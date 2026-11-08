"""SmartSpam web application.

Run locally from the project root:
    streamlit run app.py

The app uses the trained neural network in models/ for every prediction.
There are no hard-coded keyword rules.
"""

import json

import pandas as pd
import streamlit as st

from src import config
from src.batch import (
    BatchFileError,
    classify_dataframe,
    find_label_column,
    mark_correct,
    read_messages_csv,
    score_labeled_results,
    summarize,
    to_csv_bytes,
)
from src.link_check import HIGH, LOW, MEDIUM, NO_FLAGS, check_links
from src.monitor import check_messages
from src.threat_intel import add_online_findings, check_sender, fetch_blocklist, lookup_domain_age
from src.predict import InvalidMessageError, SpamDetector, format_confidence

st.set_page_config(page_title="SmartSpam — SMS Spam Detection", page_icon="📱", layout="centered")

EXAMPLES = {
    "Prize scam": "WINNER!! You have been selected to receive a £900 prize reward! "
                  "To claim call 09061701461. Valid 12 hours only.",
    "Friend": "Hey, are we still meeting for lunch tomorrow at 12?",
    "Free offer": "Congratulations! You've won a free iPhone. Text WIN to 80086 now",
    "Family": "Can you pick up milk on the way home?",
    "Delivery scam": "Your package could not be delivered. Reschedule here: http://parcel-redelivery.info",
}
# Example senders, to show the sender check (a UK premium-rate number)
EXAMPLE_SENDERS = {"Prize scam": "09061701461"}

# Blue banner under the title (same blues as the theme in .streamlit/config.toml).
# White text on these blues has a contrast ratio above 4.5:1, so it stays readable.
BANNER = """
<div style="background: linear-gradient(135deg, #1c5cab 0%, #2a78d6 100%); color: #ffffff;
            padding: 0.9rem 1.2rem; border-radius: 0.6rem; margin: 0.2rem 0 1rem 0;">
  <div style="font-size: 1.05rem; font-weight: 600;">A neural network that classifies text messages as SPAM or HAM</div>
  <div style="font-size: 0.85rem; opacity: 0.9; margin-top: 0.25rem;">
    PyTorch model &middot; Link Safety Check &middot; Sender check &middot; Batch CSV &middot; Evaluation dashboard
  </div>
</div>
"""

DISCLAIMER = (
    "SmartSpam is a machine learning class project trained on SMS messages from around 2011. "
    "It can be wrong, and it is **not** protection against phishing, scams or malware. "
    "Never rely on it to decide whether a link or request is safe."
)


@st.cache_resource(show_spinner="Loading the trained model...")
def load_detector() -> SpamDetector:
    """Load the model once per server and share it across all users and reruns."""
    return SpamDetector()


@st.cache_data
def load_json(path):
    """Read a JSON results file, or return None if it doesn't exist."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics():
    return load_json(config.METRICS_FILE)


def set_example(text: str, sender: str = "") -> None:
    st.session_state["message_input"] = text
    st.session_state["sender_input"] = sender
    st.session_state.pop("last_result", None)


# ---------------------------------------------------------------------------
# Message scanner
# ---------------------------------------------------------------------------

def show_scanner(detector: SpamDetector) -> None:
    st.subheader("Message scanner")
    st.write("Paste a text message and the neural network will predict whether it's spam.")

    st.caption("Try an example:")
    columns = st.columns(len(EXAMPLES))
    for column, (name, text) in zip(columns, EXAMPLES.items()):
        column.button(name, on_click=set_example, args=(text, EXAMPLE_SENDERS.get(name, "")), width="stretch")

    message = st.text_area("SMS message", key="message_input", height=130,
                           placeholder="e.g. Congratulations! You've won a free prize...")
    left, right = st.columns([1, 1])
    left.text_input("Sender (optional)", key="sender_input", placeholder="e.g. 09061701461 or AMAZON")
    right.checkbox("Online link checks", key="online_checks",
                   help="Checks links against the OpenPhish phishing feed and looks up how old the domain is (RDAP). "
                        "This sends the link's address to those services. Off by default.")

    if st.button("Predict", type="primary"):
        if not message.strip():
            st.warning("Please enter a message before clicking Predict.", icon="⚠️")
            st.session_state.pop("last_result", None)
        else:
            try:
                st.session_state["last_result"] = detector.predict(message)
                st.session_state["last_message"] = message   # the exact text that was predicted
                st.session_state["last_sender"] = st.session_state.get("sender_input", "")
                st.session_state["last_online"] = st.session_state.get("online_checks", False)
            except InvalidMessageError as error:
                st.warning(str(error), icon="⚠️")
                st.session_state.pop("last_result", None)

    result = st.session_state.get("last_result")
    if result is not None:
        show_result(result, detector.threshold)


def show_result(result, threshold: float) -> None:
    confidence = format_confidence(result.confidence)
    if result.is_spam:
        st.error(f"### 🚫 SPAM\nThe model predicts this message is **spam** "
                 f"with **{confidence}** confidence.")
    else:
        st.success(f"### ✅ HAM (legitimate)\nThe model predicts this message is **not spam** "
                   f"with **{confidence}** confidence.")

    st.progress(result.spam_probability,
                text=f"Spam probability: {result.spam_probability:.1%} "
                     f"(messages at {threshold:.0%} or above are labelled spam)")
    if result.is_spam and result.spam_probability < 0.5:
        st.info(f"Flagged because its spam probability is above the {threshold:.0%} cutoff. The cutoff was "
                "lowered from 50% to catch more spam, so borderline messages like this one are flagged. "
                "Treat the result with caution.", icon="ℹ️")
    elif abs(result.spam_probability - threshold) <= 0.15:
        st.info("The model is uncertain about this message. Treat the result with caution.", icon="ℹ️")

    show_link_check(result)
    show_sender_check()

    with st.expander("What did the model see?"):
        st.write("Before prediction, the message is cleaned exactly as during training: lowercased, "
                 "punctuation removed, and URLs, phone numbers, prices and numbers replaced with "
                 "placeholder words.")
        st.code(result.cleaned_text, language=None, wrap_lines=True)


RISK_DISPLAY = {
    HIGH: (st.error, "🚩", "High risk"),
    MEDIUM: (st.warning, "⚠️", "Medium risk"),
    LOW: (st.info, "ℹ️", "Low risk"),
    NO_FLAGS: (st.success, "✅", "No warning signs found"),
}


@st.cache_data(ttl=12 * 60 * 60, show_spinner="Downloading the OpenPhish phishing feed...")
def cached_blocklist():
    """The OpenPhish feed, downloaded at most every 12 hours. None if it can't be reached."""
    try:
        return fetch_blocklist()
    except Exception:   # network problems must never break the app
        return None


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def cached_domain_age(domain: str) -> dict:
    return lookup_domain_age(domain)


def show_link_check(result) -> None:
    """Rule-based checks on any links in the message (separate from the AI prediction)."""
    reports = check_links(st.session_state.get("last_message", ""))
    if not reports:
        return

    online = st.session_state.get("last_online", False)
    st.markdown("#### 🔗 Link Safety Check")
    st.caption("Rule-based checks on the links themselves, separate from the AI model, which sees every link "
               "as the same placeholder word. Links are never opened."
               + (" Online checks: OpenPhish phishing feed and domain age (RDAP)." if online else
                  " Turn on **Online link checks** to also check a phishing blocklist and the domain's age."))

    if online:
        with st.spinner("Running online checks..."):
            status = add_online_findings(reports, cached_blocklist(), age_lookup=cached_domain_age)
        if status["blocklist"] == "unavailable":
            st.caption("⚠️ The OpenPhish feed couldn't be downloaded, so the blocklist wasn't checked.")
        failed = [d for d, s in status["domain_age"].items() if s == "error"]
        if failed:
            st.caption(f"⚠️ Domain age couldn't be checked for: {', '.join(failed)}.")

    for report in reports:
        show, icon, label = RISK_DISPLAY[report.risk_level]
        details = "".join(f"\n- **{f.title}**: {f.explanation}" for f in report.findings)
        show(f"**{label}:** `{report.url}`{details}", icon=icon)

    risky = any(r.risk_level in (HIGH, MEDIUM) for r in reports)
    if risky and not result.is_spam:
        st.warning("**The AI rated the text as legitimate, but the link shows warning signs.** "
                   "Don't open the link. Modern scams often use normal-sounding text.", icon="⚠️")
    if not risky:
        st.caption("No warning signs doesn't mean a link is safe. When in doubt, go to the company's "
                   "website directly instead of using the link.")


def show_sender_check() -> None:
    """Offline checks on the optional sender field."""
    sender = st.session_state.get("last_sender", "").strip()
    if not sender:
        return
    st.markdown("#### 📇 Sender check")
    findings = check_sender(sender)
    if not findings:
        st.success(f"No warning signs in the sender `{sender}`. Phone numbers can still be faked.", icon="✅")
    for f in findings:
        show, icon, _ = RISK_DISPLAY[f.severity]
        show(f"**{f.title}** (`{sender}`): {f.explanation}", icon=icon)


# ---------------------------------------------------------------------------
# Batch CSV
# ---------------------------------------------------------------------------

def show_batch(detector: SpamDetector) -> None:
    st.subheader("Batch CSV classification")
    st.write("Upload a CSV file with a **message** column. "
             "Optionally add a **label** column (spam / ham) to check the model's accuracy.")

    sample = config.PROJECT_ROOT / "samples" / "labeled_messages.csv"
    if sample.exists():
        st.download_button("Download sample CSV", sample.read_bytes(),
                           file_name="sample_messages.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded is None:
        return

    try:
        df, column = read_messages_csv(uploaded)
    except BatchFileError as error:
        st.error(f"**Couldn't use this file.** {error}", icon="❌")
        return

    with st.spinner(f"Classifying {len(df):,} messages..."):
        results = classify_dataframe(df, column, detector)
    counts = summarize(results)
    label_column = find_label_column(df, column)

    # Summary
    tiles = st.columns(4 if counts["invalid"] else 3)
    tiles[0].metric("Messages", f"{counts['total']:,}")
    tiles[1].metric("Spam", f"{counts['spam']:,}")
    tiles[2].metric("Ham", f"{counts['ham']:,}")
    if counts["invalid"]:
        tiles[3].metric("Invalid", f"{counts['invalid']:,}")

    # Results table: only the columns needed to read the results
    table = pd.DataFrame({"Message": results[column]})
    if label_column:
        table["Actual"] = results[label_column].str.upper()
    table["Prediction"] = results["prediction"]
    table["Confidence"] = results["confidence"]
    if label_column:
        table["Correct?"] = mark_correct(results, label_column)
    if counts["invalid"]:
        table["Error"] = results["error"]
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "Message": st.column_config.TextColumn(width="large"),
        "Confidence": st.column_config.NumberColumn(format="percent"),
    })

    output_name = uploaded.name.rsplit(".", 1)[0] + "_predictions.csv"
    st.download_button("Download predictions (CSV)", to_csv_bytes(results),
                       file_name=output_name, mime="text/csv", type="primary")

    if label_column:
        show_labeled_scores(results, label_column, detector.threshold)

    show_drift_check(df[column].tolist(), detector)


def show_drift_check(messages, detector) -> None:
    """Collapsed by default: compare the uploaded messages with the data the model was checked on."""
    with st.expander("Data drift check"):
        try:
            result = check_messages(messages, detector)
        except ValueError:
            st.write("No readable messages to check.")
            return
        if result["warnings"] is None:
            st.write("No monitoring baseline found. Train the model first: `python -m src.train`.")
            return
        st.write("Compares this file with the validation data the model was checked on. Big differences "
                 "suggest the model may be less reliable for these messages (no labels needed).")
        profile, base = result["profile"], result["baseline"]
        st.table(pd.DataFrame({
            "This file": [f"{profile['unknown_word_rate']:.1%}", f"{profile['predicted_spam_rate']:.1%}",
                          f"{profile['uncertain_rate']:.1%}", f"{profile['avg_words']:.1f}"],
            "Baseline": [f"{base['unknown_word_rate']:.1%}", f"{base['predicted_spam_rate']:.1%}",
                         f"{base['uncertain_rate']:.1%}", f"{base['avg_words']:.1f}"]},
            index=["Unknown words", "Flagged as spam", "Uncertain predictions", "Words per message"]))
        if result["warnings"]:
            for w in result["warnings"]:
                st.warning(f"**{w['title']}:** {w['explanation']}", icon="⚠️")
        else:
            st.success("No drift signals found. (This doesn't prove the model is still accurate.)", icon="✅")


def show_labeled_scores(results, label_column: str, threshold: float) -> None:
    """Score predictions against the labels in the uploaded file."""
    st.subheader("How accurate was the model on this file?")
    scored = score_labeled_results(results, label_column, threshold)
    if scored is None:
        st.warning(f"The **{label_column}** column was found, but none of its values could be read. "
                   "Use **spam** or **ham** for each row.", icon="⚠️")
        return

    m = scored["metrics"]
    correct = round(m["accuracy"] * scored["rows_scored"])
    st.write(f"The model was right on **{correct} of {scored['rows_scored']}** messages.")

    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{m['accuracy']:.2%}")
    b.metric("Precision", f"{m['precision']:.2%}")
    c.metric("Recall", f"{m['recall']:.2%}")
    d.metric("F1-score", f"{m['f1']:.3f}")

    with st.expander("Confusion matrix"):
        cm = m["confusion_matrix"]
        st.table(pd.DataFrame(
            {"Predicted ham": [cm["true_negative"], cm["false_negative"]],
             "Predicted spam": [cm["false_positive"], cm["true_positive"]]},
            index=["Actually ham", "Actually spam"]))
        if scored["rows_skipped"]:
            st.caption(f"{scored['rows_skipped']} rows skipped (empty message or unrecognized label).")


# ---------------------------------------------------------------------------
# Model dashboard
# ---------------------------------------------------------------------------

def show_metric_row(current: dict, base) -> None:
    """Four metric tiles, each showing the change from the previous release when available."""
    def change(key, as_percent=True):
        if not base or base.get(key) is None or base[key] == current[key]:
            return None
        diff = current[key] - base[key]
        label = config.BASELINE_LABEL
        return f"{diff * 100:+.2f} pts vs {label}" if as_percent else f"{diff:+.4f} vs {label}"

    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{current['accuracy']:.2%}", change("accuracy"),
             help="Share of all messages classified correctly.")
    b.metric("Precision", f"{current['precision']:.2%}", change("precision"),
             help="Of the messages flagged as spam, how many really were spam.")
    c.metric("Recall", f"{current['recall']:.2%}", change("recall"),
             help="Of all real spam messages, how many were caught.")
    d.metric("F1-score", f"{current['f1']:.3f}", change("f1", as_percent=False),
             help="Balance (harmonic mean) of precision and recall.")


def show_counts(current: dict) -> None:
    cm = current["confusion_matrix"]
    lines = (f"- **{cm['true_negative']:,}** legitimate messages correctly allowed, "
             f"**{cm['false_positive']:,}** wrongly flagged as spam\n"
             f"- **{cm['true_positive']:,}** spam messages caught, **{cm['false_negative']:,}** missed")
    disguised = current.get("disguised_spam")
    if disguised and disguised["count"]:
        lines += (f"\n- **{disguised['caught_disguised']} of {disguised['count']}** spam messages still caught "
                  "when words are disguised (\"free\" → \"fr33\")")
    st.markdown(lines)


def show_dashboard(detector: SpamDetector) -> None:
    st.subheader("Model evaluation dashboard")
    metrics = load_metrics()
    if metrics is None:
        st.warning("No evaluation results found. Run `python -m src.evaluate` to create them.")
        return

    test = metrics["test"]
    baseline = load_json(config.BASELINE_METRICS_FILE) or {}
    figures = config.FIGURES_DIR

    st.markdown("#### UCI test set")
    st.write(f"{test['num_messages']:,} messages ({test['num_ham']:,} ham, {test['num_spam']:,} spam) from the "
             "original dataset that the model never saw during training.")
    show_metric_row(test, baseline.get("test"))
    show_counts(test)
    if (figures / "confusion_matrix.png").exists():
        st.image(str(figures / "confusion_matrix.png"), width=440)

    modern = metrics.get("test_modern")
    if modern:
        st.markdown("#### Modern test set")
        st.write(f"{modern['num_messages']:,} newer messages ({modern['num_ham']:,} ham, {modern['num_spam']:,} "
                 f"spam, including {modern.get('smishing_count', 0)} smishing / SMS phishing) from the Mishra & "
                 "Soni (2022) dataset, also never seen during training.")
        show_metric_row(modern, baseline.get("test_modern"))
        show_counts(modern)
        if "smishing_recall" in modern:
            st.markdown(f"- **{modern['smishing_caught']} of {modern['smishing_count']}** smishing messages "
                        f"caught ({modern['smishing_recall']:.1%})")
        if (figures / "confusion_matrix_modern.png").exists():
            st.image(str(figures / "confusion_matrix_modern.png"), width=440)

    if baseline:
        st.caption(f"Changes are compared with {config.BASELINE_LABEL}, which was trained on the UCI data only. "
                   f"The current spam threshold ({test['threshold']:.0%}) was chosen on the validation set.")
    show_calibration(test, modern, figures)

    if (figures / "training_history.png").exists():
        st.image(str(figures / "training_history.png"),
                 caption="Training vs validation loss and accuracy per epoch. Training stopped early when "
                         "validation loss stopped improving.", width="stretch")

    with st.expander("Model architecture and training details"):
        mc = detector.model_config
        training = mc["training"]
        pooling = ("`Conv1D` (3-word windows) → global max pooling" if mc.get("architecture") == "cnn"
                   else "global average pooling")
        st.markdown(
            f"""
**Architecture (PyTorch):** text → `TextVectorizer` → `Embedding({mc['vocab_size']:,} × {mc['embedding_dim']})`
→ {pooling} → `Dropout({mc['dropout']})` → `Dense({mc['hidden_dim']}, ReLU)`
→ `Dropout({mc['dropout']})` → `Dense(1)` → sigmoid

| Setting | Value |
|---|---|
| Misspelling repair ("fr33" → "free") | {"on" if detector.vectorizer.repair_misspellings else "off"} |
| Vocabulary size (training data only) | {mc['vocab_size']:,} words |
| Sequence length | {training['sequence_length']} words |
| Training / validation messages | {training['train_size']:,} / {training['val_size']:,} |
| Optimizer / learning rate | Adam / {training['learning_rate']} |
| Epochs run (early stopping) | {training['epochs_run']} (best: epoch {training['best_epoch']}) |
| Best validation accuracy | {training['best_val_accuracy']:.2%} |
| Spam decision threshold (chosen on validation) | {detector.threshold:.2f} |
| Test ROC-AUC | {test['roc_auc']:.4f} |
"""
        )

    show_experiment(config.ARCHITECTURE_COMPARISON_FILE,
                    "v3 experiment: architecture and misspelling repair (validation set)",
                    "Average pooling and a 1D CNN, with and without misspelling repair, were compared on the "
                    "**validation** set; the highest F1-score was selected (rule fixed before running). "
                    "'Disguised' is the share of validation spam still caught when words are disguised "
                    "(for information only).")
    show_experiment(config.MODEL_COMPARISON_FILE,
                    "v2 experiment: handling class imbalance (UCI validation set)",
                    "Only about 12% of messages are spam. Four approaches were compared on the **validation** "
                    "set, and the one with the highest F1-score was selected (rule fixed before running).")


def show_calibration(test: dict, modern, figures) -> None:
    """How well the confidence scores match reality, before and after temperature scaling."""
    cal = test.get("calibration")
    if not cal:
        return
    st.markdown("#### Confidence calibration")
    st.write(f"Confidence scores are calibrated with **temperature scaling** (T = {cal['temperature']:.2f}, "
             "chosen on the validation set). This changes the confidence numbers, not which messages are "
             "flagged. Calibration error (ECE, lower is better):")
    rows = [("UCI test set", cal)] + ([("Modern test set", modern["calibration"])] if modern else [])
    st.markdown("\n".join(f"- **{name}:** {c['ece_uncalibrated']:.3f} → **{c['ece']:.3f}**" for name, c in rows))
    if (figures / "calibration.png").exists():
        st.image(str(figures / "calibration.png"), width="stretch",
                 caption="Points on the dashed line are perfectly calibrated. Many bins hold only a few messages.")


EXPERIMENT_COLUMNS = {
    "variant": "Variant", "threshold": "Threshold", "val_precision": "Precision", "val_recall": "Recall",
    "val_f1": "F1", "val_spam_caught": "Spam caught", "val_false_alarms": "False alarms",
    "val_disguised_spam_recall": "Disguised",
}


def show_experiment(path, title: str, description: str) -> None:
    comparison = load_json(path)
    if not comparison:
        return
    with st.expander(title):
        st.write(description)
        table = pd.DataFrame(comparison["variants"])
        table = table[[c for c in EXPERIMENT_COLUMNS if c in table]].rename(columns=EXPERIMENT_COLUMNS)
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Precision": st.column_config.NumberColumn(format="percent"),
            "Recall": st.column_config.NumberColumn(format="percent"),
            "Disguised": st.column_config.NumberColumn(format="percent"),
            "F1": st.column_config.NumberColumn(format="%.4f")})
        st.caption(f"Selected: {comparison['selected']}.")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.title(":blue[SmartSpam] — SMS Spam Detection")
    st.markdown(BANNER, unsafe_allow_html=True)

    try:
        detector = load_detector()
    except FileNotFoundError as error:
        st.error(f"**The trained model could not be loaded.** {error}", icon="❌")
        st.stop()

    scanner_tab, batch_tab, dashboard_tab = st.tabs(["🔍 Message Scanner", "📄 Batch CSV", "📊 Model Dashboard"])
    with scanner_tab:
        show_scanner(detector)
    with batch_tab:
        show_batch(detector)
    with dashboard_tab:
        show_dashboard(detector)

    st.divider()
    st.caption(f"⚠️ {DISCLAIMER}")


main()
