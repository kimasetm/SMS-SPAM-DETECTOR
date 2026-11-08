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
from src.predict import InvalidMessageError, SpamDetector, format_confidence

st.set_page_config(page_title="SmartSpam — SMS Spam Detection", page_icon="📱", layout="centered")

EXAMPLES = {
    "Prize scam": "WINNER!! You have been selected to receive a £900 prize reward! "
                  "To claim call 09061701461. Valid 12 hours only.",
    "Friend": "Hey, are we still meeting for lunch tomorrow at 12?",
    "Free offer": "Congratulations! You've won a free iPhone. Text WIN to 80086 now",
    "Family": "Can you pick up milk on the way home?",
}

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


def set_example(text: str) -> None:
    st.session_state["message_input"] = text
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
        column.button(name, on_click=set_example, args=(text,), width="stretch")

    message = st.text_area("SMS message", key="message_input", height=130,
                           placeholder="e.g. Congratulations! You've won a free prize...")

    if st.button("Predict", type="primary"):
        if not message.strip():
            st.warning("Please enter a message before clicking Predict.", icon="⚠️")
            st.session_state.pop("last_result", None)
        else:
            try:
                st.session_state["last_result"] = detector.predict(message)
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

    with st.expander("What did the model see?"):
        st.write("Before prediction, the message is cleaned exactly as during training: lowercased, "
                 "punctuation removed, and URLs, phone numbers, prices and numbers replaced with "
                 "placeholder words.")
        st.code(result.cleaned_text, language=None, wrap_lines=True)


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

def show_dashboard(detector: SpamDetector) -> None:
    st.subheader("Model evaluation dashboard")
    metrics = load_metrics()
    if metrics is None:
        st.warning("No evaluation results found. Run `python -m src.evaluate` to create them.")
        return

    test = metrics["test"]
    st.write(f"Results on the **held-out test set**: {test['num_messages']:,} messages "
             f"({test['num_ham']:,} ham, {test['num_spam']:,} spam) that the model never saw during training.")

    # Show the change from v1.0 (fixed 50% threshold) next to each metric
    baseline = load_json(config.BASELINE_METRICS_FILE)
    base = baseline["test"] if baseline else None

    def change(key, as_percent=True):
        if base is None or base[key] == test[key]:
            return None
        diff = test[key] - base[key]
        return f"{diff * 100:+.2f} pts vs v1.0" if as_percent else f"{diff:+.4f} vs v1.0"

    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{test['accuracy']:.2%}", change("accuracy"),
             help="Share of all messages classified correctly.")
    b.metric("Precision", f"{test['precision']:.2%}", change("precision"),
             help="Of the messages flagged as spam, how many really were spam.")
    c.metric("Recall", f"{test['recall']:.2%}", change("recall"),
             help="Of all real spam messages, how many were caught.")
    d.metric("F1-score", f"{test['f1']:.3f}", change("f1", as_percent=False),
             help="Balance (harmonic mean) of precision and recall.")
    if base:
        st.caption(f"Changes compared with v1.0, which used a fixed 50% spam threshold. v2 uses a "
                   f"{test['threshold']:.0%} threshold chosen on the validation set to catch more spam.")

    cm = test["confusion_matrix"]
    st.markdown(
        f"- **{cm['true_negative']:,}** legitimate messages correctly allowed, "
        f"**{cm['false_positive']:,}** wrongly flagged as spam\n"
        f"- **{cm['true_positive']:,}** spam messages caught, **{cm['false_negative']:,}** missed"
    )

    figures = config.FIGURES_DIR
    if (figures / "confusion_matrix.png").exists():
        st.image(str(figures / "confusion_matrix.png"), width=440)
    if (figures / "training_history.png").exists():
        st.image(str(figures / "training_history.png"),
                 caption="Training vs validation loss and accuracy per epoch. Training stopped early when "
                         "validation loss stopped improving.", width="stretch")

    with st.expander("Model architecture and training details"):
        mc = detector.model_config
        training = mc["training"]
        st.markdown(
            f"""
**Architecture (PyTorch):** text → `TextVectorizer` → `Embedding({mc['vocab_size']:,} × {mc['embedding_dim']})`
→ global average pooling → `Dropout({mc['dropout']})` → `Dense({mc['hidden_dim']}, ReLU)`
→ `Dropout({mc['dropout']})` → `Dense(1)` → sigmoid

| Setting | Value |
|---|---|
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

    comparison = load_json(config.MODEL_COMPARISON_FILE)
    if comparison:
        with st.expander("Experiment: handling class imbalance (validation set)"):
            st.write("Only about 12% of messages are spam. Four approaches were compared on the **validation** "
                     "set, and the one with the highest F1-score was selected (rule fixed before running).")
            st.dataframe(pd.DataFrame(comparison["variants"]).rename(columns={
                "variant": "Variant", "spam_weight": "Spam weight", "threshold": "Threshold",
                "val_precision": "Precision", "val_recall": "Recall", "val_f1": "F1",
                "val_spam_caught": "Spam caught", "val_false_alarms": "False alarms"}),
                hide_index=True, width="stretch",
                column_config={"Precision": st.column_config.NumberColumn(format="percent"),
                               "Recall": st.column_config.NumberColumn(format="percent"),
                               "F1": st.column_config.NumberColumn(format="%.4f")})
            st.caption(f"Selected: {comparison['selected']}.")


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("SmartSpam — SMS Spam Detection")
    st.caption("A neural network that classifies text messages as **SPAM** or **HAM** (legitimate).")

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
