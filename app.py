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
from src.batch import BatchFileError, classify_dataframe, read_messages_csv, summarize, to_csv_bytes
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
def load_metrics():
    if not config.METRICS_FILE.exists():
        return None
    return json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))


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
        show_result(result)


def show_result(result) -> None:
    confidence = format_confidence(result.confidence)
    if result.is_spam:
        st.error(f"### 🚫 SPAM\nThe model predicts this message is **spam** "
                 f"with **{confidence}** confidence.")
    else:
        st.success(f"### ✅ HAM (legitimate)\nThe model predicts this message is **not spam** "
                   f"with **{confidence}** confidence.")

    st.progress(result.spam_probability,
                text=f"Spam probability: {result.spam_probability:.1%} "
                     f"(messages at {config.SPAM_THRESHOLD:.0%} or above are labelled spam)")
    if 0.35 <= result.spam_probability <= 0.65:
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
    st.write("Upload a CSV file with a column named **message**, **text**, **sms** or **body** "
             "(or a file with a single column). Maximum 5 MB and 5,000 rows.")

    sample_path = config.PROJECT_ROOT / "samples" / "sample_messages.csv"
    if sample_path.exists():
        st.download_button("Download a sample CSV", sample_path.read_bytes(),
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

    a, b, c, d = st.columns(4)
    a.metric("Messages", f"{counts['total']:,}")
    b.metric("Spam", f"{counts['spam']:,}")
    c.metric("Ham", f"{counts['ham']:,}")
    d.metric("Invalid", f"{counts['invalid']:,}")
    if counts["invalid"]:
        st.caption("Invalid rows (for example, empty messages) are listed with the reason in the **error** column.")

    st.dataframe(
        results,
        hide_index=True,
        width="stretch",
        column_config={
            column: st.column_config.TextColumn("Message", width="large"),
            "prediction": st.column_config.TextColumn("Prediction"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0,
                                                          format="percent"),
            "spam_probability": st.column_config.NumberColumn("Spam probability", format="%.4f"),
            "error": st.column_config.TextColumn("Error"),
        },
    )

    output_name = uploaded.name.rsplit(".", 1)[0] + "_predictions.csv"
    st.download_button("Download predictions (CSV)", to_csv_bytes(results),
                       file_name=output_name, mime="text/csv", type="primary")


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

    a, b, c, d = st.columns(4)
    a.metric("Accuracy", f"{test['accuracy']:.2%}", help="Share of all messages classified correctly.")
    b.metric("Precision", f"{test['precision']:.2%}",
             help="Of the messages flagged as spam, how many really were spam.")
    c.metric("Recall", f"{test['recall']:.2%}", help="Of all real spam messages, how many were caught.")
    d.metric("F1-score", f"{test['f1']:.3f}", help="Balance (harmonic mean) of precision and recall.")

    cm = test["confusion_matrix"]
    st.markdown(
        f"- **{cm['true_negative']:,}** legitimate messages correctly allowed, "
        f"**{cm['false_positive']:,}** wrongly flagged as spam\n"
        f"- **{cm['true_positive']:,}** spam messages caught, **{cm['false_negative']:,}** missed"
    )

    figures = config.FIGURES_DIR
    if (figures / "confusion_matrix.png").exists():
        st.image(str(figures / "confusion_matrix.png"), caption="Confusion matrix (test set)", width=440)
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
| Test ROC-AUC | {test['roc_auc']:.4f} |
"""
        )


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
