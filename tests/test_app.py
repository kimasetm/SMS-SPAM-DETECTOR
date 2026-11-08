"""Tests for the Streamlit web app, using Streamlit's built-in AppTest.

AppTest runs app.py without a browser and lets the test type text, click
buttons and upload files, then check what the page shows.
"""

import json

import pytest
from streamlit.testing.v1 import AppTest

from src.config import METRICS_FILE, PROJECT_ROOT

APP_FILE = str(PROJECT_ROOT / "app.py")


@pytest.fixture
def app():
    return AppTest.from_file(APP_FILE, default_timeout=60).run()


def click(app, label):
    next(b for b in app.button if b.label == label).click().run()


def test_app_loads_without_errors(app):
    assert not app.exception
    assert app.title[0].value == "SmartSpam — SMS Spam Detection"
    assert [tab.label for tab in app.tabs] == ["🔍 Message Scanner", "📄 Batch CSV", "📊 Model Dashboard"]


def test_disclaimer_is_shown(app):
    assert any("not** protection against phishing" in c.value for c in app.caption)


# ---------- message scanner ----------

def test_empty_input_shows_warning(app):
    click(app, "Predict")
    assert any("Please enter a message" in w.value for w in app.warning)
    assert not app.error and not app.success


def test_whitespace_input_shows_warning(app):
    app.text_area(key="message_input").input("    ").run()
    click(app, "Predict")
    assert any("Please enter a message" in w.value for w in app.warning)


def test_spam_message_shows_spam_result(app):
    app.text_area(key="message_input").input(
        "WINNER!! You have been selected to receive a £900 prize reward! To claim call 09061701461.").run()
    click(app, "Predict")
    assert not app.exception
    assert "SPAM" in app.error[0].value
    assert "confidence" in app.error[0].value


def test_ham_message_shows_ham_result(app):
    app.text_area(key="message_input").input("Can you pick up milk on the way home?").run()
    click(app, "Predict")
    assert "HAM" in app.success[0].value


def test_example_button_fills_text_box(app):
    click(app, "Friend")
    assert app.text_area(key="message_input").value == "Hey, are we still meeting for lunch tomorrow at 12?"
    click(app, "Predict")
    assert app.success   # predicted as HAM


def test_delivery_scam_link_is_flagged_even_when_model_says_ham(app):
    click(app, "Delivery scam")
    click(app, "Predict")
    assert not app.exception
    assert any("Link Safety Check" in m.value for m in app.markdown)
    assert any("High risk" in e.value and "parcel-redelivery.info" in e.value for e in app.error)
    if app.success:   # the model rated the text as HAM
        assert any("link shows warning signs" in w.value for w in app.warning)


def test_safe_link_shows_no_warning_signs(app):
    app.text_area(key="message_input").input("Here are the photos https://www.google.com/photos").run()
    click(app, "Predict")
    assert any("No warning signs found" in s.value for s in app.success)


def test_message_without_links_has_no_link_section(app):
    click(app, "Family")
    click(app, "Predict")
    assert not any("Link Safety Check" in m.value for m in app.markdown)


def test_link_check_uses_the_predicted_message_not_later_edits(app):
    click(app, "Delivery scam")
    click(app, "Predict")
    app.text_area(key="message_input").input("edited text without links").run()
    assert any("parcel-redelivery.info" in e.value for e in app.error)


def test_unreadable_input_shows_validation_message(app):
    app.text_area(key="message_input").input("!!! ???").run()
    click(app, "Predict")
    assert any("no words the model can read" in w.value for w in app.warning)


# ---------- batch CSV ----------

def test_csv_upload_shows_summary_and_download(app):
    sample = (PROJECT_ROOT / "samples" / "sample_messages.csv").read_bytes()
    app.file_uploader[0].upload("sample_messages.csv", sample, "text/csv").run()

    assert not app.exception
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Messages"] == "12"
    assert int(metrics["Spam"]) + int(metrics["Ham"]) + int(metrics["Invalid"]) == 12
    results_table = app.dataframe[0].value   # the batch tab's table comes first on the page
    assert list(results_table.columns[:3]) == ["Message", "Prediction", "Confidence"]
    assert len(results_table) == 12


def test_unlabelled_csv_has_no_accuracy_section(app):
    sample = (PROJECT_ROOT / "samples" / "sample_messages.csv").read_bytes()
    app.file_uploader[0].upload("sample_messages.csv", sample, "text/csv").run()
    assert not any("How accurate" in h.value for h in app.subheader)


def test_labelled_csv_shows_accuracy_on_users_file(app):
    labelled = (PROJECT_ROOT / "samples" / "labeled_messages.csv").read_bytes()
    app.file_uploader[0].upload("labeled_messages.csv", labelled, "text/csv").run()

    assert not app.exception
    assert any("How accurate" in h.value for h in app.subheader)
    assert len(app.table) == 1   # confusion matrix table
    # The first "Accuracy" metric belongs to the batch tab (the dashboard's comes later)
    accuracy = next(m.value for m in app.metric if m.label == "Accuracy")
    assert accuracy.endswith("%")


def test_unreadable_labels_show_warning(app):
    app.file_uploader[0].upload("odd.csv", b"message,label\nhello there,maybe\n", "text/csv").run()
    assert any("none of its values could be read" in w.value for w in app.warning)


def test_bad_csv_shows_error(app):
    app.file_uploader[0].upload("bad.csv", b"id,date\n1,2024-01-01\n", "text/csv").run()
    assert not app.exception
    assert any("Couldn't find the message column" in e.value for e in app.error)


# ---------- dashboard ----------

def test_dashboard_shows_real_metrics_from_file(app):
    """The dashboard must display the saved evaluation results, not hard-coded numbers."""
    test_metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))["test"]
    shown = {}
    for m in app.metric:            # the first row of tiles is the UCI test set
        shown.setdefault(m.label, m.value)
    assert shown["Recall"] == f"{test_metrics['recall']:.2%}"
    assert shown["Accuracy"] == f"{test_metrics['accuracy']:.2%}"
    assert shown["Precision"] == f"{test_metrics['precision']:.2%}"
    assert shown["Recall"] == f"{test_metrics['recall']:.2%}"
    assert shown["F1-score"] == f"{test_metrics['f1']:.3f}"


def test_dashboard_shows_change_from_previous_release(app):
    from src.config import BASELINE_LABEL
    recall = next(m for m in app.metric if m.label == "Recall")
    assert f"vs {BASELINE_LABEL}" in recall.delta


def test_dashboard_shows_modern_test_set(app):
    test_metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    if "test_modern" not in test_metrics:
        pytest.skip("modern test set not evaluated")
    assert any("Modern test set" in m.value for m in app.markdown)
    recalls = [m.value for m in app.metric if m.label == "Recall"]
    assert f"{test_metrics['test_modern']['recall']:.2%}" in recalls
