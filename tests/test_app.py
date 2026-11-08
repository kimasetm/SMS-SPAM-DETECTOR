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
    assert len(app.dataframe) == 1


def test_bad_csv_shows_error(app):
    app.file_uploader[0].upload("bad.csv", b"id,date\n1,2024-01-01\n", "text/csv").run()
    assert not app.exception
    assert any("Couldn't find the message column" in e.value for e in app.error)


# ---------- dashboard ----------

def test_dashboard_shows_real_metrics_from_file(app):
    """The dashboard must display the saved evaluation results, not hard-coded numbers."""
    test_metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))["test"]
    shown = {m.label: m.value for m in app.metric}
    assert shown["Accuracy"] == f"{test_metrics['accuracy']:.2%}"
    assert shown["Precision"] == f"{test_metrics['precision']:.2%}"
    assert shown["Recall"] == f"{test_metrics['recall']:.2%}"
    assert shown["F1-score"] == f"{test_metrics['f1']:.3f}"
