"""Tests for CSV batch classification."""

import io

import pandas as pd
import pytest

from src.batch import (
    MAX_ROWS,
    BatchFileError,
    classify_dataframe,
    find_label_column,
    main,
    read_messages_csv,
    score_labeled_results,
    summarize,
    to_csv_bytes,
)
from src.config import PROJECT_ROOT

SAMPLE_FILE = PROJECT_ROOT / "samples" / "sample_messages.csv"


def upload(text: str, encoding="utf-8") -> io.BytesIO:
    """Simulate a file uploaded through the web app."""
    return io.BytesIO(text.encode(encoding))


# ---------- reading files ----------

def test_reads_message_column_case_insensitively():
    df, column = read_messages_csv(upload("ID,Text\n1,hello there\n2,win a prize\n"))
    assert column == "Text"
    assert len(df) == 2


def test_single_column_file_is_accepted():
    df, column = read_messages_csv(upload("my_sms\nhello\n"))
    assert column == "my_sms"


def test_missing_message_column_is_explained():
    with pytest.raises(BatchFileError, match="Couldn't find the message column"):
        read_messages_csv(upload("id,date\n1,2024-01-01\n"))


def test_empty_and_header_only_files_are_rejected():
    with pytest.raises(BatchFileError, match="empty"):
        read_messages_csv(upload(""))
    with pytest.raises(BatchFileError, match="no message rows"):
        read_messages_csv(upload("message\n"))


def test_too_many_rows_are_rejected():
    text = "message\n" + "hello\n" * (MAX_ROWS + 1)
    with pytest.raises(BatchFileError, match="maximum"):
        read_messages_csv(upload(text))


def test_excel_utf8_bom_and_latin1_files_are_read():
    df, column = read_messages_csv(upload("message\nwin £100\n", encoding="utf-8-sig"))
    assert column == "message" and df[column][0] == "win £100"
    df, column = read_messages_csv(upload("message\nwin £100\n", encoding="latin-1"))
    assert df[column][0] == "win £100"


def test_cells_are_kept_as_exact_text():
    df, column = read_messages_csv(upload("message\nNA\n00123\n"))
    assert df[column].tolist() == ["NA", "00123"]


# ---------- classification ----------

def test_sample_file_is_classified():
    df, column = read_messages_csv(SAMPLE_FILE)
    results = classify_dataframe(df, column)

    assert len(results) == len(df)
    assert set(results["prediction"]) <= {"SPAM", "HAM", "INVALID"}
    assert list(results.columns[-4:]) == ["prediction", "confidence", "spam_probability", "error"]
    # Original columns are kept unchanged
    assert results[column].tolist() == df[column].tolist()


def test_invalid_rows_are_marked_not_dropped():
    df = pd.DataFrame({"message": ["see you at lunch", "", "!!!"]})
    results = classify_dataframe(df, "message")
    assert results["prediction"].tolist()[1:] == ["INVALID", "INVALID"]
    assert results["error"][1] == "Message is empty."
    assert results["error"][0] == ""
    assert summarize(results) == {"total": 3, "spam": 0, "ham": 1, "invalid": 2}


# ---------- scoring files that include labels ----------

def test_label_column_is_found_only_when_present():
    df = pd.DataFrame({"message": ["hi"], "Label": ["ham"]})
    assert find_label_column(df, "message") == "Label"
    assert find_label_column(pd.DataFrame({"message": ["hi"]}), "message") is None


def test_scores_match_hand_calculation():
    # Hand-made predictions: 2 spam (1 caught, 1 missed), 2 ham (both correct)
    results = pd.DataFrame({
        "label": ["spam", "SPAM", "ham", " Ham "],      # case and spaces are ignored
        "prediction": ["SPAM", "HAM", "HAM", "HAM"],
        "spam_probability": [0.9, 0.2, 0.1, 0.3],
    })
    scored = score_labeled_results(results, "label")
    m = scored["metrics"]
    assert scored["rows_scored"] == 4 and scored["rows_skipped"] == 0
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(1.0)
    assert m["recall"] == pytest.approx(0.5)
    assert m["confusion_matrix"]["false_negative"] == 1


def test_numeric_and_word_labels_are_understood():
    results = pd.DataFrame({"label": ["1", "0", "yes", "no", "legitimate"],
                            "prediction": ["SPAM", "HAM", "SPAM", "HAM", "HAM"],
                            "spam_probability": [0.9, 0.1, 0.8, 0.2, 0.3]})
    assert score_labeled_results(results, "label")["metrics"]["accuracy"] == 1.0


def test_invalid_rows_and_unknown_labels_are_skipped():
    results = pd.DataFrame({"label": ["spam", "ham", "maybe", "spam"],
                            "prediction": ["SPAM", "HAM", "HAM", "INVALID"],
                            "spam_probability": [0.9, 0.1, 0.2, None]})
    scored = score_labeled_results(results, "label")
    assert scored["rows_scored"] == 2
    assert scored["rows_skipped"] == 2


def test_no_usable_labels_returns_none():
    results = pd.DataFrame({"label": ["?", ""], "prediction": ["HAM", "HAM"], "spam_probability": [0.1, 0.2]})
    assert score_labeled_results(results, "label") is None


def test_labeled_sample_file_end_to_end():
    df, column = read_messages_csv(PROJECT_ROOT / "samples" / "labeled_messages.csv")
    results = classify_dataframe(df, column)
    scored = score_labeled_results(results, find_label_column(df, column))
    assert scored["rows_scored"] == len(df)
    assert 0.0 <= scored["metrics"]["accuracy"] <= 1.0


# ---------- export ----------

def test_export_escapes_spreadsheet_formulas():
    """CSV injection protection: cells starting with = + - @ must not run as formulas."""
    results = pd.DataFrame({"message": ["=HYPERLINK(\"http://evil\")", "+1", "-2", "@SUM(A1)", "normal"],
                            "prediction": ["HAM"] * 5})
    exported = pd.read_csv(io.BytesIO(to_csv_bytes(results)), encoding="utf-8-sig", dtype=str)
    assert exported["message"].tolist() == ["'=HYPERLINK(\"http://evil\")", "'+1", "'-2", "'@SUM(A1)", "normal"]


def test_export_round_trip_keeps_results():
    df = pd.DataFrame({"message": ["see you at lunch", "WIN a FREE prize, call 09061701461"]})
    results = classify_dataframe(df, "message")
    exported = pd.read_csv(io.BytesIO(to_csv_bytes(results)), encoding="utf-8-sig")
    assert exported["prediction"].tolist() == results["prediction"].tolist()


# ---------- command line ----------

def test_command_line(tmp_path, capsys):
    output = tmp_path / "out.csv"
    assert main([str(SAMPLE_FILE), str(output)]) == 0
    assert output.exists()
    assert "Classified" in capsys.readouterr().out
    assert main([str(tmp_path / "missing.csv")]) == 1
    assert main([]) == 1
