"""Tests for the prediction engine and input validation.

Most tests use the real trained model committed in models/.
"""

import pytest

from src.model import predict_probabilities
from src.predict import (
    MAX_MESSAGE_LENGTH,
    InvalidMessageError,
    PredictionResult,
    SpamDetector,
    format_confidence,
    get_detector,
    main,
    make_result,
    predict_message,
    validate_message,
)
from src.preprocessing import clean_text


@pytest.fixture(scope="module")
def detector():
    return get_detector()


# ---------- input validation ----------

@pytest.mark.parametrize("bad_input", [None, "", "   ", "\n\t"])
def test_empty_input_is_rejected(bad_input):
    with pytest.raises(InvalidMessageError, match="Message is empty"):
        validate_message(bad_input)


def test_non_text_input_is_rejected():
    with pytest.raises(InvalidMessageError, match="must be text"):
        validate_message(12345)


def test_too_long_input_is_rejected():
    with pytest.raises(InvalidMessageError, match="too long"):
        validate_message("a" * (MAX_MESSAGE_LENGTH + 1))
    assert validate_message("a " * (MAX_MESSAGE_LENGTH // 2))   # at the limit: accepted


def test_input_without_readable_words_is_rejected():
    with pytest.raises(InvalidMessageError, match="no words"):
        validate_message("!!! ??? ...")


def test_validate_returns_cleaned_text():
    assert validate_message("Call 09061701461 NOW!") == "call phonetoken now"


# ---------- results and confidence ----------

def test_confidence_is_probability_of_chosen_label():
    spam = make_result(0.9, "x")
    ham = make_result(0.1, "x")
    assert spam.label == "SPAM" and spam.confidence == pytest.approx(0.9)
    assert ham.label == "HAM" and ham.confidence == pytest.approx(0.9)


def test_threshold_boundary():
    assert make_result(0.5, "x").label == "SPAM"   # threshold is inclusive
    assert make_result(0.4999, "x").label == "HAM"
    assert make_result(0.6, "x", threshold=0.7).label == "HAM"


def test_format_confidence_never_shows_100_percent():
    assert format_confidence(0.9995) == ">99.9%"
    assert format_confidence(0.977) == "97.7%"


# ---------- predictions with the trained model ----------

def test_prediction_result_fields(detector):
    result = detector.predict("Are we still meeting for lunch tomorrow?")
    assert isinstance(result, PredictionResult)
    assert result.label in ("SPAM", "HAM")
    assert 0.0 <= result.spam_probability <= 1.0
    assert 0.5 <= result.confidence <= 1.0
    assert result.to_dict()["label"] == result.label


def test_obvious_spam_and_ham(detector):
    spam = detector.predict("WINNER!! You have been selected to receive a £900 prize reward! "
                            "To claim call 09061701461. Valid 12 hours only.")
    ham = detector.predict("Can you pick up milk on the way home?")
    assert spam.is_spam
    assert not ham.is_spam


def test_uses_same_preprocessing_as_training(detector):
    """The engine must apply clean_text + the training vocabulary, exactly like evaluation did."""
    message = "FREE entry!! Text WIN to 87121 to receive a £100 voucher. T&C's apply"
    expected = predict_probabilities(detector.model, detector.vectorizer, [clean_text(message)])[0]
    result = detector.predict(message)
    assert result.cleaned_text == clean_text(message)
    assert result.spam_probability == pytest.approx(float(expected), abs=1e-4)


def test_predictions_are_consistent(detector):
    message = "Congratulations! You've won a free iPhone. Text WIN to 80086"
    assert detector.predict(message) == detector.predict(message)


def test_case_and_punctuation_do_not_change_prediction(detector):
    assert (detector.predict("free prize call now").spam_probability
            == detector.predict("FREE PRIZE!!! Call now...").spam_probability)


def test_predict_many_matches_single_predictions_and_handles_bad_rows(detector):
    messages = ["Can you pick up milk?", "", "WIN a FREE prize, call 09061701461", None, "!!!"]
    results = detector.predict_many(messages)

    assert len(results) == len(messages)
    assert results[0] == detector.predict(messages[0])
    assert results[2] == detector.predict(messages[2])
    for i in (1, 3, 4):
        assert isinstance(results[i], InvalidMessageError)


def test_predict_many_empty_list(detector):
    assert detector.predict_many([]) == []


def test_predict_message_reuses_one_loaded_model():
    assert get_detector() is get_detector()
    assert predict_message("see you soon").label in ("SPAM", "HAM")


def test_missing_model_files_give_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.train"):
        SpamDetector(weights_path=tmp_path / "missing.pt",
                     vectorizer_path=tmp_path / "missing.json",
                     config_path=tmp_path / "missing_config.json")


# ---------- command line ----------

def test_command_line(capsys):
    assert main(["Can", "you", "pick", "up", "milk?"]) == 0
    assert "HAM" in capsys.readouterr().out
    assert main([]) == 1
    assert main(["!!!"]) == 1
    assert "Error" in capsys.readouterr().out
