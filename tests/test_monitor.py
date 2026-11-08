"""Tests for drift monitoring."""

import pandas as pd
import pytest

from src.monitor import check_messages, compare_to_baseline, compute_profile, format_report, main
from src.predict import get_detector

BASELINE = {"num_messages": 100, "unknown_word_rate": 0.04, "avg_words": 16.0, "url_rate": 0.02,
            "phone_rate": 0.10, "predicted_spam_rate": 0.15, "uncertain_rate": 0.05}


def test_no_warnings_when_batch_looks_like_baseline():
    assert compare_to_baseline(dict(BASELINE), BASELINE) == []


@pytest.mark.parametrize("signal, value", [
    ("unknown_word_rate", 0.40),      # e.g. another language
    ("predicted_spam_rate", 0.50),    # traffic changed
    ("uncertain_rate", 0.30),         # model unsure
    ("avg_words", 40.0),              # much longer messages
])
def test_large_changes_raise_a_warning(signal, value):
    warnings = compare_to_baseline({**BASELINE, signal: value}, BASELINE)
    assert [w["signal"] for w in warnings] == [signal]


def test_small_changes_do_not_raise_warnings():
    small = {**BASELINE, "unknown_word_rate": 0.06, "predicted_spam_rate": 0.20, "avg_words": 18.0}
    assert compare_to_baseline(small, BASELINE) == []


def test_profile_of_real_messages():
    detector = get_detector()
    profile = compute_profile(["call phonetoken now to claim your prize", "see you at lunch",
                               "visit urltoken"], detector.model, detector.vectorizer, detector.threshold)
    assert profile["num_messages"] == 3
    assert profile["phone_rate"] == pytest.approx(1 / 3)
    assert profile["url_rate"] == pytest.approx(1 / 3)
    assert 0 <= profile["unknown_word_rate"] <= 1
    with pytest.raises(ValueError):
        compute_profile([], detector.model, detector.vectorizer, detector.threshold)


def test_another_language_is_flagged_as_drift():
    spanish = ["Hola, nos vemos mañana para almorzar?", "Te quiero mucho, buenas noches",
               "Puedes comprar leche de camino a casa?", "Tu cuenta bancaria ha sido bloqueada"]
    result = check_messages(spanish, get_detector())
    assert "unknown_word_rate" in [w["signal"] for w in result["warnings"]]
    assert "WARNING" in format_report(result)


def test_command_line(tmp_path, capsys):
    path = tmp_path / "msgs.csv"
    pd.DataFrame({"message": ["see you at lunch", "call me later tonight"]}).to_csv(path, index=False)
    assert main([str(path)]) == 0
    assert "Messages checked: 2" in capsys.readouterr().out
    assert main([]) == 1
    assert main([str(tmp_path / "missing.csv")]) == 1
