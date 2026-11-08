"""Tests for class-imbalance handling and the tuned decision threshold."""

import json

import pandas as pd
import pytest

from src.batch import score_labeled_results
from src.config import MODEL_CONFIG_FILE
from src.metrics import choose_threshold
from src.predict import SpamDetector, get_detector
from src.train import resolve_spam_weight, train_model


# ---------- choose_threshold ----------

def test_choose_threshold_finds_best_f1_cutoff():
    # Spam probabilities 0.35 and 0.40 are only caught with a threshold <= 0.35
    y_true = [1, 1, 1, 0, 0, 0]
    y_prob = [0.9, 0.40, 0.35, 0.30, 0.1, 0.05]
    # Every threshold from 0.31 to 0.35 classifies all six correctly (F1 = 1);
    # the tie is broken by choosing the one closest to 0.5
    assert choose_threshold(y_true, y_prob) == pytest.approx(0.35)


def test_choose_threshold_prefers_value_closest_to_half_on_ties():
    # Perfectly separable: every threshold between 0.2 and 0.8 gives F1 = 1
    assert choose_threshold([1, 0], [0.8, 0.2]) == pytest.approx(0.5)


def test_choose_threshold_stays_in_range():
    assert 0.05 <= choose_threshold([0, 0, 1], [0.0, 0.0, 0.0]) <= 0.95


# ---------- spam weight ----------

def test_resolve_spam_weight():
    labels = [0] * 9 + [1]   # 9 ham : 1 spam
    assert resolve_spam_weight(None, labels) == 1.0
    assert resolve_spam_weight("balanced", labels) == pytest.approx(9.0)
    assert resolve_spam_weight("sqrt", labels) == pytest.approx(3.0)
    assert resolve_spam_weight(2.5, labels) == 2.5


def test_training_saves_threshold_and_weight():
    texts = ["win free prize now", "see you at lunch"] * 10
    labels = [1, 0] * 10
    _, _, model_config, _ = train_model(texts, labels, texts, labels,
                                        {"max_epochs": 2, "spam_weight": "balanced"}, verbose=False)
    assert 0.05 <= model_config["threshold"] <= 0.95
    assert model_config["training"]["spam_weight"] == pytest.approx(1.0)   # 10 ham : 10 spam


def test_threshold_tuning_can_be_turned_off():
    texts = ["win free prize now", "see you at lunch"] * 10
    labels = [1, 0] * 10
    _, _, model_config, _ = train_model(texts, labels, texts, labels,
                                        {"max_epochs": 1, "tune_threshold": False}, verbose=False)
    assert model_config["threshold"] == 0.5


# ---------- the saved model uses its tuned threshold everywhere ----------

def test_detector_uses_threshold_saved_with_model():
    saved = json.loads(MODEL_CONFIG_FILE.read_text(encoding="utf-8"))["threshold"]
    assert get_detector().threshold == saved


def test_detector_threshold_can_be_overridden():
    assert SpamDetector(threshold=0.9).threshold == 0.9


def test_prediction_labels_follow_the_threshold():
    detector = get_detector()
    message = "Your package could not be delivered. Reschedule here: http://parcel-redelivery.info"
    probability = detector.predict(message).spam_probability
    strict = SpamDetector(threshold=min(probability + 0.01, 1.0))
    lenient = SpamDetector(threshold=max(probability - 0.01, 0.0))
    assert not strict.predict(message).is_spam
    assert lenient.predict(message).is_spam


def test_batch_scores_use_the_same_threshold_as_predictions():
    results = pd.DataFrame({"label": ["spam", "ham"], "prediction": ["SPAM", "HAM"],
                            "spam_probability": [0.35, 0.10]})
    # With the 0.30 cutoff both rows are correct; with 0.5 the spam row would count as missed
    assert score_labeled_results(results, "label", threshold=0.30)["metrics"]["recall"] == 1.0
    assert score_labeled_results(results, "label", threshold=0.50)["metrics"]["recall"] == 0.0
