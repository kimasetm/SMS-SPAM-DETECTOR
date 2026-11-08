"""Tests for temperature scaling and calibration metrics."""

import json

import numpy as np
import pytest
import torch

from src.calibration import calibrated_threshold, fit_temperature, sigmoid
from src.config import MODEL_CONFIG_FILE
from src.metrics import brier_score, expected_calibration_error, reliability_bins
from src.model import SpamClassifier, predict_probabilities
from src.predict import get_detector
from src.text_vectorizer import TextVectorizer
from src.train import train_model


def test_fit_temperature_recovers_overconfidence():
    """If a model's logits are 3x too large, the fitted temperature should be close to 3."""
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, 20000)
    labels = rng.random(20000) < sigmoid(true_logits)       # labels drawn from the true probabilities
    assert fit_temperature(true_logits * 3, labels) == pytest.approx(3.0, rel=0.1)
    assert fit_temperature(true_logits, labels) == pytest.approx(1.0, rel=0.1)


def test_calibrated_threshold_keeps_every_decision():
    logits = np.linspace(-6, 6, 500)
    raw, temperature, threshold = sigmoid(logits), 1.7, 0.10
    new_threshold = calibrated_threshold(threshold, temperature)
    assert np.array_equal(raw >= threshold, sigmoid(logits / temperature) >= new_threshold)


def test_calibration_metrics():
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == 0.0           # perfect
    assert expected_calibration_error([0, 0, 1, 1], [0.95] * 4) == pytest.approx(0.45)
    assert brier_score([0, 1], [0.0, 1.0]) == 0.0
    assert brier_score([0, 1], [1.0, 0.0]) == 1.0
    bins = reliability_bins([0, 1, 1], [0.05, 0.92, 0.98])
    assert [b["count"] for b in bins] == [1, 2]


def test_predict_probabilities_applies_temperature():
    torch.manual_seed(0)
    vectorizer = TextVectorizer(sequence_length=4).fit(["win free prize", "see you"])
    model = SpamClassifier(vectorizer.vocab_size, embedding_dim=4, hidden_dim=2)
    raw = predict_probabilities(model, vectorizer, ["win prize"])
    model.temperature = 2.0
    softened = predict_probabilities(model, vectorizer, ["win prize"])
    assert abs(softened[0] - 0.5) <= abs(raw[0] - 0.5)       # pulled toward 50%


def test_training_with_calibration_keeps_decisions():
    texts = ["win free prize now", "free cash claim", "see you at lunch", "running late sorry"] * 6
    labels = [1, 1, 0, 0] * 6
    settings = {"max_epochs": 3, "calibrate": False}
    plain_model, vec, plain_cfg, _ = train_model(texts, labels, texts, labels, settings, verbose=False)
    cal_model, _, cal_cfg, _ = train_model(texts, labels, texts, labels, {**settings, "calibrate": True},
                                           verbose=False)
    assert "temperature" in cal_cfg and "temperature" not in plain_cfg
    plain = predict_probabilities(plain_model, vec, texts) >= plain_cfg["threshold"]
    calibrated = predict_probabilities(cal_model, vec, texts) >= cal_cfg["threshold"]
    assert np.array_equal(plain, calibrated)


def test_saved_model_loads_its_temperature():
    saved = json.loads(MODEL_CONFIG_FILE.read_text(encoding="utf-8"))
    assert get_detector().model.temperature == saved.get("temperature", 1.0)
