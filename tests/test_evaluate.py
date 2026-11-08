"""Tests for evaluation metrics and charts."""

import numpy as np
import pytest
import torch

from src.evaluate import plot_confusion_matrix, plot_training_history
from src.metrics import compute_metrics
from src.model import SpamClassifier, predict_probabilities
from src.text_vectorizer import TextVectorizer


def test_metrics_for_known_predictions():
    # 4 spam, 4 ham. The model catches 3 spam, misses 1, and wrongly flags 1 ham.
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_prob = [0.9, 0.8, 0.7, 0.2, 0.6, 0.1, 0.3, 0.05]
    m = compute_metrics(y_true, y_prob)

    assert m["confusion_matrix"] == {"true_negative": 3, "false_positive": 1,
                                     "false_negative": 1, "true_positive": 3}
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["precision"] == pytest.approx(3 / 4)   # 3 of 4 flagged messages were spam
    assert m["recall"] == pytest.approx(3 / 4)      # 3 of 4 spam messages were caught
    assert m["f1"] == pytest.approx(0.75)
    assert m["num_spam"] == 4 and m["num_ham"] == 4


def test_threshold_changes_predictions():
    y_true = [1, 0]
    y_prob = [0.6, 0.4]
    assert compute_metrics(y_true, y_prob, threshold=0.5)["accuracy"] == 1.0
    assert compute_metrics(y_true, y_prob, threshold=0.7)["recall"] == 0.0


def test_metrics_when_model_predicts_no_spam():
    """Predicting 'ham' for everything gets high accuracy but zero recall. This is why accuracy alone is misleading."""
    y_true = [0] * 9 + [1]
    m = compute_metrics(y_true, [0.1] * 10)
    assert m["accuracy"] == pytest.approx(0.9)
    assert m["precision"] == 0.0   # no division-by-zero crash
    assert m["recall"] == 0.0


def test_roc_auc_is_none_with_one_class():
    assert compute_metrics([0, 0, 0], [0.1, 0.2, 0.3])["roc_auc"] is None


def test_predict_probabilities_returns_one_probability_per_message():
    torch.manual_seed(0)
    vectorizer = TextVectorizer(sequence_length=5).fit(["win free prize", "see you soon"])
    model = SpamClassifier(vectorizer.vocab_size, embedding_dim=4, hidden_dim=2)
    texts = ["win prize", "see you", "unknown words here"]

    probs = predict_probabilities(model, vectorizer, texts, batch_size=2)   # batches of 2 and 1
    assert probs.shape == (3,)
    assert np.all((probs >= 0) & (probs <= 1))
    assert predict_probabilities(model, vectorizer, []).shape == (0,)


def test_charts_are_saved(tmp_path):
    history = {"train_loss": [0.5, 0.3, 0.2], "val_loss": [0.6, 0.4, 0.45],
               "train_accuracy": [0.8, 0.9, 0.95], "val_accuracy": [0.75, 0.85, 0.84], "best_epoch": 2}
    cm = {"true_negative": 50, "false_positive": 2, "false_negative": 5, "true_positive": 10}

    plot_training_history(history, tmp_path / "history.png")
    plot_confusion_matrix(cm, tmp_path / "figures" / "cm.png")   # creates missing folder
    assert (tmp_path / "history.png").stat().st_size > 0
    assert (tmp_path / "figures" / "cm.png").stat().st_size > 0
