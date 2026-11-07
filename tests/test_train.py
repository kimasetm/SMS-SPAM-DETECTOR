"""Tests for the training pipeline, using a tiny made-up dataset so they run in seconds."""

import torch

from src.train import train_model

SPAM = ["win free prize now", "free cash call now", "claim your prize now", "win cash free"]
HAM = ["see you at lunch", "running late sorry", "call you later tonight", "lunch at noon"]
TEXTS = (SPAM + HAM) * 5
LABELS = ([1] * len(SPAM) + [0] * len(HAM)) * 5

FAST = {"max_epochs": 40, "patience": 40, "batch_size": 8, "learning_rate": 0.01,
        "embedding_dim": 8, "hidden_dim": 4, "sequence_length": 6}


def test_model_learns_simple_pattern():
    model, vectorizer, _, history = train_model(TEXTS, LABELS, TEXTS, LABELS, FAST, verbose=False)

    assert history["train_loss"][-1] < history["train_loss"][0]
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(vectorizer.transform(["free prize now", "see you at lunch"]))))
    assert probs[0] > 0.5   # spam-like
    assert probs[1] < 0.5   # ham-like


def test_training_is_reproducible():
    run1 = train_model(TEXTS, LABELS, TEXTS, LABELS, {**FAST, "max_epochs": 3}, verbose=False)
    run2 = train_model(TEXTS, LABELS, TEXTS, LABELS, {**FAST, "max_epochs": 3}, verbose=False)
    for w1, w2 in zip(run1[0].state_dict().values(), run2[0].state_dict().values()):
        assert torch.equal(w1, w2)


def test_vocabulary_is_fit_on_training_data_only():
    _, vectorizer, _, _ = train_model(TEXTS, LABELS, ["unseen validation words"], [0],
                                      {**FAST, "max_epochs": 1}, verbose=False)
    assert "unseen" not in vectorizer.word_to_id
    assert "prize" in vectorizer.word_to_id


def test_early_stopping_and_history():
    _, _, model_config, history = train_model(
        TEXTS, LABELS, TEXTS, LABELS, {**FAST, "max_epochs": 30, "patience": 2}, verbose=False)

    epochs_run = model_config["training"]["epochs_run"]
    assert 1 <= epochs_run <= 30
    for key in ("train_loss", "train_accuracy", "val_loss", "val_accuracy"):
        assert len(history[key]) == epochs_run
    # The best epoch has the lowest validation loss
    best = model_config["training"]["best_epoch"]
    assert history["val_loss"][best - 1] == min(history["val_loss"])


def test_returned_model_is_in_eval_mode():
    model, _, _, _ = train_model(TEXTS, LABELS, TEXTS, LABELS, {**FAST, "max_epochs": 1}, verbose=False)
    assert not model.training
