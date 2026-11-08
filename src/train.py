"""Train the SmartSpam neural network.

Run from the project root (after preprocessing):
    python -m src.train

Pipeline:
  1. Load the training and validation sets (the test set is NOT used here;
     it is kept hidden until final evaluation in src/evaluate.py).
  2. Fit the TextVectorizer vocabulary on the training messages only.
  3. Train with mini-batches, checking validation loss after every epoch.
  4. Early stopping: stop when validation loss hasn't improved for a few
     epochs, and keep the weights from the best epoch.
  5. Save the model weights, vectorizer, config and training history.
"""

import copy
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src import config
from src.calibration import calibrated_threshold, fit_temperature
from src.metrics import choose_threshold
from src.model import build_model, predict_logits, predict_probabilities, save_artifacts
from src.monitor import compute_profile, save_baseline
from src.preprocessing import load_processed_split
from src.text_vectorizer import TextVectorizer

DEFAULT_SETTINGS = {
    "max_tokens": config.MAX_TOKENS,
    "sequence_length": config.SEQUENCE_LENGTH,
    "embedding_dim": config.EMBEDDING_DIM,
    "hidden_dim": config.HIDDEN_DIM,
    "dropout": config.DROPOUT,
    "batch_size": config.BATCH_SIZE,
    "learning_rate": config.LEARNING_RATE,
    "max_epochs": config.MAX_EPOCHS,
    "patience": config.EARLY_STOPPING_PATIENCE,
    "seed": config.RANDOM_SEED,
    "spam_weight": config.SPAM_WEIGHT,
    "tune_threshold": config.TUNE_THRESHOLD,
    "architecture": config.ARCHITECTURE,
    "conv_filters": config.CONV_FILTERS,
    "kernel_size": config.KERNEL_SIZE,
    "repair_misspellings": config.REPAIR_MISSPELLINGS,
    "calibrate": config.CALIBRATE,
}


def resolve_spam_weight(setting, labels) -> float:
    """Turn the spam_weight setting into a number (see SPAM_WEIGHT in config.py)."""
    if setting is None:
        return 1.0
    labels = np.asarray(labels)
    ham_to_spam = (labels == 0).sum() / max((labels == 1).sum(), 1)
    if setting == "balanced":
        return float(ham_to_spam)
    if setting == "sqrt":
        return float(np.sqrt(ham_to_spam))
    return float(setting)


def set_seed(seed: int) -> None:
    """Fix every source of randomness so training gives the same result each run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_loader(token_ids, labels, batch_size, shuffle, seed=None) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )
    # A seeded generator makes the shuffling order reproducible
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def run_epoch(model, loader, loss_fn, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise only evaluates.

    Returns (average loss, accuracy).
    """
    training = optimizer is not None
    model.train(training)   # dropout on while training, off while evaluating

    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch_ids, batch_labels in loader:
            logits = model(batch_ids)
            loss = loss_fn(logits, batch_labels)

            if training:
                optimizer.zero_grad()   # clear gradients from the previous batch
                loss.backward()         # backpropagation: compute how each weight affected the loss
                optimizer.step()        # nudge each weight to reduce the loss

            total_loss += loss.item() * len(batch_labels)
            predictions = (logits > 0).float()       # logit > 0  <=>  probability > 0.5
            correct += (predictions == batch_labels).sum().item()
            count += len(batch_labels)

    return total_loss / count, correct / count


def train_model(train_texts, train_labels, val_texts, val_labels,
                settings=None, verbose=True):
    """Train a SpamClassifier. Returns (model, vectorizer, model_config, history)."""
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    set_seed(settings["seed"])

    # Fit the vocabulary on TRAINING data only (prevents data leakage)
    vectorizer = TextVectorizer(settings["max_tokens"], settings["sequence_length"],
                                settings["repair_misspellings"])
    vectorizer.fit(train_texts)

    train_loader = make_loader(vectorizer.transform(train_texts), train_labels,
                               settings["batch_size"], shuffle=True, seed=settings["seed"])
    val_loader = make_loader(vectorizer.transform(val_texts), val_labels,
                             settings["batch_size"], shuffle=False)

    model_config = {
        "vocab_size": vectorizer.vocab_size,
        "embedding_dim": settings["embedding_dim"],
        "hidden_dim": settings["hidden_dim"],
        "dropout": settings["dropout"],
        "architecture": settings["architecture"],
    }
    if settings["architecture"] == "cnn":
        model_config["conv_filters"] = settings["conv_filters"]
        model_config["kernel_size"] = settings["kernel_size"]
    model = build_model(model_config)

    # Binary cross-entropy loss: measures how far the predicted spam
    # probability is from the true label (0 or 1). The "WithLogits" version
    # applies the sigmoid internally, which is more numerically stable.
    # pos_weight makes mistakes on spam count more, so the model doesn't just
    # learn to say "ham" (the majority class).
    spam_weight = resolve_spam_weight(settings["spam_weight"], train_labels)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(spam_weight))
    # Adam: a popular optimizer that adapts the step size for each weight
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, settings["max_epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if verbose:
            print(f"Epoch {epoch:>2} | train loss {train_loss:.4f} acc {train_acc:.4f} "
                  f"| val loss {val_loss:.4f} acc {val_acc:.4f}")

        # Early stopping: remember the best weights; stop if no improvement
        # for `patience` epochs (the model has started to overfit).
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings["patience"]:
                if verbose:
                    print(f"Early stopping: no validation improvement for {settings['patience']} epochs.")
                break

    model.load_state_dict(best_state)
    model.eval()

    # Choose the decision threshold on the VALIDATION set (never the test set)
    threshold = config.SPAM_THRESHOLD
    if settings["tune_threshold"]:
        val_probs = predict_probabilities(model, vectorizer, list(val_texts))
        threshold = choose_threshold(val_labels, val_probs)
    model_config["threshold"] = threshold

    # Calibrate confidence with temperature scaling, also on the VALIDATION set.
    # The threshold is converted to the calibrated scale, so decisions don't change.
    model.temperature = 1.0
    if settings["calibrate"]:
        temperature = fit_temperature(predict_logits(model, vectorizer, list(val_texts)), val_labels)
        model.temperature = temperature
        model_config["temperature"] = temperature
        model_config["threshold_uncalibrated"] = threshold
        model_config["threshold"] = calibrated_threshold(threshold, temperature)

    history["best_epoch"] = best_epoch
    model_config["training"] = {
        **{k: settings[k] for k in ("max_tokens", "sequence_length", "batch_size",
                                    "learning_rate", "max_epochs", "patience", "seed",
                                    "repair_misspellings")},
        "spam_weight": spam_weight,
        "tune_threshold": settings["tune_threshold"],
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": history["val_accuracy"][best_epoch - 1],
        "train_size": len(train_texts),
        "val_size": len(val_texts),
        "torch_version": torch.__version__,
    }
    return model, vectorizer, model_config, history


def main() -> None:
    train_df = load_processed_split(config.TRAIN_FILE)
    val_df = load_processed_split(config.VAL_FILE)
    print(f"Training on {len(train_df)} messages, validating on {len(val_df)} messages\n")

    model, vectorizer, model_config, history = train_model(
        train_df["clean_message"].tolist(), train_df["label_num"].tolist(),
        val_df["clean_message"].tolist(), val_df["label_num"].tolist(),
    )

    save_artifacts(model, vectorizer, model_config)
    config.HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Baseline for drift monitoring: what the validation messages look like to this model
    save_baseline(compute_profile(val_df["clean_message"].tolist(), model, vectorizer, model_config["threshold"]))

    t = model_config["training"]
    print(f"\nBest epoch: {t['best_epoch']} of {t['epochs_run']} "
          f"(val loss {t['best_val_loss']:.4f}, val accuracy {t['best_val_accuracy']:.4f})")
    print(f"Spam weight: {t['spam_weight']:.2f} | Decision threshold: {model_config['threshold']:.2f}"
          + (f" | Temperature: {model_config['temperature']:.3f}" if "temperature" in model_config else ""))
    print(f"Vocabulary size: {model_config['vocab_size']} | "
          f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Saved model files to {config.MODEL_DIR}")


if __name__ == "__main__":
    main()
