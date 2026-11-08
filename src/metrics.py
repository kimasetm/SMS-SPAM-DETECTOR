"""Classification metrics, shared by test-set evaluation and the web app."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Average gap between confidence and reality, weighted by how many messages fall in each bin.

    Messages are grouped by predicted spam probability (0-10%, 10-20%, ...).
    In each group, the average prediction is compared with the share that
    really is spam. 0 = perfectly calibrated.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.clip((y_prob * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(y_prob[in_bin].mean() - y_true[in_bin].mean())
    return float(ece)


def brier_score(y_true, y_prob) -> float:
    """Mean squared difference between predicted probability and the true label (lower is better)."""
    return float(np.mean((np.asarray(y_prob, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def reliability_bins(y_true, y_prob, n_bins: int = 10) -> list:
    """Per-bin average prediction, actual spam rate and count (for a reliability diagram)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.clip((y_prob * n_bins).astype(int), 0, n_bins - 1)
    return [{"bin": b, "mean_predicted": float(y_prob[bins == b].mean()),
             "actual_spam_rate": float(y_true[bins == b].mean()), "count": int((bins == b).sum())}
            for b in range(n_bins) if (bins == b).any()]


def choose_threshold(y_true, y_prob) -> float:
    """Pick the decision threshold with the best F1-score.

    Call this on VALIDATION data only. Choosing the threshold on the test set
    would make the test results look better than they really are.
    Ties are broken by picking the threshold closest to 0.5.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    candidates = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in candidates]
    best = max(scores)
    tied = [t for t, s in zip(candidates, scores) if s == best]
    return float(min(tied, key=lambda t: abs(t - 0.5)))


def compute_metrics(y_true, y_prob, threshold: float = config.SPAM_THRESHOLD) -> dict:
    """Compute classification metrics. Spam (1) is the "positive" class.

    - accuracy:  share of all messages classified correctly
    - precision: of the messages flagged as spam, how many really were spam
                 (low precision = real messages wrongly blocked)
    - recall:    of the real spam messages, how many were caught
                 (low recall = spam slipping through)
    - F1:        the balance (harmonic mean) of precision and recall
    - ROC-AUC:   how well the probabilities rank spam above ham, across all thresholds
    """
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "num_messages": int(len(y_true)),
        "num_spam": int(y_true.sum()),
        "num_ham": int(len(y_true) - y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # ROC-AUC is undefined when only one class is present
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true.tolist())) == 2 else None,
        "confusion_matrix": {
            "true_negative": int(tn),    # ham correctly kept
            "false_positive": int(fp),   # ham wrongly flagged as spam
            "false_negative": int(fn),   # spam that slipped through
            "true_positive": int(tp),    # spam correctly caught
        },
    }
