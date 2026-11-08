"""Confidence calibration with temperature scaling.

A model is "calibrated" when its confidence matches reality: of all messages
it rates "90% spam", about 90% really are spam. Neural networks are often
over- or under-confident.

Temperature scaling (Guo et al., 2017) fixes this with one number, T:

    calibrated probability = sigmoid(logit / T)

  T > 1 softens over-confident predictions (pulls them toward 50%)
  T < 1 sharpens under-confident ones

T is chosen on the VALIDATION set to make the probabilities match the true
labels as closely as possible (lowest log loss). Dividing by T never changes
the ORDER of the scores, so with a matching threshold every SPAM/HAM decision
stays exactly the same; only the confidence numbers change.
"""

import numpy as np

# Candidate temperatures from 0.1 to 10, evenly spaced on a log scale
TEMPERATURE_GRID = np.logspace(-1, 1, 401)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def log_loss(labels, probabilities) -> float:
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    labels = np.asarray(labels)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def fit_temperature(logits, labels) -> float:
    """Return the temperature with the lowest log loss on (validation) data."""
    logits = np.asarray(logits, dtype=float)
    losses = [log_loss(labels, sigmoid(logits / t)) for t in TEMPERATURE_GRID]
    return float(TEMPERATURE_GRID[int(np.argmin(losses))])


def calibrated_threshold(threshold: float, temperature: float) -> float:
    """Convert a threshold on uncalibrated probabilities to the calibrated scale.

    A message with uncalibrated probability exactly at `threshold` gets the
    returned calibrated probability, so every decision stays the same.
    """
    threshold = min(max(threshold, 1e-7), 1 - 1e-7)
    logit = np.log(threshold / (1 - threshold))
    return float(sigmoid(logit / temperature))
