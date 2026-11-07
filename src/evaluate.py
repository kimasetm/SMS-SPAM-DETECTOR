"""Evaluate the trained model on the held-out test set.

Run from the project root (after training):
    python -m src.evaluate

This is the first and only time the model sees the test set, so these
numbers are an honest estimate of how it performs on new messages.

Outputs:
  reports/metrics.json                       accuracy, precision, recall, F1, ROC-AUC, confusion matrix
  reports/misclassified_test_messages.csv    test messages the model got wrong (for error analysis)
  reports/figures/training_history.png       training vs validation loss and accuracy per epoch
  reports/figures/confusion_matrix.png       confusion matrix on the test set
"""

import json

import matplotlib

matplotlib.use("Agg")   # draw to image files; no window needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config
from src.model import load_artifacts, predict_probabilities
from src.preprocessing import load_processed_split

# Chart colors (validated for color-blind readers)
TRAIN_COLOR = "#2a78d6"     # blue
VAL_COLOR = "#eb6834"       # orange
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE_RAMP = ["#f0f6fe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]


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
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) == 2 else None,
        "confusion_matrix": {
            "true_negative": int(tn),    # ham correctly kept
            "false_positive": int(fp),   # ham wrongly flagged as spam
            "false_negative": int(fn),   # spam that slipped through
            "true_positive": int(tp),    # spam correctly caught
        },
    }


def _style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(AXIS)
    ax.tick_params(colors=MUTED_INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_training_history(history: dict, path) -> None:
    """Two side-by-side charts: loss and accuracy per epoch, training vs validation."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    best_epoch = history.get("best_epoch")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    panels = [("Loss", "train_loss", "val_loss"), ("Accuracy", "train_accuracy", "val_accuracy")]
    for ax, (title, train_key, val_key) in zip(axes, panels):
        _style_axes(ax)
        ax.plot(epochs, history[train_key], color=TRAIN_COLOR, linewidth=2, label="Training")
        ax.plot(epochs, history[val_key], color=VAL_COLOR, linewidth=2, linestyle="--", label="Validation")
        # Direct labels at the end of each line
        for key, color, name in ((train_key, TRAIN_COLOR, "Training"), (val_key, VAL_COLOR, "Validation")):
            ax.annotate(name, (epochs[-1], history[key][-1]), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=MUTED_INK)
        if best_epoch:
            ax.axvline(best_epoch, color=AXIS, linewidth=1, linestyle=":")
            ax.annotate(f"best epoch ({best_epoch})", (best_epoch, 0), xycoords=("data", "axes fraction"),
                        xytext=(-4, 6), textcoords="offset points", ha="right", fontsize=8, color=MUTED_INK)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Epoch", fontsize=9, color=MUTED_INK)
        ax.set_xlim(1, epochs[-1] + 2.5)   # room for the direct labels
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # One shared legend for both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=MUTED_INK,
               loc="upper right", ncol=2)
    fig.suptitle("Training history", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def plot_confusion_matrix(cm: dict, path) -> None:
    """Confusion matrix. Color shows the share of each true class (row %)."""
    counts = np.array([[cm["true_negative"], cm["false_positive"]],
                       [cm["false_negative"], cm["true_positive"]]])
    row_share = counts / counts.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    ax.imshow(row_share, cmap=cmap, vmin=0, vmax=1)

    names = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            text_color = "white" if row_share[i, j] > 0.55 else INK
            ax.text(j, i - 0.12, f"{counts[i, j]}", ha="center", va="center",
                    fontsize=18, color=text_color, fontweight="bold")
            ax.text(j, i + 0.18, f"{row_share[i, j]:.1%} · {names[i][j]}", ha="center",
                    va="center", fontsize=8.5, color=text_color)

    ax.set_xticks([0, 1], ["Ham", "Spam"])
    ax.set_yticks([0, 1], ["Ham", "Spam"])
    ax.set_xlabel("Predicted", fontsize=10, color=MUTED_INK)
    ax.set_ylabel("Actual", fontsize=10, color=MUTED_INK)
    ax.tick_params(colors=MUTED_INK, length=0)
    # White gaps between cells
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Confusion matrix (test set)", loc="left", fontsize=12, color=INK)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    model, vectorizer, model_config = load_artifacts()
    test_df = load_processed_split(config.TEST_FILE)
    texts = test_df["clean_message"].tolist()
    y_true = test_df["label_num"].to_numpy()

    y_prob = predict_probabilities(model, vectorizer, texts)
    metrics = compute_metrics(y_true, y_prob)
    y_pred = (y_prob >= config.SPAM_THRESHOLD).astype(int)

    print(f"Test set: {metrics['num_messages']} messages ({metrics['num_spam']} spam)\n")
    print(classification_report(y_true, y_pred, target_names=["ham", "spam"], digits=4))
    cm = metrics["confusion_matrix"]
    print(f"Confusion matrix: TN={cm['true_negative']} FP={cm['false_positive']} "
          f"FN={cm['false_negative']} TP={cm['true_positive']}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"test": metrics, "model": {"best_epoch": model_config["training"]["best_epoch"],
                                         "best_val_accuracy": model_config["training"]["best_val_accuracy"]}}
    config.METRICS_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Save the mistakes for error analysis (useful for the report)
    wrong = test_df.assign(spam_probability=y_prob.round(4), predicted=y_pred)[y_pred != y_true]
    wrong = wrong.rename(columns={"label_num": "actual"})
    wrong[["clean_message", "actual", "predicted", "spam_probability"]].to_csv(
        config.MISCLASSIFIED_FILE, index=False)

    history = json.loads(config.HISTORY_FILE.read_text(encoding="utf-8"))
    plot_training_history(history, config.FIGURES_DIR / "training_history.png")
    plot_confusion_matrix(cm, config.FIGURES_DIR / "confusion_matrix.png")
    print(f"\nSaved metrics, misclassified messages and figures to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
