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
import pandas as pd
from sklearn.metrics import classification_report

from src import config
from src.metrics import brier_score, compute_metrics, expected_calibration_error, reliability_bins
from src.model import load_artifacts, predict_probabilities
from src.robustness import disguised_spam_recall
from src.preprocessing import load_processed_split

# Chart colors (validated for color-blind readers)
TRAIN_COLOR = "#2a78d6"     # blue
VAL_COLOR = "#eb6834"       # orange
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE_RAMP = ["#f0f6fe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]


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


def plot_confusion_matrix(cm: dict, path, title: str = "Confusion matrix (test set)") -> None:
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
    ax.set_title(title, loc="left", fontsize=12, color=INK)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def evaluate_split(name, test_df, model, vectorizer, threshold):
    """Compute and print metrics for one test set. Returns (metrics, misclassified rows)."""
    texts = test_df["clean_message"].tolist()
    y_true = test_df["label_num"].to_numpy()
    y_prob = predict_probabilities(model, vectorizer, texts)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    y_pred = (y_prob >= threshold).astype(int)

    # Recall for smishing (SMS phishing) on its own, when the test set labels it
    if "category" in test_df and (test_df["category"] == "smishing").any():
        is_smishing = (test_df["category"] == "smishing").to_numpy()
        metrics["smishing_count"] = int(is_smishing.sum())
        metrics["smishing_caught"] = int(y_pred[is_smishing].sum())
        metrics["smishing_recall"] = float(y_pred[is_smishing].mean())

    print(f"\n=== {name}: {metrics['num_messages']} messages ({metrics['num_spam']} spam) ===")
    print(classification_report(y_true, y_pred, target_names=["ham", "spam"], digits=4, zero_division=0))
    cm = metrics["confusion_matrix"]
    print(f"Confusion matrix: TN={cm['true_negative']} FP={cm['false_positive']} "
          f"FN={cm['false_negative']} TP={cm['true_positive']}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    if "smishing_recall" in metrics:
        print(f"Smishing caught: {metrics['smishing_caught']} of {metrics['smishing_count']} "
              f"({metrics['smishing_recall']:.2%})")

    # Robustness: the same spam messages with words disguised ("free" -> "fr33")
    robust = disguised_spam_recall(model, vectorizer, test_df.loc[test_df["label_num"] == 1, "clean_message"],
                                   threshold)
    metrics["disguised_spam"] = robust
    if robust["count"]:
        print(f"Disguised spam still caught: {robust['caught_disguised']} of {robust['count']} "
              f"({robust['recall_disguised']:.2%}; undisguised: {robust['caught_original']})")

    # Calibration: how well the confidence matches reality, before and after temperature scaling
    temperature = getattr(model, "temperature", 1.0)
    model.temperature = 1.0
    y_prob_raw = predict_probabilities(model, vectorizer, texts)
    model.temperature = temperature
    metrics["calibration"] = {
        "temperature": temperature,
        "ece_uncalibrated": expected_calibration_error(y_true, y_prob_raw),
        "ece": expected_calibration_error(y_true, y_prob),
        "brier_uncalibrated": brier_score(y_true, y_prob_raw),
        "brier": brier_score(y_true, y_prob),
        "bins_uncalibrated": reliability_bins(y_true, y_prob_raw),
        "bins": reliability_bins(y_true, y_prob),
    }
    c = metrics["calibration"]
    print(f"Calibration error (ECE): {c['ece_uncalibrated']:.4f} -> {c['ece']:.4f} | "
          f"Brier score: {c['brier_uncalibrated']:.4f} -> {c['brier']:.4f} (temperature {temperature:.3f})")

    wrong = test_df.assign(spam_probability=y_prob.round(4), predicted=y_pred, test_set=name)[y_pred != y_true]
    return metrics, wrong.rename(columns={"label_num": "actual"})


def plot_reliability(panels: dict, path) -> None:
    """Reliability diagram: predicted spam probability vs the actual spam rate, per bin.

    panels: {"Panel title": calibration dict from evaluate_split}. Points on the
    dashed diagonal are perfectly calibrated; point size shows how many messages
    fall in each bin.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), squeeze=False)
    for ax, (title, cal) in zip(axes[0], panels.items()):
        _style_axes(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
        # Points only (no connecting lines): many bins hold just 1-3 messages,
        # and lines would make that noise look like a trend.
        for key, color, marker, label in (("bins_uncalibrated", TRAIN_COLOR, "s", "Before calibration"),
                                          ("bins", VAL_COLOR, "o", "After calibration")):
            bins = cal[key]
            x = [b["mean_predicted"] for b in bins]
            y = [b["actual_spam_rate"] for b in bins]
            sizes = [20 + 12 * np.sqrt(b["count"]) for b in bins]
            ax.scatter(x, y, s=sizes, color=color, marker=marker, edgecolor="white", linewidth=1.5,
                       alpha=0.85, zorder=3, label=label)
        ax.text(0.98, 0.03, "Point size = number of messages", transform=ax.transAxes, ha="right",
                fontsize=8, color=MUTED_INK)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Predicted spam probability", fontsize=9, color=MUTED_INK)
        ax.set_ylabel("Actual share of spam", fontsize=9, color=MUTED_INK)
        ax.set_title(f"{title}\nECE {cal['ece_uncalibrated']:.3f} → {cal['ece']:.3f}",
                     loc="left", fontsize=10, color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=MUTED_INK, loc="upper right", ncol=2,
               markerscale=0.6)
    fig.suptitle("Calibration (reliability diagram)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    model, vectorizer, model_config = load_artifacts()
    # Use the decision threshold chosen on the validation set during training
    threshold = model_config.get("threshold", config.SPAM_THRESHOLD)

    test_metrics, wrong = evaluate_split("UCI test set", load_processed_split(config.TEST_FILE),
                                         model, vectorizer, threshold)
    report = {"test": test_metrics}
    all_wrong = [wrong]

    # Newer messages (only if the modern dataset was downloaded and preprocessed)
    if config.MODERN_TEST_FILE.exists():
        modern_metrics, modern_wrong = evaluate_split(
            "Modern test set", load_processed_split(config.MODERN_TEST_FILE), model, vectorizer, threshold)
        report["test_modern"] = modern_metrics
        all_wrong.append(modern_wrong)

    report["model"] = {"best_epoch": model_config["training"]["best_epoch"],
                       "best_val_accuracy": model_config["training"]["best_val_accuracy"]}
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Save the mistakes for error analysis (useful for the report)
    columns = ["test_set", "clean_message", "actual", "predicted", "spam_probability"]
    pd.concat(all_wrong)[columns].to_csv(config.MISCLASSIFIED_FILE, index=False)

    history = json.loads(config.HISTORY_FILE.read_text(encoding="utf-8"))
    plot_training_history(history, config.FIGURES_DIR / "training_history.png")
    plot_confusion_matrix(test_metrics["confusion_matrix"], config.FIGURES_DIR / "confusion_matrix.png")
    if "test_modern" in report:
        plot_confusion_matrix(report["test_modern"]["confusion_matrix"],
                              config.FIGURES_DIR / "confusion_matrix_modern.png",
                              title="Confusion matrix (modern test set)")
    panels = {"UCI test set": test_metrics["calibration"]}
    if "test_modern" in report:
        panels["Modern test set"] = report["test_modern"]["calibration"]
    plot_reliability(panels, config.FIGURES_DIR / "calibration.png")
    print(f"\nSaved metrics, misclassified messages and figures to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
