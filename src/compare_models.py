"""Compare model variants using the VALIDATION set only.

Run from the project root (after preprocessing):
    python -m src.compare_models               # v3 architecture experiment (default)
    python -m src.compare_models imbalance     # v2 class-imbalance experiment

Selection rule (decided before running): pick the variant with the highest
validation F1-score. Every variant chooses its decision threshold on the
validation set. The test sets are NOT used here.

Experiments:
  imbalance (v2):     spam weighting vs. a tuned threshold, average pooling
  architecture (v3):  average pooling vs. a 1D CNN, more epochs, and
                      character-level repair of disguised words ("fr33")

For information, each variant also reports how many disguised validation
spam messages it still catches (see src/robustness.py). This is NOT used
for selection.
"""

import json
import sys

from src import config
from src.metrics import compute_metrics
from src.model import predict_probabilities
from src.preprocessing import load_processed_split
from src.robustness import disguised_spam_recall
from src.train import train_model

EXPERIMENTS = {
    "imbalance": {
        "file": config.MODEL_COMPARISON_FILE,
        # The v2 experiment used the v2 model settings
        "base": {"architecture": "avg_pool", "max_epochs": 30, "repair_misspellings": False, "calibrate": False},
        "variants": {
            "v1.0 baseline (no weight, threshold 0.5)": {"spam_weight": None, "tune_threshold": False},
            "No weight, tuned threshold": {"spam_weight": None, "tune_threshold": True},
            "Sqrt spam weight, tuned threshold": {"spam_weight": "sqrt", "tune_threshold": True},
            "Balanced spam weight, tuned threshold": {"spam_weight": "balanced", "tune_threshold": True},
        },
    },
    "architecture": {
        "file": config.ARCHITECTURE_COMPARISON_FILE,
        "base": {"spam_weight": None, "tune_threshold": True, "calibrate": False},
        "variants": {
            "Average pooling, 30 epochs (v2 settings)": {"architecture": "avg_pool", "max_epochs": 30,
                                                         "repair_misspellings": False},
            "Average pooling, 60 epochs": {"architecture": "avg_pool", "max_epochs": 60,
                                           "repair_misspellings": False},
            "Average pooling, 60 epochs + misspelling repair": {"architecture": "avg_pool", "max_epochs": 60,
                                                                "repair_misspellings": True},
            "1D CNN, 60 epochs": {"architecture": "cnn", "max_epochs": 60, "repair_misspellings": False},
            "1D CNN, 60 epochs + misspelling repair": {"architecture": "cnn", "max_epochs": 60,
                                                       "repair_misspellings": True},
        },
    },
}
# Every setting an experiment compares is stated explicitly ("base" + variant),
# so changing the defaults in config.py can't silently change an experiment.


def run_experiment(name: str) -> dict:
    experiment = EXPERIMENTS[name]
    train_df = load_processed_split(config.TRAIN_FILE)
    val_df = load_processed_split(config.VAL_FILE)
    train_texts, train_labels = train_df["clean_message"].tolist(), train_df["label_num"].tolist()
    val_texts, val_labels = val_df["clean_message"].tolist(), val_df["label_num"].tolist()
    val_spam = val_df.loc[val_df["label_num"] == 1, "clean_message"]

    rows = []
    for variant, variant_settings in experiment["variants"].items():
        settings = {**experiment["base"], **variant_settings}
        model, vectorizer, model_config, _ = train_model(
            train_texts, train_labels, val_texts, val_labels, settings, verbose=False)
        threshold = model_config["threshold"]
        m = compute_metrics(val_labels, predict_probabilities(model, vectorizer, val_texts), threshold=threshold)
        robust = disguised_spam_recall(model, vectorizer, val_spam, threshold)
        rows.append({
            "variant": variant,
            "settings": settings,
            "epochs_run": model_config["training"]["epochs_run"],
            "spam_weight": round(model_config["training"]["spam_weight"], 2),
            "threshold": threshold,
            "val_precision": m["precision"],
            "val_recall": m["recall"],
            "val_f1": m["f1"],
            "val_spam_caught": m["confusion_matrix"]["true_positive"],
            "val_false_alarms": m["confusion_matrix"]["false_positive"],
            "val_disguised_spam_recall": robust["recall_disguised"],
        })

    best = max(rows, key=lambda r: r["val_f1"])
    print(f"Validation set: {len(val_df)} messages ({sum(val_labels)} spam)\n")
    print(f"{'Variant':<50} {'Epochs':>6} {'Thresh':>6} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Disguised':>9}")
    for r in rows:
        marker = "  <- best" if r is best else ""
        print(f"{r['variant']:<50} {r['epochs_run']:>6} {r['threshold']:>6.2f} {r['val_precision']:>7.2%} "
              f"{r['val_recall']:>7.2%} {r['val_f1']:>7.4f} {r['val_disguised_spam_recall']:>9.2%}{marker}")

    result = {"selection_rule": "highest validation F1", "selected": best["variant"],
              "selected_settings": best["settings"], "variants": rows}
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    experiment["file"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSelected (highest validation F1): {best['variant']}")
    print(f"Saved comparison to {experiment['file']}")
    return result


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    name = argv[0] if argv else "architecture"
    if name not in EXPERIMENTS:
        sys.exit(f"Unknown experiment '{name}'. Choose from: {', '.join(EXPERIMENTS)}")
    run_experiment(name)


if __name__ == "__main__":
    main()
