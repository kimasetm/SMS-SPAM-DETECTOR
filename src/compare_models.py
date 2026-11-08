"""Compare ways of handling class imbalance, using the VALIDATION set only.

Run from the project root (after preprocessing):
    python -m src.compare_models

Only ~12% of messages are spam, so v1.0 tended to say "ham" when unsure and
missed about 1 in 6 spam messages. This script trains one model per variant:

  - v1.0 baseline:   no spam weight, threshold fixed at 0.5
  - tuned threshold: no spam weight, threshold chosen on validation
  - sqrt weight:     moderate spam weight, threshold chosen on validation
  - balanced weight: full ham/spam weight, threshold chosen on validation

Selection rule (decided before running): pick the variant with the highest
validation F1-score. The test set is NOT used here.
"""

import json

from src import config
from src.metrics import compute_metrics
from src.model import predict_probabilities
from src.preprocessing import load_processed_split
from src.train import train_model

VARIANTS = {
    "v1.0 baseline (no weight, threshold 0.5)": {"spam_weight": None, "tune_threshold": False},
    "No weight, tuned threshold": {"spam_weight": None, "tune_threshold": True},
    "Sqrt spam weight, tuned threshold": {"spam_weight": "sqrt", "tune_threshold": True},
    "Balanced spam weight, tuned threshold": {"spam_weight": "balanced", "tune_threshold": True},
}

RESULTS_FILE = config.MODEL_COMPARISON_FILE


def main() -> None:
    train_df = load_processed_split(config.TRAIN_FILE)
    val_df = load_processed_split(config.VAL_FILE)
    train_texts, train_labels = train_df["clean_message"].tolist(), train_df["label_num"].tolist()
    val_texts, val_labels = val_df["clean_message"].tolist(), val_df["label_num"].tolist()

    rows = []
    for name, settings in VARIANTS.items():
        model, vectorizer, model_config, _ = train_model(
            train_texts, train_labels, val_texts, val_labels, settings, verbose=False)
        probs = predict_probabilities(model, vectorizer, val_texts)
        m = compute_metrics(val_labels, probs, threshold=model_config["threshold"])
        rows.append({
            "variant": name,
            "spam_weight": round(model_config["training"]["spam_weight"], 2),
            "threshold": model_config["threshold"],
            "val_precision": m["precision"],
            "val_recall": m["recall"],
            "val_f1": m["f1"],
            "val_spam_caught": m["confusion_matrix"]["true_positive"],
            "val_false_alarms": m["confusion_matrix"]["false_positive"],
        })

    best = max(rows, key=lambda r: r["val_f1"])
    print(f"Validation set: {len(val_df)} messages ({sum(val_labels)} spam)\n")
    print(f"{'Variant':<42} {'Weight':>6} {'Thresh':>6} {'Prec':>7} {'Recall':>7} {'F1':>7}  Caught  False alarms")
    for r in rows:
        marker = "  <- best" if r is best else ""
        print(f"{r['variant']:<42} {r['spam_weight']:>6.2f} {r['threshold']:>6.2f} "
              f"{r['val_precision']:>7.2%} {r['val_recall']:>7.2%} {r['val_f1']:>7.4f}  "
              f"{r['val_spam_caught']:>6}  {r['val_false_alarms']:>12}{marker}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps({"selection_rule": "highest validation F1",
                                        "selected": best["variant"], "variants": rows}, indent=2),
                            encoding="utf-8")
    print(f"\nSelected (highest validation F1): {best['variant']}")
    print(f"Saved comparison to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
