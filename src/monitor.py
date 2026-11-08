"""Monitoring: warn when new messages look different from the data the model was checked on.

Spam changes over time ("data drift"): new scams, new slang, new languages.
A model trained on old messages slowly gets worse, often without anyone
noticing, because new messages don't come with correct answers.

This module compares a batch of new messages with a baseline profile of the
validation set (saved when the model is trained) and raises warnings when
signals that don't need labels change a lot:

  - unknown_word_rate:  share of words not in the training vocabulary
  - predicted_spam_rate: share of messages the model flags as spam
  - uncertain_rate:     share of predictions close to the decision threshold
  - avg_words, url_rate, phone_rate: what the messages look like

Usage:
    python -m src.monitor new_messages.csv
"""

import json
import sys

import numpy as np

from src import config
from src.model import predict_probabilities
from src.preprocessing import clean_text
from src.text_vectorizer import OOV_ID, PAD_ID

UNCERTAIN_BAND = 0.15   # predictions within this distance of the threshold count as uncertain


def compute_profile(cleaned_texts, model, vectorizer, threshold: float) -> dict:
    """Summary statistics of a batch of cleaned messages (no labels needed)."""
    cleaned_texts = list(cleaned_texts)
    if not cleaned_texts:
        raise ValueError("No messages to profile.")
    ids = vectorizer.transform(cleaned_texts)
    words = ids[ids != PAD_ID]
    probs = predict_probabilities(model, vectorizer, cleaned_texts)
    return {
        "num_messages": len(cleaned_texts),
        "unknown_word_rate": float(np.mean(words == OOV_ID)) if words.size else 0.0,
        "avg_words": float(np.mean([len(t.split()) for t in cleaned_texts])),
        "url_rate": float(np.mean(["urltoken" in t.split() for t in cleaned_texts])),
        "phone_rate": float(np.mean(["phonetoken" in t.split() for t in cleaned_texts])),
        "predicted_spam_rate": float(np.mean(probs >= threshold)),
        "uncertain_rate": float(np.mean(np.abs(probs - threshold) <= UNCERTAIN_BAND)),
    }


# (signal, how to describe it, rule that triggers a warning)
CHECKS = [
    ("unknown_word_rate", "Unknown words",
     lambda new, base: new > 2 * base and new - base > 0.05,
     "Many words the model has never seen: new slang, new scam types or another language."),
    ("predicted_spam_rate", "Predicted spam rate",
     lambda new, base: abs(new - base) > 0.15,
     "The share of messages flagged as spam changed a lot: different traffic, or the model is drifting."),
    ("uncertain_rate", "Uncertain predictions",
     lambda new, base: new > 2 * base and new - base > 0.05,
     "Many predictions are close to the threshold: the model is unsure about this kind of message."),
    ("avg_words", "Average words per message",
     lambda new, base: base > 0 and abs(new - base) / base > 0.5,
     "Messages are much longer or shorter than the data the model was checked on."),
]


def compare_to_baseline(profile: dict, baseline: dict) -> list:
    """Return a list of warnings ({signal, title, new, baseline, explanation}); empty means no drift found."""
    warnings = []
    for signal, title, triggered, explanation in CHECKS:
        if triggered(profile[signal], baseline[signal]):
            warnings.append({"signal": signal, "title": title, "new": profile[signal],
                             "baseline": baseline[signal], "explanation": explanation})
    return warnings


def save_baseline(profile: dict, path=None) -> None:
    path = path or config.MONITORING_BASELINE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def load_baseline(path=None):
    path = path or config.MONITORING_BASELINE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def check_messages(raw_messages, detector) -> dict:
    """Profile raw messages with a SpamDetector and compare them with the saved baseline."""
    cleaned = [c for c in (clean_text(m) for m in raw_messages) if c]
    profile = compute_profile(cleaned, detector.model, detector.vectorizer, detector.threshold)
    baseline = load_baseline()
    return {"profile": profile, "baseline": baseline,
            "warnings": compare_to_baseline(profile, baseline) if baseline else None}


def format_report(result: dict) -> str:
    profile, baseline = result["profile"], result["baseline"]
    lines = [f"Messages checked: {profile['num_messages']}", ""]
    lines.append(f"{'Signal':<24} {'New batch':>10} {'Baseline':>10}")
    for key in ("unknown_word_rate", "predicted_spam_rate", "uncertain_rate", "url_rate", "phone_rate"):
        base = f"{baseline[key]:>10.1%}" if baseline else f"{'n/a':>10}"
        lines.append(f"{key:<24} {profile[key]:>10.1%} {base}")
    base = f"{baseline['avg_words']:>10.1f}" if baseline else f"{'n/a':>10}"
    lines.append(f"{'avg_words':<24} {profile['avg_words']:>10.1f} {base}")
    lines.append("")
    if result["warnings"] is None:
        lines.append("No baseline found. Train the model first: python -m src.train")
    elif result["warnings"]:
        lines.append(f"WARNING: {len(result['warnings'])} possible drift signal(s):")
        lines += [f"  - {w['title']}: {w['explanation']}" for w in result["warnings"]]
    else:
        lines.append("OK: no drift signals found. (This doesn't prove the model is still accurate.)")
    return "\n".join(lines)


def main(argv=None) -> int:
    from src.batch import BatchFileError, read_messages_csv
    from src.predict import get_detector

    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("Usage: python -m src.monitor messages.csv")
        return 1
    try:
        df, column = read_messages_csv(argv[0])
        result = check_messages(df[column].tolist(), get_detector())
    except (BatchFileError, FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        return 1
    print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
