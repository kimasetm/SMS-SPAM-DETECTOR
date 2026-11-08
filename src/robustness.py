"""Robustness check: does the model still catch spam when words are disguised?

Spammers often swap letters for similar-looking digits to slip past filters
("free prize" -> "fr33 pr1z3"). This module creates disguised copies of real
spam messages and measures how many the model still catches.

The disguised messages are synthetic: they are only used to MEASURE
robustness, never for training or for choosing a model.
"""

import numpy as np

from src.model import predict_probabilities
from src.preprocessing import clean_text

# Letters replaced by look-alike digits
DISGUISE = str.maketrans({"e": "3", "i": "1", "o": "0", "a": "4", "s": "5"})


def disguise_text(cleaned: str) -> str:
    """Disguise every word of 4+ letters ('free' -> 'fr33', 'prize' -> 'pr1z3').

    The first letter is never changed. If it were ('entry' -> '3ntry'),
    cleaning would turn the leading digit into a "numtoken" placeholder,
    a word the model links to spam, which would make the model look MORE
    accurate on disguised text and inflate this test.
    """
    words = [w[0] + w[1:].translate(DISGUISE) if len(w) >= 4 and w.isalpha() and "token" not in w else w
             for w in cleaned.split()]
    return clean_text(" ".join(words))


def disguised_spam_recall(model, vectorizer, spam_texts, threshold: float) -> dict:
    """Share of spam messages still caught after disguising them."""
    spam_texts = list(spam_texts)
    if not spam_texts:
        return {"count": 0, "caught_original": 0, "caught_disguised": 0, "recall_disguised": None}
    original = predict_probabilities(model, vectorizer, spam_texts) >= threshold
    disguised = predict_probabilities(model, vectorizer, [disguise_text(t) for t in spam_texts]) >= threshold
    return {
        "count": len(spam_texts),
        "caught_original": int(np.sum(original)),
        "caught_disguised": int(np.sum(disguised)),
        "recall_disguised": float(np.mean(disguised)),
    }
