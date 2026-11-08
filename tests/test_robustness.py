"""Tests for the disguised-spam robustness check."""

import torch

from src.model import SpamClassifier
from src.robustness import disguise_text, disguised_spam_recall
from src.text_vectorizer import TextVectorizer


def test_disguise_swaps_letters_but_keeps_the_first_letter():
    assert disguise_text("claim your free prize") == "cl41m y0ur fr33 pr1z3"
    assert disguise_text("entry") == "entry"          # first letter kept, so no "numtoken" is created


def test_disguise_leaves_short_words_and_placeholders_alone():
    assert disguise_text("win a phonetoken now") == "win a phonetoken now"


def test_disguised_spam_recall_counts():
    torch.manual_seed(0)
    vectorizer = TextVectorizer(sequence_length=5).fit(["claim free prize", "see you"])
    model = SpamClassifier(vectorizer.vocab_size, embedding_dim=4, hidden_dim=2)
    result = disguised_spam_recall(model, vectorizer, ["claim free prize", "free prize"], threshold=0.0)
    assert result == {"count": 2, "caught_original": 2, "caught_disguised": 2, "recall_disguised": 1.0}
    assert disguised_spam_recall(model, vectorizer, [], 0.5)["count"] == 0
