"""Turn cleaned text into fixed-length sequences of integers.

PyTorch has no built-in equivalent of Keras' TextVectorization layer, so
this class does the same job:

  "free entry to win"  ->  [12, 845, 4, 57, 0, 0, 0, ...]

Each word gets an integer ID. A neural network cannot read words, but an
Embedding layer (Phase 2) can look up a learned vector for each ID.

Two special IDs are reserved:
  0 = <pad>  fills up short messages so every sequence has the same length
  1 = <oov>  "out of vocabulary": any word not seen during fit()

PREVENTING DATA LEAKAGE: the vocabulary is built by fit(), which must only
ever be called on the TRAINING set. Words that appear only in validation or
test messages become <oov>, just as unseen words will when the app is used
on brand-new messages.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np

PAD_TOKEN = "<pad>"
OOV_TOKEN = "<oov>"
PAD_ID = 0
OOV_ID = 1


class TextVectorizer:
    def __init__(self, max_tokens: int = 10000, sequence_length: int = 50):
        """
        max_tokens: maximum vocabulary size (including <pad> and <oov>).
            Rare words beyond this limit are treated as <oov>.
        sequence_length: every message is padded or cut to this many tokens.
        """
        self.max_tokens = max_tokens
        self.sequence_length = sequence_length
        self.word_to_id: dict[str, int] = {}

    @property
    def is_fitted(self) -> bool:
        return bool(self.word_to_id)

    @property
    def vocab_size(self) -> int:
        return len(self.word_to_id)

    def fit(self, texts) -> "TextVectorizer":
        """Build the vocabulary from training texts only."""
        counts = Counter(word for text in texts for word in text.split())

        # Keep the most frequent words; reserve 2 slots for <pad> and <oov>
        most_common = counts.most_common(self.max_tokens - 2)
        self.word_to_id = {PAD_TOKEN: PAD_ID, OOV_TOKEN: OOV_ID}
        for word, _ in most_common:
            self.word_to_id[word] = len(self.word_to_id)
        return self

    def encode(self, text: str) -> list[int]:
        """Convert one cleaned message into a list of exactly sequence_length IDs."""
        if not self.is_fitted:
            raise RuntimeError("TextVectorizer must be fit() before encoding text.")

        ids = [self.word_to_id.get(word, OOV_ID) for word in text.split()]
        ids = ids[: self.sequence_length]                      # truncate long messages
        ids += [PAD_ID] * (self.sequence_length - len(ids))    # pad short messages
        return ids

    def transform(self, texts) -> np.ndarray:
        """Encode many messages into a 2-D array of shape (n_messages, sequence_length)."""
        return np.array([self.encode(text) for text in texts], dtype=np.int64)

    def save(self, path) -> None:
        """Save the vocabulary and settings so prediction uses the exact same mapping."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "max_tokens": self.max_tokens,
            "sequence_length": self.sequence_length,
            "word_to_id": self.word_to_id,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "TextVectorizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vectorizer = cls(data["max_tokens"], data["sequence_length"])
        vectorizer.word_to_id = data["word_to_id"]
        return vectorizer
