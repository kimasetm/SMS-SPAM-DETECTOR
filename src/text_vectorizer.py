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
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np

PAD_TOKEN = "<pad>"
OOV_TOKEN = "<oov>"
PAD_ID = 0
OOV_ID = 1

# Digits that spammers use in place of similar-looking letters ("fr33", "pr1ze").
# "1" can stand for "i" or "l", so both are tried.
LEET_MAPS = [str.maketrans("013457", "oieast"), str.maketrans("013457", "olease")]
DIGITS = re.compile(r"\d+")


class TextVectorizer:
    def __init__(self, max_tokens: int = 10000, sequence_length: int = 50,
                 repair_misspellings: bool = False):
        """
        max_tokens: maximum vocabulary size (including <pad> and <oov>).
            Rare words beyond this limit are treated as <oov>.
        sequence_length: every message is padded or cut to this many tokens.
        repair_misspellings: map unknown words to known ones when they look like
            disguised or misspelled versions (see repair_tokens()).
        """
        self.max_tokens = max_tokens
        self.sequence_length = sequence_length
        self.repair_misspellings = repair_misspellings
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

    # ----- character-level repair of unknown words -----

    def _known(self, word: str) -> bool:
        return word in self.word_to_id

    def _repair_digits(self, word: str):
        """'fr33' -> 'free', 'pr1ze' -> 'prize' (only if the result is a known word).

        Words with fewer than 2 letters ('b4', '2u') are too ambiguous and left alone.
        """
        if len(word) < 3 or sum(ch.isalpha() for ch in word) < 2:
            return None
        for table in LEET_MAPS:
            candidate = word.translate(table)
            if candidate.isalpha() and self._known(candidate):
                return candidate
        return None

    def _repair_typo(self, word: str):
        """Fix one typo in a word of 5+ letters: 'prlze' -> 'prize', 'winnner' -> 'winner'.

        Tries every word one edit away (a deleted, inserted, replaced or swapped
        letter) and picks the most frequent known word, if any.
        """
        if len(word) < 5 or not word.isalpha():
            return None
        letters = string.ascii_lowercase
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        candidates = (
            {a + b[1:] for a, b in splits if b}                                  # deletion
            | {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}         # swap
            | {a + c + b[1:] for a, b in splits if b for c in letters}           # replacement
            | {a + c + b for a, b in splits for c in letters}                    # insertion
        )
        known = [c for c in candidates if len(c) >= 4 and self._known(c)]
        # Lower ID = more frequent in the training data
        return min(known, key=self.word_to_id.get) if known else None

    def repair_tokens(self, words: list) -> list:
        """Replace unknown words that look like disguised known words.

        Only words NOT in the training vocabulary are changed, so normal text is
        never altered. Three repairs, in order:
          1. letters spaced out: 'c a s h' -> 'cash' (3+ single letters)
          2. digits used as letters: 'fr33' -> 'free'
          3. a single typo in a longer word: 'prlze' -> 'prize'
        """
        # 1. Join runs of 3+ single letters if they spell a known word
        joined, i = [], 0
        while i < len(words):
            j = i
            while j < len(words) and len(words[j]) == 1 and words[j].isalpha():
                j += 1
            if j - i >= 3 and self._known("".join(words[i:j])):
                joined.append("".join(words[i:j]))
                i = j
            else:
                joined.append(words[i])
                i += 1

        # 2 and 3. Repair individual unknown words
        repaired = []
        for word in joined:
            if not self._known(word):
                fixed = (self._repair_digits(word) if any(ch.isdigit() for ch in word)
                         else self._repair_typo(word))
                word = fixed or word
            repaired.append(word)
        return repaired

    def tokenize(self, text: str) -> list:
        """Split cleaned text into the words used for encoding."""
        words = text.split()
        if self.repair_misspellings:
            words = self.repair_tokens(words)
        # Unknown words mixing letters and digits ("abc123") are split into the
        # letters and a "numtoken" placeholder, like other numbers.
        tokens = []
        for word in words:
            if not self._known(word) and any(ch.isdigit() for ch in word):
                tokens.extend(DIGITS.sub(" numtoken ", word).split())
            else:
                tokens.append(word)
        return tokens

    def encode(self, text: str) -> list[int]:
        """Convert one cleaned message into a list of exactly sequence_length IDs."""
        if not self.is_fitted:
            raise RuntimeError("TextVectorizer must be fit() before encoding text.")

        ids = [self.word_to_id.get(word, OOV_ID) for word in self.tokenize(text)]
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
            "repair_misspellings": self.repair_misspellings,
            "word_to_id": self.word_to_id,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "TextVectorizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vectorizer = cls(data["max_tokens"], data["sequence_length"],
                         data.get("repair_misspellings", False))   # older files don't have it
        vectorizer.word_to_id = data["word_to_id"]
        return vectorizer
