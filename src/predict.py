"""Prediction engine: classify new SMS messages with the trained model.

Usage from Python:
    from src.predict import predict_message
    result = predict_message("WINNER! Claim your free prize now")
    print(result.label, result.confidence)

Usage from the command line:
    python -m src.predict "WINNER! Claim your free prize now"

Every message goes through the SAME steps used during training:
    raw text -> clean_text() -> TextVectorizer (training vocabulary) -> model -> sigmoid
Using a different cleaning step here would give the model input it has
never seen, and its predictions would be unreliable.
"""

import sys
from dataclasses import asdict, dataclass
from functools import lru_cache

from src import config
from src.model import load_artifacts, predict_probabilities
from src.preprocessing import clean_text

# Longest message accepted. Real SMS messages are at most a few hundred
# characters (long texts are split into 153-character parts); the limit
# protects the app from huge inputs.
MAX_MESSAGE_LENGTH = 5000

SPAM_LABEL = "SPAM"
HAM_LABEL = "HAM"


class InvalidMessageError(ValueError):
    """Raised when a message can't be classified (empty, too long, wrong type...)."""


@dataclass(frozen=True)
class PredictionResult:
    label: str               # "SPAM" or "HAM"
    spam_probability: float  # model output: 0.0 = surely ham, 1.0 = surely spam
    confidence: float        # probability of the predicted label (0.5 - 1.0)
    cleaned_text: str        # what the model actually saw

    @property
    def is_spam(self) -> bool:
        return self.label == SPAM_LABEL

    def to_dict(self) -> dict:
        return asdict(self)


def format_confidence(confidence: float) -> str:
    """Format confidence as a percentage without ever claiming 100% certainty."""
    if confidence >= 0.999:
        return ">99.9%"
    return f"{confidence:.1%}"


def validate_message(message) -> str:
    """Check user input and return its cleaned version, or raise InvalidMessageError."""
    if message is None:
        raise InvalidMessageError("Please enter a message.")
    if not isinstance(message, str):
        raise InvalidMessageError(f"Message must be text, not {type(message).__name__}.")
    if not message.strip():
        raise InvalidMessageError("Please enter a message.")
    if len(message) > MAX_MESSAGE_LENGTH:
        raise InvalidMessageError(
            f"Message is too long ({len(message):,} characters). "
            f"The maximum is {MAX_MESSAGE_LENGTH:,}.")

    cleaned = clean_text(message)
    if not cleaned:
        # e.g. only punctuation or emoji: nothing left for the model to read
        raise InvalidMessageError("Message contains no words the model can read.")
    return cleaned


def make_result(spam_probability: float, cleaned_text: str,
                threshold: float = config.SPAM_THRESHOLD) -> PredictionResult:
    is_spam = spam_probability >= threshold
    # Confidence = the probability of whichever label was chosen.
    # e.g. spam probability 0.10 -> HAM with 90% confidence.
    confidence = spam_probability if is_spam else 1.0 - spam_probability
    return PredictionResult(
        label=SPAM_LABEL if is_spam else HAM_LABEL,
        spam_probability=round(float(spam_probability), 4),
        confidence=round(float(confidence), 4),
        cleaned_text=cleaned_text,
    )


class SpamDetector:
    """Loads the trained model once and classifies messages with it."""

    def __init__(self, threshold: float = config.SPAM_THRESHOLD, **artifact_paths):
        # artifact_paths lets tests point to a different model; normally the
        # files in models/ are used.
        self.model, self.vectorizer, self.model_config = load_artifacts(**artifact_paths)
        self.threshold = threshold

    def predict(self, message: str) -> PredictionResult:
        """Classify one message. Raises InvalidMessageError for bad input."""
        cleaned = validate_message(message)
        probability = predict_probabilities(self.model, self.vectorizer, [cleaned])[0]
        return make_result(probability, cleaned, self.threshold)

    def predict_many(self, messages) -> list:
        """Classify many messages at once (faster than one at a time).

        Returns one item per message: a PredictionResult, or an
        InvalidMessageError for messages that couldn't be classified. One bad
        message doesn't stop the rest from being classified.
        """
        results = [None] * len(messages)
        valid_positions, valid_cleaned = [], []
        for i, message in enumerate(messages):
            try:
                valid_cleaned.append(validate_message(message))
                valid_positions.append(i)
            except InvalidMessageError as error:
                results[i] = error

        probabilities = predict_probabilities(self.model, self.vectorizer, valid_cleaned)
        for i, cleaned, probability in zip(valid_positions, valid_cleaned, probabilities):
            results[i] = make_result(probability, cleaned, self.threshold)
        return results


@lru_cache(maxsize=1)
def get_detector() -> SpamDetector:
    """Load the model the first time it's needed, then reuse the same instance."""
    return SpamDetector()


def predict_message(message: str) -> PredictionResult:
    """Classify one SMS message with the trained model."""
    return get_detector().predict(message)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print('Usage: python -m src.predict "your message here"')
        return 1
    try:
        result = predict_message(" ".join(argv))
    except InvalidMessageError as error:
        print(f"Error: {error}")
        return 1
    print(f"{result.label}  (confidence {format_confidence(result.confidence)}, "
          f"spam probability {result.spam_probability:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
