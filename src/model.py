"""The SmartSpam neural network, plus helpers to save and load it.

Two architectures (PyTorch equivalents of Keras layers in brackets):

"avg_pool" (v1.0 / v2): treats a message as a bag of words
    word IDs  (batch, 50)
      -> Embedding             (batch, 50, 32)   [Embedding]
      -> masked average pool   (batch, 32)       [GlobalAveragePooling1D]
      -> Dropout -> Linear + ReLU (16) -> Dropout -> Linear (1)   [Dense, Dropout]

"cnn": also reads short phrases, so word order matters
    word IDs  (batch, 50)
      -> Embedding             (batch, 50, 32)   [Embedding]
      -> Conv1D, 3-word window (batch, 64, 50)   [Conv1D(64, 3, activation="relu")]
      -> masked max pool       (batch, 64)       [GlobalMaxPooling1D]
      -> Dropout -> Linear + ReLU (16) -> Dropout -> Linear (1)

Both end with a sigmoid at prediction time = probability the message is spam.
"""

import json

import numpy as np
import torch
from torch import nn

from src.config import MODEL_CONFIG_FILE, MODEL_WEIGHTS_FILE, VECTORIZER_FILE
from src.text_vectorizer import PAD_ID, TextVectorizer


ARCHITECTURES = ("avg_pool", "cnn")


class SpamClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 32,
                 hidden_dim: int = 16, dropout: float = 0.3,
                 architecture: str = "avg_pool", conv_filters: int = 64, kernel_size: int = 3):
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture '{architecture}'. Choose from {ARCHITECTURES}.")
        self.architecture = architecture

        # Embedding: a lookup table with one learned vector per word ID.
        # During training, words used in similar ways (e.g. "free", "prize",
        # "win") end up with similar vectors.
        # padding_idx keeps the <pad> vector fixed at zeros.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)

        if architecture == "cnn":
            # A 1D convolution slides a small window over `kernel_size`
            # neighbouring words and learns `conv_filters` phrase detectors
            # (e.g. one filter might respond to "claim your prize").
            # padding keeps one output per word position.
            self.conv = nn.Conv1d(embedding_dim, conv_filters, kernel_size, padding=kernel_size // 2)
            pooled_size = conv_filters
        else:
            pooled_size = embedding_dim

        # Dropout randomly zeroes a fraction of values during training only.
        # This stops the network from relying too much on any single word or
        # neuron, which reduces overfitting on a small dataset.
        self.dropout = nn.Dropout(dropout)

        # Dense (fully connected) layers learn combinations of the pooled features
        self.hidden = nn.Linear(pooled_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return one raw score (logit) per message. Positive = leaning spam."""
        embedded = self.embedding(token_ids)                    # (batch, seq_len, emb_dim)
        is_word = token_ids != PAD_ID                           # (batch, seq_len)

        if self.architecture == "cnn":
            # Conv1d expects (batch, channels, length)
            features = torch.relu(self.conv(embedded.transpose(1, 2)))   # (batch, filters, seq_len)
            # Global max pooling: keep each filter's strongest match anywhere
            # in the message. Padding positions are excluded with -inf.
            features = features.masked_fill(~is_word.unsqueeze(1), float("-inf"))
            pooled = features.max(dim=2).values                          # (batch, filters)
            pooled = torch.where(torch.isinf(pooled), torch.zeros_like(pooled), pooled)  # empty messages
        else:
            # Global average pooling over the words in each message. The mask
            # makes sure padding positions are ignored, so a short message isn't
            # "diluted" by its padding zeros.
            mask = is_word.unsqueeze(-1).float()                # (batch, seq_len, 1)
            summed = (embedded * mask).sum(dim=1)               # (batch, emb_dim)
            word_counts = mask.sum(dim=1).clamp(min=1.0)        # avoid dividing by 0
            pooled = summed / word_counts                       # (batch, emb_dim)

        x = self.dropout(pooled)
        x = torch.relu(self.hidden(x))
        x = self.dropout(x)
        return self.output(x).squeeze(-1)                       # (batch,)


def predict_probabilities(model, vectorizer, cleaned_texts, batch_size: int = 256) -> np.ndarray:
    """Return the predicted spam probability (0-1) for each *cleaned* message."""
    model.eval()
    probabilities = []
    with torch.no_grad():   # no gradients needed when only predicting
        for start in range(0, len(cleaned_texts), batch_size):
            ids = torch.tensor(vectorizer.transform(cleaned_texts[start:start + batch_size]))
            # sigmoid converts the raw score (logit) into a probability
            probabilities.append(torch.sigmoid(model(ids)).numpy())
    return np.concatenate(probabilities) if probabilities else np.array([])


def build_model(model_config: dict) -> SpamClassifier:
    return SpamClassifier(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        hidden_dim=model_config["hidden_dim"],
        dropout=model_config["dropout"],
        # Older saved models (v1.0/v2) have no architecture entry: they are avg_pool
        architecture=model_config.get("architecture", "avg_pool"),
        conv_filters=model_config.get("conv_filters", 64),
        kernel_size=model_config.get("kernel_size", 3),
    )


def save_artifacts(model: SpamClassifier, vectorizer: TextVectorizer, model_config: dict,
                   weights_path=MODEL_WEIGHTS_FILE, vectorizer_path=VECTORIZER_FILE,
                   config_path=MODEL_CONFIG_FILE) -> None:
    """Save everything needed to make predictions later.

    - weights: the numbers the network learned
    - vectorizer: the exact word -> ID vocabulary built from the training data
    - config: the layer sizes, so the same architecture can be rebuilt
    """
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    vectorizer.save(vectorizer_path)
    config_path.write_text(json.dumps(model_config, indent=2), encoding="utf-8")


def load_artifacts(weights_path=MODEL_WEIGHTS_FILE, vectorizer_path=VECTORIZER_FILE,
                   config_path=MODEL_CONFIG_FILE):
    """Load the trained model and its vectorizer. Returns (model, vectorizer, config)."""
    for path in (weights_path, vectorizer_path, config_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}. Train the model first: python -m src.train"
            )

    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    vectorizer = TextVectorizer.load(vectorizer_path)
    model = build_model(model_config)
    # weights_only=True: only load tensors, never arbitrary Python objects (safer)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()   # switch off dropout for predictions
    return model, vectorizer, model_config
