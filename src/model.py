"""The SmartSpam neural network, plus helpers to save and load it.

Architecture (PyTorch equivalents of the Keras layers in the project brief):

    word IDs  (batch, 50)
      -> Embedding             (batch, 50, 32)   Keras: Embedding
      -> masked average pool   (batch, 32)       Keras: GlobalAveragePooling1D
      -> Dropout
      -> Linear + ReLU         (batch, 16)       Keras: Dense(16, activation="relu")
      -> Dropout                                 Keras: Dropout
      -> Linear                (batch, 1)        Keras: Dense(1)
      -> sigmoid (at prediction time) = probability the message is spam
"""

import json

import torch
from torch import nn

from src.config import MODEL_CONFIG_FILE, MODEL_WEIGHTS_FILE, VECTORIZER_FILE
from src.text_vectorizer import PAD_ID, TextVectorizer


class SpamClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 32,
                 hidden_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        # Embedding: a lookup table with one learned vector per word ID.
        # During training, words used in similar ways (e.g. "free", "prize",
        # "win") end up with similar vectors.
        # padding_idx keeps the <pad> vector fixed at zeros.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)

        # Dropout randomly zeroes a fraction of values during training only.
        # This stops the network from relying too much on any single word or
        # neuron, which reduces overfitting on a small dataset.
        self.dropout = nn.Dropout(dropout)

        # Dense (fully connected) layers learn combinations of the pooled features
        self.hidden = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return one raw score (logit) per message. Positive = leaning spam."""
        embedded = self.embedding(token_ids)                    # (batch, seq_len, emb_dim)

        # Global average pooling over the words in each message. The mask
        # makes sure padding positions are ignored, so a short message isn't
        # "diluted" by its padding zeros.
        mask = (token_ids != PAD_ID).unsqueeze(-1).float()      # (batch, seq_len, 1)
        summed = (embedded * mask).sum(dim=1)                   # (batch, emb_dim)
        word_counts = mask.sum(dim=1).clamp(min=1.0)            # avoid dividing by 0
        pooled = summed / word_counts                           # (batch, emb_dim)

        x = self.dropout(pooled)
        x = torch.relu(self.hidden(x))
        x = self.dropout(x)
        return self.output(x).squeeze(-1)                       # (batch,)


def build_model(model_config: dict) -> SpamClassifier:
    return SpamClassifier(
        vocab_size=model_config["vocab_size"],
        embedding_dim=model_config["embedding_dim"],
        hidden_dim=model_config["hidden_dim"],
        dropout=model_config["dropout"],
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
