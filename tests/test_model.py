"""Tests for the neural network architecture and saving/loading."""

import pytest
import torch

from src.model import SpamClassifier, load_artifacts, save_artifacts
from src.text_vectorizer import TextVectorizer

CONFIG = {"vocab_size": 20, "embedding_dim": 8, "hidden_dim": 4, "dropout": 0.3}


def make_model():
    torch.manual_seed(0)
    return SpamClassifier(CONFIG["vocab_size"], CONFIG["embedding_dim"],
                          CONFIG["hidden_dim"], CONFIG["dropout"])


def test_output_has_one_score_per_message():
    model = make_model()
    batch = torch.randint(0, CONFIG["vocab_size"], (5, 10))
    assert model(batch).shape == (5,)


def test_padding_does_not_change_prediction():
    """Masked pooling: extra padding zeros must not change the result."""
    model = make_model().eval()
    short = torch.tensor([[3, 7, 5, 0, 0]])
    long = torch.tensor([[3, 7, 5, 0, 0, 0, 0, 0, 0, 0]])
    assert torch.allclose(model(short), model(long))


def test_all_padding_message_does_not_crash():
    model = make_model().eval()
    output = model(torch.zeros((1, 6), dtype=torch.long))
    assert torch.isfinite(output).all()


def test_dropout_only_active_in_training_mode():
    model = make_model()
    batch = torch.randint(1, CONFIG["vocab_size"], (4, 10))

    model.eval()
    assert torch.equal(model(batch), model(batch))   # deterministic in eval mode

    model.train()
    torch.manual_seed(1)
    first = model(batch)
    torch.manual_seed(2)
    second = model(batch)
    assert not torch.equal(first, second)            # random dropout in train mode


def make_cnn():
    torch.manual_seed(0)
    return SpamClassifier(CONFIG["vocab_size"], CONFIG["embedding_dim"], CONFIG["hidden_dim"],
                          CONFIG["dropout"], architecture="cnn", conv_filters=6, kernel_size=3)


def test_cnn_output_shape_and_empty_message():
    model = make_cnn().eval()
    assert model(torch.randint(1, CONFIG["vocab_size"], (4, 10))).shape == (4,)
    assert torch.isfinite(model(torch.zeros((1, 6), dtype=torch.long))).all()


def test_cnn_ignores_padding():
    model = make_cnn().eval()
    assert torch.allclose(model(torch.tensor([[3, 7, 5, 0, 0]])), model(torch.tensor([[3, 7, 5, 0, 0, 0, 0, 0]])))


def test_word_order_matters_for_cnn_but_not_average_pooling():
    """The CNN reads neighbouring words together; average pooling treats a message as a bag of words."""
    forward, reordered = torch.tensor([[3, 7, 5, 9]]), torch.tensor([[9, 5, 7, 3]])
    avg, cnn = make_model().eval(), make_cnn().eval()
    assert torch.allclose(avg(forward), avg(reordered))
    assert not torch.allclose(cnn(forward), cnn(reordered))


def test_unknown_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unknown architecture"):
        SpamClassifier(10, architecture="transformer")


def test_cnn_save_and_load(tmp_path):
    config = {**CONFIG, "architecture": "cnn", "conv_filters": 6, "kernel_size": 3}
    model = make_cnn().eval()
    vectorizer = TextVectorizer(sequence_length=6).fit(["win free prize now", "see you"])
    paths = {"weights_path": tmp_path / "m.pt", "vectorizer_path": tmp_path / "v.json",
             "config_path": tmp_path / "c.json"}
    save_artifacts(model, vectorizer, config, **paths)
    loaded, _, _ = load_artifacts(**paths)
    ids = torch.tensor(vectorizer.transform(["win free prize"]))
    assert loaded.architecture == "cnn"
    assert torch.equal(loaded(ids), model(ids))


def test_save_and_load_gives_identical_predictions(tmp_path):
    model = make_model().eval()
    vectorizer = TextVectorizer(max_tokens=20, sequence_length=6).fit(["win free prize now", "see you"])
    paths = {
        "weights_path": tmp_path / "model.pt",
        "vectorizer_path": tmp_path / "vectorizer.json",
        "config_path": tmp_path / "config.json",
    }
    save_artifacts(model, vectorizer, CONFIG, **paths)
    loaded_model, loaded_vectorizer, loaded_config = load_artifacts(**paths)

    ids = torch.tensor(vectorizer.transform(["win free prize"]))
    assert loaded_config == CONFIG
    assert not loaded_model.training                 # loaded in eval mode
    assert torch.equal(loaded_model(ids), model(ids))
    assert loaded_vectorizer.word_to_id == vectorizer.word_to_id


def test_load_artifacts_reports_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.train"):
        load_artifacts(tmp_path / "a.pt", tmp_path / "b.json", tmp_path / "c.json")
