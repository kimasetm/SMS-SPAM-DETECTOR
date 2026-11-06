"""Tests for data cleaning, splitting and text vectorization.

Run from the project root:
    python -m pytest
"""

import pandas as pd
import pytest

from src.preprocessing import clean_text, prepare_dataframe, split_data
from src.text_vectorizer import OOV_ID, PAD_ID, TextVectorizer


# ---------- clean_text ----------

def test_clean_text_lowercases_and_removes_punctuation():
    assert clean_text("Hello, WORLD!!!") == "hello world"


def test_clean_text_replaces_urls_phones_and_money():
    text = "WIN £1000 now! Call 08452810075 or visit www.prize.com"
    assert clean_text(text) == (
        "win currencytoken numtoken now call phonetoken or visit urltoken"
    )


def test_clean_text_decodes_html_entities():
    assert clean_text("I'll be there in &lt;#&gt; mins") == "i ll be there in mins"


def test_clean_text_handles_empty_and_non_string():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text(None) == ""


# ---------- prepare_dataframe / split_data ----------

def unique_word(i):
    """Turn a number into a unique letters-only word (0 -> 'a', 26 -> 'ba').

    Digits can't be used to make test messages unique, because clean_text()
    replaces every number with 'numtoken', and the duplicates would be removed.
    """
    word = ""
    while True:
        word = chr(ord("a") + i % 26) + word
        i //= 26
        if i == 0:
            return word


def make_df(n_ham=80, n_spam=20):
    messages = [f"ham message {unique_word(i)} hello" for i in range(n_ham)]
    messages += [f"spam offer {unique_word(i)} call now" for i in range(n_spam)]
    labels = ["ham"] * n_ham + ["spam"] * n_spam
    return pd.DataFrame({"label": labels, "message": messages})


def test_prepare_dataframe_encodes_labels():
    df = prepare_dataframe(pd.DataFrame({"label": ["ham", "spam"], "message": ["hi there", "win cash"]}))
    assert df["label_num"].tolist() == [0, 1]


def test_prepare_dataframe_removes_duplicates_and_empty_messages():
    df = pd.DataFrame({
        "label": ["spam", "spam", "ham", "ham"],
        "message": ["WIN NOW!", "win now", "!!!", "see you soon"],
    })
    result = prepare_dataframe(df)
    # "WIN NOW!" and "win now" clean to the same text; "!!!" cleans to ""
    assert result["clean_message"].tolist() == ["win now", "see you soon"]


def test_prepare_dataframe_rejects_unknown_labels():
    with pytest.raises(ValueError):
        prepare_dataframe(pd.DataFrame({"label": ["maybe"], "message": ["hello"]}))


def test_split_sizes_are_70_15_15_and_stratified():
    df = prepare_dataframe(make_df(n_ham=800, n_spam=200))
    train, val, test = split_data(df)

    assert len(train) + len(val) + len(test) == len(df)
    assert len(train) == 700
    assert len(val) == 150
    assert len(test) == 150
    # Stratification keeps the 20% spam ratio in every split
    for split in (train, val, test):
        assert split["label_num"].mean() == pytest.approx(0.2, abs=0.01)


def test_splits_do_not_overlap():
    df = prepare_dataframe(make_df())
    train, val, test = split_data(df)
    train_msgs, val_msgs, test_msgs = (set(s["clean_message"]) for s in (train, val, test))
    assert not train_msgs & val_msgs
    assert not train_msgs & test_msgs
    assert not val_msgs & test_msgs


# ---------- TextVectorizer ----------

def test_vectorizer_pads_and_truncates_to_sequence_length():
    vec = TextVectorizer(sequence_length=5).fit(["a b c d e f g"])
    assert len(vec.encode("a b")) == 5
    assert vec.encode("a b")[2:] == [PAD_ID, PAD_ID, PAD_ID]
    assert len(vec.encode("a b c d e f g")) == 5


def test_vectorizer_maps_unseen_words_to_oov():
    vec = TextVectorizer(sequence_length=3).fit(["hello world"])
    ids = vec.encode("hello unknownword")
    assert ids[0] not in (PAD_ID, OOV_ID)
    assert ids[1] == OOV_ID


def test_vectorizer_vocabulary_comes_only_from_training_data():
    """Data-leakage check: words that appear only in test data must not be in the vocabulary."""
    train_texts = ["free prize call now", "see you at lunch"]
    test_texts = ["exclusive bonus voucher"]
    vec = TextVectorizer().fit(train_texts)

    for word in test_texts[0].split():
        assert word not in vec.word_to_id
    assert all(i == OOV_ID for i in vec.encode(test_texts[0]) if i != PAD_ID)


def test_vectorizer_respects_max_tokens():
    vec = TextVectorizer(max_tokens=4).fit(["a a a b b c d e"])
    # 2 special tokens + the 2 most frequent words
    assert vec.vocab_size == 4
    assert set(vec.word_to_id) == {"<pad>", "<oov>", "a", "b"}


def test_vectorizer_requires_fit():
    with pytest.raises(RuntimeError):
        TextVectorizer().encode("hello")


def test_vectorizer_save_and_load_roundtrip(tmp_path):
    vec = TextVectorizer(max_tokens=100, sequence_length=8).fit(["win a free prize", "call me later"])
    path = tmp_path / "vectorizer.json"
    vec.save(path)
    loaded = TextVectorizer.load(path)

    assert loaded.word_to_id == vec.word_to_id
    assert loaded.sequence_length == 8
    assert loaded.encode("win free call") == vec.encode("win free call")
