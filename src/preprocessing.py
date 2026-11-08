"""Load, clean and split the SMS Spam Collection dataset.

Run from the project root (after downloading the data):
    python -m src.preprocessing

Steps:
  1. Load the raw tab-separated file (label <TAB> message).
  2. Clean each message with clean_text().
  3. Convert labels to numbers (ham -> 0, spam -> 1).
  4. Remove duplicate messages so the same text cannot be in both the
     training set and the test set (a form of data leakage).
  5. Split into train / validation / test sets, keeping the spam ratio
     the same in each (stratified split).
  6. Save the three sets as CSV files in data/processed/.
"""

import csv
import html
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    LABEL_MAP,
    MODERN_RAW_FILE,
    MODERN_TEST_FILE,
    MODERN_TEST_SIZE,
    MODERN_VAL_SIZE,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_FILE,
    TEST_FILE,
    TEST_SIZE,
    TRAIN_FILE,
    VAL_FILE,
    VAL_SIZE,
)

# Regular expressions used by clean_text()
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_PATTERN = re.compile(r"\S+@\S+\.\S+")
PHONE_PATTERN = re.compile(r"\d{5,}")         # 5+ digits in a row: phone numbers, short codes
# Numbers that stand alone or start a word ("150p", "2nite"). Digits inside a
# word that starts with a letter ("fr33", "pr1ze", "b4") are kept as part of the
# word, so the vectorizer can recognize SMS slang and repair disguised words.
NUMBER_PATTERN = re.compile(r"(?<![a-z0-9])\d+")
CURRENCY_PATTERN = re.compile(r"[£$€]")
NON_WORD_PATTERN = re.compile(r"[^a-z0-9\s]")  # anything that is not a letter, digit or space
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize one SMS message.

    This exact function is reused at prediction time, so the model always
    sees text processed the same way it was during training.

    Instead of deleting URLs, phone numbers and currency symbols, we replace
    them with placeholder words. The exact number is not useful to the model,
    but the fact that a message *contains* a phone number or a price is a
    strong spam signal.
    """
    if not isinstance(text, str):
        return ""

    # The dataset contains HTML entities such as "&lt;#&gt;"; decode them to
    # real characters so they don't become fake words like "lt" and "gt".
    text = html.unescape(text)
    text = text.lower()
    text = URL_PATTERN.sub(" urltoken ", text)
    text = EMAIL_PATTERN.sub(" emailtoken ", text)
    text = PHONE_PATTERN.sub(" phonetoken ", text)
    text = CURRENCY_PATTERN.sub(" currencytoken ", text)
    text = NUMBER_PATTERN.sub(" numtoken ", text)
    text = NON_WORD_PATTERN.sub(" ", text)          # drop punctuation
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def duplicate_key(text: str) -> str:
    """A fixed version of the cleaned text, used only to find duplicates and split the data.

    It treats every digit as a number (like the original v1.0 cleaning), so
    'fr33' and 'fr44' count as the same message. Keeping this key fixed means
    improving clean_text() never changes which messages land in the test set,
    so results stay comparable across versions.
    """
    return WHITESPACE_PATTERN.sub(" ", re.sub(r"\d+", " numtoken ", clean_text(text))).strip()


def load_raw_data(path=RAW_DATA_FILE) -> pd.DataFrame:
    """Read the raw dataset into a DataFrame with columns 'label' and 'message'."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python -m src.download_data"
        )
    # QUOTE_NONE: messages contain quote characters that must be kept as text,
    # not interpreted as CSV quoting.
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean messages, encode labels and remove duplicates and empty rows."""
    df = df.copy()
    df["clean_message"] = df["message"].apply(clean_text)
    df["duplicate_key"] = df["message"].apply(duplicate_key)

    # Convert text labels to numbers the neural network can learn from
    df["label_num"] = df["label"].map(LABEL_MAP)
    if df["label_num"].isna().any():
        unknown = df.loc[df["label_num"].isna(), "label"].unique()
        raise ValueError(f"Unknown labels in dataset: {unknown}")
    df["label_num"] = df["label_num"].astype(int)

    # Drop messages that became empty after cleaning (e.g. only punctuation)
    df = df[df["duplicate_key"] != ""]

    # Drop duplicates. Many spam messages were sent repeatedly; if one copy
    # lands in training and another in testing, the test score would be
    # inflated because the model has already "seen" that exact message.
    df = df.drop_duplicates(subset="duplicate_key", keep="first")

    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, val_size: float = VAL_SIZE, test_size: float = TEST_SIZE):
    """Split into train / validation / test sets (70 / 15 / 15 by default).

    stratify=... keeps the same ham/spam ratio in every split. This matters
    because only about 1 in 8 messages is spam.
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=val_size + test_size,
        stratify=df["label_num"],
        random_state=RANDOM_SEED,
    )
    # Split the remainder into validation and test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        stratify=temp_df["label_num"],
        random_state=RANDOM_SEED,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_modern_data(path=MODERN_RAW_FILE) -> pd.DataFrame:
    """Read the Mishra & Soni dataset as columns 'label' (ham/spam), 'category' and 'message'.

    Its labels are ham, spam and smishing (SMS phishing). Smishing is a kind of
    spam, so both become "spam" for the model; 'category' keeps the original
    label so results can be reported for smishing separately.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run: python -m src.download_data")
    raw = pd.read_csv(path, encoding="utf-8", dtype=str, keep_default_na=False)
    category = raw["LABEL"].str.strip().str.lower()
    unknown = set(category) - {"ham", "spam", "smishing"}
    if unknown:
        raise ValueError(f"Unknown labels in modern dataset: {unknown}")
    return pd.DataFrame({
        "label": category.replace({"smishing": "spam"}),
        "category": category,
        "message": raw["TEXT"],
    })


def prepare_modern_dataframe(modern_df: pd.DataFrame, existing_keys) -> pd.DataFrame:
    """Clean the newer messages and keep only those NOT already in the UCI data.

    Many messages in this dataset were copied from the UCI collection. Keeping
    them would duplicate data and could put the same message in both training
    and test sets, so any message whose cleaned text appears anywhere in the
    UCI data is removed.
    """
    df = prepare_dataframe(modern_df)
    return df[~df["duplicate_key"].isin(set(existing_keys))].reset_index(drop=True)


def load_processed_split(path) -> pd.DataFrame:
    """Load a train/val/test CSV created by main()."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run: python -m src.preprocessing"
        )
    # keep_default_na=False: read every message as text. Otherwise pandas
    # would turn messages such as "na" or "null" into missing values.
    return pd.read_csv(path, keep_default_na=False)


def describe_split(name: str, df: pd.DataFrame) -> None:
    spam = int(df["label_num"].sum())
    print(f"  {name:<14} {len(df):>5} messages | spam: {spam:>4} ({spam / len(df):.1%})")


def main() -> None:
    # --- 1. UCI SMS Spam Collection (split exactly as in v1.0/v2) ---
    raw_df = load_raw_data()
    print(f"UCI: loaded {len(raw_df)} raw messages")
    uci = prepare_dataframe(raw_df).assign(source="uci")
    uci["category"] = uci["label"]
    print(f"UCI: {len(uci)} after removing duplicates/empty rows ({len(raw_df) - len(uci)} removed)")
    uci_train, uci_val, uci_test = split_data(uci)

    # --- 2. Newer messages (Mishra & Soni, 2022), if downloaded ---
    if MODERN_RAW_FILE.exists():
        modern_raw = load_modern_data()
        modern = prepare_modern_dataframe(modern_raw, uci["duplicate_key"]).assign(source="modern")
        print(f"Modern: loaded {len(modern_raw)} messages; {len(modern)} are new "
              f"(not in UCI, no duplicates) — {(modern['category'] == 'smishing').sum()} smishing")
        mod_train, mod_val, mod_test = split_data(modern, MODERN_VAL_SIZE, MODERN_TEST_SIZE)
    else:
        print(f"Modern dataset not found at {MODERN_RAW_FILE}; using UCI data only. "
              "Run: python -m src.download_data")
        mod_train = mod_val = mod_test = uci.iloc[0:0]

    # Training and validation use both sources. Test sets are kept separate so
    # UCI results stay comparable with earlier versions.
    train_df = pd.concat([uci_train, mod_train], ignore_index=True)
    val_df = pd.concat([uci_val, mod_val], ignore_index=True)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["clean_message", "label_num", "source", "category"]
    train_df[columns].to_csv(TRAIN_FILE, index=False)
    val_df[columns].to_csv(VAL_FILE, index=False)
    uci_test[columns].to_csv(TEST_FILE, index=False)
    if len(mod_test):
        mod_test[columns].to_csv(MODERN_TEST_FILE, index=False)

    print("Saved splits:")
    describe_split("train", train_df)
    describe_split("validation", val_df)
    describe_split("test (UCI)", uci_test)
    if len(mod_test):
        describe_split("test (modern)", mod_test)


if __name__ == "__main__":
    main()
