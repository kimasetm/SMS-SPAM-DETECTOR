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
NUMBER_PATTERN = re.compile(r"\d+")
CURRENCY_PATTERN = re.compile(r"[£$€]")
NON_WORD_PATTERN = re.compile(r"[^a-z\s]")    # anything that is not a letter or space
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

    # Convert text labels to numbers the neural network can learn from
    df["label_num"] = df["label"].map(LABEL_MAP)
    if df["label_num"].isna().any():
        unknown = df.loc[df["label_num"].isna(), "label"].unique()
        raise ValueError(f"Unknown labels in dataset: {unknown}")
    df["label_num"] = df["label_num"].astype(int)

    # Drop messages that became empty after cleaning (e.g. only punctuation)
    df = df[df["clean_message"] != ""]

    # Drop duplicates. Many spam messages were sent repeatedly; if one copy
    # lands in training and another in testing, the test score would be
    # inflated because the model has already "seen" that exact message.
    df = df.drop_duplicates(subset="clean_message", keep="first")

    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """Split into train / validation / test sets (70 / 15 / 15).

    stratify=... keeps the same ham/spam ratio in every split. This matters
    because only about 1 in 8 messages is spam.
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=VAL_SIZE + TEST_SIZE,
        stratify=df["label_num"],
        random_state=RANDOM_SEED,
    )
    # Split the remaining 30% in half: 15% validation, 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        stratify=temp_df["label_num"],
        random_state=RANDOM_SEED,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


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
    print(f"  {name:<10} {len(df):>5} messages | spam: {spam:>4} ({spam / len(df):.1%})")


def main() -> None:
    raw_df = load_raw_data()
    print(f"Loaded {len(raw_df)} raw messages")

    df = prepare_dataframe(raw_df)
    print(f"After cleaning and removing duplicates/empty rows: {len(df)} messages "
          f"({len(raw_df) - len(df)} removed)")

    train_df, val_df, test_df = split_data(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["clean_message", "label_num"]
    train_df[columns].to_csv(TRAIN_FILE, index=False)
    val_df[columns].to_csv(VAL_FILE, index=False)
    test_df[columns].to_csv(TEST_FILE, index=False)

    print("Saved splits:")
    describe_split("train", train_df)
    describe_split("validation", val_df)
    describe_split("test", test_df)


if __name__ == "__main__":
    main()
