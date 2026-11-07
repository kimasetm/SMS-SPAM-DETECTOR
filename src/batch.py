"""Classify many SMS messages from a CSV file.

Used by the web app's upload feature, and from the command line:
    python -m src.batch samples/sample_messages.csv predictions.csv

The input CSV needs a column of messages, named "message", "text", "sms" or
"body" (any capitalization). A file with a single column is also accepted.
"""

import io
import sys
from pathlib import Path

import pandas as pd

from src.predict import InvalidMessageError, get_detector

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024   # 5 MB
MAX_ROWS = 5000
MESSAGE_COLUMN_NAMES = ("message", "text", "sms", "body")

# Columns added to the results
RESULT_COLUMNS = ["prediction", "confidence", "spam_probability", "error"]

# Spreadsheet programs treat cells starting with these characters as formulas.
# A malicious CSV could use this to run commands when the results are opened
# in Excel ("CSV injection"), so such cells are escaped on export.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


class BatchFileError(ValueError):
    """Raised when an uploaded CSV file can't be used."""


def read_messages_csv(source) -> tuple[pd.DataFrame, str]:
    """Read a CSV file (path or uploaded file object).

    Returns (dataframe, name of the message column). Raises BatchFileError
    with a user-friendly explanation if the file can't be used.
    """
    raw = Path(source).read_bytes() if isinstance(source, (str, Path)) else source.read()

    if len(raw) == 0:
        raise BatchFileError("The file is empty.")
    if len(raw) > MAX_FILE_SIZE_BYTES:
        raise BatchFileError(f"The file is too large. The maximum size is "
                             f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.")

    # Most files are UTF-8 ("utf-8-sig" also removes the marker Excel adds);
    # fall back to Latin-1, which can decode any byte sequence.
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    try:
        # dtype=str + keep_default_na=False: keep every cell as the exact text
        df = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise BatchFileError(f"Could not read the file as CSV: {error}") from error

    if df.empty:
        raise BatchFileError("The file has no message rows.")
    if len(df) > MAX_ROWS:
        raise BatchFileError(f"The file has {len(df):,} rows. The maximum is {MAX_ROWS:,}.")

    return df, find_message_column(df)


def find_message_column(df: pd.DataFrame) -> str:
    for column in df.columns:
        if str(column).strip().lower() in MESSAGE_COLUMN_NAMES:
            return column
    if len(df.columns) == 1:
        return df.columns[0]
    raise BatchFileError(
        f"Couldn't find the message column. Name it one of: {', '.join(MESSAGE_COLUMN_NAMES)}. "
        f"Columns found: {', '.join(map(str, df.columns))}.")


def classify_dataframe(df: pd.DataFrame, message_column: str, detector=None) -> pd.DataFrame:
    """Return a copy of df with prediction, confidence, spam_probability and error columns."""
    detector = detector or get_detector()
    results = detector.predict_many(df[message_column].tolist())

    output = df.copy()
    output["prediction"] = [r.label if not isinstance(r, InvalidMessageError) else "INVALID"
                            for r in results]
    output["confidence"] = [r.confidence if not isinstance(r, InvalidMessageError) else None
                            for r in results]
    output["spam_probability"] = [r.spam_probability if not isinstance(r, InvalidMessageError) else None
                                  for r in results]
    output["error"] = [str(r) if isinstance(r, InvalidMessageError) else "" for r in results]
    return output


def summarize(results: pd.DataFrame) -> dict:
    counts = results["prediction"].value_counts()
    return {
        "total": len(results),
        "spam": int(counts.get("SPAM", 0)),
        "ham": int(counts.get("HAM", 0)),
        "invalid": int(counts.get("INVALID", 0)),
    }


def escape_formula(value):
    """Prefix risky cells with an apostrophe so spreadsheets show them as plain text."""
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def to_csv_bytes(results: pd.DataFrame) -> bytes:
    """Convert results to CSV bytes that are safe to open in a spreadsheet."""
    # Check every cell (escape_formula leaves non-text values unchanged)
    safe = results.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(escape_formula)
    # utf-8-sig so Excel displays characters like £ correctly
    return safe.to_csv(index=False).encode("utf-8-sig")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or len(argv) > 2:
        print("Usage: python -m src.batch input.csv [output.csv]")
        return 1

    input_path = Path(argv[0])
    output_path = Path(argv[1]) if len(argv) == 2 else input_path.with_name(input_path.stem + "_predictions.csv")
    try:
        df, column = read_messages_csv(input_path)
    except (BatchFileError, FileNotFoundError) as error:
        print(f"Error: {error}")
        return 1

    results = classify_dataframe(df, column)
    output_path.write_bytes(to_csv_bytes(results))
    s = summarize(results)
    print(f"Classified {s['total']} messages: {s['spam']} spam, {s['ham']} ham, {s['invalid']} invalid")
    print(f"Saved predictions to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
