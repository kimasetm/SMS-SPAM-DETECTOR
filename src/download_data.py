"""Download the UCI SMS Spam Collection dataset into data/raw/.

Run from the project root:
    python -m src.download_data
"""

import io
import urllib.request
import zipfile

from src.config import DATASET_URL, RAW_DATA_DIR, RAW_DATA_FILE


def download_dataset(force: bool = False) -> None:
    """Download the dataset zip and extract the SMSSpamCollection file."""
    if RAW_DATA_FILE.exists() and not force:
        print(f"Dataset already exists at {RAW_DATA_FILE}")
        return

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATASET_URL} ...")
    with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
        zip_bytes = response.read()

    # The zip contains "SMSSpamCollection" (the data) and a "readme" file.
    # We only extract the data file.
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        RAW_DATA_FILE.write_bytes(archive.read("SMSSpamCollection"))

    print(f"Saved dataset to {RAW_DATA_FILE}")


if __name__ == "__main__":
    download_dataset()
