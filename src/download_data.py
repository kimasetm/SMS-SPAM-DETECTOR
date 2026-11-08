"""Download the datasets into data/raw/.

Run from the project root:
    python -m src.download_data

1. UCI SMS Spam Collection (Almeida et al., 2011): the main dataset.
2. SMS Phishing Dataset (Mishra & Soni, 2022, CC BY 4.0): newer messages,
   including smishing (SMS phishing).
"""

import io
import urllib.request
import zipfile

from src.config import (
    DATASET_URL,
    MODERN_DATASET_URL,
    MODERN_RAW_FILE,
    RAW_DATA_DIR,
    RAW_DATA_FILE,
)


def _download_zip(url: str) -> zipfile.ZipFile:
    # Some servers (e.g. Mendeley) reject Python's default User-Agent with 403 Forbidden
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (SmartSpam dataset download)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return zipfile.ZipFile(io.BytesIO(response.read()))


def download_dataset(force: bool = False) -> None:
    """Download the UCI zip and extract the SMSSpamCollection file."""
    if RAW_DATA_FILE.exists() and not force:
        print(f"Dataset already exists at {RAW_DATA_FILE}")
        return

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATASET_URL} ...")
    # The zip contains "SMSSpamCollection" (the data) and a "readme" file.
    # We only extract the data file.
    with _download_zip(DATASET_URL) as archive:
        RAW_DATA_FILE.write_bytes(archive.read("SMSSpamCollection"))
    print(f"Saved dataset to {RAW_DATA_FILE}")


def download_modern_dataset(force: bool = False) -> None:
    """Download the Mishra & Soni smishing dataset and extract its CSV file."""
    if MODERN_RAW_FILE.exists() and not force:
        print(f"Dataset already exists at {MODERN_RAW_FILE}")
        return

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading SMS Phishing Dataset (Mishra & Soni, 2022) ...")
    with _download_zip(MODERN_DATASET_URL) as archive:
        # Take only the data file (by name), never paths chosen by the archive
        member = next(name for name in archive.namelist() if name.endswith(MODERN_RAW_FILE.name))
        MODERN_RAW_FILE.write_bytes(archive.read(member))
    print(f"Saved dataset to {MODERN_RAW_FILE}")


if __name__ == "__main__":
    download_dataset()
    download_modern_dataset()
