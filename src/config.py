"""Central configuration: file paths and settings shared by every script.

Keeping these in one place means training, evaluation and prediction
always read and write the same files.
"""

from pathlib import Path

# Project root = the folder that contains src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data locations
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "SMSSpamCollection"

TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_FILE = PROCESSED_DATA_DIR / "val.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

# Official UCI download link for the SMS Spam Collection dataset
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

# Labels: the model outputs a number, so each class gets an integer
LABEL_MAP = {"ham": 0, "spam": 1}

# Data split: 70% train, 15% validation, 15% test
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Fixed random seed so every run produces the same split (reproducibility)
RANDOM_SEED = 42
