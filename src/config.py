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
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"                    # UCI test set (same as v1.0/v2)
MODERN_TEST_FILE = PROCESSED_DATA_DIR / "test_modern.csv"      # test set of newer messages

# Official UCI download link for the SMS Spam Collection dataset
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

# Second, newer dataset: "SMS Phishing Dataset for Machine Learning and Pattern
# Recognition" by S. Mishra and D. Soni (Mendeley Data, 2022, CC BY 4.0).
# 5,971 messages labelled ham / spam / smishing. Many overlap with the UCI
# collection; only messages NOT already in UCI are used.
MODERN_DATASET_URL = ("https://data.mendeley.com/public-files/datasets/f45bkkt8pr/files/"
                      "edb361de-918d-469f-9106-e84823830665/file_downloaded")
MODERN_RAW_FILE = RAW_DATA_DIR / "Dataset_5971.csv"
# The newer messages are split 60 / 15 / 25: a larger test share gives a more
# reliable estimate of performance on modern messages.
MODERN_VAL_SIZE = 0.15
MODERN_TEST_SIZE = 0.25

# Labels: the model outputs a number, so each class gets an integer
LABEL_MAP = {"ham": 0, "spam": 1}

# Data split: 70% train, 15% validation, 15% test
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Fixed random seed so every run produces the same split and the same
# trained model (reproducibility)
RANDOM_SEED = 42

# Saved model files (created by src/train.py, used by evaluation and the app)
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_WEIGHTS_FILE = MODEL_DIR / "spam_classifier.pt"    # learned weights
VECTORIZER_FILE = MODEL_DIR / "vectorizer.json"          # vocabulary from training data
MODEL_CONFIG_FILE = MODEL_DIR / "model_config.json"      # architecture + training settings
HISTORY_FILE = MODEL_DIR / "training_history.json"       # loss/accuracy per epoch

# Evaluation outputs (created by src/evaluate.py)
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_FILE = REPORTS_DIR / "metrics.json"
BASELINE_METRICS_FILE = REPORTS_DIR / "metrics_v1.json"   # v1.0 results, for before/after comparison
MODEL_COMPARISON_FILE = REPORTS_DIR / "model_comparison.json"
MISCLASSIFIED_FILE = REPORTS_DIR / "misclassified_test_messages.csv"

# Default decision threshold: a message is spam when its probability is >= this.
# Training now chooses a better threshold on the validation set and saves it
# in model_config.json; this value is only used if a model has none saved.
SPAM_THRESHOLD = 0.5

# --- Text vectorization ---
MAX_TOKENS = 10000       # maximum vocabulary size
SEQUENCE_LENGTH = 50     # words per message (covers >95% of training messages)

# --- Neural network architecture ---
EMBEDDING_DIM = 32       # size of the learned vector for each word
HIDDEN_DIM = 16          # neurons in the hidden dense layer
DROPOUT = 0.3            # fraction of neurons randomly switched off during training

# --- Training ---
BATCH_SIZE = 32          # messages per weight update
LEARNING_RATE = 0.001    # step size for the Adam optimizer
MAX_EPOCHS = 30          # upper limit on passes over the training data
EARLY_STOPPING_PATIENCE = 3   # stop after this many epochs without validation improvement

# --- Handling class imbalance (only ~12% of messages are spam) ---
# How much more a mistake on a spam message counts during training:
#   None      = all messages count equally (v1.0 behavior)
#   "sqrt"    = square root of the ham/spam ratio (moderate boost)
#   "balanced"= the full ham/spam ratio (spam and ham count equally in total)
SPAM_WEIGHT = None
# Choose the decision threshold that maximizes F1 on the validation set
TUNE_THRESHOLD = True
