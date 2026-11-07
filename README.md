# SmartSpam — SMS Spam Detection System

[![Tests](https://github.com/kimasetm/SMS-SPAM-DETECTOR/actions/workflows/tests.yml/badge.svg)](https://github.com/kimasetm/SMS-SPAM-DETECTOR/actions/workflows/tests.yml)

A machine learning web application that classifies SMS messages as **SPAM** or **HAM** (legitimate).
The model is a PyTorch neural network trained on the
[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) dataset.

> Project under development. Completed so far: data preprocessing and the neural network.

## Project structure

```
├── .github/workflows/      # CI: runs tests on pull requests
├── data/
│   ├── raw/                # SMSSpamCollection (downloaded, not stored in git)
│   └── processed/          # train.csv, val.csv, test.csv (generated)
├── models/                 # trained weights, vocabulary, config, training history
├── src/
│   ├── config.py           # shared paths, hyperparameters and settings
│   ├── download_data.py    # downloads the UCI dataset
│   ├── preprocessing.py    # cleaning, label encoding, train/val/test split
│   ├── text_vectorizer.py  # converts text into integer sequences for the model
│   ├── model.py            # the neural network + save/load helpers
│   └── train.py            # training pipeline with early stopping
├── tests/                  # pytest unit tests
└── requirements.txt
```

## Setup

Requires Python 3.10 or newer.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## Get and prepare the data

```bash
# Download the dataset into data/raw/
python -m src.download_data

# Clean the messages and create the train / validation / test splits
python -m src.preprocessing
```

If the download fails, download the zip manually from the
[UCI page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
and place the `SMSSpamCollection` file from it in `data/raw/`.

### What preprocessing does

1. **Cleans** each message: lowercase, decode HTML entities, replace URLs, emails, phone numbers,
   currency symbols and numbers with placeholder words (`urltoken`, `phonetoken`, …), and remove punctuation.
2. **Encodes labels**: `ham → 0`, `spam → 1`.
3. **Removes duplicates** (478 duplicate or empty messages) so the same message can't appear in both training and test data.
4. **Splits** the data 70% / 15% / 15% into training, validation and test sets, keeping the same spam ratio (~12%) in each.

The vocabulary used by the model is built **only from the training set**, to prevent data leakage.

## Machine learning architecture

```
SMS text
  → clean_text()                 lowercase, placeholder tokens, remove punctuation
  → TextVectorizer               words → integer IDs, padded/truncated to 50
  → Embedding (6,314 × 32)       a learned 32-number vector for each word
  → Global average pooling       average of the word vectors (padding ignored) → 1 vector per message
  → Dropout (0.3)
  → Dense (16, ReLU)
  → Dropout (0.3)
  → Dense (1) → sigmoid          probability that the message is spam
```

Built with PyTorch (`src/model.py`); 202,593 trainable parameters.
The project brief allowed TensorFlow/Keras or PyTorch. PyTorch was chosen because TensorFlow did not
provide packages for Python 3.14, the development environment's Python version.
`TextVectorizer` and the masked average pooling are the PyTorch counterparts of Keras'
`TextVectorization` and `GlobalAveragePooling1D` layers.

## Train the model

```bash
python -m src.train
```

- Loss: binary cross-entropy · Optimizer: Adam (learning rate 0.001) · Batch size: 32
- **Early stopping:** training stops when validation loss hasn't improved for 3 epochs, and the best epoch's weights are kept.
- **Reproducible:** all random seeds are fixed, so retraining on the same machine produces identical weights.
- The **test set is not used during training**; it's reserved for final evaluation.

The trained model is saved in `models/` (`spam_classifier.pt`, `vectorizer.json`, `model_config.json`,
`training_history.json`). These files are committed, so the app can run without retraining.
In the current run, training stopped after epoch 18; the best epoch was 15, with a validation accuracy of 97.91%.
Test-set results will be reported after evaluation.

## Run tests

```bash
python -m pytest
```

Tests also run automatically with GitHub Actions (`.github/workflows/tests.yml`) on every pull request
into `dev` or `main`, on Python 3.12 and 3.14.
