# SmartSpam — SMS Spam Detection System

A machine learning web application that classifies SMS messages as **SPAM** or **HAM** (legitimate).
The model is a PyTorch neural network trained on the
[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) dataset.

> Project under development. Completed so far: project setup and data preprocessing.

## Project structure

```
├── data/
│   ├── raw/                # SMSSpamCollection (downloaded, not stored in git)
│   └── processed/          # train.csv, val.csv, test.csv (generated)
├── src/
│   ├── config.py           # shared paths and settings
│   ├── download_data.py    # downloads the UCI dataset
│   ├── preprocessing.py    # cleaning, label encoding, train/val/test split
│   └── text_vectorizer.py  # converts text into integer sequences for the model
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

## Run tests

```bash
python -m pytest
```
