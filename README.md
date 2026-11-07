# SmartSpam — SMS Spam Detection System

[![Tests](https://github.com/kimasetm/SMS-SPAM-DETECTOR/actions/workflows/tests.yml/badge.svg)](https://github.com/kimasetm/SMS-SPAM-DETECTOR/actions/workflows/tests.yml)

A machine learning web application that classifies SMS messages as **SPAM** or **HAM** (legitimate).
The model is a PyTorch neural network trained on the
[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) dataset.

> Project under development. Completed so far: data preprocessing, the neural network, model evaluation and the prediction engine.

## Project structure

```
├── .github/workflows/      # CI: runs tests on pull requests
├── data/
│   ├── raw/                # SMSSpamCollection (downloaded, not stored in git)
│   └── processed/          # train.csv, val.csv, test.csv (generated)
├── models/                 # trained weights, vocabulary, config, training history
├── reports/                # metrics.json and figures/ (evaluation charts)
├── samples/                # example CSV for batch classification
├── src/
│   ├── config.py           # shared paths, hyperparameters and settings
│   ├── download_data.py    # downloads the UCI dataset
│   ├── preprocessing.py    # cleaning, label encoding, train/val/test split
│   ├── text_vectorizer.py  # converts text into integer sequences for the model
│   ├── model.py            # the neural network + save/load helpers
│   ├── train.py            # training pipeline with early stopping
│   ├── evaluate.py         # test-set metrics, confusion matrix, charts
│   ├── predict.py          # prediction engine: message → SPAM/HAM + confidence
│   └── batch.py            # CSV batch classification and safe export
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

![Training history](reports/figures/training_history.png)

## Evaluation results

```bash
python -m src.evaluate
```

Measured on the **held-out test set** (765 messages: 675 ham, 90 spam), which was not used during
training or model selection. Spam is the positive class; threshold = 0.5.

| Metric | Value |
|---|---|
| Accuracy | **97.65%** |
| Precision (spam) | **95.00%**: of the messages flagged as spam, 95% were spam |
| Recall (spam) | **84.44%**: the model caught 76 of the 90 spam messages |
| F1-score (spam) | **0.8941** |
| ROC-AUC | **0.9863** |

<img src="reports/figures/confusion_matrix.png" alt="Confusion matrix" width="420">

**What the results mean:** the model is cautious. It wrongly flagged only 4 of 675 legitimate
messages (0.6%), but it missed 14 of 90 spam messages (15.6%). Most missed spam messages read like
ordinary conversation or ads without typical spam words (e.g. *"for sale arsenal dartboard good condition"*).
Because only ~12% of messages are spam, accuracy alone is misleading: a model that labels everything
"ham" would still score 88%. That's why precision, recall and F1 are reported.

Full results are saved in [`reports/metrics.json`](reports/metrics.json). Running the evaluation also writes
`reports/misclassified_test_messages.csv` locally for error analysis. It isn't committed, because it contains raw dataset text.

## Make predictions

**One message (command line):**
```bash
python -m src.predict "WINNER!! You have been selected to receive a £900 prize reward! Call 09061701461"
# SPAM  (confidence 97.7%, spam probability 0.9774)
```

**From Python:**
```python
from src.predict import predict_message

result = predict_message("Are we still meeting for lunch tomorrow?")
result.label             # "HAM"
result.confidence        # probability of the predicted label (0.5–1.0)
result.spam_probability  # model output: 0 = ham, 1 = spam
```

**Many messages from a CSV file:**
```bash
python -m src.batch samples/sample_messages.csv predictions.csv
# Classified 12 messages: 5 spam, 6 ham, 1 invalid
```
The CSV needs a column named `message`, `text`, `sms` or `body` (or just a single column). Limits: 5 MB and 5,000 rows.
The output adds `prediction`, `confidence`, `spam_probability` and `error` columns.
Rows that can't be classified, such as empty messages, are marked `INVALID` instead of stopping the whole file.

**How it works:** every message goes through the same `clean_text()` → `TextVectorizer` steps used in training,
so the model always gets input in the format it learned from. The model is loaded once and reused.
Input is validated (empty, non-text, too long, no readable words). Exported CSVs are protected against
**CSV formula injection**: cells starting with `=`, `+`, `-` or `@` are escaped, so spreadsheet programs
don't run them as formulas.

## Run tests

```bash
python -m pytest
```

Tests also run automatically with GitHub Actions (`.github/workflows/tests.yml`) on every pull request
into `dev` or `main`, on Python 3.12 and 3.14.
