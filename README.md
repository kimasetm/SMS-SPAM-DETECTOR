# SmartSpam — SMS Spam Detection System

[![Tests](https://github.com/kimasetm/SMS-SPAM-DETECTOR/actions/workflows/tests.yml/badge.svg)](https://github.com/kimasetm/SMS-SPAM-DETECTOR/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-neural%20network-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-web%20app-ff4b4b)

SmartSpam is a machine learning web application that classifies SMS text messages as **SPAM** or **HAM**
(legitimate). A PyTorch neural network, trained on the UCI SMS Spam Collection, powers a Streamlit app with a
message scanner, CSV batch classification and a model evaluation dashboard.

> ⚠️ SmartSpam is a text classification project for learning purposes. It is **not** protection against
> phishing, scams or malware. See [Limitations](#limitations).

**Live demo:** not deployed yet. See [docs/deployment.md](docs/deployment.md) to deploy it on Streamlit Community Cloud.

## Screenshots

| Message scanner | Batch CSV with accuracy check |
|---|---|
| ![Message scanner](docs/screenshots/scanner.png) | ![Batch CSV](docs/screenshots/batch.png) |

<details>
<summary><b>Model evaluation dashboard</b></summary>

![Model dashboard](docs/screenshots/dashboard.png)
</details>

## Features

- **Message scanner:** paste a message and get SPAM or HAM with a confidence score, a spam-probability bar,
  example messages, an empty-input warning, and a view of the cleaned text the model actually reads.
- **Batch CSV classification:** upload up to 5,000 messages, review the results and download the predictions.
  If the file includes a `label` column, the app also reports accuracy, precision, recall and F1 on your file.
- **Model evaluation dashboard:** real test-set metrics, the confusion matrix and training charts.
- **Trained neural network:** every prediction comes from the model; there are no keyword rules.
- **Reproducible pipeline:** fixed random seeds, so retraining produces identical weights.
- **Input validation and safe export:** clear errors for empty, overly long or unreadable input; exported CSVs are
  protected against spreadsheet **formula injection**.
- **86 automated tests**, run by **GitHub Actions** on every pull request (Python 3.12 and 3.14).

## Technologies

| Area | Tools |
|---|---|
| Language | Python |
| Neural network | PyTorch |
| Data processing | pandas, NumPy |
| Evaluation | scikit-learn |
| Charts | Matplotlib |
| Web app | Streamlit |
| Testing and CI | pytest, Streamlit AppTest, GitHub Actions |
| Version control | Git, GitHub (feature branches and pull requests) |

## Machine learning architecture

```
SMS text
  → clean_text()                 lowercase, decode HTML, replace URLs / phone numbers / prices / numbers
                                 with placeholder words, remove punctuation
  → TextVectorizer               words → integer IDs (vocabulary built from training data only),
                                 padded or truncated to 50 words
  → Embedding (6,314 × 32)       a learned 32-number vector for each word
  → Global average pooling       averages the word vectors (ignoring padding) → one vector per message
  → Dropout (0.3)
  → Dense (16, ReLU)
  → Dropout (0.3)
  → Dense (1) → sigmoid          probability that the message is spam (≥ 0.5 → SPAM)
```

- 202,593 trainable parameters.
- Training: binary cross-entropy loss, Adam optimizer (learning rate 0.001), batch size 32.
- **Early stopping:** stops when validation loss hasn't improved for 3 epochs and keeps the best weights.
  The current model stopped after epoch 18; the best epoch was 15.
- The same `clean_text()` and vocabulary are used for training and prediction, so the model always gets input
  in the format it learned from.

The project brief allowed TensorFlow/Keras or PyTorch. PyTorch was chosen because TensorFlow didn't provide packages
for Python 3.14, the development environment. `TextVectorizer` and masked average pooling are the PyTorch
counterparts of Keras' `TextVectorization` and `GlobalAveragePooling1D`.

![Training history](reports/figures/training_history.png)

## Dataset

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) by Almeida, Gómez Hidalgo
and Yamakami (2011): 5,574 English SMS messages labelled ham or spam.

| Step | Result |
|---|---|
| Raw messages | 5,574 (about 13% spam) |
| After removing duplicate and empty messages | 5,096 (478 removed) |
| Training / validation / test split (stratified 70 / 15 / 15) | 3,567 / 764 / 765 |

**Preventing data leakage:** duplicates are removed *before* splitting, so no message appears in both training and
test data; the vocabulary is built from the training set only; and the test set is used once, for final evaluation.
The dataset isn't stored in this repository; `python -m src.download_data` downloads it.

## Evaluation results

Measured on the **held-out test set**: 765 messages (675 ham, 90 spam) that weren't used for training or model selection.

| Metric | Value |
|---|---|
| Accuracy | **97.65%** |
| Precision (spam) | **95.00%**: of the messages flagged as spam, 95% were spam |
| Recall (spam) | **84.44%**: the model caught 76 of the 90 spam messages |
| F1-score (spam) | **0.8941** |
| ROC-AUC | **0.9863** |

<img src="reports/figures/confusion_matrix.png" alt="Confusion matrix" width="420">

The model rarely raises false alarms (4 of 675 legitimate messages flagged), but it misses about 1 in 6 spam
messages. Because only ~12% of messages are spam, accuracy alone is misleading: labelling everything "ham" would
score 88%. That's why precision, recall and F1 are reported. Full results are in [`reports/metrics.json`](reports/metrics.json).

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/kimasetm/SMS-SPAM-DETECTOR.git
cd SMS-SPAM-DETECTOR

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

The trained model is included in `models/`, so you can run the app right away.

## Usage

### Run the web app

```bash
streamlit run app.py
```
Then open <http://localhost:8501>.

### Train and evaluate the model

```bash
python -m src.download_data     # download the dataset into data/raw/
python -m src.preprocessing     # clean, de-duplicate and split the data
python -m src.train             # train the network (about 15 seconds on a laptop CPU)
python -m src.evaluate          # test-set metrics and charts → reports/
```

### Predict from the command line

```bash
python -m src.predict "WINNER!! You have been selected to receive a £900 prize reward! Call 09061701461"
# SPAM  (confidence 97.7%, spam probability 0.9774)

python -m src.batch samples/labeled_messages.csv predictions.csv
```

The CSV needs a `message` column (`text`, `sms` or `body` also work). An optional `label` column (spam/ham) enables
accuracy scoring. Limits: 5 MB and 5,000 rows.

### Run the tests

```bash
python -m pytest
```

## Project structure

```
├── app.py                  # Streamlit web app
├── src/
│   ├── config.py           # paths, hyperparameters and settings
│   ├── download_data.py    # downloads the UCI dataset
│   ├── preprocessing.py    # cleaning, label encoding, de-duplication, train/val/test split
│   ├── text_vectorizer.py  # words → integer sequences (fitted on training data only)
│   ├── model.py            # the neural network, save/load, batch inference
│   ├── train.py            # reproducible training with early stopping
│   ├── metrics.py          # accuracy, precision, recall, F1, ROC-AUC, confusion matrix
│   ├── evaluate.py         # test-set evaluation and charts
│   ├── predict.py          # prediction engine with input validation
│   └── batch.py            # CSV reading, batch classification, label scoring, safe export
├── tests/                  # 86 pytest tests, including Streamlit UI tests
├── models/                 # trained weights, vocabulary, config, training history
├── reports/                # metrics.json and evaluation charts
├── samples/                # example CSV files
├── docs/                   # report, limitations, deployment guide, demo outline, screenshots
├── data/                   # dataset (downloaded, not committed)
├── .github/workflows/      # CI: tests on every pull request
└── .streamlit/config.toml  # app theme and upload limit (no secrets)
```

## Development workflow

The project was developed incrementally with Git branches:

- `main`: stable releases
- `dev`: integration branch where finished features are combined and tested
- Feature branches cut from `dev` and merged back by pull request: `feature/data-preprocessing`, `feature/neural-network`,
  `feature/model-evaluation`, `feature/prediction-engine`, `feature/streamlit-ui`, `feature/testing-docs`

GitHub Actions runs the full test suite on every pull request into `dev` or `main`.

## Limitations

SmartSpam reads only the **words** of a message, learned from data collected around 2011. Key limitations:

- It misses about 1 in 6 spam messages on the test set, and handles modern scams poorly. For example, it labelled a
  parcel-delivery phishing text as HAM.
- It can't inspect links, senders or attachments; every URL becomes the same placeholder word.
- It ignores word order and can be fooled by misspellings like "fr33 pr1ze".
- It only understands English, and its confidence scores aren't calibrated.

Full details, with measured examples: **[docs/limitations.md](docs/limitations.md)**.

## Future improvements

- Improve spam recall with class weighting, and tune the decision threshold on the validation set.
- Train on newer and more diverse data, including modern phishing (smishing) messages and other languages.
- Try sequence-aware models (LSTM, 1D CNN or a small pre-trained transformer) that understand word order.
- Add character-level features to resist misspellings.
- Combine text classification with URL reputation and sender checks, as real spam filters do.
- Calibrate confidence scores and monitor the model over time.
- Deploy publicly on Streamlit Community Cloud.

## Documentation

- [Project report](docs/report.md): what was built, technology choices, problems solved, lessons learned
- [Limitations](docs/limitations.md): what the model can't do, with measured examples
- [Deployment guide](docs/deployment.md): Streamlit Community Cloud
- [Demo outline](docs/presentation.md): five-minute presentation script and recording checklist

## Acknowledgments

Dataset: T. A. Almeida, J. M. Gómez Hidalgo and A. Yamakami, "Contributions to the Study of SMS Spam Filtering:
New Collection and Results," *Proceedings of the 2011 ACM Symposium on Document Engineering (DocEng '11)*.
Available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
