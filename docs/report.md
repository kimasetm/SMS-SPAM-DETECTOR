# SmartSpam — SMS Spam Detection System: Project Report

**Course:** Artificial Intelligence / Machine Learning · **Instructor:** Professor Griffin
**Repository:** <https://github.com/kimasetm/SMS-SPAM-DETECTOR>

> **Draft.** Review and rewrite this report in your own words before submitting it.

## 1. What I built

SmartSpam is a web application that classifies SMS text messages as **spam** or **ham** (legitimate) with a
neural network. A user types a message, clicks **Predict**, and sees the prediction with a confidence score.
The project includes:

- **A data pipeline** that downloads the UCI SMS Spam Collection (5,574 messages), cleans the text, removes
  478 duplicate or empty messages, and splits the remaining 5,096 messages into training (70%), validation
  (15%) and test (15%) sets, keeping the ~12% spam ratio in each.
- **A neural network** built with PyTorch: text vectorization → embedding layer → global average pooling →
  dense layers with dropout → a spam probability. It has 202,593 trainable parameters.
- **A reproducible training pipeline** with fixed random seeds, validation after every epoch and early stopping.
- **An evaluation** on the held-out test set with accuracy, precision, recall, F1-score, ROC-AUC, a confusion
  matrix and training charts.
- **A prediction engine** that applies exactly the same text processing used in training, validates input and
  returns a label with a confidence score.
- **A Streamlit web app** with a message scanner, CSV batch classification (with optional accuracy scoring
  when the file includes labels) and a model evaluation dashboard.
- **86 automated tests**, run by GitHub Actions on every pull request, and documentation that includes the
  model's limitations.

### Results (held-out test set: 765 messages, 90 spam)

| Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|
| 97.65% | 95.00% | 84.44% | 0.894 | 0.986 |

The model wrongly flagged 4 of 675 legitimate messages and missed 14 of 90 spam messages.

## 2. Why I chose these technologies

| Technology | Why |
|---|---|
| **Python** | The standard language for machine learning, with mature libraries for every step. |
| **PyTorch** | The brief allowed TensorFlow/Keras or PyTorch. TensorFlow didn't offer packages for Python 3.14, my development environment, while PyTorch did. PyTorch also made me write the training loop myself, which helped me understand forward passes, loss, backpropagation and optimizer steps. |
| **pandas** | Loading, cleaning, de-duplicating and saving the dataset as tables. |
| **scikit-learn** | A reliable stratified train/validation/test split and standard evaluation metrics, so I didn't have to implement them by hand. |
| **Matplotlib** | Charts of training history and the confusion matrix. |
| **Streamlit** | Turns a Python script into a web app without HTML or JavaScript, and can be deployed free on Streamlit Community Cloud. |
| **pytest + GitHub Actions** | Automated tests that run on every pull request, so broken code is caught before it's merged. |
| **Git and GitHub** | Version control with a `main` / `dev` / feature-branch workflow and pull requests. |

### Model design

| Layer | Purpose |
|---|---|
| Text vectorization | Converts each word to an integer ID using a vocabulary of 6,314 words learned **only from the training set**; pads or truncates every message to 50 words. PyTorch has no built-in equivalent of Keras' `TextVectorization`, so I wrote a `TextVectorizer` class. |
| Embedding (32 dimensions) | Learns a vector for each word; words used in similar contexts end up with similar vectors. |
| Global average pooling | Averages the word vectors into one vector per message, ignoring padding (the PyTorch equivalent of `GlobalAveragePooling1D` with masking). |
| Dense (16, ReLU) + dropout (0.3) | Learns combinations of features; dropout randomly disables neurons during training to reduce overfitting. |
| Dense (1) + sigmoid | Outputs the probability that the message is spam. A message is labelled spam at ≥ 0.5. |

Training used binary cross-entropy loss, the Adam optimizer (learning rate 0.001) and batches of 32.
Training stopped after epoch 18 because validation loss hadn't improved for 3 epochs, and the weights from
the best epoch (15) were kept.

## 3. Problems I encountered and how I solved them

1. **TensorFlow wasn't available for my Python version.** TensorFlow had no packages for Python 3.14. Instead of
   installing an older Python, I used PyTorch and wrote my own text vectorizer.
2. **Duplicate messages could leak into the test set.** The dataset contains many repeated spam messages. If one
   copy were in training and another in testing, the test score would be inflated. I removed 478 duplicate or
   empty messages *before* splitting, and wrote a test that checks the splits don't overlap.
3. **Preventing data leakage from preprocessing.** The vocabulary is fitted only on training messages; validation
   and test words not seen in training become an "unknown" token. A test verifies this.
4. **HTML codes in the data.** Messages contained text like `&lt;#&gt;`, which became fake words ("lt", "gt").
   I decoded HTML entities during cleaning.
5. **A misleading accuracy number.** In the first two epochs, the model had 88.35% validation accuracy while
   predicting "ham" for everything, because 88% of messages are ham. That's why I report precision, recall and
   F1, and why a test demonstrates this case.
6. **Spam recall is lower than the other metrics (84%).** It was tempting to tune the model after seeing the test
   results, but that would make the test score dishonest. I kept the test set untouched and listed better
   approaches (class weighting, tuning the threshold on the validation set) as future improvements.
7. **A security bug caught by testing.** I protect exported CSV files against *formula injection*: cells starting
   with `=`, `+`, `-` or `@` could run as formulas in Excel. My first version didn't work, because pandas 3
   stores text in a new data type that my check skipped. A test row in the sample file exposed it; I fixed it
   and added an automated test.
8. **Large installs.** The default PyTorch package for Linux includes about 2 GB of GPU libraries. I configured
   the CPU-only version so CI runs quickly and the app fits within Streamlit Community Cloud's limits.
9. **Sensitive content.** The list of misclassified test messages contains explicit spam text, so I kept it out of
   the public repository and generate it locally instead.

## 4. What I learned

- **Machine learning concepts:** how embeddings represent words as vectors, what pooling does, how dropout
  reduces overfitting, how binary cross-entropy and backpropagation train a network, and how early stopping
  uses a validation set to decide when to stop.
- **Evaluation:** why a held-out test set matters, why accuracy is misleading on imbalanced data, the trade-off
  between precision (false alarms) and recall (missed spam), and how to read a confusion matrix.
- **Data leakage:** how duplicates and preprocessing fitted on all the data can make results look better than
  they really are.
- **Reproducibility:** fixing random seeds so that retraining produces identical weights.
- **Software engineering:** organizing code into reusable modules, validating input, writing automated tests,
  using CI, and using a branch-and-pull-request workflow.
- **Responsible AI:** a model is only as good as its data. My model handles 2011-style prize spam well but
  misses modern delivery scams, so it must not be presented as security protection.

## 5. Limitations and possible improvements

The full discussion is in [limitations.md](limitations.md). In short: the training data is old and narrow; the
model reads only words (not links or senders); averaging ignores word order; it can be fooled by misspellings;
and it misses about 1 in 6 spam messages.

Possible improvements:

1. **Improve spam recall** with class weights during training, and choose the decision threshold on the
   validation set.
2. **Use newer, more diverse data,** including modern phishing (smishing) messages and other languages.
3. **Use sequence-aware models** such as an LSTM, a 1D CNN or a small pre-trained transformer, which understand
   word order.
4. **Add character-level features** to handle misspellings like "fr33".
5. **Combine signals:** check URLs against threat-intelligence lists and use sender reputation, as real spam
   filters do.
6. **Calibrate confidence scores** so that "90% confident" is right about 90% of the time.
7. **Monitor and retrain** the model as spam evolves.

## 6. Use of AI tools

I used an AI coding assistant (Claude Code) during development to help write and review code, tests and
documentation. I reviewed, ran and tested the work at each phase, and I can explain how each part works.

## 7. How to run the project

See the [README](../README.md) for installation, training, evaluation, testing and app instructions.
