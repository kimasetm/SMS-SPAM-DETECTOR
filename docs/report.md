# SmartSpam — SMS Spam Detection System: Project Report

**Course:** Artificial Intelligence / Machine Learning · **Instructor:** Professor Griffin
**Repository:** <https://github.com/kimasetm/SMS-SPAM-DETECTOR>

> **Draft.** Review and rewrite this report in your own words before submitting it.

## 1. What I built

SmartSpam is a web application that classifies SMS text messages as **spam** or **ham** (legitimate) with a
neural network. A user types a message, clicks **Predict**, and sees the prediction with a confidence score, plus
separate checks on any links and on the sender. I built it in three releases, each driven by the measured
weaknesses of the previous one:

- **v1.0 — the core system.** A data pipeline for the UCI SMS Spam Collection (5,574 messages; 478 duplicate or
  empty messages removed; stratified 70/15/15 split), a PyTorch neural network (text vectorization → embedding →
  global average pooling → dense layers with dropout), a reproducible training pipeline with early stopping, a
  test-set evaluation, a prediction engine, and a Streamlit app with a message scanner, CSV batch classification
  and an evaluation dashboard.
- **v2.0 — catching more spam and judging links.** A decision threshold chosen by a controlled experiment, and a
  rule-based **Link Safety Check** that flags phishing warning signs in links.
- **v3.0 — modern data and robustness.** 778 newer messages (including 180 smishing) from a second dataset and a
  separate **modern test set**; **misspelling repair** ("fr33" → "free"); a 1D CNN experiment; **calibrated
  confidence** (temperature scaling); **drift monitoring**; and opt-in **threat intelligence** (OpenPhish blocklist,
  domain age via RDAP) plus a **sender check**.

The project has **195 automated tests**, run by GitHub Actions on every pull request, and documentation that
includes the system's limitations.

### Results

**UCI test set** (765 messages, 90 spam; identical across all versions, so results are comparable):

| Version | Accuracy | Precision | Recall | F1-score | False alarms |
|---|---|---|---|---|---|
| v1.0 (threshold 0.50) | 97.65% | 95.00% | 84.44% | 0.894 | 4 of 675 |
| v2.0 (threshold 0.30) | 97.65% | 93.90% | 85.56% | 0.895 | 5 of 675 |
| **v3.0** | **96.86%** | **83.67%** | **91.11%** | **0.872** | **16 of 675** |

**Modern test set** (195 newer messages, 80 spam, of which 51 smishing):

| Version | Accuracy | Precision | Recall | F1-score | Smishing caught |
|---|---|---|---|---|---|
| v2.0 | 91.28% | 100.00% | 78.75% | 0.881 | 44 of 51 |
| **v3.0** | **97.44%** | **97.47%** | **96.25%** | **0.969** | **49 of 51** |

v3.0 catches far more modern spam and more UCI spam, but flags more legitimate UCI messages (see section 3,
problem 14).

## 2. Why I chose these technologies

| Technology | Why |
|---|---|
| **Python** | The standard language for machine learning, with mature libraries for every step. |
| **PyTorch** | The brief allowed TensorFlow/Keras or PyTorch. TensorFlow didn't offer packages for Python 3.14, my development environment, while PyTorch did. PyTorch also made me write the training loop myself, which helped me understand forward passes, loss, backpropagation and optimizer steps. |
| **pandas** | Loading, cleaning, de-duplicating and saving the datasets as tables. |
| **scikit-learn** | A reliable stratified train/validation/test split and standard evaluation metrics. |
| **Matplotlib** | Charts of training history, confusion matrices and calibration. |
| **Streamlit** | Turns a Python script into a web app without HTML or JavaScript, and can be deployed free on Streamlit Community Cloud. |
| **pytest + GitHub Actions** | Automated tests that run on every pull request, so broken code is caught before it's merged. |
| **Git and GitHub** | Version control with a `main` / `dev` / feature-branch workflow, pull requests and tagged releases. |
| **OpenPhish and RDAP** | Free threat-intelligence sources that need no API keys, so there are no secrets to protect. |

### Model design (v3.0)

| Step | Purpose |
|---|---|
| Text cleaning | Lowercases, decodes HTML, replaces URLs, phone numbers, prices and numbers with placeholder words, and keeps digits inside words ("fr33") so disguised words can be repaired. |
| Text vectorization | Converts each word to an integer ID using a vocabulary of 6,913 words learned **only from the training set**, and pads or truncates to 50 words. Unknown words that look disguised are repaired first. PyTorch has no built-in equivalent of Keras' `TextVectorization`, so I wrote a `TextVectorizer` class. |
| Embedding (32 dimensions) | Learns a vector for each word; words used in similar contexts end up with similar vectors. |
| Global average pooling | Averages the word vectors into one vector per message, ignoring padding (the PyTorch equivalent of `GlobalAveragePooling1D` with masking). |
| Dense (16, ReLU) + dropout (0.3) | Learns combinations of features; dropout randomly disables neurons during training to reduce overfitting. |
| Dense (1) → ÷ temperature → sigmoid | Outputs the calibrated probability that the message is spam. Temperature T = 1.55 was fitted on the validation set. A message is labelled spam at ≥ 0.19 on the calibrated scale (0.10 on the raw probability, chosen on the validation set). |

Training used binary cross-entropy loss, the Adam optimizer (learning rate 0.001) and batches of 32. Training
stopped after epoch 23 because validation loss hadn't improved for 3 epochs, and the weights from the best epoch
(20) were kept. The model has 221,761 trainable parameters.

## 3. Problems I encountered and how I solved them

**v1.0**

1. **TensorFlow wasn't available for my Python version.** Instead of installing an older Python, I used PyTorch
   and wrote my own text vectorizer.
2. **Duplicate messages could leak into the test set.** I removed 478 duplicate or empty messages *before*
   splitting, and wrote a test that checks the splits don't overlap.
3. **Preventing data leakage from preprocessing.** The vocabulary is fitted only on training messages. A test
   verifies this.
4. **HTML codes in the data.** Text like `&lt;#&gt;` became fake words ("lt", "gt"), so I decoded HTML entities.
5. **A misleading accuracy number.** In the first two epochs, the model had 88% validation accuracy while
   predicting "ham" for everything. That's why I report precision, recall and F1.
6. **A security bug caught by testing.** My CSV formula-injection protection (cells starting with `=`, `+`, `-`
   or `@`) first didn't work, because pandas 3 stores text in a new data type that my check skipped. A test row
   exposed it.
7. **Large installs.** I configured the CPU-only PyTorch build, avoiding about 2 GB of GPU libraries in CI and
   on Streamlit Community Cloud.
8. **Sensitive content.** Misclassified test messages contain explicit spam text, so I keep them out of the
   public repository.

**v2.0**

9. **Low spam recall (84%).** I compared four approaches **on the validation set only**, with the selection rule
   (highest validation F1) fixed before running. Class weighting didn't help; lowering the threshold to 0.30 did.
   On the test set, the gain was small (+1 spam caught, +1 false alarm): improvements measured on a small
   validation set often shrink on new data.
10. **The model couldn't judge links.** Every URL becomes the word `urltoken`, so I added a separate, clearly
    labelled rule-based Link Safety Check that never opens links. I fixed a bug where email addresses were
    treated as links.

**v3.0**

11. **Old training data.** I found a properly licensed newer dataset (Mishra & Soni, 2022, CC BY 4.0). Many of its
    messages were copies of UCI messages, so I kept only the 778 new ones and gave them their own test set. Training
    on them raised modern-test recall from 78.8% to 91.3%. (Downloading it failed at first with "403 Forbidden"
    because the server rejected Python's default User-Agent.)
12. **Improving the cleaning shifted the test set.** Keeping digits inside words changed which messages counted as
    duplicates, and one spam message moved out of the UCI test set, which would have broken comparisons with earlier
    versions. I separated the *duplicate key* (fixed, based on the original cleaning) from the *model input*, and
    proved that every split is row-for-row identical to before.
13. **The CNN didn't win.** A 1D CNN, which reads word order, scored lower validation F1 than average pooling
    (0.911 vs 0.924 with misspelling repair), so I kept the simpler model. On a small dataset, more complex isn't
    automatically better.
14. **A trade-off I had to accept.** The model chosen by the rule (average pooling + misspelling repair) has a low
    threshold, which tripled UCI false alarms (6 → 16) while catching more spam. Switching to another model *after*
    seeing test results would have made the test numbers unreliable, so I kept it and documented the trade-off.
15. **Bugs in my own experiments.** A rerun silently gave every variant misspelling repair, because variants
    inherited a changed default; I made every experiment setting explicit. My robustness test also inflated scores
    (changing a word's first letter created a "number" token that the model links to spam), and I first overstated
    how much misspelling repair improved robustness. When I separated the effects, most of the gain came from the
    lower threshold; at a fixed threshold, repair still raised disguised modern spam caught from 69 to 75 of 80.
16. **Calibration helped only partly.** Temperature scaling improved calibration on the UCI test set but not on the
    modern set, where the model is underconfident; one temperature can't fit both kinds of data.
17. **Threat-intelligence edge cases.** Domain age was looked up for shared platforms like `github.io`, which says
    nothing about one user's page, so those are skipped. Online checks are opt-in, because they send link addresses
    to third parties, and network failures are reported instead of crashing the app.

## 4. What I learned

- **Machine learning concepts:** embeddings, pooling, convolutions, dropout, binary cross-entropy, backpropagation,
  early stopping, and confidence calibration with temperature scaling.
- **Evaluation:** why held-out test sets matter, why accuracy is misleading on imbalanced data, the precision/recall
  trade-off, and why testing on a *separate, newer* test set reveals weaknesses that the original test set hides.
- **Data leakage and reproducibility:** removing duplicates before splitting, fitting preprocessing on training data
  only, fixing random seeds, and keeping splits stable when the code changes.
- **Running fair experiments:** fixing the selection rule before looking at results, making choices on the
  validation set, isolating one change at a time (ablation), and reporting modest or negative results honestly.
- **Security thinking:** how phishing links hide their real destination, how threat intelligence and domain age
  help, why "no warning signs" doesn't mean safe, and the privacy cost of online lookups.
- **Software engineering:** reusable modules, input validation, automated tests (including tests that fake network
  calls), CI, feature branches, pull requests and tagged releases.
- **Responsible AI:** a model is only as good as its data; monitoring matters because models fail silently on new
  kinds of messages (a Spanish test showed a normal-looking spam rate while most scams were missed).

## 5. Limitations and possible improvements

The full discussion is in [limitations.md](limitations.md). In short: the model still makes mistakes in both
directions and has more false alarms than v1.0; social-engineering scams without links or prizes are missed; the
data is small and English-only; word order is ignored; calibration is only partly effective; and the link and sender
checks are rule-based, never visit links and can be evaded.

Possible improvements:

1. **A more stable threshold choice** (cross-validation or a cost-based rule) to reduce false alarms.
2. **More, newer and multilingual data**, which could let sequence models such as LSTMs or small transformers help.
3. **Calibration per data source**, since one temperature can't fit both old and modern messages.
4. **Stronger link intelligence**, such as Google Safe Browsing (API key in Streamlit secrets) and sandboxed link expansion.
5. **Continuous monitoring and retraining** based on aggregate drift statistics.

## 6. Use of AI tools

I used an AI coding assistant (Claude Code) during development to help write and review code, tests and
documentation. I reviewed, ran and tested the work at each phase, and I can explain how each part works.

## 7. How to run the project

See the [README](../README.md) for installation, training, evaluation, testing and app instructions.
