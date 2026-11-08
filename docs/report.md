# SmartSpam — SMS Spam Detection System: Project Report

**Course:** Artificial Intelligence / Machine Learning · **Instructor:** Professor Griffin
**Repository:** <https://github.com/kimasetm/SMS-SPAM-DETECTOR>

## 1. What I built

SmartSpam is a web application that uses a neural network to classify SMS text messages as **spam** or **ham**
(legitimate). A user types or pastes a message, clicks **Predict**, and sees whether the trained model thinks it is
spam, along with a confidence score. The app has three tabs:

- **Message Scanner:** classifies one message. Separate checks also look at any links in the message (phishing
  warning signs, plus an optional phishing blocklist and domain-age lookup) and at the sender (for example,
  premium-rate numbers).
- **Batch CSV:** classifies up to 5,000 messages from an uploaded file and lets the user download the results.
  If the file includes the correct answers, the app also scores its own accuracy.
- **Model Dashboard:** shows how well the model performs on test data it never saw during training.

I built the project in three releases. Each release fixed weaknesses that I had measured in the previous one:

| Release | Main additions |
|---|---|
| **v1.0** | Data pipeline, PyTorch neural network, training with early stopping, evaluation, prediction engine, Streamlit app |
| **v2.0** | Decision threshold chosen by an experiment; rule-based Link Safety Check |
| **v3.0** | Newer training data with SMS phishing ("smishing"); repair of disguised words like "fr33"; calibrated confidence; drift monitoring; phishing blocklist, domain-age and sender checks |

The model is trained on two public datasets: the **UCI SMS Spam Collection** (5,574 messages from about 2011) and
778 newer messages from the **SMS Phishing Dataset** by Mishra and Soni (2022), of which 180 are smishing. I removed
duplicate messages before splitting the data into training (70%), validation (15%) and test (15%) sets, so that no
message appears in more than one set. The newer messages have their own test set, so I can measure performance on
modern scams separately.

### The neural network

| Step | What it does |
|---|---|
| Text cleaning | Lowercases the text and replaces URLs, phone numbers, prices and numbers with placeholder words |
| Text vectorization | Turns each word into a number using a vocabulary learned from the training data only; repairs disguised words such as "fr33" → "free" |
| Embedding (32 dimensions) | Learns a vector for each word, so words used in similar ways get similar vectors |
| Global average pooling | Averages the word vectors into one vector per message |
| Dense layer (16, ReLU) with dropout (0.3) | Learns combinations of features; dropout randomly switches off neurons during training to reduce overfitting |
| Output layer + sigmoid | Produces the probability that the message is spam, adjusted by temperature scaling so the confidence is more realistic |

I trained it with binary cross-entropy loss and the Adam optimizer, and used **early stopping**: training stops when
the validation loss stops improving, and the best weights are kept. All random seeds are fixed, so retraining gives
exactly the same model.

### Results

The results come from test messages the model never saw during training. The UCI test set is identical for every
release, so the numbers can be compared directly.

| | v1.0 | v2.0 | **v3.0** |
|---|---|---|---|
| UCI test (765 messages): spam caught (recall) | 84.4% | 85.6% | **91.1%** |
| UCI test: legitimate messages wrongly flagged | 4 of 675 | 5 of 675 | **16 of 675** |
| UCI test: F1-score | 0.894 | 0.895 | 0.872 |
| Modern test (195 messages): spam caught | — | 78.8% | **96.3%** |
| Modern test: smishing caught | — | 44 of 51 | **49 of 51** |

v3.0 catches much more spam, especially modern scams, but it also flags more legitimate messages. I explain this
trade-off in section 3.

## 2. Why I chose these technologies

| Technology | Why I chose it |
|---|---|
| **Python** | The standard language for machine learning, with libraries for every step of the project |
| **PyTorch** | The assignment allowed TensorFlow/Keras or PyTorch. TensorFlow didn't support Python 3.14, which I was using, but PyTorch did. PyTorch also made me write the training loop myself, which helped me understand how a network actually learns |
| **pandas** | Loading, cleaning and splitting the datasets |
| **scikit-learn** | Reliable data splitting and standard evaluation metrics (accuracy, precision, recall, F1, confusion matrix) |
| **Matplotlib** | Charts of the training progress, confusion matrices and calibration |
| **Streamlit** | Turns Python code into a web app without writing HTML or JavaScript, and can be hosted for free |
| **pytest and GitHub Actions** | 195 automated tests that run on every pull request, so mistakes are caught before code is merged |
| **Git and GitHub** | Version control with `main`, `dev` and feature branches, pull requests and tagged releases |

## 3. Problems I encountered

1. **TensorFlow didn't work with my Python version.** I switched to PyTorch. Because PyTorch has no ready-made
   text-vectorization layer, I wrote my own.

2. **Duplicate messages could make the results look too good.** The dataset contains many copies of the same spam
   message. If one copy is in the training data and another in the test data, the model is tested on something it
   has already seen. I removed 478 duplicates before splitting the data, and later I also removed newer messages
   that were copies of UCI messages.

3. **Accuracy was misleading.** Only about 12% of the messages are spam, so a model that labels everything "ham"
   still scores 88% accuracy. That actually happened in the first epochs of training. That's why I focused on
   precision, recall and F1-score.

4. **The model missed too much spam.** v1.0 caught only 84% of the spam. I compared several fixes on the
   validation set, deciding in advance that I would choose the one with the best F1-score. Giving spam more weight
   during training didn't help; lowering the decision threshold did, but only slightly on the test set.

5. **The model couldn't judge links.** It sees every link as the same placeholder word, so it can't tell a safe
   link from a phishing link. I added separate, clearly labelled link checks that never open the link. One bug I
   found: email addresses were being treated as links.

6. **The training data was old.** v2.0 labelled a modern parcel-delivery scam as legitimate. I found a newer,
   properly licensed dataset with smishing messages. Adding it raised the share of modern spam caught from 79% to
   91%, and the final model, which also repairs disguised words, catches 96%.

7. **Improving the text cleaning moved a message out of the test set.** That would have made comparisons with
   earlier versions unfair. I separated the key used to find duplicates from the text the model reads, and checked
   that every data split stayed exactly the same.

8. **A more complex model didn't help.** I built a 1D convolutional network (CNN), which reads short phrases, so word
   order matters. On the validation set it performed slightly worse than the simpler model, so I kept the simpler one.

9. **More spam caught, but more false alarms.** The model chosen by my selection rule uses a low threshold, so false
   alarms on the UCI test set rose to 16 out of 675 (the model before it had 6). I kept it anyway. Choosing a different model after seeing
   the test results would have made those results unreliable, so instead I documented the trade-off.

10. **Mistakes in my own experiments.** Twice my testing gave misleading numbers: a setting changed silently between
    experiments, and my robustness test for disguised words inflated the results. When I separated the effects, I
    found that I had overstated how much the misspelling repair helped. I fixed the code and corrected the
    documentation.

11. **Security and privacy.** I protected downloaded CSV files against formula injection, which caught a real bug
    during testing. I made the online link checks optional, because they send the link to outside services, and I
    kept explicit spam messages out of the public repository.

## 4. What I learned

- **How a neural network for text works:** embeddings, pooling, dense layers, dropout, loss functions,
  backpropagation and early stopping.
- **How to evaluate a model honestly:** keep a separate test set, look beyond accuracy, understand the trade-off
  between precision and recall, and test on newer data to find weaknesses the original test set hides.
- **How to avoid data leakage:** remove duplicates before splitting, and build the vocabulary from training data only.
- **How to run fair experiments:** decide how to choose a model before seeing the results, make choices on the
  validation set, change one thing at a time, and report results honestly, even when an idea doesn't work.
- **Security thinking:** how phishing links disguise their destination, and why simple, explainable rules are a useful
  addition to a machine learning model.
- **Software engineering:** organizing code into modules, writing automated tests, using continuous integration, and
  working with branches, pull requests and releases.
- **Responsible AI:** a model is only as good as its data, and it can fail silently on new kinds of messages, which is
  why monitoring matters.

## 5. Limitations and possible improvements

SmartSpam is a learning project, **not** protection against phishing or scams. Its main limitations:

- It still makes mistakes in both directions, and it flags more legitimate messages than v1.0 did.
- It misses scams that read like normal conversation, such as *"Hi mum, I dropped my phone, can you send me £200?"*
- It only understands English, and its training data is small.
- It ignores word order, and the link and sender checks are simple rules that can be evaded.

Full details, with measured examples, are in [limitations.md](limitations.md).

Possible improvements:

1. **A more stable way to choose the threshold**, such as cross-validation, to reduce false alarms.
2. **More, newer and multilingual training data**, which could also let more advanced models (LSTMs or transformers) help.
3. **Separate calibration for older and newer messages**, since one setting couldn't fit both.
4. **Stronger link checks**, such as Google Safe Browsing, with the API key stored securely.
5. **Continuous monitoring and retraining** as spam changes over time.

## 6. Use of AI tools

I used an AI coding assistant (Claude Code) to help write and review code, tests and documentation, including this
report. I decided what to build, reviewed and ran each phase, merged every change myself after testing it, and I can
explain how each part of the project works.

## 7. How to run the project

The [README](../README.md) explains how to install the project, train and evaluate the model, run the tests and start
the app.
