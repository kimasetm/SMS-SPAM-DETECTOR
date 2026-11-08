# Five-Minute Demo Outline and Screen-Recording Checklist

## Before the demo (10 minutes ahead)

- [ ] Open a terminal in the project folder and activate the environment: `.venv\Scripts\activate`
- [ ] Start the app: `streamlit run app.py`, then open <http://localhost:8501>
      (or open the deployed app link so it has time to wake up)
- [ ] Have `samples/labeled_messages.csv` ready to upload (or download it from the Batch CSV tab)
- [ ] Open the GitHub repository in a second browser tab: <https://github.com/kimasetm/SMS-SPAM-DETECTOR>
- [ ] Close unrelated tabs and turn off notifications
- [ ] Zoom the browser to about 125% so text is readable on a projector or recording

## Demo script (≈ 5 minutes)

| Time | Section | What to show | What to say |
|---|---|---|---|
| 0:00–0:30 | **Introduction** | App title | "SmartSpam is a web app that uses a neural network to classify text messages as spam or legitimate. I trained it on the UCI SMS Spam Collection, about 5,500 real messages." |
| 0:30–1:45 | **Message Scanner** | Click **Prize scam** → **Predict** (SPAM). Click **Friend** → **Predict** (HAM). Clear the box → **Predict** (warning). Open **What did the model see?** | "The app loads my trained PyTorch model; there are no keyword rules. Before predicting, the message is cleaned exactly like the training data: phone numbers become `phonetoken`, and so on. Using the same processing in training and prediction is essential." |
| 1:45–2:45 | **Batch CSV** | Upload `labeled_messages.csv`. Point to the ✅/❌ column and "right on 19 of 20". Point to the one ❌ (the parcel-delivery scam). | "Users can upload many messages at once and download the results. If the file includes the correct answers, the app scores itself. It got 19 of 20 right. The one it missed is a modern delivery scam; the training data is from around 2011, so it hasn't seen this kind of message." |
| 2:45–4:00 | **Model Dashboard** | Metric tiles, confusion matrix, training chart | "These are results on a test set of 765 messages the model never saw during training: 97.65% accuracy and 95% precision, but 84% recall, so it misses about 1 in 6 spam messages. Accuracy alone is misleading here: predicting 'not spam' for everything would score 88%. The chart shows early stopping: training stopped when validation loss stopped improving, to avoid overfitting." |
| 4:00–4:40 | **How it's built** | GitHub: branches, a pull request with green checks | "The network is an embedding layer, average pooling, and dense layers with dropout. I developed it in six feature branches merged through pull requests, and 86 automated tests run on GitHub Actions for every pull request." |
| 4:40–5:00 | **Limitations & wrap-up** | Disclaimer at the bottom of the app | "This is a text classifier, not security protection; it can't check links and it misses some modern scams. Improvements would be newer data, class weighting to catch more spam, and models that understand word order. Thank you." |

**If something goes wrong:** if the deployed app is slow, switch to the local one. If the local app won't
start, show the screenshots in the README and run `python -m src.predict "Congratulations! You've won a free iPhone"` in the terminal.

## Likely questions

- **Why PyTorch instead of TensorFlow?** TensorFlow had no packages for Python 3.14; the brief allowed either.
  PyTorch also made me write the training loop myself.
- **How did you prevent data leakage?** I removed duplicate messages before splitting, built the vocabulary from
  the training set only, and used the test set exactly once, for final evaluation.
- **Why is recall lower than precision?** Spam is only 12% of the data, and many missed spam messages read like
  normal texts. I didn't tune the model on the test results, because that would make the score dishonest. Class
  weighting would be the next step.
- **What does dropout do?** It randomly switches off 30% of the neurons during training so the network can't rely
  on any single feature, which reduces overfitting. It's turned off when making predictions.
- **What is the confidence score?** The model's probability for the label it chose. For example, a spam
  probability of 0.03 means HAM with 97% confidence.
- **Does the dashboard change when I type messages?** No. Accuracy needs correct answers, which typed messages
  don't have, and the model doesn't learn from user input.

## Screen-recording checklist

- [ ] Record at 1080p. On Windows, use **Win + Alt + R** (Xbox Game Bar) or OBS Studio.
- [ ] Test your microphone; record somewhere quiet.
- [ ] Do one practice run with a timer: aim for **4:30**, leaving a buffer under the 5-minute limit.
- [ ] Follow the demo script above, in order.
- [ ] Don't show passwords, emails, notifications or personal messages on screen.
- [ ] Watch the recording once: check that the audio is clear and the text is readable.
- [ ] Submit the **recording** and the **GitHub repository URL**: <https://github.com/kimasetm/SMS-SPAM-DETECTOR>
