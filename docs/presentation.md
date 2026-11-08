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
| 0:00–0:25 | **Introduction** | App title | "SmartSpam is a web app that uses a neural network to classify text messages as spam or legitimate. I trained it on the UCI SMS Spam Collection, about 5,500 real messages." |
| 0:25–1:20 | **Message Scanner** | Click **Prize scam** → **Predict** (SPAM). Click **Friend** → **Predict** (HAM). Clear the box → **Predict** (warning). | "The app loads my trained PyTorch model; there are no keyword rules. Before predicting, the message is cleaned exactly like the training data. Using the same processing in training and prediction is essential." |
| 1:20–2:10 | **Link Safety Check** | Click **Delivery scam** → **Predict**. Point to HAM, then to the red **High risk** link box and the warning below it. | "Here's a modern scam. The AI says HAM: its training data is from 2011, and it sees every link as the same placeholder word. So I added a separate Link Safety Check. It never opens the link; it looks for phishing signs like abused domain endings and words like 'redelivery'. It flags this link as high risk and warns the user." |
| 2:10–2:50 | **Batch CSV** | Upload `labeled_messages.csv`. Point to the ✅/❌ column and "right on 19 of 20". | "Users can upload many messages at once and download the results. If the file includes the correct answers, the app scores itself. It got 19 of 20 right; the miss is that same kind of delivery scam." |
| 2:50–3:55 | **Model Dashboard** | Metric tiles with the change from v1.0, confusion matrix, training chart | "These are results on 765 test messages the model never saw: 97.65% accuracy. Version 1 missed about 1 in 6 spam messages. I compared four fixes on the validation set; class weighting didn't help, but lowering the decision threshold to 30% did. On the test set, recall went from 84.4% to 85.6%, a modest gain, at the cost of one extra false alarm. Accuracy alone is misleading here: predicting 'not spam' for everything would score 88%." |
| 3:55–4:30 | **How it's built** | GitHub: branches, a pull request with green checks | "The network is an embedding layer, average pooling, and dense layers with dropout. I developed it in feature branches merged through pull requests, with 127 automated tests running on GitHub Actions for every pull request." |
| 4:30–4:50 | **Limitations & wrap-up** | Disclaimer at the bottom of the app | "This is a learning project, not security protection: the link check uses simple rules, and the model still misses some modern scams. Next steps would be newer data and models that understand word order. Thank you." |

**If something goes wrong:** if the deployed app is slow, switch to the local one. If the local app won't
start, show the screenshots in the README and run `python -m src.predict "Congratulations! You've won a free iPhone"` in the terminal.

## Likely questions

- **Why PyTorch instead of TensorFlow?** TensorFlow had no packages for Python 3.14; the brief allowed either.
  PyTorch also made me write the training loop myself.
- **How did you prevent data leakage?** I removed duplicate messages before splitting, built the vocabulary from
  the training set only, and used the test set exactly once, for final evaluation.
- **Why is recall lower than precision?** Spam is only 12% of the data, and many missed spam messages read like
  normal texts. I tried class weighting and a lower threshold, comparing them on the validation set only so the
  test score stays honest. The lower threshold worked best, but the test-set gain was small.
- **Why didn't class weighting help?** With weighting, the model's probabilities shift upward, so the best
  threshold also moved up (to 0.68–0.89), and the net effect was similar or worse. On this data, adjusting the
  threshold alone was enough.
- **Isn't the Link Safety Check a keyword rule?** It's a separate layer that checks links, not the message text,
  and it's labelled as rule-based in the app. Every SPAM/HAM prediction still comes from the neural network.
  Real spam filters combine ML with rules like these.
- **Does the link check visit the website?** No. It only reads the URL text, which is safer, but it means it
  can't see where shortened links lead.
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
