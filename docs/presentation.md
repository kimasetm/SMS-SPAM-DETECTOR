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
| 0:00–0:25 | **Introduction** | App title | "SmartSpam is a web app that uses a neural network to classify text messages as spam or legitimate. I trained it on two public datasets: about 5,500 messages from 2011 plus newer messages that include SMS phishing." |
| 0:25–1:15 | **Message Scanner** | Click **Prize scam** → **Predict** (SPAM, plus the red **Premium-rate number** sender warning). Click **Friend** → **Predict** (HAM). Clear the box → **Predict** (warning). | "Every prediction comes from my trained PyTorch model; there are no keyword rules. The sender check is a separate rule: numbers starting with 09 are UK premium-rate lines. The same text cleaning is used in training and prediction, which is essential." |
| 1:15–2:00 | **Link Safety Check** | Click **Delivery scam** → **Predict**. Point to SPAM, then to the red **High risk** link box. Optionally type `FR33 PR1ZE! Cl41m n0w` → **Predict**. | "This is a modern delivery scam. Version 2 called it legitimate because it was trained only on 2011 data; adding newer training data fixed that. The model sees every link as the same placeholder word, so a separate Link Safety Check looks at the link itself, without ever opening it. Version 3 also repairs disguised words like 'fr33 pr1ze'." |
| 2:00–2:35 | **Batch CSV** | Upload `labeled_messages.csv`. Point to the ✅ column and "right on 20 of 20". | "Users can upload many messages at once and download the results. If the file includes the correct answers, the app scores itself: 20 of 20 on this small sample, which is too small to prove much on its own." |
| 2:35–3:50 | **Model Dashboard** | Recall tiles for both test sets with the change from v2, the modern test set section, then the experiment expander | "These are results on messages the model never saw. On newer messages, recall went from 79% to 96%, and it catches 49 of 51 phishing texts. But there's a trade-off: on the original test set, false alarms rose from 6 to 16 out of 675, because of the threshold chosen on validation. I kept that model because I fixed the selection rule before testing. I also tested a CNN that reads word order, and it didn't beat the simpler model, so I didn't use it." |
| 3:50–4:25 | **How it's built** | GitHub: releases v1.0 → v3.0, a pull request with green checks | "I built this in three releases using feature branches and pull requests, with 195 automated tests running on GitHub Actions for every pull request. Each release fixed measured weaknesses of the previous one." |
| 4:25–4:50 | **Limitations & wrap-up** | Disclaimer at the bottom of the app | "This is a learning project, not security protection. It still misses scams that are pure conversation, like 'Hi mum, I lost my phone, send money', and it only understands English. Thank you." |

**Online link checks:** leave the checkbox **off** during the demo unless you've tested the internet connection
beforehand. It needs the OpenPhish and RDAP services to respond.

**If something goes wrong:** if the deployed app is slow, switch to the local one. If the local app won't
start, show the screenshots in the README and run `python -m src.predict "Congratulations! You've won a free iPhone"` in the terminal.

## Likely questions

- **Why PyTorch instead of TensorFlow?** TensorFlow had no packages for Python 3.14; the brief allowed either.
  PyTorch also made me write the training loop myself.
- **How did you prevent data leakage?** I removed duplicate messages before splitting (including messages in the
  second dataset that were copies of UCI messages), built the vocabulary from the training set only, and made every
  model choice on the validation set.
- **Why are there more false alarms in v3?** The threshold that maximized F1 on the validation set is low, so the
  model catches more spam but flags more legitimate messages (16 of 675 on the UCI test set). I kept it because the
  selection rule was fixed before testing; picking a model after seeing test results would make those results
  unreliable. A more stable threshold choice is the next improvement.
- **Why didn't the CNN win?** It was more cautious (higher precision, lower recall) and scored lower validation F1.
  With only about 4,000 training messages, the extra complexity didn't pay off.
- **What is temperature scaling?** The model's raw score is divided by one number (1.55, fitted on validation data)
  before turning it into a probability, so confidence better matches reality. It changes no decisions. It helped on
  the original test set but not on the modern one.
- **How does drift monitoring work without labels?** It compares new messages with the validation data: the share of
  unknown words, the predicted spam rate and how many predictions are uncertain. Spanish messages, for example, had
  62.5% unknown words versus 4.4% normally.
- **Why didn't class weighting help?** With weighting, the model's probabilities shift upward, so the best
  threshold also moved up (to 0.68–0.89), and the net effect was similar or worse. On this data, adjusting the
  threshold alone was enough.
- **Isn't the Link Safety Check a keyword rule?** It's a separate layer that checks links, not the message text,
  and it's labelled as rule-based in the app. Every SPAM/HAM prediction still comes from the neural network.
  Real spam filters combine ML with rules like these.
- **Does the link check visit the website?** No. It only reads the URL text, which is safer, but it means it
  can't see where shortened links lead. The optional online checks only look the address up in a phishing list and
  ask the domain registry how old the domain is.
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
