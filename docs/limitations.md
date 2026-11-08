# Limitations of SmartSpam

SmartSpam is a **text classification** project: it estimates whether the *wording* of an SMS message looks
like the spam it was trained on. It is **not** a security tool and does **not** provide protection against
phishing, scams, malware or other cyberattacks. This page explains why, using results measured in this project.

## 1. It misses some spam, even on its own test set

On the held-out test set, the model caught **76 of 90** spam messages (recall 84.4%). Roughly **1 in 6 spam
messages got through**. The test set contains only 90 spam messages, so this estimate is itself uncertain:
a 95% confidence interval for recall is roughly **76%–90%**.

The model favors avoiding false alarms (only 4 of 675 legitimate messages were flagged), so its mistakes
tend to be spam slipping through rather than real messages being blocked.

## 2. The training data is old and narrow

The [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) was published in
2011–2012. Its messages come mainly from UK and Singapore sources and are mostly **prize, ringtone,
premium-rate and dating spam** from that era. Modern scams look different, and the model handles them poorly:

| Message (written for testing, not from the dataset) | Model output |
|---|---|
| "Your package could not be delivered. Reschedule here: http://parcel-redelivery.info" | **HAM**, spam probability 19%: missed |
| "URGENT: Your bank account has been suspended. Verify now at http://secure-bank-login.com" | SPAM, but only 57%: barely caught |
| "WINNER!! You have been selected to receive a £900 prize reward! …" (2011-style) | SPAM, 98% |

Delivery scams, bank and account-lock phishing, one-time-code (MFA) fraud, cryptocurrency scams and
messages impersonating real companies are under-represented or absent in the training data.

## 3. It only reads words; it can't inspect links, senders or attachments

During cleaning, every URL becomes the same placeholder word (`urltoken`), and phone numbers become
`phonetoken`. The model therefore **cannot tell a safe link from a malicious one**, check where a link leads,
look up a sender's reputation, or scan attachments. Real anti-phishing systems combine many of these signals.

## 4. Word order is ignored

The model averages the word vectors in a message (global average pooling). This makes it fast and simple,
but it means **word order has no effect at all**: the same words in any order get exactly the same score.

| Message | Spam probability |
|---|---|
| "call me now about the free prize" | 0.3995 |
| "prize free the about now me call" | 0.3995 (identical) |

Individual words such as "not" still affect the average, but the model can't understand how words relate
to each other in a sentence.

## 5. It can be fooled on purpose

Spammers can avoid known spam words with misspellings or substitutions. Words never seen in training
become a single "unknown" token that carries little meaning. For example, "fr33 pr1ze c-a-s-h claim now"
was classified as **HAM** (spam probability 0.405), even though its meaning is obvious to a person.

## 6. English only

The vocabulary was learned from English messages with British and Singaporean texting slang. Messages in
other languages, or mostly made of emoji, are effectively unknown to the model.

## 7. Confidence isn't a guarantee

The confidence score is the model's own probability estimate. It has not been calibrated against real-world
data, and the model can be confidently wrong. A message shown as "97% HAM" can still be a scam.

## 8. Privacy

The app doesn't save messages that are typed or uploaded. If it is deployed publicly, however, messages are
processed on the hosting provider's servers. **Don't paste personal, financial or confidential messages into
a public deployment.**

## What responsible use looks like

- Treat SmartSpam's output as an **educational demonstration** of text classification, not as advice.
- Never decide whether to click a link, call a number or share information based on its result.
- Report real scam texts to your carrier (in the US, forward them to **7726**) and follow official guidance
  from your bank or government.

## How these limitations could be reduced

See *Future improvements* in the [README](../README.md#future-improvements): newer and more diverse training
data, class weighting and threshold tuning to improve recall, sequence-aware models such as LSTMs or
transformers, and combining text classification with URL and sender reputation checks.
