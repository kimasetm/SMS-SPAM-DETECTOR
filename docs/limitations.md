# Limitations of SmartSpam

SmartSpam is a **text classification** project: it estimates whether the *wording* of an SMS message looks
like the spam it was trained on, plus some rule-based and optional online checks on links and senders. It is
**not** a security tool and does **not** provide protection against phishing, scams, malware or other
cyberattacks. This page explains why, using results measured in this project (v3.0 unless stated otherwise).

## 1. It still makes mistakes in both directions

| Test set | Spam caught | Legitimate messages wrongly flagged |
|---|---|---|
| UCI (765 messages) | 82 of 90 (91.1%) | **16 of 675 (2.4%)** |
| Modern (195 messages) | 77 of 80 (96.3%) | 2 of 115 |

The test sets are small, so these numbers are uncertain. 95% confidence intervals: UCI recall roughly **83%–95%**,
modern recall **90%–99%**, UCI false-alarm rate **1.5%–3.8%**.

**More false alarms than earlier versions.** v1.0 flagged 4 of 675 legitimate UCI messages. The current model's
decision threshold, chosen on the validation set to maximize F1, is low (0.10 on the raw probability), so it
catches more spam but also flags more real messages. Example: *"Your verification code is 482913. Do not share
it with anyone."* is flagged as **SPAM** (69.7%), although such messages are usually legitimate.

## 2. Scams that rely on conversation, not links or prizes, are missed

The model learned what 2011-era spam and some newer smishing look like: prizes, premium-rate numbers, links,
urgent account warnings. Social-engineering messages without those features look like normal conversation:

| Message (written for testing, not from the datasets) | Model output |
|---|---|
| "Hi mum, I dropped my phone, this is my new number. Can you send me £200 for rent?" | **HAM** (0.8% spam): missed |
| "Your package could not be delivered. Reschedule here: http://parcel-redelivery.info" | SPAM (65.9%) |
| "URGENT: Your bank account has been suspended. Verify now at http://secure-bank-login.com" | SPAM (79.7%) |
| "WINNER!! You have been selected to receive a £900 prize reward! …" (2011-style) | SPAM (96.5%) |

(v2, trained on UCI data only, rated the delivery scam HAM at 19%; adding newer training data fixed that one.)

## 3. The training data is small and narrow

Training uses the UCI SMS Spam Collection (2011–2012, mainly UK and Singapore) plus **778 newer messages** from the
Mishra & Soni (2022) dataset that weren't already in UCI. That's still only ~4,000 training messages, all in
English. Many modern scam types (one-time-code fraud, crypto scams, "Hi mum" scams, job scams) are rare or absent.

## 4. Link checks never visit links and can be evaded

The neural network sees every URL as the same placeholder word (`urltoken`), so links are judged by separate checks:

- **Rule-based Link Safety Check** (always on): raw IP addresses, the `@` trick, punycode look-alikes, brand names
  on unofficial domains, link shorteners, frequently abused domain endings, phishing words, unusual ports, plain `http`.
- **Online checks** (opt-in): the OpenPhish community phishing feed and domain age via RDAP.

Their limits:

- Links are **never visited**, so the checks can't see where a shortened link leads or what the page contains.
- The OpenPhish community feed holds only a few hundred currently active URLs and updates about every 12 hours, so
  new phishing sites are often not on it yet.
- Some registries don't publish RDAP data, so "no registration record" doesn't mean a domain is fake.
- Rules can be evaded: a scam on an established, plain-looking `.com` domain shows no warning signs, and a legitimate
  site on a `.info` domain may be flagged. **"No warning signs" never means "safe".**
- Certificates, page content and attachments aren't checked.

## 5. Sender checks are simple

The optional sender check recognizes UK/US premium-rate numbers, email senders, short codes and sender names. It
can't verify who really sent a message: phone numbers and sender names can be **spoofed**, and it doesn't know
regional number formats beyond those patterns.

## 6. Word order is still ignored

The model averages the word vectors in a message (global average pooling), so **word order has no effect at all**:

| Message | Spam probability |
|---|---|
| "call me now about the free prize" | 0.6285 |
| "prize free the about now me call" | 0.6285 (identical) |

A 1D CNN, which reads 3-word phrases, was implemented and compared on the validation set, but it didn't beat
average pooling on this small dataset, so it isn't used.

## 7. Misspelling repair helps, but has limits

Misspelling repair maps unknown words to known ones when they look disguised: "fr33 pr1ze c-a-s-h claim now" is now
classified as SPAM with 98.5% spam probability, whereas v1.0 gave it 40.5% and labelled it HAM (below its 50% threshold). It only handles single-letter swaps, spaced-out letters
and one typo per word; heavier disguises, new words, images of text and emoji aren't understood. Compared at the same
threshold, repair raised disguised modern spam caught from 69 to 75 of 80, but it isn't a guarantee.

## 8. English only

The vocabulary was learned from English messages. Messages in other languages are mostly unknown words to the model:
for example, the Spanish message "Hola, nos vemos mañana para almorzar?" is rated HAM simply because nothing in it
looks familiar, and Spanish scams were mostly missed in testing. The **drift monitor** flags such batches (62.5% unknown
words vs a 4.4% baseline), but it can't make the model understand them.

## 9. Confidence is calibrated only partly

Temperature scaling (fitted on the validation set) improved calibration on the UCI test set (ECE 0.017 → 0.012) but
made it worse on the modern test set (0.073 → 0.083), where the model tends to be *under*confident. A message shown
as "95% HAM" can still be a scam.

## 10. Monitoring needs batches and doesn't measure accuracy

The drift monitor compares a batch of messages with the validation data using signals that need no labels (unknown
words, predicted spam rate, uncertain predictions, length). It can't tell whether the model is right, and with very
small batches its numbers are noisy. A warning means "look more closely", not "the model is wrong".

## 11. Privacy

The app doesn't save messages that are typed or uploaded. If it is deployed publicly, messages are processed on the
hosting provider's servers, and with **online link checks** turned on, link addresses are sent to OpenPhish and RDAP
services. **Don't paste personal, financial or confidential messages into a public deployment.**

## What responsible use looks like

- Treat SmartSpam's output as an **educational demonstration** of text classification, not as advice.
- Never decide whether to click a link, call a number or share information based on its result.
- Report real scam texts to your carrier (in the US, forward them to **7726**) and follow official guidance
  from your bank or government.

## How these limitations could be reduced

See *Limitations and possible improvements* in the [project report](report.md#5-limitations-and-possible-improvements): a more stable threshold choice, more and
multilingual data (which could let sequence models help), calibration per data source, stronger link intelligence,
and continuous monitoring with retraining.
