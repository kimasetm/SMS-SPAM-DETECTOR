# Deploying SmartSpam to Streamlit Community Cloud

Streamlit Community Cloud hosts Streamlit apps for free, straight from a GitHub repository.

**Deployed at:** <https://kimasetm-sms-spam-detector-app-ee33vq.streamlit.app/> (from the `main` branch). The
post-deployment checklist below passed on November 9, 2026.

## What's already prepared

- **`app.py`** is in the repository root (the app's entry point).
- **The trained model is committed** in `models/` (weights, vocabulary, calibration temperature, threshold and drift
  baseline), so the cloud server doesn't need to download the datasets or train.
- **`requirements.txt`** installs the CPU-only PyTorch build, avoiding about 2 GB of unneeded GPU libraries. It
  resolves correctly with `uv`, the installer Streamlit Cloud uses (checked for Linux and Python 3.12).
- **`.streamlit/config.toml`** sets the theme and a 5 MB upload limit. It contains no secrets.
- **No secrets are needed.** The app uses no API keys, passwords or databases. The optional online link checks use
  the keyless OpenPhish feed and RDAP. `.streamlit/secrets.toml` is git-ignored in case a key is ever added.
- **Online link checks** need outbound internet access, which Streamlit Cloud provides. They're off by default for
  each visitor, and if a service is unreachable the app says so instead of failing.

## Steps

1. Make sure the code you want to deploy is merged into `main` on GitHub.
2. Go to <https://share.streamlit.io> and sign in with the GitHub account that owns the repository.
3. Click **Create app**, then choose **Deploy a public app from GitHub**.
4. Fill in:
   - **Repository:** `kimasetm/SMS-SPAM-DETECTOR`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** choose a name, e.g. `smartspam-sms`
5. Open **Advanced settings** and select **Python 3.12**. Leave **Secrets** empty.
6. Click **Deploy**. The first build takes a few minutes while PyTorch installs.
7. Copy the app URL into the README's demo link.

## Checklist after deploying

- [ ] The app opens and shows the three tabs.
- [ ] **Prize scam** → **Predict** shows SPAM and the premium-rate sender warning.
- [ ] **Delivery scam** → **Predict** shows SPAM and a high-risk link.
- [ ] Batch CSV: **Download sample CSV**, upload it, and see "right on 20 of 20".
- [ ] Model Dashboard shows both test sets and the charts.
- [ ] Optional: tick **Online link checks** and predict the Delivery scam; it should finish within a few seconds.

## Updating and troubleshooting

- Every push to `main` redeploys the app automatically.
- If the build fails, open **Manage app → Logs** in the bottom-right corner of the app page to see the error.
- Free apps go to sleep after a period without visitors. The first visit after that takes about a minute
  while the app wakes up, so open it before a live demo.
