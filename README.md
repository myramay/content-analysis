# ED / Skinnytok Content Detection — Portfolio Project

A three-part system that detects eating-disorder-promoting content on the web: an ML training pipeline, a REST inference API, and a Chrome browser extension.

---

## Project Structure

```
content-analysis/
├── train.py              # ML pipeline — generates model
├── api.py                # FastAPI inference server
├── requirements.txt
├── model/                # Created by train.py (gitignore this)
│   └── detector.joblib
└── extension/            # Chrome extension (Manifest V3)
    ├── manifest.json
    ├── content.js
    ├── popup.html
    └── popup.js
```

---

## 1. Train the Model

```bash
pip install -r requirements.txt
python train.py
```

This generates `model/detector.joblib` and prints a classification report + confusion matrix on an 80/20 hold-out split.

**What the pipeline does:**
- 200 synthetic labeled examples across three classes: `SAFE`, `WARN`, `HARMFUL`
- TF-IDF vectorizer with `ngram_range=(1,2)`, `max_features=5000`
- Domain lexicon feature: counts hits of 30+ ED-associated terms (e.g., *thinspo*, *restrict*, *sw/gw/cw*, *purge*) with word-boundary matching
- VADER sentiment (neg/neu/pos/compound) as additional features
- Logistic Regression with `class_weight="balanced"` for probability outputs

---

## 2. Run the API

```bash
uvicorn api:app --reload --port 8000
```

**Endpoint: `POST /classify`**

Request:
```json
{ "text": "thinspo goals, sw 140 gw 105" }
```

Response:
```json
{
  "verdict": "HARMFUL",
  "score": 0.9312,
  "flagged_terms": ["thinspo", "sw", "gw"]
}
```

- `verdict` — `SAFE`, `WARN`, or `HARMFUL`
- `score` — confidence (0–1) for the predicted class
- `flagged_terms` — lexicon matches found in the input

Health check: `GET http://localhost:8000/health`

---

## 3. Load the Chrome Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked** and select the `extension/` folder

**How it works:**
- Scans `<p>`, `<li>`, `<blockquote>`, and other text elements every 3 seconds
- Uses `MutationObserver` to catch dynamically loaded content (infinite scroll, etc.)
- Sends text chunks to `http://localhost:8000/classify`
- **WARN** → applies a soft yellow blur with a click-to-reveal badge
- **HARMFUL** → replaces the element with a red intervention card and a "Show anyway" toggle
- The popup icon shows live counts of flagged items on the current page

> **Note:** The API must be running on `localhost:8000` for the extension to work.

---

## Architecture

```
Browser extension (content.js)
        │  POST /classify
        ▼
FastAPI (api.py, port 8000)
        │  load features
        ▼
TF-IDF + Lexicon + VADER → LogisticRegression (model/detector.joblib)
        │
        ▼
{ verdict, score, flagged_terms }
```

---

## Limitations

- The classifier is trained on **synthetic data** — performance on real-world content will vary. Retraining on real labeled data is strongly recommended before any production use.
- Short or context-dependent text (e.g., single words) may produce unreliable verdicts.
- The browser extension processes a maximum of 5 elements per 3-second cycle to avoid overwhelming the local API.
- This tool is intended for **content moderation research and education**, not as a replacement for professional clinical assessment.
