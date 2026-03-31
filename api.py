"""
api.py — FastAPI inference server for ED content classifier
POST /classify  {"text": "..."}
             -> {"verdict": "SAFE|WARN|HARMFUL", "score": float, "flagged_terms": [...]}
"""

import re
from contextlib import asynccontextmanager
from typing import List

import joblib
import numpy as np
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Startup: load model bundle
# ---------------------------------------------------------------------------
bundle: dict = {}
vader = SentimentIntensityAnalyzer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        bundle.update(joblib.load("model/detector.joblib"))
        print("Model loaded from model/detector.joblib")
    except FileNotFoundError:
        raise RuntimeError("Model not found. Run `python train.py` first.")
    yield


app = FastAPI(title="ED Content Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Feature helpers (must mirror train.py exactly)
# ---------------------------------------------------------------------------

def lexicon_count(text: str) -> int:
    text_lower = text.lower()
    count = 0
    for term in bundle["lexicon"]:
        if " " in term or "-" in term:
            count += 1 if term in text_lower else 0
        else:
            count += 1 if re.search(r"\b" + re.escape(term) + r"\b", text_lower) else 0
    return count


def flagged_terms(text: str) -> List[str]:
    text_lower = text.lower()
    found = []
    for term in bundle["lexicon"]:
        if " " in term or "-" in term:
            if term in text_lower:
                found.append(term)
        else:
            if re.search(r"\b" + re.escape(term) + r"\b", text_lower):
                found.append(term)
    return found


def build_features(text: str):
    tfidf = bundle["vectorizer"].transform([text])
    lex = np.array([[lexicon_count(text)]], dtype=np.float32)
    s = vader.polarity_scores(text)
    vdr = np.array([[s["neg"], s["neu"], s["pos"], s["compound"]]], dtype=np.float32)
    return sp.hstack([tfidf, sp.csr_matrix(lex), sp.csr_matrix(vdr)])


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    verdict: str
    score: float
    flagged_terms: List[str]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")
    if len(text) > 5000:
        raise HTTPException(status_code=422, detail="text exceeds 5000 character limit")

    features = build_features(text)
    proba = bundle["model"].predict_proba(features)[0]
    classes = bundle["model"].classes_

    idx = int(np.argmax(proba))
    verdict = classes[idx]
    score = float(proba[idx])

    return ClassifyResponse(
        verdict=verdict,
        score=round(score, 4),
        flagged_terms=flagged_terms(text),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(bundle)}
