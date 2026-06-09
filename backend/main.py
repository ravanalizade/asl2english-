import base64
import os
import time

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocess import preprocess_crop
from model import letter_model

load_dotenv()

CONFIDENCE_THRESHOLD = 0.6  # below this, emit NO letter (suppresses uncertain guesses)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

app = FastAPI(title="ASL Fingerspelling Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LetterRequest(BaseModel):
    image: str


class AssembleRequest(BaseModel):
    text: str


def _decode_image(data_url: str):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.get("/health")
def health():
    return {"status": "ok", "placeholder_model": letter_model.placeholder}


@app.post("/letter")
def letter(req: LetterRequest):
    img = _decode_image(req.image)
    if img is None:
        return {"letter": None, "confidence": 0.0, "error": "bad image"}

    processed = preprocess_crop(img)
    pred, conf = letter_model.predict(processed)

    if conf < CONFIDENCE_THRESHOLD:
        return {"letter": None, "confidence": conf}
    return {"letter": pred, "confidence": conf}


@app.post("/assemble")
def assemble(req: AssembleRequest):
    text = req.text.strip()
    if not text:
        return {"sentence": ""}

    if not GEMINI_API_KEY:
        return {
            "sentence": text.capitalize(),
            "note": "No GEMINI_API_KEY set; returned letters without AI correction.",
        }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    prompt = (
        "You are helping a sign-language speller. The user spelled these letters "
        "with no spaces or punctuation. Turn them into the most likely correct, "
        "natural English sentence. Reply with ONLY the sentence.\n\n"
        f"Letters: {text}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    last_err = None
    for attempt in range(4):
        try:
            r = requests.post(url, json=payload, timeout=20)
            r.raise_for_status()
            data = r.json()
            sentence = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {"sentence": sentence}
        except Exception as e:
            last_err = e
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (503, 429) and attempt < 3:
                time.sleep(1.5 * (attempt + 1))
                continue
            break
    return {"error": f"Sentence service failed: {last_err}"}