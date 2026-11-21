"""Asisten untuk uji chatbot: MEMAKAI kode layanan yang sama (app/genai.py + app/classifier.py), jadi yang diuji
adalah prompt, knowledge.json, dan model classifier yang benar-benar dipakai FastAPI. API key dari env GEMINI_API_KEY."""
import os, sys, json, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app import genai, config            # noqa: E402
from app.classifier import Classifier    # noqa: E402
PRIMARY_MODEL, BACKUP_MODEL = config.GEMINI_PRIMARY, config.GEMINI_BACKUP
_clf = None

def classify(q):
    global _clf
    _clf = _clf or Classifier()
    return _clf.predict(" ".join(q.split()))

def build_system_prompt(label, yakin=True): return genai.build_system_prompt(label, yakin, genai.load_kb())

def ask(question, label, yakin=True, model=PRIMARY_MODEL, timeout=60):
    key = config.gemini_key()
    if not key: raise RuntimeError("GEMINI_API_KEY belum di-set")
    body = {"systemInstruction": {"parts": [{"text": build_system_prompt(label, yakin)}]},
            "contents": [{"role": "user", "parts": [{"text": question}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512}}
    req = urllib.request.Request(genai.API.format(model=model), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as resp: data = json.load(resp)
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()
