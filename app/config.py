"""Konfigurasi dari environment variable. Jangan menaruh secret di kode."""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

def _f(name, default): return float(os.environ.get(name, default))
def _i(name, default): return int(os.environ.get(name, default))

MODEL_PATH = Path(os.environ.get("AE_MODEL_PATH", BASE / "model" / "classifier_11label_logreg.joblib"))
MODEL_VERSION = os.environ.get("AE_MODEL_VERSION", "logreg-tfidf-11label-2026-11-21")
KNOWLEDGE_PATH = Path(os.environ.get("AE_KNOWLEDGE_PATH", BASE / "app" / "data" / "knowledge.json"))
PROMPT_PATH = Path(os.environ.get("AE_PROMPT_PATH", BASE / "app" / "data" / "system_prompt_template.md"))

MAX_LEN = _i("AE_MAX_LEN", 500)
# Ambang 0,50: dipilih dari probabilitas set VALIDASI dataset DS (lihat docs/evaluation/threshold_val.json, selection.json)
CONF_THRESHOLD = _f("AE_CONF_THRESHOLD", 0.50)

GEMINI_PRIMARY = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
GEMINI_BACKUP = os.environ.get("GEMINI_BACKUP_MODEL", "gemini-2.5-flash-lite")
GEMINI_TIMEOUT = _f("GEMINI_TIMEOUT", 10)          # detik per percobaan
GEMINI_RETRIES = _i("GEMINI_RETRIES", 2)           # percobaan ke model utama
GEMINI_BACKOFF = _f("GEMINI_BACKOFF", 1.0)         # detik, dikali 2 tiap ulang
GEMINI_TOTAL_BUDGET = _f("GEMINI_TOTAL_BUDGET", 25) # batas total semua percobaan

def gemini_key(): return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
def internal_key(): return os.environ.get("AE_INTERNAL_KEY", "")
def cors_origins():
    v = os.environ.get("AE_CORS_ORIGINS", "").strip()
    return [o.strip() for o in v.split(",") if o.strip()]
