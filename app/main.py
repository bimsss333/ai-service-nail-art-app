"""Layanan AE The Nail Palace: classifier pertanyaan + asisten generatif. Dipanggil backend Node (FS),
bukan frontend. Reservasi TIDAK bergantung pada layanan ini."""
import hmac, logging, time
from fastapi import FastAPI, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from . import config, genai
from .classifier import Classifier

# Log tanpa isi pertanyaan (bisa berisi data pribadi): hanya label, confidence, sumber, durasi.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("ae")

app = FastAPI(title="The Nail Palace AE Service", version="0.1.0", docs_url="/docs", redoc_url=None)
if config.cors_origins():
    app.add_middleware(CORSMiddleware, allow_origins=config.cors_origins(), allow_methods=["GET", "POST"], allow_headers=["*"])

STATE = {"clf": None, "error": None}
try:
    STATE["clf"] = Classifier()
except Exception as e:  # service tetap hidup, /health melaporkan masalah
    STATE["error"] = type(e).__name__

def err(status, code, message):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})

class ApiError(Exception):
    def __init__(self, status, code, message): self.status, self.code, self.message = status, code, message

@app.exception_handler(ApiError)
async def _api_err(_, e: ApiError): return err(e.status, e.code, e.message)

@app.exception_handler(RequestValidationError)
async def _val_err(_, e):
    return err(400, "VALIDATION_ERROR", "Body harus JSON dengan field 'pertanyaan' berupa teks.")

@app.exception_handler(404)
async def _nf(_, e): return err(404, "NOT_FOUND", "Endpoint tidak ditemukan.")

@app.exception_handler(405)
async def _na(_, e): return err(405, "METHOD_NOT_ALLOWED", "Metode HTTP tidak diizinkan untuk endpoint ini.")

@app.exception_handler(Exception)
async def _any(_, e):
    log.error("error tak terduga: %s", type(e).__name__)
    return err(500, "INTERNAL_ERROR", "Terjadi kesalahan di layanan AE.")

def require_internal_key(request: Request):
    expected = config.internal_key()
    if not expected: return  # mode dev: tanpa kunci
    got = request.headers.get("X-Internal-Key", "")
    if not hmac.compare_digest(got.encode(), expected.encode()):
        raise ApiError(401, "UNAUTHORIZED", "Header X-Internal-Key tidak ada atau salah.")

class Pertanyaan(BaseModel):
    pertanyaan: str

def clean(p: Pertanyaan) -> str:
    text = " ".join(p.pertanyaan.split())
    if not text: raise ApiError(400, "VALIDATION_ERROR", "Pertanyaan tidak boleh kosong.")
    if len(text) > config.MAX_LEN:
        raise ApiError(400, "VALIDATION_ERROR", f"Pertanyaan terlalu panjang (maksimal {config.MAX_LEN} karakter).")
    return text

def get_clf():
    if STATE["clf"] is None: raise ApiError(503, "MODEL_UNAVAILABLE", "Model classifier belum siap.")
    return STATE["clf"]

@app.get("/health")
def health():
    ok = STATE["clf"] is not None
    return JSONResponse(status_code=200 if ok else 503, content={
        "status": "ok" if ok else "error", "model_loaded": ok, "model_version": config.MODEL_VERSION,
        "labels": STATE["clf"].labels if ok else [], "conf_threshold": config.CONF_THRESHOLD,
        "gemini_configured": bool(config.gemini_key()), "gemini_model": config.GEMINI_PRIMARY,
        "gemini_backup_model": config.GEMINI_BACKUP, "auth_required": bool(config.internal_key()),
        **({"model_error": STATE["error"]} if not ok else {})})

@app.post("/api/v1/classify", dependencies=[Depends(require_internal_key)])
def classify(body: Pertanyaan):
    text = clean(body); r = get_clf().predict(text)
    log.info("classify label=%s conf=%.3f", r["label"], r["confidence"])
    return {"label": r["label"], "confidence": r["confidence"], "yakin": r["yakin"],
            "ambang": config.CONF_THRESHOLD, "probabilities": r["probabilities"], "model_version": config.MODEL_VERSION}

@app.post("/api/v1/chat", dependencies=[Depends(require_internal_key)])
async def chat(body: Pertanyaan):
    text = clean(body); r = get_clf().predict(text)
    t0 = time.monotonic()
    g = await run_in_threadpool(genai.answer, text, r["label"], r["yakin"])
    ms = int((time.monotonic() - t0) * 1000)
    log.info("chat label=%s conf=%.3f sumber=%s model=%s ms=%d percobaan=%d", r["label"], r["confidence"], g["sumber"], g["model_dipakai"], ms, len(g["percobaan"]))
    return {"jawaban": g["jawaban"], "label": r["label"], "confidence": r["confidence"], "yakin": r["yakin"],
            "sumber": g["sumber"], "model_dipakai": g["model_dipakai"], "percobaan": len(g["percobaan"]), "durasi_ms": ms}
