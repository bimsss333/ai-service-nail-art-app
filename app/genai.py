"""Jawaban generatif Gemini (REST) yang dibatasi knowledge.json, dengan timeout, retry+backoff,
model cadangan, lalu jawaban statis per label kalau semua gagal."""
import json, time, logging
import httpx
from . import config

log = logging.getLogger("ae.genai")
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE = {408, 429, 500, 502, 503, 504}
LABEL_DESC = {
    "layanan_harga": "pertanyaan tentang jenis layanan, detail, durasi, atau harga",
    "info_operasional": "pertanyaan tentang jam/hari buka, lokasi, kontak, atau info operasional studio",
    "cara_reservasi": "pertanyaan tentang cara memesan, ketersediaan slot, atau status booking",
    "pembatalan_keluhan": "pertanyaan tentang pembatalan, perubahan booking, atau keluhan",
}

def rupiah(n): return "Rp" + f"{n:,}".replace(",", ".")

def load_kb(): return json.loads(config.KNOWLEDGE_PATH.read_text(encoding="utf-8"))

def build_system_prompt(label, yakin, kb):
    r = kb["cara_reservasi"]
    layanan = "\n".join(f"- {s['nama']} ({s['kategori']}): {rupiah(s['harga'])}, durasi {s['durasi']}" for s in kb["layanan"])
    cara = "\n".join([f"- {r['tanpa_akun']}", f"- Pesan di halaman {r['halaman_pesan']} dalam {len(r['langkah'])} langkah:"]
                     + [f"  {i+1}. {s}" for i, s in enumerate(r["langkah"])] + [f"- {r['setelah_kirim']}", f"- {r['cek_status']}", f"- {r['slot']}"])
    d = kb["studio"]["deskripsi"]
    prompt = config.PROMPT_PATH.read_text(encoding="utf-8").format(
        deskripsi_studio=d[0].lower() + d[1:], label=label, label_desc=LABEL_DESC.get(label, "tidak diketahui"),
        versi=kb["versi"], daftar_layanan=layanan, cara_reservasi=cara,
        aturan_booking="\n".join(f"- {a}" for a in kb["aturan_booking"]),
        tidak_diketahui="\n".join(f"- {t}" for t in kb["tidak_diketahui"]), arahan_kontak=kb["arahan_kontak"])
    if not yakin:
        prompt += ("\n\n## Catatan keyakinan\nClassifier TIDAK YAKIN dengan label di atas (keyakinan rendah). "
                   "Abaikan label itu dan jawab berdasarkan isi pertanyaan. Kalau maksud pelanggan kurang jelas, "
                   "minta pelanggan memperjelas pertanyaannya dengan singkat.")
    return prompt

def _cek_path(r):
    import re
    m = re.search(r"(/cek-[a-z-]+)", r.get("cek_status", ""))
    return m.group(1) if m else "/cek-booking"

KONTAK = "Untuk info yang belum tersedia, silakan hubungi studio The Nail Palace secara langsung."

def fallback_answer(label, kb, yakin=True):
    r = kb["cara_reservasi"]; kontak = KONTAK
    if not yakin:  # label ragu: jangan menebak topik, beri arahan umum
        return (f"Maaf, asisten sedang sibuk. Untuk memesan, buka halaman {r['halaman_pesan']}; untuk cek status booking, "
                f"buka halaman {_cek_path(r)}. Untuk pertanyaan lain, silakan hubungi studio The Nail Palace secara langsung.")
    if label == "layanan_harga":
        top = ", ".join(f"{s['nama']} {rupiah(s['harga'])}" for s in kb["layanan"][:3])
        return (f"Maaf, asisten sedang sibuk. Daftar layanan dan harga lengkap bisa dilihat di halaman {r['halaman_pesan']} "
                f"(contoh: {top}). {kontak}")
    if label == "info_operasional":
        return (f"Maaf, asisten sedang sibuk. {kb['studio']['nama']} berada di Kota Jambi. "
                f"Untuk jam buka, alamat lengkap, dan kontak, silakan hubungi studio secara langsung.")
    if label == "cara_reservasi":
        return (f"Maaf, asisten sedang sibuk. Kamu tetap bisa memesan di halaman {r['halaman_pesan']} tanpa akun "
                f"(pilih layanan, pilih tanggal dan jam, isi nama dan nomor HP, lalu kirim). "
                f"Status booking bisa dicek di halaman {_cek_path(r)}.")
    if label == "pembatalan_keluhan":
        return ("Maaf, asisten sedang sibuk. Reservasi yang sudah dibatalkan tidak bisa diaktifkan kembali; "
                f"untuk pesan ulang, buat booking baru di halaman {r['halaman_pesan']}. "
                f"Untuk pembatalan atau keluhan, silakan hubungi studio secara langsung.")
    return f"Maaf, asisten sedang sibuk. Silakan pesan di halaman {r['halaman_pesan']} atau hubungi studio secara langsung."

class GeminiError(Exception):
    def __init__(self, kind, status=None): super().__init__(kind); self.kind, self.status = kind, status

def call_gemini(client, model, system_prompt, question, timeout):
    body = {"systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": question}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512}}
    try:
        resp = client.post(API.format(model=model), json=body, timeout=timeout,
                           headers={"x-goog-api-key": config.gemini_key()})
    except httpx.TimeoutException: raise GeminiError("timeout")
    except httpx.HTTPError: raise GeminiError("network")
    if resp.status_code != 200: raise GeminiError("http", resp.status_code)
    try:
        parts = resp.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, ValueError): raise GeminiError("bad_response")
    if not text: raise GeminiError("empty")
    return text

def answer(question, label, yakin, client=None, sleep=time.sleep, clock=time.monotonic):
    """Return dict(jawaban, sumber, model_dipakai, percobaan). Tidak pernah raise ke pemanggil."""
    kb = load_kb()
    if not config.gemini_key():
        return {"jawaban": fallback_answer(label, kb, yakin), "sumber": "fallback", "model_dipakai": None, "percobaan": []}
    prompt = build_system_prompt(label, yakin, kb)
    plan = [config.GEMINI_PRIMARY] * max(1, config.GEMINI_RETRIES) + [config.GEMINI_BACKUP]
    own = client is None
    client = client or httpx.Client()
    start, attempts = clock(), []
    try:
        i = 0
        while i < len(plan):
            model = plan[i]
            remaining = config.GEMINI_TOTAL_BUDGET - (clock() - start)
            if remaining <= 1: break
            try:
                text = call_gemini(client, model, prompt, question, min(config.GEMINI_TIMEOUT, remaining))
                attempts.append({"model": model, "hasil": "ok"})
                return {"jawaban": text, "sumber": "gemini", "model_dipakai": model, "percobaan": attempts}
            except GeminiError as e:
                attempts.append({"model": model, "hasil": e.kind if e.status is None else f"http_{e.status}"})
                log.warning("gemini gagal model=%s jenis=%s status=%s", model, e.kind, e.status)
                if e.kind == "http" and e.status not in RETRYABLE and e.status != 404:
                    break  # 400/401/403: kunci/permintaan salah, mengulang tidak membantu
                nxt = i + 1
                if e.kind == "http" and e.status == 404:
                    while nxt < len(plan) and plan[nxt] == model: nxt += 1  # model tidak ada: lompat ke cadangan
                elif nxt < len(plan) and plan[nxt] == model:
                    sleep(config.GEMINI_BACKOFF * (2 ** i))  # ulang model yang sama dengan backoff
                i = nxt
    finally:
        if own: client.close()
    return {"jawaban": fallback_answer(label, kb, yakin), "sumber": "fallback", "model_dipakai": None, "percobaan": attempts}
