"""Asisten generatif The Nail Palace: rakit system prompt dari knowledge.json lalu panggil Gemini API (REST, tanpa library tambahan).
API key dibaca dari env GEMINI_API_KEY atau GOOGLE_API_KEY. Jangan taruh key di kode."""
import json, os, urllib.request, urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIMARY_MODEL, BACKUP_MODEL = "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"
LABEL_DESC = {
    "layanan_harga": "pertanyaan tentang jenis layanan, detail, durasi, atau harga",
    "info_operasional": "pertanyaan tentang jam/hari buka, lokasi, kontak, atau info operasional studio",
    "cara_reservasi": "pertanyaan tentang cara memesan, ketersediaan slot, atau status booking",
    "pembatalan_keluhan": "pertanyaan tentang pembatalan, perubahan booking, atau keluhan",
}

def rupiah(n): return "Rp" + f"{n:,}".replace(",", ".")

def build_system_prompt(label, kb_path=HERE / "knowledge.json", tpl_path=HERE / "system_prompt_template.md"):
    kb = json.loads(Path(kb_path).read_text(encoding="utf-8")); r = kb["cara_reservasi"]
    layanan = "\n".join(f"- {s['nama']} ({s['kategori']}): {rupiah(s['harga'])}, durasi {s['durasi']}" for s in kb["layanan"])
    cara = "\n".join([f"- {r['tanpa_akun']}", f"- Pesan di halaman {r['halaman_pesan']} dalam {len(r['langkah'])} langkah:"]
                     + [f"  {i+1}. {s}" for i, s in enumerate(r["langkah"])] + [f"- {r['setelah_kirim']}", f"- {r['cek_status']}", f"- {r['slot']}"])
    return Path(tpl_path).read_text(encoding="utf-8").format(
        deskripsi_studio=kb["studio"]["deskripsi"][0].lower() + kb["studio"]["deskripsi"][1:], label=label, label_desc=LABEL_DESC.get(label, "tidak diketahui"),
        versi=kb["versi"], daftar_layanan=layanan, cara_reservasi=cara,
        aturan_booking="\n".join(f"- {a}" for a in kb["aturan_booking"]),
        tidak_diketahui="\n".join(f"- {t}" for t in kb["tidak_diketahui"]), arahan_kontak=kb["arahan_kontak"])

def ask(question, label, model=PRIMARY_MODEL, temperature=0.2, timeout=60):
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key: raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY belum di-set")
    body = {"systemInstruction": {"parts": [{"text": build_system_prompt(label)}]},
            "contents": [{"role": "user", "parts": [{"text": question}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 512}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.load(resp)
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()

def ask_with_fallback(question, label):
    try: return ask(question, label, PRIMARY_MODEL), PRIMARY_MODEL
    except urllib.error.HTTPError as e:
        if e.code in (404, 429, 500, 503): return ask(question, label, BACKUP_MODEL), BACKUP_MODEL
        raise

if __name__ == "__main__":
    import sys
    print(build_system_prompt(sys.argv[1] if len(sys.argv) > 1 else "layanan_harga"))
