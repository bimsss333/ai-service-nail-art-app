"""Jawaban generatif Gemini (REST) yang dibatasi knowledge.json, dengan timeout, retry+backoff,
model cadangan, lalu jawaban statis per label kalau semua gagal."""
import json, re, time, logging
import httpx
from . import config

log = logging.getLogger("ae.genai")
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRYABLE = {408, 429, 500, 502, 503, 504}
# Routing 11 label (label.json tim DS v1.0): deskripsi untuk prompt + aksi asisten.
# aksi: jawab_dari_fakta | arahkan_ke_reservasi | arahkan_ke_cek_reservasi | balas_sapaan | arahkan_ke_kontak
ROUTES = {
    "harga": ("pertanyaan tentang harga, tarif, atau kisaran biaya layanan", "jawab_dari_fakta"),
    "layanan_desain": ("pertanyaan tentang jenis layanan atau desain yang tersedia", "jawab_dari_fakta"),
    "durasi": ("pertanyaan tentang lama pengerjaan layanan", "jawab_dari_fakta"),
    "jam_operasional": ("pertanyaan tentang jam buka/tutup, jam istirahat, hari buka, atau hari libur", "jawab_dari_fakta"),
    "lokasi": ("pertanyaan tentang alamat, arah, parkir, atau patokan studio", "jawab_dari_fakta"),
    "cara_reservasi": ("pertanyaan tentang cara memesan, ketersediaan slot, atau ingin membuat reservasi", "arahkan_ke_reservasi"),
    "cek_ubah_batal": ("pertanyaan tentang cek status, pembatalan, atau jadwal ulang reservasi yang sudah ada", "arahkan_ke_cek_reservasi"),
    "pembayaran": ("pertanyaan tentang metode bayar, DP, bukti transfer, atau waktu pembayaran", "jawab_dari_fakta"),
    "perawatan_keluhan": ("keluhan hasil layanan, garansi, atau cara merawat kuku", "jawab_dari_fakta"),
    "sapaan": ("sapaan, ucapan terima kasih, atau salam penutup tanpa pertanyaan", "balas_sapaan"),
    "di_luar_konteks": ("pertanyaan di luar topik studio atau terlalu ambigu", "arahkan_ke_kontak"),
}
LABEL_DESC = {k: v[0] for k, v in ROUTES.items()}
AKSI_INSTRUKSI = {
    "jawab_dari_fakta": "jawab langsung dari bagian FAKTA yang relevan; kalau infonya tidak ada, bilang jujur lalu arahkan ke kontak studio.",
    "arahkan_ke_reservasi": "jelaskan singkat cara pesan dan arahkan ke halaman /pesan. Jangan memastikan slot/tanggal tertentu kosong atau penuh.",
    "arahkan_ke_cek_reservasi": "arahkan ke halaman /cek-booking (kode booking + nomor HP) untuk cek status atau membatalkan; jadwal ulang lewat studio. Kamu tidak bisa melihat atau mengubah booking.",
    "balas_sapaan": "balas sapaan dengan singkat dan ramah, lalu tawarkan bantuan (layanan, harga, jam buka, reservasi).",
    "arahkan_ke_kontak": "kalau di luar topik studio, tolak dengan sopan dan tawarkan bantuan seputar studio; kalau ambigu, minta pelanggan memperjelas atau arahkan ke kontak studio.",
}
def aksi(label): return ROUTES.get(label, ROUTES["di_luar_konteks"])[1]

def rupiah(n): return "Rp" + f"{n:,}".replace(",", ".")

def load_kb(): return json.loads(config.KNOWLEDGE_PATH.read_text(encoding="utf-8"))

def _durasi(m): h, mm = divmod(m, 60); return " ".join(x for x in [f"{h} jam" if h else "", f"{mm} menit" if mm else ""] if x)

def _catatan(c):  # kisaran harga sudah ditulis di kolom harga, jangan diulang
    return re.sub(r"Kisaran harga .*?\.(\s+|$)", "", c or "").strip()

def _harga(s): return f"{rupiah(s['harga_mulai'])}–{rupiah(s['harga_maks'])}" if s.get("harga_maks") else rupiah(s["harga_mulai"])

def fakta_sections(kb):
    """Rakit teks FAKTA dari knowledge.json (dipakai prompt dan tes)."""
    L = "\n".join(f"- {s['nama']} ({s['kategori']}): {_harga(s)}, durasi {_durasi(s['durasi_menit'])}" + (f". {_catatan(s['catatan'])}" if _catatan(s.get("catatan")) else "")
                  for s in kb["layanan"])
    jam = "\n".join(f"- {j['hari']}: " + (f"{j['jam']}, istirahat {j['istirahat']}" if j["buka"] else "tutup") for j in kb["jam_operasional"])
    libur = "\n".join(f"- {h['tanggal']}: {h['keterangan']}" for h in kb["hari_libur_tercatat"])
    r, c, p, k, st, ld = kb["cara_reservasi"], kb["cek_ubah_batal"], kb["pembayaran"], kb["perawatan_keluhan"], kb["studio"], kb["layanan_desain"]
    cara = "\n".join([f"- {r['tanpa_akun']}", f"- Pesan di halaman {r['halaman_pesan']}:"] + [f"  {i+1}. {s}" for i, s in enumerate(r["langkah"])]
                     + [f"- {r['kode_booking']}"] + [f"- {a}" for a in r["aturan"]])
    return dict(daftar_layanan=L, jam_operasional=jam, hari_libur=libur,
                layanan_desain=f"- Kategori: {', '.join(ld['kategori'])}\n- {ld['desain_custom']}\n- {ld['cakupan']}",
                lokasi=f"- {st['kota']} (alamat lengkap belum ada)\n- Instagram: {st['instagram']}",
                cara_reservasi=cara, cek_ubah_batal="\n".join([f"- Halaman cek: {c['halaman_cek']}"] + [f"- {a}" for a in c["aturan"]]),
                pembayaran=f"- Metode: {', '.join(p['metode'])}\n- {p['waktu']}\n- {p['dp']}",
                perawatan="\n".join([f"- {k['keluhan']}"] + [f"- {x}" for x in k["catatan_katalog"]]))

def build_system_prompt(label, yakin, kb):
    d = kb["studio"]["deskripsi"]
    prompt = config.PROMPT_PATH.read_text(encoding="utf-8").format(
        deskripsi_studio=d[0].lower() + d[1:], label=label, label_desc=LABEL_DESC.get(label, "tidak diketahui"),
        aksi_instruksi=AKSI_INSTRUKSI[aksi(label)], versi=kb["versi"], **fakta_sections(kb),
        tidak_diketahui="\n".join(f"- {t}" for t in kb["tidak_diketahui"]), arahan_kontak=kb["arahan_kontak"])
    if not yakin:
        prompt += ("\n\n## Catatan keyakinan\nClassifier TIDAK YAKIN dengan label di atas (keyakinan rendah). "
                   "Abaikan label itu dan jawab berdasarkan isi pertanyaan. Kalau maksud pelanggan kurang jelas, "
                   "minta pelanggan memperjelas pertanyaannya dengan singkat.")
    return prompt

def fallback_answer(label, kb, yakin=True):
    """Jawaban statis per label kalau Gemini gagal / key kosong. Semua isi dari knowledge.json."""
    r, c, st = kb["cara_reservasi"], kb["cek_ubah_batal"], kb["studio"]
    ig = f"Instagram {st['instagram']}"
    kontak = f"Untuk info lain, silakan hubungi studio lewat {ig}."
    pre = "Maaf, asisten sedang sibuk."
    if not yakin:  # label ragu: jangan menebak topik, beri arahan umum
        return (f"{pre} Untuk memesan, buka halaman {r['halaman_pesan']}; untuk cek status atau membatalkan booking, buka halaman "
                f"{c['halaman_cek']}. Untuk pertanyaan lain, silakan hubungi studio lewat {ig}.")
    a = aksi(label)
    if label in ("harga", "layanan_desain", "durasi"):
        top = ", ".join(f"{s['nama']} mulai {rupiah(s['harga_mulai'])} ({_durasi(s['durasi_menit'])})" for s in kb["layanan"][:3])
        return f"{pre} Daftar layanan, harga, dan durasi lengkap bisa dilihat di halaman {r['halaman_pesan']} (contoh: {top}). {kontak}"
    if label == "jam_operasional":
        buka = [j for j in kb["jam_operasional"] if j["buka"]]; tutup = [j["hari"] for j in kb["jam_operasional"] if not j["buka"]]
        return (f"{pre} {st['nama']} buka {buka[0]['hari']}–{buka[-1]['hari']} pukul {buka[0]['jam']} WIB"
                + (f", tutup hari {', '.join(tutup)}" if tutup else "") + f". Slot yang masih kosong bisa dicek di halaman {r['halaman_pesan']}.")
    if label == "lokasi":
        return f"{pre} {st['nama']} berada di {st['kota']}. Untuk alamat lengkap dan petunjuk arah, silakan hubungi studio lewat {ig}."
    if label == "pembayaran":
        p = kb["pembayaran"]
        return f"{pre} Pembayaran bisa {', '.join(p['metode'])}. {p['waktu']}. {p['dp']}. {kontak}"
    if label == "perawatan_keluhan":
        return f"{pre} Untuk keluhan hasil layanan, silakan hubungi studio lewat {ig} supaya bisa dibantu langsung."
    if a == "arahkan_ke_reservasi":
        return (f"{pre} Kamu tetap bisa memesan di halaman {r['halaman_pesan']} tanpa akun (pilih layanan, pilih tanggal dan jam, "
                f"isi nama dan nomor HP, lalu kirim). Status booking bisa dicek di halaman {c['halaman_cek']}.")
    if a == "arahkan_ke_cek_reservasi":
        return (f"{pre} Cek status atau batalkan booking di halaman {c['halaman_cek']} dengan kode booking dan nomor HP. "
                f"Untuk jadwal ulang, silakan hubungi studio lewat {ig}.")
    if a == "balas_sapaan":
        return f"Halo, terima kasih sudah menghubungi {st['nama']}! Ada yang bisa dibantu soal layanan, harga, jam buka, atau reservasi?"
    return f"{pre} Untuk pertanyaan ini, silakan hubungi studio lewat {ig}. Untuk memesan, buka halaman {r['halaman_pesan']}."

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
