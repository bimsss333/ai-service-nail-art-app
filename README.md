# Layanan AE — The Nail Palace (A26-CS032) v0.1

Layanan FastAPI untuk bagian AI Engineer:
1. **Classifier pertanyaan** buatan sendiri (TF-IDF + Logistic Regression, scikit-learn 1.9.1), 4 label:
   `layanan_harga`, `info_operasional`, `cara_reservasi`, `pembatalan_keluhan`. Macro F1 0,901 (split grouped, lihat `../eval/`).
2. **Asisten generatif** (Gemini `gemini-3.1-flash-lite`, cadangan `gemini-2.5-flash-lite`) yang hanya boleh menjawab
   dari `app/data/knowledge.json`.

Layanan ini **dipanggil oleh backend Node (FS)**, bukan langsung oleh frontend. **Reservasi tidak bergantung pada layanan ini**:
kalau layanan AE mati, halaman `/pesan` dan `/cek-booking` tetap harus jalan.

Kontrak API untuk FS: `../kontrak/kontrak-api-ae-v0.1.md`.

## Struktur
```
app/main.py          endpoint, validasi, format error, auth header
app/classifier.py    muat model + predict_proba
app/genai.py         prompt dari knowledge.json, panggilan Gemini (timeout, retry, cadangan, fallback statis)
app/config.py        semua pengaturan dari env var
app/data/            knowledge.json (satu-satunya sumber fakta), system_prompt_template.md
model/               best_model_LogReg.joblib
tests/               pytest (Gemini di-mock)
smoke_live.sh        smoke test live dengan curl
threshold_analysis.py  analisis ambang keyakinan (butuh dataset di ../data/)
test_output/         hasil tes asli
```

## Menjalankan di lokal
```bash
cd ae-service
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export GEMINI_API_KEY=...            # dari Google AI Studio; jangan ditulis ke file yang di-commit
export AE_INTERNAL_KEY=rahasia-dev   # opsional
uvicorn app.main:app --reload --port 8000
```
Buka `http://localhost:8000/health` atau dokumentasi otomatis di `http://localhost:8000/docs`.

Tes:
```bash
pytest -v                                   # tanpa internet, Gemini di-mock
BASE=http://127.0.0.1:8000 ./smoke_live.sh  # butuh server jalan + GEMINI_API_KEY asli
```

## Environment variable
| Nama | Wajib | Default | Fungsi |
|---|---|---|---|
| `GEMINI_API_KEY` | untuk jawaban generatif | – | Kalau kosong, `/chat` selalu memakai jawaban fallback statis |
| `AE_INTERNAL_KEY` | disarankan di produksi | kosong | Kalau diisi, `/api/v1/*` wajib header `X-Internal-Key` yang sama. Kosong = mode dev |
| `GEMINI_MODEL` / `GEMINI_BACKUP_MODEL` | tidak | `gemini-3.1-flash-lite` / `gemini-2.5-flash-lite` | Model utama dan cadangan |
| `GEMINI_TIMEOUT` | tidak | `10` | Detik per percobaan |
| `GEMINI_RETRIES` | tidak | `2` | Jumlah percobaan ke model utama |
| `GEMINI_BACKOFF` | tidak | `1.0` | Jeda awal sebelum mengulang (dikali 2) |
| `GEMINI_TOTAL_BUDGET` | tidak | `25` | Batas total detik semua percobaan Gemini |
| `AE_CONF_THRESHOLD` | tidak | `0.55` | Di bawah ini label dianggap ragu (`yakin: false`) |
| `AE_MAX_LEN` | tidak | `500` | Panjang maksimal pertanyaan (karakter) |
| `AE_CORS_ORIGINS` | tidak | kosong (CORS mati) | Isi hanya kalau disepakati frontend memanggil langsung |

Lihat `.env.example`. File `.env` tidak ikut di-commit.

## Alur `/api/v1/chat`
1. Validasi (tidak kosong, maksimal 500 karakter).
2. Classifier memberi label + confidence. Kalau confidence < 0,55, label tetap dikirim tapi `yakin: false`,
   dan prompt Gemini diberi tahu bahwa label tidak yakin (jawab dari isi pertanyaan, minta diperjelas kalau perlu).
3. Gemini dipanggil: model utama maksimal 2 kali (timeout 10 detik, jeda 1 detik), lalu model cadangan 1 kali,
   semua dibatasi total 25 detik. HTTP 400/401/403 tidak diulang; 404 (model tidak ada) langsung ke cadangan.
4. Kalau semua gagal (atau key tidak ada), dikirim **jawaban statis per label** dari `knowledge.json` dengan `sumber: "fallback"`.
   Endpoint tetap HTTP 200.

## Ambang keyakinan 0,55 — kenapa
Dihitung dari probabilitas out-of-fold (split grouped 5 fold, hiperparameter model final), lihat
`test_output/threshold_analysis.txt`:
- Median confidence prediksi benar 0,864, prediksi salah 0,544.
- Ambang 0,55 menandai 12,3% pertanyaan sebagai ragu dan menangkap 51,9% prediksi salah;
  akurasi pada yang di atas ambang naik dari 0,910 ke 0,951.
- Ambang 0,60 menangkap 61,1% kesalahan tapi menandai 16,5% pertanyaan; 0,50 hanya menangkap 35,2%.

Ambang ini bisa diubah lewat `AE_CONF_THRESHOLD` tanpa mengubah kode.

## Privasi
- Isi pertanyaan **tidak** ditulis ke log; yang dicatat hanya label, confidence, sumber, model, durasi.
- Asisten tidak meminta nomor HP atau data pribadi dan tidak bisa melihat data booking.
- API key dibaca dari env saja, tidak pernah dikirim balik (`/health` hanya `gemini_configured: true/false`).

## Deploy ke Render (Free)
1. Push folder `ae-service` ke repo GitHub (tanpa `.env` dan `.venv`).
2. Render → New → Web Service → pilih repo, **Root Directory** `ae-service` (kalau di subfolder).
3. Runtime **Python**, lalu:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Health check path: `/health`
4. Environment: `PYTHON_VERSION=3.13.5`, `GEMINI_API_KEY`, `AE_INTERNAL_KEY` (isi juga di env backend Node dengan nilai sama).
   Alternatif: pakai `render.yaml` (Blueprint) yang sudah disiapkan.
5. **Cold start:** instance Free tidur setelah sekitar 15 menit tanpa trafik; request pertama setelah tidur bisa
   butuh puluhan detik sampai sekitar 1 menit. Backend harus tetap memakai timeout (lihat kontrak) dan menampilkan pesan
   ramah kalau habis waktu. Opsional: backend memanggil `GET /health` saat halaman chatbot dibuka untuk "membangunkan" layanan.

Catatan: konfigurasi Render di atas belum dicoba deploy sungguhan; yang sudah diuji baru jalan lokal.

## Hasil uji (14 Nov 2026 WIB)
- `pytest`: 32 lulus (`test_output/pytest_output.txt`).
- Smoke live dengan Gemini asli: `test_output/live_smoke.txt`, `live_low_confidence.txt`, `live_fallback.txt`.
