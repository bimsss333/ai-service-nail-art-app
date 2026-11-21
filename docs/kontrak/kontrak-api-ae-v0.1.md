# Kontrak API Layanan AE — The Nail Palace (A26-CS032)
Versi: 0.1 — **DRAF, perlu disepakati bersama Arief (FS) / Thoha (FS) / Bima (AE)**  
Tanggal: 14 Nov 2026 WIB  
Pelengkap dari: `kontrak-api-web-inti-v0.1.md` (bagian 6: "FastAPI AE — kontrak terpisah")  
Kode: `ae-service/` (FastAPI). Semua contoh JSON di bawah diambil dari **uji live 14 Nov 2026 pukul 14.18–14.20 WIB**
(`ae-service/test_output/`), bukan karangan.

---

## 1. Posisi layanan AE dalam sistem

```
Browser (React) ──► Backend Node (FS) ──► Layanan AE (FastAPI) ──► Gemini API
                        │
                        └─► MySQL (reservasi)  ← jalur utama, TIDAK lewat AE
```

- **Frontend tidak memanggil FastAPI langsung.** Frontend memanggil endpoint chatbot di backend Node
  (usulan: `POST /api/v1/chatbot` di backend), lalu backend meneruskan ke layanan AE.
- **Reservasi tidak boleh bergantung pada layanan AE.** Kalau AE mati/lambat, `/pesan`, `/cek-booking`,
  dan dasbor admin tetap jalan normal. Chatbot hanya fitur informasi.
- Base URL AE (usulan): `https://<ae-host>.onrender.com` dengan endpoint di bawah `/api/v1`.
- `Content-Type: application/json`, encoding UTF-8.

## 2. Autentikasi antar-layanan

| Header | Wajib | Keterangan |
|---|---|---|
| `X-Internal-Key` | Ya, kalau env `AE_INTERNAL_KEY` di AE diisi | Rahasia bersama. Simpan di env backend Node (mis. `AE_INTERNAL_KEY`), **jangan** di kode frontend |

Kalau env `AE_INTERNAL_KEY` di AE kosong (mode dev lokal), header tidak dicek. `GET /health` selalu terbuka (untuk health check Render).

## 3. Endpoint

### 3.1 `GET /health`
Cek layanan hidup dan model termuat. Tidak membocorkan API key.

**Response 200** (live)
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "logreg-tfidf-2026-11-14",
  "labels": ["cara_reservasi", "info_operasional", "layanan_harga", "pembatalan_keluhan"],
  "conf_threshold": 0.55,
  "gemini_configured": true,
  "gemini_model": "gemini-3.1-flash-lite",
  "gemini_backup_model": "gemini-2.5-flash-lite",
  "auth_required": true
}
```
**503** kalau model gagal dimuat (`status: "error"`, `model_loaded: false`).

### 3.2 `POST /api/v1/classify`
Klasifikasi pertanyaan saja (tanpa Gemini). Cepat (±2–6 ms di lokal).

**Request**
```json
{ "pertanyaan": "Berapa harga gel polish kak?" }
```
**Response 200** (live)
```json
{
  "label": "layanan_harga",
  "confidence": 0.9479,
  "yakin": true,
  "ambang": 0.55,
  "probabilities": {
    "layanan_harga": 0.9479,
    "info_operasional": 0.0238,
    "cara_reservasi": 0.0197,
    "pembatalan_keluhan": 0.0086
  },
  "model_version": "logreg-tfidf-2026-11-14"
}
```
Contoh label ragu (live, `"halo"`):
```json
{ "label": "info_operasional", "confidence": 0.3437, "yakin": false, "ambang": 0.55, "probabilities": { "info_operasional": 0.3437, "layanan_harga": 0.2649, "cara_reservasi": 0.2469, "pembatalan_keluhan": 0.1445 }, "model_version": "logreg-tfidf-2026-11-14" }
```

| Field | Tipe | Arti |
|---|---|---|
| `label` | string | Salah satu dari 4 label |
| `confidence` | number 0–1 | Probabilitas label teratas |
| `yakin` | boolean | `confidence >= ambang`. Kalau `false`, label tetap dikirim tapi jangan dipakai untuk keputusan penting |
| `ambang` | number | Ambang yang dipakai (default 0,55) |
| `probabilities` | object | Probabilitas tiap label, urut dari terbesar |

### 3.3 `POST /api/v1/chat`
Klasifikasi lalu jawaban Gemini yang dibatasi fakta di `knowledge.json`. Kalau Gemini gagal, tetap **HTTP 200**
dengan jawaban fallback statis.

**Request**
```json
{ "pertanyaan": "Berapa harga nail art premium?" }
```
**Response 200 — dari Gemini** (live; percobaan pertama timeout, percobaan kedua berhasil)
```json
{
  "jawaban": "Harga untuk layanan Nail Art Premium adalah Rp250.000 dengan durasi pengerjaan 2 jam. Jika ada informasi lain yang ingin ditanyakan, silakan hubungi pihak studio secara langsung ya.",
  "label": "layanan_harga",
  "confidence": 0.9035,
  "yakin": true,
  "sumber": "gemini",
  "model_dipakai": "gemini-3.1-flash-lite",
  "percobaan": 2,
  "durasi_ms": 11785
}
```
**Response 200 — fakta yang tidak ada** (live)
```json
{
  "jawaban": "Mohon maaf, untuk saat ini saya belum memiliki informasi mengenai layanan ombre di daftar layanan kami. Saya sarankan Anda menghubungi pihak studio The Nail Palace secara langsung untuk menanyakan ketersediaan layanan tersebut. Apakah ada layanan lain dari daftar kami yang ingin Anda ketahui?",
  "label": "layanan_harga",
  "confidence": 0.893,
  "yakin": true,
  "sumber": "gemini",
  "model_dipakai": "gemini-3.1-flash-lite",
  "percobaan": 1,
  "durasi_ms": 732
}
```
**Response 200 — fallback** (live; Gemini sengaja dibuat gagal total dengan nama model salah)
```json
{
  "jawaban": "Maaf, asisten sedang sibuk. Daftar layanan dan harga lengkap bisa dilihat di halaman /pesan (contoh: Full Set Acrylic Rp350.000, Gel Extension Rp300.000, Gel Polish Rp120.000). Untuk info yang belum tersedia, silakan hubungi studio The Nail Palace secara langsung.",
  "label": "layanan_harga",
  "confidence": 0.9479,
  "yakin": true,
  "sumber": "fallback",
  "model_dipakai": null,
  "percobaan": 2,
  "durasi_ms": 271
}
```

| Field | Tipe | Arti |
|---|---|---|
| `jawaban` | string | Teks untuk ditampilkan ke pelanggan (bisa berisi baris baru `\n` dan daftar bernomor) |
| `label`, `confidence`, `yakin` | | Sama seperti `/classify` |
| `sumber` | `"gemini"` \| `"fallback"` | `fallback` = jawaban statis karena Gemini gagal / key tidak ada |
| `model_dipakai` | string \| null | Model Gemini yang menjawab; `null` kalau fallback |
| `percobaan` | integer | Jumlah panggilan ke Gemini |
| `durasi_ms` | integer | Waktu bagian Gemini (ms) |

## 4. Error

Format sama dengan kontrak web inti: `{ "error": { "code": "...", "message": "..." } }`, pesan dalam Bahasa Indonesia.

| HTTP | `code` | Kapan | Contoh `message` (asli) |
|---|---|---|---|
| 400 | `VALIDATION_ERROR` | `pertanyaan` kosong/spasi saja | `Pertanyaan tidak boleh kosong.` |
| 400 | `VALIDATION_ERROR` | lebih dari 500 karakter (setelah spasi dirapikan) | `Pertanyaan terlalu panjang (maksimal 500 karakter).` |
| 400 | `VALIDATION_ERROR` | body bukan JSON / field hilang / bukan teks | `Body harus JSON dengan field 'pertanyaan' berupa teks.` |
| 401 | `UNAUTHORIZED` | `X-Internal-Key` tidak ada/salah | `Header X-Internal-Key tidak ada atau salah.` |
| 404 | `NOT_FOUND` | endpoint salah | `Endpoint tidak ditemukan.` |
| 405 | `METHOD_NOT_ALLOWED` | metode HTTP salah | `Metode HTTP tidak diizinkan untuk endpoint ini.` |
| 500 | `INTERNAL_ERROR` | error tak terduga | `Terjadi kesalahan di layanan AE.` |
| 503 | `MODEL_UNAVAILABLE` | model classifier gagal dimuat | `Model classifier belum siap.` |

Kegagalan Gemini **bukan** error HTTP: dibalas 200 dengan `sumber: "fallback"`.

## 5. Timeout yang disarankan untuk backend Node

| Panggilan | Timeout backend | Alasan |
|---|---|---|
| `GET /health` | 5 detik (atau 60 detik kalau dipakai untuk membangunkan Render) | |
| `POST /api/v1/classify` | 5 detik | Lokal ±2–6 ms |
| `POST /api/v1/chat` | **30 detik** | AE membatasi semua percobaan Gemini maks 25 detik (10 dtk/percobaan, utama 2x + cadangan 1x) |

Di uji live, jawaban Gemini biasanya 0,7–5,5 detik; tapi panggilan Gemini pertama setelah server start dua kali kena
timeout 10 detik lalu berhasil di percobaan kedua (total ±11,8 detik).

**Cold start Render Free:** layanan tidur setelah ±15 menit tanpa trafik; request pertama bisa butuh hingga ±1 menit.
Kalau backend timeout karena ini, perlakukan sama seperti AE tidak tersedia (lihat bagian 6). Opsional: panggil
`GET /health` tanpa menunggu hasil saat pelanggan membuka widget chatbot.

Backend **jangan mengulang** `/chat` otomatis (AE sudah retry ke Gemini sendiri).

## 6. Perilaku fallback dan apa yang ditampilkan frontend

| Kondisi | Yang dilakukan backend | Yang ditampilkan frontend |
|---|---|---|
| 200, `sumber: "gemini"` | Teruskan `jawaban` | Tampilkan jawaban biasa |
| 200, `sumber: "fallback"` | Teruskan `jawaban` | Tampilkan jawaban (sudah berisi arahan ke `/pesan`, `/cek-booking`, atau hubungi studio). Boleh diberi tanda kecil "mode terbatas" |
| 200, `yakin: false` | Teruskan | Tampilkan jawaban biasa (Gemini sudah diberi tahu labelnya ragu) |
| 400 `VALIDATION_ERROR` | Teruskan `message` | Tampilkan di bawah kotak input; sebaiknya input frontend juga dibatasi 500 karakter dan tombol kirim mati kalau kosong |
| 401 | Catat di log backend (konfigurasi key salah) | Pesan umum (baris berikut) |
| Timeout / 5xx / koneksi gagal | Balas ke frontend dengan pesan umum | "Maaf, asisten sedang tidak tersedia. Kamu tetap bisa pesan di halaman /pesan atau cek status di /cek-booking." + tombol/tautan ke `/pesan` |

Widget chatbot tidak boleh memblokir halaman: error chatbot tidak boleh membuat halaman pemesanan gagal dimuat.

Jawaban fallback per label (dari `knowledge.json`):
- `layanan_harga`: arahan ke `/pesan` + 3 contoh harga dari daftar layanan
- `info_operasional`: studio di Kota Jambi, jam/alamat/kontak hubungi studio
- `cara_reservasi`: langkah singkat di `/pesan` dan cek status di `/cek-booking`
- `pembatalan_keluhan`: booking batal tidak bisa diaktifkan lagi, pesan ulang lewat `/pesan`, hubungi studio
- label ragu (`yakin: false`): arahan umum `/pesan`, `/cek-booking`, hubungi studio

## 7. Environment variable

**Layanan AE (Render):**
| Nama | Wajib | Default |
|---|---|---|
| `GEMINI_API_KEY` | untuk jawaban generatif | – |
| `AE_INTERNAL_KEY` | disarankan | kosong (mode dev) |
| `GEMINI_MODEL` / `GEMINI_BACKUP_MODEL` | tidak | `gemini-3.1-flash-lite` / `gemini-2.5-flash-lite` |
| `GEMINI_TIMEOUT`, `GEMINI_RETRIES`, `GEMINI_BACKOFF`, `GEMINI_TOTAL_BUDGET` | tidak | `10`, `2`, `1.0`, `25` |
| `AE_CONF_THRESHOLD`, `AE_MAX_LEN` | tidak | `0.55`, `500` |
| `AE_CORS_ORIGINS` | tidak | kosong = CORS mati |

**Backend Node (usulan nama):**
| Nama | Isi |
|---|---|
| `AE_BASE_URL` | mis. `https://nailpalace-ae.onrender.com` |
| `AE_INTERNAL_KEY` | sama persis dengan di layanan AE |
| `AE_TIMEOUT_MS` | `30000` |

## 8. Privasi
- Layanan AE tidak menyimpan pertanyaan dan tidak menulis isi pertanyaan ke log.
- Backend sebaiknya **tidak** mengirim data pelanggan (nama, nomor HP, kode booking) ke AE; cukup teks pertanyaan.
- Asisten tidak bisa melihat atau mengubah booking.

## 9. Checklist kesepakatan (centang saat disepakati FS)
- [ ] Frontend → backend Node → AE (bukan frontend langsung ke AE)
- [ ] Nama endpoint chatbot di backend Node (usulan `POST /api/v1/chatbot`)
- [ ] Bentuk request/response `/api/v1/chat`
- [ ] Header `X-Internal-Key` + cara menyimpan rahasia di env Render
- [ ] Timeout backend 30 detik untuk `/chat`, tanpa retry di backend
- [ ] Pesan error chatbot di frontend
- [ ] Batas 500 karakter juga di input frontend
- [ ] Apakah `confidence`/`label` ditampilkan ke pelanggan (usulan: tidak, hanya untuk log/evaluasi)

Perubahan setelah disepakati wajib naik versi (`v0.2`) dan dicatat di progress tim.
