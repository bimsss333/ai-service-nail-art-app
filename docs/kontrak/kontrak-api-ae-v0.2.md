# Kontrak API Layanan AE — The Nail Palace (A26-CS032)
Versi: 0.2 — **DRAF, perlu disepakati bersama Arief (FS) / Thoha (FS) / Bima (AE)**  
Tanggal: 21 Nov 2026 WIB (menggantikan v0.1, 14 Nov 2026)  

> ⚠️ **BREAKING CHANGE untuk backend (Arief/Thoha).** Label classifier berubah dari **4 label** menjadi **11 label**
> (dataset pertanyaan tim DS v1.0, 21 Nov 2026). Kode backend yang mencocokkan nama label lama
> (`layanan_harga`, `info_operasional`, `pembatalan_keluhan`) harus diubah. Ringkasan perubahan ada di [bagian 0](#0-perubahan-dari-v01).

## 0. Perubahan dari v0.1
| Bagian | v0.1 (14 Nov) | v0.2 (21 Nov) | Dampak ke backend |
|---|---|---|---|
| Label | 4: `layanan_harga`, `info_operasional`, `cara_reservasi`, `pembatalan_keluhan` | 11: `harga`, `layanan_desain`, `durasi`, `jam_operasional`, `lokasi`, `cara_reservasi`, `cek_ubah_batal`, `pembayaran`, `perawatan_keluhan`, `sapaan`, `di_luar_konteks` | **Breaking** kalau backend memakai nama label |
| Field baru `aksi` | – | Di `/classify` dan `/chat`: `jawab_dari_fakta` \| `arahkan_ke_reservasi` \| `arahkan_ke_cek_reservasi` \| `balas_sapaan` \| `arahkan_ke_kontak` \| `ragu` | Tambahan (tidak merusak). Bisa dipakai frontend untuk tombol ke `/pesan` atau `/cek-booking` |
| Ambang `yakin` | 0,55 | **0,50** | Nilai saja |
| `model_version` | `logreg-tfidf-2026-11-14` | `logreg-tfidf-11label-2026-11-21` | Nilai saja |
| Fakta chatbot | 8 layanan web | **15 layanan katalog DS (S01–S15)**, jam buka, pembayaran, Instagram | Harga di web perlu disamakan dengan katalog DS |
| Endpoint, request, format error, auth, timeout | – | **Tidak berubah** | – |

Pelengkap dari: `kontrak-api-web-inti-v0.1.md` (bagian 6: "FastAPI AE — kontrak terpisah")  
Kode: `ae-service/` (FastAPI). Semua contoh JSON di bawah diambil dari **uji live 14 Nov 2026 pukul 14.18–14.20 WIB**
(`ae-service/test_output/`), bukan karangan.
Contoh v0.2 diambil dari **uji live 21 Nov 2026 pukul 14.48 WIB** (`test_output/live_smoke_20261121.txt`, `live_fallback_20261121.txt`).

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

**Response 200** (live 21 Nov)
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "logreg-tfidf-11label-2026-11-21",
  "labels": ["cara_reservasi", "cek_ubah_batal", "di_luar_konteks", "durasi", "harga", "jam_operasional",
             "layanan_desain", "lokasi", "pembayaran", "perawatan_keluhan", "sapaan"],
  "conf_threshold": 0.5,
  "gemini_configured": true,
  "gemini_model": "gemini-3.1-flash-lite",
  "gemini_backup_model": "gemini-2.5-flash-lite",
  "auth_required": true
}
```
**503** kalau model gagal dimuat (`status: "error"`, `model_loaded: false`).

### 3.2 `POST /api/v1/classify`
Klasifikasi pertanyaan saja (tanpa Gemini). Cepat (±2–5 ms di lokal).

**Request**
```json
{ "pertanyaan": "Berapa harga gel polish kak?" }
```
**Response 200** (live, probabilitas dipotong)
```json
{
  "label": "harga",
  "confidence": 0.9538,
  "yakin": true,
  "ambang": 0.5,
  "aksi": "jawab_dari_fakta",
  "probabilities": { "harga": 0.9538, "durasi": 0.0138, "layanan_desain": 0.0116, "...": "11 label, urut terbesar" },
  "model_version": "logreg-tfidf-11label-2026-11-21"
}
```
Contoh label ragu (live, `"Kalau mau batalin reservasi gimana?"`):
```json
{ "label": "cek_ubah_batal", "confidence": 0.4314, "yakin": false, "ambang": 0.5, "aksi": "ragu", "probabilities": { "cek_ubah_batal": 0.4314, "cara_reservasi": 0.2186, "...": "..." }, "model_version": "logreg-tfidf-11label-2026-11-21" }
```

| Field | Tipe | Arti |
|---|---|---|
| `label` | string | Salah satu dari 11 label (tabel di bawah) |
| `confidence` | number 0–1 | Probabilitas label teratas |
| `yakin` | boolean | `confidence >= ambang`. Kalau `false`, label tetap dikirim tapi jangan dipakai untuk keputusan penting |
| `ambang` | number | Ambang yang dipakai (default 0,50) |
| `aksi` | string | Aksi asisten untuk label ini (tabel di bawah); `"ragu"` kalau `yakin: false` |
| `probabilities` | object | Probabilitas tiap label, urut dari terbesar |

**11 label dan aksinya** (sumber: `label.json` tim DS v1.0)
| Label | Arti | `aksi` |
|---|---|---|
| `harga` | harga, tarif, kisaran biaya | `jawab_dari_fakta` |
| `layanan_desain` | jenis layanan/desain yang tersedia | `jawab_dari_fakta` |
| `durasi` | lama pengerjaan | `jawab_dari_fakta` |
| `jam_operasional` | jam buka/tutup/istirahat, hari libur | `jawab_dari_fakta` |
| `lokasi` | alamat, arah, parkir | `jawab_dari_fakta` |
| `cara_reservasi` | cara pesan, slot, ingin reservasi | `arahkan_ke_reservasi` (ke `/pesan`) |
| `cek_ubah_batal` | cek status, batal, jadwal ulang | `arahkan_ke_cek_reservasi` (ke `/cek-booking`) |
| `pembayaran` | metode bayar, DP, bukti transfer | `jawab_dari_fakta` |
| `perawatan_keluhan` | keluhan hasil, garansi, perawatan | `jawab_dari_fakta` (+ kontak studio) |
| `sapaan` | sapaan, terima kasih | `balas_sapaan` |
| `di_luar_konteks` | di luar topik / ambigu | `arahkan_ke_kontak` |

### 3.3 `POST /api/v1/chat`
Klasifikasi lalu jawaban Gemini yang dibatasi fakta di `knowledge.json`. Kalau Gemini gagal, tetap **HTTP 200**
dengan jawaban fallback statis per label.

**Request**
```json
{ "pertanyaan": "Saya mau batalin booking, bisa?" }
```
**Response 200 — dari Gemini** (live 21 Nov)
```json
{
  "jawaban": "Tentu, Anda bisa membatalkan reservasi melalui halaman /cek-booking dengan memasukkan kode booking (NP-YYYYMMDD-XXXX) dan nomor HP yang digunakan saat memesan. Perlu diingat, pembatalan hanya dapat dilakukan sebelum jadwal layanan dimulai. Jika ada kendala, silakan hubungi kami melalui Instagram https://www.instagram.com/thenailpalace____/.",
  "label": "cek_ubah_batal",
  "confidence": 0.9888,
  "yakin": true,
  "aksi": "arahkan_ke_cek_reservasi",
  "sumber": "gemini",
  "model_dipakai": "gemini-3.1-flash-lite",
  "percobaan": 1,
  "durasi_ms": 2792
}
```
**Response 200 — fallback** (live; Gemini sengaja dibuat gagal dengan nama model salah)
```json
{
  "jawaban": "Maaf, asisten sedang sibuk. The Nail Palace buka Senin–Sabtu pukul 10:00–19:00 WIB, tutup hari Minggu. Slot yang masih kosong bisa dicek di halaman /pesan.",
  "label": "jam_operasional",
  "confidence": 0.9915,
  "yakin": true,
  "aksi": "jawab_dari_fakta",
  "sumber": "fallback",
  "model_dipakai": null,
  "percobaan": 2,
  "durasi_ms": 207
}
```

| Field | Tipe | Arti |
|---|---|---|
| `jawaban` | string | Teks untuk ditampilkan ke pelanggan (bisa berisi baris baru `\n` dan daftar) |
| `label`, `confidence`, `yakin`, `aksi` | | Sama seperti `/classify` |
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
- `harga` / `layanan_desain` / `durasi`: arahan ke `/pesan` + 3 contoh layanan dengan harga dan durasi
- `jam_operasional`: Senin–Sabtu 10:00–19:00 WIB, Minggu tutup, slot di `/pesan`
- `lokasi`: Kota Jambi, alamat lengkap lewat Instagram studio
- `cara_reservasi`: langkah singkat di `/pesan` dan cek status di `/cek-booking`
- `cek_ubah_batal`: cek/batal di `/cek-booking` (kode booking + no HP), jadwal ulang lewat studio
- `pembayaran`: tunai/transfer/QRIS, dibayar di studio setelah layanan, tanpa DP
- `perawatan_keluhan`: hubungi studio lewat Instagram
- `sapaan`: balasan sapaan + tawaran bantuan
- `di_luar_konteks` dan label ragu (`yakin: false`): arahan umum `/pesan`, `/cek-booking`, kontak Instagram

## 7. Environment variable

**Layanan AE (Render):**
| Nama | Wajib | Default |
|---|---|---|
| `GEMINI_API_KEY` | untuk jawaban generatif | – |
| `AE_INTERNAL_KEY` | disarankan | kosong (mode dev) |
| `GEMINI_MODEL` / `GEMINI_BACKUP_MODEL` | tidak | `gemini-3.1-flash-lite` / `gemini-2.5-flash-lite` |
| `GEMINI_TIMEOUT`, `GEMINI_RETRIES`, `GEMINI_BACKOFF`, `GEMINI_TOTAL_BUDGET` | tidak | `10`, `2`, `1.0`, `25` |
| `AE_CONF_THRESHOLD`, `AE_MAX_LEN` | tidak | `0.50`, `500` |
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
- [ ] **v0.2:** backend tidak lagi memakai nama 4 label lama; kalau perlu, pakai field `aksi` (mis. tampilkan tombol ke `/pesan` untuk `arahkan_ke_reservasi`)
- [ ] **v0.2:** harga layanan di web disamakan dengan katalog DS 15 layanan (S01–S15)

Perubahan setelah disepakati wajib naik versi (`v0.3`) dan dicatat di progress tim. v0.1 tetap disimpan di `kontrak-api-ae-v0.1.md` sebagai riwayat.
