# Bukti uji chatbot (Gemini)

Versi aktif: **katalog 15 layanan dari paket tim DS (21 Nov 2026)**, classifier 11 label.

| File | Isi |
|---|---|
| `test_cases.json` | 49 kasus: fakta per label (harga, layanan/desain, durasi, jam, lokasi, reservasi, cek/batal, pembayaran, keluhan, sapaan), info yang belum diketahui, batas kemampuan, **10 kasus di luar topik X01–X10** dan **4 prompt injection J01–J04** (dipertahankan dari set lama; J02 disesuaikan ke harga baru) |
| `results_gemini-3.1-flash-lite_20261121_1443.csv` | Uji 49 kasus (21 Nov 2026 14.43 WIB): **49/49 lolos cek otomatis**, 0 error server, semua jawaban dibaca manual |
| `run_tests.py`, `assistant.py` | Skrip uji. Memakai **kode layanan yang sama** (`app/genai.py`, `app/classifier.py`): pertanyaan → classifier → label + `yakin` → prompt dari `knowledge.json` → Gemini |
| `archive_8layanan_2026-11-14/` | Set lama (47 kasus, katalog web 8 layanan, 4 label) beserta hasilnya. Tidak berlaku lagi |

Menjalankan ulang (butuh `GEMINI_API_KEY` di env, jangan ditulis ke file):
```bash
cd docs/chatbot-tests
python3 run_tests.py --sleep 3            # semua kasus
python3 run_tests.py --only X01,J02       # sebagian
```
`auto_pass` = cek pola regex otomatis. `manual_pass` diisi setelah jawaban dibaca.
