# Bukti uji chatbot (Gemini)

| File | Isi |
|---|---|
| `test_cases.json` | 47 kasus uji (39 kasus awal + 8 kasus di luar topik X03–X10 yang ditambah 21 Nov 2026) |
| `results_gemini-3.1-flash-lite_20261114_1330.csv` | Uji 39 kasus dengan prompt v2 (14 Nov 2026 13.30 WIB): 34/39 lolos, 5 gagal karena error server Gemini (503/timeout) |
| `results_gemini-3.1-flash-lite_20261114_1335.csv` | Uji ulang 5 kasus yang kena 503/timeout: 5/5 lolos |
| `results_gemini-3.1-flash-lite_20261121_1147.csv` | Uji 10 kasus di luar topik X01–X10 (21 Nov 2026 11.47 WIB): 10/10 lolos cek otomatis + dibaca manual |
| `ringkasan_hasil_terakhir.csv` | Hasil terakhir per kasus (gabungan tiga file di atas) |
| `run_tests.py`, `assistant.py` | Skrip uji (salinan dari folder kerja AE) |

Menjalankan ulang (butuh `GEMINI_API_KEY` di env, jangan ditulis ke file):
```bash
cd docs/chatbot-tests
cp ../../app/data/knowledge.json ../../app/data/system_prompt_template.md .
python3 run_tests.py --only X01,X02,X03 --sleep 6
```
`auto_pass` = cek pola regex otomatis. `manual_pass` diisi setelah jawaban dibaca.
