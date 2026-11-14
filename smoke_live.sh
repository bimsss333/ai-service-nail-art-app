#!/usr/bin/env bash
# Smoke test live: jalankan uvicorn dulu, lalu: BASE=http://127.0.0.1:8077 AE_INTERNAL_KEY=... ./smoke_live.sh
BASE=${BASE:-http://127.0.0.1:8000}
H=(-H "Content-Type: application/json"); [ -n "$AE_INTERNAL_KEY" ] && H+=(-H "X-Internal-Key: $AE_INTERNAL_KEY")
post() { echo "### POST $1  $2"; curl -s -w '\n[HTTP %{http_code}, %{time_total}s]\n\n' "${H[@]}" -X POST "$BASE$1" -d "$2"; }
echo "### waktu: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "### GET /health"; curl -s -w '\n[HTTP %{http_code}, %{time_total}s]\n\n' "$BASE/health"
for q in "Berapa harga gel polish kak?" "Studio buka jam berapa ya?" "Gimana cara booking buat besok?" "Kalau mau batalin reservasi gimana?" "halo"; do
  post /api/v1/classify "{\"pertanyaan\": \"$q\"}"; done
for q in "Berapa harga nail art premium?" "Cara pesan jadwal gimana ya kak?" "Ada layanan ombre nggak? berapa harganya?" "Saya mau batalin booking, bisa?"; do
  post /api/v1/chat "{\"pertanyaan\": \"$q\"}"; done
echo "### Error cases"
post /api/v1/classify '{"pertanyaan": ""}'
post /api/v1/classify "{\"pertanyaan\": \"$(printf 'a%.0s' {1..501})\"}"
echo "### POST tanpa X-Internal-Key"; curl -s -w '\n[HTTP %{http_code}]\n' -H "Content-Type: application/json" -X POST "$BASE/api/v1/classify" -d '{"pertanyaan":"harga?"}'
