"""Uji prompt asisten The Nail Palace ke Gemini API (dev saja, bukan pelanggan asli).
Pakai:  export GEMINI_API_KEY=...   lalu   python3 run_tests.py [--model gemini-3.1-flash-lite] [--only H01,R03]
Tanpa key: hanya membuat results_template.csv (tidak ada jawaban palsu).
Hasil: results_<model>_<waktu>.csv/.json. auto_pass = cek regex otomatis; kolom manual_pass WAJIB dicek manusia."""
import argparse, csv, json, os, re, sys, time
from datetime import datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assistant import ask, PRIMARY_MODEL, build_system_prompt

FIELDS = ["id", "label", "kategori", "pertanyaan", "expected", "sumber", "model", "jawaban", "auto_pass", "auto_alasan", "manual_pass", "catatan"]

def check(ans, case):
    reasons = []
    if case["must_any"] and not any(re.search(p, ans, re.I | re.S) for p in case["must_any"]):
        reasons.append("tidak memuat pola wajib: " + " | ".join(case["must_any"]))
    for p in case["must_not"]:
        if re.search(p, ans, re.I | re.S | re.M): reasons.append("memuat pola terlarang: " + p)
    return (not reasons), "; ".join(reasons)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=PRIMARY_MODEL); ap.add_argument("--only", default="")
    ap.add_argument("--sleep", type=float, default=1.0); a = ap.parse_args()
    cases = json.loads((HERE / "test_cases.json").read_text(encoding="utf-8"))
    if a.only: cases = [c for c in cases if c["id"] in a.only.split(",")]
    for c in cases:  # validasi pola regex & prompt bisa dirakit
        [re.compile(p) for p in c["must_any"] + c["must_not"]]; build_system_prompt(c["label"])
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        with open(HERE / "results_template.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, FIELDS); w.writeheader()
            for c in cases: w.writerow({k: c.get(k, "") for k in FIELDS} | {"model": "", "jawaban": "", "auto_pass": "", "auto_alasan": "BELUM DIJALANKAN: API key belum ada", "manual_pass": ""})
        print(f"API key tidak ditemukan. {len(cases)} kasus valid, template ditulis ke results_template.csv. Set GEMINI_API_KEY lalu jalankan ulang."); return 2
    rows, stamp = [], datetime.now().strftime("%Y%m%d_%H%M")
    for c in cases:
        try: ans = ask(c["pertanyaan"], c["label"], model=a.model); ok, why = check(ans, c)
        except Exception as e: ans, ok, why = "", False, f"ERROR: {e}"
        rows.append({k: c.get(k, "") for k in FIELDS} | {"model": a.model, "jawaban": ans, "auto_pass": ok, "auto_alasan": why, "manual_pass": "", "catatan": ""})
        print(f"[{'PASS' if ok else 'FAIL'}] {c['id']} {c['pertanyaan'][:50]} -> {why}"); time.sleep(a.sleep)
    base = HERE / f"results_{a.model}_{stamp}"
    with open(f"{base}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(f"{base}.json", "w"), indent=2, ensure_ascii=False)
    n = sum(r["auto_pass"] for r in rows); print(f"\nAuto pass: {n}/{len(rows)} ({n/len(rows):.1%}). Hasil: {base}.csv — cek manual kolom manual_pass.")
    return 0

if __name__ == "__main__": sys.exit(main())
