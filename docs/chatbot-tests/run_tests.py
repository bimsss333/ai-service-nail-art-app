"""Uji chatbot The Nail Palace ke Gemini API (dev saja). Alur sama dengan /api/v1/chat:
pertanyaan -> classifier (model layanan) -> label + yakin -> prompt dari knowledge.json -> Gemini.
Pakai:  export GEMINI_API_KEY=...  lalu  python3 run_tests.py [--model gemini-3.1-flash-lite] [--only H01,X03] [--sleep 4]
auto_pass = cek regex otomatis; manual_pass diisi manusia setelah membaca jawaban.
Kasus yang gagal karena error server (HTTP 503/429/timeout) ditandai ERROR, bukan jawaban salah."""
import argparse, csv, json, re, sys, time
from datetime import datetime
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from assistant import ask, classify, PRIMARY_MODEL, build_system_prompt

FIELDS = ["id", "label", "label_model", "confidence", "yakin", "kategori", "pertanyaan", "expected", "model", "jawaban",
          "auto_pass", "auto_alasan", "manual_pass", "catatan"]

def check(ans, case):
    reasons = []
    if case["must_any"] and not any(re.search(p, ans, re.I | re.S) for p in case["must_any"]):
        reasons.append("tidak memuat pola wajib: " + " | ".join(case["must_any"]))
    for p in case["must_not"]:
        if re.search(p, ans, re.I | re.S | re.M): reasons.append("memuat pola terlarang: " + p)
    return (not reasons), "; ".join(reasons)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=PRIMARY_MODEL); ap.add_argument("--only", default="")
    ap.add_argument("--sleep", type=float, default=2.0); ap.add_argument("--tag", default=""); a = ap.parse_args()
    cases = json.loads((HERE / "test_cases.json").read_text(encoding="utf-8"))
    if a.only: cases = [c for c in cases if c["id"] in a.only.split(",")]
    for c in cases: [re.compile(p) for p in c["must_any"] + c["must_not"]]; build_system_prompt(c["label"])
    rows, stamp = [], datetime.now().strftime("%Y%m%d_%H%M")
    for c in cases:
        r = classify(c["pertanyaan"])
        for attempt in range(3):
            try:
                ans = ask(c["pertanyaan"], r["label"], r["yakin"], model=a.model); ok, why = check(ans, c); break
            except Exception as e:
                ans, ok, why = "", False, f"ERROR: {type(e).__name__} {getattr(e, 'code', '')}".strip(); time.sleep(5 * (attempt + 1))
        rows.append({"id": c["id"], "label": c["label"], "label_model": r["label"], "confidence": r["confidence"], "yakin": r["yakin"],
                     "kategori": c["kategori"], "pertanyaan": c["pertanyaan"], "expected": c["expected"], "model": a.model,
                     "jawaban": ans, "auto_pass": ok, "auto_alasan": why, "manual_pass": "", "catatan": ""})
        print(f"[{'PASS' if ok else 'FAIL'}] {c['id']} ({r['label']} {r['confidence']:.2f}) {c['pertanyaan'][:45]} -> {why}"); time.sleep(a.sleep)
    base = HERE / f"results_{a.model}_{stamp}{a.tag}"
    with open(f"{base}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS); w.writeheader(); w.writerows(rows)
    n = sum(r["auto_pass"] for r in rows); err = sum(r["auto_alasan"].startswith("ERROR") for r in rows)
    print(f"\nAuto pass: {n}/{len(rows)} ({n/len(rows):.1%}), error server: {err}. Hasil: {base.name}.csv")
    return 0

if __name__ == "__main__": sys.exit(main())
