"""Evaluasi classifier pertanyaan The Nail Palace (A26-CS032) — dataset tim DS v1.0 (21 Nov 2026), 11 label.
TF-IDF + LogisticRegression vs TF-IDF + LinearSVC, metrik utama macro F1.

Split mengikuti paket DS (train 875 / val 172 / test 164, dibagi per `split_group`, checksum di meta.json):
  1. Tuning  : GridSearchCV di TRAIN, cv = StratifiedGroupKFold(5) per group_id (parafrase satu kalimat dasar tidak terpisah).
  2. Estimasi: nested CV di TRAIN (outer StratifiedGroupKFold 5, inner 3) -> mean ± SD macro F1.
  3. Seleksi : model terbaik tiap kandidat di-fit di train, dinilai di VAL. Pilihan model + ambang dikunci (selection.json)
               SEBELUM test dibuka.
  4. Ambang  : dari probabilitas VAL (LogReg). Prediksi di bawah ambang -> `di_luar_konteks` (aturan fallback DS).
  5. Final   : model terpilih di-fit ulang di TRAIN+VAL, dinilai SEKALI di TEST.
Pakai: python eval_classifier_11label.py [folder dataset_pertanyaan]
"""
import hashlib, json, sys, platform
from pathlib import Path
import numpy as np, pandas as pd, sklearn, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

RS = 42
D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/capstone/ds_hasil_21nov/datasciencs-nail-art/output/dataset_pertanyaan")
OUT = Path(__file__).resolve().parent
MODEL_OUT = OUT.parent.parent / "model" / "classifier_11label_logreg.joblib"
FALLBACK = "di_luar_konteks"
log_lines = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); log_lines.append(s)

# --- data + integritas ---
meta = json.loads((D / "meta.json").read_text())
for f in ["train.csv", "val.csv", "test.csv"]:
    h = hashlib.sha256((D / f).read_bytes()).hexdigest()
    assert h == meta["sha256"][f], f"checksum {f} tidak cocok"
log("checksum train/val/test cocok dengan meta.json")
LABELS = sorted(json.loads((D / "label.json").read_text())["label"].keys())
def load(name):
    df = pd.read_csv(D / f"{name}.csv"); n0 = len(df)
    df["teks"] = df["teks"].astype(str).str.strip(); df = df[df["teks"] != ""].reset_index(drop=True)
    log(f"{name}: N={len(df)} (dibuang {n0-len(df)} teks kosong)")
    return df
tr, va, te = load("train"), load("val"), load("test")
assert not (set(tr.group_id) & set(va.group_id) | set(tr.group_id) & set(te.group_id) | set(va.group_id) & set(te.group_id)), "leakage group_id"
assert not (set(tr.teks.str.lower()) & set(te.teks.str.lower())), "teks sama di train & test"
log("tidak ada group_id / teks yang sama lintas split")
log("Label:", LABELS)
log("Jumlah per label (train/val/test):")
log(pd.DataFrame({s: d.label.value_counts() for s, d in [("train", tr), ("val", va), ("test", te)]}).loc[LABELS].to_string())
Xtr, ytr, gtr = tr.teks.to_numpy(), tr.label.to_numpy(), tr.group_id.to_numpy()
Xva, yva, Xte, yte = va.teks.to_numpy(), va.label.to_numpy(), te.teks.to_numpy(), te.label.to_numpy()

# --- grid (ukuran sama untuk kedua model) ---
def vec(kind):
    if kind == "word12": return TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2))
    if kind == "word11": return TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 1))
    if kind == "char35": return TfidfVectorizer(lowercase=True, sublinear_tf=True, analyzer="char_wb", ngram_range=(3, 5))
VECS = ["word11", "word12", "char35"]
GRID = {"tfidf": [vec(k) for k in VECS], "clf__C": [0.1, 1, 10], "clf__class_weight": [None, "balanced"]}
MODELS = {"LogReg": lambda: LogisticRegression(max_iter=5000, random_state=RS),
          "LinearSVC": lambda: LinearSVC(random_state=RS)}
def pipe(m): return Pipeline([("tfidf", vec("word12")), ("clf", MODELS[m]())])
def vname(v): return "char_wb(3,5)" if v.analyzer == "char_wb" else f"word{v.ngram_range}"
def pstr(p): return {"tfidf": vname(p["tfidf"]), "C": p["clf__C"], "class_weight": p["clf__class_weight"]}

summary, best = {}, {}
for m in MODELS:
    # nested CV (estimasi jujur di train)
    outer, scores, accs, fold_params = StratifiedGroupKFold(5, shuffle=True, random_state=RS), [], [], []
    for k, (a, b) in enumerate(outer.split(Xtr, ytr, gtr)):
        assert not set(gtr[a]) & set(gtr[b])
        gs = GridSearchCV(pipe(m), GRID, scoring="f1_macro", cv=StratifiedGroupKFold(3, shuffle=True, random_state=RS), n_jobs=-1)
        gs.fit(Xtr[a], ytr[a], groups=gtr[a]); p = gs.predict(Xtr[b])
        scores.append(f1_score(ytr[b], p, average="macro")); accs.append(accuracy_score(ytr[b], p)); fold_params.append(pstr(gs.best_params_))
    # tuning di seluruh train (CV grouped 5) -> dinilai di val
    gs = GridSearchCV(pipe(m), GRID, scoring="f1_macro", cv=StratifiedGroupKFold(5, shuffle=True, random_state=RS), n_jobs=-1)
    gs.fit(Xtr, ytr, groups=gtr); best[m] = gs.best_estimator_
    pv = gs.predict(Xva)
    cvres = pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    cvres["params"] = cvres["params"].map(lambda p: json.dumps(pstr(p))); cvres.sort_values("rank_test_score").to_csv(OUT / f"tuning_{m}.csv", index=False)
    summary[m] = {"nested_cv_f1_mean": float(np.mean(scores)), "nested_cv_f1_sd": float(np.std(scores)),
                  "nested_cv_acc_mean": float(np.mean(accs)), "nested_fold_f1": [float(s) for s in scores], "nested_fold_params": fold_params,
                  "best_params_train_cv": pstr(gs.best_params_), "train_cv_f1": float(gs.best_score_),
                  "val_f1": float(f1_score(yva, pv, average="macro")), "val_acc": float(accuracy_score(yva, pv))}
    log(f"\n== {m} ==\nnested CV (train, grouped 5x3) macro F1 = {np.mean(scores):.4f} ± {np.std(scores):.4f} (acc {np.mean(accs):.4f})")
    log("  per fold:", [round(s, 4) for s in scores]); log("  params per fold:", fold_params)
    log(f"tuning train CV5 grouped: best {pstr(gs.best_params_)} f1={gs.best_score_:.4f}")
    log(f"VAL macro F1 = {summary[m]['val_f1']:.4f}  acc = {summary[m]['val_acc']:.4f}")
    (OUT / f"report_val_{m}.txt").write_text(classification_report(yva, pv, labels=LABELS, digits=3, zero_division=0))

# --- seleksi (dikunci sebelum test) ---
diff = summary["LogReg"]["val_f1"] - summary["LinearSVC"]["val_f1"]
sd = max(summary["LogReg"]["nested_cv_f1_sd"], summary["LinearSVC"]["nested_cv_f1_sd"])
if diff >= -sd:   # LogReg tidak kalah jelas -> LogReg (punya predict_proba untuk ambang)
    chosen, reason = "LogReg", f"selisih val F1 LogReg-LinearSVC = {diff:+.4f}, |selisih| vs SD nested terbesar {sd:.4f}; LogReg punya predict_proba untuk ambang keyakinan"
else:
    chosen, reason = "LinearSVC", f"LinearSVC unggul val F1 {-diff:.4f} > SD {sd:.4f}"
log(f"\nDIPILIH: {chosen} ({reason})")

# --- ambang dari VAL (model fit train) ---
thr_rows, chosen_thr = [], None
if chosen == "LogReg":
    m = best["LogReg"]; P = m.predict_proba(Xva); conf = P.max(1); pred = m.classes_[P.argmax(1)]; ok = pred == yva
    log(f"VAL: median conf benar={np.median(conf[ok]):.3f}, salah={np.median(conf[~ok]):.3f} (n salah={int((~ok).sum())})")
    log("thr  %ragu  acc_di_atas  n_di_atas  %salah_tertangkap  macroF1(ragu->di_luar_konteks)")
    for t in np.round(np.arange(0.15, 0.71, 0.05), 2):
        lo = conf < t; mapped = np.where(lo, FALLBACK, pred)
        r = {"thr": float(t), "pct_ragu": float(lo.mean()), "acc_di_atas": float(ok[~lo].mean()) if (~lo).any() else None,
             "n_di_atas": int((~lo).sum()), "pct_salah_tertangkap": float((~ok & lo).sum() / max(1, (~ok).sum())),
             "f1_macro_mapped": float(f1_score(yva, mapped, average="macro"))}
        thr_rows.append(r)
        log(f"{t:.2f}  {r['pct_ragu']*100:5.1f}%  {r['acc_di_atas']:.4f}  {r['n_di_atas']:4d}  {r['pct_salah_tertangkap']*100:5.1f}%  {r['f1_macro_mapped']:.4f}")
    base_f1 = summary["LogReg"]["val_f1"]
    # aturan: ambang terbesar yang macro F1 (setelah ragu -> di_luar_konteks) tidak turun > 0,01 dari tanpa ambang
    ok_rows = [r for r in thr_rows if r["f1_macro_mapped"] >= base_f1 - 0.01]
    chosen_thr = max(r["thr"] for r in ok_rows) if ok_rows else 0.0
    log(f"AMBANG DIPILIH: {chosen_thr} (terbesar dengan macro F1 val setelah pemetaan >= {base_f1:.4f} - 0,01)")
sel = {"chosen": chosen, "reason": reason, "threshold": chosen_thr, "threshold_rule": "ambang terbesar dgn macro F1 val (ragu->di_luar_konteks) >= F1 val tanpa ambang - 0.01",
       "params": summary[chosen]["best_params_train_cv"], "locked_before_test": True}
(OUT / "selection.json").write_text(json.dumps(sel, indent=2, ensure_ascii=False))
json.dump(thr_rows, open(OUT / "threshold_val.json", "w"), indent=2)

# --- final: fit train+val, test SEKALI ---
Xall, yall = np.concatenate([Xtr, Xva]), np.concatenate([ytr, yva])
final = {}
for m in MODELS:
    f = clone(best[m]).fit(Xall, yall); pt = f.predict(Xte); final[m] = f
    summary[m]["test_f1"] = float(f1_score(yte, pt, average="macro")); summary[m]["test_acc"] = float(accuracy_score(yte, pt))
    (OUT / f"report_test_{m}.txt").write_text(classification_report(yte, pt, labels=LABELS, digits=3, zero_division=0))
    cm = confusion_matrix(yte, pt, labels=LABELS)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(OUT / f"cm_test_{m}.csv")
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, xticks_rotation=60, colorbar=False)
    ax.set_title(f"{m} — test (n={len(yte)}), macro F1 {summary[m]['test_f1']:.3f}"); fig.tight_layout(); fig.savefig(OUT / f"cm_test_{m}.png", dpi=110); plt.close(fig)
    pairs = [(LABELS[i], LABELS[j], int(cm[i, j])) for i in range(len(LABELS)) for j in range(len(LABELS)) if i != j and cm[i, j] > 0]
    summary[m]["top_confusions"] = sorted(pairs, key=lambda x: -x[2])[:5]
    log(f"\nTEST {m}{' (TERPILIH)' if m == chosen else ' (pembanding, seleksi sudah dikunci)'}: macro F1 = {summary[m]['test_f1']:.4f}, acc = {summary[m]['test_acc']:.4f}")
    log(classification_report(yte, pt, labels=LABELS, digits=3, zero_division=0))
    log("  kebingungan terbanyak (asli -> prediksi):", summary[m]["top_confusions"])
    errs = te.assign(prediksi=pt)[pt != yte][["id", "teks", "label", "prediksi"]]; errs.to_csv(OUT / f"errors_test_{m}.csv", index=False)
if chosen == "LogReg":
    f = final["LogReg"]; P = f.predict_proba(Xte); conf = P.max(1); pred = f.classes_[P.argmax(1)]
    mapped = np.where(conf < chosen_thr, FALLBACK, pred)
    summary["LogReg"]["test_with_threshold"] = {"thr": chosen_thr, "pct_ragu": float((conf < chosen_thr).mean()),
        "f1_macro_mapped": float(f1_score(yte, mapped, average="macro")), "acc_mapped": float(accuracy_score(yte, mapped)),
        "acc_di_atas": float((pred == yte)[conf >= chosen_thr].mean())}
    log("TEST dengan ambang:", summary["LogReg"]["test_with_threshold"])
joblib.dump(final[chosen], MODEL_OUT)
log(f"\nModel final ({chosen}, fit train+val n={len(yall)}) disimpan: {MODEL_OUT.name}")
env = {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
       "matplotlib": matplotlib.__version__, "joblib": joblib.__version__}
json.dump({"dataset": str(D), "dataset_versi_label": meta.get("versi_label"), "n": {"train": len(tr), "val": len(va), "test": len(te)},
           "labels": LABELS, "selection": sel, "summary": summary, "env": env}, open(OUT / "results.json", "w"), indent=2, ensure_ascii=False)
rows = [{"model": m, "nested_cv_f1": f"{s['nested_cv_f1_mean']:.4f} ± {s['nested_cv_f1_sd']:.4f}", "val_f1": round(s["val_f1"], 4),
         "val_acc": round(s["val_acc"], 4), "test_f1": round(s["test_f1"], 4), "test_acc": round(s["test_acc"], 4),
         "best_params": json.dumps(s["best_params_train_cv"]), "dipilih": m == chosen} for m, s in summary.items()]
pd.DataFrame(rows).to_csv(OUT / "results_summary.csv", index=False)
log("env:", env)
(OUT / "run_log.txt").write_text("\n".join(log_lines) + "\n")
