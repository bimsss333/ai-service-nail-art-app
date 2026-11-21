"""Evaluasi classifier pertanyaan The Nail Palace (A26-CS032).
TF-IDF + LogisticRegression vs TF-IDF + LinearSVC, metrik macro F1.
Dua skema split:
  - grouped : StratifiedGroupKFold berdasarkan kelompok_intent (anti-leakage parafrase)
  - random  : StratifiedKFold biasa (pembanding, cenderung terlalu optimis)
Nested CV: inner GridSearchCV (f1_macro) untuk tuning, outer fold untuk skor.
"""
import json, sys, platform
from pathlib import Path
import numpy as np, pandas as pd, sklearn, joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

RS = 42
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/capstone/data/dataset_classifier_v2_label_pp.csv")
OUT = Path(__file__).resolve().parent
LABELS = ["layanan_harga", "info_operasional", "cara_reservasi", "pembatalan_keluhan"]

df = pd.read_csv(DATA)
n0 = len(df)
df["pertanyaan"] = df["pertanyaan"].astype(str).str.strip()
df = df[df["pertanyaan"] != ""].reset_index(drop=True)
X, y, g = df["pertanyaan"].to_numpy(), df["label"].to_numpy(), df["kelompok_intent"].to_numpy()
print(f"N={len(df)} (dibuang {n0-len(df)} teks kosong)"); print(df["label"].value_counts().to_string())

grid_tfidf = {"tfidf__ngram_range": [(1, 1), (1, 2)], "tfidf__min_df": [1, 2], "tfidf__sublinear_tf": [True]}
MODELS = {
    "LogReg": (lambda: LogisticRegression(max_iter=5000, random_state=RS), {"clf__C": [0.1, 1, 10]}),
    "LinearSVC": (lambda: LinearSVC(random_state=RS), {"clf__C": [0.1, 1, 10]}),
}
SPLITS = {
    "grouped": (StratifiedGroupKFold(5, shuffle=True, random_state=RS), lambda: StratifiedGroupKFold(3, shuffle=True, random_state=RS)),
    "random": (StratifiedKFold(5, shuffle=True, random_state=RS), lambda: StratifiedKFold(3, shuffle=True, random_state=RS)),
}

def make_pipe(clf):
    return Pipeline([("tfidf", TfidfVectorizer(lowercase=True)), ("clf", clf)])

results, rows = {}, []
for sname, (outer, inner_f) in SPLITS.items():
    for mname, (clf_f, g_clf) in MODELS.items():
        scores, accs, yt, yp, params, ids = [], [], [], [], [], []
        split_iter = outer.split(X, y, g) if sname == "grouped" else outer.split(X, y)
        for k, (tr, te) in enumerate(split_iter):
            if sname == "grouped":
                assert not set(g[tr]) & set(g[te]), "leakage grup!"
            gs = GridSearchCV(make_pipe(clf_f()), {**grid_tfidf, **g_clf}, scoring="f1_macro", cv=inner_f(), n_jobs=-1)
            if sname == "grouped":
                gs.fit(X[tr], y[tr], groups=g[tr])
            else:
                gs.fit(X[tr], y[tr])
            p = gs.predict(X[te])
            f = f1_score(y[te], p, average="macro"); a = accuracy_score(y[te], p)
            scores.append(f); accs.append(a); params.append({kk: str(v) for kk, v in gs.best_params_.items()})
            yt += list(y[te]); yp += list(p); ids += list(df["id"].iloc[te])
            rows.append({"split": sname, "model": mname, "fold": k + 1, "n_test": len(te), "macro_f1": f, "accuracy": a, "best_params": json.dumps(params[-1])})
        key = f"{sname}_{mname}"
        rep = classification_report(yt, yp, labels=LABELS, digits=3)
        (OUT / f"report_{key}.txt").write_text(rep)
        cm = confusion_matrix(yt, yp, labels=LABELS)
        disp = ConfusionMatrixDisplay(cm, display_labels=LABELS)
        fig, ax = plt.subplots(figsize=(7, 6)); disp.plot(ax=ax, xticks_rotation=30, colorbar=False)
        ax.set_title(f"{mname} — split {sname} (out-of-fold, N={len(yt)})"); fig.tight_layout(); fig.savefig(OUT / f"cm_{key}.png", dpi=120); plt.close(fig)
        pd.DataFrame({"id": ids, "y_true": yt, "y_pred": yp}).merge(df[["id", "pertanyaan", "kelompok_intent"]], on="id") \
          .query("y_true != y_pred").to_csv(OUT / f"errors_{key}.csv", index=False)
        results[key] = {"split": sname, "model": mname, "macro_f1_mean": float(np.mean(scores)), "macro_f1_std": float(np.std(scores)),
                        "macro_f1_folds": [float(s) for s in scores], "accuracy_mean": float(np.mean(accs)),
                        "macro_f1_pooled_oof": float(f1_score(yt, yp, average="macro")),
                        "best_params_per_fold": params, "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()}}
        print(f"\n== {key}: macro F1 {np.mean(scores):.4f} ± {np.std(scores):.4f} | folds {np.round(scores,4).tolist()}"); print(rep)

pd.DataFrame(rows).to_csv(OUT / "results_folds.csv", index=False)
summary = pd.DataFrame([{k: v for k, v in r.items() if k in ("split", "model", "macro_f1_mean", "macro_f1_std", "accuracy_mean", "macro_f1_pooled_oof")} for r in results.values()])
summary.to_csv(OUT / "results_summary.csv", index=False); print(summary.to_string(index=False))

# Model terbaik dipilih dari split grouped (estimasi yang jujur), lalu dilatih ulang di seluruh data
best = max(["LogReg", "LinearSVC"], key=lambda m: results[f"grouped_{m}"]["macro_f1_mean"])
clf_f, g_clf = MODELS[best]
gs = GridSearchCV(make_pipe(clf_f()), {**grid_tfidf, **g_clf}, scoring="f1_macro", cv=StratifiedGroupKFold(5, shuffle=True, random_state=RS), n_jobs=-1)
gs.fit(X, y, groups=g)
joblib.dump(gs.best_estimator_, OUT / f"best_model_{best}.joblib")
versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__, "joblib": joblib.__version__}
json.dump({"data": str(DATA), "n": int(len(df)), "label_counts": df["label"].value_counts().to_dict(), "n_groups": int(df["kelompok_intent"].nunique()),
           "random_state": RS, "results": results, "best_model": best, "best_model_final_params": {k: str(v) for k, v in gs.best_params_.items()},
           "best_model_final_inner_cv_f1": float(gs.best_score_), "versions": versions}, open(OUT / "results.json", "w"), indent=2, ensure_ascii=False)
print("\nBEST (grouped):", best, gs.best_params_, "| versions:", versions)
