"""Analisis ambang keyakinan: probabilitas out-of-fold (split grouped, 5 fold, random_state=42)
dengan hiperparameter model final (TF-IDF ngram (1,2), sublinear_tf, C=10)."""
import json, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
import joblib
df = pd.read_csv("/workspace/capstone/data/dataset_classifier_v2_label_pp.csv")
X, y, g = df["pertanyaan"].to_numpy(), df["label"].to_numpy(), df["kelompok_intent"].to_numpy()
base = joblib.load("/workspace/capstone/eval/best_model_LogReg.joblib")
conf = np.zeros(len(y)); pred = np.empty(len(y), dtype=object)
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, g):
    m = clone(base).fit(X[tr], y[tr]); p = m.predict_proba(X[te])
    conf[te] = p.max(1); pred[te] = m.classes_[p.argmax(1)]
ok = pred == y
out = {"n": int(len(y)), "acc_oof": float(ok.mean()),
       "conf_median_benar": float(np.median(conf[ok])), "conf_median_salah": float(np.median(conf[~ok])), "rows": []}
print(f"OOF acc={ok.mean():.4f}  median conf benar={np.median(conf[ok]):.3f} salah={np.median(conf[~ok]):.3f}")
print("thr  %flagged  acc_diatas  n_diatas  %salah_tertangkap")
for t in [0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7]:
    hi = conf >= t
    r = {"thr": t, "pct_flagged": float((~hi).mean()), "acc_diatas": float(ok[hi].mean()), "n_diatas": int(hi.sum()),
         "pct_salah_tertangkap": float((~ok & ~hi).sum() / max(1, (~ok).sum()))}
    out["rows"].append(r)
    print(f"{t:.2f}  {r['pct_flagged']*100:5.1f}%   {r['acc_diatas']:.4f}   {r['n_diatas']}   {r['pct_salah_tertangkap']*100:5.1f}%")
json.dump(out, open("test_output/threshold_analysis.json", "w"), indent=2)
