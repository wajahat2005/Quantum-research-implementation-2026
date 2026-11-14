# %% [markdown]
# # 01 — Upama et al. (2026): causal (DML) feature ranking → SVM / RF / QSVM
#
# **Method (ML with Applications):** rank features by a double-machine-learning (LinearDML)
# treatment effect on the outcome, keep the top k (10 on Sylhet, 7 on Pima), then train a linear
# SVM, a random forest and a QSVM. The paper ran the DML ranking on **all rows before** a random
# 80/20 split (Sylhet).
#
# DML here is the partialling-out estimator written out by hand (no `econml` needed): for each
# feature T, residualise T and y on the other features with 3-fold cross-fitted ridge models;
# effect = Σ r_T r_y / Σ r_T²; features ranked by |effect × SD(T)|.
#
# **Protocols**
# * `paper` — DML ranking + scaling on all rows, then 80/20 stratified split (seed 42), ungrouped.
# * `clean` — ranking and scaling inside each training fold; grouped/stratified 5-fold × seeds {0, 1, 42}.

# %%
import os, sys
from pathlib import Path
for p in [Path.cwd(), Path.cwd() / "notebooks2", Path("/content/notebooks2"), *Path("/kaggle/input").glob("*"), *Path("/kaggle/input").glob("*/notebooks2")]:
    if (p / "nb2_common.py").exists():
        sys.path.insert(0, str(p)); break
from nb2_common import *
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
sanity_check_kernels()
DATA = load_all()

# %%
TOP_K = {"pima": 7, "sylhet": 10, "cdc": 10, "hosp130": 10}


def dml_rank(X, y, seed=0):
    """Return feature indices sorted by |DML effect x SD|, largest first. X already imputed+scaled."""
    d = X.shape[1]
    eff = np.zeros(d)
    kf = KFold(3, shuffle=True, random_state=seed)
    for j in range(d):
        W = np.delete(X, j, axis=1); T = X[:, j]
        rT, ry = np.zeros(len(y)), np.zeros(len(y))
        for a, b in kf.split(X):
            rT[b] = T[b] - Ridge(alpha=1.0).fit(W[a], T[a]).predict(W[b])
            ry[b] = y[b] - Ridge(alpha=1.0).fit(W[a], y[a]).predict(W[b])
        eff[j] = (rT @ ry) / max(rT @ rT, 1e-12) * np.std(T)
    return np.argsort(-np.abs(eff))


def make_method(dataset):
    k = TOP_K[dataset]

    def upama_method(Xtr, ytr, Xte, seed, groups_tr, X_prep):
        Sp, Sr, Se = impute_scale(X_prep, Xtr, Xte)
        # ranking uses the rows preprocessing is fitted on: all rows (paper) or train fold (clean)
        y_prep = Y_ALL[dataset] if len(X_prep) != len(Xtr) else ytr
        keep = dml_rank(Sp, y_prep, seed)[:k]
        RANKED.append(dict(dataset=dataset, leak=len(X_prep) != len(Xtr), top=tuple(sorted(keep))))
        Sr, Se = Sr[:, keep], Se[:, keep]
        out = {}
        svm = SVC(kernel="linear", C=1.0, random_state=seed).fit(Sr, ytr)
        out[f"Linear SVM (DML top-{k})"] = (svm.predict(Se), svm.decision_function(Se))
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(Sr, ytr)
        out[f"RF (DML top-{k})"] = (rf.predict(Se), rf.predict_proba(Se)[:, 1])
        Qr, Qe = to_qubits(Sr, Se, seed=seed)  # q = min(k, 8); fitted on the training rows
        out[f"Angle-QSVM (DML top-{k})"] = svm_precomputed(angle_kernel(Qr, Qr), ytr, angle_kernel(Qe, Qr), seed=seed)
        # reference without causal selection (all features)
        rf_all = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        A_r, A_e = prep(impute_scale, X_prep, Xtr, Xte)
        rf_all.fit(A_r, ytr)
        out["RF (all features)"] = (rf_all.predict(A_e), rf_all.predict_proba(A_e)[:, 1])
        return out
    return upama_method


RANKED = []
Y_ALL = {n: DATA[n]["y"] for n in DATASETS}
rec = Recorder("P01_Upama", "01_upama", config=dict(
    method="hand-written DML partialling-out (ridge nuisances, 3-fold cross-fit) top-k -> linear SVM, RF, Angle-QSVM",
    top_k=TOP_K, paper_protocol="DML + scaling on all rows, 80/20 stratified split seed 42, ungrouped"))
print("paper protocol")
for n in DATASETS:
    run_paper_holdout(rec, DATA, make_method(n), test_size=0.2, global_prep=True, datasets=[n])
print("clean protocol")
for n in DATASETS:
    run_clean(rec, DATA, make_method(n), datasets=[n])
df = rec.save()

# %%
pd.set_option("display.width", 220, "display.max_rows", 200)
summarise(df)

# %% [markdown]
# ## Stability of the causal ranking: distinct top-k sets chosen (paper = 1 run; clean = 15 folds)

# %%
r = pd.DataFrame(RANKED)
r["features"] = [", ".join(DATA[d]["feature_names"][i] for i in t) for d, t in zip(r.dataset, r.top)]
r.groupby(["dataset", "leak"])["features"].agg(lambda s: s.value_counts().to_dict())
