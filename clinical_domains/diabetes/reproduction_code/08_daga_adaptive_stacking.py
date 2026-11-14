# %% [markdown]
# # 08 — Daga et al. (2026): adaptive quantum-kernel stacking, on all 4 diabetes datasets
#
# **Method (Sci. Rep. 16:26840):** QSVMs with Angle, ZZ and Amplitude fidelity kernels, a classical
# RBF-SVM, hard voting, and an out-of-fold (OOF) stacked meta-learner (logistic regression) over the
# kernel models' decision scores. Original data: Pima only.
#
# **Protocols**
# * `paper` — the released script's protocol: scaling fitted on **all rows** (global), ungrouped
#   stratified 5-fold CV (seed 42); pooled OOF predictions get a bootstrap CI (as in Table 9).
# * `clean` — leakage-free: everything fitted inside the outer training fold; grouped CV when
#   feature vectors repeat; seeds {0, 1, 42}. Stacking meta-features come from an inner 5-fold CV
#   on the training fold only.
#
# Quantum kernels are exact noiseless simulations (`nb2_common.py`); q = min(d, 8) qubits, PCA
# fitted in-fold when d > 8. Positive class per dataset is printed by `load_all()`.

# %%
import os, sys
from pathlib import Path
for p in [Path.cwd(), Path.cwd() / "notebooks2", Path("/content/notebooks2"), *Path("/kaggle/input").glob("*"), *Path("/kaggle/input").glob("*/notebooks2")]:
    if (p / "nb2_common.py").exists():
        sys.path.insert(0, str(p)); break
from nb2_common import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
sanity_check_kernels()
DATA = load_all()

# %%
KERNEL_MODELS = ["Angle-QSVM", "ZZ-QSVM", "Amplitude-QSVM", "RBF-SVM"]


def kernel_blocks(Xq_a, Xq_b, Xs_a, Xs_b):
    """Gram blocks between row sets a and b for the three quantum kernels."""
    return {"Angle-QSVM": angle_kernel(Xq_a, Xq_b),
            "ZZ-QSVM": zz_kernel(Xq_a, Xq_b, reps=2),
            "Amplitude-QSVM": amplitude_kernel(Xs_a, Xs_b)}


def fit_scores(Ktr, Kte, Xs_tr, Xs_te, ytr, seed):
    """Fit the 4 kernel SVMs; return {model: (pred, decision score)} on the test block."""
    out = {m: svm_precomputed(Ktr[m], ytr, Kte[m], C=1.0, seed=seed) for m in Ktr}
    rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed).fit(Xs_tr, ytr)
    out["RBF-SVM"] = (rbf.predict(Xs_te), rbf.decision_function(Xs_te))
    return out


def daga_method(Xtr, ytr, Xte, seed, groups_tr, X_prep):
    Xq_tr, Xq_te = prep(to_qubits, X_prep, Xtr, Xte, seed=seed)
    Xs_tr, Xs_te = prep(impute_scale, X_prep, Xtr, Xte)
    K_trtr = kernel_blocks(Xq_tr, Xq_tr, Xs_tr, Xs_tr)
    K_tetr = kernel_blocks(Xq_te, Xq_tr, Xs_te, Xs_tr)
    out = fit_scores(K_trtr, K_tetr, Xs_tr, Xs_te, ytr, seed)

    # OOF stacking: meta-features = inner-CV decision scores of the 4 kernel models (train fold only)
    meta_tr = np.zeros((len(ytr), len(KERNEL_MODELS)))
    for itr, iva in inner_splits(ytr, groups_tr, seed, n_splits=5):
        Ka = {m: K_trtr[m][np.ix_(itr, itr)] for m in K_trtr}
        Kb = {m: K_trtr[m][np.ix_(iva, itr)] for m in K_trtr}
        o = fit_scores(Ka, Kb, Xs_tr[itr], Xs_tr[iva], ytr[itr], seed)
        meta_tr[iva] = np.column_stack([o[m][1] for m in KERNEL_MODELS])
    meta_te = np.column_stack([out[m][1] for m in KERNEL_MODELS])
    meta = LogisticRegression(max_iter=1000).fit(meta_tr, ytr)
    out["OOF-Stacking (LR meta)"] = (meta.predict(meta_te), meta.predict_proba(meta_te)[:, 1])

    votes = np.column_stack([out[m][0] for m in KERNEL_MODELS])
    out["Hard-Voting (4 kernels)"] = ((votes.mean(1) > 0.5).astype(int), votes.mean(1))

    lr = LogisticRegression(max_iter=1000).fit(Xs_tr, ytr)
    out["Logistic Regression"] = (lr.predict(Xs_te), lr.predict_proba(Xs_te)[:, 1])
    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1).fit(Xs_tr, ytr)
    out["Random Forest"] = (rf.predict(Xs_te), rf.predict_proba(Xs_te)[:, 1])
    out["Majority class"] = (np.full(len(Xte), int(ytr.mean() >= 0.5)), np.zeros(len(Xte)))
    return out


# %%
rec = Recorder("P08_Daga", "08_daga", config=dict(
    method="Angle/ZZ(reps2,linear)/Amplitude QSVM C=1, RBF-SVM C=1, inner-5-fold OOF LR stacking, hard voting",
    paper_protocol="global scaling + ungrouped stratified 5-fold seed 42"))
print("paper protocol"); run_paper_cv(rec, DATA, daga_method, global_prep=True)
print("clean protocol"); run_clean(rec, DATA, daga_method)
df = rec.save()

# %% [markdown]
# ## Results: mean ± SD over folds (and seeds for `clean`)

# %%
pd.set_option("display.width", 220, "display.max_rows", 200)
summarise(df)

# %% [markdown]
# ## Pooled out-of-fold accuracy and recall (paper vs clean), stacking vs RBF vs Angle
# Bootstrap CIs and significance tests are computed in `09_summary.ipynb`.
# Hard voting over 4 models counts a 2–2 tie as negative.

# %%
rows = []
for (dset, proto, seed, model), g in df.groupby(["dataset", "protocol", "seed", "model"]):
    if model not in ("OOF-Stacking (LR meta)", "RBF-SVM", "Angle-QSVM"):
        continue
    # rebuild pooled confusion counts -> accuracy CI via per-fold sums
    acc = (g.tp.sum() + g.tn.sum()) / g.n_test.sum()
    rec_ = g.tp.sum() / max(g.tp.sum() + g.fn.sum(), 1)
    rows.append(dict(dataset=dset, protocol=proto, seed=seed, model=model, pooled_acc=round(acc, 4),
                     pooled_recall=round(rec_, 4)))
pd.DataFrame(rows).pivot_table(index=["dataset", "model"], columns="protocol",
                               values=["pooled_acc", "pooled_recall"], aggfunc="mean").round(4)
