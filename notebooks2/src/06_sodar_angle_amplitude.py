# %% [markdown]
# # 06 — Sodar et al. (2026): Angle vs Amplitude QSVM, transferred to 4 diabetes datasets
#
# **Method (EPJ Quantum Technol.):** QSVM with angle encoding and with amplitude encoding,
# originally on URL-traffic data (70/30 split, N = 3,000). That data is not diabetes, so only the
# *method* is transferred here.
#
# **Protocols**
# * `paper` — 70/30 random stratified split (seed 42), ungrouped, scaling fitted on all rows.
# * `clean` — leakage-free grouped/stratified 5-fold × seeds {0, 1, 42}, in-fold preprocessing.
#
# Amplitude encoding is run on two inputs: MinMax [0, π] features (as in the paper — every vector
# is non-negative, so the kernel only sees direction) and standardised features (the repair used in
# the audit). RBF-SVM and LR are the classical references.

# %%
import os, sys
from pathlib import Path
for p in [Path.cwd(), Path.cwd() / "notebooks2", Path("/content/notebooks2"), *Path("/kaggle/input").glob("*"), *Path("/kaggle/input").glob("*/notebooks2")]:
    if (p / "nb2_common.py").exists():
        sys.path.insert(0, str(p)); break
from nb2_common import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
sanity_check_kernels()
DATA = load_all()

# %%
def sodar_method(Xtr, ytr, Xte, seed, groups_tr, X_prep):
    Xq_tr, Xq_te = prep(to_qubits, X_prep, Xtr, Xte, seed=seed)
    Xs_tr, Xs_te = prep(impute_scale, X_prep, Xtr, Xte)
    Xm_tr, Xm_te = prep(impute_scale, X_prep, Xtr, Xte, kind="minmax")
    out = {
        "Angle-QSVM": svm_precomputed(angle_kernel(Xq_tr, Xq_tr), ytr, angle_kernel(Xq_te, Xq_tr), seed=seed),
        "Amplitude-QSVM (MinMax, paper)": svm_precomputed(amplitude_kernel(Xm_tr, Xm_tr), ytr,
                                                          amplitude_kernel(Xm_te, Xm_tr), seed=seed),
        "Amplitude-QSVM (standardised)": svm_precomputed(amplitude_kernel(Xs_tr, Xs_tr), ytr,
                                                         amplitude_kernel(Xs_te, Xs_tr), seed=seed),
    }
    rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed).fit(Xs_tr, ytr)
    out["RBF-SVM"] = (rbf.predict(Xs_te), rbf.decision_function(Xs_te))
    lr = LogisticRegression(max_iter=1000).fit(Xs_tr, ytr)
    out["Logistic Regression"] = (lr.predict(Xs_te), lr.predict_proba(Xs_te)[:, 1])
    out["Majority class"] = (np.full(len(Xte), int(ytr.mean() >= 0.5)), np.zeros(len(Xte)))
    return out


rec = Recorder("P06_Sodar", "06_sodar", config=dict(
    method="Angle-QSVM, Amplitude-QSVM on MinMax and on standardised input, RBF-SVM, LR",
    paper_protocol="70/30 stratified hold-out, seed 42, global scaling"))
print("paper protocol"); run_paper_holdout(rec, DATA, sodar_method, test_size=0.3, global_prep=True)
print("clean protocol"); run_clean(rec, DATA, sodar_method)
df = rec.save()

# %%
pd.set_option("display.width", 220, "display.max_rows", 200)
summarise(df)

# %% [markdown]
# ## Does Amplitude (MinMax) collapse to one class? (`n_pred_pos` = predicted positives per fold)

# %%
df.groupby(["dataset", "protocol", "model"])[["n_pred_pos", "n_pos_test", "recall", "specificity"]].mean().round(3)
