# %% [markdown]
# # 07 — Yadav et al. (2026): Z-map vs ZZ-map QSVM, on 4 diabetes datasets
#
# **Method (PeerJ CS):** QSVC with the unentangled Z feature map and the entangling ZZ feature map
# (reps = 2), 80/20 split. Their benchmark was WDBC (breast cancer, out of scope here); only the
# method is transferred. The hybrid QNN head is not repeated (notebook 05 covers VQCs).
#
# **Protocols**
# * `paper` — 80/20 stratified hold-out (seed 42), ungrouped, scaling on all rows, bandwidth 1.
# * `clean` — grouped/stratified 5-fold × seeds {0, 1, 42}; models at bandwidth 1 **and** with the
#   bandwidth chosen by inner 3-fold CV (balanced accuracy) on the training fold from
#   {0.1, 0.25, 0.5, 1.0}. The outer test fold is never used for the choice.

# %%
import os, sys
from pathlib import Path
for p in [Path.cwd(), Path.cwd() / "notebooks2", Path("/content/notebooks2"), *Path("/kaggle/input").glob("*"), *Path("/kaggle/input").glob("*/notebooks2")]:
    if (p / "nb2_common.py").exists():
        sys.path.insert(0, str(p)); break
from nb2_common import *
from sklearn.svm import SVC
sanity_check_kernels()
DATA = load_all()

# %%
SCALES = [0.1, 0.25, 0.5, 1.0]
MAPS = {
    "Z-QSVM": lambda X, s: pauli_states(X, reps=2, scale=s, pairs=False),
    "ZZ-QSVM (linear)": lambda X, s: pauli_states(X, reps=2, scale=s, entangle="linear"),
    "ZZ-QSVM (full)": lambda X, s: pauli_states(X, reps=2, scale=s, entangle="full"),
}


def pick_scale(fmap, Xq, y, groups, seed):
    best, best_s = -1, 1.0
    for s in SCALES:
        K = state_kernel(fmap(Xq, s), fmap(Xq, s))
        scores = []
        for a, b in inner_splits(y, groups, seed, n_splits=3):
            p, _ = svm_precomputed(K[np.ix_(a, a)], y[a], K[np.ix_(b, a)], seed=seed)
            scores.append(balanced_accuracy_score(y[b], p))
        if np.mean(scores) > best:
            best, best_s = np.mean(scores), s
    return best_s


def make_method(tune):
    def yadav_method(Xtr, ytr, Xte, seed, groups_tr, X_prep):
        Xq_tr, Xq_te = prep(to_qubits, X_prep, Xtr, Xte, seed=seed)
        out = {}
        for name, fmap in MAPS.items():
            scales = [1.0] + ([pick_scale(fmap, Xq_tr, ytr, groups_tr, seed)] if tune else [])
            for j, s in enumerate(scales):
                Ptr, Pte = fmap(Xq_tr, s), fmap(Xq_te, s)
                label = name + (" bw=1" if j == 0 else " bw=inner-CV")
                out[label] = svm_precomputed(state_kernel(Ptr, Ptr), ytr, state_kernel(Pte, Ptr), seed=seed)
                if j == 1:
                    CHOSEN.append(dict(map=name, scale=s))
        Xs_tr, Xs_te = prep(impute_scale, X_prep, Xtr, Xte)
        rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed).fit(Xs_tr, ytr)
        out["RBF-SVM"] = (rbf.predict(Xs_te), rbf.decision_function(Xs_te))
        return out
    return yadav_method


CHOSEN = []
rec = Recorder("P07_Yadav", "07_yadav", config=dict(
    method="Z / ZZ-linear / ZZ-full feature maps reps=2, QSVC C=1; bandwidth grid " + str(SCALES),
    paper_protocol="80/20 stratified hold-out seed 42, global scaling, bandwidth 1"))
print("paper protocol"); run_paper_holdout(rec, DATA, make_method(tune=False), test_size=0.2, global_prep=True)
print("clean protocol"); run_clean(rec, DATA, make_method(tune=True))
df = rec.save()

# %%
pd.set_option("display.width", 220, "display.max_rows", 200)
summarise(df)

# %% [markdown]
# ## Bandwidths picked by inner CV (counts over datasets × seeds × folds)

# %%
pd.DataFrame(CHOSEN).value_counts().rename("times chosen").reset_index()
