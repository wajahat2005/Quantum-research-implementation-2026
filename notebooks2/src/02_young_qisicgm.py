# %% [markdown]
# # 02 — Young (2025) QISICGM: phase feature map + RF/ET/MLP ensemble, GMM augmentation
#
# **Method (arXiv 2509.12259):** standardise, expand each feature x into (cos x, sin x)
# ("quantum-inspired" phase map), average the probabilities of a random forest, extra trees and an
# MLP. The paper added 2,000 GMM-synthetic diabetic rows to Pima's 768 **before** 5-fold CV, so
# synthetic rows also sat in the validation folds.
#
# **Protocols**
# * `paper` — synthetic positives (GMM, 5 components, fitted on all real positives) added to the
#   dataset before ungrouped stratified 5-fold CV (seed 42); scored on all validation rows,
#   synthetic included. Synthetic count = 2000/768 × N (capped at 5,000).
# * `paper_real_only` — same augmented training, but scored on real validation rows only.
# * `clean` — no synthetic rows in any test fold: GMM fitted on the training-fold positives only and
#   used to augment the training fold only; grouped/stratified 5-fold × seeds {0, 1, 42}.
#   The same ensemble without augmentation is also reported.

# %%
import os, sys
from pathlib import Path
for p in [Path.cwd(), Path.cwd() / "notebooks2", Path("/content/notebooks2"), *Path("/kaggle/input").glob("*"), *Path("/kaggle/input").glob("*/notebooks2")]:
    if (p / "nb2_common.py").exists():
        sys.path.insert(0, str(p)); break
from nb2_common import *
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
sanity_check_kernels()
DATA = load_all()

# %%
def phase_map(X):
    return np.hstack([np.cos(X), np.sin(X)])


def ensemble(Ztr, ytr, Zte, seed):
    P = [RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(Ztr, ytr).predict_proba(Zte)[:, 1],
         ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(Ztr, ytr).predict_proba(Zte)[:, 1],
         MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=seed).fit(Ztr, ytr).predict_proba(Zte)[:, 1]]
    p = np.mean(P, axis=0)
    return (p >= 0.5).astype(int), p


def n_synth(n):
    return int(min(5000, round(2000 / 768 * n)))


def gmm_positives(S_pos, n, seed):
    g = GaussianMixture(n_components=min(5, max(1, len(S_pos) // 10)), covariance_type="full",
                        reg_covar=1e-3, random_state=seed).fit(S_pos)
    return g.sample(n)[0]


def clean_method(Xtr, ytr, Xte, seed, groups_tr, X_prep):
    Sr, Se = prep(impute_scale, X_prep, Xtr, Xte)
    out = {"QISICGM ensemble (no augmentation)": ensemble(phase_map(Sr), ytr, phase_map(Se), seed)}
    syn = gmm_positives(Sr[ytr == 1], n_synth(len(ytr)), seed)
    Sa = np.vstack([Sr, syn]); ya = np.r_[ytr, np.ones(len(syn), int)]
    out["QISICGM ensemble (+GMM, train-fold only)"] = ensemble(phase_map(Sa), ya, phase_map(Se), seed)
    return out


def run_paper(rec, seed=PAPER_SEED):
    for name in DATASETS:
        ds = DATA[name]
        S = impute_scale(ds["X"])[0]                          # global scaling (paper)
        syn = gmm_positives(S[ds["y"] == 1], n_synth(len(ds["y"])), seed)
        SA = np.vstack([S, syn]); yA = np.r_[ds["y"], np.ones(len(syn), int)]
        is_real = np.r_[np.ones(len(S), bool), np.zeros(len(syn), bool)]
        for k, (tr, te) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(SA, yA)):
            pred, p = ensemble(phase_map(SA[tr]), yA[tr], phase_map(SA[te]), seed)
            rec.add(name, "paper", seed, k, "QISICGM ensemble (+GMM, global)", yA[te], pred, p,
                    synthetic_in_test=int((~is_real[te]).sum()))
            r = is_real[te]
            rec.add(name, "paper_real_only", seed, k, "QISICGM ensemble (+GMM, global)", yA[te][r], pred[r], p[r])
        print(f"  paper  {name:8s} done ({len(syn)} synthetic positives)")


rec = Recorder("P02_Young", "02_young", config=dict(
    method="phase map [cos x, sin x] -> mean prob of RF(100), ET(100), MLP(32,16); GMM(<=5 comp) synthetic positives",
    synthetic_ratio="2000/768 x N, cap 5000"))
print("paper protocol"); run_paper(rec)
print("clean protocol"); run_clean(rec, DATA, clean_method)
df = rec.save()

# %%
pd.set_option("display.width", 220, "display.max_rows", 200)
summarise(df, ("accuracy", "f1", "recall", "specificity", "roc_auc"))
