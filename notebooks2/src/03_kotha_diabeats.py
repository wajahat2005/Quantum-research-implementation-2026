# %% [markdown]
# # 03 — Kotha et al. (2026) "DiaBeats": random forest, single split vs cross-validation
#
# **Method (IEEE I3CTCON):** a random forest on the raw clinical features, reported from one
# 80/20 test split (Pima). Here the same model runs on all 4 diabetes datasets.
#
# **Protocols**
# * `paper` — one 80/20 stratified split (seed 42), ungrouped, no scaling (RF on raw features;
#   NaNs median-imputed on all rows).
# * `paper_10splits` — the same single-split recipe repeated with 10 different split seeds, to show
#   how much a single-split number can move.
# * `clean` — grouped/stratified 5-fold × seeds {0, 1, 42}, in-fold imputation.
#
# A threshold-tuned variant (threshold chosen on the training fold by inner CV to maximise
# balanced accuracy) is added in `clean` only.

# %%
import os, sys
from pathlib import Path
for p in [Path.cwd(), Path.cwd() / "notebooks2", Path("/content/notebooks2"), *Path("/kaggle/input").glob("*"), *Path("/kaggle/input").glob("*/notebooks2")]:
    if (p / "nb2_common.py").exists():
        sys.path.insert(0, str(p)); break
from nb2_common import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
sanity_check_kernels()
DATA = load_all()

# %%
def impute_only(X_fit, *others):
    imp = SimpleImputer(strategy="median").fit(X_fit)
    return [imp.transform(X_fit)] + [imp.transform(o) for o in others]


def make_method(tune_threshold):
    def kotha_method(Xtr, ytr, Xte, seed, groups_tr, X_prep):
        Ar, Ae = prep(impute_only, X_prep, Xtr, Xte)
        rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1).fit(Ar, ytr)
        p = rf.predict_proba(Ae)[:, 1]
        out = {"DiaBeats RF": ((p >= 0.5).astype(int), p)}
        if tune_threshold:
            folds = inner_splits(ytr, groups_tr, seed, n_splits=3)
            oof = cross_val_predict(RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
                                    Ar, ytr, cv=folds, method="predict_proba")[:, 1]
            grid = np.linspace(0.05, 0.95, 19)
            t = grid[np.argmax([balanced_accuracy_score(ytr, (oof >= g).astype(int)) for g in grid])]
            out["DiaBeats RF (threshold by inner CV)"] = ((p >= t).astype(int), p)
        Sr, Se = prep(impute_scale, X_prep, Xtr, Xte)
        lr = LogisticRegression(max_iter=1000).fit(Sr, ytr)
        out["Logistic Regression"] = (lr.predict(Se), lr.predict_proba(Se)[:, 1])
        return out
    return kotha_method


rec = Recorder("P03_Kotha", "03_kotha", config=dict(
    method="RF n_estimators=100 on raw features; LR reference; inner-CV threshold variant (clean only)",
    paper_protocol="single 80/20 stratified split seed 42; paper_10splits = seeds 0..9"))
print("paper protocol"); run_paper_holdout(rec, DATA, make_method(False), test_size=0.2, global_prep=True)
for s in range(10):
    run_paper_holdout(rec, DATA, make_method(False), test_size=0.2, global_prep=True, seed=s,
                      protocol="paper_10splits")
print("clean protocol"); run_clean(rec, DATA, make_method(True))
df = rec.save()

# %%
pd.set_option("display.width", 220, "display.max_rows", 200)
summarise(df)

# %% [markdown]
# ## Spread of the single-split accuracy across 10 split seeds (min / max)

# %%
df[df.protocol == "paper_10splits"].groupby(["dataset", "model"])["accuracy"].agg(["min", "max", "mean", "std"]).round(3)
