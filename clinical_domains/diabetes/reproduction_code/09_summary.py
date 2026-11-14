# %% [markdown]
# # 09 — Summary: every paper's method × every diabetes dataset
#
# Reads `results/0[1-8]_*.csv` (run notebooks 01–08 first) and builds:
# 1. the main table (clean protocol): balanced accuracy, recall, specificity, AUC, with the
#    majority-class rate as a floor;
# 2. paper-protocol vs clean-protocol gap (how much each paper's own protocol inflates the number);
# 3. paired tests of every model vs the RBF-SVM of notebook 08 on the **same** clean folds
#    (Nadeau–Bengio corrected t-test on balanced accuracy, Holm correction within each dataset);
# 4. bootstrap 95 % CIs (B = 2000) for pooled paper-protocol accuracy;
# 5. a heatmap and `results/summary.md`.
#
# Clean folds are identical across notebooks (same `clean_splits(dataset, seed)`), which is what
# makes the fold-paired tests valid. Zhang (04) rows enter at noise rate 0 only.

# %%
import os, sys
from pathlib import Path
for p in [Path.cwd(), Path.cwd() / "notebooks2", Path("/content/notebooks2"), *Path("/kaggle/input").glob("*"), *Path("/kaggle/input").glob("*/notebooks2")]:
    if (p / "nb2_common.py").exists():
        sys.path.insert(0, str(p)); break
from nb2_common import *
import matplotlib.pyplot as plt
RD = find_results_dir()
files = sorted(RD.glob("0[1-8]_*.csv"))
print("found:", [f.name for f in files])
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
if "noise_rate" in df:
    df = df[df.noise_rate.isna() | (df.noise_rate == 0)]
df["paper_model"] = df.paper.str.replace(r"^P0\d_", "", regex=True) + ": " + df.model
DATA = load_all()
FLOOR = {n: DATA[n]["majority_rate"] for n in DATASETS}

# %% [markdown]
# ## 1. Main table — clean protocol (mean over 3 seeds × 5 folds)

# %%
pd.set_option("display.width", 250, "display.max_rows", 300, "display.max_colwidth", 70)
clean = df[df.protocol == "clean"]
main = clean.groupby(["dataset", "paper_model"])[["balanced_accuracy", "recall", "specificity", "roc_auc", "accuracy"]].mean()
main["majority_floor"] = [FLOOR[d] for d in main.index.get_level_values(0)]
main = main.round(3).sort_values(["dataset", "balanced_accuracy"], ascending=[True, False])
main

# %% [markdown]
# ## 2. Paper protocol vs clean protocol (accuracy; positive gap = paper protocol looks better)

# %%
proto = df[df.protocol.isin(["paper", "clean"])].groupby(["dataset", "paper_model", "protocol"])["accuracy"].mean().unstack()
proto = proto.dropna()
proto["gap (paper - clean)"] = proto["paper"] - proto["clean"]
proto.round(3).sort_values("gap (paper - clean)", ascending=False)

# %% [markdown]
# ## 3. Fold-paired tests vs RBF-SVM (notebook 08), balanced accuracy, Holm within dataset

# %%
ref = clean[(clean.paper == "P08_Daga") & (clean.model == "RBF-SVM")].set_index(["dataset", "seed", "fold"])["balanced_accuracy"]
tests = []
for (dset, pm), g in clean.groupby(["dataset", "paper_model"]):
    if pm == "Daga: RBF-SVM":
        continue
    g = g.set_index(["dataset", "seed", "fold"])["balanced_accuracy"]
    g = g[~g.index.duplicated()]
    common = g.index.intersection(ref.index)
    d = (g.loc[common] - ref.loc[common]).values
    n = len(DATA[dset]["y"]); n_te = n / N_SPLITS
    t, p = nadeau_bengio(d, n - n_te, n_te)
    tests.append(dict(dataset=dset, model=pm, n_pairs=len(d), mean_diff=np.mean(d), t=t, p=p))
tests = pd.DataFrame(tests)
tests["p_holm"] = np.nan
for dset, g in tests.groupby("dataset"):
    tests.loc[g.index, "p_holm"] = holm(g.p.values)
tests["verdict"] = np.where(tests.p_holm < 0.05, np.where(tests.mean_diff > 0, "better", "worse"), "n.s.")
tests.round(4).sort_values(["dataset", "mean_diff"], ascending=[True, False])

# %% [markdown]
# ## 4. Bootstrap 95 % CI of pooled paper-protocol accuracy (rows rebuilt from confusion counts)

# %%
ci = []
rng = np.random.RandomState(42)
for (dset, pm), g in df[df.protocol == "paper"].groupby(["dataset", "paper_model"]):
    tp, tn, fp, fn = (int(g[c].sum()) for c in ["tp", "tn", "fp", "fn"])
    y = np.r_[np.ones(tp + fn), np.zeros(tn + fp)].astype(int)
    pr = np.r_[np.ones(tp), np.zeros(fn), np.zeros(tn), np.ones(fp)].astype(int)
    acc = (y == pr).mean()
    boots = [(y[i] == pr[i]).mean() for i in (rng.randint(0, len(y), len(y)) for _ in range(2000))]
    ci.append(dict(dataset=dset, model=pm, n=len(y), accuracy=acc,
                   ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5)))
ci = pd.DataFrame(ci).round(4)
ci

# %% [markdown]
# ## 5. Heatmap — clean balanced accuracy

# %%
hm = clean.groupby(["paper_model", "dataset"])["balanced_accuracy"].mean().unstack()[DATASETS]
fig, ax = plt.subplots(figsize=(8, 0.28 * len(hm) + 1.5))
im = ax.imshow(hm.values, cmap="viridis", aspect="auto", vmin=0.5, vmax=1.0)
ax.set_xticks(range(len(DATASETS)), DATASETS); ax.set_yticks(range(len(hm)), hm.index, fontsize=7)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w" if v < 0.75 else "k")
plt.colorbar(im, label="balanced accuracy (clean)"); plt.title("Paper method × diabetes dataset")
plt.tight_layout(); plt.savefig(RD / "09_heatmap_balanced_accuracy.png", dpi=160); plt.show()

# %% [markdown]
# ## 6. Write `results/summary.md` (all numbers come from the tables above)

# %%
def md(t):
    try:
        return t.to_markdown()
    except ImportError:
        return "```\n" + t.to_string() + "\n```"

with open(RD / "summary.md", "w", encoding="utf-8") as f:
    f.write("# notebooks2 summary — paper methods × diabetes datasets\n\n")
    f.write("Positive class: " + "; ".join(f"{n}: {DATA[n]['positive']}" for n in DATASETS) + "\n\n")
    f.write("## Clean protocol (mean over seeds × folds)\n\n" + md(main) + "\n\n")
    f.write("## Paper vs clean protocol (accuracy)\n\n" + md(proto.round(3)) + "\n\n")
    f.write("## Paired tests vs RBF-SVM (balanced accuracy, Nadeau-Bengio, Holm per dataset)\n\n" + md(tests.round(4)) + "\n\n")
    f.write("## Paper-protocol pooled accuracy, bootstrap 95 % CI\n\n" + md(ci) + "\n")
print("wrote", RD / "summary.md")
