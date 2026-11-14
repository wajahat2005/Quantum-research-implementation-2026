import os, sys, json, glob
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# Load all CSVs
results_dir = Path("notebooks2/results")
csv_files = sorted(results_dir.glob("0*.csv"))
dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)
all_df = pd.concat(dfs, ignore_index=True)

# Standardize model names
# In all_df, paper is P01_Upama, etc.
# Let's create full_model_name: paper_prefix + ": " + model
paper_map = {
    "P01_Upama": "Upama",
    "P02_Young": "Young",
    "P03_Kotha": "Kotha",
    "P04_Zhang": "Zhang",
    "P05_Kiran": "Kiran",
    "P06_Sodar": "Sodar",
    "P07_Yadav": "Yadav",
    "P08_Daga": "Daga"
}
all_df["paper_short"] = all_df["paper"].map(paper_map)
all_df["full_model"] = all_df["paper_short"] + ": " + all_df["model"]

clean_df = all_df[all_df["protocol"] == "clean"].copy()

print("Clean df total rows:", len(clean_df))
print("Datasets:", clean_df["dataset"].unique())
print("Seeds:", clean_df["seed"].unique())
print("Folds per seed:", clean_df["fold"].unique())

# Let's inspect Daga Stacking vs RBF-SVM
daga_clean = clean_df[clean_df["paper"] == "P08_Daga"]
print("\nDaga models in clean:", daga_clean["model"].unique())

stacking = daga_clean[daga_clean["model"] == "OOF-Stacking (LR meta)"]
rbf = daga_clean[daga_clean["model"] == "RBF-SVM"]

merged = pd.merge(stacking, rbf, on=["dataset", "seed", "fold"], suffixes=("_stacking", "_rbf"))
print("\nMerged Stacking vs RBF rows:", len(merged))

# Check difference across all datasets and folds
for ds in ["pima", "sylhet", "cdc", "hosp130"]:
    sub = merged[merged["dataset"] == ds]
    diff_ba = sub["balanced_accuracy_stacking"] - sub["balanced_accuracy_rbf"]
    diff_acc = sub["accuracy_stacking"] - sub["accuracy_rbf"]
    diff_rec = sub["recall_stacking"] - sub["recall_rbf"]
    diff_spec = sub["specificity_stacking"] - sub["specificity_rbf"]
    print(f"\n--- Dataset: {ds} (n={len(sub)}) ---")
    print("Diff Balanced Accuracy (stacking - rbf):")
    print(list(np.round(diff_ba.values, 6)))
    print("Diff Accuracy:")
    print(list(np.round(diff_acc.values, 6)))
    print("Diff Recall:")
    print(list(np.round(diff_rec.values, 6)))
    print("Diff Specificity:")
    print(list(np.round(diff_spec.values, 6)))
    
    # Check confusion matrix identical count
    same_cm = (sub["tn_stacking"] == sub["tn_rbf"]) & (sub["fp_stacking"] == sub["fp_rbf"]) & \
              (sub["fn_stacking"] == sub["fn_rbf"]) & (sub["tp_stacking"] == sub["tp_rbf"])
    print(f"Folds with IDENTICAL confusion matrix: {same_cm.sum()} / {len(sub)}")

