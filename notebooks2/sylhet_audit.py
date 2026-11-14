import os, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

# Load Sylhet
p = Path("data/sylhet/early_stage_diabetes_uci529.csv")
df = pd.read_csv(p)
print("Sylhet shape:", df.shape)

# Target
y = (df["class"].str.strip().str.lower() == "positive").astype(int).values
X = df.drop(columns=["class"]).copy()
X["gender"] = (X["gender"].str.strip().str.lower() == "male").astype(int)
for c in X.columns:
    if X[c].dtype == object:
        X[c] = (X[c].str.strip().str.lower() == "yes").astype(int)

# Create unique profile key
keys = pd.DataFrame(X).astype(str).agg("|".join, axis=1)
groups, uniques = pd.factorize(keys)
n_unique = len(uniques)
n_total = len(df)
n_redundant = n_total - n_unique
print(f"Total rows: {n_total}, Unique profiles: {n_unique}, Redundant rows: {n_redundant}")

# Profile frequency breakdown
counts = pd.Series(groups).value_counts()
print("\nProfile frequency distribution:")
print(counts.value_counts().sort_index())

# Check label consistency within duplicate profiles
df_grouped = pd.DataFrame({"group": groups, "y": y})
grouped_y = df_grouped.groupby("group")["y"].nunique()
inconsistent_groups = grouped_y[grouped_y > 1]
print(f"\nGroups with inconsistent labels: {len(inconsistent_groups)}")

# Profile table: breakdown of top duplicate profiles
top_profiles = counts.head(10)
print("\nTop 10 most frequent profiles:")
for grp, cnt in top_profiles.items():
    grp_rows = df[groups == grp]
    label_dist = grp_rows["class"].value_counts().to_dict()
    print(f"Group ID {grp}: Count = {cnt}, Labels = {label_dist}, Age = {grp_rows['Age'].iloc[0]}, Gender = {grp_rows['gender'].iloc[0]}")

# Simulation of CV splits
seeds = [0, 1, 42]
print("\n=== UNGROUPED vs GROUPED CV SPLITS ===")
for seed in seeds:
    print(f"\n--- SEED {seed} ---")
    # Ungrouped StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    print("Ungrouped StratifiedKFold:")
    for f_idx, (tr, te) in enumerate(skf.split(X, y)):
        g_tr = set(groups[tr])
        leaks = sum(groups[i] in g_tr for i in te)
        pct = (leaks / len(te)) * 100
        print(f"  Fold {f_idx}: Test size={len(te)}, Pos={y[te].sum()}, Neg={len(te)-y[te].sum()}, Leaks={leaks} ({pct:.1f}%)")

    # Grouped StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    print("Grouped StratifiedGroupKFold:")
    for f_idx, (tr, te) in enumerate(sgkf.split(X, y, groups=groups)):
        g_tr = set(groups[tr])
        leaks = sum(groups[i] in g_tr for i in te)
        n_grps_te = len(set(groups[te]))
        pct = (leaks / len(te)) * 100
        print(f"  Fold {f_idx}: Test size={len(te)}, Unique Groups={n_grps_te}, Pos={y[te].sum()}, Neg={len(te)-y[te].sum()}, Leaks={leaks} ({pct:.1f}%)")
