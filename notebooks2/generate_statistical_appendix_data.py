import os, sys, json, glob
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

results_dir = Path("notebooks2/results")
data_dir = Path("data")

# 1. Config audit
print("--- 1. CONFIG AUDIT ---")
configs = {}
for p in sorted(results_dir.glob("*_config.json")):
    with open(p, "r") as f:
        configs[p.name] = json.load(f)

config_rows = []
for name, cfg in configs.items():
    config_rows.append({
        "config_file": name,
        "paper": cfg.get("paper"),
        "clean_seeds": str(cfg.get("clean_seeds")),
        "paper_seed": cfg.get("paper_seed"),
        "n_splits": cfg.get("n_splits"),
        "subsample_n": cfg.get("subsample_n"),
        "subsample_seed": cfg.get("subsample_seed"),
        "max_qubits": cfg.get("max_qubits"),
        "runtime_s": cfg.get("runtime_s"),
        "n_rows": cfg.get("n_rows"),
        "python": cfg.get("python"),
        "numpy": cfg.get("numpy"),
        "sklearn": cfg.get("sklearn"),
        "platform": cfg.get("platform"),
        "sha_pima": cfg.get("data_sha256", {}).get("pima")[:8] + "...",
        "sha_sylhet": cfg.get("data_sha256", {}).get("sylhet")[:8] + "...",
        "sha_cdc": cfg.get("data_sha256", {}).get("cdc")[:8] + "...",
        "sha_hosp130": cfg.get("data_sha256", {}).get("hosp130")[:8] + "...",
    })
df_cfg = pd.DataFrame(config_rows)
print(df_cfg.to_string())

# 2. Sylhet duplicate profiling
print("\n--- 2. SYLHET DUPLICATE PROFILING ---")
df_sylhet = pd.read_csv(data_dir / "sylhet/early_stage_diabetes_uci529.csv")
y_syl = (df_sylhet["class"].str.strip().str.lower() == "positive").astype(int).values
X_syl = df_sylhet.drop(columns=["class"]).copy()
X_syl["gender"] = (X_syl["gender"].str.strip().str.lower() == "male").astype(int)
for c in X_syl.columns:
    if X_syl[c].dtype == object:
        X_syl[c] = (X_syl[c].str.strip().str.lower() == "yes").astype(int)

keys = pd.DataFrame(X_syl).astype(str).agg("|".join, axis=1)
groups, uniques = pd.factorize(keys)
n_total = len(df_sylhet)
n_unique = len(uniques)
n_dups = n_total - n_unique

print(f"Total: {n_total}, Unique: {n_unique}, Duplicates: {n_dups}")

# Frequency distribution
counts = pd.Series(groups).value_counts()
freq_dist = counts.value_counts().sort_index()
freq_table = []
cum_rows = 0
for k, num_profiles in freq_dist.items():
    rows_k = k * num_profiles
    cum_rows += rows_k
    freq_table.append({
        "multiplicity_k": k,
        "n_unique_profiles": num_profiles,
        "total_rows": rows_k,
        "cum_rows": cum_rows,
        "pct_total_rows": (cum_rows / n_total) * 100
    })
df_freq = pd.DataFrame(freq_table)
print(df_freq.to_string())

# Inconsistent labels?
df_syl_grp = pd.DataFrame({"group": groups, "y": y_syl})
inconsistencies = df_syl_grp.groupby("group")["y"].nunique()
print("Inconsistent groups (with both 0 and 1):", (inconsistencies > 1).sum())

# CV split leak simulation
cv_sim_rows = []
for seed in [0, 1, 42]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(X_syl, y_syl)):
        g_tr = set(groups[tr])
        leaks = sum(groups[i] in g_tr for i in te)
        cv_sim_rows.append({
            "seed": seed, "fold": fold, "split_type": "Ungrouped (SKFold)",
            "n_test": len(te), "n_pos": y_syl[te].sum(), "n_neg": len(te) - y_syl[te].sum(),
            "n_unique_groups": len(set(groups[te])), "leak_rows": leaks,
            "leak_pct": (leaks / len(te)) * 100
        })
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(sgkf.split(X_syl, y_syl, groups=groups)):
        g_tr = set(groups[tr])
        leaks = sum(groups[i] in g_tr for i in te)
        cv_sim_rows.append({
            "seed": seed, "fold": fold, "split_type": "Grouped (SGKFold)",
            "n_test": len(te), "n_pos": y_syl[te].sum(), "n_neg": len(te) - y_syl[te].sum(),
            "n_unique_groups": len(set(groups[te])), "leak_rows": leaks,
            "leak_pct": (leaks / len(te)) * 100
        })
df_cv_sim = pd.DataFrame(cv_sim_rows)
print("\nCV Simulation summary (Ungrouped vs Grouped):")
print(df_cv_sim.groupby("split_type")[["n_test", "leak_rows", "leak_pct"]].agg(["min", "mean", "max"]).to_string())

# 3. Load all clean CSV rows and compute per-model variance
print("\n--- 3. PER-FOLD AND PER-SEED METRIC VARIANCE ---")
csv_files = sorted(results_dir.glob("0*.csv"))
dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)
all_df = pd.concat(dfs, ignore_index=True)

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

# Focus models for variance analysis:
# Top models per dataset + representative quantum/classical
stats_records = []
t_crit = stats.t.ppf(0.975, df=14) # 15-1=14 df

for (ds, m), grp in clean_df.groupby(["dataset", "full_model"]):
    if len(grp) != 15:
        continue
    # Balanced Accuracy
    ba_vals = grp["balanced_accuracy"].values
    ba_mean = np.mean(ba_vals)
    ba_sd_total = np.std(ba_vals, ddof=1)
    ba_se = ba_sd_total / np.sqrt(15)
    ba_ci_low = ba_mean - t_crit * ba_se
    ba_ci_high = ba_mean + t_crit * ba_se
    # Seed means
    seed_means_ba = grp.groupby("seed")["balanced_accuracy"].mean().values
    ba_sd_between_seeds = np.std(seed_means_ba, ddof=1)
    ba_sd_within_seeds = np.mean(grp.groupby("seed")["balanced_accuracy"].std(ddof=1).values)

    # Recall
    rec_vals = grp["recall"].values
    rec_mean = np.mean(rec_vals)
    rec_sd_total = np.std(rec_vals, ddof=1)
    rec_se = rec_sd_total / np.sqrt(15)
    rec_ci_low = rec_mean - t_crit * rec_se
    rec_ci_high = rec_mean + t_crit * rec_se
    seed_means_rec = grp.groupby("seed")["recall"].mean().values
    rec_sd_between_seeds = np.std(seed_means_rec, ddof=1)
    rec_sd_within_seeds = np.mean(grp.groupby("seed")["recall"].std(ddof=1).values)

    # Specificity
    spec_vals = grp["specificity"].values
    spec_mean = np.mean(spec_vals)
    spec_sd_total = np.std(spec_vals, ddof=1)
    spec_se = spec_sd_total / np.sqrt(15)
    spec_ci_low = spec_mean - t_crit * spec_se
    spec_ci_high = spec_mean + t_crit * spec_se
    seed_means_spec = grp.groupby("seed")["specificity"].mean().values
    spec_sd_between_seeds = np.std(seed_means_spec, ddof=1)
    spec_sd_within_seeds = np.mean(grp.groupby("seed")["specificity"].std(ddof=1).values)

    # Confusion matrix sums across 15 folds
    tn_sum = grp["tn"].sum()
    fp_sum = grp["fp"].sum()
    fn_sum = grp["fn"].sum()
    tp_sum = grp["tp"].sum()
    n_test_sum = grp["n_test"].sum()

    stats_records.append({
        "dataset": ds,
        "full_model": m,
        "n_folds": 15,
        "ba_mean": ba_mean, "ba_sd": ba_sd_total, "ba_se": ba_se,
        "ba_ci_low": ba_ci_low, "ba_ci_high": ba_ci_high,
        "ba_sd_between": ba_sd_between_seeds, "ba_sd_within": ba_sd_within_seeds,
        "rec_mean": rec_mean, "rec_sd": rec_sd_total, "rec_se": rec_se,
        "rec_ci_low": rec_ci_low, "rec_ci_high": rec_ci_high,
        "rec_sd_between": rec_sd_between_seeds, "rec_sd_within": rec_sd_within_seeds,
        "spec_mean": spec_mean, "spec_sd": spec_sd_total, "spec_se": spec_se,
        "spec_ci_low": spec_ci_low, "spec_ci_high": spec_ci_high,
        "spec_sd_between": spec_sd_between_seeds, "spec_sd_within": spec_sd_within_seeds,
        "tn_sum": int(tn_sum), "fp_sum": int(fp_sum), "fn_sum": int(fn_sum), "tp_sum": int(tp_sum),
        "n_test_sum": int(n_test_sum)
    })

df_stats = pd.DataFrame(stats_records)
print(f"Computed stats for {len(df_stats)} dataset-model combinations.")

# 4. Class Imbalance Breakdown on CDC and Hosp130
print("\n--- 4. CLASS IMBALANCE BREAKDOWN (CDC and HOSP130) ---")
imbal_models = [
    # CDC & HOSP130 quantum models
    "Daga: Angle-QSVM", "Daga: ZZ-QSVM", "Daga: Amplitude-QSVM",
    "Kiran: Angle-QSVM (same 4 PCs)", "Sodar: Angle-QSVM", "Sodar: Amplitude-QSVM (MinMax, paper)",
    "Yadav: ZZ-QSVM (full) bw=1", "Yadav: Z-QSVM bw=1",
    # Cost-weighted / thresholded classical models
    "Kiran: RBF-SVM (same 4 PCs, balanced)", "Kiran: Logistic Regression (same 4 PCs, balanced)",
    "Kotha: DiaBeats RF (threshold by inner CV)",
    # Unweighted classical baselines
    "Daga: RBF-SVM", "Daga: Random Forest", "Daga: Logistic Regression",
    "Daga: OOF-Stacking (LR meta)", "Daga: Majority class"
]

sub_imbal = df_stats[df_stats["dataset"].isin(["cdc", "hosp130"]) & df_stats["full_model"].isin(imbal_models)].copy()
sub_imbal = sub_imbal.sort_values(["dataset", "ba_mean"], ascending=[True, False])
imbal_cols = ["dataset", "full_model", "ba_mean", "rec_mean", "spec_mean", "tp_sum", "fn_sum", "fp_sum", "tn_sum", "n_test_sum"]
print(sub_imbal[imbal_cols].to_string())

# 5. Daga Stacking vs RBF-SVM Fold-by-Fold Table
print("\n--- 5. DAGA STACKING VS RBF-SVM PAIRED FOLD COMPARISON ---")
daga_clean = clean_df[clean_df["paper"] == "P08_Daga"]
stacking = daga_clean[daga_clean["model"] == "OOF-Stacking (LR meta)"]
rbf = daga_clean[daga_clean["model"] == "RBF-SVM"]
merged = pd.merge(stacking, rbf, on=["dataset", "seed", "fold"], suffixes=("_stack", "_rbf"))

paired_rows = []
for ds in ["pima", "sylhet", "cdc", "hosp130"]:
    sub = merged[merged["dataset"] == ds].sort_values(["seed", "fold"])
    diff = sub["balanced_accuracy_stack"] - sub["balanced_accuracy_rbf"]
    
    # Nadeau-Bengio correction:
    # J = 15 folds (3 seeds * 5 folds). n_test / n_train = 1 / 4 = 0.25
    K = len(diff)
    mean_d = np.mean(diff)
    var_d = np.var(diff, ddof=1)
    sd_d = np.std(diff, ddof=1)
    
    # Corrected variance
    corr_factor = (1.0 / K) + 0.25
    corr_var = corr_factor * var_d
    corr_se = np.sqrt(corr_var)
    t_stat = mean_d / corr_se if corr_se > 1e-12 else 0.0
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=K - 1)) if corr_se > 1e-12 else 1.0
    
    paired_rows.append({
        "dataset": ds,
        "n_folds": K,
        "mean_diff": mean_d,
        "sd_diff": sd_d,
        "corr_se": corr_se,
        "t_stat": t_stat,
        "p_val": p_val,
        "identical_cm_count": int(sum((sub["tn_stack"] == sub["tn_rbf"]) & (sub["fp_stack"] == sub["fp_rbf"]) &
                                      (sub["fn_stack"] == sub["fn_rbf"]) & (sub["tp_stack"] == sub["tp_rbf"])))
    })

df_paired = pd.DataFrame(paired_rows)
print(df_paired.to_string())

# Save outputs to JSON for formatting
output_data = {
    "config_audit": config_rows,
    "sylhet_freq": freq_table,
    "cv_simulation": cv_sim_rows,
    "stats_records": stats_records,
    "paired_stacking_rbf": merged[[
        "dataset", "seed", "fold",
        "balanced_accuracy_stack", "balanced_accuracy_rbf",
        "accuracy_stack", "accuracy_rbf",
        "recall_stack", "recall_rbf",
        "specificity_stack", "specificity_rbf",
        "tn_stack", "fp_stack", "fn_stack", "tp_stack",
        "tn_rbf", "fp_rbf", "fn_rbf", "tp_rbf"
    ]].to_dict(orient="records")
}

def json_converter(o):
    if isinstance(o, (np.integer, np.int64)):
        return int(o)
    elif isinstance(o, (np.floating, np.float64)):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

with open("notebooks2/results/statistical_audit_data.json", "w") as f:
    json.dump(output_data, f, indent=2, default=json_converter)

print("\nSuccessfully wrote notebooks2/results/statistical_audit_data.json!")
