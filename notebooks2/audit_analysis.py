import os, sys, json, glob
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

results_dir = Path("notebooks2/results")

# 1. Config audit
configs = {}
for p in sorted(results_dir.glob("*_config.json")):
    with open(p, "r") as f:
        configs[p.name] = json.load(f)

print("=== CONFIG AUDIT ===")
prereg = {
    "clean_seeds": [0, 1, 42],
    "paper_seed": 42,
    "n_splits": 5,
    "subsample_n": 2000,
    "subsample_seed": 42,
    "max_qubits": 8,
    "sha256": {
        "pima": "6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af",
        "sylhet": "f6bf14c5d2e939feccd78750da2542f131b78d80355ccf35f29554b1f03fcb97",
        "cdc": "9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964",
        "hosp130": "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97",
    }
}

for name, cfg in configs.items():
    match_clean = cfg.get("clean_seeds") == prereg["clean_seeds"]
    match_paper_seed = cfg.get("paper_seed") == prereg["paper_seed"]
    match_splits = cfg.get("n_splits") == prereg["n_splits"]
    match_subsample_n = cfg.get("subsample_n") == prereg["subsample_n"]
    match_subsample_seed = cfg.get("subsample_seed") == prereg["subsample_seed"]
    match_max_qubits = cfg.get("max_qubits") == prereg["max_qubits"]
    match_sha = cfg.get("data_sha256") == prereg["sha256"]
    
    all_match = all([match_clean, match_paper_seed, match_splits, match_subsample_n, match_subsample_seed, match_max_qubits, match_sha])
    print(f"{name}: All Match={all_match} | runtime={cfg.get('runtime_s')}s | n_rows={cfg.get('n_rows')} | platform={cfg.get('platform')}")

# Load all CSVs
csv_files = sorted(results_dir.glob("0*.csv"))
dfs = []
for f in csv_files:
    df = pd.read_csv(f)
    dfs.append(df)
all_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal rows in all CSVs: {len(all_df)}")
print(f"Datasets: {all_df['dataset'].unique()}")
print(f"Protocols: {all_df['protocol'].unique()}")
print(f"Clean rows: {len(all_df[all_df['protocol'] == 'clean'])}")
print(f"Paper rows: {len(all_df[all_df['protocol'] == 'paper'])}")

