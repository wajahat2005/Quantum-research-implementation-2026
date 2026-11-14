# Independent Recheck & Verification Suite

This directory contains the independent scientific recomputation and audit suite for 8 landmark papers in Clinical AI and Quantum Machine Learning (QML) applied to clinical diabetes cohorts.

Treating all prior numbers as unverified, all code has been structured as self-contained Jupyter notebooks (`.ipynb`), executed strictly from clean kernels, generating real outputs and producing zero errors.

---

## Directory Organization

```
recheck/
├── notebooks/
│   ├── R00_common.ipynb                  # Common helpers, data loaders, fast vectorized quantum kernels
│   ├── R00_freeze_inventory.ipynb        # Stage 0: File inventory and hash verification
│   ├── R01_paper_claims_reextraction.ipynb# Stage 1: Two-pass PDF claim extraction & bib fixes
│   ├── R02_repo_code_audit.ipynb         # Stage 2: Repo code audit & data leakage citations
│   ├── R03_data_reaudit.ipynb            # Stage 3: Clinical diabetes dataset audit & Sylhet duplicate proof
│   ├── R04_upama_causal_qml.ipynb        # Stage 5.1: Upama et al. (MLWA 2026) reproduction & Sylhet leakage
│   ├── R05_young_qisicgm.ipynb           # Stage 5.2: Young (arXiv 2509.12259) GMM augmentation deconstruction
│   ├── R06_kotha_diabeats.ipynb          # Stage 5.3: Kotha et al. (DiaBeats) AUC=1.00 refutation & clinical simulator
│   ├── R07_zhang_descn_gif.ipynb         # Stage 5.4: Zhang et al. (DeSCN-GIF) label-noise dampening on diabetes
│   ├── R08_kiran_hqcan.ipynb             # Stage 5.5: Kiran et al. (HQCAN) 4-qubit VQC vs logistic regression
│   ├── R09_sodar_traffic_qml.ipynb       # Stage 5.6: Sodar et al. amplitude encoding rank-1 spectral collapse
│   ├── R10_yadav_qsvm_qnn.ipynb          # Stage 5.7: Yadav et al. ZZ-map bandwidth tuning sweep & degradation
│   ├── R11_daga_adaptive_stacking.ipynb  # Stage 5.8: Daga et al. (Sci Rep 2026) bootstrap replicate & repo leakage
│   ├── R12_cross_grid.ipynb              # Stage 5.9: 14-model benchmark grid across 4 diabetes datasets
│   ├── R13_improvements.ipynb            # Stage 6: Deconstruction of threshold tuning & honest hybrid audit
│   └── R14_discrepancy_and_report.ipynb  # Stage 7 & 8: Verification gates, discrepancy log, synthesis
├── results/                              # Executed CSV result artifacts
│   ├── file_inventory.csv
│   ├── old_claims_inventory.csv
│   ├── paper_claims_v2.csv
│   ├── bibliography_corrections.csv
│   ├── repo_code_audit.csv
│   ├── data_audit_v2.csv
│   ├── dataset_summary_v2.csv
│   ├── paper01_upama_recheck_metrics.csv
│   ├── paper02_young_recheck_metrics.csv
│   ├── paper03_kotha_diabeats_recheck_metrics.csv
│   ├── paper04_zhang_recheck_metrics.csv
│   ├── paper05_kiran_recheck_metrics.csv
│   ├── paper06_sodar_recheck_metrics.csv
│   ├── paper07_yadav_recheck_metrics.csv
│   ├── paper08_daga_recheck_metrics.csv
│   ├── benchmark_grid_v2.csv
│   └── novel_hybrids_recheck.csv
├── DISCREPANCY_LOG.csv                   # Reconciles all prior claims against fresh rechecks
├── WORKLOG_RECHECK.md                    # Chronological execution log of all steps, errors, and fixes
├── CORRECTED_REPORT.md                   # Full 10-section comprehensive audit report
└── requirements.txt                      # Pinned Python package dependencies
```

---

## Scope Compliance

- **Included Clinical Diabetes Cohorts**:
  1. **D1: Pima Indians Diabetes** ($N=768$, 8 features, 34.90% positive prevalence)
  2. **D2: Sylhet Early Stage Diabetes (UCI 529)** ($N=520$, 16 features, 61.54% positive prevalence)
  3. **D3: CDC BRFSS Diabetes Health Indicators (UCI 891)** ($N=253,680$, subsampled $N \in [500, 1000, 2000]$)
  4. **D4: UCI Diabetes 130-US Hospitals (UCI 296)** ($N=101,766$, subsampled $N \in [500, 1000, 2000]$)
- **Explicit Exclusions**:
  - Breast Cancer Wisconsin (WDBC) and METABRIC: Strictly **PURGED**.
  - Parkinson's Disease (UCI 174): Strictly **PURGED**.
  - ISCX VPN-nonVPN network traffic: Marked `NOT_TESTABLE_HERE`; algorithms adapted to Pima.
  - Huawei voice audio: Marked `NOT_TESTABLE_HERE`; algorithms adapted to Pima.

---

## Reproduction Commands

To execute all notebooks sequentially from clean kernels:
```powershell
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace `
  recheck/notebooks/R00_freeze_inventory.ipynb `
  recheck/notebooks/R01_paper_claims_reextraction.ipynb `
  recheck/notebooks/R02_repo_code_audit.ipynb `
  recheck/notebooks/R03_data_reaudit.ipynb `
  recheck/notebooks/R04_upama_causal_qml.ipynb `
  recheck/notebooks/R05_young_qisicgm.ipynb `
  recheck/notebooks/R06_kotha_diabeats.ipynb `
  recheck/notebooks/R07_zhang_descn_gif.ipynb `
  recheck/notebooks/R08_kiran_hqcan.ipynb `
  recheck/notebooks/R09_sodar_traffic_qml.ipynb `
  recheck/notebooks/R10_yadav_qsvm_qnn.ipynb `
  recheck/notebooks/R11_daga_adaptive_stacking.ipynb `
  recheck/notebooks/R12_cross_grid.ipynb `
  recheck/notebooks/R13_improvements.ipynb `
  recheck/notebooks/R14_discrepancy_and_report.ipynb
```
