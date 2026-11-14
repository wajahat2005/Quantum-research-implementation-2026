# notebooks2 — each paper's method on each diabetes dataset

The method of every audited paper runs on all four diabetes datasets, each under two protocols:
* **`paper`**: the paper's own split and preprocessing, with its leaks left in.
* **`clean`**: leakage-free.

Cancer (WDBC) and Parkinson's data are deliberately left out.

| # | Notebook | Paper | Method transferred |
|---|---|---|---|
| 01 | `01_upama_causal_qml.ipynb` | Upama et al. 2026 | DML causal feature ranking → linear SVM / RF / Angle-QSVM |
| 02 | `02_young_qisicgm.ipynb` | Young 2025 | phase map + RF/ET/MLP ensemble, GMM synthetic positives |
| 03 | `03_kotha_diabeats.ipynb` | Kotha et al. 2026 | random forest; single split vs 10 splits vs CV |
| 04 | `04_zhang_gif_label_noise.ipynb` | Zhang et al. 2025 | GIF sample weights (LightGBM) under 0–40 % label noise |
| 05 | `05_kiran_hqcan_vqc.ipynb` | Kiran et al. 2025 | 4-qubit VQC vs LR / RBF-SVM on the same 4 PCs |
| 06 | `06_sodar_angle_amplitude.ipynb` | Sodar et al. 2026 | Angle vs Amplitude QSVM |
| 07 | `07_yadav_z_zz_qsvm.ipynb` | Yadav et al. 2026 | Z / ZZ feature-map QSVM, bandwidth by inner CV |
| 08 | `08_daga_adaptive_stacking.ipynb` | Daga et al. 2026 | Angle/ZZ/Amplitude QSVM, RBF, OOF stacking, voting |
| 09 | `09_summary.ipynb` | — | combined tables, paired tests + Holm, bootstrap CIs, heatmap, `summary.md` |

## Datasets (SHA-256 checked on load)
| Id | File | Positive class | Rows used |
|---|---|---|---|
| pima | `data/pima/pima-indians-diabetes.csv` | diabetes | 768 |
| sylhet | `data/sylhet/early_stage_diabetes_uci529.csv` | early-stage diabetes | 520 (251 unique vectors → grouped CV) |
| cdc | `data/cdc/cdc_diabetes_health_indicators_uci891.csv` | diabetes or prediabetes (self-reported) | stratified 2,000-row subsample, seed 42 |
| hosp130 | `data/diabetes_130_hospitals/diabetic_data.csv` | readmitted < 30 days (diabetic inpatients) | first encounter per patient, then stratified 2,000-row subsample, seed 42 |

## Protocols
* **clean**: 5-fold CV × seeds {0, 1, 42}. Imputation, scaling, PCA, feature ranking, GMM, GIF weights, tuning and stacking are all fitted on the training fold only. When identical feature vectors exist (Sylhet, and possibly CDC), `StratifiedGroupKFold` keeps every copy on one side of the split.
* **paper**: each notebook's header states exactly what the paper did (hold-out size, global scaling, etc.). Every result row carries `leak_rows`, the number of test rows whose feature vector also appears in training.
* Quantum kernels are exact noiseless simulations in numpy, with q = min(d, 8) qubits and in-fold PCA when d > 8. A self-test (`sanity_check_kernels()`) runs at the top of every notebook. The VQC (05) uses PennyLane.

## How to run
**Colab**
1. Upload the `notebooks2/` folder **and** the project's `data/` folder (both under `/content`, i.e. `/content/notebooks2` and `/content/data`), or put both in Drive and `cd` there.
2. Open a notebook, set the working directory to `notebooks2/` (`%cd /content/notebooks2`), and run all cells.
3. Run 01–08 in any order, then 09. Download `notebooks2/results/` and the executed `.ipynb` files.

**Kaggle**: add `notebooks2/` and `data/` as one dataset. The setup cell finds `nb2_common.py` and the data under `/kaggle/input`. Results are written to `/kaggle/working/results`.

If the data lives somewhere else, set `os.environ["NB2_DATA"] = "/path/to/data"` before the import cell. `NB2_RESULTS` sets the output folder.

Rough CPU runtimes: 01 ≈ 10 min, 02 ≈ 10 min, 03 ≈ 3 min, 04 ≈ 10 min, 05 ≈ 15–25 min, 06 ≈ 5 min, 07 ≈ 20–30 min (bandwidth grid), 08 ≈ 15–25 min (inner-CV stacking), 09 < 1 min.

## Outputs (`results/`)
* `NN_<paper>.csv` has one row per dataset × protocol × seed × fold × model, with accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, ROC-AUC, PR-AUC, confusion counts and `n_pred_pos`.
* `NN_<paper>_config.json` records the configuration, package versions and runtime.
* `09_heatmap_balanced_accuracy.png`, `04_zhang_gmean_vs_noise.png` and `summary.md`.

## Editing
Do not edit the `.ipynb` files by hand. Edit `src/*.py` (percent format) and rebuild:
`powershell -ExecutionPolicy Bypass -File notebooks2\build_notebooks.ps1`
The build also copies `src/nb2_common.py` to `notebooks2/nb2_common.py`.

## Pre-registered settings
Seeds, split sizes, subsample size, qubit cap, bandwidth grid, VQC training schedule and noise rates were all fixed before any run, and none is chosen from test-fold results. Record any later change here, with the reason.
