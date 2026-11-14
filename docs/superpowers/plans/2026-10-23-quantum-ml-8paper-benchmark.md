# Quantum Machine Learning & Clinical ML Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reimplement, audit for data leakage/methodological flaws, and benchmark 8 recent QML and clinical machine learning papers across 5 clinical datasets using executed Jupyter notebooks (`notebooks/00_...ipynb` through `10_...ipynb`).

**Architecture:** Programmatic Jupyter notebook generation via `nbformat` and execution via `jupyter nbconvert --to notebook --execute --inplace` or `papermill`. All Python code resides inside notebook cells. Shared functions are exposed via standard fit/predict interfaces or imported via `%run`/`nbformat`. Evaluation uses repeated stratified K-fold cross-validation with strict inner/outer fold isolation.

**Tech Stack:** Python 3.11, Jupyter/nbformat/papermill, PennyLane (`lightning.qubit`), Qiskit/Qiskit Aer, PyTorch, Scikit-Learn, EconML, NetworkX, SHAP, LIME, Optuna, Pandas, Matplotlib/Seaborn.

**Spec:** User prompt instructions (8 papers audit, notebooks 00-10, dataset downloads, repos cloning, leakage-free benchmarking).

## Global Constraints

- Every piece of Python must live inside Jupyter notebooks (`.ipynb`). No `.py` files, scripts, or modules outside `repos/`.
- Executed notebooks must contain saved outputs, cell execution numbers, tables, and PNG figures.
- All 8 datasets and repos must be downloaded/cloned into `data/` and `repos/`.
- Seed all random number generators globally (`SEED = 42`).
- Strict nested/fold-local preprocessing: no fit on test folds, no data leakage.

## Review Focus

1. **Synthetic data leakage (Paper 2):** GMM or SMOTE sampling must occur strictly inside training folds, not before splitting or on validation sets.
2. **Causal/Statistical feature selection leakage (Paper 1):** LinearDML / t-test feature selection must run strictly inside training folds.
3. **Data imputation & scaling leakage (Paper 3, 5, 6, 7, 8):** Zero-as-missing median imputation, standard scaling, and PCA must fit only on training folds.
4. **Metric calculation consistency:** G-mean, Brier score, MCC, and AUC must handle zero positive predictions without throwing unhandled exceptions.
5. **Quantum simulation scalability:** PCA or feature truncation to 4–8 qubits to avoid exponentially slow CPU statevector simulation.

---

### Task 0: Setup, Paper PDF Audit, Repos, Data Pipeline, and Red Flags (Notebook 00)

**Files:**
- Create: `notebooks/00_setup_and_audit.ipynb`
- Create/Populate: `data/` (Pima, Sylhet UCI 529, BRFSS UCI 891 subsample, Breast Cancer Wisconsin, Parkinson's UCI 174)
- Create/Populate: `repos/` (cloned QISICGM, Zenodo archive for Paper 7, Adaptive Quantum Kernel Selection repo for Paper 8)

**Interfaces:**
- Produces: Dataset dictionary and SHA-256 checksums, structured summaries for all 8 papers, and empirical red-flag verification code in `00_setup_and_audit.ipynb`.

- [ ] **Step 1: Programmatically build `notebooks/00_setup_and_audit.ipynb`**
- [ ] **Step 2: Add Stage 0 structured paper summaries for all 8 papers**
- [ ] **Step 3: Add Stage 1 repository cloning (Paper 2, Paper 7 Zenodo, Paper 8) and dataset downloading scripts into notebook cells**
- [ ] **Step 4: Execute `notebooks/00_setup_and_audit.ipynb` using `jupyter nbconvert --to notebook --execute --inplace`**
- [ ] **Step 5: Verify notebook outputs and dataset checksums in cell outputs**

---

### Task 1: Reimplement Paper 1 — Upama et al. Causal QML (Notebook 01)

**Files:**
- Create: `notebooks/01_upama_causal_qml.ipynb`
- Output: `results/paper1_upama_results.csv`

**Interfaces:**
- Produces: `fit_upama_pipeline(X_train, y_train, use_quantum=True)`, `predict_upama(model, X_test)`

- [ ] **Step 1: Programmatically build `notebooks/01_upama_causal_qml.ipynb`**
- [ ] **Step 2: Implement LinearDML (EconML) causal feature selection vs t-test/Cohen's d, MI, SHAP**
- [ ] **Step 3: Implement Classical SVM, QSVM (Qiskit ZZFeatureMap), RF, depth-capped "QRF", and variational quantum classifier (VQC) alternative**
- [ ] **Step 4: Replicate single 104-sample split (original protocol) and repeated stratified 5-fold CV (leakage-free protocol)**
- [ ] **Step 5: Execute notebook end-to-end and save outputs**

---

### Task 2: Reimplement Paper 2 — Young QISICGM (Notebook 02)

**Files:**
- Create: `notebooks/02_qisicgm.ipynb`
- Output: `results/paper2_qisicgm_results.csv`

**Interfaces:**
- Produces: `fit_qisicgm_pipeline(X_train, y_train)`, `predict_qisicgm(model, X_test)`

- [ ] **Step 1: Programmatically build `notebooks/02_qisicgm.ipynb`**
- [ ] **Step 2: Implement phase feature map ($\cos(\alpha x), \sin(\alpha x)$), kNN concept graph (NetworkX), neighbor-sequence Transformer/CNN, RF/ET/FFNN base learners, isotonic calibration, and Logistic Regression meta-learner**
- [ ] **Step 3: Test pre-CV GMM synthetic augmentation vs in-fold GMM synthetic augmentation to quantify score inflation**
- [ ] **Step 4: Execute notebook end-to-end and save outputs**

---

### Task 3: Reimplement Paper 3 — Kotha et al. DiaBeats (Notebook 03)

**Files:**
- Create: `notebooks/03_diabeats.ipynb`
- Output: `results/paper3_diabeats_results.csv`

**Interfaces:**
- Produces: `fit_diabeats_pipeline(X_train, y_train)`, `predict_diabeats(model, X_test)`

- [ ] **Step 1: Programmatically build `notebooks/03_diabeats.ipynb`**
- [ ] **Step 2: Implement zero-as-missing median imputation, StandardScaler, RandomForest, threshold tuning, and probability calibration**
- [ ] **Step 3: Implement inline interactive widget / batch prediction UI demo inside notebook**
- [ ] **Step 4: Execute notebook end-to-end and save outputs**

---

### Task 4: Reimplement Paper 4 — Zhang et al. DeSCN-GIF (Notebook 04)

**Files:**
- Create: `notebooks/04_descn_gif.ipynb`
- Output: `results/paper4_descn_gif_results.csv`

**Interfaces:**
- Produces: `fit_descn_gif(X_train, y_train, noise_rate=0.0)`, `predict_descn_gif(model, X_test)`

- [ ] **Step 1: Programmatically build `notebooks/04_descn_gif.ipynb`**
- [ ] **Step 2: Implement Graph Intuitionistic Fuzzy scoring ($\mu, \nu, \pi, \Theta$), incremental Deep Stochastic Configuration Network with supervisory inequality, and weighted ridge output weights**
- [ ] **Step 3: Evaluate on public diabetes datasets with 0%, 5%, 10%, 20%, 40% label-flip noise vs RVFL, SCN, DSCN baselines**
- [ ] **Step 4: Execute notebook end-to-end and save outputs**

---

### Task 5: Reimplement Paper 5 — Kiran et al. HQCAN-BSTOA (Notebook 05)

**Files:**
- Create: `notebooks/05_hqcan_bstoa.ipynb`
- Output: `results/paper5_hqcan_results.csv`

**Interfaces:**
- Produces: `fit_hqcan_pipeline(X_train, y_train)`, `predict_hqcan(model, X_test)`

- [ ] **Step 1: Programmatically build `notebooks/05_hqcan_bstoa.ipynb`**
- [ ] **Step 2: Implement Iterative Self-Guided Filtering (ISGF), Scale-Aware Modulation Transformer feature extractor, 4-qubit PennyLane hybrid network, and Boosted Sooty Tern Optimization (BSTOA / Optuna fallback)**
- [ ] **Step 3: Test for full-dataset filtering/transformer leakage vs in-fold processing to audit the claimed 99.7% accuracy**
- [ ] **Step 4: Execute notebook end-to-end and save outputs**

---

### Task 6: Reimplement Paper 6 — Sodar et al. Encrypted Traffic QML (Notebook 06)

**Files:**
- Create: `notebooks/06_encrypted_traffic_qml.ipynb`
- Output: `results/paper6_encrypted_traffic_results.csv`

**Interfaces:**
- Produces: `fit_traffic_qml(X_train, y_train, encoding="angle")`, `predict_traffic_qml(model, X_test)`

- [ ] **Step 1: Programmatically build `notebooks/06_encrypted_traffic_qml.ipynb`**
- [ ] **Step 2: Implement PennyLane angle & amplitude encodings for QSVM, QKNN (fidelity kernel/distance), and Pauli-Z expectation hybrid feature extractor with classical SVM/KNN**
- [ ] **Step 3: Compare 70/30 single split vs repeated 5-fold CV on tabular clinical datasets**
- [ ] **Step 4: Execute notebook end-to-end and save outputs**

---

### Task 7: Reimplement Paper 7 — Yadav et al. QSVM-QNN (Notebook 07)

**Files:**
- Create: `notebooks/07_qsvm_qnn_breast_cancer.ipynb`
- Output: `results/paper7_qsvm_qnn_results.csv`

**Interfaces:**
- Produces: `fit_qsvm_qnn(X_train, y_train, map_type="ZZ")`, `predict_qsvm_qnn(model, X_test)`

- [ ] **Step 1: Programmatically build `notebooks/07_qsvm_qnn_breast_cancer.ipynb`**
- [ ] **Step 2: Implement Qiskit / PennyLane Z, ZZ, Pauli feature maps with FidelityQuantumKernel, QSVC, EstimatorQNN / VQC, and hybrid QSVM-QNN**
- [ ] **Step 3: Evaluate with paired t-tests and F-tests on Breast Cancer Wisconsin and Diabetes datasets**
- [ ] **Step 4: Execute notebook end-to-end and save outputs**

---

### Task 8: Reimplement Paper 8 — Daga et al. Adaptive Quantum Kernel Stacking (Notebook 08)

**Files:**
- Create: `notebooks/08_adaptive_quantum_kernel_stacking.ipynb`
- Output: `results/paper8_adaptive_stacking_results.csv`

**Interfaces:**
- Produces: `fit_quantum_stacking(X_train, y_train)`, `predict_quantum_stacking(model, X_test)`

- [ ] **Step 1: Programmatically build `notebooks/08_adaptive_quantum_kernel_stacking.ipynb`**
- [ ] **Step 2: Implement Angle, Amplitude, ZZ Gram matrix generation for QSVM & QKNN, Logistic Regression OOF meta-learner, nested CV (5 inner / 10 outer), 1000-iteration bootstrap CIs, VQC baseline, and noise simulation**
- [ ] **Step 3: Execute notebook end-to-end and save outputs**

---

### Task 9: Cross-Dataset x Cross-Method Benchmark (Notebook 09)

**Files:**
- Create: `notebooks/09_cross_dataset_cross_method_benchmark.ipynb`
- Output: `results/benchmark_grid.csv`, `results/figures/` (heatmaps, critical-difference diagram, ROC/PR curves, calibration curves, runtime plots)

**Interfaces:**
- Consumes: Preprocessing adapters and model fit/predict functions from Tasks 1–8.
- Produces: Full benchmark grid across 8 method families + 5 classical baselines x 5 clinical datasets under repeated stratified 5-fold CV.

- [ ] **Step 1: Programmatically build `notebooks/09_cross_dataset_cross_method_benchmark.ipynb`**
- [ ] **Step 2: Implement dataset adapters and common evaluation harness computing Accuracy, Balanced Acc, Precision, Recall, Specificity, F1, G-Mean, ROC-AUC, PR-AUC, Brier, MCC, and Runtime**
- [ ] **Step 3: Run full evaluation grid with parallelized execution and bootstrap CIs + Friedman/Nemenyi non-parametric statistical tests**
- [ ] **Step 4: Generate publication-ready figures and save `results/benchmark_grid.csv`**
- [ ] **Step 5: Execute notebook end-to-end and verify outputs**

---

### Task 10: Final Synthesis Report & Documentation (Notebook 10, README.md, requirements.txt)

**Files:**
- Create: `notebooks/10_final_report.ipynb`
- Create: `README.md`
- Create: `requirements.txt`

**Interfaces:**
- Produces: Comprehensive comparison table (Paper Claimed vs Reproduced vs Leakage-Free), top 5 empirical findings, execution guide, and pinned dependency specification.

- [ ] **Step 1: Programmatically build `notebooks/10_final_report.ipynb` containing paper claims vs reproduction vs leakage-free metrics, statistical significance verdicts, and top 5 findings**
- [ ] **Step 2: Execute `notebooks/10_final_report.ipynb` end-to-end**
- [ ] **Step 3: Generate root `README.md` and `requirements.txt` with exact execution instructions**
- [ ] **Step 4: Run final verification across all 11 executed notebooks to confirm zero errors and complete output rendering**
