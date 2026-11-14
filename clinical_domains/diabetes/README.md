# Clinical Diabetes Quantum Machine Learning Audit & Cross-Dataset Benchmark

## 1. Overview & Clinical Cohorts
This domain encapsulates the comprehensive, multi-dataset investigation of Quantum Machine Learning (QML) applied to clinical diabetes diagnostics and 30-day hospital readmission:

1. **D1: Pima Indians Diabetes Database (PIDD)**:
   - $N = 768$ adult female Pima Indian patients, 8 clinical biomarker features, 34.9% positive prevalence.
2. **D2: Early Stage Diabetes Risk Prediction (Sylhet Diabetes)**:
   - $N = 520$ patient questionnaire responses from Sylhet Diabetes Hospital (UCI 529), 16 symptom indicators, 61.5% positive prevalence.
   - **Audit Discovery**: Contains 269 duplicate records (51.7%) across only 251 unique patient questionnaires.
3. **D3: CDC BRFSS Diabetes Health Indicators**:
   - $N = 253,680$ survey respondents (UCI 891), 21 epidemiological risk factors, 13.9% positive prevalence.
4. **D4: Diabetes 130-US Hospitals Readmission Benchmark**:
   - $N = 101,766$ inpatient clinical encounters (UCI 296), 47 clinical lab and medication features, 11.2% 30-day readmission prevalence.

---

## 2. Directory Structure
```
clinical_domains/diabetes/
├── publications_and_reports/
│   ├── IEEE_MULTI_DATASET_PAPER.pdf      # IEEE Transactions-style multi-dataset paper
│   ├── MULTI_DATASET_SLIDES.pdf          # 22-slide conference presentation (16:9 widescreen)
│   ├── MASTER_MULTI_DATASET_REPORT.pdf   # Extended technical synthesis report
│   ├── DIABETES_REIMPLEMENTATION_REPORT_v2.pdf # Detailed re-implementation audit
│   ├── DIABETES_REIMPLEMENTATION_SLIDES.pdf    # Re-implementation slides
│   └── MASTER_DIABETES_QML_AUDIT.pdf     # Master diabetes audit report
├── reproduction_code/
│   ├── 01_upama_causal_qml.py            # Paper 01: Upama et al. (DML feature selection & GroupKFold)
│   ├── 02_young_qisicgm.py               # Paper 02: Young (QISICGM GMM test contamination audit)
│   ├── 03_kotha_diabeats.py              # Paper 03: Kotha et al. (DiaBeats Mann-Whitney AUC bound)
│   ├── 04_zhang_gif_label_noise.py       # Paper 04: Zhang et al. (GIF sample weighting under 0-40% label noise)
│   ├── 05_kiran_hqcan_vqc.py             # Paper 05: Kiran et al. (HQCAN 4-qubit VQC error bound)
│   ├── 06_sodar_angle_amplitude.py       # Paper 06: Sodar et al. (Amplitude state collapse audit)
│   ├── 07_yadav_z_zz_qsvm.py             # Paper 07: Yadav et al. (Z-Map vs ZZ-Map entanglement penalty)
│   ├── 08_daga_adaptive_stacking.py      # Paper 08: Daga et al. (Adaptive kernel stacking cross-benchmark)
│   └── 09_summary.py                     # Cross-benchmark synthesis & statistical comparison
├── benchmark_results/
│   ├── benchmark_grid.csv                # Complete model-by-dataset evaluation matrix
│   ├── benchmark_summary.csv             # Summary metrics across all clinical protocols
│   ├── dataset_audit_table.csv           # Cryptographic hashes, missingness, and leakage stats
│   ├── novel_hybrids_benchmark.csv       # Stage 6 cost-sensitive screening hybrid results
│   ├── grand_classical_vs_quantum_summary.csv # Statistical quantum vs classical comparisons
│   └── paper01_upama_metrics.csv ... paper08_daga_metrics.csv # Per-paper metric records
└── README.md                             # This domain documentation file
```

---

## 3. Audited Publications & Empirical Findings

| ID | Landmark Paper | Claimed Metric | Honest Leakage-Free Metric | Discrepancy & Root Cause |
| :--- | :--- | :---: | :---: | :--- |
| **P01** | Upama et al. (2026) | 97.12% Accuracy | **83.51%** (RF) / **71.18%** (QSVM) | 269 duplicate patient questionnaires leaked across random train/test splits. Corrected via `GroupKFold`. "Quantum RF" is classical RF. |
| **P02** | Young (2025) | 96.3% AUC / 0.934 F1 | **82.8%** AUC / **0.619** F1 | 2,000 synthetic minority samples generated via GMM prior to evaluation. Removing synthetic rows drops CDC Balanced Acc from 0.951 to 0.622. |
| **P03** | Kotha et al. (2026) | 1.0000 ROC-AUC | $\le \mathbf{0.9700}$ (Max Upper Bound) | Mann-Whitney U theorem: Published 15 FN + 10 FP forces $\ge 150$ rank inversions. Claimed 1.00 AUC is mathematically impossible. |
| **P04** | Zhang et al. (2026) | Robust at 40% noise | Tested at 0%, 20%, 40% noise | GIF sample weighting assists on Pima, but on large imbalanced cohorts (CDC & Hosp 130), balanced accuracy remains near chance (0.50-0.53). |
| **P05** | Kiran et al. (2025) | 99.70% Accuracy | $\le \mathbf{96.0\%}$ (Text) / **74.74%** (CV) | Manuscript states "4% error rate" ($\le 96\%$ accuracy). 5-fold CV shows 4-qubit VQC matches Logistic Regression (74.0%). |
| **P06** | Sodar et al. (2026) | 98.40% Accuracy | **35.00%** (Amplitude QSVM) | $L_2$ unit normalization collapses collinear clinical vectors, causing 0.00% recall. Angle encoding preserves biomarker scale (76.6%). |
| **P07** | Yadav et al. (2026) | 85.00% (ZZ QSVM) | **74.74%** (Z) vs **69.14%** (ZZ) | Entanglement penalty: 2-qubit ZZ phase over-rotation fragments tabular boundaries. Unentangled Z-Map outperforms ZZ-Map by $+5.60\%$. |
| **P08** | Daga et al. (2026) | Adaptive Stacking | **77.60%** (Pima) / **79.52%** (Sylhet) | Adaptive kernel stacking gains $+2\%$ to $+4\%$ via diversity, but Classical LightGBM/ExtraTrees matches performance at $100\times$ less compute. |

---

## 4. Master Cross-Dataset Comparison (Balanced Accuracy / Sensitivity)

| Audited Pipeline | Pima (D1) | Sylhet Leak-Free (D2) | CDC BRFSS (D3) | Hosp. 130 Readmit (D4) |
| :--- | :---: | :---: | :---: | :---: |
| **Upama: Angle QSVM (DML top-K)** | 0.72 / 0.58 | 0.71 / 0.76 | 0.50 / 0.00 | 0.50 / 0.00 |
| **Upama: Classical Random Forest** | 0.73 / 0.61 | 0.84 / 0.88 | 0.63 / 0.41 | 0.51 / 0.05 |
| **Young: QISICGM Ensemble (Real rows)** | 0.74 / 0.62 | 0.85 / 0.89 | 0.62 / 0.40 | 0.51 / 0.04 |
| **Kotha: DiaBeats Tuned RF** | 0.73 / 0.71 | 0.84 / 0.88 | 0.64 / 0.48 | 0.52 / 0.12 |
| **Daga: Adaptive Quantum Stacking** | 0.74 / 0.61 | 0.79 / 0.83 | 0.52 / 0.08 | 0.50 / 0.02 |
| **Classical Reference: RBF-SVM** | 0.72 / 0.57 | 0.83 / 0.86 | 0.58 / 0.28 | 0.51 / 0.03 |
| **Majority Class Baseline** | 0.50 / 0.00 | 0.50 / 0.00 | 0.50 / 0.00 | 0.50 / 0.00 |
