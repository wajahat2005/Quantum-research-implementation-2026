# Autonomous Audit, Reproduction, and Cross-Dataset Benchmark of Quantum Machine Learning in Clinical Diabetes Diagnostics

A rigorous, reproducible machine learning research audit evaluating **eight landmark Quantum Machine Learning (QML) and Healthcare AI publications (2025–2026)** applied strictly to **clinical diabetes diagnostics and readmission benchmarks**.

---

## Key Findings & Critical Audits

1. **Sylhet Questionnaire Duplicate Leakage (Paper 01 - Upama et al., 2026)**:
   - The Sylhet Early Stage Diabetes dataset (UCI 529) contains **269 duplicate records** (51.7% of the 520 rows) across 251 unique patient profiles.
   - Standard random train/test splitting leaks identical patient questionnaires across partitions, inflating Classical RF to 97.1% and QSVM to 97.1%.
   - Under leakage-free `GroupKFold` cross-validation, Classical RF drops to **83.5%** and QSVM drops to **71.2%**. The paper's "Quantum Random Forest" is merely a classical random forest with capped `max_depth`.
2. **Synthetic Minority Test Contamination (Paper 02 - Kenneth G. Young II, 2025)**:
   - 2,000 synthetic minority samples were generated via a Gaussian Mixture Model (GMM) prior to cross-validation evaluation.
   - Evaluating on clean, real clinical patients drops OOF F1 from **0.934** to **0.619**, ROC-AUC from **0.963** to **0.828**, and Accuracy to **75.9%**. The "quantum-inspired" phase map is classical trigonometric $[\cos(\alpha x), \sin(\alpha x)]$ expansion.
3. **Mathematical Contradiction in DiaBeats (Paper 03 - Kotha et al., 2026)**:
   - Published confusion matrix (Fig. 4, $N=150$) reports 25 classification errors ($\text{TN}=90, \text{FP}=10, \text{FN}=15, \text{TP}=35$).
   - By Mann-Whitney U theorem, 15 false negatives and 10 false positives guarantee at least 150 misordered pairs, establishing an absolute theoretical upper bound of $\text{AUC}_{\max} \le \frac{5,000 - 150}{5,000} = 0.9700$. Published $\text{AUC} = 1.00$ is mathematically impossible.
4. **Internal Inconsistency in HQCAN (Paper 05 - Kiran et al., 2025)**:
   - The manuscript text states a "4% error rate", which mathematically bounds accuracy at $\le 96.0\%$, directly contradicting the claimed 99.7% headline accuracy.
   - Honest 5-fold CV reveals the true capability of the 4-qubit VQC on Pima is **74.7%**, equivalent to classical logistic regression (74.0%).
5. **Clinical Scale Erasure in Amplitude Encoding (Paper 06 - Sodar et al., 2026)**:
   - Amplitude encoding enforces $L_2$ unit normalization ($\|x\|_2 = 1$). In clinical diagnostics, absolute biomarker levels (e.g., blood glucose 180 vs. 90 mg/dL) are decisive.
   - Collinear patient vectors map to identical quantum states, causing Amplitude-QSVM to collapse to majority-class prediction with **0.00% sensitivity**. Angle encoding preserves feature scale and reaches 76.6% accuracy.
6. **Entanglement Fragmentation on Clinical Data (Paper 07 - Yadav et al., 2026)**:
   - Unentangled Z-Map QSVM (**74.7%** accuracy, **0.808** AUC) significantly outperforms entangling ZZ-Map QSVM (**69.1%** accuracy, **0.711** AUC). High-frequency 2-qubit phase over-rotation fragments tabular clinical decision boundaries.
7. **Stacking Ensemble Diversity vs. Quantum Speedup (Paper 08 - Daga et al., 2026)**:
   - Out-of-fold adaptive kernel stacking outperforms individual quantum kernels by $+2\%$ to $+4\%$, confirming the algorithmic value of ensemble diversity. However, classical ensembles (LightGBM, ExtraTrees, Classical Stacking) match or exceed quantum stacking at $100\times$ lower computational cost.
8. **Novel Cost-Sensitive Screening Hybrid (Stage 6)**:
   - By combining in-fold MICE imputation (resolving biologically impossible zeros) with cost-sensitive threshold tuning, the novel hybrid boosts clinical Sensitivity from 60.1% to **89.56%** while maintaining 0.828 ROC-AUC, reducing missed diabetic cases by 74%.

---

## Repository Structure

```
├── papers/                                         # Original paper PDFs (01 to 08)
├── repos/                                          # Cloned public repos and downloaded archives
│   ├── Adaptive-Quantum-Kernel-Selection/          # Daga et al. (2026) repo
│   ├── QISICGM/                                    # Young (2025) repo
│   └── yadav_zenodo/                               # Yadav et al. (2026) Zenodo archive
├── data/                                           # Processed clinical benchmarks
│   ├── pima/                                       # D1: Pima Indians Diabetes (768 rows, 8 features)
│   ├── sylhet/                                     # D2: Early Stage Diabetes (UCI 529, 520 rows)
│   ├── cdc/                                        # D3: CDC BRFSS Diabetes Indicators (UCI 891)
│   └── diabetes_130_hospitals/                     # D4: Diabetes 130-US Hospitals (UCI 296)
├── notebooks/                                      # Fully executed in-place Jupyter Notebooks
│   ├── 00_common.ipynb                             # Shared PennyLane kernels, loaders, statistical tests
│   ├── 00_setup_and_audit.ipynb                    # Cryptographic audit & baseline distributions
│   ├── 01_upama_causal_qml.ipynb                   # Paper 1 reproduction & GroupKFold leakage correction
│   ├── 02_young_qisicgm.ipynb                      # Paper 2 reproduction on augmented vs real Pima
│   ├── 03_kotha_diabeats.ipynb                     # Paper 3 DiaBeats & interactive risk simulator
│   ├── 04_zhang_descn_gif.ipynb                    # Paper 4 DeSCN-GIF noise robustness
│   ├── 05_kiran_hqcan.ipynb                        # Paper 5 4-qubit VQC HQCAN reproduction
│   ├── 06_sodar_traffic_qml.ipynb                  # Paper 6 Angle vs Amplitude clinical scale erasure
│   ├── 07_yadav_qsvm_qnn.ipynb                     # Paper 7 Z vs ZZ vs Pauli quantum kernels
│   ├── 08_daga_adaptive_stacking.ipynb             # Paper 8 Multi-kernel stacking reproduction
│   ├── 09_cross_dataset_cross_method_benchmark.ipynb # 14-model x 4-dataset grand cross-benchmark
│   └── 10_final_report.ipynb                       # Master audit table & novel hybrid improvements
├── results/                                        # Verified CSV metrics and publication PNG figures
│   ├── paper_claims.csv                            # Dual-pass extracted claims from 8 PDFs
│   ├── dataset_audit_table.csv                     # Ingestion metadata & SHA-256 checksums
│   ├── benchmark_grid.csv                          # 280-run cross-dataset benchmark results
│   ├── benchmark_summary.csv                       # Summary metrics across all models and benchmarks
│   ├── master_paper_audit_table.csv                # Reported vs reproduced vs corrected table
│   ├── novel_hybrids_benchmark.csv                 # Evaluation of novel improvements & hybrids
│   └── *.png                                       # Publication figures
├── WORKLOG.md                                      # Chronological audit worklog
├── FINDINGS.md                                     # Comprehensive scientific findings report
├── README.md                                       # Repository overview & navigation guide
└── requirements.txt                                # Pinned Python dependencies
```

---

## Reproducibility Guide

All experiments are implemented purely in Jupyter Notebooks (`.ipynb`) and execute cleanly top-to-bottom:

```bash
# 1. Set up and activate Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run data ingestion and cryptographic audit
jupyter nbconvert --to notebook --execute --inplace notebooks/00_setup_and_audit.ipynb

# 3. Run individual paper reproductions (01 through 08)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_upama_causal_qml.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_young_qisicgm.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_kotha_diabeats.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_zhang_descn_gif.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_kiran_hqcan.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_sodar_traffic_qml.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/07_yadav_qsvm_qnn.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/08_daga_adaptive_stacking.ipynb

# 4. Run grand cross-dataset benchmark
jupyter nbconvert --to notebook --execute --inplace notebooks/09_cross_dataset_cross_method_benchmark.ipynb

# 5. Run final report & novel improvements
jupyter nbconvert --to notebook --execute --inplace notebooks/10_final_report.ipynb
```

For complete scientific analyses, mathematical proofs, and literature comparison tables, see [FINDINGS.md](file:///C:/Research/Adaptive-Quantum-Kernel-Selection/FINDINGS.md).
