# Autonomous Audit, Reproduction, and Cross-Dataset Benchmark of Quantum Machine Learning in Clinical Diagnostics

A rigorous, reproducible machine learning research audit evaluating **eight landmark Quantum Machine Learning (QML) and Healthcare AI publications (2025–2026)** with distinct, dedicated clinical domain implementations for **Clinical Diabetes** and **Breast Cancer (WDBC)**.

---

## Clinical Domain Separation

To provide clean separation between clinical conditions, this repository is organized into dedicated domain trees under `clinical_domains/`:

```
clinical_domains/
├── breast_cancer/                     # Wisconsin Diagnostic Breast Cancer (WDBC) Benchmark
│   ├── checkpoints/                   # Quantum Gram matrices (Amplitude, Angle, ZZ-Map)
│   ├── results/                       # Table 8 reproduction, bootstrap & parametric CIs, confusion matrices
│   ├── reports/                       # WDBC_AUDIT_REPORT.pdf
│   └── README.md                      # Detailed WDBC audit summary & analysis
│
└── diabetes/                          # Multi-Dataset Clinical Diabetes Benchmark
    ├── publications_and_reports/      # IEEE Multi-Dataset Paper, Slides, and Synthesis Reports
    ├── reproduction_code/             # Python scripts replicating Papers 01 through 08
    ├── benchmark_results/             # Grand cross-dataset evaluation matrix (Pima, Sylhet, CDC, Hosp 130)
    └── README.md                      # Detailed diabetes audit findings & mathematical derivations
```

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
├── clinical_domains/                               # Dedicated clinical disease domains
│   ├── breast_cancer/                              # WDBC dataset, Gram matrices, Table 8 audit
│   └── diabetes/                                   # Diabetes publications, scripts, metrics
├── data/                                           # Processed clinical benchmarks
│   ├── pima/                                       # D1: Pima Indians Diabetes (768 rows, 8 features)
│   ├── sylhet/                                     # D2: Early Stage Diabetes (UCI 529, 520 rows)
│   ├── cdc/                                        # D3: CDC BRFSS Diabetes Indicators (UCI 891)
│   └── diabetes_130_hospitals/                     # D4: Diabetes 130-US Hospitals (UCI 296)
├── latex_report/                                   # LaTeX source files & compiled publications
│   ├── IEEE_MULTI_DATASET_PAPER.pdf                # Multi-dataset IEEE Transactions publication
│   ├── MULTI_DATASET_SLIDES.pdf                    # 22-slide conference presentation (16:9 widescreen)
│   ├── MASTER_MULTI_DATASET_REPORT.pdf             # Extended multi-dataset technical synthesis
│   └── figures/                                    # Publication vector plots and TikZ assets
├── notebooks/                                      # Executed Jupyter reproduction notebooks (00 to 10)
├── notebooks2/                                     # Phase 2 audit pipelines and scripts
│   ├── src/                                        # Python implementation modules (01 to 09)
│   └── results/                                    # Structured CSV results and configs
├── results/                                        # Comprehensive benchmark outputs
│   ├── breast_cancer/                              # WDBC metrics and confusion matrices
│   ├── diabetes/                                   # Cross-dataset diabetes benchmarks
│   ├── confusion_matrices/                         # Raw confusion matrix records
│   ├── extracted_figs/                             # Author figure comparisons
│   └── metrics/                                    # Per-paper and bootstrap metric tables
├── FINDINGS.md                                     # Exhaustive scientific findings & proofs
├── WORKLOG.md                                      # Audit execution logs
├── README.md                                       # Repository overview & navigation guide
└── requirements.txt                                # Pinned Python dependencies
```

---

## Reproducibility Guide

All experiments are reproducible using the provided Python scripts and Jupyter Notebooks:

```bash
# 1. Set up and activate Python environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run data ingestion and cryptographic audit
jupyter nbconvert --to notebook --execute --inplace notebooks/00_setup_and_audit.ipynb

# 3. Run individual paper reproductions (01 through 08)
python notebooks2/src/01_upama_causal_qml.py
python notebooks2/src/02_young_qisicgm.py
python notebooks2/src/03_kotha_diabeats.py
python notebooks2/src/04_zhang_gif_label_noise.py
python notebooks2/src/05_kiran_hqcan_vqc.py
python notebooks2/src/06_sodar_angle_amplitude.py
python notebooks2/src/07_yadav_z_zz_qsvm.py
python notebooks2/src/08_daga_adaptive_stacking.py

# 4. Generate synthesis metrics and master comparison tables
python notebooks2/src/09_summary.py
```

For complete scientific analyses, mathematical derivations, and literature comparison tables, see [FINDINGS.md](FINDINGS.md).
