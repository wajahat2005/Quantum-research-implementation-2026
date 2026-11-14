# Wisconsin Diagnostic Breast Cancer (WDBC) Quantum Machine Learning Audit

## 1. Overview & Clinical Cohort
- **Dataset**: Wisconsin Diagnostic Breast Cancer (WDBC / UCI Machine Learning Repository)
- **Cohort Size**: 569 patients (357 Benign, 212 Malignant)
- **Features**: 30 continuous real-valued features computed from digitized images of fine needle aspirates (FNA) of breast masses (mean, standard error, and "worst" values for: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).
- **Clinical Task**: Binary classification between benign tissue and malignant breast carcinomas.

---

## 2. Directory Structure
```
clinical_domains/breast_cancer/
├── checkpoints/
│   ├── gram_amp_breast_cancer.npy       # Quantum Amplitude Encoding Gram matrix (N=569 x 569)
│   ├── gram_angle_breast_cancer.npy     # Quantum Angle Encoding Gram matrix (N=569 x 569)
│   └── gram_zz_breast_cancer.npy        # Quantum Entangled ZZ Feature Map Gram matrix (N=569 x 569)
├── results/
│   ├── breast_cancer_bootstrap.csv      # 1,000-fold bootstrap resampled confidence intervals
│   ├── breast_cancer_parametric.csv     # Parametric normal-theory confidence intervals
│   ├── breast_cancer_table_8_comparison.csv # Exact Table 8 reproduction & discrepancy audit
│   └── breast_cancer_confusion_matrices.json # Per-kernel out-of-fold confusion matrices
└── reports/
    └── WDBC_AUDIT_REPORT.pdf            # Standalone comprehensive audit report for WDBC
```

---

## 3. Key Findings & Quantum Kernel Comparison (Table 8 Reproduction)

Auditing the reported baseline in **Daga et al. (2026)** against independent exact reproduction:

| Quantum Kernel / Model | Published Table 8 Accuracy | Independent Re-implementation Accuracy | 95% Bootstrap CI | Sensitivity (Recall) | Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Angle Encoding QSVM** | **97.37%** | **96.49%** | [94.74%, 98.25%] | 95.28% | 97.20% |
| **ZZ Feature Map QSVM** | 94.74% | 94.74% | [92.11%, 96.49%] | 92.45% | 96.08% |
| **Amplitude Encoding QSVM** | 91.23% | 91.23% | [88.60%, 93.86%] | 87.74% | 93.28% |
| **Classical RBF-SVM** | 97.37% | 97.37% | [95.61%, 98.25%] | 96.23% | 98.04% |
| **Classical Random Forest** | 96.49% | 96.49% | [94.74%, 98.25%] | 95.28% | 97.20% |

### Key Clinical Observations:
1. **Classical Parity**: Classical RBF-SVM achieves 97.37% accuracy, matching or slightly exceeding the best quantum kernel (Angle QSVM at 96.49%--97.37%), with zero quantum runtime overhead.
2. **Entanglement Penalty**: Entangling 2-qubit phase gates (ZZ-Map) underperform unentangled Angle encoding by $-1.75\%$ due to high-frequency phase over-rotation on tabular biomarker data.
3. **Amplitude Normalization Loss**: Amplitude encoding suffers a $-5.26\%$ performance degradation compared to Angle encoding, as $L_2$ unit normalization erases absolute tumor dimensions (e.g., perimeter and area magnitude).
