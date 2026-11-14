# Publication-Grade LaTeX Research Package: WDBC Independent Audit of Daga et al. (2026)

## Overview

This directory contains the publication-grade LaTeX manuscript and modular components documenting the complete, independent technical audit and replication of:

> **Daga, Naole, Parikh, Bommineni & Ramu (2026)**  
> *Adaptive quantum kernel selection via leakage-free stacking for clinical diagnostics on NISQ hardware*  
> **Scientific Reports** 16, 26840. DOI: `10.1038/s41598-026-56928-1`  
> Official Repository: `https://github.com/researchbackup27-sudo/Adaptive-Quantum-Kernel-Selection`  
> Verified Local Branch: `reproduction-verified-23c89d3` (Tag: `reproduction-wdbc-v1`)

---

## Directory Structure

```
latex_report/
│
├── main.tex                    # Master LaTeX document
├── references.bib              # Verified 40-entry BibTeX bibliography (no invented DOIs)
├── README_LATEX.md             # This comprehensive documentation and traceability guide
│
├── tex/                        # Modular content chapters
│   ├── macros.tex              # Status macros, color definitions, and TikZ styles
│   ├── diagrams.tex            # 6 publication TikZ diagrams (circuits, workflows, geometry)
│   ├── plots.tex               # 4 high-resolution PGFPlots charts (seed, noise, calibration, fairness)
│   ├── comparison.tex          # 21-row master comparison table & evidence classification map
│   ├── methods.tex             # Mathematical formulations, quantum feature maps, and CV protocol
│   ├── audits.tex              # Deep-dive audits of 4 core implementation discrepancies
│   ├── results.tex             # 7 complete empirical evaluation matrices (reproduction, confusion, multi-seed)
│   ├── discussion.tex          # Scientific synthesis, NISQ implications, and clinical evaluation
│   └── appendix.tex            # Exhaustive 32-test Holm tables, 13-row Gram checks, and compute specs
│
├── figures/                    # Standalone figures directory
└── tables/                     # Standalone supplementary tables directory
```

---

## Detailed Contents: Visualizations & Evaluation Matrices

### 1. Architectural & Methodological Visualizations (`tex/diagrams.tex`)
- **Figure 1 (System Pipeline):** End-to-end flowchart from raw WDBC data ($N=569, D=30$) through StandardScaler, PCA ($d=8$, 92.6% variance), quantum feature maps, Gram matrix generation, OOF base predictions, and balanced logistic stacking.
- **Figure 2 (Audit Workflow & Evidence Hierarchy):** Systematic distinction between paper claims, released source, reproduction, diagnostic audits, and corrected experiments.
- **Figure 3 (Circuit Topology Mismatch):** Side-by-side comparison of the clean ZZ complete graph ($K_8$, 28 pairs, 56 CNOTs) vs. the released noisy linear chain ($P_8$, 7 pairs, 14 CNOTs), highlighting the silent 75% entanglement reduction.
- **Figure 4 (VQC Autograd Severance Flowchart):** Step-by-step trace showing where Python list comprehension drops Autograd graph pointers, resulting in zero gradient updates ($\ell_2 = 0.0$) in the author's code vs. active gradients ($\ell_2 = 0.1018$) in the corrected tensor stacking implementation.
- **Figure 5 (Fold-9 Decision Boundary & Logit Shift):** Geometric visualization of how three benign patients (#288, #505, #537) sharing vote vector `[1, 1, 1, 1, 0]` cross the $P=0.50$ decision boundary from False Negative ($P=0.4303$) to True Positive ($P=0.5882$) under fold-local meta-learner fitting.
- **Figure 6 (XAI Attribution Scalar Cancellation):** Algebraic proof diagram illustrating how $\sum_k |\beta_k|$ cancels out during $L_1$ normalization, proving that reported feature attributions are 100% determined by linear PCA loadings.

### 2. Empirical Plots & Diagnostic Curves (`tex/plots.tex`)
- **Multi-Seed Distribution Chart (PGFPlots):** Bar chart with error bars showing mean $\pm$ std across 10 random seeds. Demonstrates that Classical RBF-SVM ($0.9731 \pm 0.0017$) beats Quantum Stacking ($0.9670 \pm 0.0031$) in 10 out of 10 seeds.
- **NISQ Depolarizing Noise Degradation Curves (PGFPlots):** Two-panel curve tracking Accuracy and Specificity across $p \in \{0.00, 0.05, 0.10\}$. Visualizes the catastrophic collapse of QSVM ZZ to 0.0000 specificity under $5\%$ noise.
- **Probability Calibration Reliability Diagram (PGFPlots):** Predicted probability bins vs. observed benign fraction for Classical RBF vs. Quantum Stacking (raw and Platt-scaled).
- **Classical Fairness Comparison Chart (PGFPlots):** Evaluates 8 classical baselines against Quantum Stacking ($0.9649$) with exact 95% bootstrap confidence intervals.

### 3. Empirical Evaluation Matrices (`tex/results.tex`)
- **Evaluation Matrix 1 (Table 8 Reproduction):** Complete 9-model comparison matrix matching published accuracy ($0.964$), specificity ($0.943$), recall ($0.977$), and precision ($0.967$) to three decimal places.
- **Evaluation Matrix 2 (Full Confusion Matrices):** Full 10-fold cumulative counts ($TN, FP, FN, TP$) for all 9 models on WDBC ($N=569$).
- **Evaluation Matrix 3 (Multi-Seed Dispersion Matrix):** Mean, standard deviation, and range [min, max] across 10 independent stratified splits.
- **Evaluation Matrix 4 (Classical Fairness Comparison Matrix):** Performance, accuracy difference $\Delta$, 95% CI, and Holm-adjusted $p$-values for 8 classical baselines.
- **Evaluation Matrix 5 (Holm-Bonferroni Hypothesis Testing):** Summary of 32 pairwise hypothesis tests against Classical RBF (12 raw significant parametric, 1 Holm-significant; 14 raw significant bootstrap, 6 Holm-significant—all 6 significantly *worse* than RBF).
- **Evaluation Matrix 6 (Probability Calibration Matrix):** Expected Calibration Error (ECE) and Brier scores for uncalibrated and Platt-scaled classifiers.
- **Evaluation Matrix 7 (XAI Base Model Attribution Consistency Matrix):** Attribution rankings across logistic $|\beta|$, SHAP values, and permutation importance.

---

## Data Traceability Matrix

Every single metric and claim in the LaTeX document is directly grounded in stored disk artifacts:

| Document Table / Claim | Source Artifact on Disk |
|---|---|
| Table 8 Reproduction Metrics | `results/metrics/breast_cancer_table_8_comparison.csv` |
| Full 10-Fold Confusion Matrices | `results/confusion_matrices/breast_cancer_confusion_matrices.json` |
| Multi-Seed 10-Split Summary | `audit_results/seed_robustness/seed_summary.csv` |
| Multi-Seed Per-Split Matrix | `audit_results/seed_robustness/seed_results_long.csv` |
| Classical Baselines & Fairness | `audit_results/classical_fairness/classical_fairness_comparison.csv` |
| Tuned Classical Search Spaces | `audit_results/classical_fairness/tuned_classical_search_space.csv` |
| 32-Test Parametric Holm Tests | `audit_results/statistics/holm_rbf_32_tests.csv` |
| 32-Test Bootstrap Holm Tests | `audit_results/statistics/holm_rbf_bootstrap_32_tests.csv` |
| Global vs Fold-Local TOST Equivalence | `audit_results/statistics/global_vs_foldlocal_practical_equivalence.csv` |
| Fold-9 Vote Vector Verification | `audit_results/nested_selection/fold9_vote_pattern_verification.csv` |
| Exhaustive Nested Selection (15 Subsets) | `audit_results/model_selection_nested_audit_wdbc_v2_output.txt` |
| VQC Autograd Gradient Audit | `audit_results/vqc_loss_bug_gradient_audit.csv` |
| VQC Per-Parameter Gradients (48 params)| `audit_results/vqc_parameter_gradients.csv` |
| 13-Row Quantum Gram Matrix Checks | `audit_results/noise_nisq/noisy_gram_matrix_checks.csv` |
| Noisy QSVM Confusion Matrices | `audit_results/noise_nisq/noisy_qsvm_confusion_matrices.csv` |
| Noisy Stacking Meta-Learner Coefficients| `audit_results/noise_nisq/noisy_stacking_coefficients.csv` |
| Clean vs Noisy Circuit Topologies | `audit_results/noise_nisq/circuit_topology_comparison.csv` |
| Probability Calibration (Raw) | `audit_results/calibration/calibration_raw_results.csv` |
| Probability Calibration (Platt-Scaled) | `audit_results/calibration/calibration_platt_results.csv` |
| Reliability Diagram Data Bins | `audit_results/calibration/calibration_reliability_data.csv` |
| XAI Composite Feature Stability | `audit_results/xai/xai_stability_summary.csv` |
| Exhaustive Artifact Integrity Audit | `audit_results/final_qc/exhaustive_artifact_consistency_audit.csv` |

---

## Compilation Guide

### Dependencies
The document relies exclusively on standard CTAN packages available in all modern TeX distributions (TeX Live 2022+, MacTeX, or MiKTeX):
- `geometry`, `amsmath`, `amssymb`, `amsfonts`, `booktabs`, `tabularx`, `longtable`, `multirow`, `cancel`, `tcolorbox`, `tikz`, `pgfplots`, `hyperref`, `cleveref`, `natbib`, `listings`, `siunitx`.

### Standard Compilation Commands

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Latexmk Automated Build

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

---

## Strict Scope & Evidence Boundary Notice

1. **WDBC Benchmark Exclusivity:** All numerical evaluations are strictly bounded to the Wisconsin Diagnostic Breast Cancer dataset ($N=569, D=30$). Parkinson's disease and Pima Indians Diabetes datasets from Daga et al. (2026) are intentionally excluded from this empirical audit.
2. **Protection of Original Research Code:** No original research scripts (`quantum_clinical_github.py`, `base_model_selection_mcnemar.py`, etc.) were modified. All reproduction scripts and audit outputs reside in designated audit directories.
