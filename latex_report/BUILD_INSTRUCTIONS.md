# Compilation and Reproduction Guide: Clinical AI & QML Audit Presentation

This document outlines the exact environment, dependencies, and shell commands required to reproduce, compile, and validate the 37-slide research presentation `clinical_qml_audit_presentation.pdf`.

---

## 1. System Requirements & Environment

- **Operating System:** Windows 10 / 11 (or Linux/macOS with standard TeX Live).
- **TeX Engine:** MiKTeX (pdfTeX 3.141592653-2.6-1.40.29 / MiKTeX 26.5 or later).
- **Python Environment:** Python 3.10+ with Jupyter, NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, LightGBM, CatBoost, Qiskit 1.2.0, PennyLane 0.38.0.
- **Shell:** PowerShell or Bash.

---

## 2. Directory Structure

Ensure the following directory structure is in place within the workspace `Adaptive-Quantum-Kernel-Selection`:

```text
Adaptive-Quantum-Kernel-Selection/
├── latex_report/
│   ├── clinical_qml_audit_presentation.tex    # Main Beamer source
│   ├── clinical_qml_audit_presentation.pdf    # Compiled slide deck
│   ├── presentation_references.bib            # BibTeX references for 8 papers
│   ├── verified_results_summary.csv           # Traceable results across 8 papers
│   ├── BUILD_INSTRUCTIONS.md                  # This compilation guide
│   ├── presentation_validation.md             # Verification checklist & reconciliation
│   └── figures/                               # 12 programmatic vector PDF figures
│       ├── fig01_audit_pipeline.pdf
│       ├── fig02_sylhet_leakage.pdf
│       ├── fig03_young_gmm_shift.pdf
│       ├── fig04_kotha_confusion_bound.pdf
│       ├── fig05_kiran_hqcan_vqc.pdf
│       ├── fig06_amplitude_spectral_collapse.pdf
│       ├── fig07_yadav_z_vs_zz.pdf
│       ├── fig08_daga_stacking_ci.pdf
│       ├── fig09_cross_benchmark_auc.pdf
│       ├── fig10_threshold_tradeoff.pdf
│       ├── fig11_missingness_dca.pdf
│       └── fig12_verdicts_summary.pdf
├── recheck/
│   ├── notebooks/                             # Executed audit notebooks R00 to R16
│   ├── results/                               # Verified metrics CSV files
│   ├── CORRECTED_REPORT.md                    # Comprehensive audit report
│   └── DISCREPANCY_LOG.csv                    # Empirical reconciliation records
└── final_presentation_and_reports_bundle/     # Consolidated distribution bundle
```

---

## 3. Step 1: Re-generating Figures from Verified CSVs

All 12 vector figures are generated programmatically from executed notebook results in `recheck/results/`. To re-generate all figures from scratch:

```powershell
# Navigate to the workspace root
cd C:\Research\Adaptive-Quantum-Kernel-Selection

# Execute the figure generator notebook via Jupyter nbconvert
jupyter nbconvert --to notebook --execute recheck/notebooks/R16_figure_generator.ipynb --inplace
```

This updates all vector PDF figures in `latex_report/figures/`.

---

## 4. Step 2: Compiling the LaTeX Beamer Presentation

Compile the presentation using `pdflatex`. Two passes are required to resolve Beamer frame counters, section headers, and cross-references.

### Pass 1: Initial Typesetting & Auxiliary Generation

```powershell
cd C:\Research\Adaptive-Quantum-Kernel-Selection\latex_report
pdflatex -interaction=nonstopmode clinical_qml_audit_presentation.tex
```

### Pass 2: Final Resolution of Total Frame Count & Page Numbers

```powershell
pdflatex -interaction=nonstopmode clinical_qml_audit_presentation.tex
```

Expected output:
- `clinical_qml_audit_presentation.pdf` (37 pages, approx. 850 KB, exit code `0`).

---

## 5. Troubleshooting & Common TeX Warnings

1. **Unescaped Alignment Characters:** In Beamer, `&` characters in frame titles must be written as `\&`.
2. **Path Resolution:** Figures are referenced as `figures/figXX_...pdf`. Always execute `pdflatex` from inside the `latex_report/` directory or provide `-output-directory=latex_report`.
3. **Overfull Boxes:** Tables have been scaled using `\resizebox{\linewidth}{!}` and `\resizebox{0.96\textwidth}{!}` to fit the 16:9 widescreen layout without clipping.

---

## 6. Reproducing the Complete Audit Pipeline

To re-execute the entire scientific verification pipeline from Stage 0 through Stage 6:

```powershell
cd C:\Research\Adaptive-Quantum-Kernel-Selection

# Sequential execution of verification suite
jupyter nbconvert --to notebook --execute recheck/notebooks/R00_freeze_inventory.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R01_paper_claims_reextraction.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R02_repo_code_audit.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R03_data_reaudit.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R04_upama_causal_qml.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R05_young_qisicgm.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R06_kotha_diabeats.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R07_zhang_descn_gif.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R08_kiran_hqcan.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R09_sodar_traffic_qml.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R10_yadav_qsvm_qnn.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R11_daga_adaptive_stacking.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R12_cross_grid.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R13_improvements.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R14_discrepancy_and_report.ipynb --inplace
jupyter nbconvert --to notebook --execute recheck/notebooks/R16_figure_generator.ipynb --inplace
```
