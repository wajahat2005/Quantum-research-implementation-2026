# Final Presentation and Reports Distribution Bundle
## Independent Scientific Audit of Clinical AI and Quantum Machine Learning

This bundle contains the complete, verified, publication-grade deliverables resulting from the exhaustive independent recheck and audit of eight clinical AI and quantum machine learning (QML) studies.

---

## 📦 Contents of this Distribution Bundle

1. **Research Presentation PDF:**
   - [`clinical_qml_audit_presentation.pdf`](clinical_qml_audit_presentation.pdf) (853 KB, 37 pages): Complete 16:9 widescreen LaTeX Beamer presentation with custom deep-purple/teal theme, embedding all 12 vector figures and comprehensive cross-dataset benchmark tables.
2. **Comprehensive Audit Report PDF:**
   - [`corrected_report.pdf`](corrected_report.pdf) (168 KB): The complete 10-section audited manuscript compiled with real outputs from executed notebooks.
3. **Beamer Presentation Source:**
   - [`clinical_qml_audit_presentation.tex`](clinical_qml_audit_presentation.tex): Full LaTeX source code for the presentation.
4. **BibTeX References:**
   - [`presentation_references.bib`](presentation_references.bib): Verified bibliographic entries for all eight audited studies.
5. **Traceable Results Summary:**
   - [`verified_results_summary.csv`](verified_results_summary.csv): Machine-readable study-by-study matrix mapping published claims, verified metrics, source notebooks, and final verdicts.
6. **Validation & Reconciliation Checklist:**
   - [`presentation_validation.md`](presentation_validation.md): Detailed verification audit checklist, page inspections, and discrepancy resolutions.
7. **Reproduction & Build Guide:**
   - [`BUILD_INSTRUCTIONS.md`](BUILD_INSTRUCTIONS.md): Step-by-step instructions for recompiling the slides and re-generating figures on Windows.
8. **Markdown Audit Manuscript & Discrepancy Log:**
   - [`CORRECTED_REPORT.md`](CORRECTED_REPORT.md): Complete textual report with markdown formatting.
   - [`DISCREPANCY_LOG.csv`](DISCREPANCY_LOG.csv): Core audit log of all 15 reconciled methodological discrepancies.
9. **Programmatic Vector Figures (`figures/`):**
   - `fig01_audit_pipeline.pdf`: 5-gate scientific verification pipeline.
   - `fig02_sylhet_leakage.pdf`: Sylhet 269-row duplicate leakage breakdown (98.08% random vs 85.77% / 83.46% grouped).
   - `fig03_young_gmm_shift.pdf`: GMM prior shift and real-patient precision collapse (50.11%).
   - `fig04_kotha_confusion_bound.pdf`: Mathematical refutation of ROC-AUC = 1.00 via Mann-Whitney bound ($\le 0.9700$).
   - `fig05_kiran_hqcan_vqc.pdf`: HQCAN mathematical bound ($\le 96.0\%$) and honest 4-qubit VQC vs classical LR (64.97% vs 72.14%).
   - `fig06_amplitude_spectral_collapse.pdf`: Amplitude encoding rank-1 spectral collapse ($\lambda_1/\sum\lambda = 85.3\%$) and standardization repair.
   - `fig07_yadav_z_vs_zz.pdf`: Z-map vs ZZ-map entanglement sweep and classical RBF comparison (75.78% vs 64.97% / 74.74% vs 76.17%).
   - `fig08_daga_stacking_ci.pdf`: Stacking bootstrap 95% CIs and precomputation leakage audit.
   - `fig09_cross_benchmark_auc.pdf`: Cross-dataset ROC-AUC comparison across D1--D4.
   - `fig10_threshold_tradeoff.pdf`: Clinical screening threshold tuning ($\tau=0.35$, 81.3% sensitivity).
   - `fig11_missingness_dca.pdf`: MICE missingness analysis and Decision Curve Analysis (DCA) net benefit.
   - `fig12_verdicts_summary.pdf`: Categorical synthesis of all eight study verdicts.

---

## 🎯 Verification Standard
- **Zero `.py` files:** Every number was computed and verified in executed `.ipynb` Jupyter notebooks (`recheck/notebooks/`).
- **Scope Compliance:** Restricted strictly to clinical diabetes mellitus (Pima $N=768$, Sylhet $N=520$, CDC BRFSS $N=253,680$, UCI 130-Hosp $N=101,766$). Breast Cancer and Parkinson's datasets were strictly purged.
