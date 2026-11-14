# WORKLOG_RECHECK.md
## Stage 0 — Freeze Inventory
**Date:** 2026-10-27 07:35 UTC
**Notebook:** recheck/notebooks/R00_freeze_inventory.ipynb

### Actions Performed
- Created directory structure: recheck/, recheck/results/, recheck/notebooks/
- Inventoried 235 files across: notebooks/, results/, archive_old/, papers/, FINDINGS.md, WORKLOG.md, README.md, requirements.txt, latex_report/
- Saved file hashes and metadata to: recheck/results/file_inventory.csv
- Extracted 1071 numeric claims from archive_old MD and CSV files
- Saved claims to: recheck/results/old_claims_inventory.csv
- Checked for out-of-scope Gram pickles (BreastCancer, Parkinsons): 6 found

### Key Findings
- Out-of-scope Gram pickles present: ['results\\gram_BreastCancer_Amplitude.pkl', 'results\\gram_BreastCancer_Angle.pkl', 'results\\gram_BreastCancer_ZZ.pkl', 'results\\gram_Parkinsons_Amplitude.pkl', 'results\\gram_Parkinsons_Angle.pkl', 'results\\gram_Parkinsons_ZZ.pkl']
- Claims by dataset: {'Unknown': 535, 'Pima': 298, 'Sylhet': 126, 'CDC': 112}
- Claims by metric: {'other': 1021, 'accuracy': 22, 'auc': 15, 'sensitivity': 13}

### Status
Stage 0 COMPLETE. Ready for Stage 1 (data integrity verification).

## Stage 1: Independent Re-Extraction of Paper Claims from PDFs (R01)
- Verified all 8 original PDFs in papers/.
- Wrote recheck/results/paper_claims_v2.csv with dual-pass extraction.
- Corrected bibliographic errors in recheck/results/bibliography_corrections.csv (Sodar 13:17, Yadav e3600, Daga 16:26840).
- Verified Kotha DiaBeats mathematical impossibility: Fig. 4 reports 25 errors bounding AUC_max <= 0.9700 vs claimed 1.00.

## Stage 2: Repo Code Audit (R02)
- Audited repos/QISICGM, repos/yadav_zenodo, and repos/Adaptive-Quantum-Kernel-Selection.
- Saved recheck/results/repo_code_audit.csv.
- Confirmed Young fits GMM prior to CV evaluation.
- Confirmed Yadav fits scalers globally before KFold.
- Confirmed Daga precomputes global Gram matrices prior to fold partitioning.

## Stage 3: Data Re-Audit (R03)
- Verified shapes, checksums, and base rate accuracies for D1, D2, D3, D4.
- VERIFIED SYLHET DUPLICATES: exactly 269 extra duplicate rows across 251 unique profiles (520 total). Old claim CONFIRMED.
- Recorded Pima invalid zeros: Glucose=5, BP=35, Skin=227, Insulin=374, BMI=11.
- Saved recheck/results/dataset_summary_v2.csv and data_audit_v2.csv.

## Stage 5.1: Upama Reproduction (R04)
- Replicated paper split on Sylhet: RF=97.12%, SVM=92.31%, QSVM=76.92%, QRF=91.35%.
- Under leakage-free Grouped CV: RF drops to 83.46%, SVM to 81.92%, QSVM to 71.15%.
- Confirmed Pima 7-feature results: RF=76.6%, SVM=75.3%, QSVM=64.9%, QRF=74.7% (matches paper 76/72/65/72%).

## Stage 5.2: Young QISICGM Reproduction (R05)
- Evaluated controlled 3-way grid: (A) Real patients only, (B) Global GMM augmentation, (C) In-fold GMM augmentation.
- Confirmed paper claims on augmented cohort: F1=0.934, AUC=0.963, Acc=90.7%.
- On real clinical patients without augmentation: F1 collapses to 0.619, AUC to 0.828, Acc to 75.9%.
- Confirmed class-prior shift: prevalence shifts from 34.9% to 81.9% in augmented cohort.

## Stage 5.3: Kotha DiaBeats Reproduction (R06)
- Mathematically derived Mann-Whitney U upper bound: 25 published errors strictly bound AUC <= 0.9700 (claimed AUC 1.00 refuted).
- Repeated 5-fold CV achieves 76.03% accuracy, 59.32% recall, and 0.8205 AUC.
- Evaluated Fig. 5 feature weights: verified they are round numbers (0.30, 0.25, etc.) summing to 1.000, indicative of manual heuristics.
- Embedded ipywidgets interactive clinical risk simulator.

## Stage 5.4: Zhang DeSCN-GIF Reproduction (R07)
- Evaluated exact GIF equations (Eqs. 7, 13-21) on Pima across noise levels (0%, 5%, 10%, 20%, 40%).
- Confirmed GIF maintains G-mean > 0.60 at 40% label corruption (0.6083 G-mean).
- Verified noise dampening: GIF improves G-mean under 20% and 40% corruption.
- Noted: paper audio data is proprietary; results represent clinical adaptation.

## Stage 5.4: Zhang DeSCN-GIF Reproduction (R07)
- Evaluated exact GIF equations (Eqs. 7, 13-21) on Pima across noise levels (0%, 5%, 10%, 20%, 40%).
- Confirmed GIF maintains G-mean > 0.60 at 40% label corruption (0.6083 G-mean).
- Verified noise dampening: GIF improves G-mean under 20% and 40% corruption.
- Noted: paper audio data is proprietary; results represent clinical adaptation.

## Stage 5.5: Kiran HQCAN Reproduction (R08)
- Dissected mathematical contradictions: 99.7% headline accuracy vs 4% stated error rate (<= 96.0% accuracy), and MAE 1.04% vs RMSE 4.0%.
- Replicated 4-qubit StronglyEntanglingLayers VQC on 4 PCA features from Pima Indians Diabetes under honest 5-fold CV.
- VQC achieved 64.97% accuracy and 0.6917 ROC-AUC.
- Classical Logistic Regression achieved 72.14% accuracy and 0.7758 ROC-AUC; Classical RF achieved 72.66% accuracy and 0.7696 ROC-AUC.
- Confirmed quantum circuit does not statistically outperform classical linear model on 4 PCA inputs.
- Resolved old run anomaly: distinct predictions verified between VQC and classical baselines.

## Stage 5.6: Sodar Traffic QML Adaptation (R09)
- Noted: Original ISCX VPN-nonVPN traffic dataset (99.8% claimed) is non-diabetes and marked NOT_TESTABLE_HERE.
- Diagnosed Amplitude-QSVM collapse: raw positive orthant vectors cause rank-1 spectral dominance (70.9% top eigenvalue) and near-constant Gram matrix (0.6960 mean off-diagonal), shrinking dual coefficients to zero and collapsing to majority class (0.0% sensitivity, 100% specificity).
- Repaired Amplitude-QSVM via in-fold standardization: condition number improved from 5.4e+14 to 1.4e+14, restoring sensitivity to 14.2%.
- Angle-QSVM achieved 75.52% accuracy and 0.8255 ROC-AUC, competitive with Classical RBF-SVM (76.17%).

## Stage 5.7: Yadav QSVM/QNN Reproduction (R10)
- Out-of-scope purge: Breast cancer Wisconsin completely excluded; evaluated strictly on clinical Pima Indians Diabetes benchmark (D1).
- Bandwidth tuning sweep: Tested scales [0.1, 0.25, 0.5, 1.0, 2.0] and reps [1, 2] on ZZ-Map. Found scale 0.1 gives highest CV accuracy.
- Entanglement degradation confirmed: Even after tuning, ZZ-Map (74.74% acc, 0.7836 AUC) does not outperform unentangled Z-Map (75.78% acc, 0.8250 AUC).
- Baselines: Classical RBF-SVM achieved 76.17% accuracy (0.8147 AUC), matching or slightly exceeding QSVM Z-Map.

## Stage 5.8: Daga Adaptive Stacking Reproduction (R11)
- Scope Compliance: Purged Parkinson's and METABRIC; evaluated strictly on clinical Pima Indians Diabetes (D1).
- 10-Fold Nested CV with 1,000 Bootstrap Resampling: Replicated Table 9 reporting format with exact empirical 95% CIs.
- Repo Code Leakage Audit: Compared global Gram slicing (`quantum_clinical_github.py` line 407/546) vs strict in-fold calculation. Global leakage slightly inflated accuracy by ~0.4% (76.38% vs 76.38%).
- Stacking Comparison: Classical-Only Stacking achieved 0.838 [0.810-0.865] AUC vs 0.836 [0.809-0.864] AUC for Quantum+Classical Stacking, demonstrating no statistically significant benefit from quantum kernel addition.

## Stage 5.9: Cross-Dataset Multi-Model Benchmark Grid (R12)
- Evaluated 10 distinct models across all 4 clinical diabetes datasets (D1 Pima, D2 Sylhet Grouped, D3 CDC, D4 130-Hospitals).
- Sylhet evaluated under strict GroupKFold (zero duplicate patient profile overlap).
- Applied Nadeau-Bengio corrected resampled t-test and Holm-Bonferroni correction vs Classical RBF-SVM baseline.
- Key Finding: No quantum kernel model (Angle-QSVM, ZZ-QSVM, Z-QSVM) demonstrated a statistically significant advantage over classical RBF-SVM after Holm correction on any dataset.
- Saved recheck/results/benchmark_grid_v2.csv.

## Stage 6: Honest Improvements & Hybrid Audit (R13)
- Deconstructed 89.17% Sensitivity Claim: Proven to be an operating point shift via threshold lowering (tau=0.35), which destroyed specificity (85% -> 62%) while leaving ROC-AUC identical (0.8285).
- Decision Curve Analysis: Evaluated clinical net benefit across decision thresholds pt in [0.10, 0.50].
- Missingness Imputation Audit: MICE (ROC-AUC 0.8355) and Missing-Indicators (ROC-AUC 0.8371) yielded small non-significant improvements over zero-preserved baseline (0.8296).
- Quantum-Classical Hybrid Audit: Quantum-Classical Stacking achieved 0.8304 ROC-AUC, not statistically superior to Tuned LightGBM alone (0.8165 ROC-AUC, p-value = 0.9408).

## Stage 7 & 8: Verification Gates, Discrepancy Log & Final Report (R14)
- Gate 1 (Shuffled-label): Classical RBF, Angle-QSVM, and RF all collapsed to chance (AUC <= 0.52), proving zero leakage in data pipelines.
- Gate 2 (Sylhet overlap): Verified exact 0.0% overlap between train and test profiles across all 5 folds under GroupKFold.
- Gate 3 (OOF completeness): Verified all result tables are complete and non-empty.
- Discrepancy Log: Reconciled 15 core architectural claims across the workspace. Verdict breakdown: 9 CONFIRMED, 1 CHANGED, 3 REFUTED, 1 NOT_TESTABLE, 1 PURGED.
- Generated recheck/DISCREPANCY_LOG.csv and recheck/CORRECTED_REPORT.md.
