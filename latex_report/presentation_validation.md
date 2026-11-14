# Presentation Validation & Audit Verification Checklist

This document provides a comprehensive verification checklist for `clinical_qml_audit_presentation.pdf`, certifying that every metric, mathematical bound, figure, and claim is grounded directly in executed Jupyter notebooks, verified CSV files, and audited source code.

---

## 1. Document Overview & Compilation Verification

| Item | Expected Specification | Verified Status |
| :--- | :--- | :--- |
| **Output File** | `latex_report/clinical_qml_audit_presentation.pdf` | **VERIFIED** (853,184 bytes, 37 pages) |
| **Compiler Engine** | MiKTeX pdfTeX 3.141592653-2.6-1.40.29 | **VERIFIED** (Exit code `0`) |
| **Aspect Ratio** | 16:9 Widescreen (`aspectratio=169`) | **VERIFIED** |
| **Color Scheme** | Deep Purple (`#2A1B4E`), Teal (`#0D9488`), Coral (`#E11D48`) | **VERIFIED** |
| **Slide Count** | 28 Narrative Slides + 1 Divider + 8 Technical Appendix Slides (37 Total) | **VERIFIED** |
| **Embedded Figures** | 12 Programmatic Vector PDF Figures (`figures/fig01_...` to `figures/fig12_...`) | **VERIFIED** (All 12 present & legible) |

---

## 2. Scope & Dataset Compliance Verification

- [x] **Clinical Diabetes Exclusivity:** All benchmarking, evaluation, and comparative conclusions are strictly restricted to clinical diabetes mellitus datasets:
  - **D1 Pima Indians Diabetes:** $N=768$, 8 features, 34.90% positive prevalence.
  - **D2 Sylhet Early-Stage Diabetes:** $N=520$, 16 features, 61.54% positive prevalence.
  - **D3 CDC BRFSS Diabetes Health Indicators:** Full $N=253,680$ (stratified clinical sample $N=1,000$ for QML benchmarking).
  - **D4 UCI Diabetes 130-US Hospitals:** Full $N=101,766$ (stratified inpatient sample $N=1,000$ for QML benchmarking).
- [x] **Strict Scope Purge:** Breast Cancer Wisconsin (WDBC), METABRIC, and Parkinson's (UCI 174) datasets have been **strictly purged** from the clinical benchmark.
- [x] **Cross-Domain Adaptation Disclaimers:** Non-diabetes datasets (Sodar's ISCX network traffic and Zhang's Huawei speech audio) are explicitly labeled as cross-domain algorithmic adaptations or `NOT_TESTABLE_HERE`.

---

## 3. Study-by-Study Claim vs. Verification Traceability

| Study & Venue | Published Claim | Verified Metric / Finding | Source Notebook & CSV | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Upama et al.**<br>*(MLWA 2026)* | Sylhet RF Acc: 98.08% / 97.12% | **269 duplicate rows** (51.7%). Random 80/20 gives 98.08%; `StratifiedGroupKFold` drops to **85.77%** (unstratified: **83.46%**). | `R04_upama_causal_qml.ipynb`<br>`paper01_upama_recheck_metrics.csv` | **CONFIRMED DUPLICATE LEAKAGE** |
| **Young**<br>*(arXiv 2509.12259)* | Pima + 2k GMM OOF F1: 0.893--0.940 | Prior shifted from 34.9% to 81.9%. F1 = 0.940 on synthetic folds; on real patients, precision collapses to **50.11%** (F1: **0.6357**). | `R05_young_qisicgm.ipynb`<br>`paper02_young_recheck_metrics.csv` | **CONFIRMED SYNTHETIC SHIFT** |
| **Kotha et al.**<br>*(DiaBeats 2026)* | Claimed ROC-AUC = 1.00 on Pima | Published confusion matrix reports 15 FN, 10 FP. Mann-Whitney bound proves $\text{ROC-AUC} \le \mathbf{0.9700}$. 50-run CV yields **0.8240** (Acc: 76.65%). | `R06_kotha_diabeats.ipynb`<br>`paper03_kotha_recheck_metrics.csv` | **MATHEMATICALLY REFUTED** |
| **Zhang et al.**<br>*(IEEE TFS 2025)* | G-mean: 0.883 on Huawei voice | Audio data unreleased (`NOT_TESTABLE_HERE`). Evaluated GIF mechanism on Pima: at 20% label noise, LightGBM+GIF retains **0.6634** F1 vs 0.6181 (+0.045 gain). | `R07_zhang_descn_gif.ipynb`<br>`paper04_zhang_recheck_metrics.csv` | **NOT TESTABLE (MECHANISM VERIFIED)** |
| **Kiran et al.**<br>*(ICICI 2025)* | Claimed 99.7% headline accuracy | Section 4 reports 4% test error, bounding $\text{Accuracy} \le \mathbf{96.0\%}$. Honest 4-qubit VQC on Pima achieves **64.97%** acc (AUC 0.6917), trailing Logistic Regression (72.14%). | `R08_kiran_hqcan.ipynb`<br>`paper05_kiran_recheck_metrics.csv` | **MATHEMATICALLY REFUTED** |
| **Sodar et al.**<br>*(EPJ QT 2026)* | 99.8% acc on ISCX traffic | Raw amplitude encoding causes rank-1 spectral collapse ($\lambda_1/\sum\lambda = 85.3\%$, **0.00% sensitivity**). Z-score standardization restores rank ($\lambda_1/\sum\lambda = 36.1\%$, 14.18% sensitivity). | `R09_sodar_traffic_qml.ipynb`<br>`paper06_sodar_recheck_metrics.csv` | **CONFIRMED SPECTRAL COLLAPSE** |
| **Yadav et al.**<br>*(PeerJ CS 2026)* | High performance on WDBC | WDBC purged; tested on Pima: Z-map (no entanglement) yields **75.78%** acc; ZZ-map (entangled) drops to **64.97%** default (74.74% tuned) vs Classical RBF **76.17%**. | `R10_yadav_qsvm_qnn.ipynb`<br>`paper07_yadav_recheck_metrics.csv` | **CONFIRMED NO ADVANTAGE** |
| **Daga et al.**<br>*(Sci Rep 2026)* | Table 9 Nested CV on Pima | Full-matrix precomputation creates slight leakage bias (+0.08% AUC in 1,000-bootstrap). Hybrid stacking (AUC **0.836**) does not exceed Classical-only stacking (AUC **0.838**). | `R11_daga_adaptive_stacking.ipynb`<br>`paper08_daga_recheck_metrics.csv` | **CONFIRMED HYBRID = CLASSICAL** |

---

## 4. Discrepancy Reconciliation Tracking

Every variance between preliminary exploratory runs and final verified reporting has been fully reconciled:

1. **Sylhet Deduplication Drop (85.77% vs 83.46%):**
   - *Exploratory value:* 83.46% ($434/520$) was obtained via a single unstratified `GroupKFold` split.
   - *Verified value:* 85.77% ($\pm 1.8\%$) was derived from repeated 5-fold `StratifiedGroupKFold` across 5 independent seeds.
   - *Verdict:* Both prove identical findings: isolating duplicate patient profiles causes a 12.3% to 14.6% accuracy collapse from the 98.08% random-split illusion.

2. **DiaBeats Repeated CV Accuracy (76.65% vs 76.03%):**
   - *Exploratory value:* 76.03% (ROC-AUC 0.8205) was an initial single 5-fold cross-validation seed.
   - *Verified value:* 76.65% (ROC-AUC 0.8240) is the grand mean across 50 repeated 5-fold CV runs in `R06`.
   - *Verdict:* Fully reconciled and documented in Slide 23.

3. **Daga Stacking Precomputation Leakage (+0.08% vs +0.40%):**
   - *Exploratory value:* +0.40% AUC was observed on a single fold split.
   - *Verified value:* 1,000-iteration bootstrap in `R11` established the true leakage bias as +0.08% ROC-AUC ($0.8311$ leaked vs $0.8303$ strictly honest).
   - *Verdict:* Documented in Slide 15 and Slide 23.

4. **CDC Angle-QSVM ROC-AUC (0.5520 vs 0.7634):**
   - *Exploratory value:* 0.7634 was an early exploratory subset run without class-imbalance controls.
   - *Verified value:* 0.5520 (with 10.07% sensitivity) represents the balanced cross-validated metric on the representative stratified CDC clinical sample in `benchmark_grid_v2.csv`.
   - *Verdict:* Fully updated in benchmark tables and figures.

5. **Kiran / Yadav Shared Fraction Anomaly (74.74% vs 64.97%):**
   - *Exploratory observation:* 74.74% represented a discrete rational count ($574/768$) that appeared identically across preliminary reports.
   - *Verified value:* Independent 4-qubit VQC optimization with Adam in `R08` produces distinct, honest metrics: **64.97% accuracy, 0.6917 ROC-AUC**.
   - *Verdict:* Verified and distinguished across slides.

---

## 5. Statistical Rigor & Hypothesis Testing

- [x] **Independence Correction:** All cross-validation comparisons employ the **Nadeau-Bengio corrected resampled t-test**, adjusting variance by $\left(\frac{1}{J \cdot K} + \frac{n_{\text{test}}}{n_{\text{train}}}\right)$ to account for the 80% training set overlap in 5-fold CV.
- [x] **Multiple Testing Adjustment:** All pairwise model comparisons vs. CatBoost are adjusted via the **Holm-Bonferroni step-down procedure** to strictly control the Family-Wise Error Rate (FWER) at $\alpha = 0.05$.
- [x] **Significance Thresholds:** Classical models (CatBoost, LightGBM, Classical Stacking, Logistic Regression) exhibit statistically significant advantages over quantum models (Angle-QSVM $p=0.0014$, Fidelity-QSVM $p=0.0004$, 4-Qubit VQC $p < 0.0001$).

---

## 6. Actionable Clinical Improvements

- [x] **Missing Value Profiling:** Physiological zero values in Pima (Glucose, Blood Pressure, Skinfold, Insulin, BMI) are documented and treated as missing. MICE preserves non-linear covariance structures.
- [x] **Clinical Threshold Optimization:** Moving from $\tau=0.50$ to $\tau=0.35$ increases sensitivity from **58.2% to 81.3%** (+23.1% absolute gain) while maintaining specificity at 74.1%.
- [x] **Decision Curve Analysis (DCA):** Net benefit analysis demonstrates superior clinical utility for calibrated thresholding across all realistic decision thresholds ($p_t \in [0.15, 0.65]$).

---

## 7. Slide Layout & Visual Inspection Certification

- [x] **No Text Clipping / Overflow:** All slides fit within the 16:9 widescreen canvas.
- [x] **Table Scaling:** All data tables are scaled using `\resizebox` with comfortable font sizes.
- [x] **High-Contrast Styling:** Custom Beamer theme uses high-contrast text on white/light-gray backgrounds with distinct colored callout cards.
- [x] **Figure Integration:** All 12 figures are scaled with `height=0.68\textheight, keepaspectratio` to prevent vertical page spills.
- [x] **Page Numbers:** Two-pass compilation ensures `Slide X / 37` footer reflects accurate progress across the deck.
