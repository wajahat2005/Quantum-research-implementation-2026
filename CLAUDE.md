# CLAUDE.md — Adaptive Quantum Kernel Selection (audit / re-implementation)

## What this repository is
An audit and independent re-implementation of Daga et al. (2026), *Adaptive quantum
kernel selection via leakage-free stacking for clinical diagnostics on NISQ hardware*
(Sci. Rep. 16:26840). The released authors' script is `quantum_clinical_github.py`; the
repository's own audit papers are `IEEE_QUANTUM_AUDIT_PAPER*.pdf` and `audit_results/`.

* **Completed work (do not redo or modify):** the Pima diabetes re-implementation in
  `reimplementation/`. Its results are in `reimplementation/results/diabetes/`. Its reports
  are `latex_report/DIABETES_REIMPLEMENTATION_REPORT.tex` and `_v2.tex`, plus the slides.
  All 108 published Table 9 numbers are reproduced, and the 8 checks in
  `reimplementation/verify_independent.py` pass.
* **How the pipeline works:** see `reimplementation/README.md`. The shared code is
  `reimplementation/qks_common.py`.
* **Current work:** the multi-dataset extension. Its plan, decisions and deviations log is
  `reimplementation/PLAN_MULTIDATASET.md`. **Read it before changing anything.**

## Multi-dataset extension: phases
0. Plan — done.
1. Refactor `qks_common.py` behind a dataset registry without changing Pima. The regression gate (`reimplementation/regression_gate.py` + `verify_independent.py`, output redirected to `results/_regression/`) must pass before going on.
2. Per-dataset runs into `reimplementation/results/<dataset>/`:
   * data audit;
   * the full model set (QSVM/QKNN × Angle/ZZ/Amplitude, OOF stacking, hard voting, RBF, tuned RBF, XGBoost, LR, fixed-gradient VQC, closed-form classical kernels);
   * 10 seeds; strictly nested and outer-fold-local preprocessing;
   * bootstrap (B = 10⁴) and parametric CIs, McNemar, Holm;
   * noise test with the benchmark kernel as the 0 % reference;
   * a one-page `summary.md`.
3. Short exploratory cross-dataset section (see the decisions below).
4. Figures, `latex_report/MULTIDATASET_REPORT.tex` (tectonic, no warnings, findings first, in the style of `DIABETES_REIMPLEMENTATION_REPORT_v2.tex`), and new checks in `verify_independent.py`.

Stop and ask the user at the end of every phase, and whenever a decision is not covered
by the plan.

## Rules
* Windows, Python 3.11 in `.venv`. Do not upgrade pinned packages (`requirements.txt`). `ucimlrepo` is not installed; the UCI CSVs are read directly.
* **Never write into `reimplementation/results/diabetes/`, `reimplementation/checkpoints/gram_*_diabetes.npy`, `reimplementation/tex/` or `latex_report/` existing files.** Redirect output with `QKS_RES_DIR`, `QKS_CKP_DIR`, `QKS_TEX_DIR`.
* Pin every seed. Log every run's config to JSON beside its results.
* Never tune on test folds. Never pick settings after seeing test results. Any change to a pre-registered setting goes into `PLAN_MULTIDATASET.md` §8 (deviations log) with the reason.
* Report negative results plainly. Do not drop seeds, datasets or models.
* No hand-typed numbers in reports: every number comes from a result file or a checking script. Values read off figures are labelled as such.
* Raw data in `reimplementation/data/raw/` are pinned by SHA-256 in `qks_common.py`; loaders refuse a changed file.

## Decisions fixed on 2026-10-20 (details: PLAN_MULTIDATASET.md §7)
* **Frankfurt (Kaggle `johndasilva/diabetes`) is excluded as an independent dataset.** 1,981 of its 2,000 rows are identical to Pima rows; it has 744 unique rows, of which only 10 are not Pima rows. It is used only for a 10-fold duplicate-inflation analysis (raw 2,000 vs 744 unique), plus the overlap counts.
* **Headline = within-dataset replication on Sylhet (UCI 529) and CDC BRFSS (UCI 891, cited as the BRFSS 2014 extract, as UCI documents it)** under the same protocol as Pima.
  * The pooled study is dropped, as are Pima ↔ Frankfurt, Pima ↔ Sylhet, Sylhet ↔ CDC and the source-shortcut test.
  * **The only cross-dataset analysis is Pima ↔ CDC-females** on shared features (BMI, age band): exploratory, classical models only, labelled *limited*. No shrunk quantum circuits.
  * A limitations table (plan §7.5) maps the shared features of every pair, to show why the other transfers are not meaningful.
* **CDC label:** the UCI 891 label is used as is (positive = prediabetes or diabetes). The Kaggle 3-class file is not substituted. Every CDC table, figure and summary labels the positive class **"diabetes or prediabetes (self-reported)"**. The Pima ↔ CDC-females section states the outcome mismatch as an extra limitation.
* **Pima** is re-run under the new protocol into `results/pima_multidataset/`.
* **Duplicates.**
  * Sylhet main analysis: all 520 rows with grouped CV (`StratifiedGroupKFold`, group = identical feature vector), outer and inner folds.
  * Sylhet sensitivity: the 251 deduplicated rows.
  * Sylhet leakage demonstration: ungrouped CV on all 520 rows.
  * CDC: 5 disjoint 2,000-row subsamples drawn from the raw rows, stratified on the label at natural prevalence (279 positives each), grouped inside each subsample.
  * Datasets with no repeated feature vector use plain `StratifiedKFold`, so Pima's folds stay unchanged.
* **Seeds:** {0, 1, 2, 3, 4, 5, 10, 42, 100, 999} (outer = inner CV seed). The Holm families use **seed 42**; the other seeds give robustness counts only.
* **Tuning:** 12 configurations per tuned model, selected by inner-CV balanced accuracy. The VQC is untuned in both arms; seeds: all 10 on Pima and Sylhet; on CDC **{0, 1, 2, 42}** (the first three of the list, plus 42 so the secondary "stacking vs VQC" test runs at the Holm seed). CDC VQC descriptive summaries use all 4 seeds; the test uses seed 42 only.
* **Holm:** one family per dataset (the same 32 tests as the Pima audit: 8 models × acc/prec/rec/spec vs faithful RBF), never pooled across datasets. For CDC, the out-of-fold predictions of the 5 subsamples (10,000 rows) are combined and tested once, with the bootstrap stratified on (subsample, label); per-subsample results are descriptive only, with the count of subsamples agreeing in direction. A precision test is dropped (family 32 − k, listed) if the model or RBF predicts no positives; any zero-positive model is reported in `summary.md` as a result (recall 0 vs the floor).
* **Reporting:** recall of the diabetic class is primary, but it is **always** shown with specificity, balanced accuracy, AUC and the majority-class rate as a floor. Sylhet tables include the all-positive baseline (recall 1.0), because the majority class there is diabetic.
* **Preprocessing:** strictly nested = main analysis; outer-fold-local = secondary. Pima-style zeros: counts only, no imputation.
* `verify_independent.py` gets a check that no feature vector appears in both train and test in any fold, for every dataset.

## File ownership
`reimplementation/PLAN_MULTIDATASET.md` and this `CLAUDE.md` are owned by the session
running the multi-dataset work. Other Claude sessions must treat them as read-only.

## Token Efficiency & Budget Optimization (STRICT)
* **Never load large files or raw datasets into context:** Never read raw `.csv`, `.data`, `.pdf`, or binary files directly with `View` or `read_file`. Always inspect schema and data shapes with brief 1-line Python commands (e.g., `pd.read_csv(..., nrows=2)` or `df.info()`).
* **Keep responses concise:** Provide direct solutions, code diffs, or execution summaries without lengthy preambles or repeating large existing files.
* **Suppress verbose command outputs:** When running Python scripts, tests, or bash commands, always suppress or pipe verbose outputs (e.g., `python script.py | select -First 20` or `-q` flags) to avoid flooding the context window.
* **Proactive session management:** Suggest `/compact` or `/clear` to the user when completing a major phase so tokens are not carried over across unrelated tasks.
