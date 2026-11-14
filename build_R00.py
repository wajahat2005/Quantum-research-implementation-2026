
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

cells = []

# Cell 0: Markdown header
cells.append(new_markdown_cell("""# R00 — Freeze Inventory
**Stage 0 of Recheck Pipeline**

Inventories all tracked files, extracts numeric claims from archive_old documents,
checks for out-of-scope Gram pickle files, and writes WORKLOG_RECHECK.md.
"""))

# Cell 1: Setup directories
cells.append(new_code_cell("""
import os, hashlib, re, csv, json, glob, warnings
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r"C:\\Research\\Adaptive-Quantum-Kernel-Selection")
RECHECK_DIR   = ROOT / "recheck"
RESULTS_DIR   = RECHECK_DIR / "results"
NOTEBOOKS_DIR = RECHECK_DIR / "notebooks"

for d in [RESULTS_DIR, NOTEBOOKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("Directories ready:")
print(f"  {RECHECK_DIR}")
print(f"  {RESULTS_DIR}")
print(f"  {NOTEBOOKS_DIR}")
"""))

# Cell 2: File inventory
cells.append(new_code_cell("""
def sha256_file(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"ERROR:{e}"

def scan_dir(base_path, rel_prefix=""):
    rows = []
    p = ROOT / base_path
    if not p.exists():
        print(f"  SKIP (not found): {p}")
        return rows
    if p.is_file():
        stat = p.stat()
        rows.append({
            "path": str(Path(rel_prefix) / p.name) if rel_prefix else base_path,
            "size_bytes": stat.st_size,
            "sha256": sha256_file(p),
            "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })
    else:
        for fp in sorted(p.rglob("*")):
            if fp.is_file():
                stat = fp.stat()
                rows.append({
                    "path": str(fp.relative_to(ROOT)),
                    "size_bytes": stat.st_size,
                    "sha256": sha256_file(fp),
                    "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                })
    return rows

SCAN_TARGETS = [
    "notebooks",
    "results",
    "archive_old",
    "papers",
    "FINDINGS.md",
    "WORKLOG.md",
    "README.md",
    "requirements.txt",
    "latex_report",
]

all_rows = []
for target in SCAN_TARGETS:
    rows = scan_dir(target)
    print(f"  {target}: {len(rows)} files")
    all_rows.extend(rows)

inventory_df = pd.DataFrame(all_rows, columns=["path","size_bytes","sha256","mtime"])
inventory_df.to_csv(RESULTS_DIR / "file_inventory.csv", index=False)
print(f"\\nTotal files inventoried: {len(inventory_df)}")
print(f"Saved: {RESULTS_DIR / 'file_inventory.csv'}")
"""))

# Cell 3: Parse claims from archive_old markdown files
cells.append(new_code_cell("""
DATASET_PATTERNS = {
    "Pima":      r"(?i)pima|pima\\s+indian",
    "Sylhet":    r"(?i)sylhet",
    "CDC":       r"(?i)cdc|brfss",
    "130-Hosp":  r"(?i)130[- ]hosp|diabetic.*130|hospital.*130",
}
PROTOCOL_PATTERNS = {
    "random_split": r"(?i)random.?split|train.?test.?split|hold.?out",
    "grouped_cv":   r"(?i)group(?:ed|s)?\\s+(?:cross.?val|cv|k.?fold)",
    "5fold":        r"(?i)5[- ]fold|five[- ]fold",
}
METRIC_PATTERNS = {
    "accuracy":    r"(?i)accur",
    "auc":         r"(?i)\\bauc\\b|roc.?auc|area.under",
    "f1":          r"(?i)\\bf1\\b|f.?measure|f.?score",
    "sensitivity": r"(?i)sensitiv|recall|true.?positive.?rate",
    "specificity": r"(?i)specif|true.?negative.?rate",
}

def infer_dataset(text):
    for ds, pat in DATASET_PATTERNS.items():
        if re.search(pat, text):
            return ds
    return "Unknown"

def infer_protocol(text):
    for proto, pat in PROTOCOL_PATTERNS.items():
        if re.search(pat, text):
            return proto
    return "unknown"

def infer_metric(text):
    for metric, pat in METRIC_PATTERNS.items():
        if re.search(pat, text):
            return metric
    return "other"

def get_surrounding_sentence(text, match):
    start = text.rfind('.', 0, match.start())
    if start == -1:
        start = 0
    else:
        start += 1
    end = text.find('.', match.end())
    if end == -1:
        end = len(text)
    else:
        end += 1
    sentence = text[start:end].strip()
    return sentence[:300]  # cap at 300 chars

NUM_RE = re.compile(r'\\b(\\d+\\.\\d+|0\\.\\d+|\\d{2,3}\\.\\d+)\\b')

def parse_md_claims(filepath):
    rows = []
    try:
        content = Path(filepath).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return rows
    
    # Track current section heading
    current_section = "Unknown"
    lines = content.split("\\n")
    for i, line in enumerate(lines):
        heading_m = re.match(r'^(#{1,4})\\s+(.*)', line)
        if heading_m:
            current_section = heading_m.group(2).strip()
        
        for m in NUM_RE.finditer(line):
            num = float(m.group(0))
            if num < 0.01 or num > 200:
                continue  # skip trivially irrelevant
            claim_text = get_surrounding_sentence(content, m)
            rows.append({
                "source_file": str(Path(filepath).relative_to(ROOT)),
                "section": current_section,
                "claim_text": claim_text,
                "number": num,
                "dataset": infer_dataset(claim_text),
                "protocol": infer_protocol(claim_text),
                "metric": infer_metric(claim_text),
            })
    return rows

claim_rows = []

for md_file in ["archive_old/FINDINGS.md", "archive_old/WORKLOG.md"]:
    fp = ROOT / md_file
    if fp.exists():
        rows = parse_md_claims(fp)
        print(f"  {md_file}: {len(rows)} numeric claims extracted")
        claim_rows.extend(rows)
    else:
        print(f"  SKIP (not found): {md_file}")

print(f"\\nMD claims so far: {len(claim_rows)}")
"""))

# Cell 4: Parse CSV files
cells.append(new_code_cell("""
def parse_csv_claims(filepath):
    rows = []
    try:
        df = pd.read_csv(filepath, dtype=str)
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return rows
    
    rel = str(Path(filepath).relative_to(ROOT))
    
    for col in df.columns:
        for idx, val in df[col].items():
            if pd.isna(val):
                continue
            for m in NUM_RE.finditer(str(val)):
                num = float(m.group(0))
                if num < 0.01 or num > 200:
                    continue
                # build a context string from the whole row
                row_ctx = " | ".join(str(v) for v in df.loc[idx].values if pd.notna(v))
                rows.append({
                    "source_file": rel,
                    "section": col,
                    "claim_text": row_ctx[:300],
                    "number": num,
                    "dataset": infer_dataset(row_ctx),
                    "protocol": infer_protocol(row_ctx),
                    "metric": infer_metric(row_ctx),
                })
    return rows

CSV_TARGETS = [
    "archive_old/master_paper_audit_table.csv",
    "archive_old/benchmark_summary.csv",
    "archive_old/novel_hybrids_benchmark.csv",
]

for csv_file in CSV_TARGETS:
    fp = ROOT / csv_file
    if fp.exists():
        rows = parse_csv_claims(fp)
        print(f"  {csv_file}: {len(rows)} numeric claims extracted")
        claim_rows.extend(rows)
    else:
        print(f"  SKIP (not found): {csv_file}")

print(f"\\nTotal claims (MD + CSV): {len(claim_rows)}")
"""))

# Cell 5: Save claims CSV
cells.append(new_code_cell("""
claims_df = pd.DataFrame(claim_rows, columns=[
    "source_file","section","claim_text","number","dataset","protocol","metric"
])
claims_df.to_csv(RESULTS_DIR / "old_claims_inventory.csv", index=False)
print(f"Saved: {RESULTS_DIR / 'old_claims_inventory.csv'}")
print(f"Shape: {claims_df.shape}")
"""))

# Cell 6: Summary
cells.append(new_code_cell("""
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total files inventoried : {len(inventory_df)}")
print(f"Total old numeric claims: {len(claims_df)}")
print()
print("Claims by dataset:")
print(claims_df["dataset"].value_counts().to_string())
print()
print("Claims by metric:")
print(claims_df["metric"].value_counts().to_string())
print()
print("Claims by source_file:")
print(claims_df["source_file"].value_counts().to_string())
"""))

# Cell 7: Check out-of-scope Gram pickles
cells.append(new_code_cell("""
print("=" * 60)
print("OUT-OF-SCOPE GRAM PICKLE CHECK")
print("=" * 60)

oos_patterns = [
    "results/gram_BreastCancer*.pkl",
    "results/gram_Parkinsons*.pkl",
]

oos_found = []
for pat in oos_patterns:
    matches = sorted(glob.glob(str(ROOT / pat)))
    for m in matches:
        rel = str(Path(m).relative_to(ROOT))
        sz = Path(m).stat().st_size
        oos_found.append({"path": rel, "size_bytes": sz})
        print(f"  ⚠  OUT-OF-SCOPE: {rel}  ({sz:,} bytes)")

if not oos_found:
    print("  None found.")
else:
    print(f"\\n  Total out-of-scope Gram pickles: {len(oos_found)}")
    oos_df = pd.DataFrame(oos_found)
    oos_df.to_csv(RESULTS_DIR / "oos_gram_pickles.csv", index=False)
    print(f"  Saved: {RESULTS_DIR / 'oos_gram_pickles.csv'}")
"""))

# Cell 8: Write WORKLOG_RECHECK.md
cells.append(new_code_cell("""
now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

worklog_content = f\"\"\"# WORKLOG_RECHECK.md
## Stage 0 — Freeze Inventory
**Date:** {now_str}
**Notebook:** recheck/notebooks/R00_freeze_inventory.ipynb

### Actions Performed
- Created directory structure: recheck/, recheck/results/, recheck/notebooks/
- Inventoried {len(inventory_df)} files across: notebooks/, results/, archive_old/, papers/, FINDINGS.md, WORKLOG.md, README.md, requirements.txt, latex_report/
- Saved file hashes and metadata to: recheck/results/file_inventory.csv
- Extracted {len(claims_df)} numeric claims from archive_old MD and CSV files
- Saved claims to: recheck/results/old_claims_inventory.csv
- Checked for out-of-scope Gram pickles (BreastCancer, Parkinsons): {len(oos_found)} found

### Key Findings
- Out-of-scope Gram pickles present: {[x['path'] for x in oos_found] if oos_found else 'None'}
- Claims by dataset: {claims_df['dataset'].value_counts().to_dict()}
- Claims by metric: {claims_df['metric'].value_counts().to_dict()}

### Status
Stage 0 COMPLETE. Ready for Stage 1 (data integrity verification).
\"\"\"

worklog_path = RECHECK_DIR / "WORKLOG_RECHECK.md"
worklog_path.write_text(worklog_content, encoding="utf-8")
print(f"Written: {worklog_path}")
print()
print(worklog_content)
"""))

# Cell 9: Final confirmation
cells.append(new_code_cell("""
print("=" * 60)
print("R00 COMPLETE")
print("=" * 60)
print(f"  file_inventory.csv    : {len(inventory_df)} rows")
print(f"  old_claims_inventory  : {len(claims_df)} rows")
print(f"  oos_gram_pickles      : {len(oos_found)} found")
print(f"  WORKLOG_RECHECK.md    : written")
print()
# Also save a run manifest
manifest = {
    "notebook": "R00_freeze_inventory",
    "run_time_utc": datetime.now(tz=timezone.utc).isoformat(),
    "files_inventoried": len(inventory_df),
    "claims_extracted": len(claims_df),
    "oos_pickles": len(oos_found),
}
import json
with open(RESULTS_DIR / "R00_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest saved: {RESULTS_DIR / 'R00_manifest.json'}")
"""))

nb.cells = cells

from pathlib import Path as _Path
import nbformat as _nbformat

_ROOT = _Path(r"C:\Research\Adaptive-Quantum-Kernel-Selection")
nb_path = _ROOT / "recheck" / "notebooks" / "R00_freeze_inventory.ipynb"

_nbformat.write(nb, str(nb_path))
print(f"Notebook written: {nb_path}")
