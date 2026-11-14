"""nb2_common.py — shared code for notebooks2 (cross-paper x cross-dataset, diabetes only).

Every notebook in notebooks2/ does `from nb2_common import *`.

Contents
--------
* locate data + results folders (works locally, on Colab and on Kaggle)
* SHA-256-pinned loaders for the 4 diabetes datasets (D1 Pima, D2 Sylhet, D3 CDC, D4 130-Hospitals)
* CV splitters: `clean_splits` (grouped when feature vectors repeat) and `paper_split`
* in-fold preprocessing helpers
* exact numpy quantum-kernel simulators (Angle, Z, ZZ, Amplitude) — noiseless, no shots
* metrics, bootstrap CI, Nadeau-Bengio corrected t-test, Holm correction
* result recording to notebooks2/results/<file>.csv + a JSON config beside it
"""
import os, sys, json, glob, hashlib, time, platform, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import hadamard

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef, roc_auc_score,
                             average_precision_score, confusion_matrix)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants (pre-registered; change only with a note in notebooks2/README.md)
# ─────────────────────────────────────────────────────────────────────────────
CLEAN_SEEDS = [0, 1, 42]          # outer-CV seeds for the leakage-free protocol
PAPER_SEED = 42                   # seed for each paper's own single split
N_SPLITS = 5                      # outer folds (clean protocol)
SUBSAMPLE_N = 2000                # rows drawn from CDC and 130-Hospitals
SUBSAMPLE_SEED = 42
MAX_QUBITS = 8                    # quantum encodings use q = min(d, 8); PCA in-fold if d > 8
DATASETS = ["pima", "sylhet", "cdc", "hosp130"]
DATASET_LABEL = {
    "pima": "D1 Pima (diabetes, OGTT)",
    "sylhet": "D2 Sylhet (early-stage diabetes, symptoms)",
    "cdc": "D3 CDC BRFSS 2014 (diabetes or prediabetes, self-reported)",
    "hosp130": "D4 130-Hospitals (diabetic patients, readmitted <30 d)",
}
SHA256 = {
    "pima": "6bfe5d0f379d17a0e0819b996407e3c09bf80febd4287f2ed212190dfff154af",
    "sylhet": "f6bf14c5d2e939feccd78750da2542f131b78d80355ccf35f29554b1f03fcb97",
    "cdc": "9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964",
    "hosp130": "0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97",
}
RELPATH = {
    "pima": "pima/pima-indians-diabetes.csv",
    "sylhet": "sylhet/early_stage_diabetes_uci529.csv",
    "cdc": "cdc/cdc_diabetes_health_indicators_uci891.csv",
    "hosp130": "diabetes_130_hospitals/diabetic_data.csv",
}


# ─────────────────────────────────────────────────────────────────────────────
# Locations
# ─────────────────────────────────────────────────────────────────────────────
def find_data_root():
    """Return the folder that contains pima/, sylhet/, cdc/, diabetes_130_hospitals/."""
    cands = []
    if os.environ.get("NB2_DATA"):
        cands.append(Path(os.environ["NB2_DATA"]))
    here = Path.cwd()
    cands += [here / "data", here.parent / "data", here.parent.parent / "data",
              Path("/content/data"), Path("/content/notebooks2/data"),
              Path("/content/drive/MyDrive/notebooks2/data")]
    cands += [Path(p).parent.parent for p in glob.glob("/kaggle/input/**/pima-indians-diabetes.csv", recursive=True)]
    for c in cands:
        if (c / RELPATH["pima"]).exists():
            return c.resolve()
    raise FileNotFoundError(
        "Data folder not found. Upload the project's `data/` folder (pima/, sylhet/, cdc/, "
        "diabetes_130_hospitals/) next to the notebooks, or set os.environ['NB2_DATA'].")


def find_results_dir():
    p = Path(os.environ.get("NB2_RESULTS", Path.cwd() / "results"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checked_path(name):
    path = find_data_root() / RELPATH[name]
    got = sha256_of(path)
    if got != SHA256[name]:
        raise ValueError(f"{name}: SHA-256 mismatch ({got}); refusing a changed data file.")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Loaders. Each returns a dict:
#   X (float ndarray, NaN allowed), y (int 0/1), feature_names, groups (identical-feature-vector
#   id), has_dups (bool), name, label, positive (text), majority_rate
# ─────────────────────────────────────────────────────────────────────────────
def _finish(name, X, y, feats, positive, note=""):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    keys = pd.DataFrame(np.nan_to_num(X, nan=-9999.0)).astype(str).agg("|".join, axis=1)
    groups = pd.factorize(keys)[0]
    has_dups = len(np.unique(groups)) < len(groups)
    return dict(name=name, label=DATASET_LABEL[name], X=X, y=y, feature_names=list(feats),
                groups=groups, has_dups=bool(has_dups), n_unique=int(len(np.unique(groups))),
                positive=positive, majority_rate=float(max(y.mean(), 1 - y.mean())), note=note)


def _subsample(df, ycol, n=SUBSAMPLE_N, seed=SUBSAMPLE_SEED):
    idx, _ = train_test_split(np.arange(len(df)), train_size=n, stratify=df[ycol].values,
                              random_state=seed)
    return df.iloc[np.sort(idx)].reset_index(drop=True)


def load_pima():
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age", "Outcome"]
    df = pd.read_csv(_checked_path("pima"), header=None, names=cols)
    # zeros-as-missing are left as recorded (counts only, no imputation), as in the audit
    return _finish("pima", df[cols[:-1]].values, df["Outcome"].values, cols[:-1], "diabetes")


def load_sylhet():
    df = pd.read_csv(_checked_path("sylhet"))
    y = (df["class"].str.strip().str.lower() == "positive").astype(int).values
    X = df.drop(columns=["class"]).copy()
    X["gender"] = (X["gender"].str.strip().str.lower() == "male").astype(int)
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = (X[c].str.strip().str.lower() == "yes").astype(int)
    return _finish("sylhet", X.values, y, X.columns, "diabetes (early stage)",
                   note="many identical feature vectors -> grouped CV in the clean protocol")


def load_cdc():
    df = pd.read_csv(_checked_path("cdc")).drop(columns=["ID"])
    df = _subsample(df, "Diabetes_binary")
    y = df["Diabetes_binary"].astype(int).values
    X = df.drop(columns=["Diabetes_binary"])
    return _finish("cdc", X.values, y, X.columns, "diabetes or prediabetes (self-reported)",
                   note=f"stratified {SUBSAMPLE_N}-row subsample, seed {SUBSAMPLE_SEED}")


_AGE_MID = {f"[{a}-{a+10})": a + 5 for a in range(0, 100, 10)}


def load_hosp130():
    df = pd.read_csv(_checked_path("hosp130"), low_memory=False)
    # one encounter per patient (first encounter) so no patient appears twice
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    df["y"] = (df["readmitted"] == "<30").astype(int)
    df = _subsample(df, "y")
    y = df["y"].values
    df = df.drop(columns=["encounter_id", "patient_nbr", "readmitted", "y",
                          "weight", "payer_code", "medical_specialty"])  # >40 % missing / IDs
    df = df.replace("?", np.nan)
    df["age"] = df["age"].map(_AGE_MID)
    for c in ["diag_1", "diag_2", "diag_3"]:  # ICD-9 code -> number; V/E codes -> NaN
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if df[c].dtype == object:
            # label-independent ordinal coding (category order is alphabetical, no target used)
            df[c] = pd.Categorical(df[c]).codes.astype(float)
            df.loc[df[c] < 0, c] = np.nan
    return _finish("hosp130", df.values, y, df.columns, "readmitted within 30 days",
                   note=f"first encounter per patient; stratified {SUBSAMPLE_N}-row subsample, seed {SUBSAMPLE_SEED}")


LOADERS = {"pima": load_pima, "sylhet": load_sylhet, "cdc": load_cdc, "hosp130": load_hosp130}


def load_dataset(name):
    return LOADERS[name]()


def load_all():
    out = {}
    for n in DATASETS:
        d = load_dataset(n)
        out[n] = d
        print(f"{n:8s} N={len(d['y']):5d} d={d['X'].shape[1]:3d} prevalence={d['y'].mean():.3f} "
              f"unique={d['n_unique']:5d} grouped_cv={d['has_dups']}  [{d['positive']}]")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Splits
# ─────────────────────────────────────────────────────────────────────────────
def clean_splits(ds, seed, n_splits=N_SPLITS):
    """Leakage-free outer CV. Grouped (identical feature vector never on both sides)
    when the dataset has repeated vectors; otherwise plain stratified K-fold."""
    X, y = ds["X"], ds["y"]
    if ds["has_dups"]:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(X, y, groups=ds["groups"]))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X, y))


def inner_splits(y, groups, seed, n_splits=3):
    """Inner CV used for tuning / stacking inside a training fold."""
    if len(np.unique(groups)) < len(groups):
        return list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    .split(np.zeros(len(y)), y, groups=groups))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                .split(np.zeros(len(y)), y))


def paper_split(ds, test_size=0.2, seed=PAPER_SEED):
    """A paper-style random stratified hold-out split (ungrouped: duplicates may straddle it)."""
    idx = np.arange(len(ds["y"]))
    return train_test_split(idx, test_size=test_size, stratify=ds["y"], random_state=seed)


def leak_count(ds, tr, te):
    """How many test rows have an identical feature vector in the training rows."""
    g_tr = set(ds["groups"][tr])
    return int(sum(g in g_tr for g in ds["groups"][te]))


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing (always fit on the rows passed as `fit`)
# ─────────────────────────────────────────────────────────────────────────────
def impute_scale(X_fit, *others, kind="standard"):
    imp = SimpleImputer(strategy="median").fit(X_fit)
    sc = (StandardScaler() if kind == "standard" else MinMaxScaler((0, np.pi))).fit(imp.transform(X_fit))
    outs = [sc.transform(imp.transform(X_fit))] + [sc.transform(imp.transform(o)) for o in others]
    if kind != "standard":
        outs = [np.clip(o, 0, np.pi) for o in outs]
    return outs


def to_qubits(X_fit, *others, q=MAX_QUBITS, seed=0):
    """Standardise -> (PCA to q if d > q) -> MinMax to [0, pi]. All fitted on X_fit only."""
    Z = impute_scale(X_fit, *others, kind="standard")
    if Z[0].shape[1] > q:
        pca = PCA(n_components=q, random_state=seed).fit(Z[0])
        Z = [pca.transform(z) for z in Z]
    mm = MinMaxScaler((0, np.pi)).fit(Z[0])
    return [np.clip(mm.transform(z), 0, np.pi) for z in Z]


# ─────────────────────────────────────────────────────────────────────────────
# Quantum kernels — exact, noiseless simulation (fidelity kernels |<phi(a)|phi(b)>|^2)
# ─────────────────────────────────────────────────────────────────────────────
def angle_kernel(A, B, scale=1.0):
    """RY/RX angle embedding (product state): K = prod_i cos^2(scale*(a_i - b_i)/2)."""
    out = np.ones((len(A), len(B)))
    for i in range(A.shape[1]):
        out *= np.cos(scale * (A[:, [i]] - B[None, :, i]) / 2.0) ** 2
    return out


def _signs(n):
    bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[None, ::-1]) & 1
    return 1.0 - 2.0 * bits  # (2^n, n) Z eigenvalues


def pauli_states(X, reps=2, scale=1.0, entangle="linear", pairs=True):
    """Statevectors of the Havlicek ZZ feature map (Qiskit ZZFeatureMap convention):
    reps x [ H^n ; exp(-i * (sum_i phi_i Z_i + sum_(i,j) phi_ij Z_i Z_j)) ],
    phi_i = scale*x_i, phi_ij = scale*(pi - x_i)(pi - x_j).  pairs=False gives the Z map."""
    X = np.asarray(X, float)
    n = X.shape[1]
    S = _signs(n)
    H = hadamard(2 ** n) / np.sqrt(2 ** n)
    if entangle == "full":
        P = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        P = [(i, i + 1) for i in range(n - 1)]
    phase = scale * X @ S.T                                   # (N, 2^n)
    if pairs:
        for (i, j) in P:
            phase += scale * np.outer((np.pi - X[:, i]) * (np.pi - X[:, j]), S[:, i] * S[:, j])
    D = np.exp(-1j * phase)
    psi = np.zeros((len(X), 2 ** n), complex)
    psi[:, 0] = 1.0
    for _ in range(reps):
        psi = (psi @ H.T) * D
    return psi


def state_kernel(PA, PB):
    return np.abs(PA @ PB.conj().T) ** 2


def zz_kernel(A, B, reps=2, scale=1.0, entangle="linear"):
    return state_kernel(pauli_states(A, reps, scale, entangle), pauli_states(B, reps, scale, entangle))


def z_kernel(A, B, reps=2, scale=1.0):
    return state_kernel(pauli_states(A, reps, scale, pairs=False), pauli_states(B, reps, scale, pairs=False))


def amplitude_kernel(A, B):
    """Amplitude encoding of the L2-normalised (zero-padded) vector: K = (a.b)^2/(|a|^2|b|^2)."""
    na = np.linalg.norm(A, axis=1, keepdims=True); na[na == 0] = 1
    nb = np.linalg.norm(B, axis=1, keepdims=True); nb[nb == 0] = 1
    return np.clip(((A / na) @ (B / nb).T) ** 2, 0, 1)


def svm_precomputed(K_tr, y_tr, K_te, C=1.0, seed=0):
    from sklearn.svm import SVC
    m = SVC(kernel="precomputed", C=C, random_state=seed).fit(K_tr, y_tr)
    return m.predict(K_te), m.decision_function(K_te)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics & statistics
# ─────────────────────────────────────────────────────────────────────────────
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc",
           "roc_auc", "pr_auc", "n_pred_pos"]


def metrics(y, pred, score=None):
    y = np.asarray(y).astype(int); pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two = len(np.unique(y)) == 2
    s = None if score is None else np.asarray(score, float)
    ok = two and s is not None and np.all(np.isfinite(s))
    return dict(
        accuracy=accuracy_score(y, pred),
        balanced_accuracy=balanced_accuracy_score(y, pred),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        specificity=tn / (tn + fp) if (tn + fp) else np.nan,
        f1=f1_score(y, pred, zero_division=0),
        mcc=matthews_corrcoef(y, pred),
        roc_auc=roc_auc_score(y, s) if ok else np.nan,
        pr_auc=average_precision_score(y, s) if ok else np.nan,
        n_pred_pos=int(pred.sum()), n_test=int(len(y)), n_pos_test=int(y.sum()),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def bootstrap_ci(y, pred, score=None, metric="accuracy", B=2000, seed=42, alpha=0.05):
    rng = np.random.RandomState(seed)
    y = np.asarray(y); pred = np.asarray(pred)
    vals = []
    for _ in range(B):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(metrics(y[i], pred[i], None if score is None else np.asarray(score)[i])[metric])
    vals = np.asarray(vals, float)
    return float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def nadeau_bengio(diffs, n_train, n_test):
    """Corrected resampled t-test (Nadeau & Bengio 2003) on per-fold paired differences."""
    d = np.asarray(diffs, float); d = d[np.isfinite(d)]
    k = len(d)
    if k < 2:
        return np.nan, np.nan
    v = np.var(d, ddof=1) * (1 / k + n_test / n_train)
    if v <= 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt(v)
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]; m = len(q)
    order = np.argsort(q)
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * q[order]))
    tmp = np.empty(m); tmp[order] = adj
    out[ok] = tmp
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Recording
# ─────────────────────────────────────────────────────────────────────────────
class Recorder:
    """Collects one row per (dataset, protocol, seed, fold, model) and saves CSV + config JSON."""

    def __init__(self, paper, file_stem, config=None):
        self.paper, self.stem, self.rows = paper, file_stem, []
        self.config = dict(config or {})
        self.t0 = time.time()

    def add(self, dataset, protocol, seed, fold, model, y, pred, score=None, **extra):
        r = dict(paper=self.paper, dataset=dataset, protocol=protocol, seed=seed, fold=fold, model=model)
        r.update(metrics(y, pred, score)); r.update(extra)
        self.rows.append(r)

    def add_all(self, dataset, protocol, seed, fold, y, outputs, **extra):
        """outputs: {model_name: (pred, score)}"""
        for m, (p, s) in outputs.items():
            self.add(dataset, protocol, seed, fold, m, y, p, s, **extra)

    def frame(self):
        return pd.DataFrame(self.rows)

    def save(self):
        rd = find_results_dir()
        df = self.frame()
        df.to_csv(rd / f"{self.stem}.csv", index=False)
        cfg = dict(paper=self.paper, clean_seeds=CLEAN_SEEDS, paper_seed=PAPER_SEED, n_splits=N_SPLITS,
                   subsample_n=SUBSAMPLE_N, subsample_seed=SUBSAMPLE_SEED, max_qubits=MAX_QUBITS,
                   python=sys.version.split()[0], numpy=np.__version__, sklearn=sklearn.__version__,
                   platform=platform.platform(), runtime_s=round(time.time() - self.t0, 1),
                   n_rows=len(df), data_sha256=SHA256)
        cfg.update(self.config)
        with open(rd / f"{self.stem}_config.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        print(f"saved {rd / (self.stem + '.csv')}  ({len(df)} rows)")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# Generic runners.  method(Xtr, ytr, Xte, seed, groups_tr, X_prep) -> {model: (pred, score)}
#   X_prep = rows the preprocessing is fitted on (training fold = clean; all rows = paper leak)
# ─────────────────────────────────────────────────────────────────────────────
def prep(transform, X_prep, Xtr, Xte, **kw):
    """Fit `transform` on X_prep, return transformed (Xtr, Xte).
    (Public name on purpose: `from nb2_common import *` skips names starting with '_'.)"""
    out = transform(X_prep, Xtr, Xte, **kw)
    return out[1], out[2]


def run_clean(rec, data, method, seeds=CLEAN_SEEDS, datasets=DATASETS, n_splits=N_SPLITS):
    for name in datasets:
        ds, t0 = data[name], time.time()
        for seed in seeds:
            for k, (tr, te) in enumerate(clean_splits(ds, seed, n_splits)):
                Xtr, Xte = ds["X"][tr], ds["X"][te]
                out = method(Xtr, ds["y"][tr], Xte, seed, ds["groups"][tr], Xtr)
                rec.add_all(name, "clean", seed, k, ds["y"][te], out, leak_rows=leak_count(ds, tr, te))
        print(f"  clean  {name:8s} done in {time.time() - t0:6.1f}s")


def run_paper_holdout(rec, data, method, test_size=0.2, global_prep=True, datasets=DATASETS,
                      seed=PAPER_SEED, protocol="paper"):
    """Single random stratified hold-out split, ungrouped, as most papers did.
    global_prep=True fits scalers/PCA on ALL rows (the leak present in the papers' code)."""
    for name in datasets:
        ds = data[name]
        tr, te = paper_split(ds, test_size, seed)
        X_prep = ds["X"] if global_prep else ds["X"][tr]
        out = method(ds["X"][tr], ds["y"][tr], ds["X"][te], seed, ds["groups"][tr], X_prep)
        rec.add_all(name, protocol, seed, 0, ds["y"][te], out, leak_rows=leak_count(ds, tr, te))
        print(f"  {protocol:6s} {name:8s} test rows sharing a feature vector with train: {leak_count(ds, tr, te)}")


def run_paper_cv(rec, data, method, global_prep=True, datasets=DATASETS, seed=PAPER_SEED,
                 n_splits=N_SPLITS, protocol="paper"):
    """Ungrouped stratified K-fold with (optionally) global preprocessing, as in Daga / Young."""
    for name in datasets:
        ds = data[name]
        for k, (tr, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed)
                                     .split(ds["X"], ds["y"])):
            X_prep = ds["X"] if global_prep else ds["X"][tr]
            out = method(ds["X"][tr], ds["y"][tr], ds["X"][te], seed, ds["groups"][tr], X_prep)
            rec.add_all(name, protocol, seed, k, ds["y"][te], out, leak_rows=leak_count(ds, tr, te))
        print(f"  {protocol:6s} {name:8s} done")


def summarise(df,metrics_=("accuracy", "balanced_accuracy", "recall", "specificity", "roc_auc")):
    """Mean +- SD over seeds x folds for each dataset x protocol x model."""
    g = df.groupby(["dataset", "protocol", "model"])[list(metrics_)]
    m, s = g.mean(), g.std()
    out = m.round(3).astype(str) + " ± " + s.round(3).astype(str)
    out["n_runs"] = g.size()
    return out.reset_index()


def sanity_check_kernels():
    """Cheap self-test: kernels are PSD-ish, have unit diagonal and match closed forms."""
    rng = np.random.RandomState(0)
    A = rng.uniform(0, np.pi, (6, 3))
    for name, K in [("angle", angle_kernel(A, A)), ("zz", zz_kernel(A, A)),
                    ("z", z_kernel(A, A)), ("amp", amplitude_kernel(A + 0.1, A + 0.1))]:
        assert np.allclose(np.diag(K), 1, atol=1e-9), name
        assert np.allclose(K, K.T, atol=1e-9), name
        assert np.linalg.eigvalsh(K).min() > -1e-8, name
    # Z map with reps=1 is a product state: K = prod cos^2(a_i - b_i)
    K1 = z_kernel(A, A, reps=1)
    K2 = np.prod(np.cos(A[:, None, :] - A[None, :, :]) ** 2, axis=2)
    assert np.allclose(K1, K2, atol=1e-9), "z reps=1 closed form"
    print("kernel self-test passed")


print(f"nb2_common loaded: numpy {np.__version__}, sklearn {sklearn.__version__}")
