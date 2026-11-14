# %% [markdown]
# # 05 — Kiran et al. (2025) HQCAN: 4-qubit variational quantum classifier vs classical models on the same 4 PCs
#
# **Method (ICICI):** PCA to 4 components → AngleEmbedding(RY) → 2 StronglyEntanglingLayers →
# ⟨Z₀⟩ → logistic output; 80/20 split on Pima (claimed 99.7 % accuracy). Here the same circuit runs
# on all 4 diabetes datasets, next to logistic regression and an RBF-SVM **on the same 4 PCs**, so any
# gap is due to the model, not the features.
#
# **Protocols**
# * `paper` — scaling + PCA fitted on all rows, 80/20 stratified split (seed 42), ungrouped.
# * `clean` — scaling + PCA in-fold; grouped/stratified 5-fold × seeds {0, 1, 42}.
#
# Training is fixed in advance (no tuning on test folds): Adam, lr 0.05, 60 full-batch epochs,
# class-balanced BCE loss, weights seeded per run. Needs `pennylane` + `torch` (cell below installs
# PennyLane if missing; Colab/Kaggle already have torch). Runtime ≈ 10–25 min on CPU.

# %%
import os, sys, subprocess
from pathlib import Path
for p in [Path.cwd(), Path.cwd() / "notebooks2", Path("/content/notebooks2"), *Path("/kaggle/input").glob("*"), *Path("/kaggle/input").glob("*/notebooks2")]:
    if (p / "nb2_common.py").exists():
        sys.path.insert(0, str(p)); break
try:
    import pennylane as qml
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pennylane"]); import pennylane as qml
import torch
from nb2_common import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
sanity_check_kernels()
DATA = load_all()
print("pennylane", qml.__version__, "torch", torch.__version__)

# %%
NQ, LAYERS, EPOCHS, LR = 4, 2, 60, 0.05
dev = qml.device("default.qubit", wires=NQ)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(x, w):
    qml.AngleEmbedding(x, wires=range(NQ), rotation="Y")
    qml.StronglyEntanglingLayers(w, wires=range(NQ))
    return qml.expval(qml.PauliZ(0))


def train_vqc(Xtr, ytr, Xte, seed):
    torch.manual_seed(seed)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=LAYERS, n_wires=NQ)
    w = torch.nn.Parameter(0.1 * torch.randn(shape, dtype=torch.float64))
    a = torch.nn.Parameter(torch.tensor([2.0, 0.0], dtype=torch.float64))  # logit = a0*<Z> + a1
    xt = torch.tensor(Xtr, dtype=torch.float64); yt = torch.tensor(ytr, dtype=torch.float64)
    pos = max(ytr.mean(), 1e-3)
    pw = torch.tensor((1 - pos) / pos, dtype=torch.float64)
    opt = torch.optim.Adam([w, a], lr=LR)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = lossf(a[0] * circuit(xt, w) + a[1], yt)
        loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.sigmoid(a[0] * circuit(torch.tensor(Xte, dtype=torch.float64), w) + a[1]).numpy()
    return (p >= 0.5).astype(int), p


def pcs(X_fit, *others, seed=0):
    """standardise -> PCA(4) -> MinMax [0, pi], all fitted on X_fit."""
    return to_qubits(X_fit, *others, q=NQ, seed=seed)


def kiran_method(Xtr, ytr, Xte, seed, groups_tr, X_prep):
    Qr, Qe = prep(pcs, X_prep, Xtr, Xte, seed=seed)
    out = {"HQCAN VQC (4q, 2 layers)": train_vqc(Qr, ytr, Qe, seed)}
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Qr, ytr)
    out["Logistic Regression (same 4 PCs, balanced)"] = (lr.predict(Qe), lr.predict_proba(Qe)[:, 1])
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed).fit(Qr, ytr)
    out["RBF-SVM (same 4 PCs, balanced)"] = (svm.predict(Qe), svm.decision_function(Qe))
    out["Angle-QSVM (same 4 PCs)"] = svm_precomputed(angle_kernel(Qr, Qr), ytr, angle_kernel(Qe, Qr), seed=seed)
    return out


rec = Recorder("P05_Kiran", "05_kiran", config=dict(
    method=f"VQC: PCA{NQ} -> AngleEmbedding(Y) -> StronglyEntanglingLayers x{LAYERS} -> <Z0>; Adam lr {LR}, {EPOCHS} epochs, balanced BCE",
    paper_protocol="global scaling+PCA, 80/20 stratified split seed 42",
    pennylane=qml.__version__, torch=torch.__version__))
print("paper protocol"); run_paper_holdout(rec, DATA, kiran_method, test_size=0.2, global_prep=True)
print("clean protocol"); run_clean(rec, DATA, kiran_method)
df = rec.save()

# %%
pd.set_option("display.width", 220, "display.max_rows", 200)
summarise(df)

# %% [markdown]
# ## Is the VQC better than LR on the same 4 PCs? Paired per-fold difference (clean protocol)

# %%
c = df[df.protocol == "clean"].pivot_table(index=["dataset", "seed", "fold"], columns="model", values="balanced_accuracy")
diff = (c["HQCAN VQC (4q, 2 layers)"] - c["Logistic Regression (same 4 PCs, balanced)"]).groupby("dataset")
diff.agg(["mean", "std", "min", "max"]).round(3)
