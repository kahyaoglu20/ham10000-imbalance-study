"""train_step.py NAME EPOCHS_THIS_CALL TARGET_TOTAL
Checkpointed trainer for the real HAM10000 cache (64x64).

Loads the saved checkpoint (model state, optimizer state, history) for NAME
if present; trains EPOCHS_THIS_CALL more epochs (capped at TARGET_TOTAL);
saves checkpoint back to disk. Produces test-set predictions JSON when the
target epoch count is reached.
"""
import sys, os, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support)

CACHE = Path("/sessions/loving-nice-maxwell/mnt/outputs/ham_cache")
RES   = Path("/sessions/loving-nice-maxwell/mnt/outputs/real_results")
CKPT  = RES / "ckpt"
RES.mkdir(parents=True, exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(2)

# ---------- Load 64x64 cache (fast) ----------
ranges = [(0,1500),(1500,3000),(3000,4500),(4500,6000),(6000,7500),(7500,9000),(9000,10015)]
N = 10015; NC = 7
X = np.empty((N, 3, 64, 64), dtype=np.uint8)
for s, e in ranges: X[s:e] = np.load(CACHE / f"chunk64_{s:05d}_{e:05d}.npy")
y = np.load(CACHE / "y_all.npy")
manifest = json.loads((CACHE / "manifest.json").read_text())
CLASSES = manifest["classes"]

# ---------- Stratified 70/15/15 split (deterministic) ----------
rng = np.random.default_rng(SEED)
tr=[]; va=[]; te=[]
for c in range(NC):
    ix = np.where(y == c)[0]; rng.shuffle(ix); n = len(ix)
    n1, n2 = int(round(n*0.70)), int(round(n*0.15))
    tr += list(ix[:n1]); va += list(ix[n1:n1+n2]); te += list(ix[n1+n2:])
rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
tr, va, te = np.array(tr), np.array(va), np.array(te)
np.save(RES / "test_idx.npy", te)
train_counts = np.bincount(y[tr], minlength=NC).tolist()

# ---------- Dataset ----------
MEAN = np.array([0.485,0.456,0.406], dtype=np.float32).reshape(3,1,1)
STD  = np.array([0.229,0.224,0.225], dtype=np.float32).reshape(3,1,1)

class DS(Dataset):
    def __init__(s, idx, augment=False):
        s.idx, s.aug = idx, augment
    def __len__(s): return len(s.idx)
    def __getitem__(s, i):
        gi = s.idx[i]
        img = X[gi].astype(np.float32) / 255.0
        if s.aug and np.random.rand() < 0.5: img = img[:, :, ::-1].copy()
        if s.aug and np.random.rand() < 0.5: img = img[:, ::-1, :].copy()
        img = (img - MEAN) / STD
        return torch.from_numpy(img.astype(np.float32)), int(y[gi])

# ---------- Model: 4-block CNN, ~106k params ----------
class SkinCNN(nn.Module):
    def __init__(s, nc=NC):
        super().__init__()
        def b(ic, oc):
            return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        s.features = nn.Sequential(b(3,16), b(16,32), b(32,64), b(64,128))
        s.gap = nn.AdaptiveAvgPool2d(1); s.dr = nn.Dropout(0.4)
        s.fc1 = nn.Linear(128, 64); s.fc2 = nn.Linear(64, nc)
    def forward(s, x):
        x = s.features(x); x = s.gap(x).flatten(1); x = s.dr(x)
        x = F.relu(s.fc1(x)); x = s.dr(x); return s.fc2(x)

# ---------- Loss helpers ----------
def w_inv(c):
    c = np.array(c, dtype=np.float64); w = 1/c; return torch.tensor(w*(len(c)/w.sum()), dtype=torch.float32)
def w_cb(c, beta=0.999):
    c = np.array(c, dtype=np.float64); e = 1-np.power(beta,c); w=(1-beta)/e
    return torch.tensor(w*(len(c)/w.sum()), dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(s, gamma=2.0, alpha=None): super().__init__(); s.g=gamma; s.a=alpha
    def forward(s, l, t):
        lp = F.log_softmax(l, 1); lpt = lp.gather(1, t.unsqueeze(1)).squeeze(1); pt = lpt.exp()
        fl = (1 - pt) ** s.g * (-lpt)
        if s.a is not None: fl = s.a.to(l.device)[t] * fl
        return fl.mean()

WI, WC = w_inv(train_counts), w_cb(train_counts)
sw = np.array([1/train_counts[int(c)] for c in y[tr]], dtype=np.float64)
sampler = WeightedRandomSampler(torch.from_numpy(sw), num_samples=len(sw), replacement=True)

CONFIGS = {
    "Standard CE":       (nn.CrossEntropyLoss(),               None),
    "Weighted CE":       (nn.CrossEntropyLoss(weight=WI),      None),
    "Focal (gamma=2)":   (FocalLoss(2.0, alpha=WI),            None),
    "Class-Balanced CE": (nn.CrossEntropyLoss(weight=WC),      None),
    "Std CE + Sampler":  (nn.CrossEntropyLoss(),               sampler),
}

# ---------- Resume / train one or two more epochs ----------
NAME = sys.argv[1]
EPOCHS_NOW = int(sys.argv[2])
TARGET     = int(sys.argv[3])
slug = NAME.replace(" ","_").replace("(","").replace(")","").replace("=","").replace("+","plus")
crit, sm = CONFIGS[NAME]

ckpt_path = CKPT / f"{slug}.pt"
hist_path = CKPT / f"{slug}_hist.json"

torch.manual_seed(SEED)
model = SkinCNN()
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
hist = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
done_epochs = 0
if ckpt_path.exists() and hist_path.exists():
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
    hist = json.loads(hist_path.read_text()); done_epochs = len(hist["train_loss"])
    print(f"Resuming from epoch {done_epochs}.")
else:
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Fresh model. Total params: {n_params:,}")

train_ds = DS(tr, augment=True); val_ds = DS(va, augment=False)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
if sm is None:
    loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
else:
    loader = DataLoader(train_ds, batch_size=128, sampler=sm, num_workers=0)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ys, ps, ls = [], [], []
    for x, t in loader:
        o = model(x); ls.append(F.cross_entropy(o, t).item() * x.size(0))
        ys.append(t.numpy()); ps.append(o.argmax(1).numpy())
    return sum(ls)/len(loader.dataset), accuracy_score(np.concatenate(ys), np.concatenate(ps))

t0 = time.time()
to_run = min(EPOCHS_NOW, TARGET - done_epochs)
print(f"[{NAME}] running {to_run} epoch(s) (have {done_epochs}, target {TARGET})")
for ep in range(to_run):
    model.train(); rl=0.0; rc=0; n=0
    for x, t in loader:
        opt.zero_grad(); o = model(x); l = crit(o, t); l.backward(); opt.step()
        rl += l.item()*x.size(0); rc += (o.argmax(1)==t).sum().item(); n += x.size(0)
    tl, ta = rl/n, rc/n
    vl, va_acc = evaluate(model, val_loader)
    hist["train_loss"].append(tl); hist["val_loss"].append(vl)
    hist["train_acc"].append(ta);  hist["val_acc"].append(va_acc)
    print(f"  ep {done_epochs+ep+1}/{TARGET}  tl={tl:.3f} vl={vl:.3f} ta={ta:.3f} va={va_acc:.3f}  ({time.time()-t0:.1f}s)")

torch.save({"model": model.state_dict(), "opt": opt.state_dict()}, ckpt_path)
hist_path.write_text(json.dumps(hist))
print(f"Checkpoint saved. Total epochs done: {len(hist['train_loss'])}/{TARGET}")

# If we finished, also produce test predictions
if len(hist["train_loss"]) >= TARGET:
    test_ds = DS(te, augment=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    model.eval(); ys, pr = [], []
    with torch.no_grad():
        for x, t in test_loader:
            pr.append(F.softmax(model(x), 1).numpy()); ys.append(t.numpy())
    yt = np.concatenate(ys); P = np.concatenate(pr); yp = P.argmax(1)
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, labels=list(range(NC)), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    out = dict(name=NAME, target_epochs=TARGET, history=hist,
               y_true=yt.tolist(), y_pred=yp.tolist(),
               per_p=p.tolist(), per_r=r.tolist(), per_f1=f1.tolist(),
               macro_p=float(mp), macro_r=float(mr), macro_f1=float(mf),
               acc=float(accuracy_score(yt, yp)),
               bal_acc=float(balanced_accuracy_score(yt, yp)),
               n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)),
               train_counts=train_counts, classes=CLASSES)
    if NAME == "Standard CE":
        np.save(RES / "probs_standard.npy", P)
    (RES / f"{slug}.json").write_text(json.dumps(out))
    print(f"FINAL: {slug}.json   acc={out['acc']:.3f} macroF1={out['macro_f1']:.3f}")
