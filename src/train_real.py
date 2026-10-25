"""train_real.py NAME [EPOCHS] - run ONE experiment on the full real
HAM10000 cache. Save history + test predictions to JSON for later plotting.

Each call fits within the 45-second sandbox budget by training only one
configuration; running all five experiments thus needs five sandbox calls."""
import sys, os, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support)

CACHE  = Path("/sessions/loving-nice-maxwell/mnt/outputs/ham_cache")
RES    = Path("/sessions/loving-nice-maxwell/mnt/outputs/real_results")
RES.mkdir(parents=True, exist_ok=True)
SEED   = 42
torch.manual_seed(SEED); np.random.seed(SEED)
torch.set_num_threads(2)

# ---------- Load cache ----------
manifest = json.loads((CACHE / "manifest.json").read_text())
CLS = manifest["classes"]; NC = len(CLS); N = manifest["n"]
ranges = [(0,1500),(1500,3000),(3000,4500),(4500,6000),(6000,7500),(7500,9000),(9000,10015)]
X = np.empty((N, 3, 96, 96), dtype=np.uint8)
y = np.empty((N,), dtype=np.int64)
for s, e in ranges:
    X[s:e] = np.load(CACHE / f"chunk_{s:05d}_{e:05d}.npy")
    y[s:e] = np.load(CACHE / f"labels_{s:05d}_{e:05d}.npy")

# ---------- Stratified 70/15/15 split ----------
rng = np.random.default_rng(SEED)
tr, va, te = [], [], []
for c in range(NC):
    ix = np.where(y == c)[0]; rng.shuffle(ix); n = len(ix)
    n1, n2 = int(round(n*0.70)), int(round(n*0.15))
    tr += list(ix[:n1]); va += list(ix[n1:n1+n2]); te += list(ix[n1+n2:])
rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
tr, va, te = np.array(tr), np.array(va), np.array(te)

# ---------- Dataset (uint8 -> float, ImageNet normalisation, light aug on train) ----------
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)

class CacheDS(Dataset):
    def __init__(self, idx, augment=False):
        self.idx = idx; self.aug = augment
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        gi = self.idx[i]
        img = X[gi].astype(np.float32) / 255.0  # CHW float32 in [0,1]
        if self.aug:
            if np.random.rand() < 0.5:                    img = img[:, :, ::-1].copy()
            if np.random.rand() < 0.5:                    img = img[:, ::-1, :].copy()
        img = (img - MEAN) / STD
        return torch.from_numpy(img.astype(np.float32)), int(y[gi])

train_ds = CacheDS(tr, augment=True)
val_ds   = CacheDS(va, augment=False)
test_ds  = CacheDS(te, augment=False)

train_counts = np.bincount(y[tr], minlength=NC).tolist()

# ---------- Model ----------
class SkinCNN(nn.Module):
    def __init__(self, nc=NC):
        super().__init__()
        def blk(ic, oc):
            return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(blk(3,32), blk(32,64), blk(64,128), blk(128,256))
        self.gap = nn.AdaptiveAvgPool2d(1); self.dr = nn.Dropout(0.4)
        self.fc1 = nn.Linear(256, 128); self.fc2 = nn.Linear(128, nc)
    def forward(self, x):
        x = self.features(x); x = self.gap(x).flatten(1); x = self.dr(x)
        x = F.relu(self.fc1(x)); x = self.dr(x); return self.fc2(x)

# ---------- Losses + samplers ----------
def w_inv(c):
    c = np.array(c, dtype=np.float64); w = 1/c
    return torch.tensor(w * (len(c)/w.sum()), dtype=torch.float32)
def w_cb(c, beta=0.999):
    c = np.array(c, dtype=np.float64); e = 1 - np.power(beta, c); w = (1-beta)/e
    return torch.tensor(w * (len(c)/w.sum()), dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None): super().__init__(); self.g, self.a = gamma, alpha
    def forward(self, l, t):
        lp = F.log_softmax(l, 1); lpt = lp.gather(1, t.unsqueeze(1)).squeeze(1); pt = lpt.exp()
        fl = (1 - pt) ** self.g * (-lpt)
        if self.a is not None: fl = self.a.to(l.device)[t] * fl
        return fl.mean()

WI = w_inv(train_counts); WC = w_cb(train_counts)
sw = np.array([1/train_counts[int(c)] for c in y[tr]], dtype=np.float64)
sampler = WeightedRandomSampler(torch.from_numpy(sw), num_samples=len(sw), replacement=True)

CONFIGS = {
    "Standard CE":       (nn.CrossEntropyLoss(),               None),
    "Weighted CE":       (nn.CrossEntropyLoss(weight=WI),      None),
    "Focal (gamma=2)":   (FocalLoss(2.0, alpha=WI),            None),
    "Class-Balanced CE": (nn.CrossEntropyLoss(weight=WC),      None),
    "Std CE + Sampler":  (nn.CrossEntropyLoss(),               sampler),
}

NAME   = sys.argv[1]
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
BATCH  = 96
LR     = 1e-3
crit, sm = CONFIGS[NAME]
print(f"Training: {NAME}  epochs={EPOCHS}  batch={BATCH}  lr={LR}")
print(f"Train={len(tr)} Val={len(va)} Test={len(te)}  per-class train={train_counts}")

torch.manual_seed(SEED)
model = SkinCNN()
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
if sm is None:
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
else:
    loader = DataLoader(train_ds, batch_size=BATCH, sampler=sm,    num_workers=0)
val_loader  = DataLoader(val_ds,  batch_size=256, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); ys, ps, ls = [], [], []
    for x, t in loader:
        o = model(x); ls.append(nn.functional.cross_entropy(o, t).item() * x.size(0))
        ys.append(t.numpy()); ps.append(o.argmax(1).numpy())
    return sum(ls)/len(loader.dataset), accuracy_score(np.concatenate(ys), np.concatenate(ps))

@torch.no_grad()
def predict_proba(model, loader):
    model.eval(); ys, pr = [], []
    for x, t in loader:
        pr.append(F.softmax(model(x), 1).numpy()); ys.append(t.numpy())
    return np.concatenate(ys), np.concatenate(pr)

hist = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
t0 = time.time()
for ep in range(EPOCHS):
    model.train(); rl=0.; rc=0; n=0
    for x, t in loader:
        opt.zero_grad(); o = model(x); loss = crit(o, t); loss.backward(); opt.step()
        rl += loss.item() * x.size(0); rc += (o.argmax(1) == t).sum().item(); n += x.size(0)
    tl, ta = rl/n, rc/n
    vl, va_acc = evaluate(model, val_loader)
    hist["train_loss"].append(tl); hist["val_loss"].append(vl)
    hist["train_acc"].append(ta);  hist["val_acc"].append(va_acc)
    print(f"  ep {ep+1}/{EPOCHS}  tl={tl:.3f} vl={vl:.3f} ta={ta:.3f} va={va_acc:.3f}  ({time.time()-t0:.1f}s)")

yt, pr = predict_proba(model, test_loader)
yp = pr.argmax(1)
p, r, f1, _ = precision_recall_fscore_support(yt, yp, labels=list(range(NC)), zero_division=0)
mp, mr, mf, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
out = dict(name=NAME, epochs=EPOCHS, history=hist,
           y_true=yt.tolist(), y_pred=yp.tolist(),
           probs_top=pr[np.arange(len(yp)), yp].tolist(),
           per_p=p.tolist(), per_r=r.tolist(), per_f1=f1.tolist(),
           macro_p=float(mp), macro_r=float(mr), macro_f1=float(mf),
           acc=float(accuracy_score(yt, yp)),
           bal_acc=float(balanced_accuracy_score(yt, yp)),
           test_idx=te.tolist(), n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)),
           train_counts=train_counts)

# Save the full prob matrix only for the FIRST experiment - it's used for error analysis.
if NAME == "Standard CE":
    np.save(RES / "probs_standard.npy", pr)

# Save numpy file of test_idx for error analysis (just once)
if not (RES / "test_idx.npy").exists():
    np.save(RES / "test_idx.npy", te)

slug = NAME.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("+", "plus")
(RES / f"{slug}.json").write_text(json.dumps(out))
print(f"  total time: {time.time()-t0:.1f}s -> {slug}.json")
