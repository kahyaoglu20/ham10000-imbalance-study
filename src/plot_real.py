"""plot_real.py - regenerates ALL report/slide figures from the real
HAM10000 experiment results in /real_results/all_real.json."""
import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

OUT = Path("/sessions/loving-nice-maxwell/mnt/outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)
RES = Path("/sessions/loving-nice-maxwell/mnt/outputs/real_results")
CACHE = Path("/sessions/loving-nice-maxwell/mnt/outputs/ham_cache")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
PALETTE = sns.color_palette("colorblind", 7)

D = json.loads((RES / "all_real.json").read_text())
EXP = D["experiments"]
ORDER = D["ordered_names"]
CLASSES = D["classes"]   # alphabetical: akiec, bcc, bkl, df, mel, nv, vasc
NUM = len(CLASSES)
EPOCHS = len(EXP[ORDER[0]]["history"]["train_loss"])

# Real HAM10000 counts
manifest = json.loads((CACHE / "manifest.json").read_text())
HAM_COUNTS_DICT = manifest["class_counts"]
HAM_COUNTS = [HAM_COUNTS_DICT[c] for c in CLASSES]

# ---------------------------------------------------------------------------
# 1. Class distribution (real HAM10000 counts)
# ---------------------------------------------------------------------------
order_idx = np.argsort(HAM_COUNTS)[::-1]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
labels   = [CLASSES[i] for i in order_idx]
counts   = [HAM_COUNTS[i] for i in order_idx]
colors   = [PALETTE[i] for i in order_idx]
bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.4)
total = sum(HAM_COUNTS)
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 70,
            f"{c}\n({100*c/total:.1f}%)", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("Number of dermoscopic images")
ax.set_xlabel("Diagnostic class (HAM10000)")
ax.set_title("HAM10000 class distribution (severe imbalance, 7 classes)")
ax.set_ylim(0, max(HAM_COUNTS) * 1.20)
plt.tight_layout()
plt.savefig(OUT / "class_distribution.png", dpi=180); plt.close()

# ---------------------------------------------------------------------------
# 2. Loss curves
# ---------------------------------------------------------------------------
def plot_curves(key_train, key_val, ylabel, title, fname, ylim=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    xs = np.arange(1, EPOCHS + 1)
    for i, name in enumerate(ORDER):
        h = EXP[name]["history"]
        axes[0].plot(xs, h[key_train], label=name, color=PALETTE[i], linewidth=1.6, marker="o", markersize=3)
        axes[1].plot(xs, h[key_val],   label=name, color=PALETTE[i], linewidth=1.6, marker="o", markersize=3)
    axes[0].set_title(f"Train {ylabel}");  axes[0].set_xlabel("Epoch"); axes[0].set_ylabel(ylabel)
    axes[1].set_title(f"Validation {ylabel}"); axes[1].set_xlabel("Epoch")
    axes[1].legend(loc="best", fontsize=8, frameon=True)
    if ylim: [a.set_ylim(*ylim) for a in axes]
    fig.suptitle(title); plt.tight_layout()
    plt.savefig(OUT / fname, dpi=180); plt.close()

plot_curves("train_loss", "val_loss", "Loss",
            f"Per-experiment loss curves (real HAM10000, {EPOCHS} epochs)", "loss_curves.png")
plot_curves("train_acc",  "val_acc",  "Accuracy",
            f"Per-experiment accuracy curves (real HAM10000, {EPOCHS} epochs)",
            "acc_curves.png", ylim=(0, 1.0))

# ---------------------------------------------------------------------------
# 3. Annotated under/overfitting (Standard CE)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 3.6))
h = EXP["Standard CE"]["history"]
xs = np.arange(1, EPOCHS + 1)
ax.plot(xs, h["train_loss"], label="Train loss", color=PALETTE[0], marker="o", markersize=4, linewidth=1.8)
ax.plot(xs, h["val_loss"],   label="Val loss",   color=PALETTE[3], marker="o", markersize=4, linewidth=1.8)
ymax = max(max(h["train_loss"]), max(h["val_loss"]))
ax.set_ylim(0, ymax * 1.25)
ax.axvspan(0.5, 1.5, color="orange", alpha=0.10)
ax.text(1.0, ymax * 0.65, "Underfitting\n(both losses high)", ha="center",
        fontsize=8, color="darkorange")
gap_epoch = int(np.argmin(h["val_loss"])) + 1
ax.axvline(gap_epoch, color="grey", linestyle="--", alpha=0.7)
ax.text(gap_epoch + 0.15, ymax * 0.97, f"min val loss\n@ epoch {gap_epoch}",
        fontsize=8, color="grey")
# Mark overfitting region (last few epochs where train continues down but val plateaus)
ax.axvspan(EPOCHS - 1.5, EPOCHS + 0.5, color="red", alpha=0.07)
ax.text(EPOCHS - 0.5, ymax * 0.50, "Overfitting risk\n(val plateau)",
        ha="center", fontsize=8, color="darkred")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Standard CE - locating under/overfitting on the loss curve")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUT / "overfitting_annotated.png", dpi=180); plt.close()

# ---------------------------------------------------------------------------
# 4. Macro-F1 + minority-class recall
# ---------------------------------------------------------------------------
macro_f1 = [EXP[n]["macro_f1"] for n in ORDER]
rare_idx = CLASSES.index("df")
rare_rec = [EXP[n]["per_r"][rare_idx] for n in ORDER]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].barh(ORDER, macro_f1, color=PALETTE[:len(ORDER)], edgecolor="black", linewidth=0.4)
for i, v in enumerate(macro_f1):
    axes[0].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
axes[0].set_xlabel("Macro F1 on test set"); axes[0].set_title("Macro F1 by experiment")
axes[0].set_xlim(0, max(macro_f1 + [0.1]) * 1.30)
axes[0].invert_yaxis()

axes[1].barh(ORDER, rare_rec, color=PALETTE[:len(ORDER)], edgecolor="black", linewidth=0.4)
for i, v in enumerate(rare_rec):
    axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
axes[1].set_xlabel(f"Recall on rarest class ('df', 1.1% of data)")
axes[1].set_title("Minority-class recall by experiment")
axes[1].set_xlim(0, 1.0)
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig(OUT / "metrics_comparison.png", dpi=180); plt.close()

# ---------------------------------------------------------------------------
# 5. Per-class recall heatmap
# ---------------------------------------------------------------------------
RM = np.array([[EXP[n]["per_r"][c] for c in range(NUM)] for n in ORDER])
fig, ax = plt.subplots(figsize=(7.4, 3.2))
sns.heatmap(RM, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
            xticklabels=CLASSES, yticklabels=ORDER, cbar_kws={"label": "Per-class recall"})
ax.set_title("Per-class recall - all five experiments (real HAM10000)")
ax.set_xlabel("Class")
plt.tight_layout()
plt.savefig(OUT / "per_class_recall_heatmap.png", dpi=180); plt.close()

# ---------------------------------------------------------------------------
# 6. Confusion matrices: baseline vs best macro-F1
# ---------------------------------------------------------------------------
def plot_cm(yt, yp, title, fname):
    cm = confusion_matrix(yt, yp, labels=list(range(NUM)))
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, cbar=False, ax=ax,
                annot_kws={"fontsize": 8})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout()
    plt.savefig(OUT / fname, dpi=180); plt.close()

best_name = max(ORDER, key=lambda n: EXP[n]["macro_f1"])
plot_cm(np.array(EXP["Standard CE"]["y_true"]),
        np.array(EXP["Standard CE"]["y_pred"]),
        "Confusion matrix - Standard CE (baseline)", "cm_standard.png")
plot_cm(np.array(EXP[best_name]["y_true"]),
        np.array(EXP[best_name]["y_pred"]),
        f"Confusion matrix - {best_name} (best macro-F1)", "cm_best.png")
(OUT / "best_experiment.txt").write_text(best_name + "\n")
print("Best macro-F1 experiment:", best_name)

# ---------------------------------------------------------------------------
# 7. Error analysis: 3 most-confident WRONG predictions of Standard CE
# ---------------------------------------------------------------------------
test_idx = np.load(RES / "test_idx.npy")
probs = np.load(RES / "probs_standard.npy")
yp = probs.argmax(1)
yt = np.array(EXP["Standard CE"]["y_true"])

# Reload the original HAM 96x96 chunks to display the actual test images.
ranges = [(0,1500),(1500,3000),(3000,4500),(4500,6000),(6000,7500),(7500,9000),(9000,10015)]
X96 = np.empty((10015, 3, 96, 96), dtype=np.uint8)
for s, e in ranges:
    X96[s:e] = np.load(CACHE / f"chunk_{s:05d}_{e:05d}.npy")

wrong = np.where(yp != yt)[0]
confs = probs[wrong, yp[wrong]]
top3 = wrong[np.argsort(-confs)[:3]]
fig, axes = plt.subplots(1, 3, figsize=(9, 3.8))
fig.subplots_adjust(top=0.78, bottom=0.03, wspace=0.10)
for ax, li in zip(axes, top3):
    gi = test_idx[li]
    img = X96[gi].transpose(1, 2, 0)  # CHW -> HWC
    ax.imshow(img); ax.axis("off")
    p = probs[li, yp[li]]
    ax.set_title(f"True: {CLASSES[yt[li]]}    Pred: {CLASSES[yp[li]]}\nconfidence: {p*100:.1f}%",
                 fontsize=10, pad=6)
fig.suptitle("Standard CE: most-confident WRONG predictions on real HAM10000",
             fontsize=11, y=0.97)
plt.savefig(OUT / "error_analysis_baseline.png", dpi=180, bbox_inches="tight"); plt.close()

# Same analysis for the best model (compute test-set probs from saved checkpoint).
import torch, torch.nn as nn, torch.nn.functional as F
class SkinCNN(nn.Module):
    def __init__(s, nc=NUM):
        super().__init__()
        def b(ic,oc): return nn.Sequential(nn.Conv2d(ic,oc,3,padding=1,bias=False),
                                            nn.BatchNorm2d(oc), nn.ReLU(True), nn.MaxPool2d(2))
        s.features = nn.Sequential(b(3,16), b(16,32), b(32,64), b(64,128))
        s.gap = nn.AdaptiveAvgPool2d(1); s.dr = nn.Dropout(0.4)
        s.fc1 = nn.Linear(128, 64); s.fc2 = nn.Linear(64, nc)
    def forward(s, x):
        x = s.features(x); x = s.gap(x).flatten(1); x = s.dr(x)
        x = F.relu(s.fc1(x)); x = s.dr(x); return s.fc2(x)

slug_best = best_name.replace(" ","_").replace("(","").replace(")","").replace("=","").replace("+","plus")
ckpt = torch.load(RES / "ckpt" / f"{slug_best}.pt", map_location="cpu", weights_only=False)
m = SkinCNN(); m.load_state_dict(ckpt["model"]); m.eval()

X64 = np.empty((10015, 3, 64, 64), dtype=np.uint8)
for s, e in ranges:
    X64[s:e] = np.load(CACHE / f"chunk64_{s:05d}_{e:05d}.npy")

MEAN = np.array([0.485,0.456,0.406], dtype=np.float32).reshape(3,1,1)
STD  = np.array([0.229,0.224,0.225], dtype=np.float32).reshape(3,1,1)
all_probs = []
with torch.no_grad():
    for i in range(0, len(test_idx), 256):
        bi = test_idx[i:i+256]
        bx = X64[bi].astype(np.float32) / 255.0
        bx = (bx - MEAN) / STD
        all_probs.append(F.softmax(m(torch.from_numpy(bx.astype(np.float32))), 1).numpy())
P_best = np.concatenate(all_probs)
yp_b = P_best.argmax(1); yt_b = np.array(EXP[best_name]["y_true"])
wrong_b = np.where(yp_b != yt_b)[0]
confs_b = P_best[wrong_b, yp_b[wrong_b]]
top3_b = wrong_b[np.argsort(-confs_b)[:3]]
fig, axes = plt.subplots(1, 3, figsize=(9, 3.8))
fig.subplots_adjust(top=0.78, bottom=0.03, wspace=0.10)
for ax, li in zip(axes, top3_b):
    gi = test_idx[li]
    img = X96[gi].transpose(1, 2, 0)
    ax.imshow(img); ax.axis("off")
    p = P_best[li, yp_b[li]]
    ax.set_title(f"True: {CLASSES[yt_b[li]]}    Pred: {CLASSES[yp_b[li]]}\nconfidence: {p*100:.1f}%",
                 fontsize=10, pad=6)
fig.suptitle(f"{best_name}: most-confident WRONG predictions on real HAM10000",
             fontsize=11, y=0.97)
plt.savefig(OUT / "error_analysis.png", dpi=180, bbox_inches="tight"); plt.close()

print("All real-data figures saved in:", OUT)
for p in sorted(OUT.glob("*.png")):
    print("  ", p.name, f"{p.stat().st_size//1024} KB")
