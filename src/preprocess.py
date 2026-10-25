"""preprocess.py — load HAM10000 from the two ZIPs, resize each image to
96x96, and save tensor chunks. Each invocation processes a CHUNK of indices
(set by argv) so the 45-second sandbox call budget is respected.

Uses multiprocessing for speed. Atomic rename prevents partial writes."""
import sys, zipfile, io, time, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from multiprocessing import Pool

DATA = Path("/sessions/loving-nice-maxwell/mnt/MYZ/dataset")
CACHE = Path("/sessions/loving-nice-maxwell/mnt/outputs/ham_cache")
CACHE.mkdir(parents=True, exist_ok=True)

ZIPS = [str(DATA / "HAM10000_images_part_1.zip"),
        str(DATA / "HAM10000_images_part_2.zip")]
META_CSV = DATA / "HAM10000_metadata"
IMG_SIZE = 96

# ---------- Phase 0: build manifest if needed ------------------------------
MANIFEST = CACHE / "manifest.json"
if not MANIFEST.exists():
    df = pd.read_csv(META_CSV)
    location = {}
    for zi, zp in enumerate(ZIPS):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.lower().endswith(".jpg"):
                    iid = Path(name).stem
                    location[iid] = (zi, name)
    df["zip_idx"]  = df["image_id"].map(lambda i: location[i][0])
    df["zip_path"] = df["image_id"].map(lambda i: location[i][1])
    df = df[df["zip_path"].notna()].reset_index(drop=True)
    classes = sorted(df["dx"].unique().tolist())
    cls2idx = {c: i for i, c in enumerate(classes)}
    df["label"] = df["dx"].map(cls2idx)
    manifest = {
        "n": int(len(df)), "classes": classes,
        "class_counts": {c: int((df["dx"] == c).sum()) for c in classes},
        "image_ids": df["image_id"].tolist(),
        "labels":    df["label"].tolist(),
        "zip_idx":   df["zip_idx"].tolist(),
        "zip_path":  df["zip_path"].tolist(),
    }
    MANIFEST.write_text(json.dumps(manifest))
    print("Manifest built. Total:", manifest["n"], "Classes:", classes)

manifest = json.loads(MANIFEST.read_text())
N = manifest["n"]

# ---------- Phase 1: process one chunk via multiprocessing -----------------
if len(sys.argv) >= 3:
    start = int(sys.argv[1]); end = int(sys.argv[2])
else:
    print("Usage: preprocess.py START END"); sys.exit(0)
end = min(end, N)
out_path = CACHE / f"chunk_{start:05d}_{end:05d}.npy"
if out_path.exists():
    try:
        a = np.load(out_path); assert a.shape == (end - start, 3, IMG_SIZE, IMG_SIZE)
        print("Already done:", out_path); sys.exit(0)
    except Exception:
        pass  # fall through and re-run

# Each worker opens its own zipfile handle (they're not pickleable).
def worker(args):
    gi, zi, name = args
    if not hasattr(worker, "zfs"):
        worker.zfs = [zipfile.ZipFile(z) for z in ZIPS]
    with worker.zfs[zi].open(name) as f:
        img = Image.open(io.BytesIO(f.read())).convert("RGB").resize(
            (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return gi, np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)

tasks = [(gi, manifest["zip_idx"][gi], manifest["zip_path"][gi])
         for gi in range(start, end)]

t0 = time.time()
buf = np.empty((end - start, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
with Pool(2) as pool:
    for gi, arr in pool.imap_unordered(worker, tasks, chunksize=64):
        buf[gi - start] = arr

labs = np.array(manifest["labels"][start:end], dtype=np.int64)
# np.save adds .npy if missing — write to a path that already ends in .npy
tmp_no_ext = str(out_path) + ".tmp"   # ham_cache/chunk_xxx.npy.tmp
np.save(tmp_no_ext, buf)              # actually writes ham_cache/chunk_xxx.npy.tmp.npy
os.replace(tmp_no_ext + ".npy", out_path)
np.save(CACHE / f"labels_{start:05d}_{end:05d}.npy", labs)
print(f"Wrote {out_path.name} {buf.shape} in {time.time() - t0:.1f}s")
