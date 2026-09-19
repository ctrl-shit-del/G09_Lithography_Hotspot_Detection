"""Phase 0 - ICCAD-12 dataset inspection, verification, preprocessing and split generation.

Usage
-----
    python -m src.data --root <dataset_root> --inspect-only     # step 1: discover layout
    python -m src.data --root <dataset_root> --size 64          # step 2: verify + preprocess

Outputs
-------
    data/bN_{train,test}_{X,y}.npy   X: uint8 (N,1,S,S)   y: uint8 (1 = HS, 0 = NHS)
    data/splits/bN_seed{S}.json      stratified 80/20 train/val index split (versioned artifact)
    data/meta.json                   native image sizes, resize method, realized counts

Resize method
-------------
Native clips are square in ICCAD-12 but we do not assume it: each image is converted to
single-channel 'L', zero-padded (centred) to a square of side max(H, W) so the aspect ratio is
preserved, then downsampled to SxS with PIL's BOX filter (area averaging). Area averaging is
used rather than nearest-neighbour so thin polygons are not aliased away at 64x64; the result
is a grey-level "coverage" map in [0, 255]. NO ImageNet mean/std normalisation; batches are
cast to float and divided by 255 at batch time (see `to_float`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image  # Pillow ships as a matplotlib dependency; no new requirement.

# ----------------------------------------------------------------------------------------------
# Published ICCAD-12 benchmark statistics (hard constraint: fail loudly on mismatch)
# ----------------------------------------------------------------------------------------------
EXPECTED = {
    1: dict(train_hs=99,  train_nhs=340,  test_hs=226,  test_nhs=3869),
    2: dict(train_hs=174, train_nhs=5285, test_hs=498,  test_nhs=41298),
    3: dict(train_hs=909, train_nhs=4643, test_hs=1808, test_nhs=46333),
    4: dict(train_hs=95,  train_nhs=4452, test_hs=177,  test_nhs=31890),
    5: dict(train_hs=26,  train_nhs=2716, test_hs=41,   test_nhs=19327),
}
BENCHES = (1, 2, 3, 4, 5)
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DATA_DIR = Path("data")


# ----------------------------------------------------------------------------------------------
# Step 1: inspection (assumes nothing about the layout)
# ----------------------------------------------------------------------------------------------
def _classify_dir(path: Path):
    """Infer (bench, split, label) from the path components, or None if not inferable.

    bench : first 'iccad<N>' / 'bench<N>' / 'b<N>' component (case-insensitive)
    split : 'train' or 'test' appearing in any component
    label : component ending in 'nhs' -> NHS, else ending in 'hs' -> HS (leaf checked first)
    """
    parts = [p.lower() for p in path.parts]
    bench = split = label = None
    for p in parts:
        m = re.search(r"(?:iccad|bench|benchmark|b)[_-]?(\d+)$", p)
        if m and bench is None:
            bench = int(m.group(1))
        if "train" in p and split is None:
            split = "train"
        elif "test" in p and split is None:
            split = "test"
    for p in reversed(parts):
        if re.search(r"(^|[_-])nhs$", p):
            label = "NHS"
            break
        if re.search(r"(^|[_-])hs$", p):
            label = "HS"
            break
    if bench is None or split is None or label is None:
        return None
    return bench, split, label


def inspect_dataset(root, max_depth_print: int = 6, sample_images: int = 2):
    """Walk `root`, print the discovered tree with per-directory image counts, and return
    {(bench, split, label): [file paths]} for every leaf directory that could be classified.

    Nothing about the directory layout is assumed; the classification heuristics are printed
    alongside the raw tree so a wrong inference is visible immediately.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    print(f"[inspect] root = {root.resolve()}")

    dir_counts: dict = {}
    leaf_files: dict = defaultdict(list)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        d = Path(dirpath)
        c = Counter(Path(f).suffix.lower() for f in filenames)
        dir_counts[d] = c
        for f in filenames:
            if Path(f).suffix.lower() in IMG_EXT:
                leaf_files[d].append(d / f)

    print("\n[inspect] directory tree (image files per directory; non-image suffixes in braces)")
    for d in sorted(dir_counts):
        rel = d.relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth_print:
            continue
        c = dir_counts[d]
        n_img = sum(v for k, v in c.items() if k in IMG_EXT)
        other = {k: v for k, v in c.items() if k not in IMG_EXT}
        cls = _classify_dir(rel) if n_img else None
        tag = f"  -> bench={cls[0]} split={cls[1]} label={cls[2]}" if cls else ""
        print(f"  {'  ' * depth}{rel.name or '.'}/  [{n_img} images]"
              + (f" {other}" if other else "") + tag)

    discovered: dict = defaultdict(list)
    unclassified = []
    for d, files in leaf_files.items():
        cls = _classify_dir(d.relative_to(root))
        if cls is None:
            unclassified.append((d, len(files)))
        else:
            discovered[cls].extend(files)
    if unclassified:
        print("\n[inspect] WARNING: image directories that could not be classified:")
        for d, n in unclassified:
            print(f"   {d}  ({n} images)")

    print("\n[inspect] per-class file counts")
    print(f"  {'bench':>5} {'train HS':>9} {'train NHS':>10} {'test HS':>8} {'test NHS':>9}")
    for b in sorted({k[0] for k in discovered}):
        row = [len(discovered.get((b, s, l), [])) for s in ("train", "test") for l in ("HS", "NHS")]
        print(f"  {b:>5} {row[0]:>9} {row[1]:>10} {row[2]:>8} {row[3]:>9}")

    print("\n[inspect] sample image properties (mode, size, distinct pixel values)")
    sizes: Counter = Counter()
    for key in sorted(discovered):
        for p in discovered[key][:sample_images]:
            with Image.open(p) as im:
                arr = np.asarray(im)
                u = np.unique(arr)
                us = u.tolist() if u.size <= 8 else f"{u.size} values in [{u.min()},{u.max()}]"
                sizes[(key[0], im.mode, im.size)] += 1
                print(f"  b{key[0]} {key[1]:5s} {key[2]:3s} {p.name:20s} mode={im.mode:4s} "
                      f"size={im.size} dtype={arr.dtype} shape={arr.shape} values={us}")
    print("\n[inspect] (bench, mode, size) of sampled images:", dict(sizes))
    return dict(discovered)


def verify_counts(discovered: dict, accept: dict | None = None) -> dict:
    """Raise AssertionError listing every (bench, split, label) whose count differs from
    the published statistics, unless that exact (key, count) pair was explicitly accepted
    via --accept-count. Returns the dict of overrides actually used (recorded in meta.json)."""
    accept = accept or {}
    errors, used = [], {}
    for b in BENCHES:
        for s in ("train", "test"):
            for l in ("hs", "nhs"):
                key = f"b{b}_{s}_{l}"
                exp = EXPECTED[b][f"{s}_{l}"]
                got = len(discovered.get((b, s, l.upper()), []))
                if got == exp:
                    continue
                if accept.get(key) == got:
                    used[key] = dict(published=exp, realized=got)
                    print(f"[verify] OVERRIDE accepted: {key} published={exp} realized={got}")
                else:
                    errors.append(f"{key}: expected {exp}, found {got}")
    if errors:
        raise AssertionError("Dataset count mismatch against published ICCAD-12 statistics:\n  "
                             + "\n  ".join(errors)
                             + "\nIf the discrepancy is a known property of your copy of the dataset, "
                               "re-run with --accept-count KEY=COUNT (recorded in data/meta.json).")
    print(f"[verify] {20 - len(used)} of 20 (bench, split, class) counts match the published statistics"
          + (f"; {len(used)} explicit override(s) recorded." if used else "."))
    return used


# ----------------------------------------------------------------------------------------------
# Step 2: preprocessing
# ----------------------------------------------------------------------------------------------
def load_clip(path: Path, size: int) -> np.ndarray:
    """Read one clip -> uint8 (S, S). Aspect ratio preserved by centred zero-padding to a
    square before BOX (area-average) downsampling. Documented in the module docstring."""
    with Image.open(path) as im:
        im = im.convert("L")
        w, h = im.size
        if w != h:
            side = max(w, h)
            canvas = Image.new("L", (side, side), 0)
            canvas.paste(im, ((side - w) // 2, (side - h) // 2))
            im = canvas
        if im.size != (size, size):
            im = im.resize((size, size), Image.BOX)
        return np.asarray(im, dtype=np.uint8)


def _load_chunk(args):
    paths, size = args
    return np.stack([load_clip(p, size) for p in paths])


def load_many(paths: list, size: int, workers: int) -> np.ndarray:
    if not paths:
        return np.zeros((0, size, size), np.uint8)
    chunk = max(64, len(paths) // (workers * 8) + 1)
    chunks = [(paths[i:i + chunk], size) for i in range(0, len(paths), chunk)]
    if workers <= 1:
        out = [_load_chunk(c) for c in chunks]
    else:
        with ProcessPoolExecutor(workers) as ex:
            out = list(ex.map(_load_chunk, chunks))
    return np.concatenate(out)


def preprocess(discovered: dict, size: int, out_dir: Path, workers: int, overrides: dict | None = None) -> dict:
    """Write data/bN_{train,test}_{X,y}.npy. HS files come first, then NHS, each sorted by
    filename, so the order is deterministic and re-derivable from the raw tree."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    for b in BENCHES:
        for s in ("train", "test"):
            hs = sorted(discovered[(b, s, "HS")])
            nhs = sorted(discovered[(b, s, "NHS")])
            X = load_many(hs + nhs, size, workers)[:, None]          # (N,1,S,S)
            y = np.concatenate([np.ones(len(hs), np.uint8), np.zeros(len(nhs), np.uint8)])
            assert X.shape == (len(y), 1, size, size) and X.dtype == np.uint8
            np.save(out_dir / f"b{b}_{s}_X.npy", X)
            np.save(out_dir / f"b{b}_{s}_y.npy", y)
            with Image.open(hs[0]) as im:
                native = list(im.size)
            meta[f"b{b}_{s}"] = dict(n=len(y), hs=len(hs), nhs=len(nhs), native_size=native,
                                    mean_pixel=float(X.mean()), nonzero_frac=float((X > 0).mean()))
            print(f"[preprocess] b{b} {s:5s} X={X.shape} HS={len(hs):5d} NHS={len(nhs):5d} "
                  f"native={native} mean_px={X.mean():.2f}")
    meta["size"] = size
    meta["count_overrides"] = overrides or {}
    meta["resize_method"] = ("convert('L'); centred zero-pad to square (aspect preserved); "
                             "PIL BOX (area-average) downsample to SxS; uint8; /255 at batch time")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


# ----------------------------------------------------------------------------------------------
# Splits (stratified 80/20 of the official TRAIN split, one JSON per seed)
# ----------------------------------------------------------------------------------------------
def make_splits(out_dir: Path, seeds: list, val_frac: float = 0.2) -> None:
    from sklearn.model_selection import StratifiedShuffleSplit
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for b in BENCHES:
        y = np.load(out_dir / f"b{b}_train_y.npy")
        for seed in seeds:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
            tr, va = next(sss.split(np.zeros(len(y)), y))
            tr, va = np.sort(tr), np.sort(va)
            assert len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == len(y)
            rec = dict(bench=b, seed=seed, val_frac=val_frac, n_total=int(len(y)),
                       train_idx=tr.tolist(), val_idx=va.tolist(),
                       train_hs=int(y[tr].sum()), train_nhs=int((y[tr] == 0).sum()),
                       val_hs=int(y[va].sum()), val_nhs=int((y[va] == 0).sum()))
            (split_dir / f"b{b}_seed{seed}.json").write_text(json.dumps(rec))


def print_split_table(out_dir: Path, seeds: list) -> None:
    print("\n[splits] realized class counts (fit split = 80% of official train; val = 20%)")
    hdr = (f"{'bench':>5} {'seed':>4} | {'fit HS':>6} {'fit NHS':>7} | {'val HS':>6} {'val NHS':>7} | "
           f"{'test HS':>7} {'test NHS':>8} | {'NHS/HS train':>12} {'NHS/HS test':>11}")
    print(hdr)
    print("-" * len(hdr))
    for b in BENCHES:
        yte = np.load(out_dir / f"b{b}_test_y.npy")
        ytr = np.load(out_dir / f"b{b}_train_y.npy")
        for seed in seeds:
            r = json.loads((out_dir / "splits" / f"b{b}_seed{seed}.json").read_text())
            print(f"{b:>5} {seed:>4} | {r['train_hs']:>6} {r['train_nhs']:>7} | {r['val_hs']:>6} "
                  f"{r['val_nhs']:>7} | {int(yte.sum()):>7} {int((yte == 0).sum()):>8} | "
                  f"{(ytr == 0).sum() / ytr.sum():>12.1f} {(yte == 0).sum() / yte.sum():>11.1f}")


# ----------------------------------------------------------------------------------------------
# Runtime loader API (used by train.py / metrics.py). File system is touched once, here.
# ----------------------------------------------------------------------------------------------
def load_bench(b: int, data_dir: Path = DATA_DIR, mmap_test: bool = True):
    """Return dict of uint8 arrays: Xtr, ytr (memory-resident), Xte, yte. By default Xte is a
    read-only memmap so the test set (up to 48k x 128 x 128 = 790 MB) is never materialised;
    consumers must slice it in batches and cast each batch to float (see train.predict)."""
    return dict(Xtr=np.load(data_dir / f"b{b}_train_X.npy"), ytr=np.load(data_dir / f"b{b}_train_y.npy"),
                Xte=np.load(data_dir / f"b{b}_test_X.npy", mmap_mode="r" if mmap_test else None),
                yte=np.load(data_dir / f"b{b}_test_y.npy"))


def load_split(b: int, seed: int, data_dir: Path = DATA_DIR):
    r = json.loads((data_dir / "splits" / f"b{b}_seed{seed}.json").read_text())
    return np.asarray(r["train_idx"]), np.asarray(r["val_idx"])


def to_float(x_uint8):
    """Batch-time cast: uint8 [0,255] -> float32 [0,1]. Works for numpy and torch tensors."""
    if hasattr(x_uint8, "float"):
        return x_uint8.float().div_(255.0)
    return x_uint8.astype(np.float32) / 255.0


def file_sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dataset root (extracted zip)")
    ap.add_argument("--out", default=str(DATA_DIR))
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--inspect-only", action="store_true", help="print discovered structure and exit")
    ap.add_argument("--accept-count", nargs="*", default=[], metavar="KEY=COUNT",
                    help="explicitly accept a known deviation from the published counts, "
                         "e.g. b1_test_nhs=4679; anything else still fails loudly")
    a = ap.parse_args(argv)

    discovered = inspect_dataset(a.root)
    if a.inspect_only:
        return
    accept = {k: int(v) for k, v in (kv.split("=") for kv in a.accept_count)}
    overrides = verify_counts(discovered, accept)
    out = Path(a.out)
    preprocess(discovered, a.size, out, a.workers, overrides)
    make_splits(out, a.seeds)
    print_split_table(out, a.seeds)


if __name__ == "__main__":
    main()
