"""Backfill raw scores for finished runs from their checkpoints - inference only, no retraining.

    python -m src.evaluate --runs runs                # every run dir with best.pt + config.json
    python -m src.evaluate --runs runs --tag ls0.1    # one experiment
    python -m src.evaluate --dirs runs/ls0.1/b1/seed0 # explicit run dirs

For each run: rebuild the model from config.json, load best.pt, run batched inference over the
uint8 test memmap (never materialised as float32), and write probs.npy / labels.npy /
logits.npy / eval.json next to test.json. Existing files are NEVER overwritten.

Verification: the ROC-AUC of the fresh logits must equal test.json's recorded at_0p5.auc to
within 1e-6, and if the run saved test_logits.npy the fresh logits must match them
element-wise. On any mismatch nothing is written and the run is reported.

A run that has a checkpoint but no test.json (e.g. an evaluation that was killed) gets a
test.json created with the same structure train.py writes, tagged "source": "evaluate backfill".

    python -m src.evaluate --val --runs runs_gpu      # backfill val_logits.npy / val_labels.npy

`--val` re-scores the run's own val split (data/splits/b{B}_seed{S}.json) with best.pt and
writes val_logits.npy / val_labels.npy (needed by metrics.py temperature scaling and by the
val-cost thresholds in sweep.py). Verification: the Youden threshold recomputed from the fresh
val logits must match test.json's val_thr_youden to 1e-3 (GPU vs CPU arithmetic) and the val
AUC must match to 1e-6; on a mismatch nothing is written.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .data import load_bench, load_split
from .device import get_device
from .metrics import binary_metrics, recall_at_fpr, youden_threshold
from .model import build_model
from .scores import SCORE_FILES, VAL_FILES, save_scores, save_val_scores
from .train import _class_subset, predict


def load_run(run_dir: Path, dev):
    cfg = json.loads((run_dir / "config.json").read_text())
    model = build_model(cfg["model"], size=cfg.get("size"), dropout=cfg.get("dropout", 0.3)).to(dev)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=dev))
    model.eval()
    return cfg, model


def test_arrays(cfg: dict):
    d = load_bench(cfg["bench"], Path(cfg["data"]), mmap_test=True)
    Xte, yte = d["Xte"], d["yte"]
    if cfg.get("smoke"):
        k = _class_subset(yte, 16)
        Xte, yte = np.array(Xte[k]), yte[k]
    return d, Xte, yte


def backfill(run_dir: Path, dev, tol: float = 1e-6, allow_missing: bool = False) -> dict:
    run_dir = Path(run_dir)
    rep = dict(run=str(run_dir))
    have = [f for f in SCORE_FILES if (run_dir / f).exists()]
    if have:
        rep.update(status="skipped", reason=f"already has {have}")
        return rep
    if not (run_dir / "test.json").exists() and not allow_missing:
        # no test.json normally means training is still in progress (best.pt is rewritten every
        # improving epoch) or was killed; only backfill such a run when explicitly asked
        rep.update(status="skipped", reason="no test.json (training in progress or killed); pass --allow-missing-test-json")
        return rep
    cfg, model = load_run(run_dir, dev)
    d, Xte, yte = test_arrays(cfg)

    t0 = time.time()
    logits = predict(model, Xte, dev, batch=512)
    infer_s = time.time() - t0
    auc = float(roc_auc_score(yte, logits))
    rep.update(bench=cfg["bench"], seed=cfg["seed"], n=int(len(yte)), auc_fresh=auc, inference_s=infer_s)

    tj = run_dir / "test.json"
    if tj.exists():
        rec = json.loads(tj.read_text())["at_0p5"]["auc"]
        rep.update(auc_recorded=rec, auc_absdiff=abs(auc - rec))
        if abs(auc - rec) > tol:
            rep.update(status="MISMATCH", reason=f"auc differs by {abs(auc - rec):.3e} > {tol}")
            return rep
    old = run_dir / "test_logits.npy"
    if old.exists():
        prev = np.load(old)
        md = float(np.abs(prev - logits).max()) if prev.shape == logits.shape else float("inf")
        rep.update(logits_maxabsdiff=md)
        if md > 1e-4:
            rep.update(status="MISMATCH", reason=f"logits differ from test_logits.npy by {md:.3e}")
            return rep

    if not tj.exists():
        # reproduce train.py's test.json from the same checkpoint: thresholds come from the val split
        tr_idx, va_idx = load_split(cfg["bench"], cfg["seed"], Path(cfg["data"]))
        Xval, yval = d["Xtr"][va_idx], d["ytr"][va_idx]
        pv = predict(model, Xval, dev)
        thr_y = youden_threshold(yval, pv)
        _, thr_fpr = recall_at_fpr(yval, pv, cfg.get("target_fpr", 0.01))
        hist = json.loads((run_dir / "history.json").read_text())
        best = max(hist, key=lambda r: (r["val_auc"], -r["val_loss"]))
        test = dict(best_epoch=best["epoch"], epochs_run=len(hist), val_auc=float(best["val_auc"]), inference_s=infer_s,
                    threshold_space="logit", val_thr_youden=thr_y, val_thr_fpr=thr_fpr,
                    at_0p5=binary_metrics(yte, logits, 0.0),
                    at_val_youden=binary_metrics(yte, logits, thr_y),
                    at_val_fpr=binary_metrics(yte, logits, thr_fpr),
                    target_fpr=cfg.get("target_fpr", 0.01), train_time_s=hist[-1]["elapsed"],
                    source="evaluate backfill (training completed; original test evaluation was killed)")
        tj.write_text(json.dumps(test, indent=2))
        rep["test_json"] = "created"

    save_scores(run_dir, logits, yte, infer_s, dev, extra=dict(source="evaluate backfill", auc_check=rep.get("auc_absdiff")))
    rep.update(status="ok")
    return rep


def backfill_val(run_dir: Path, dev, thr_tol: float = 1e-3, auc_tol: float = 1e-6) -> dict:
    run_dir = Path(run_dir)
    rep = dict(run=str(run_dir))
    have = [f for f in VAL_FILES if (run_dir / f).exists()]
    if have:
        rep.update(status="skipped", reason=f"already has {have}")
        return rep
    tj = run_dir / "test.json"
    if not tj.exists():
        rep.update(status="skipped", reason="no test.json")
        return rep
    cfg, model = load_run(run_dir, dev)
    d = load_bench(cfg["bench"], Path(cfg["data"]), mmap_test=True)
    _, va_idx = load_split(cfg["bench"], cfg["seed"], Path(cfg["data"]))
    Xval, yval = d["Xtr"][va_idx], d["ytr"][va_idx]
    pv = predict(model, Xval, dev)
    t = json.loads(tj.read_text())
    thr = youden_threshold(yval, pv)
    auc = float(roc_auc_score(yval, pv)) if 0 < yval.sum() < len(yval) else float("nan")
    rep.update(bench=cfg["bench"], seed=cfg["seed"], n_val=int(len(yval)), n_val_hs=int(yval.sum()),
               thr_fresh=thr, thr_recorded=t["val_thr_youden"], thr_absdiff=abs(thr - t["val_thr_youden"]),
               auc_fresh=auc, auc_recorded=t["val_auc"], auc_absdiff=abs(auc - t["val_auc"]))
    if rep["thr_absdiff"] > thr_tol or rep["auc_absdiff"] > auc_tol:
        rep.update(status="MISMATCH", reason=f"thr diff {rep['thr_absdiff']:.2e}, auc diff {rep['auc_absdiff']:.2e}")
        return rep
    save_val_scores(run_dir, pv, yval)
    rep.update(status="ok")
    return rep


def refresh_scores(run_dir: Path, dev, auc_tol: float = 1e-5) -> dict:
    """Recompute probs/labels/logits (and val_logits/val_labels) from best.pt and OVERWRITE the files
    on disk, but only when the fresh scores reproduce test.json's at_0p5 counts exactly and its AUC
    to `auc_tol`. Used to repair fold directories whose score files were left over from an earlier
    training pass (oof.py < 2026-09-19 skipped save_scores when probs.npy already existed).
    The original eval.json timing fields are kept (they were measured on the training device);
    a `scores_refreshed` record documents the repair."""
    run_dir = Path(run_dir)
    rep = dict(run=str(run_dir))
    tj = run_dir / "test.json"
    if not tj.exists():
        rep.update(status="skipped", reason="no test.json")
        return rep
    t = json.loads(tj.read_text())
    cfg, model = load_run(run_dir, dev)
    d, Xte, yte = test_arrays(cfg)
    # stale check first: do the files on disk reproduce test.json?
    if (run_dir / "logits.npy").exists():
        old = np.load(run_dir / "logits.npy")
        m = binary_metrics(yte, old, 0.0)
        rep["disk_matches"] = (m["tp"], m["fp"]) == (t["at_0p5"]["tp"], t["at_0p5"]["fp"]) and abs(m["auc"] - t["at_0p5"]["auc"]) <= auc_tol
        if rep["disk_matches"]:
            rep.update(status="ok (already consistent)")
            return rep
    t0 = time.time()
    logits = predict(model, Xte, dev, batch=512)
    infer_s = time.time() - t0
    m = binary_metrics(yte, logits, 0.0)
    rep.update(n=int(len(yte)), auc_fresh=m["auc"], auc_recorded=t["at_0p5"]["auc"], auc_absdiff=abs(m["auc"] - t["at_0p5"]["auc"]),
               counts_fresh=(m["tp"], m["fp"]), counts_recorded=(t["at_0p5"]["tp"], t["at_0p5"]["fp"]), infer_s=infer_s)
    if rep["counts_fresh"] != rep["counts_recorded"] or rep["auc_absdiff"] > auc_tol:
        rep.update(status="MISMATCH", reason="fresh scores do not reproduce test.json either; nothing written")
        return rep
    # val split: 80/20 split for train.py runs, the held-out fold for oof.py folds
    if "fold" in cfg:
        fold_of = np.load(run_dir.parent / "fold_of.npy")
        va_idx = np.flatnonzero(fold_of == cfg["fold"])
    else:
        _, va_idx = load_split(cfg["bench"], cfg["seed"], Path(cfg["data"]))
    Xval, yval = d["Xtr"][va_idx], d["ytr"][va_idx]
    pv = predict(model, Xval, dev)
    thr = youden_threshold(yval, pv)
    rep.update(val_thr_fresh=thr, val_thr_recorded=t["val_thr_youden"], val_thr_absdiff=abs(thr - t["val_thr_youden"]))
    if rep["val_thr_absdiff"] > 1e-3:
        rep.update(status="MISMATCH", reason="val threshold differs; nothing written")
        return rep
    ev = json.loads((run_dir / "eval.json").read_text()) if (run_dir / "eval.json").exists() else {}
    for f in SCORE_FILES + VAL_FILES:
        (run_dir / f).unlink(missing_ok=True)
    save_scores(run_dir, logits, yte, ev.get("inference_s", infer_s), ev.get("device", str(dev)),
                extra=dict(**{k: v for k, v in ev.items() if k not in ("n", "n_hs", "inference_s", "inference_ms_per_clip", "device", "torch", "written")},
                           scores_refreshed=dict(device=str(dev), inference_s_on_this_device=infer_s, auc_absdiff=rep["auc_absdiff"],
                                                 val_thr_absdiff=rep["val_thr_absdiff"], written=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                 reason="fold score files were from an earlier training pass; recomputed from best.pt")))
    save_val_scores(run_dir, pv, yval)
    rep.update(status="refreshed")
    return rep


def find_runs(root: Path, tag: str | None):
    pats = [f"{tag}/b*/seed*", f"{tag}/b*/seed*/fold*"] if tag else ["*/b*/seed*", "*/b*/seed*/fold*"]
    return sorted(p for pat in pats for p in root.glob(pat) if (p / "best.pt").exists() and (p / "config.json").exists())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dirs", nargs="*", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-missing-test-json", action="store_true",
                    help="also backfill runs whose training finished but whose test evaluation was killed")
    ap.add_argument("--val", action="store_true", help="backfill val_logits.npy / val_labels.npy instead of test scores")
    ap.add_argument("--refresh-stale", action="store_true",
                    help="recompute and overwrite score files that do not reproduce test.json (verified against it first)")
    a = ap.parse_args(argv)
    dev = get_device(a.device)
    dirs = [Path(d) for d in a.dirs] if a.dirs else find_runs(Path(a.runs), a.tag)
    if a.refresh_stale:
        for d in dirs:
            r = refresh_scores(d, dev)
            extra = {k: r[k] for k in ("counts_fresh", "counts_recorded", "auc_absdiff", "val_thr_absdiff", "infer_s") if k in r}
            print(f"{r['run']:40s} {r['status']:26s} {extra}" + (f"  ({r['reason']})" if "reason" in r else ""), flush=True)
        return
    if a.val:
        print(f"{'run':34s} {'n_val':>6s} {'HS':>4s} {'thr fresh':>10s} {'thr rec':>10s} {'|diff|':>9s} {'auc |diff|':>10s}  status")
        for d in dirs:
            r = backfill_val(d, dev)
            f = lambda k, w: (f"{r[k]:{w}}" if k in r else " " * int(w.rstrip("dfe").split(".")[0]))
            print(f"{r['run']:34s} {f('n_val','6d')} {f('n_val_hs','4d')} {f('thr_fresh','10.5f')} {f('thr_recorded','10.5f')} "
                  f"{f('thr_absdiff','9.1e')} {f('auc_absdiff','10.1e')}  {r['status']}" + (f"  ({r['reason']})" if "reason" in r else ""))
        return
    print(f"{'run':34s} {'n':>6s} {'auc fresh':>10s} {'auc rec':>10s} {'|diff|':>9s} {'logit diff':>10s} {'infer s':>8s}  status")
    for d in dirs:
        r = backfill(d, dev, allow_missing=a.allow_missing_test_json)
        f = lambda k, w: (f"{r[k]:{w}}" if k in r else " " * int(w.rstrip("dfe").split(".")[0]))
        print(f"{r['run']:34s} {f('n','6d')} {f('auc_fresh','10.6f')} {f('auc_recorded','10.6f')} {f('auc_absdiff','9.1e')} "
              f"{f('logits_maxabsdiff','10.1e')} {f('inference_s','8.1f')}  {r['status']}"
              + (f"  ({r['reason']})" if "reason" in r else "") + ("  test.json created" if r.get("test_json") else ""))


if __name__ == "__main__":
    main()
