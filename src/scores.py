"""Raw-score persistence shared by train.py (fresh runs) and evaluate.py (backfill).

Every run directory ends up with
    probs.npy    float32 sigmoid(test logits), computed in float64 then cast
    labels.npy   uint8 test labels in the same order
    logits.npy   float32 test logits
    eval.json    inference wall time, n, device, torch version
and, for temperature scaling / val-cost thresholds (metrics.py, sweep.py),
    val_logits.npy   float32 logits of the run's val split (best checkpoint)
    val_labels.npy   uint8 val labels, same order
Files are never overwritten: a second call on a run that already has them raises.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

SCORE_FILES = ("probs.npy", "labels.npy", "logits.npy", "eval.json")
VAL_FILES = ("val_logits.npy", "val_labels.npy")


def sigmoid64(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def save_scores(run_dir: Path, logits: np.ndarray, labels: np.ndarray, inference_s: float, dev, extra: dict | None = None):
    run_dir = Path(run_dir)
    clash = [f for f in SCORE_FILES if (run_dir / f).exists()]
    if clash:
        raise FileExistsError(f"{run_dir}: refusing to overwrite {clash}")
    logits = np.asarray(logits, dtype=np.float32).ravel()
    labels = np.asarray(labels).astype(np.uint8).ravel()
    assert len(logits) == len(labels) and len(labels) > 0
    np.save(run_dir / "probs.npy", sigmoid64(logits))
    np.save(run_dir / "labels.npy", labels)
    np.save(run_dir / "logits.npy", logits)
    rec = dict(n=int(len(labels)), n_hs=int(labels.sum()), inference_s=float(inference_s),
               inference_ms_per_clip=1000.0 * inference_s / len(labels), device=str(dev),
               torch=torch.__version__, written=time.strftime("%Y-%m-%dT%H:%M:%S"), **(extra or {}))
    (run_dir / "eval.json").write_text(json.dumps(rec, indent=2))
    return rec


def save_val_scores(run_dir: Path, logits: np.ndarray, labels: np.ndarray):
    run_dir = Path(run_dir)
    clash = [f for f in VAL_FILES if (run_dir / f).exists()]
    if clash:
        raise FileExistsError(f"{run_dir}: refusing to overwrite {clash}")
    logits = np.asarray(logits, dtype=np.float32).ravel()
    labels = np.asarray(labels).astype(np.uint8).ravel()
    assert len(logits) == len(labels) and len(labels) > 0
    np.save(run_dir / "val_logits.npy", logits)
    np.save(run_dir / "val_labels.npy", labels)


def load_scores(run_dir: Path):
    run_dir = Path(run_dir)
    return np.load(run_dir / "probs.npy"), np.load(run_dir / "labels.npy")
