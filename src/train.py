"""Phase 2 - train one model per (benchmark, seed) and evaluate on the official test split.

    python -m src.train --bench 1 --seed 0                    # single run
    python -m src.train --bench 1 2 3 4 5 --seed 0 1 2 3 4    # full grid (25 runs)
    python -m src.train --bench 1 --seed 0 --smoke            # 2 epochs, tiny subset, CPU

Outputs (one directory per run)
    runs/<tag>/b{B}/seed{S}/config.json     resolved args + device + reproducibility info
                          history.json    per-epoch train loss + val metrics
                          best.pt         state_dict at the best val epoch
                          test.json       test-set metrics at logit 0 (p=0.5), val-Youden thr, val-FPR-budget thr
                          probs.npy       float32 sigmoid test probabilities  (raw scores; metrics/sweep read these)
                          labels.npy      uint8 test labels, same order as probs.npy
                          logits.npy      float32 test logits (thresholds in test.json are in this space)
                          eval.json       inference wall time, n, device
                          val_logits.npy  float32 val-split logits of the selected checkpoint (+ val_labels.npy);
                                          metrics.py fits temperature scaling / val-cost thresholds on these

Protocol
    * fit set = 80 % of the official train split, val = 20 % (data/splits, fixed per seed)
    * the official TEST split is touched exactly once, after training, with the best-val checkpoint
    * checkpoint selection: highest val ROC-AUC, ties broken by lower val loss
    * class imbalance: BCE-with-logits with pos_weight = n_nhs / n_fit_hs (default), or
      --imbalance oversample (WeightedRandomSampler to a balanced stream), or none
    * augmentation: the 8 dihedral symmetries (flips + 90-degree rotations). Layout hotspots
      are invariant to these; nothing else is applied
    * reference-paper arm (src/ref_model.py): --model ref --data data150 --optimizer nadam
      --sched const --select last --epochs 10 --no-augment --imbalance none --label-smoothing 0
      --weight-decay 0 --clip 0 --patience 0 --batch-size 32. `ref` takes 3 channels; the single
      grey channel is duplicated at batch time (`expand_channels`), exactly like a Keras RGB loader
    * overconfidence control: --label-smoothing EPS (default 0.1) trains against targets
      y*(1-EPS) + EPS/2 instead of {0,1}. The loss is then minimised at a finite logit
      (logit(1-EPS/2) ~ 2.9 for EPS=0.1) rather than at +/-inf, so (a) val loss plateaus once
      the val set is separated and early stopping can fire, and (b) test logits stay in a range
      where a threshold chosen on val is meaningful instead of everything saturating at p=1.0
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .data import DATA_DIR, load_bench, load_split, to_float
from .device import describe_device, get_device, seed_everything, worker_init_fn
from .metrics import binary_metrics, fmt, recall_at_fpr, youden_threshold
from .model import build_model, count_params, in_channels
from .ref_model import keras_param_count
from .scores import save_scores, save_val_scores


# ----------------------------------------------------------------------------------------------
def dihedral_augment(x: torch.Tensor) -> torch.Tensor:
    """Apply an independent random element of D4 to every sample in the batch (N,C,H,W)."""
    n = x.shape[0]
    k = torch.randint(0, 4, (n,), device=x.device)
    flip = torch.rand(n, device=x.device) < 0.5
    out = x.clone()
    for r in range(1, 4):
        m = k == r
        if m.any():
            out[m] = torch.rot90(x[m], r, dims=(2, 3))
    if flip.any():
        out[flip] = out[flip].flip(3)
    return out


def expand_channels(x: torch.Tensor, in_ch: int) -> torch.Tensor:
    """(N,1,H,W) -> (N,in_ch,H,W) by duplicating the grey channel (a view, no copy)."""
    return x if x.shape[1] == in_ch else x.expand(-1, in_ch, -1, -1)


def make_loader(X: np.ndarray, y: np.ndarray, batch: int, shuffle: bool, gen, sampler=None, workers: int = 0):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle and sampler is None, sampler=sampler,
                      generator=gen, num_workers=workers, worker_init_fn=worker_init_fn if workers else None,
                      pin_memory=False, drop_last=False)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, dev: torch.device, batch: int = 1024) -> np.ndarray:
    """Logits (not probabilities): thresholds are chosen in logit space so a confidently
    separated val set does not collapse every candidate threshold onto sigmoid(x) == 1.0."""
    model.eval()
    out, c = [], in_channels(model)
    for i in range(0, len(X), batch):
        xb = to_float(torch.from_numpy(np.array(X[i:i + batch]))).to(dev, non_blocking=True)  # np.array: memmap -> RAM, one batch
        out.append(model(expand_channels(xb, c)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


@torch.no_grad()
def val_loss(model, X, y, dev, crit, batch=1024) -> float:
    model.eval()
    tot, c = 0.0, in_channels(model)
    for i in range(0, len(X), batch):
        xb = to_float(torch.from_numpy(np.array(X[i:i + batch]))).to(dev)
        yb = torch.from_numpy(y[i:i + batch]).float().to(dev)
        tot += crit(model(expand_channels(xb, c)), yb).item() * len(yb)
    return tot / max(1, len(X))


class SmoothedBCE(nn.Module):
    """BCE-with-logits against label-smoothed targets; pos_weight applies to the smoothed target
    exactly as torch does for hard targets (loss = -[pw*t*log s + (1-t)*log(1-s)])."""
    def __init__(self, eps: float, pos_weight: torch.Tensor):
        super().__init__()
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t = y * (1.0 - self.eps) + 0.5 * self.eps if self.eps > 0 else y
        return self.bce(logits, t)


def _class_subset(y: np.ndarray, n_hs: int, nhs_ratio: int = 4) -> np.ndarray:
    """Indices of the first n_hs hotspots and nhs_ratio*n_hs non-hotspots (smoke mode only)."""
    return np.sort(np.concatenate([np.flatnonzero(y == 1)[:n_hs], np.flatnonzero(y == 0)[:n_hs * nhs_ratio]]))


# ----------------------------------------------------------------------------------------------
def fit(Xfit, yfit, Xval, yval, a: argparse.Namespace, dev: torch.device, gen, out: Path, extra_cfg: dict | None = None):
    """Train one model on (Xfit, yfit), monitoring (Xval, yval) every epoch. Writes config.json,
    history.json and best.pt into `out` and returns (model, history, selected_epoch, best_key, t0).

    Checkpoint selection (`--select`): 'best_val' = highest val AUC, ties by lower val loss;
    'last' = final epoch (the reference-paper protocol: a fixed epoch budget, no model selection).
    Shared by train.run_one (80/20 split) and oof.py (5-fold CV over the full train pool)."""
    n_hs, n_nhs = int(yfit.sum()), int((yfit == 0).sum())
    sampler = None
    pos_weight = torch.tensor(1.0, device=dev)
    if a.imbalance == "pos_weight":
        pos_weight = torch.tensor(n_nhs / max(1, n_hs), device=dev)
    elif a.imbalance == "oversample":
        w = np.where(yfit == 1, 0.5 / max(1, n_hs), 0.5 / max(1, n_nhs))
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(yfit),
                                        replacement=True, generator=gen)
    loader = make_loader(Xfit, yfit, a.batch_size, shuffle=True, gen=gen, sampler=sampler, workers=a.workers)

    size = int(Xfit.shape[-1])
    model = build_model(a.model, size=size, dropout=a.dropout).to(dev)
    in_ch = in_channels(model)
    crit = SmoothedBCE(a.label_smoothing, pos_weight)
    if a.optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    elif a.optimizer == "nadam":   # Keras Nadam: beta1 .9, beta2 .999, eps 1e-7 (torch default eps 1e-8; immaterial)
        opt = torch.optim.NAdam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    else:
        raise ValueError(a.optimizer)
    steps = a.epochs * len(loader)
    if a.sched == "onecycle":
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(1, steps), pct_start=0.15)
    else:
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)

    cfg = dict(**(extra_cfg or {}), **{k: v for k, v in vars(a).items() if k not in ("bench", "seed", "device")},
               device=str(dev), device_desc=describe_device(dev), params=count_params(model),
               params_keras=keras_param_count(model), in_ch=in_ch, size=size,
               n_fit=len(yfit), n_fit_hs=n_hs, n_fit_nhs=n_nhs, n_val=len(yval),
               pos_weight=float(pos_weight), torch=torch.__version__)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    hist, best, best_key, bad_epochs = [], None, (-1.0, -math.inf), 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb = to_float(xb).to(dev, non_blocking=True)
            yb = yb.float().to(dev, non_blocking=True)
            if a.augment:
                xb = dihedral_augment(xb)
            loss = crit(model(expand_channels(xb, in_ch)), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if a.clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            sched.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        train_loss = tot / max(1, n)

        pv = predict(model, Xval, dev)
        vm = binary_metrics(yval, pv, 0.0)                 # logit 0 == prob 0.5
        vl = val_loss(model, Xval, yval, dev, crit)
        rec = dict(epoch=ep, train_loss=train_loss, val_loss=vl, lr=sched.get_last_lr()[0],
                   elapsed=time.time() - t0, **{f"val_{k}": v for k, v in vm.items()})
        hist.append(rec)
        key = (vm["auc"] if not math.isnan(vm["auc"]) else -1.0, -vl)  # higher is better on both
        if a.select == "last":
            improved = True
            best_key, best = key, ep
        else:
            improved = key > best_key
            if improved:
                best_key, best, bad_epochs = key, ep, 0
            else:
                bad_epochs += 1
        if improved:
            torch.save(model.state_dict(), out / "best.pt")
        star = "*" if improved else ""
        print(f"  ep {ep:3d}/{a.epochs} loss {train_loss:.4f} vloss {vl:.4f} | val {fmt(vm)} {star}"
              f"  [{rec['elapsed']:.0f}s]")
        (out / "history.json").write_text(json.dumps(hist, indent=1))
        if a.patience and bad_epochs >= a.patience:
            print(f"  early stop: no val improvement for {a.patience} epochs")
            break
    return model, hist, best, best_key, t0


def run_one(bench: int, seed: int, a: argparse.Namespace, dev: torch.device) -> dict:
    out = Path(a.out) / a.tag / f"b{bench}" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    gen = seed_everything(seed)

    d = load_bench(bench, Path(a.data))
    tr_idx, va_idx = load_split(bench, seed, Path(a.data))
    Xfit, yfit = d["Xtr"][tr_idx], d["ytr"][tr_idx]
    Xval, yval = d["Xtr"][va_idx], d["ytr"][va_idx]
    Xte, yte = d["Xte"], d["yte"]
    if a.smoke:  # tiny, class-preserving subset so the pipeline is exercised end to end quickly
        k = _class_subset(yfit, 16); Xfit, yfit = Xfit[k], yfit[k]
        k = _class_subset(yte, 16); Xte, yte = Xte[k], yte[k]

    n_hs, n_nhs = int(yfit.sum()), int((yfit == 0).sum())
    print(f"\n=== b{bench} seed{seed} | fit {len(yfit)} (HS {n_hs}, NHS {n_nhs}) | val {len(yval)} "
          f"(HS {int(yval.sum())}) | test {len(yte)} (HS {int(yte.sum())}) | {describe_device(dev)}")

    model, hist, best, best_key, t0 = fit(Xfit, yfit, Xval, yval, a, dev, gen, out,
                                          extra_cfg=dict(bench=bench, seed=seed, n_test=len(yte)))

    # ---- official test split, evaluated exactly once with the selected checkpoint ----
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    pv = predict(model, Xval, dev)
    thr_y = youden_threshold(yval, pv)
    _, thr_fpr = recall_at_fpr(yval, pv, a.target_fpr)
    t1 = time.time()
    pt = predict(model, Xte, dev)
    infer_s = time.time() - t1
    save_scores(out, pt, yte, infer_s, dev)                      # probs.npy / labels.npy / logits.npy / eval.json
    save_val_scores(out, pv, yval)                               # val_logits.npy / val_labels.npy
    test = dict(best_epoch=best, epochs_run=len(hist), select=a.select, val_auc=float(best_key[0]), inference_s=infer_s,
                threshold_space="logit", val_thr_youden=thr_y, val_thr_fpr=thr_fpr,
                at_0p5=binary_metrics(yte, pt, 0.0),
                at_val_youden=binary_metrics(yte, pt, thr_y),
                at_val_fpr=binary_metrics(yte, pt, thr_fpr),
                target_fpr=a.target_fpr, train_time_s=time.time() - t0)
    (out / "test.json").write_text(json.dumps(test, indent=2))
    print(f"  best epoch {best} (val auc {best_key[0]:.4f})")
    print(f"  TEST @0.5           {fmt(test['at_0p5'])}")
    print(f"  TEST @val-Youden    {fmt(test['at_val_youden'])}")
    print(f"  TEST @val-fpr{a.target_fpr:<6g}{fmt(test['at_val_fpr'])}")
    return test


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    """The training recipe's flags; shared with oof.py so a CV run uses exactly the same options."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", type=int, nargs="+", default=[1])
    ap.add_argument("--seed", type=int, nargs="+", default=[0])
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default="baseline", help="sub-directory under --out grouping this experiment")
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optimizer", choices=["adamw", "nadam"], default="adamw")
    ap.add_argument("--sched", choices=["onecycle", "const"], default="onecycle")
    ap.add_argument("--select", choices=["best_val", "last"], default="best_val",
                    help="checkpoint used for the test split: best val AUC (default) or the final epoch")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0, help="grad-norm clip; 0 disables")
    ap.add_argument("--patience", type=int, default=12, help="early-stop patience in epochs; 0 disables")
    ap.add_argument("--imbalance", choices=["pos_weight", "oversample", "none"], default="pos_weight")
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="overconfidence control; 0 disables (see module docstring)")
    ap.add_argument("--target-fpr", type=float, default=0.01, help="val FPR budget for the third test threshold")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers; data is in RAM so 0 is fine")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads; 0 = torch default")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on a tiny subset; pipeline check only")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.smoke:
        a.epochs, a.patience = 2, 0
        if a.tag == "baseline":
            a.tag = "smoke"
    dev = get_device(a.device, a.deterministic)
    if a.threads > 0:
        torch.set_num_threads(a.threads)

    summary = []
    for b in a.bench:
        for s in a.seed:
            t = run_one(b, s, a, dev)
            summary.append((b, s, t["val_auc"], t["at_0p5"]["auc"], t["at_val_youden"]["accuracy"],
                            t["at_val_youden"]["false_alarms"]))
    print("\nbench seed | val AUC | test AUC | test acc@Youden | FA@Youden")
    for b, s, va, ta, acc, fa in summary:
        print(f"{b:5d} {s:4d} | {va:7.4f} | {ta:8.4f} | {acc:15.3f} | {fa}")


if __name__ == "__main__":
    main()
