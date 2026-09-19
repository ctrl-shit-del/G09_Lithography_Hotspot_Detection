"""Task 4 - out-of-fold (OOF) threshold selection by 5-fold stratified CV over the full official
training pool (fit + val of the 80/20 protocol, i.e. every official training clip).

    python -m src.oof --bench 1 4 5 --seed 0 --out runs_gpu --tag oof_arm0 --imbalance none
    python -m src.oof --bench 1 4 5 --seed 0 --out runs_gpu --tag oof_ls0.1                # pos_weight arm
    python -m src.oof --bench 5 --smoke                                                     # CPU pipeline check

Accepts every train.py flag (same recipe: model, epochs, label smoothing, imbalance handling,
augmentation, selection); `--seed` fixes the fold assignment and the init of every fold.

Why: the 80/20 val split of b5 holds 5 hotspots and its val AUC is 1.0 by epoch 3, so the
val-Youden threshold is an arbitrary point inside the separating gap. Pooling out-of-fold
scores gives a threshold chosen on all 26 (b5) / 99 (b1) / 95 (b4) training hotspots, scored
by models that never saw them.

Protocol
    * StratifiedKFold(5, shuffle=True, random_state=seed) over (Xtr, ytr) of the benchmark.
    * fold k: train on the other four folds with fold k as the val split for checkpoint
      selection (identical to train.fit: best val AUC, ties by lower val loss, early stopping).
      The fold-k model scores (a) fold k -> its slice of the OOF vector, (b) the official test
      split (evaluated once, with the selected checkpoint).
    * OOF logits are concatenated in original training-pool order and thresholds are chosen on
      them: Youden (TPR - FPR) and, for the report's cost-optimal rows, NEC(r)-minimising
      thresholds at r in {1, 2, 5, 10, 20, 50, 100, 200} and at r = N_test / P_test.
    * Each fold model's per-fold val-Youden threshold is kept alongside (the "val split"
      thresholds a single 80/20 run would have used), as is the 80/20 seed run's threshold when
      that run exists under --out/<arm tag>.
    * Test metrics: every fold model at the OOF threshold (5 single models -> mean +/- std, the
      primary numbers, since the OOF threshold was calibrated on single-model scores) and the
      5-fold mean-logit ensemble at the same threshold (secondary).

Outputs   <out>/<tag>/b{B}/seed{S}/
    fold{k}/config.json history.json best.pt probs.npy labels.npy logits.npy eval.json
            val_logits.npy val_labels.npy test.json           (train.py layout, fold k)
    oof_logits.npy  oof_labels.npy  oof_index.npy  fold_of.npy (training-pool order)
    ensemble_logits.npy                                        (mean test logit over folds)
    oof.json        thresholds (OOF-Youden, per-fold val-Youden, OOF-cost(r)), test metrics per
                    fold and for the ensemble at every threshold rule, val AUC per fold
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from . import metrics as M
from .data import load_bench
from .device import describe_device, get_device, seed_everything
from .train import _class_subset, build_parser, fit, predict
from .scores import SCORE_FILES, VAL_FILES, save_scores, save_val_scores

RS = (1, 2, 5, 10, 20, 50, 100, 200)
N_FOLDS = 5


def _test_block(yte, logits, thr_logit: float) -> dict:
    m = M.at_threshold(yte, M.sigmoid(logits), M.sigmoid(thr_logit))
    m["thr_logit"] = float(thr_logit)
    m.update(M.ranking_metrics(yte, M.sigmoid(logits)))
    return m


def run_bench(bench: int, seed: int, a: argparse.Namespace, dev: torch.device) -> dict:
    out = Path(a.out) / a.tag / f"b{bench}" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    d = load_bench(bench, Path(a.data))
    Xpool, ypool = d["Xtr"], d["ytr"]
    Xte, yte = d["Xte"], d["yte"]
    if a.smoke:
        k = _class_subset(ypool, 10, nhs_ratio=4); Xpool, ypool = Xpool[k], ypool[k]
        k = _class_subset(yte, 16); Xte, yte = np.array(Xte[k]), yte[k]
    n = len(ypool)
    print(f"\n=== OOF b{bench} seed{seed} | pool {n} (HS {int(ypool.sum())}) | test {len(yte)} (HS {int(yte.sum())}) "
          f"| {N_FOLDS} folds | {describe_device(dev)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, np.float32)
    fold_of = np.full(n, -1, np.int8)
    folds, test_logits = [], []
    t_start = time.time()
    for k, (tr, va) in enumerate(skf.split(np.zeros(n), ypool)):
        fdir = out / f"fold{k}"
        fdir.mkdir(exist_ok=True)
        gen = seed_everything(seed * 100 + k)
        Xfit, yfit, Xval, yval = Xpool[tr], ypool[tr], Xpool[va], ypool[va]
        print(f"\n--- fold {k}: fit {len(tr)} (HS {int(yfit.sum())}) | held-out {len(va)} (HS {int(yval.sum())})")
        model, hist, best, best_key, t0 = fit(Xfit, yfit, Xval, yval, a, dev, gen, fdir,
                                              extra_cfg=dict(bench=bench, seed=seed, fold=k, n_folds=N_FOLDS,
                                                             n_test=len(yte), protocol="oof"))
        model.load_state_dict(torch.load(fdir / "best.pt", map_location=dev))
        pv = predict(model, Xval, dev)
        oof[va], fold_of[va] = pv, k
        thr_y = M.youden_threshold(yval, pv)
        t1 = time.time()
        pt = predict(model, Xte, dev)
        infer_s = time.time() - t1
        test_logits.append(pt)
        for f in SCORE_FILES + VAL_FILES:                       # a re-run retrains the fold: stale scores must go
            (fdir / f).unlink(missing_ok=True)
        save_scores(fdir, pt, yte, infer_s, dev, extra=dict(protocol="oof", fold=k))
        save_val_scores(fdir, pv, yval)
        tj = dict(best_epoch=best, epochs_run=len(hist), select=a.select, val_auc=float(best_key[0]),
                  inference_s=infer_s, threshold_space="logit", val_thr_youden=thr_y,
                  at_0p5=M.binary_metrics(yte, pt, 0.0), at_val_youden=M.binary_metrics(yte, pt, thr_y),
                  train_time_s=time.time() - t0, fold=k, n_val=int(len(va)), n_val_hs=int(yval.sum()))
        (fdir / "test.json").write_text(json.dumps(tj, indent=2))
        folds.append(dict(fold=k, best_epoch=best, epochs_run=len(hist), val_auc=float(best_key[0]),
                          val_thr_youden=thr_y, n_val=int(len(va)), n_val_hs=int(yval.sum()), inference_s=infer_s))
        print(f"  fold {k}: best ep {best} val AUC {best_key[0]:.4f} | fold-val Youden thr {thr_y:+.3f} | "
              f"TEST @0.5 {M.fmt(tj['at_0p5'])}")
    assert not np.isnan(oof).any()

    # ---- thresholds on the pooled out-of-fold scores (logit space, like train.py) ----
    oof_p = M.sigmoid(oof)
    thr_oof = M.youden_threshold(ypool, oof)
    r_test = M.implied_r(yte)
    cost_thr = {}
    for key, r in [(f"{r:g}", r) for r in RS] + [("N/P", r_test)]:
        _, t = M.min_nec(ypool, oof_p, r)                       # prob space -> stored as a logit
        cost_thr[key] = dict(r=float(r), thr=float(M.logit(t)))
    oof_rank = M.ranking_metrics(ypool, oof_p)

    np.save(out / "oof_logits.npy", oof)
    np.save(out / "oof_labels.npy", ypool.astype(np.uint8))
    np.save(out / "oof_index.npy", np.arange(n))
    np.save(out / "fold_of.npy", fold_of)
    ens = np.mean(np.stack(test_logits), axis=0).astype(np.float32)
    np.save(out / "ensemble_logits.npy", ens)

    # ---- test metrics under every threshold rule ----
    rules = {"0.5": 0.0, "OOF-Youden": thr_oof}
    for name, rec in cost_thr.items():
        rules[f"OOF-cost r={name}"] = rec["thr"]
    per_fold = {name: [_test_block(yte, tl, thr) for tl in test_logits] for name, thr in rules.items()}
    per_fold["fold-val-Youden"] = [_test_block(yte, tl, f["val_thr_youden"]) for tl, f in zip(test_logits, folds)]
    ensemble = {name: _test_block(yte, ens, thr) for name, thr in rules.items()}

    # the 80/20 seed run's val-Youden threshold, when that run exists (for side-by-side reporting)
    seed_run = None
    if a.seed_run_tag:
        sj = Path(a.out) / a.seed_run_tag / f"b{bench}" / f"seed{seed}" / "test.json"
        if sj.exists():
            t = json.loads(sj.read_text())
            seed_run = dict(tag=a.seed_run_tag, val_thr_youden=t["val_thr_youden"], at_val_youden=t["at_val_youden"])

    rec = dict(bench=bench, seed=seed, tag=a.tag, n_folds=N_FOLDS, n_pool=n, n_pool_hs=int(ypool.sum()),
               n_test=int(len(yte)), n_test_hs=int(yte.sum()), r_test=r_test, smoke=a.smoke,
               threshold_space="logit", oof_thr_youden=thr_oof, oof_cost_thr=cost_thr, oof_auc=oof_rank["auc"],
               oof_ap=oof_rank["ap"], oof_pauc95=oof_rank["pauc95"], folds=folds,
               fold_val_thr_youden=[f["val_thr_youden"] for f in folds], seed_run=seed_run,
               test_per_fold=per_fold, test_ensemble=ensemble, total_time_s=time.time() - t_start,
               device=describe_device(dev), torch=torch.__version__)
    (out / "oof.json").write_text(json.dumps(rec, indent=1, default=float))

    def _ms(vals):
        v = np.asarray(vals, float)
        return f"{v.mean():.4f} ± {v.std(ddof=0):.4f}"
    print(f"\n  OOF AUC {oof_rank['auc']:.4f} AP {oof_rank['ap']:.4f} | OOF-Youden thr {thr_oof:+.3f} (logit) | "
          f"fold-val Youden thrs {[round(f['val_thr_youden'], 3) for f in folds]}")
    print(f"  {'rule':22s} {'recall':>17s} {'FA':>17s} {'bal.acc':>17s} | ensemble recall / FA")
    for name in list(rules) + ["fold-val-Youden"]:
        pf = per_fold[name]
        e = ensemble.get(name)
        print(f"  {name:22s} {_ms([m['recall'] for m in pf]):>17s} {_ms([m['fp'] for m in pf]):>17s} "
              f"{_ms([m['balanced_accuracy'] for m in pf]):>17s} | " + (f"{e['recall']:.4f} / {e['fp']}" if e else "-"))
    return rec


def main(argv=None):
    ap = build_parser(__doc__)
    ap.add_argument("--seed-run-tag", default=None,
                    help="tag of the 80/20 runs under --out whose val-Youden threshold is reported alongside")
    ap.set_defaults(bench=[1, 4, 5], tag="oof", out="runs_gpu")
    a = ap.parse_args(argv)
    if a.smoke:
        a.epochs, a.patience = 2, 0
        if a.tag == "oof":
            a.tag = "oof_smoke"
    dev = get_device(a.device, a.deterministic)
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    for b in a.bench:
        for s in a.seed:
            run_bench(b, s, a, dev)


if __name__ == "__main__":
    main()
