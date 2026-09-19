"""Phase 3 - hotspot-detection metrics. Two layers:

  1. training-time helpers (binary_metrics, youden_threshold, recall_at_fpr, fmt) used by
     train.py / evaluate.py; scores may be probabilities OR logits and thresholds come back in
     the same space.
  2. offline analysis (Task 5) that reads only a run's probs.npy / labels.npy (plus
     val_logits.npy / val_labels.npy for temperature scaling): confusion matrix and rates at a
     threshold, normalised expected cost NEC(r), the Elkan cost-optimal threshold and the
     cost ratio implied by a threshold, partial AUC at high recall, temperature scaling with
     ECE / reliability diagrams, and an exact McNemar test between two runs.

    python -m src.metrics runs_gpu/arm0/b1/seed0                       # one run, all numbers
    python -m src.metrics runs_gpu/arm0/b1/seed0 --r 1 10 100          # NEC at these cost ratios
    python -m src.metrics runs_gpu/arm0/b1/seed0 --vs runs_gpu/ls0.1/b1/seed0   # + McNemar

Conventions (ICCAD-12 / hotspot-detection literature):
    accuracy      = recall on the hotspot class  (TP / (TP + FN))   -- NOT overall accuracy
    false_alarms  = number of non-hotspots predicted as hotspots (FP), reported as a count
                    because the benchmark papers do; `fpr` is the rate for comparison
    NEC(r)        = (r*FN + FP) / (r*P + N): expected cost when a missed hotspot costs r times a
                    false alarm, normalised so that getting everything wrong scores 1.0 and
                    the trivial "all negative" rule scores r*P / (r*P + N)
    Elkan t*      = 1 / (1 + r): the Bayes-optimal threshold on a *calibrated* posterior for
                    cost ratio r (Elkan 2001). Conversely a threshold t implies r = (1 - t) / t.
    implied r     = N / P: Youden's J = TPR - FPR is maximised by the same rule as the expected
                    cost with r = N / P (each class weighted to equal mass), so a val-Youden
                    threshold is the cost-optimal rule at r = N_val / P_val; the test-set N / P
                    is emitted beside it because that is the ratio the benchmark scores at.
    pAUC          = area of the ROC region with TPR >= 0.95, i.e. mean specificity while keeping
                    at least 95 % recall; normalised by 0.05 so 1.0 is perfect and a random
                    scorer gets 0.025.
    ECE           = 15 equal-width bins on the positive-class probability, sum_b n_b/n *
                    |mean p_b - HS fraction_b|; the same quantities feed the reliability diagram.
    prior shift   = a model trained with pos_weight = w estimates q = w p / (w p + 1 - p), not the
                    posterior p; logit(p) = logit(q) - log w undoes this exactly (the same
                    correction as re-weighting the class prior). Applied before temperature
                    scaling whenever config.json records pos_weight != 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _np(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


# ==============================================================================================
# 1. training-time helpers (unchanged API; train.py and evaluate.py import these)
# ==============================================================================================
def binary_metrics(y_true, prob, thr: float = 0.5) -> dict:
    y, p = _np(y_true).astype(np.int64).ravel(), _np(prob).astype(np.float64).ravel()
    pred = (p >= thr).astype(np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    n_pos, n_neg = tp + fn, fp + tn
    both = n_pos > 0 and n_neg > 0
    return dict(
        thr=float(thr), n=int(len(y)), n_hs=n_pos, n_nhs=n_neg,
        tp=tp, fp=fp, fn=fn, tn=tn,
        accuracy=tp / n_pos if n_pos else float("nan"),          # hotspot recall
        false_alarms=fp,
        fpr=fp / n_neg if n_neg else float("nan"),
        precision=tp / (tp + fp) if (tp + fp) else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        overall_acc=(tp + tn) / len(y) if len(y) else float("nan"),
        auc=float(roc_auc_score(y, p)) if both else float("nan"),
        ap=float(average_precision_score(y, p)) if both else float("nan"),
    )


def _cut_points(y_true, score):
    """Sort by descending score and return (y_sorted, score_sorted, tpr, fpr, midpoints) where
    entry i describes the operating point 'predict positive for the i+1 highest scores' and
    midpoints[i] is a threshold realising it that sits half-way to the next score (so a
    threshold never coincides with a sample and survives saturation / float rounding).
    Entries whose score equals the next one are not valid cut points (tpr/fpr set to nan)."""
    y, s = _np(y_true).astype(np.int64).ravel(), _np(score).astype(np.float64).ravel()
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    P, N = ys.sum(), (1 - ys).sum()
    tpr = np.cumsum(ys) / max(P, 1)
    fpr = np.cumsum(1 - ys) / max(N, 1)
    nxt = np.r_[ss[1:], ss[-1] - 1.0]                       # score after this one (or below the min)
    mid = (ss + nxt) / 2.0
    valid = np.r_[ss[:-1] != ss[1:], True]
    tpr, fpr = np.where(valid, tpr, np.nan), np.where(valid, fpr, np.nan)
    return ys, ss, tpr, fpr, mid


def youden_threshold(y_true, score) -> float:
    """Threshold (in the score's own space: prob or logit) maximising TPR - FPR. Ties ->
    lowest threshold, i.e. favour recall, since a missed hotspot is the expensive error."""
    y = _np(y_true).ravel()
    if y.min() == y.max():
        return 0.5
    _, _, tpr, fpr, mid = _cut_points(y, score)
    j = np.nan_to_num(tpr - fpr, nan=-np.inf)
    i = np.flatnonzero(j == j.max())[-1]
    return float(mid[i])


def recall_at_fpr(y_true, score, max_fpr: float = 0.01) -> tuple[float, float]:
    """(recall, threshold) at the largest operating point whose FPR <= max_fpr."""
    y = _np(y_true).ravel()
    if y.min() == y.max():
        return float("nan"), 0.5
    _, _, tpr, fpr, mid = _cut_points(y, score)
    ok = np.flatnonzero(np.nan_to_num(fpr, nan=np.inf) <= max_fpr)
    if len(ok) == 0:
        return 0.0, float(mid[0]) + 1.0                      # nothing admissible: predict all negative
    i = ok[np.argmax(tpr[ok])]
    return float(tpr[i]), float(mid[i])


def fmt(m: dict) -> str:
    return (f"auc={m['auc']:.4f} ap={m['ap']:.4f} | thr={m['thr']:.3f} "
            f"acc(HS recall)={m['accuracy']:.3f} FA={m['false_alarms']} fpr={m['fpr']:.4f} "
            f"prec={m['precision']:.3f} f1={m['f1']:.3f}")


# ==============================================================================================
# 2. offline analysis on saved scores (probability space throughout)
# ==============================================================================================
def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def confusion_matrix(y_true, prob, thr: float) -> dict:
    """Counts at `prob >= thr` (the same inequality train.py used, so a val threshold that was
    chosen as a midpoint between two scores reproduces the training-time split exactly)."""
    y, p = _np(y_true).astype(np.int64).ravel(), _np(prob).astype(np.float64).ravel()
    pred = p >= thr
    pos = y == 1
    tp, fp = int((pred & pos).sum()), int((pred & ~pos).sum())
    fn, tn = int((~pred & pos).sum()), int((~pred & ~pos).sum())
    return dict(thr=float(thr), tp=tp, fp=fp, fn=fn, tn=tn, P=tp + fn, N=fp + tn)


def rates(cm: dict) -> dict:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    P, N = tp + fn, fp + tn
    rec = tp / P if P else float("nan")
    spec = tn / N if N else float("nan")
    return dict(
        recall=rec,                                              # == hotspot 'accuracy'
        specificity=spec,
        balanced_accuracy=(rec + spec) / 2,
        precision=tp / (tp + fp) if (tp + fp) else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        fpr=1 - spec,
        false_alarms=fp,
    )


def at_threshold(y_true, prob, thr: float) -> dict:
    cm = confusion_matrix(y_true, prob, thr)
    return {**cm, **rates(cm)}


def nec(cm: dict, r: float) -> float:
    """Normalised expected cost: (r*FN + FP) / (r*P + N)."""
    P, N = cm["tp"] + cm["fn"], cm["fp"] + cm["tn"]
    return (r * cm["fn"] + cm["fp"]) / (r * P + N)


def elkan_threshold(r: float) -> float:
    """Bayes-optimal probability threshold for cost ratio r (FN costs r times FP): 1 / (1 + r)."""
    return 1.0 / (1.0 + r)


def cost_ratio_of_threshold(t: float) -> float:
    """Inverse of elkan_threshold: the r for which a probability threshold t is Bayes-optimal."""
    return (1.0 - t) / t


def implied_r(y_true) -> float:
    """N / P of a label vector: the cost ratio at which Youden's J and expected cost agree."""
    y = _np(y_true).ravel()
    P = int((y == 1).sum())
    return (len(y) - P) / P if P else float("inf")


def nec_curve(y_true, prob, r: float):
    """NEC(r) at every attainable operating point. Returns (thresholds, nec) sorted by threshold
    ascending; thresholds are midpoints between consecutive distinct scores plus the
    all-negative rule (a threshold just above the highest score)."""
    ys, ss, tpr, fpr, mid = _cut_points(y_true, prob)
    P, N = int(ys.sum()), int(len(ys) - ys.sum())
    ok = ~np.isnan(tpr)
    fn = P - np.rint(tpr[ok] * max(P, 1)).astype(int)
    fp = np.rint(fpr[ok] * max(N, 1)).astype(int)
    vals = (r * fn + fp) / (r * P + N)
    thr = np.r_[mid[ok], np.nextafter(ss[0], np.inf)]
    vals = np.r_[vals, r * P / (r * P + N)]
    o = np.argsort(thr)
    return thr[o], vals[o]


def min_nec(y_true, prob, r: float) -> tuple[float, float]:
    """(NEC, threshold) of the cost-optimal operating point ON THIS SET (an oracle bound)."""
    thr, vals = nec_curve(y_true, prob, r)
    i = int(np.argmin(vals))
    return float(vals[i]), float(thr[i])


def partial_auc_high_recall(y_true, prob, min_recall: float = 0.95) -> dict:
    """Area of the ROC region with TPR >= min_recall, expressed as mean specificity over that
    recall range and normalised to [0, 1] (1 = zero false alarms at >= 95 % recall; random =
    (1 - min_recall) / 2). Also returns the FPR at exactly min_recall (linear interpolation)."""
    y, p = _np(y_true).astype(np.int64).ravel(), _np(prob).astype(np.float64).ravel()
    fpr, tpr, _ = roc_curve(y, p)
    # integrate (1 - fpr) d(tpr) over tpr in [min_recall, 1]; the ROC path from roc_curve is a
    # piecewise-linear monotone curve so trapezoids between its vertices are exact
    fpr_at = float(np.interp(min_recall, tpr, fpr))          # tpr is non-decreasing
    keep = tpr >= min_recall
    t = np.r_[min_recall, tpr[keep]]
    f = np.r_[fpr_at, fpr[keep]]
    area = float(np.trapezoid(1.0 - f, t)) if len(t) > 1 else 0.0
    return dict(pauc=area / (1.0 - min_recall), fpr_at_recall=fpr_at, min_recall=min_recall,
                specificity_at_recall=1.0 - fpr_at)


def ranking_metrics(y_true, prob) -> dict:
    y, p = _np(y_true).astype(np.int64).ravel(), _np(prob).astype(np.float64).ravel()
    both = 0 < y.sum() < len(y)
    pa = partial_auc_high_recall(y, p)
    return dict(auc=float(roc_auc_score(y, p)) if both else float("nan"),
                ap=float(average_precision_score(y, p)) if both else float("nan"),
                pauc95=pa["pauc"], fpr_at_recall95=pa["fpr_at_recall"])


# ---- calibration --------------------------------------------------------------------------------
def nll(logits, y, T: float = 1.0) -> float:
    z = np.asarray(logits, np.float64) / T
    y = np.asarray(y, np.float64)
    return float(np.mean(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)))   # stable log-sigmoid


def fit_temperature(val_logits, val_labels, bounds=(0.01, 100.0)) -> dict:
    """Guo et al. temperature scaling: T minimising the val NLL of sigmoid(z / T). Optimised over
    log T on a bounded interval; `at_bound` flags the degenerate case (a perfectly separated
    val set drives T -> 0, i.e. the NLL has no interior minimum)."""
    z, y = np.asarray(val_logits, np.float64).ravel(), np.asarray(val_labels, np.float64).ravel()
    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    res = optimize.minimize_scalar(lambda u: nll(z, y, np.exp(u)), bounds=(lo, hi), method="bounded",
                                   options=dict(xatol=1e-6))
    T = float(np.exp(res.x))
    at_bound = bool(np.log(T) - lo < 1e-3 or hi - np.log(T) < 1e-3)
    return dict(T=T, nll_before=nll(z, y, 1.0), nll_after=nll(z, y, T), at_bound=at_bound, n_val=len(y),
                n_val_hs=int(y.sum()))


def apply_temperature(logits, T: float) -> np.ndarray:
    return sigmoid(np.asarray(logits, np.float64) / T)


def prior_shift(logits, pos_weight: float) -> np.ndarray:
    """Undo a pos_weight-weighted loss: logit(p) = logit(q) - log(pos_weight)."""
    return np.asarray(logits, np.float64) - np.log(pos_weight)


def reliability(y_true, prob, n_bins: int = 15) -> dict:
    """Per-bin mean predicted positive probability vs observed HS fraction (equal-width bins)."""
    y, p = _np(y_true).astype(np.float64).ravel(), _np(prob).astype(np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins)
    conf = np.bincount(idx, weights=p, minlength=n_bins) / np.maximum(count, 1)
    acc = np.bincount(idx, weights=y, minlength=n_bins) / np.maximum(count, 1)
    ece = float(np.sum(count / max(1, len(y)) * np.abs(conf - acc)))
    return dict(edges=edges, count=count, conf=np.where(count > 0, conf, np.nan),
                acc=np.where(count > 0, acc, np.nan), ece=ece, n=int(len(y)))


def ece(y_true, prob, n_bins: int = 15) -> float:
    return reliability(y_true, prob, n_bins)["ece"]


def calibration_report(val_logits, val_labels, test_logits, test_labels, n_bins: int = 15, pos_weight: float = 1.0) -> dict:
    """Prior-shift by pos_weight (no-op for 1.0), fit T on val, report ECE + reliability diagram
    data on test before (raw sigmoid) and after (shift + T). `probs_after` are the calibrated
    test probabilities the cost-sensitive thresholds in sweep.py act on."""
    zv, zt = prior_shift(val_logits, pos_weight), prior_shift(test_logits, pos_weight)
    fit = fit_temperature(zv, val_labels)
    before = reliability(test_labels, sigmoid(test_logits), n_bins)
    p_after = apply_temperature(zt, fit["T"])
    after = reliability(test_labels, p_after, n_bins)
    return dict(**fit, pos_weight=float(pos_weight), log_shift=float(-np.log(pos_weight)),
                ece_before=before["ece"], ece_after=after["ece"],
                nll_test_before=nll(test_logits, test_labels, 1.0), nll_test_after=nll(zt, test_labels, fit["T"]),
                reliability_before=before, reliability_after=after, probs_after=p_after,
                val_probs_after=apply_temperature(zv, fit["T"]))


# ---- paired comparison --------------------------------------------------------------------------
def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact (binomial) two-sided McNemar test on the discordant pairs of two classifiers scored
    on the same samples. b = A right & B wrong, c = A wrong & B right. Also splits the
    discordants by class so the direction (recall vs false alarms) is visible."""
    y = _np(y_true).astype(bool).ravel()
    a, b_ = _np(pred_a).astype(bool).ravel(), _np(pred_b).astype(bool).ravel()
    ra, rb = a == y, b_ == y
    b, c = int((ra & ~rb).sum()), int((~ra & rb).sum())
    n = b + c
    p = float(stats.binomtest(min(b, c), n, 0.5).pvalue) if n else 1.0
    return dict(b=b, c=c, n_discordant=n, p_value=p,
                b_hs=int((ra & ~rb & y).sum()), c_hs=int((~ra & rb & y).sum()),
                b_nhs=int((ra & ~rb & ~y).sum()), c_nhs=int((~ra & rb & ~y).sum()),
                acc_a=float(ra.mean()), acc_b=float(rb.mean()))


# ---- run I/O ------------------------------------------------------------------------------------
def load_run(run_dir) -> dict:
    """probs / labels (+ val logits & labels when present) and the thresholds train.py recorded,
    converted from logit space to probability space. Only .npy / .json files are read."""
    d = Path(run_dir)
    out = dict(run=str(d), probs=np.load(d / "probs.npy").astype(np.float64),
               labels=np.load(d / "labels.npy").astype(np.int64))
    tj = d / "test.json"
    if tj.exists():
        t = json.loads(tj.read_text())
        conv = sigmoid if t.get("threshold_space", "logit") == "logit" else float
        out["thr_youden"] = float(conv(t["val_thr_youden"]))
        if "val_thr_fpr" in t:
            out["thr_fpr"] = float(conv(t["val_thr_fpr"]))
        out["test_json"] = t
    for name in ("config", "eval"):
        if (d / f"{name}.json").exists():
            out[name] = json.loads((d / f"{name}.json").read_text())
    if (d / "val_logits.npy").exists():
        out["val_logits"] = np.load(d / "val_logits.npy").astype(np.float64)
        out["val_labels"] = np.load(d / "val_labels.npy").astype(np.int64)
    if (d / "logits.npy").exists():
        out["logits"] = np.load(d / "logits.npy").astype(np.float64)
    return out


def summarize_run(run: dict, rs=(1, 2, 5, 10, 20, 50, 100, 200)) -> dict:
    """Everything Task 5 defines, for one run, at p=0.5 and at the val-Youden threshold."""
    y, p = run["labels"], run["probs"]
    r_np = implied_r(y)
    rows = {"0.5": 0.5}
    if "thr_youden" in run:
        rows["val-Youden"] = run["thr_youden"]
    res = dict(run=run["run"], n=int(len(y)), P=int(y.sum()), N=int(len(y) - y.sum()), implied_r=r_np,
               **ranking_metrics(y, p), rules={})
    for name, thr in rows.items():
        m = at_threshold(y, p, thr)
        m["implied_r_of_thr"] = cost_ratio_of_threshold(thr)
        m["nec"] = {str(r): nec(m, r) for r in rs}
        m["nec_at_N_over_P"] = nec(m, r_np)
        res["rules"][name] = m
    res["elkan"] = {}
    for r in rs:
        m = at_threshold(y, p, elkan_threshold(r))
        best, bthr = min_nec(y, p, r)
        res["elkan"][str(r)] = dict(thr=m["thr"], tp=m["tp"], fp=m["fp"], fn=m["fn"], tn=m["tn"],
                                    nec=nec(m, r), nec_oracle=best, thr_oracle=bthr)
    if "val_logits" in run and "logits" in run:
        cal = calibration_report(run["val_logits"], run["val_labels"], run["logits"], y,
                                 pos_weight=run.get("config", {}).get("pos_weight", 1.0))
        res["calibration"] = {k: v for k, v in cal.items() if not (k.startswith("reliability") or k.endswith("probs_after"))}
    return res


def _print_summary(s: dict):
    print(f"{s['run']}: n={s['n']} P={s['P']} N={s['N']}  implied r=N/P={s['implied_r']:.2f}")
    print(f"  AUC {s['auc']:.4f}  AP {s['ap']:.4f}  pAUC(recall>=.95) {s['pauc95']:.4f}  FPR@recall.95 {s['fpr_at_recall95']:.4f}")
    for name, m in s["rules"].items():
        print(f"  @{name:<11s} thr={m['thr']:.4f} (r_of_thr={m['implied_r_of_thr']:.2f}) TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']} "
              f"bal.acc={m['balanced_accuracy']:.4f} prec={m['precision']:.4f} rec={m['recall']:.4f} "
              f"spec={m['specificity']:.4f} f1={m['f1']:.4f}  NEC(N/P)={m['nec_at_N_over_P']:.4f}")
        print("     NEC(r): " + "  ".join(f"r={r}:{v:.4f}" for r, v in m["nec"].items()))
    print("  Elkan t*=1/(1+r) on raw probs   [NEC | oracle-min NEC on test]")
    for r, e in s["elkan"].items():
        print(f"     r={r:>4s} t*={e['thr']:.4f} TP={e['tp']} FP={e['fp']} FN={e['fn']}  NEC={e['nec']:.4f} | {e['nec_oracle']:.4f} @ {e['thr_oracle']:.4f}")
    if "calibration" in s:
        c = s["calibration"]
        print(f"  calibration: prior shift {c['log_shift']:+.3f} (pos_weight {c['pos_weight']:.2f}), T={c['T']:.3f}"
              f"{' (AT BOUND: val set separable)' if c['at_bound'] else ''} "
              f"val NLL {c['nll_before']:.4f}->{c['nll_after']:.4f} | test ECE {c['ece_before']:.4f}->{c['ece_after']:.4f} "
              f"test NLL {c['nll_test_before']:.4f}->{c['nll_test_after']:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run directory containing probs.npy / labels.npy")
    ap.add_argument("--vs", default=None, help="second run on the same test set for a McNemar test")
    ap.add_argument("--r", type=float, nargs="+", default=[1, 2, 5, 10, 20, 50, 100, 200])
    ap.add_argument("--rule", choices=["0.5", "val-Youden"], default="val-Youden", help="threshold rule for McNemar")
    ap.add_argument("--json", default=None, help="write the summary here")
    a = ap.parse_args(argv)
    run = load_run(a.run)
    s = summarize_run(run, a.r)
    _print_summary(s)
    if a.vs:
        other = load_run(a.vs)
        assert np.array_equal(run["labels"], other["labels"]), "runs are not on the same test set"
        ta = 0.5 if a.rule == "0.5" else run["thr_youden"]
        tb = 0.5 if a.rule == "0.5" else other["thr_youden"]
        m = mcnemar_exact(run["labels"], run["probs"] >= ta, other["probs"] >= tb)
        s["mcnemar"] = dict(vs=str(a.vs), rule=a.rule, **m)
        print(f"  McNemar vs {a.vs} @{a.rule}: b={m['b']} (HS {m['b_hs']}, NHS {m['b_nhs']}) c={m['c']} "
              f"(HS {m['c_hs']}, NHS {m['c_nhs']})  exact p={m['p_value']:.3g}")
    if a.json:
        Path(a.json).write_text(json.dumps(s, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()
