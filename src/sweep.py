"""Task 6 - cost-ratio sweep over saved probabilities (no retraining, no model loading).

    python -m src.sweep --runs runs_gpu                      # numbers: results/sweep_runs.csv, *.json, sweep_summary.txt
    python -m src.sweep --runs runs_gpu --figures            # + figures/*.png, figures/captions.md
    python -m src.sweep --runs runs_gpu --arms arm0 ls0.1    # restrict arms (default: arm0 ls0.1 ref64)

Reads, per run: probs.npy, labels.npy, test.json (val-Youden threshold), config.json (pos_weight)
and, when present, val_logits.npy / val_labels.npy (backfilled by `python -m src.evaluate --val`).

Cost ratio r = cost(FN) / cost(FP); NEC(r) = (r FN + FP) / (r P + N). Five threshold rules are
swept over r in R = {1, 2, 5, 10, 20, 50, 100, 200} (and a fine log grid for the curves):

    val-Youden   fixed threshold maximising TPR - FPR on the val split (train.py's choice). NEC
                 varies with r only through the weighting, so the curve is (r FN + FP)/(r P + N)
                 with fixed counts. Its implied cost ratio is N_val / P_val (emitted beside it).
    elkan-raw    t* = 1/(1+r) applied to the raw sigmoid output. Label smoothing (eps = 0.1) caps
                 raw probabilities inside ~[0.05, 0.95] and pos_weight shifts them by log(w), so
                 this rule degenerates for r >~ 20 (everything positive); kept to show that.
    elkan-cal    t* = 1/(1+r) applied to calibrated probabilities: logits prior-shifted by
                 -log(pos_weight) then temperature-scaled with T fitted on the val logits
                 (metrics.calibration_report). Flagged when T hit its bound (separable val set).
    val-cost     threshold minimising NEC(r) on the val split (empirical cost minimisation, no
                 calibration assumption), applied to the test probabilities.
    oracle       threshold minimising NEC(r) on the TEST set: a lower bound, not a deployable rule.

Crossover: the r where NEC_ls0.1(r) - NEC_arm0(r) changes sign, reported with the arm that is
better ABOVE it ('R>ls' = ls0.1 better for r > R). At its val-Youden point the pos_weight arm is
often the more conservative one (fewer false alarms, lower recall), so the crossing can run
either way. For val-Youden the counts are fixed and the sign changes at most once, at
r* = (FP1 - FP0) / (FN0 - FN1) in closed form; the r-dependent rules are read off the fine r
grid by log-linear interpolation and may cross more than once (all crossings are kept).

OOF rules (from src/oof.py outputs, --oof-tags): the five fold models of a benchmark are scored
on the official test split with thresholds chosen on the pooled out-of-fold scores:
    oof-Youden   fixed OOF Youden threshold;   oof-cost   NEC(r)-minimising OOF threshold per r.
When the OOF pool is perfectly separated (OOF AUC = 1.0: b1 and b5 for both arms) every r selects
the same threshold inside the separating gap, so oof-cost == oof-Youden and the NEC(r) curve is
just the fixed-count curve (r FN + FP)/(r P + N); the figure notes this instead of plotting it.
The oof-cost curve is drawn for b4 (OOF AUC 0.994 / 0.997), the primary cost-curve panel.

Multi-seed benchmarks (b2 / b4 / b5): mean +/- std (ddof = 0) over the five seeds and a paired
comparison arm0 - ls0.1 by seed (same split per seed) with a paired t-test and an exact
Wilcoxon signed-rank test (n = 5 -> the smallest attainable two-sided Wilcoxon p is 0.0625).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from . import metrics as M

RS = (1, 2, 5, 10, 20, 50, 100, 200)
RULES = ("0.5", "val-Youden", "elkan-raw", "elkan-cal", "val-cost", "oracle")
FIXED = ("0.5", "val-Youden")
ARMS_DEFAULT = ("arm0", "ls0.1", "ref64")
BENCHES = (1, 2, 3, 4, 5)
R_GRID = np.logspace(np.log10(0.5), np.log10(1000), 241)


# ----------------------------------------------------------------------------------------------
def find_runs(root: Path, arms):
    out = {}
    for arm in arms:
        for d in sorted((root / arm).glob("b*/seed*")):
            if (d / "probs.npy").exists() and (d / "labels.npy").exists():
                b, s = int(d.parent.name[1:]), int(d.name[4:])
                out[(arm, b, s)] = d
    return out


def _cm_at(y, p, thr):
    return M.confusion_matrix(y, p, thr)


def sweep_run(run: dict, rs=RS, r_grid=R_GRID) -> dict:
    """All rules for one run at r in rs (full counts) and on r_grid (NEC only, for curves)."""
    y, p = run["labels"], run["probs"]
    P, N = int(y.sum()), int(len(y) - y.sum())
    r_test = N / P
    res = dict(run=run["run"], n=len(y), P=P, N=N, r_test=r_test, **M.ranking_metrics(y, p))

    # calibrated probabilities (needs val logits + test logits)
    cal = None
    if "val_logits" in run and "logits" in run:
        cal = M.calibration_report(run["val_logits"], run["val_labels"], run["logits"], y,
                                   pos_weight=run.get("config", {}).get("pos_weight", 1.0))
        res["calibration"] = dict(T=cal["T"], at_bound=cal["at_bound"], pos_weight=cal["pos_weight"],
                                  ece_before=cal["ece_before"], ece_after=cal["ece_after"],
                                  nll_test_before=cal["nll_test_before"], nll_test_after=cal["nll_test_after"],
                                  n_val=cal["n_val"], n_val_hs=cal["n_val_hs"])
    have_val = "val_logits" in run
    if have_val:
        pv, yv = M.sigmoid(run["val_logits"]), run["val_labels"]
        res["r_val"] = M.implied_r(yv)
        res["n_val"], res["n_val_hs"] = int(len(yv)), int(yv.sum())

    # fixed rules
    fixed = {"0.5": 0.5}
    if "thr_youden" in run:
        fixed["val-Youden"] = run["thr_youden"]
    res["fixed"] = {}
    for name, thr in fixed.items():
        m = M.at_threshold(y, p, thr)
        m["nec"] = {str(r): M.nec(m, r) for r in rs}
        m["nec_r_test"] = M.nec(m, r_test)
        m["nec_grid"] = [M.nec(m, r) for r in r_grid]
        m["implied_r_of_thr"] = M.cost_ratio_of_threshold(thr)
        res["fixed"][name] = m

    # r-dependent rules
    def rule_thr(rule, r):
        if rule == "elkan-raw":
            return M.elkan_threshold(r), p
        if rule == "elkan-cal":
            if cal is None:
                return None, None
            return M.elkan_threshold(r), cal["probs_after"]
        if rule == "val-cost":
            if not have_val:
                return None, None
            _, t = M.min_nec(yv, pv, r)
            return t, p
        if rule == "oracle":
            _, t = M.min_nec(y, p, r)
            return t, p
        raise KeyError(rule)

    res["rules"] = {}
    for rule in ("elkan-raw", "elkan-cal", "val-cost", "oracle"):
        rows, grid = {}, []
        for r in rs:
            t, pp = rule_thr(rule, r)
            if t is None:
                rows[str(r)] = None
                continue
            m = M.at_threshold(y, pp, t)
            m["nec"] = M.nec(m, r)
            rows[str(r)] = m
        for r in r_grid:
            t, pp = rule_thr(rule, r)
            grid.append(float("nan") if t is None else M.nec(_cm_at(y, pp, t), r))
        t, pp = rule_thr(rule, r_test)
        res["rules"][rule] = dict(at=rows, nec_grid=grid,
                                  nec_r_test=(float("nan") if t is None else M.nec(_cm_at(y, pp, t), r_test)),
                                  thr_r_test=(float("nan") if t is None else float(t)))
    return res


def nec_grid_of(res: dict, rule: str):
    if rule in FIXED:
        return np.asarray(res["fixed"][rule]["nec_grid"], float)
    return np.asarray(res["rules"][rule]["nec_grid"], float)


def nec_r_test_of(res: dict, rule: str) -> float:
    if rule in FIXED:
        return res["fixed"][rule]["nec_r_test"]
    return res["rules"][rule]["nec_r_test"]


def crossover(diff: np.ndarray, r_grid=R_GRID) -> dict:
    """Sign structure of diff(r) = NEC_ls0.1(r) - NEC_arm0(r) along increasing r.

    kind      'ls0.1'  : cost arm no worse at every r (and strictly better somewhere)
              'arm0'   : control no worse everywhere
              'tie'    : identical curves
              'cross'  : sign changes; `r` is the first strict crossing (log-linear interpolated)
                         and `above` names the arm that is better just above it. Every crossing is
                         listed in `all` as (r, arm better above).
    Zeros (both arms saturating to the same rule) are neutral and never count as a crossing."""
    d = np.asarray(diff, float)
    ok = ~np.isnan(d)
    if ok.sum() < 2:
        return dict(r=float("nan"), kind="undefined", above=None, all=[])
    d, g = d[ok], r_grid[ok]
    if np.all(d == 0):
        return dict(r=float("nan"), kind="tie", above=None, all=[])
    if np.all(d <= 0):
        return dict(r=0.0, kind="ls0.1", above="ls0.1", all=[])
    if np.all(d >= 0):
        return dict(r=float("inf"), kind="arm0", above="arm0", all=[])
    xs, last_sign, last_i = [], 0, None
    for i in range(len(d)):
        sgn = int(np.sign(d[i]))
        if sgn == 0:
            continue
        if last_sign and sgn != last_sign:
            a, b = d[last_i], d[i]
            f = a / (a - b)
            r = float(np.exp(np.log(g[last_i]) + f * (np.log(g[i]) - np.log(g[last_i]))))
            xs.append((r, "ls0.1" if sgn < 0 else "arm0"))
        last_sign, last_i = sgn, i
    return dict(r=xs[0][0], kind="cross", above=xs[0][1], all=xs)


def youden_crossover_closed_form(m0: dict, m1: dict) -> dict:
    """Two fixed operating points on the same P, N: NEC1 - NEC0 has the sign of
    r (FN1 - FN0) + (FP1 - FP0), which changes sign at most once, at r* = (FP1 - FP0)/(FN0 - FN1)."""
    dfn, dfp = m1["fn"] - m0["fn"], m1["fp"] - m0["fp"]
    if dfn == 0 and dfp == 0:
        return dict(r=float("nan"), kind="tie", above=None)
    if dfn <= 0 and dfp <= 0:
        return dict(r=0.0, kind="ls0.1", above="ls0.1")
    if dfn >= 0 and dfp >= 0:
        return dict(r=float("inf"), kind="arm0", above="arm0")
    r = -dfp / dfn
    return dict(r=float(r), kind="cross", above="ls0.1" if dfn < 0 else "arm0")


# ----------------------------------------------------------------------------------------------
def _ms(v):
    v = np.asarray([x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if len(v) == 0:
        return "n/a"
    return f"{v.mean():.4f}" if len(v) == 1 else f"{v.mean():.4f} ± {v.std(ddof=0):.4f}"


def _msi(v):
    v = np.asarray(v, float)
    return f"{int(v[0])}" if len(v) == 1 else f"{v.mean():.0f} ± {v.std(ddof=0):.0f}"


def paired(a, b) -> dict:
    """arm0 - ls0.1 paired by seed: mean diff, std, paired t, exact Wilcoxon."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    out = dict(n=int(len(d)), mean_diff=float(d.mean()), std_diff=float(d.std(ddof=0)))
    if len(d) >= 2 and np.any(d != 0):
        out["t_p"] = float(stats.ttest_rel(a, b).pvalue)
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(a, b, method="exact").pvalue)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    else:
        out["t_p"] = out["wilcoxon_p"] = float("nan")
    return out


def aggregate(results: dict, arms, rs=RS) -> dict:
    """results: {(arm, bench, seed): sweep_run(...)}. Returns per-benchmark summaries."""
    summary = {}
    for b in BENCHES:
        per_arm = {}
        for arm in arms:
            seeds = sorted(s for (a, bb, s) in results if a == arm and bb == b)
            if not seeds:
                continue
            rr = [results[(arm, b, s)] for s in seeds]
            e = dict(seeds=seeds, n_seeds=len(seeds), P=rr[0]["P"], N=rr[0]["N"], r_test=rr[0]["r_test"],
                     auc=[r["auc"] for r in rr], ap=[r["ap"] for r in rr], pauc95=[r["pauc95"] for r in rr],
                     fpr_at_recall95=[r["fpr_at_recall95"] for r in rr],
                     thr_youden=[r["fixed"]["val-Youden"]["thr"] for r in rr],
                     r_val=[r.get("r_val", float("nan")) for r in rr],
                     n_val_hs=[r.get("n_val_hs") for r in rr],
                     T=[r.get("calibration", {}).get("T", float("nan")) for r in rr],
                     pos_weight=float(np.mean([r.get("calibration", {}).get("pos_weight", 1.0) for r in rr])),
                     T_at_bound=[r.get("calibration", {}).get("at_bound") for r in rr],
                     ece_before=[r.get("calibration", {}).get("ece_before", float("nan")) for r in rr],
                     ece_after=[r.get("calibration", {}).get("ece_after", float("nan")) for r in rr],
                     nec={}, nec_r_test={}, fa_youden=[r["fixed"]["val-Youden"]["fp"] for r in rr],
                     recall_youden=[r["fixed"]["val-Youden"]["recall"] for r in rr],
                     nec_grid_mean={})
            for rule in RULES:
                e["nec"][rule] = {str(r): [(r_["fixed"][rule]["nec"][str(r)] if rule in FIXED
                                            else (r_["rules"][rule]["at"][str(r)] or {}).get("nec", float("nan")))
                                           for r_ in rr] for r in rs}
                e["nec_r_test"][rule] = [nec_r_test_of(r_, rule) for r_ in rr]
                e["nec_grid_mean"][rule] = np.nanmean(np.stack([nec_grid_of(r_, rule) for r_ in rr]), axis=0).tolist()
            per_arm[arm] = e
        comp = {}
        if "arm0" in per_arm and "ls0.1" in per_arm:
            s0, s1 = per_arm["arm0"]["seeds"], per_arm["ls0.1"]["seeds"]
            common = sorted(set(s0) & set(s1))
            r0 = {s: results[("arm0", b, s)] for s in common}
            r1 = {s: results[("ls0.1", b, s)] for s in common}
            comp["seeds"] = common
            comp["auc"] = paired([r0[s]["auc"] for s in common], [r1[s]["auc"] for s in common])
            comp["ap"] = paired([r0[s]["ap"] for s in common], [r1[s]["ap"] for s in common])
            comp["pauc95"] = paired([r0[s]["pauc95"] for s in common], [r1[s]["pauc95"] for s in common])
            comp["nec_r_test"] = {rule: paired([nec_r_test_of(r0[s], rule) for s in common],
                                              [nec_r_test_of(r1[s], rule) for s in common]) for rule in RULES}
            # crossovers: per seed (paired) and on the seed-mean curves
            comp["crossover"] = {}
            for rule in RULES:
                per_seed = [crossover(nec_grid_of(r1[s], rule) - nec_grid_of(r0[s], rule)) for s in common]
                mean_curve = crossover(np.asarray(per_arm["ls0.1"]["nec_grid_mean"][rule]) - np.asarray(per_arm["arm0"]["nec_grid_mean"][rule]))
                rec = dict(per_seed=per_seed, mean_curve=mean_curve)
                if rule == "val-Youden":
                    rec["closed_form_per_seed"] = [youden_crossover_closed_form(r0[s]["fixed"]["val-Youden"], r1[s]["fixed"]["val-Youden"]) for s in common]
                comp["crossover"][rule] = rec
        summary[b] = dict(arms=per_arm, arm0_vs_ls01=comp)
    return summary


# ----------------------------------------------------------------------------------------------
def _fmt_cross(c: dict) -> str:
    """'12.3>ls' = above r = 12.3 the ls0.1 arm is better; 'ls' / 'arm0' = that arm better at every
    r; 'tie' = identical; further crossings are appended after ';'."""
    if c["kind"] == "cross":
        xs = c.get("all") or [(c["r"], c["above"])]
        return ";".join(f"{r:.3g}>{'ls' if a == 'ls0.1' else 'arm0'}" for r, a in xs)
    if c["kind"] == "ls0.1":
        return "ls"
    return c["kind"]


def print_summary(summary: dict, rs=RS):
    for b, S in summary.items():
        if not S["arms"]:
            continue
        any_arm = next(iter(S["arms"].values()))
        print(f"\n=== b{b}: test P={any_arm['P']} N={any_arm['N']}  r_test = N/P = {any_arm['r_test']:.2f} ===")
        print(f"{'arm':6s} {'seeds':>5s} {'AUC':>17s} {'AP':>17s} {'pAUC95':>17s} {'FPR@rec95':>17s} {'FA@Youden':>14s} {'rec@Youden':>17s}")
        for arm, e in S["arms"].items():
            print(f"{arm:6s} {e['n_seeds']:>5d} {_ms(e['auc']):>17s} {_ms(e['ap']):>17s} {_ms(e['pauc95']):>17s} "
                  f"{_ms(e['fpr_at_recall95']):>17s} {_msi(e['fa_youden']):>14s} {_ms(e['recall_youden']):>17s}")
        print("  val-Youden thr (prob) / implied r_val = N_val/P_val / calibration T (bound flags) / test ECE before -> after")
        for arm, e in S["arms"].items():
            flags = sum(1 for f in e["T_at_bound"] if f)
            bound = f" [{flags}/{e['n_seeds']} at bound]" if flags else ""
            print(f"  {arm:6s} thr={_ms(e['thr_youden'])}  r_val={_ms(e['r_val'])} (val HS {e['n_val_hs'][0]})  "
                  f"T={_ms(e['T'])}{bound}  ECE {_ms(e['ece_before'])} -> {_ms(e['ece_after'])}")
        print("  NEC(r), mean over seeds  (rows: rule; cols: r)          | NEC at r_test")
        hdr = "  " + f"{'arm':6s} {'rule':11s}" + "".join(f"{('r=' + str(r)):>9s}" for r in rs) + f"{'r=N/P':>11s}"
        print(hdr)
        starred = False
        for arm, e in S["arms"].items():
            for rule in RULES:
                vals = [np.nanmean(np.asarray(e["nec"][rule][str(r)], float)) if len(e["nec"][rule][str(r)]) else float("nan") for r in rs]
                name = rule
                if rule == "0.5" and e.get("pos_weight", 1.0) != 1.0:
                    name, starred = "0.5 *", True
                print("  " + f"{arm:6s} {name:11s}" + "".join(f"{v:>9.4f}" for v in vals) + f"{np.nanmean(np.asarray(e['nec_r_test'][rule], float)):>11.4f}")
        if starred:
            print("  * p = 0.5 is not a valid operating point for a pos_weight arm (sigmoid is not a posterior; logit 0 flags every negative when w >> 1)")
        C = S["arm0_vs_ls01"]
        if C:
            print(f"  paired arm0 - ls0.1 over seeds {C['seeds']}  (mean diff ± std; paired t p; exact Wilcoxon p)")
            for k in ("auc", "ap", "pauc95"):
                c = C[k]
                print(f"    {k:14s} {c['mean_diff']:+.4f} ± {c['std_diff']:.4f}   t p={c['t_p']:.3f}   W p={c['wilcoxon_p']:.3f}")
            for rule in RULES:
                c = C["nec_r_test"][rule]
                print(f"    NEC@N/P {rule:9s} {c['mean_diff']:+.4f} ± {c['std_diff']:.4f}   t p={c['t_p']:.3f}   W p={c['wilcoxon_p']:.3f}")
            print("  crossover of NEC(r) between arms ('R>ls' = ls0.1 better above r = R; 'ls'/'arm0' = better at every r): per seed | seed-mean curves")
            for rule in RULES:
                x = C["crossover"][rule]
                ps = ", ".join(_fmt_cross(c) for c in x["per_seed"])
                print(f"    {rule:11s} {ps:52s} | {_fmt_cross(x['mean_curve'])}")
                if rule == "val-Youden":
                    print(f"    {'':11s} closed form: {', '.join(_fmt_cross(c) for c in x['closed_form_per_seed'])}")


def write_outputs(results: dict, summary: dict, out: Path, rs=RS):
    out.mkdir(parents=True, exist_ok=True)
    # flat CSV: one row per run x rule x r
    with open(out / "sweep_runs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "bench", "seed", "rule", "r", "thr", "tp", "fp", "fn", "tn", "recall", "specificity",
                    "balanced_accuracy", "precision", "f1", "nec", "auc", "ap", "pauc95", "r_test", "r_val", "T", "T_at_bound"])
        for (arm, b, s), R in sorted(results.items()):
            common = [R["auc"], R["ap"], R["pauc95"], R["r_test"], R.get("r_val", ""),
                      R.get("calibration", {}).get("T", ""), R.get("calibration", {}).get("at_bound", "")]
            for name, m in R["fixed"].items():
                for r in rs:
                    w.writerow([arm, b, s, name, r, m["thr"], m["tp"], m["fp"], m["fn"], m["tn"], m["recall"], m["specificity"],
                                m["balanced_accuracy"], m["precision"], m["f1"], m["nec"][str(r)]] + common)
            for rule, rec in R["rules"].items():
                for r in rs:
                    m = rec["at"][str(r)]
                    if m is None:
                        continue
                    w.writerow([arm, b, s, rule, r, m["thr"], m["tp"], m["fp"], m["fn"], m["tn"], m["recall"], m["specificity"],
                                m["balanced_accuracy"], m["precision"], m["f1"], m["nec"]] + common)
    slim = {f"b{b}": dict(arms={a: {k: v for k, v in e.items() if k != "nec_grid_mean"} for a, e in S["arms"].items()},
                          arm0_vs_ls01=S["arm0_vs_ls01"]) for b, S in summary.items()}
    (out / "sweep_summary.json").write_text(json.dumps(slim, indent=1, default=_json_default))
    grids = {f"{a}/b{b}/seed{s}": {rule: nec_grid_of(R, rule).tolist() for rule in RULES} for (a, b, s), R in results.items()}
    (out / "nec_grid.json").write_text(json.dumps(dict(r_grid=R_GRID.tolist(), curves=grids), default=_json_default))


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return str(o)
    return str(o)


# ----------------------------------------------------------------------------------------------
def find_oof(root: Path, oof_tags):
    """{(arm, bench): dir} for --oof-tags entries 'tag:arm'."""
    out = {}
    for spec in oof_tags:
        tag, _, arm = spec.partition(":")
        arm = arm or tag
        for oj in sorted((root / tag).glob("b*/seed*/oof.json")):
            b = int(oj.parent.parent.name[1:])
            out[(arm, b)] = oj.parent
    return out


def sweep_oof(d: Path, rs=RS, r_grid=R_GRID) -> dict:
    """oof-Youden and oof-cost rules for the five fold models of one (arm, bench)."""
    rec = json.loads((d / "oof.json").read_text())
    oof_p, oof_y = M.sigmoid(np.load(d / "oof_logits.npy")), np.load(d / "oof_labels.npy").astype(int)
    folds = [M.load_run(d / f"fold{k}") for k in range(rec["n_folds"])]
    y = folds[0]["labels"]
    r_test = M.implied_r(y)
    thr_y = M.sigmoid(rec["oof_thr_youden"])
    thr_grid = np.array([M.min_nec(oof_y, oof_p, r)[1] for r in r_grid])
    thr_rs = {str(r): M.min_nec(oof_y, oof_p, r)[1] for r in rs}
    thr_rt = M.min_nec(oof_y, oof_p, r_test)[1]
    n_distinct = int(len(np.unique(np.round(thr_grid, 6))))
    degenerate = n_distinct == 1
    res = dict(dir=str(d), n_folds=rec["n_folds"], oof_auc=rec["oof_auc"], oof_ap=rec["oof_ap"], n_pool=rec["n_pool"],
               n_pool_hs=rec["n_pool_hs"], P=int(y.sum()), N=int(len(y) - y.sum()), r_test=r_test, thr_youden=thr_y,
               thr_grid=thr_grid.tolist(), n_distinct_thr=n_distinct, degenerate=degenerate,
               fold_val_thr_youden=[M.sigmoid(t) for t in rec["fold_val_thr_youden"]],
               fold_val_auc=[f["val_auc"] for f in rec["folds"]], folds=[])
    for k, fr in enumerate(folds):
        p = fr["probs"]
        fy = M.at_threshold(y, p, thr_y)
        fv = M.at_threshold(y, p, res["fold_val_thr_youden"][k])
        f = dict(fold=k, auc=M.ranking_metrics(y, p)["auc"],
                 oof_youden=dict(**fy, nec_grid=[M.nec(fy, r) for r in r_grid], nec_r_test=M.nec(fy, r_test),
                                 nec={str(r): M.nec(fy, r) for r in rs}),
                 fold_val_youden=dict(**fv, nec_grid=[M.nec(fv, r) for r in r_grid], nec_r_test=M.nec(fv, r_test)),
                 oof_cost=dict(nec_grid=[M.nec(_cm_at(y, p, t), r) for t, r in zip(thr_grid, r_grid)],
                               at={str(r): dict(**M.at_threshold(y, p, thr_rs[str(r)]),
                                               nec=M.nec(_cm_at(y, p, thr_rs[str(r)]), r)) for r in rs},
                               nec_r_test=M.nec(_cm_at(y, p, thr_rt), r_test), thr_r_test=float(thr_rt)))
        res["folds"].append(f)
    for rule in ("oof_youden", "fold_val_youden", "oof_cost"):
        res[f"{rule}_grid_mean"] = np.mean([f[rule]["nec_grid"] for f in res["folds"]], axis=0).tolist()
    return res


def print_oof(oof: dict, rs=RS):
    by_bench = {}
    for (arm, b), o in oof.items():
        by_bench.setdefault(b, {})[arm] = o
    for b in sorted(by_bench):
        print(f"\n=== OOF b{b} (5 fold models each; thresholds chosen on the pooled out-of-fold scores) ===")
        for arm, o in by_bench[b].items():
            flag = "  [DEGENERATE: OOF pool separable -> one threshold for every r; oof-cost == oof-Youden]" if o["degenerate"] else ""
            print(f"  {arm:6s} OOF AUC {o['oof_auc']:.4f} AP {o['oof_ap']:.4f} (pool {o['n_pool']}, HS {o['n_pool_hs']}) | "
                  f"fold val AUC {[round(v, 3) for v in o['fold_val_auc']]} | distinct oof-cost thr over r grid: {o['n_distinct_thr']}{flag}")
            fv = [f["fold_val_youden"] for f in o["folds"]]
            fy = [f["oof_youden"] for f in o["folds"]]
            print(f"  {'':6s} fold-val-Youden thr {_ms(o['fold_val_thr_youden'])}  recall {_ms([m['recall'] for m in fv])}  FA {_msi([m['fp'] for m in fv])}  "
                  f"NEC@N/P {_ms([m['nec_r_test'] for m in fv])}")
            print(f"  {'':6s} OOF-Youden      thr {o['thr_youden']:.4f}           recall {_ms([m['recall'] for m in fy])}  FA {_msi([m['fp'] for m in fy])}  "
                  f"NEC@N/P {_ms([m['nec_r_test'] for m in fy])}")
            hdr = "".join(f"{('r=' + str(r)):>9s}" for r in rs) + f"{'r=N/P':>11s}"
            print(f"  {'':6s} {'rule':16s}{hdr}")
            for rule, key in (("oof-Youden", "oof_youden"), ("oof-cost", "oof_cost")):
                vals = [np.mean([f[key]["nec"][str(r)] if key == "oof_youden" else f[key]["at"][str(r)]["nec"] for f in o["folds"]]) for r in rs]
                print(f"  {'':6s} {rule:16s}" + "".join(f"{v:>9.4f}" for v in vals) + f"{np.mean([f[key]['nec_r_test'] for f in o['folds']]):>11.4f}")
            print(f"  {'':6s} {'oof-cost recall':16s}" + "".join(f"{np.mean([f['oof_cost']['at'][str(r)]['recall'] for f in o['folds']]):>9.4f}" for r in rs))
            print(f"  {'':6s} {'oof-cost FA':16s}" + "".join(f"{np.mean([f['oof_cost']['at'][str(r)]['fp'] for f in o['folds']]):>9.0f}" for r in rs))


CAPTIONS = {
    "roc": "ROC on the official test split, FPR on a log axis, one thin line per seed. Open squares mark the "
           "p = 0.5 operating point, filled circles the val-Youden threshold. For the pos_weight arm (ls0.1) the "
           "p = 0.5 point sits at FPR = 1 on b2/b4/b5: with pos_weight 30-104 the sigmoid is not a posterior and "
           "logit 0 flags every negative, so that square is not a valid operating point.",
    "nec": "Normalised expected cost NEC(r) = (r FN + FP)/(r P + N) against the cost ratio r for each threshold "
           "rule (thin = seeds, heavy = seed mean; dashed grey = r_test = N/P; dotted black = crossings of the "
           "seed-mean curves, labelled with the arm that is better above them). 'elkan-raw' applies t* = 1/(1+r) "
           "to the raw sigmoid and degenerates (everything positive) for r >~ 20 because label smoothing caps the "
           "outputs at 0.95; 'elkan-cal' applies it after the prior shift -log(pos_weight) and val-fitted "
           "temperature; 'val-cost' minimises NEC(r) on the val split; 'oracle' minimises it on the test set (lower bound).",
    "nec_oof": "Primary cost curve, b4: NEC(r) of the five fold models under the OOF-cost rule (threshold "
               "minimising NEC(r) on the pooled out-of-fold scores of the full training pool; thin = folds, heavy = "
               "mean), with the 80/20 val-cost rule of the five seed runs dashed for comparison. b1 and b5 are not "
               "shown: their OOF pools are perfectly separated (OOF AUC = 1.0 for both arms), so every r selects the "
               "same threshold in the separating gap, oof-cost coincides with oof-Youden and the curve is the "
               "fixed-count line (r FN + FP)/(r P + N) - five flat, identical curves per arm.",
    "reliability": "Reliability diagrams (15 equal-width bins on P(HS), seed 0) before and after calibration: "
                   "prior shift by -log(pos_weight) then temperature T fitted on the val logits. T marked * hit its "
                   "bound (val set perfectly separated: NLL has no interior minimum).",
}


def make_oof_figure(oof: dict, results: dict, summary: dict, out: Path, primary_bench: int = 4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"arm0": "#1f77b4", "ls0.1": "#ff7f0e"}
    lab = {"arm0": "arm0 (control)", "ls0.1": "ls0.1 (pos_weight)"}
    arms = [a for a in col if (a, primary_bench) in oof]
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for arm in arms:
        o = oof[(arm, primary_bench)]
        for f in o["folds"]:
            ax.plot(R_GRID, f["oof_cost"]["nec_grid"], color=col[arm], lw=0.7, alpha=0.35)
        ax.plot(R_GRID, o["oof_cost_grid_mean"], color=col[arm], lw=2.2, label=f"{lab[arm]}: OOF-cost (5 folds, OOF AUC {o['oof_auc']:.3f})")
        S = summary.get(primary_bench, {}).get("arms", {}).get(arm)
        if S:
            ax.plot(R_GRID, S["nec_grid_mean"]["val-cost"], color=col[arm], lw=1.4, ls="--", label=f"{lab[arm]}: val-cost (5 seed runs, 80/20)")
    ax.axvline(oof[(arms[0], primary_bench)]["r_test"], color="0.5", ls="--", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("cost ratio r = cost(FN) / cost(FP)"); ax.set_ylabel("NEC(r)")
    ax.set_title(f"b{primary_bench}: expected cost vs r, OOF-cost rule (dashed grey: r = N/P = {oof[(arms[0], primary_bench)]['r_test']:.0f})", fontsize=10)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7.5)
    degenerate = sorted({b for (a, b), o in oof.items() if o["degenerate"]})
    if degenerate:
        import textwrap
        note = (f"b{', b'.join(map(str, degenerate))} not shown: OOF pool perfectly separated (OOF AUC 1.0), so every r selects "
                "the same threshold and OOF-cost == OOF-Youden (flat fixed-count curves (r FN + FP)/(r P + N)).")
        fig.text(0.01, 0.01, textwrap.fill(note, 118), fontsize=7, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(out / f"nec_oof_b{primary_bench}.png", dpi=160); plt.close(fig)


def write_captions(out: Path, oof: dict):
    lines = ["# Figure captions", ""]
    for b in BENCHES:
        lines += [f"**roc_b{b}.png** — {CAPTIONS['roc']}", "", f"**nec_b{b}.png** — {CAPTIONS['nec']}", "",
                  f"**reliability_b{b}.png** — {CAPTIONS['reliability']}", ""]
    if oof:
        lines += [f"**nec_oof_b4.png** — {CAPTIONS['nec_oof']}", ""]
    (out / "captions.md").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------------------------
def make_figures(results: dict, summary: dict, out: Path):
    """ROC overlays with the 0.5 / val-Youden operating points, NEC-vs-r overlays with crossovers,
    and reliability diagrams before/after calibration. Colours: arm0 = blue, ls0.1 = orange,
    ref64 = grey; light lines = individual seeds, heavy = seed mean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    out.mkdir(parents=True, exist_ok=True)
    col = {"arm0": "#1f77b4", "ls0.1": "#ff7f0e", "ref64": "#7f7f7f"}
    lab = {"arm0": "arm0 (control)", "ls0.1": "ls0.1 (pos_weight)", "ref64": "ref64"}
    for b, S in summary.items():
        arms = [a for a in S["arms"] if a in col]
        if not arms:
            continue
        # ---- ROC ----
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        for arm in arms:
            seeds = S["arms"][arm]["seeds"]
            for i, s in enumerate(seeds):
                R = results[(arm, b, s)]
                run = M.load_run(R["run"])
                fpr, tpr, _ = roc_curve(run["labels"], run["probs"])
                ax.plot(np.maximum(fpr, 1e-5), tpr, color=col[arm], lw=0.9, alpha=0.55 if len(seeds) > 1 else 1.0,
                        label=f"{lab[arm]} (AUC {np.mean(S['arms'][arm]['auc']):.3f})" if i == 0 else None)
                for name, mk in (("0.5", "s"), ("val-Youden", "o")):
                    m = R["fixed"][name]
                    invalid = name == "0.5" and m["fp"] == m["N"]          # pos_weight arm: logit 0 flags every negative
                    ax.plot(max(m["fpr"], 1e-5), m["recall"], marker="x" if invalid else mk, ms=7 if invalid else 6,
                            mfc="white" if name == "0.5" else col[arm], mec=col[arm], ls="none",
                            label=(f"{name}" if (i == 0 and arm == arms[0]) else ("p=0.5 invalid (FA = N)" if invalid and i == 0 else None)))
        ax.set_xscale("log"); ax.set_xlim(1e-4, 1); ax.set_ylim(0, 1.02)
        ax.set_xlabel("false-positive rate (log)"); ax.set_ylabel("recall (hotspot accuracy)")
        ax.set_title(f"b{b}: ROC, {len(S['arms'][arms[0]]['seeds'])} seed(s); open = p 0.5, filled = val-Youden", fontsize=10)
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout(); fig.savefig(out / f"roc_b{b}.png", dpi=160); plt.close(fig)

        # ---- NEC vs r, one panel per rule ----
        rules = list(RULES)
        fig, axes = plt.subplots(1, len(rules), figsize=(3.4 * len(rules), 3.6), sharey=False)
        for ax, rule in zip(axes, rules):
            for arm in arms:
                e = S["arms"][arm]
                for s in e["seeds"]:
                    g = nec_grid_of(results[(arm, b, s)], rule)
                    ax.plot(R_GRID, g, color=col[arm], lw=0.7, alpha=0.35 if len(e["seeds"]) > 1 else 0.0)
                ax.plot(R_GRID, e["nec_grid_mean"][rule], color=col[arm], lw=2, label=lab[arm])
            C = S["arm0_vs_ls01"]
            if C:
                mc = C["crossover"][rule]["mean_curve"]
                if mc["kind"] == "cross":
                    for r_, above in mc["all"]:
                        ax.axvline(r_, color="k", ls=":", lw=0.8)
                    r_, above = mc["all"][0]
                    more = f" (+{len(mc['all']) - 1})" if len(mc["all"]) > 1 else ""
                    ax.text(r_, ax.get_ylim()[1] * 0.95, f" r*={r_:.3g}{more}\n {above} above", fontsize=7, va="top")
            ax.axvline(S["arms"][arms[0]]["r_test"], color="0.5", ls="--", lw=0.8)
            ax.set_xscale("log"); ax.set_xlabel("cost ratio r = cost(FN)/cost(FP)")
            ax.set_title(rule, fontsize=10); ax.grid(alpha=0.3, which="both")
        axes[0].set_ylabel("NEC(r)"); axes[0].legend(fontsize=8)
        fig.suptitle(f"b{b}: normalised expected cost vs r (dashed grey = r_test = N/P = {S['arms'][arms[0]]['r_test']:.0f})", fontsize=10)
        fig.tight_layout(); fig.savefig(out / f"nec_b{b}.png", dpi=160); plt.close(fig)

        # ---- reliability (seed 0, before / after) ----
        fig, axes = plt.subplots(1, len(arms), figsize=(3.6 * len(arms), 3.6), squeeze=False)
        for ax, arm in zip(axes[0], arms):
            s = S["arms"][arm]["seeds"][0]
            run = M.load_run(results[(arm, b, s)]["run"])
            if "val_logits" not in run or "logits" not in run:
                ax.set_axis_off(); continue
            cal = M.calibration_report(run["val_logits"], run["val_labels"], run["logits"], run["labels"],
                                       pos_weight=run.get("config", {}).get("pos_weight", 1.0))
            for key, c, name in (("reliability_before", "0.5", "raw"), ("reliability_after", col[arm], f"shift+T (T={cal['T']:.2f}{'*' if cal['at_bound'] else ''})")):
                rel = cal[key]
                mid = (rel["edges"][:-1] + rel["edges"][1:]) / 2
                ok = rel["count"] > 0
                ax.plot(rel["conf"][ok], rel["acc"][ok], "o-", color=c, ms=4, lw=1.2, label=f"{name}: ECE {rel['ece']:.3f}")
            ax.plot([0, 1], [0, 1], "k:", lw=0.8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("mean predicted P(HS)"); ax.set_ylabel("observed HS fraction")
            ax.set_title(f"b{b} {arm} seed{s}", fontsize=10); ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out / f"reliability_b{b}.png", dpi=160); plt.close(fig)
    print(f"figures written to {out}")


# ----------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs_gpu")
    ap.add_argument("--arms", nargs="+", default=list(ARMS_DEFAULT))
    ap.add_argument("--out", default="results")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--oof-tags", nargs="*", default=["oof_arm0:arm0", "oof_ls0.1:ls0.1"], help="'tag:arm' OOF run tags under --runs")
    ap.add_argument("--primary-bench", type=int, default=4, help="benchmark drawn in the OOF cost-curve figure")
    a = ap.parse_args(argv)
    runs = find_runs(Path(a.runs), a.arms)
    if not runs:
        raise SystemExit(f"no runs under {a.runs} for arms {a.arms}")
    results = {}
    for key, d in runs.items():
        results[key] = sweep_run(M.load_run(d))
    summary = aggregate(results, a.arms)
    oof = {key: sweep_oof(d) for key, d in find_oof(Path(a.runs), a.oof_tags).items()}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_summary(summary)
        print_oof(oof)
    print(buf.getvalue())
    write_outputs(results, summary, Path(a.out))
    (Path(a.out) / "sweep_summary.txt").write_text(buf.getvalue(), encoding="utf-8")
    (Path(a.out) / "oof_sweep.json").write_text(json.dumps({f"{arm}/b{b}": o for (arm, b), o in oof.items()}, default=_json_default))
    print(f"\nwrote {Path(a.out) / 'sweep_runs.csv'}, sweep_summary.json, nec_grid.json, sweep_summary.txt")
    if a.figures:
        make_figures(results, summary, Path(a.fig_dir))
        make_oof_figure(oof, results, summary, Path(a.fig_dir), a.primary_bench)
        write_captions(Path(a.fig_dir), oof)


if __name__ == "__main__":
    main()
