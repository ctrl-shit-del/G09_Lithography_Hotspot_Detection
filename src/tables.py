"""Task 7 - report tables from saved scores: per benchmark plus a macro-average row, one block per
(arm, threshold rule).

    python -m src.tables --runs runs_gpu                       # results/results.md + results.csv
    python -m src.tables --runs runs_gpu --arms arm0 ls0.1     # subset of arms
    python -m src.tables --runs runs_gpu --oof-tags oof_arm0:arm0 oof_ls0.1:ls0.1

Columns: TP / FP / FN / TN, balanced accuracy, precision, recall (= hotspot 'accuracy'),
specificity, F1, ROC-AUC, AP, false alarms (= FP), inference time (ms per clip and total s on
the device recorded in eval.json), parameter count (torch trainable params; Keras-convention
count in parentheses when it differs), and the threshold rule of the row:

    p=0.5                sigmoid(logit) >= 0.5, i.e. logit >= 0
    val-Youden           threshold maximising TPR - FPR on the run's 20 % val split
    val-cost r=N/P       threshold minimising NEC(r) on the val split at r = N_test / P_test
                         (the cost ratio at which Youden and expected cost agree; needs the
                         backfilled val_logits.npy)
    OOF-Youden           threshold maximising TPR - FPR on the pooled 5-fold out-of-fold scores of
                         the full training pool (src/oof.py); metrics are the mean over the five
                         fold models on the official test split
    OOF-cost r=N/P       NEC-minimising threshold on the OOF scores at r = N_test / P_test

p = 0.5 rows of a pos_weight arm are kept but starred and footnoted: with pos_weight w = 30-104 the
weighted model's sigmoid is q = w p / (w p + 1 - p), not the posterior p, and logit 0 (q = 0.5,
i.e. p = 1/(1+w)) flags every negative on b2/b4/b5 (FA = N exactly). It is evidence that the
weighted sigmoid is not a posterior, not an operating point.

A threshold-stability section compares, per benchmark and arm, the five fold models' own
val-Youden thresholds (each chosen on one held-out fold) with the single OOF-Youden threshold
(chosen on the pooled out-of-fold scores), reporting test recall / false alarms mean ± std
across the five fold models (numbers as recorded by src/oof.py on the training device).

Multi-seed cells are mean ± std (ddof = 0) over seeds. The macro-average row averages the
per-benchmark means (counts included, so "false alarms" in that row is a mean count, the way
the ICCAD-12 papers average). Benchmarks without a run for an arm are printed as "not run";
the average row then says which benchmarks it covers.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from . import metrics as M

BENCHES = (1, 2, 3, 4, 5)
ARMS = ("arm0", "ls0.1", "ref64")
ARM_DESC = {"arm0": "CNN, label smoothing 0.1, no re-weighting (control)",
            "ls0.1": "CNN, label smoothing 0.1, pos_weight = N/P (cost arm)",
            "ref64": "reference-paper net (12-ch, 2 blocks) at 1x64x64"}
COLS = ["tp", "fp", "fn", "tn", "balanced_accuracy", "precision", "recall", "specificity", "f1", "auc", "ap",
        "false_alarms", "infer_ms_per_clip", "infer_s", "params"]
HDR = ["TP", "FP", "FN", "TN", "bal. acc", "precision", "recall", "specificity", "F1", "AUC", "AP",
       "false alarms", "infer ms/clip", "infer s", "params"]
INT_COLS = {"tp", "fp", "fn", "tn", "false_alarms", "params"}


# ----------------------------------------------------------------------------------------------
def row_from_run(run: dict, thr: float, label: str, r: float = float("nan")) -> dict:
    y, p = run["labels"], run["probs"]
    m = M.at_threshold(y, p, thr)
    m.update(M.ranking_metrics(y, p))
    ev, cfg = run.get("eval", {}), run.get("config", {})
    m["infer_s"] = ev.get("inference_s", float("nan"))
    m["infer_ms_per_clip"] = ev.get("inference_ms_per_clip", float("nan"))
    m["params"] = cfg.get("params", float("nan"))
    m["params_keras"] = cfg.get("params_keras", float("nan"))
    m["pos_weight"] = float(cfg.get("pos_weight", 1.0))
    m["N"] = int((y == 0).sum())
    m["device"] = ev.get("device", "?")
    m["rule"] = label
    m["thr_prob"] = float(thr)
    m["r"] = r
    return m


def rows_for_arm(root: Path, arm: str) -> dict:
    """{bench: {rule_label: [row per seed]}} for one arm."""
    out = {}
    for b in BENCHES:
        dirs = sorted((root / arm / f"b{b}").glob("seed*")) if (root / arm / f"b{b}").exists() else []
        dirs = [d for d in dirs if (d / "probs.npy").exists()]
        if not dirs:
            continue
        per_rule = {}
        for d in dirs:
            run = M.load_run(d)
            r_test = M.implied_r(run["labels"])
            rules = {"p=0.5": 0.5}
            if "thr_youden" in run:
                rules["val-Youden"] = run["thr_youden"]
            if "val_logits" in run:
                _, t = M.min_nec(run["val_labels"], M.sigmoid(run["val_logits"]), r_test)
                rules["val-cost r=N/P"] = t
            for label, thr in rules.items():
                per_rule.setdefault(label, []).append(row_from_run(run, thr, label, r_test if "cost" in label else float("nan")))
        out[b] = per_rule
    return out


def rows_for_oof(root: Path, tag: str) -> dict:
    """{bench: {rule_label: [row per fold model]}} from src/oof.py outputs (seed 0 by default)."""
    out = {}
    for b in BENCHES:
        seeds = sorted((root / tag / f"b{b}").glob("seed*/oof.json")) if (root / tag / f"b{b}").exists() else []
        if not seeds:
            continue
        per_rule = {}
        for oj in seeds:
            rec = json.loads(oj.read_text())
            r_test = rec["r_test"]
            fold_runs = [M.load_run(oj.parent / f"fold{k}") for k in range(rec["n_folds"])]
            rules = {"OOF-Youden": M.sigmoid(rec["oof_thr_youden"]),
                     "OOF-cost r=N/P": M.sigmoid(rec["oof_cost_thr"]["N/P"]["thr"])}
            for label, thr in rules.items():
                for fr in fold_runs:
                    per_rule.setdefault(label, []).append(row_from_run(fr, thr, label, r_test if "cost" in label else float("nan")))
        out[b] = per_rule
    return out


# ----------------------------------------------------------------------------------------------
def _agg(rows: list, col: str):
    v = np.asarray([r[col] for r in rows], float)
    return float(np.nanmean(v)), (float(np.nanstd(v, ddof=0)) if len(v) > 1 else float("nan")), len(v)


def _cell(mean, std, n, col):
    if np.isnan(mean):
        return "n/a"
    if col in INT_COLS:
        return f"{mean:.0f}" if n == 1 else f"{mean:.0f} ± {std:.0f}"
    if col in ("infer_s",):
        return f"{mean:.2f}" if n == 1 else f"{mean:.2f} ± {std:.2f}"
    if col in ("infer_ms_per_clip",):
        return f"{mean:.3f}" if n == 1 else f"{mean:.3f} ± {std:.3f}"
    return f"{mean:.4f}" if n == 1 else f"{mean:.4f} ± {std:.4f}"


def block_table(arm: str, per_bench: dict, rule_labels: list, title: str) -> tuple[str, list]:
    """Markdown table for one arm: rows = (bench, rule) + avg per rule. Returns (markdown, flat rows)."""
    lines = [f"### {title}", "",
             "| bench | seeds | rule | thr (prob) | r | " + " | ".join(HDR) + " |",
             "|" + "---|" * (len(HDR) + 5)]
    flat, starred = [], False
    for label in rule_labels:
        covered, means = [], []
        for b in BENCHES:
            rows = per_bench.get(b, {}).get(label)
            if not rows:
                lines.append(f"| b{b} | – | {label} | – | – | " + " | ".join(["not run"] + ["–"] * (len(HDR) - 1)) + " |")
                continue
            cells, rec = [], dict(arm=arm, bench=f"b{b}", n=len(rows), rule=label, thr_prob=float(np.mean([r["thr_prob"] for r in rows])),
                                  r=float(rows[0]["r"]))
            for col in COLS:
                mean, std, n = _agg(rows, col)
                pk = rows[0].get("params_keras", float("nan"))
                if col == "params":
                    keras = f" ({int(pk):,} Keras)" if not np.isnan(pk) and pk != rows[0]["params"] else ""
                    cells.append(f"{int(rows[0]['params']):,}{keras}")
                else:
                    cells.append(_cell(mean, std, n, col))
                rec[col] = mean
                rec[col + "_std"] = std
            thr_s = _cell(rec["thr_prob"], float(np.std([r["thr_prob"] for r in rows])), len(rows), "thr")
            r_s = "–" if np.isnan(rec["r"]) else f"{rec['r']:.1f}"
            name = label
            if label == "p=0.5" and rows[0]["pos_weight"] != 1.0:
                starred = True
                all_neg = all(r["fp"] == r["N"] for r in rows)
                name = f"p=0.5 \*{' (FA = N)' if all_neg else ''}"
                rec["invalid"] = True
            lines.append(f"| b{b} | {len(rows)} | {name} | {thr_s} | {r_s} | " + " | ".join(cells) + " |")
            covered.append(b)
            means.append(rec)
            flat.append(rec)
        if means:
            rec = dict(arm=arm, bench="avg", n=len(means), rule=label, thr_prob=float("nan"), r=float("nan"))
            cells = []
            for col in COLS:
                v = np.asarray([m[col] for m in means], float)
                rec[col], rec[col + "_std"] = float(np.nanmean(v)), float("nan")
                cells.append("–" if col == "params" else _cell(rec[col], float("nan"), 1, col))
            cover = "" if len(covered) == len(BENCHES) else f" (b{', b'.join(map(str, covered))} only)"
            name = f"{label} \*" if (label == "p=0.5" and any(m.get("invalid") for m in means)) else label
            lines.append(f"| **avg{cover}** | – | {name} | – | – | " + " | ".join(cells) + " |")
            flat.append(rec)
    if starred:
        w = sorted({round(r["pos_weight"], 1) for rows in per_bench.values() for r in rows.get("p=0.5", [])})
        lines += ["", f"\* Not a valid operating point. This arm is trained with pos_weight w = {w[0]:g}–{w[-1]:g} (N/P of the fit "
                  "split), so its sigmoid estimates q = w·p/(w·p + 1 − p) rather than the posterior p; logit 0 corresponds to "
                  "p = 1/(1 + w) and on b2/b4/b5 (w ≥ 30) flags every negative — FA = N exactly. The row is kept only as "
                  "evidence that the weighted model's sigmoid is not a posterior; its metrics are excluded from any conclusion."]
    return "\n".join(lines), flat


def stability_table(root: Path, oof_tags: list) -> str:
    """fold-val-Youden vs OOF-Youden: test recall / FA / bal.acc mean ± std over the five fold models,
    from the numbers src/oof.py recorded (oof.json test_per_fold)."""
    rows = []
    for spec in oof_tags:
        tag, _, arm = spec.partition(":")
        arm = arm or tag
        for oj in sorted((root / tag).glob("b*/seed*/oof.json")):
            rec = json.loads(oj.read_text())
            b = rec["bench"]
            for rule, thr_list in (("fold-val-Youden", [M.sigmoid(t) for t in rec["fold_val_thr_youden"]]),
                                   ("OOF-Youden", [M.sigmoid(rec["oof_thr_youden"])])):
                pf = rec["test_per_fold"][rule]
                g = lambda k: np.asarray([m[k] for m in pf], float)
                rows.append(dict(bench=b, arm=arm, oof_auc=rec["oof_auc"], n_pool_hs=rec["n_pool_hs"],
                                 fold_val_auc=[f["val_auc"] for f in rec["folds"]], rule=rule,
                                 thr_mean=float(np.mean(thr_list)), thr_std=float(np.std(thr_list)) if len(thr_list) > 1 else float("nan"),
                                 recall=g("recall"), fa=g("fp"), bal=g("balanced_accuracy"), f1=g("f1")))
    if not rows:
        return ""
    out = ["### Threshold stability: per-fold val-Youden vs pooled OOF-Youden (5 fold models on the official test split)", "",
           "| bench | arm | OOF AUC (pool HS) | fold val AUC | rule | threshold (prob) | test recall | false alarms | bal. acc | F1 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    ms = lambda v, fmt: f"{v.mean():{fmt}} ± {v.std(ddof=0):{fmt}}"
    for r in sorted(rows, key=lambda r: (r["bench"], r["arm"], r["rule"] != "fold-val-Youden")):
        thr = f"{r['thr_mean']:.4f}" + ("" if np.isnan(r["thr_std"]) else f" ± {r['thr_std']:.4f}")
        fva = "[" + ", ".join(f"{v:.3f}" for v in r["fold_val_auc"]) + "]"
        out.append(f"| b{r['bench']} | {r['arm']} | {r['oof_auc']:.4f} ({r['n_pool_hs']}) | {fva} | {r['rule']} | {thr} | "
                   f"{ms(r['recall'], '.3f')} | {ms(r['fa'], '.0f')} | {ms(r['bal'], '.4f')} | {ms(r['f1'], '.3f')} |")
    out += ["", "Each fold model is scored on the official test split with (a) the Youden threshold of its own held-out fold "
            "(the threshold a single 80/20 run would have used) and (b) the one Youden threshold chosen on all five folds' "
            "out-of-fold scores. On b1 and b5 the OOF pool is perfectly separated (OOF AUC 1.0), so the OOF threshold is a "
            "point inside the separating gap chosen with all 99 / 26 training hotspots rather than 20 / 5; the recall "
            "spread across fold models collapses while false alarms move with the (lower) threshold. On b4 the OOF "
            "threshold sits well below the per-fold ones and the false-alarm spread across fold models widens sharply.", ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs_gpu")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--oof-tags", nargs="*", default=["oof_arm0:arm0", "oof_ls0.1:ls0.1"],
                    help="OOF run tag under --runs, optionally ':arm' to attach its rows to that arm's block")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    root, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    md = ["# Results tables", "",
          f"Source: `{root}` (probs.npy / labels.npy / val_logits.npy / test.json / eval.json / config.json). "
          "Cells are mean ± std over seeds where more than one seed exists. Recall is the ICCAD 'accuracy'; "
          "false alarms = FP. Inference time is on the device the run recorded in eval.json (T4 for runs_gpu). "
          "'not run' marks (arm, benchmark) combinations without a run; OOF rows appear once src/oof.py has been run.", ""]
    flat_all, missing = [], []
    for arm in a.arms:
        per_bench = rows_for_arm(root, arm)
        oof_tag = next((t.split(":")[0] for t in a.oof_tags if t.endswith(f":{arm}")), None)
        oof_rows = rows_for_oof(root, oof_tag) if oof_tag else {}
        if oof_tag and not oof_rows:
            missing.append(f"{arm}: no OOF runs under {root / oof_tag} (run `python -m src.oof --tag {oof_tag} ...` on the GPU)")
        for b, rules in oof_rows.items():
            per_bench.setdefault(b, {}).update(rules)
        labels = []
        for b in BENCHES:
            for label in per_bench.get(b, {}):
                if label not in labels:
                    labels.append(label)
        # order: p=0.5, val-Youden, val-cost, OOF-Youden, OOF-cost
        key = lambda l: (0 if l == "p=0.5" else 1 if l == "val-Youden" else 2 if l.startswith("val-cost") else 3 if l == "OOF-Youden" else 4, l)
        labels.sort(key=key)
        if not per_bench:
            missing.append(f"{arm}: no runs under {root / arm}")
            continue
        table, flat = block_table(arm, per_bench, labels, f"{arm} — {ARM_DESC.get(arm, '')}")
        md += [table, ""]
        flat_all += flat
        absent = [b for b in BENCHES if b not in per_bench]
        if absent:
            missing.append(f"{arm}: not run on b{', b'.join(map(str, absent))}")
    stab = stability_table(root, a.oof_tags)
    if stab:
        md += [stab, ""]
    if missing:
        md += ["### Missing", ""] + [f"- {m}" for m in missing] + [""]
    text = "\n".join(md)
    (out / "results.md").write_text(text, encoding="utf-8")
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["arm", "bench", "n", "rule", "thr_prob", "r"] + [c for col in COLS for c in (col, col + "_std")]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(flat_all)
    print(text)
    print(f"\nwrote {out / 'results.md'} and results.csv")


if __name__ == "__main__":
    main()
