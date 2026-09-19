"""Phase 4 - aggregate runs/<tag>/b{B}/seed{S}/test.json into a results table.

    python -m src.report --tag ls0.1                # markdown table to stdout
    python -m src.report --tag ls0.1 --csv out.csv  # also write a flat CSV (one row per run)

With several seeds per benchmark, mean +/- std over seeds is reported; with one seed the
raw numbers are shown. `--op` picks which test operating point to tabulate:
    youden  threshold maximising TPR-FPR on the val split  (default)
    fpr     threshold at the val FPR budget (--target-fpr at train time)
    0p5     logit 0, i.e. probability 0.5
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

OPS = {"youden": "at_val_youden", "fpr": "at_val_fpr", "0p5": "at_0p5"}
COLS = ["auc", "ap", "accuracy", "false_alarms", "fpr", "precision", "f1"]


def load_runs(root: Path):
    rows = []
    for tj in sorted(root.glob("b*/seed*/test.json")):
        t = json.loads(tj.read_text())
        c = json.loads((tj.parent / "config.json").read_text())
        rows.append(dict(bench=c["bench"], seed=c["seed"], best_epoch=t["best_epoch"], epochs_run=t["epochs_run"],
                         val_auc=t["val_auc"], train_time_s=t["train_time_s"], n_test_hs=t["at_0p5"]["n_hs"],
                         n_test_nhs=t["at_0p5"]["n_nhs"], **{f"{op}_{k}": t[key][k] for op, key in OPS.items() for k in COLS}))
    return rows


def _ms(vals):
    v = np.asarray(vals, float)
    return (f"{v.mean():.4f}" if len(v) == 1 else f"{v.mean():.4f} ± {v.std(ddof=0):.4f}")


def _msi(vals):
    v = np.asarray(vals, float)
    return (f"{int(v[0])}" if len(v) == 1 else f"{v.mean():.0f} ± {v.std(ddof=0):.0f}")


def markdown(rows, op: str) -> str:
    key = op
    benches = sorted({r["bench"] for r in rows})
    out = [f"| bench | seeds | test HS / NHS | best ep | val AUC | test AUC | test AP | accuracy (HS recall) | false alarms | FPR | precision |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in benches:
        rs = [r for r in rows if r["bench"] == b]
        g = lambda k: [r[k] for r in rs]
        out.append(f"| b{b} | {len(rs)} | {rs[0]['n_test_hs']} / {rs[0]['n_test_nhs']} | {_msi(g('best_epoch'))} | "
                   f"{_ms(g('val_auc'))} | {_ms(g(f'{key}_auc'))} | {_ms(g(f'{key}_ap'))} | {_ms(g(f'{key}_accuracy'))} | "
                   f"{_msi(g(f'{key}_false_alarms'))} | {_ms(g(f'{key}_fpr'))} | {_ms(g(f'{key}_precision'))} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--op", choices=list(OPS), default="youden")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args(argv)
    rows = load_runs(Path(a.runs) / a.tag)
    if not rows:
        raise SystemExit(f"no test.json found under {Path(a.runs) / a.tag}")
    print(f"## {a.tag}  (operating point: {a.op})\n")
    print(markdown(rows, a.op))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
