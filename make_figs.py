"""Regenerate every reported artefact from the saved run scores (no training, no model loading
except in the optional --refresh step).

    python make_figs.py                     # results/*.csv|json|txt + figures/*.png from runs_gpu/
    python make_figs.py --runs runs_gpu     # explicit run root
    python make_figs.py --no-figures        # tables and sweep numbers only

Equivalent to:
    python -m src.sweep  --runs runs_gpu --figures --out results --fig-dir figures
    python -m src.tables --runs runs_gpu --out results
Requires the run directories (probs.npy / labels.npy / val_logits.npy / test.json per run), which
are not part of the repository; see README.md for how they are produced.
"""
from __future__ import annotations

import argparse

from src import sweep, tables


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs_gpu")
    ap.add_argument("--results", default="results")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    sweep_args = ["--runs", a.runs, "--out", a.results, "--fig-dir", a.figures] + ([] if a.no_figures else ["--figures"])
    sweep.main(sweep_args)
    tables.main(["--runs", a.runs, "--out", a.results])


if __name__ == "__main__":
    main()
