# Cost-sensitive hotspot detection on ICCAD-12: where the operating point actually comes from

Loss-level cost re-weighting (`pos_weight = N/P` in the BCE) does **not** improve a CNN lithography
hotspot detector on the ICCAD-12 benchmarks — it only relocates its operating point. Across five
seeds on b2/b4/b5 the weighted and unweighted arms rank the test clips the same (no significant
AUC difference; on b5 the weighted arm's average precision is significantly *worse*, 0.582 vs
0.725, all five seeds agreeing), no paired difference in normalised expected cost at r = N/P is
significant under any threshold rule that can be chosen without test labels, and no cost ratio
r ∈ [0.5, 1000] makes the weighted arm cheaper on all five benchmarks — on b4 the sign of the
difference itself depends on r. The reliable gain is in **threshold selection**: choosing the
threshold on pooled 5-fold out-of-fold scores of the full training pool, instead of on a single
20 % val split (5 hotspots on b5), cuts the recall spread across training runs to roughly a third
(std 0.04–0.05 → 0.01 on b4/b5), at the price of more false alarms on b4. A 7.8k-parameter
re-implementation of the reference paper's network matches the 1.17M-parameter CNN's AUC within
0.004 on b2–b4 and beats it on b1 and b5 (at lower AP on b2–b4). The paper is at [`paper/paper.pdf`](paper/paper.pdf).

## Data

The benchmark is the **ICCAD 2012 CAD Contest, Problem C (fuzzy pattern matching for physical
verification)** suite of five layout-clip sets (b1–b5), 1200×1200 binary PNG clips labelled
hotspot / non-hotspot with fixed train and test splits (J. A. Torres, *ICCAD-2012 CAD contest in
fuzzy pattern matching for physical verification and benchmark suite*, ICCAD 2012). It is
redistributed under the contest organiser's terms, so **it is not part of this repository** —
obtain it from the organisers / the benchmark-suite paper and extract it to `iccad_official/`
(any layout works: `src/data.py` discovers `iccad<N>` / `train|test` / `hs|nhs` directories and
prints what it inferred).

Preprocess to the 64×64 coverage maps and the per-seed 80/20 splits used here:

```bash
python -m src.data --root iccad_official --inspect-only                      # check the discovered layout
python -m src.data --root iccad_official --size 64 --accept-count b1_test_nhs=4679
```

`--accept-count` records a known property of the public copy: b1 has 4,679 test non-hotspots
where the published statistic says 3,869; every other count matches and any other deviation
fails loudly. This writes `data/b{1..5}_{train,test}_{X,y}.npy`, `data/splits/` and
`data/meta.json` (all git-ignored).

## Reproduce

Training was run on a Colab Tesla T4 (`torch 2.11.0+cu128`); everything downstream of the saved
scores runs on CPU. Install with the pins in `requirements.txt`.

```bash
# 1. the two CNN arms (label smoothing 0.1 in both; the only difference is the loss weighting)
python -m src.train --bench 1 3 --seed 0           --out runs_gpu --tag arm0  --imbalance none
python -m src.train --bench 2 4 5 --seed 0 1 2 3 4 --out runs_gpu --tag arm0  --imbalance none
python -m src.train --bench 1 3 --seed 0           --out runs_gpu --tag ls0.1                     # pos_weight = N/P (default)
python -m src.train --bench 2 4 5 --seed 0 1 2 3 4 --out runs_gpu --tag ls0.1

# 2. the reference-paper network on the same 1x64x64 input, its own protocol
python -m src.train --bench 1 2 3 4 5 --seed 0 --out runs_gpu --tag ref64 --model ref64 \
    --optimizer nadam --sched const --select last --epochs 10 --no-augment --imbalance none \
    --label-smoothing 0 --weight-decay 0 --clip 0 --patience 0 --batch-size 32

# 3. out-of-fold threshold selection (5-fold stratified CV over the full training pool)
python -m src.oof --bench 1 4 5 --seed 0 --out runs_gpu --tag oof_arm0  --imbalance none
python -m src.oof --bench 1 4 5 --seed 0 --out runs_gpu --tag oof_ls0.1

# 4. everything reported, from the saved scores only (no training, no model loading)
python make_figs.py                 # results/*.csv|json|txt and figures/*.png
python -m pytest tests -q           # 17 tests: confusion matrix / NEC against recorded runs
```

Each run directory holds `config.json`, `history.json`, `best.pt`, `test.json`, and the raw
scores `probs.npy` / `logits.npy` / `labels.npy` / `val_logits.npy` / `val_labels.npy` that every
analysis script reads. `python -m src.evaluate --val` back-fills val logits for runs made before
they were saved, and `--refresh-stale` re-scores a checkpoint and overwrites score files only if
the result reproduces that run's `test.json`. `python -m src.metrics <run>` prints all Task-5
numbers for one run (`--vs <run>` adds an exact McNemar test).

## Layout

| path | contents |
|---|---|
| `src/data.py` | dataset inspection, count verification against the published statistics, 64×64 preprocessing, stratified splits |
| `src/model.py`, `src/ref_model.py` | the CNN (1.17 M params) and the reference network re-derived from its published parameter count (see `paper/NOTES.md`) |
| `src/train.py` | one run per (benchmark, seed): 80/20 fit/val split, best-val-AUC checkpoint, label smoothing, dihedral augmentation, optional `pos_weight` / oversampling |
| `src/oof.py` | 5-fold out-of-fold scores over the full training pool and the thresholds chosen on them |
| `src/metrics.py` | offline metrics on saved scores: confusion matrix and rates, NEC(r), Elkan threshold and implied cost ratio, partial AUC at recall ≥ 0.95, prior shift + temperature scaling with ECE / reliability, exact McNemar |
| `src/sweep.py` | cost-ratio sweep r ∈ {1…200} under five threshold rules, per-benchmark paired tests, crossovers, all figures |
| `src/tables.py` | the results tables (per benchmark + macro average, threshold rule per row, threshold-stability section) |
| `src/evaluate.py`, `src/scores.py`, `src/device.py` | score back-fill / verification, score-file I/O, device + seeding |
| `tests/` | unit tests for the confusion-matrix and NEC functions against hand-computed values from recorded runs, plus synthetic checks |
| `results/` | `results.md` / `results.csv` (all tables), `sweep_runs.csv` (one row per run × rule × r), `sweep_summary.txt` / `.json`, `oof_sweep.json`, `nec_grid.json` |
| `figures/` | ROC, NEC-vs-r and reliability diagrams per benchmark, the b4 OOF cost curve; `captions.md` |
| `paper/` | the paper and slides; `NOTES.md` records the reference-network derivation, the abandoned 150 px arm, and every methodological caveat |

## Headline results

Official test split, threshold chosen on the run's val split by Youden's J (mean ± std over five
seeds where available). Recall is the ICCAD "accuracy"; false alarms are counts. Full tables
with all threshold rules (p = 0.5, val-Youden, val-cost at r = N/P, OOF-Youden, OOF-cost) are in
[`results/results.md`](results/results.md).

| bench | arm | seeds | AUC | AP | recall | false alarms |
|---|---|---|---|---|---|---|
| b1 | arm0 (control) | 1 | 0.917 | 0.364 | 0.996 | 1790 |
| b1 | ls0.1 (pos_weight) | 1 | 0.913 | 0.346 | 0.991 | 1790 |
| b1 | ref64 | 1 | 0.982 | 0.779 | 0.996 | 976 |
| b2 | arm0 | 5 | 0.997 ± 0.001 | 0.979 ± 0.027 | 0.988 ± 0.003 | 111 ± 159 |
| b2 | ls0.1 | 5 | 0.998 ± 0.001 | 0.994 ± 0.002 | 0.989 ± 0.004 | 80 ± 111 |
| b2 | ref64 | 1 | 0.996 | 0.834 | 0.962 | 553 |
| b3 | arm0 | 1 | 0.993 | 0.908 | 0.978 | 1488 |
| b3 | ls0.1 | 1 | 0.989 | 0.886 | 0.978 | 1454 |
| b3 | ref64 | 1 | 0.989 | 0.785 | 0.986 | 3821 |
| b4 | arm0 | 5 | 0.989 ± 0.006 | 0.851 ± 0.045 | 0.957 ± 0.017 | 670 ± 344 |
| b4 | ls0.1 | 5 | 0.993 ± 0.007 | 0.859 ± 0.046 | 0.944 ± 0.022 | 414 ± 171 |
| b4 | ref64 | 1 | 0.989 | 0.641 | 0.904 | 660 |
| b5 | arm0 | 5 | 0.982 ± 0.003 | 0.725 ± 0.042 | 0.941 ± 0.037 | 61 ± 28 |
| b5 | ls0.1 | 5 | 0.985 ± 0.005 | 0.582 ± 0.031 | 0.937 ± 0.033 | 107 ± 15 |
| b5 | ref64 | 1 | 0.989 | 0.746 | 0.756 | 27 |
| avg | arm0 | – | 0.976 | 0.765 | 0.972 | 824 |
| avg | ls0.1 | – | 0.976 | 0.733 | 0.968 | 769 |
| avg | ref64 | – | 0.989 | 0.757 | 0.921 | 1207 |

Threshold stability (five fold models of the OOF run, each scored on the test split with its own
fold's val-Youden threshold vs the one threshold chosen on the pooled out-of-fold scores):

| bench | arm | fold-val-Youden recall / FA | OOF-Youden recall / FA |
|---|---|---|---|
| b4 | arm0 | 0.937 ± 0.041 / 616 ± 352 | 0.965 ± 0.014 / 1840 ± 1547 |
| b4 | ls0.1 | 0.934 ± 0.050 / 542 ± 309 | 0.958 ± 0.010 / 1229 ± 1040 |
| b5 | arm0 | 0.932 ± 0.036 / 53 ± 13 | 0.971 ± 0.010 / 77 ± 12 |
| b5 | ls0.1 | 0.917 ± 0.037 / 101 ± 43 | 0.966 ± 0.012 / 165 ± 37 |

Two things the tables make explicit rather than hide: the p = 0.5 rows of the pos_weight arm are
starred because with w = 30–104 the weighted model's sigmoid is not a posterior and logit 0
flags every negative (FA = N exactly on b2/b4/b5); and the raw Elkan threshold 1/(1+r) is
meaningless for label-smoothed models (outputs are capped at 0.95), so cost-sensitive thresholds
are chosen either on calibrated probabilities (prior shift + temperature fitted on val) or by
minimising the expected cost directly on val / OOF scores.
