# Where Should Cost Live?

**Cost-sensitive lithography hotspot detection on the ICCAD-12 benchmarks — an injection-site study.**

Loss-level cost re-weighting does not improve a CNN hotspot detector. It only relocates the
detector's operating point on the same ROC curve. The reliable gain is in how the decision
threshold is *chosen*.

📄 [Paper](paper/paper.pdf) · 🖼 [Slides](paper/slides.pdf) · 📊 [Full result tables](results/results.md)

---

## Findings

Across five seeds on b2/b4/b5, with architecture, splits, schedule, seeds and device held fixed
and only the cost mechanism varying:

- **Ranking is unchanged.** No significant AUC difference between the weighted
  (`pos_weight = N/P`) and unweighted arms. On b5 the weighted arm is significantly *worse* —
  AP 0.582 vs 0.725, all five seeds agreeing (*t*-test *p* < 0.001).
- **Expected cost is unchanged.** No paired difference in normalised expected cost at
  `r = N/P` is significant under any threshold rule selectable without test labels.
- **No cost ratio makes weighting win outright.** Over `r ∈ [0.5, 1000]` the weighted arm is
  not cheaper on all five benchmarks; on b4 the *sign* of the difference flips with `r`.
- **Threshold selection is where the gain is.** Choosing the threshold on pooled 5-fold
  out-of-fold scores, rather than on a single 20 % val split (5 hotspots on b5), cuts the recall
  spread across training runs to roughly a third — std 0.04–0.05 → 0.01 on b4/b5 — at the price
  of more false alarms on b4.
- **Capacity is not the bottleneck.** A 7.8k-parameter re-implementation of the reference
  network matches the 1.17M-parameter CNN's AUC within 0.004 on b2–b4 and beats it on b1 and b5,
  at lower AP on b2–b4.

---

## Data

The benchmark is the **ICCAD 2012 CAD Contest, Problem C** (fuzzy pattern matching for physical
verification): five sets of 1200×1200 binary layout clips (b1–b5) labelled hotspot / non-hotspot,
with fixed train and test splits.¹

It is redistributed under the contest organiser's terms, so **it is not part of this repository**.
Obtain it from the organisers and extract to `iccad_official/`. Any directory layout works —
`src/data.py` discovers `iccad<N>/{train,test}/{hs,nhs}` and prints what it inferred.

```bash
python -m src.data --root iccad_official --inspect-only          # check the discovered layout
python -m src.data --root iccad_official --size 64 --accept-count b1_test_nhs=4679
```

`--accept-count` records a known property of the public copy: b1 has **4,679** test
non-hotspots where the published statistic says 3,869. Every other count matches, and any other
deviation fails loudly rather than passing silently.

Writes `data/b{1..5}_{train,test}_{X,y}.npy`, `data/splits/` and `data/meta.json` — all
git-ignored.

---

## Reproduce

Training ran on a Colab Tesla T4 (torch 2.11.0+cu128). Everything downstream of the saved scores
runs on CPU. Install with the pins in [`requirements.txt`](requirements.txt).

**1 — the two CNN arms.** Label smoothing 0.1 in both; the only difference is the loss weighting.

```bash
python -m src.train --bench 1 3     --seed 0         --out runs_gpu --tag arm0  --imbalance none
python -m src.train --bench 2 4 5   --seed 0 1 2 3 4 --out runs_gpu --tag arm0  --imbalance none
python -m src.train --bench 1 3     --seed 0         --out runs_gpu --tag ls0.1
python -m src.train --bench 2 4 5   --seed 0 1 2 3 4 --out runs_gpu --tag ls0.1
```

**2 — the reference-paper network**, same 1×64×64 input, its own protocol.

```bash
python -m src.train --bench 1 2 3 4 5 --seed 0 --out runs_gpu --tag ref64 --model ref64 \
    --optimizer nadam --sched const --select last --epochs 10 --no-augment --imbalance none \
    --label-smoothing 0 --weight-decay 0 --clip 0 --patience 0 --batch-size 32
```

**3 — out-of-fold threshold selection**, 5-fold stratified CV over the full training pool.

```bash
python -m src.oof --bench 1 4 5 --seed 0 --out runs_gpu --tag oof_arm0  --imbalance none
python -m src.oof --bench 1 4 5 --seed 0 --out runs_gpu --tag oof_ls0.1
```

**4 — every reported number**, from the saved scores only — no training, no model loading.

```bash
python make_figs.py          # results/*.csv|json|txt and figures/*.png
python -m pytest tests -q    # 17 tests: confusion matrix / NEC against recorded runs
```

Each run directory holds `config.json`, `history.json`, `best.pt`, `test.json`, and the raw
scores `probs.npy` / `logits.npy` / `labels.npy` / `val_logits.npy` / `val_labels.npy` that every
analysis script reads.

| Utility | What it does |
|---|---|
| `python -m src.evaluate --val` | back-fills val logits for runs made before they were saved |
| `python -m src.evaluate --refresh-stale` | re-scores a checkpoint, overwriting score files **only** if the result reproduces that run's `test.json` |
| `python -m src.metrics <run>` | prints every metric for one run; `--vs <run>` adds an exact McNemar test |

---

## Layout

| Path | Contents |
|---|---|
| `src/data.py` | dataset inspection, count verification against the published statistics, 64×64 preprocessing, stratified splits |
| `src/model.py`, `src/ref_model.py` | the CNN (1.17M params) and the reference network re-derived from its published parameter count (see [`paper/NOTES.md`](paper/NOTES.md)) |
| `src/train.py` | one run per (benchmark, seed): 80/20 fit/val split, best-val-AUC checkpoint, label smoothing, dihedral augmentation, optional `pos_weight` / oversampling |
| `src/oof.py` | 5-fold out-of-fold scores over the full training pool, and the thresholds chosen on them |
| `src/metrics.py` | offline metrics on saved scores: confusion matrix and rates, NEC(r), Elkan threshold and implied cost ratio, partial AUC at recall ≥ 0.95, prior shift + temperature scaling with ECE / reliability, exact McNemar |
| `src/sweep.py` | cost-ratio sweep `r ∈ {1…200}` under five threshold rules, per-benchmark paired tests, crossovers, all figures |
| `src/tables.py` | result tables: per benchmark + macro average, threshold rule per row, threshold-stability section |
| `src/evaluate.py`, `src/scores.py`, `src/device.py` | score back-fill and verification, score-file I/O, device and seeding |
| `tests/` | unit tests for the confusion-matrix and NEC functions against hand-computed values from recorded runs, plus synthetic checks |
| `results/` | `results.md` / `results.csv` (all tables), `sweep_runs.csv` (one row per run × rule × r), `sweep_summary.txt` / `.json`, `oof_sweep.json`, `nec_grid.json` |
| `figures/` | ROC, NEC-vs-r and reliability diagrams per benchmark, confusion matrices, the b4 OOF cost curve; `captions.md` |
| `paper/` | the paper and slides; `NOTES.md` records the reference-network derivation, the abandoned 150 px arm, and every methodological caveat |

---

## Headline results

Official test split, threshold chosen on the run's val split by Youden's *J*; mean ± std over five
seeds where available. Recall is the ICCAD "accuracy"; false alarms are counts. Full tables under
all five threshold rules — p = 0.5, val-Youden, val-cost at `r = N/P`, OOF-Youden, OOF-cost — are
in [`results/results.md`](results/results.md).

| Bench | Arm | Seeds | AUC | AP | Recall | False alarms |
|---|---|:-:|---|---|---|---|
| **b1** | arm0 (control) | 1 | 0.917 | 0.364 | 0.996 | 1790 |
| | ls0.1 (pos_weight) | 1 | 0.913 | 0.346 | 0.991 | 1790 |
| | ref64 | 1 | **0.982** | **0.779** | 0.996 | **976** |
| **b2** | arm0 | 5 | 0.997 ± 0.001 | 0.979 ± 0.027 | 0.988 ± 0.003 | 111 ± 159 |
| | ls0.1 | 5 | **0.998 ± 0.001** | **0.994 ± 0.002** | 0.989 ± 0.004 | **80 ± 111** |
| | ref64 | 1 | 0.996 | 0.834 | 0.962 | 553 |
| **b3** | arm0 | 1 | **0.993** | **0.908** | 0.978 | 1488 |
| | ls0.1 | 1 | 0.989 | 0.886 | 0.978 | **1454** |
| | ref64 | 1 | 0.989 | 0.785 | **0.986** | 3821 |
| **b4** | arm0 | 5 | 0.989 ± 0.006 | 0.851 ± 0.045 | **0.957 ± 0.017** | 670 ± 344 |
| | ls0.1 | 5 | **0.993 ± 0.007** | **0.859 ± 0.046** | 0.944 ± 0.022 | **414 ± 171** |
| | ref64 | 1 | 0.989 | 0.641 | 0.904 | 660 |
| **b5** | arm0 | 5 | 0.982 ± 0.003 | 0.725 ± 0.042 | **0.942 ± 0.037** | 61 ± 28 |
| | ls0.1 | 5 | 0.985 ± 0.005 | 0.582 ± 0.031 | 0.937 ± 0.033 | 107 ± 15 |
| | ref64 | 1 | **0.989** | **0.747** | 0.756 | **27** |
| **avg** | arm0 | – | 0.976 | **0.765** | **0.972** | 824 |
| | ls0.1 | – | 0.976 | 0.733 | 0.968 | **769** |
| | ref64 | – | **0.989** | 0.757 | 0.921 | 1207 |

### Threshold stability

Five fold models of the OOF run, each scored on the test split with its own fold's val-Youden
threshold, versus the single threshold chosen on the pooled out-of-fold scores.

| Bench | Arm | fold-val-Youden — recall / FA | OOF-Youden — recall / FA |
|---|---|---|---|
| b1 | arm0 | 0.999 ± 0.002 / 1794 ± 3 | 1.000 ± **0.000** / 1802 ± 6 |
| b1 | ls0.1 | 0.996 ± 0.003 / 1766 ± 22 | 0.998 ± **0.004** / 1794 ± 4 |
| b4 | arm0 | 0.937 ± 0.041 / 616 ± 352 | 0.965 ± **0.014** / 1840 ± 1547 |
| b4 | ls0.1 | 0.934 ± 0.050 / 542 ± 309 | 0.958 ± **0.010** / 1229 ± 1040 |
| b5 | arm0 | 0.932 ± 0.036 / 53 ± 13 | 0.971 ± **0.010** / 77 ± 12 |
| b5 | ls0.1 | 0.917 ± 0.037 / 101 ± 43 | 0.966 ± **0.012** / 165 ± 37 |

Higher recall, a third of the spread, no retraining and no architecture change. On b4 it buys
stability rather than cheapness — false alarms rise, and NEC at `r = N/P` is worse than the
80/20 rule (0.064 vs 0.032 for arm0), because the pool's few hard out-of-fold hotspots
(OOF AUC 0.994) pull the minimiser down.

---

## Two things the tables make explicit rather than hide

**The weighted model's sigmoid is not a posterior.** With `w = 30–104`, a BCE trained with
weight `w` estimates `q = wp/(wp + 1 − p)`, so logit 0 corresponds to `p = 1/(1+w)` — below every
test score. At p = 0.5 the `pos_weight` arm therefore flags *every* negative: FA = N exactly on
b2/b4/b5, balanced accuracy exactly 0.5000 ± 0.0000. Those rows are starred in every table and
excluded from every conclusion. Test ECE 0.63–0.85 falls to ≈ 0.002 once `log w` is subtracted
from the logit.

**The raw Elkan threshold is unusable for label-smoothed models.** Label smoothing caps outputs
at ≈ 0.95, so `t* = 1/(1+r)` falls outside the attainable range for `r ≳ 20` and the rule
degenerates to flagging everything. Cost-sensitive thresholds are therefore chosen either on
calibrated probabilities (prior shift + temperature fitted on val) or by minimising expected cost
directly on val / OOF scores.

---

¹ J. A. Torres, "ICCAD-2012 CAD contest in fuzzy pattern matching for physical verification and
benchmark suite," *Proc. IEEE/ACM ICCAD*, 2012, pp. 349–350.
