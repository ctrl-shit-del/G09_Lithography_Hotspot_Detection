# Report notes (2026-09-19)

## What is reported from where

* **`runs_gpu/`** (Tesla T4, torch 2.11.0+cu128) is the reporting set, complete as of 2026-09-19:
  `arm0` and `ls0.1` with seeds 0–4 on b2/b4/b5 and seed 0 on b1/b3; `ref64` seed 0 on all five;
  `oof_arm0` and `oof_ls0.1` (5-fold CV over the full training pool, `src/oof.py`) on b1/b4/b5.
  Every run carries `val_logits.npy` / `val_labels.npy`.
* **`runs/`** (CPU laptop, torch 2.12.0+cpu) is superseded and used only for the reproducibility
  notes below.
* **Repair of the OOF fold score files (inference only, no training).** In the delivered zip the
  per-fold `probs.npy` / `logits.npy` / `val_logits.npy` of all 30 fold directories came from an
  earlier training pass: `oof.py` (as first written) skipped `save_scores` when `probs.npy` already
  existed, so a re-run in Colab left the first pass's scores next to the second pass's `best.pt`,
  `test.json`, `oof.json` and `oof_logits.npy` (which are mutually consistent — same best epoch,
  same counts). `python -m src.evaluate --refresh-stale --runs runs_gpu` re-scored each fold's
  `best.pt` on CPU and overwrote the score files **only after** the fresh scores reproduced that
  fold's `test.json` (TP/FP at logit 0 exactly, AUC to ≤ 1e-5, val-Youden threshold to ≤ 1e-3);
  the 36 single-model runs were checked the same way and were already consistent. Each repaired
  `eval.json` keeps the T4 timing fields and records a `scores_refreshed` block. `oof.py` now
  deletes stale score files before saving.

## The abandoned 150 px reference arm

The reference paper reports **12,873** parameters for its Keras network
(3×[Conv 12, 3×3] + BN → pool → MaxPool(5) → same block → Flatten → Dropout → Dense 10 → Dense 1)
but neither the input size nor the channel count. The count pins both down (`src/ref_model.py`):

| piece | count |
|---|---|
| conv1 | in_ch·12·9 + 12 = **336** (in_ch = 3) / 120 (in_ch = 1) |
| conv2–6 | 5 · (12·12·9 + 12) = 6,540 |
| BatchNorm ×2 | 2 · 4 · 12 = 96 (Keras counts γ, β, moving mean, moving var) |
| dense | 12·h·h·10 + 10 + 11 = 1,200·h² + 21 |
| **total** | 6,993 + 1,200·h² (in_ch = 3) ⇒ h² = 49, **h = 7** ; 6,777 + 1,200·h² (in_ch = 1) ⇒ h² = 5.08, impossible |

So the reference used a **3-channel input** (a grey clip duplicated into RGB, the Keras image-loader
default) and its second block flattens to 12×7×7. Walking the spatial chain back with 'same'
padding (7 ← pool2 ← 15 ← pool5 ← 75..79 ← pool2 ← 150..159) the smallest input reproducing the
count is **150 px**; no 'valid'-padding input in 8..512 px gives h = 7 with 3 channels.

We re-implemented and ran this exact configuration (`--model ref --data data150`, Nadam,
10 epochs, no augmentation, unweighted BCE, last epoch) on the CPU laptop:

| run | input | params (Keras) | test AUC | AP | recall / FA @0.5 | recall / FA @val-Youden | train time | infer ms/clip |
|---|---|---|---|---|---|---|---|---|
| `runs/ref/b1/seed0` | 3×150×150 | 12,873 | 0.9233 | 0.3114 | 0.040 / 18 | 1.000 / 1653 | 115 s | 8.37 (CPU) |
| `runs/ref/b2/seed0` | 3×150×150 | 12,873 | 0.9958 | 0.9109 | 0.351 / 4 | 0.986 / 377 | 977 s | 4.22 (CPU) |
| `runs/ref64/b1/seed0` | 1×64×64 | 7,857 | 0.9575 | 0.5931 | 0.792 / 242 | 0.982 / 1479 | 27 s | 1.46 (CPU) |
| `runs/ref64/b2/seed0` | 1×64×64 | 7,857 | 0.9959 | 0.8834 | 0.295 / 8 | 0.970 / 490 | 298 s | 1.32 (CPU) |

The 150 px, 3-channel input gave no better ranking quality than the same network on the repo's
1×64×64 clips (b2 AUC 0.9958 vs 0.9959; b1 0.923 vs 0.958) at 4–6× the inference cost and
~10× the training time, so the arm was abandoned after these two benchmarks (`runs/ref/b3`
was started and killed before evaluation). `ref64` on the T4 (`runs_gpu/ref64`) is the reference
comparison used in the tables. Note that the reference's own operating point (p = 0.5, last epoch,
unweighted BCE) catches almost no b1 hotspots (recall 0.04) because the sigmoid outputs of an
unsmoothed, unweighted model sit far below 0.5 for the rare class; every number in the tables is
therefore labelled with its threshold rule.

## Threshold rules and the cost sweep (Tasks 5 / 6)

* Label smoothing ε = 0.1 caps raw probabilities inside ≈ [0.05, 0.95], and `pos_weight = w`
  shifts every logit by ≈ log w (3.4–4.6 for the ls0.1 arm), so **the raw Elkan threshold
  t* = 1/(1+r) is meaningless for these models**: for r ≳ 20 it lies below every score
  (everything positive) and for the ls0.1 arm even p = 0.5 predicts every test clip positive on
  b2/b4/b5. `results/sweep_summary.txt` keeps the `elkan-raw` row to show this.
* Calibrated Elkan (`elkan-cal`): logits are prior-shifted by −log w (exact for a weighted BCE)
  and temperature-scaled with T fitted on the val logits. The shift alone brings the ls0.1 test
  ECE from 0.61–0.85 down to ≈ 0.002. On b1 (88 val clips, perfectly separated) and on 3/5 b2
  seeds T hits its lower bound (0.01), i.e. the val NLL has no interior minimum — those fits are
  flagged `at_bound` and their `elkan-cal` rows are effectively the p = 0.5 rule.
* `val-cost r` (threshold minimising NEC(r) on the val split) needs no calibration assumption and
  is the "cost-optimal at stated r" rule used in the tables, at r = N_test/P_test. Youden's J is
  maximised by the same rule as expected cost at r = N/P, so `val-Youden` ≈ `val-cost` at
  r = N_val/P_val (3.4 / 30.2 / 5.1 / 46.9 / 108.8 for b1–b5), which is emitted beside every
  Youden threshold together with the test-set N/P (20.7 / 82.9 / 25.6 / 180.2 / 471.4).
* Paired tests on b2/b4/b5 use n = 5 seeds; the exact Wilcoxon two-sided p cannot go below 0.0625,
  so the paired t-test p is reported alongside.

## OOF thresholds (Task 4) and the threshold-stability table

* On b1 and b5 the pooled out-of-fold scores are perfectly separated (OOF AUC 1.0 for both
  arms), so the OOF Youden threshold is a point inside the separating gap — but one chosen with
  all 99 / 26 training hotspots instead of the 20 / 5 in a single val split. For every r the
  NEC(r)-minimising OOF threshold is that same point, so `OOF-cost r=N/P` == `OOF-Youden` there
  and the OOF cost curve is the fixed-count line; the cost-curve figure therefore shows b4 only
  (`figures/nec_oof_b4.png`, OOF AUC 0.994 / 0.997) and says so in its caption.
* Headline of the stability section (`results/results.md`): the test-recall spread across
  the five fold models collapses when the per-fold val thresholds are replaced by the single
  OOF threshold — b5 arm0 0.932 ± 0.036 → 0.971 ± 0.010, ls0.1 0.917 ± 0.037 → 0.966 ± 0.012;
  b4 arm0 0.937 ± 0.041 → 0.965 ± 0.014, ls0.1 0.934 ± 0.050 → 0.958 ± 0.010. The price is a
  lower threshold: false alarms rise on b5 (53 → 77, 101 → 165) and on b4 the FA spread across fold
  models widens sharply (616 ± 352 → 1840 ± 1547; 542 ± 309 → 1229 ± 1040) because the OOF
  threshold sits below most per-fold ones.
* `p = 0.5` rows of the pos_weight arm are starred in every table: with w = 30–104 on b2/b4/b5,
  logit 0 flags every negative (FA = N exactly). It is evidence that the weighted sigmoid is not
  a posterior, not an operating point, and it is excluded from every conclusion.
* b4 cost curve (`nec_oof_b4.png`, refreshed fold scores): under the OOF-cost rule the mean NEC of
  the five fold models is 0.0137 / 0.0155 / 0.0297 / 0.0496 / 0.0620 (arm0) and 0.0045 / 0.0127 /
  0.0389 / 0.0477 / 0.0572 (ls0.1) at r = 1 / 5 / 20 / 100 / 200, i.e. 0.064 / 0.059 at r = N/P,
  against 0.032 / 0.035 for the val-cost rule of the 80/20 seed runs. The pooled OOF threshold is
  the more *stable* choice (recall spread 0.041 → 0.014, 0.050 → 0.010) but on b4 it is not the
  cheaper one: the pool's few hard out-of-fold hotspots (OOF AUC 0.994 / 0.997, not 1.0) pull the
  NEC-minimising threshold down (0.13 / 0.77 at r = N/P) and false alarms up (3374 ± 2584 /
  3208 ± 1892 vs 670 / 415). The two rules answer different questions — variance across
  training runs vs. expected cost on this test set — and the report should say so rather than
  present OOF as uniformly better.
