"""Unit tests for the offline metrics (Task 5).

    python -m pytest tests -q

The confusion-matrix / NEC checks use the recorded counts in two real run directories
(runs_gpu/arm0/b1/seed0 and runs_gpu/ls0.1/b5/seed0, written by train.py on the GPU) and
NEC values worked out by hand from those counts; the remaining checks use tiny synthetic
vectors where the answer is known in closed form.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src import metrics as M

ROOT = Path(__file__).resolve().parents[1]
RUN_B1 = ROOT / "runs_gpu" / "arm0" / "b1" / "seed0"
RUN_B5 = ROOT / "runs_gpu" / "ls0.1" / "b5" / "seed0"
needs_runs = pytest.mark.skipif(not (RUN_B1 / "probs.npy").exists() or not (RUN_B5 / "probs.npy").exists(),
                                reason="runs_gpu not extracted")


# ---- confusion matrix against test.json ----------------------------------------------------------
@needs_runs
@pytest.mark.parametrize("run_dir,rule,key", [
    (RUN_B1, "0.5", "at_0p5"), (RUN_B1, "val-Youden", "at_val_youden"),
    (RUN_B5, "0.5", "at_0p5"), (RUN_B5, "val-Youden", "at_val_youden"),
])
def test_confusion_matrix_matches_test_json(run_dir, rule, key):
    run = M.load_run(run_dir)
    rec = json.loads((run_dir / "test.json").read_text())[key]
    thr = 0.5 if rule == "0.5" else run["thr_youden"]
    cm = M.confusion_matrix(run["labels"], run["probs"], thr)
    assert (cm["tp"], cm["fp"], cm["fn"], cm["tn"]) == (rec["tp"], rec["fp"], rec["fn"], rec["tn"])
    assert cm["P"] == rec["n_hs"] and cm["N"] == rec["n_nhs"]
    # the recorded thresholds are logits; the prob-space threshold must be their sigmoid
    assert math.isclose(thr, 1 / (1 + math.exp(-rec["thr"])), rel_tol=1e-9)


@needs_runs
def test_b1_arm0_recorded_counts_are_what_we_think():
    """Pin the counts the hand computations below are based on (val-Youden == 0.5 rule on b1)."""
    rec = json.loads((RUN_B1 / "test.json").read_text())["at_val_youden"]
    assert (rec["tp"], rec["fp"], rec["fn"], rec["tn"]) == (225, 1790, 1, 2889)


@needs_runs
def test_rates_match_test_json_b1():
    run = M.load_run(RUN_B1)
    rec = json.loads((RUN_B1 / "test.json").read_text())["at_val_youden"]
    m = M.at_threshold(run["labels"], run["probs"], run["thr_youden"])
    assert math.isclose(m["recall"], rec["accuracy"], abs_tol=1e-12)          # hotspot 'accuracy'
    assert math.isclose(m["precision"], rec["precision"], abs_tol=1e-12)
    assert math.isclose(m["f1"], rec["f1"], abs_tol=1e-12)
    assert math.isclose(m["fpr"], rec["fpr"], abs_tol=1e-12)
    assert m["false_alarms"] == rec["false_alarms"] == 1790
    # hand: recall = 225/226, specificity = 2889/4679, balanced = mean of the two
    assert math.isclose(m["specificity"], 2889 / 4679, abs_tol=1e-12)
    assert math.isclose(m["balanced_accuracy"], (225 / 226 + 2889 / 4679) / 2, abs_tol=1e-12)


# ---- NEC against hand-computed values ------------------------------------------------------------
@needs_runs
def test_nec_hand_computed_b1():
    # b1 arm0 seed0 @ val-Youden: TP 225, FP 1790, FN 1, TN 2889 -> P = 226, N = 4679
    run = M.load_run(RUN_B1)
    cm = M.confusion_matrix(run["labels"], run["probs"], run["thr_youden"])
    assert math.isclose(M.nec(cm, 1), 1791 / 4905, abs_tol=1e-12)           # (1*1 + 1790)/(226 + 4679)   = 0.365138
    assert math.isclose(M.nec(cm, 10), 1800 / 6939, abs_tol=1e-12)          # (10 + 1790)/(2260 + 4679)   = 0.259403
    assert math.isclose(M.nec(cm, 100), 1890 / 27279, abs_tol=1e-12)        # (100 + 1790)/(22600 + 4679) = 0.069284
    assert math.isclose(M.nec(cm, 1), 0.365138, abs_tol=5e-7)
    assert math.isclose(M.nec(cm, 10), 0.259403, abs_tol=5e-7)
    assert math.isclose(M.nec(cm, 100), 0.069284, abs_tol=5e-7)
    # at r = N/P the NEC is 1 - balanced accuracy
    r = M.implied_r(run["labels"])
    assert math.isclose(r, 4679 / 226, rel_tol=1e-12)
    assert math.isclose(M.nec(cm, r), 1 - M.rates(cm)["balanced_accuracy"], abs_tol=1e-12)
    assert math.isclose(M.nec(cm, r), (1 / 226 + 1790 / 4679) / 2, abs_tol=1e-12)     # = 0.193492


@needs_runs
def test_nec_hand_computed_b5():
    # b5 ls0.1 seed0: @0.5 everything is positive (TP 41, FP 19327); @val-Youden TP 37, FP 96, FN 4, TN 19231
    run = M.load_run(RUN_B5)
    cm05 = M.confusion_matrix(run["labels"], run["probs"], 0.5)
    assert (cm05["tp"], cm05["fp"], cm05["fn"], cm05["tn"]) == (41, 19327, 0, 0)
    assert math.isclose(M.nec(cm05, 1), 19327 / 19368, abs_tol=1e-12)                 # 0.997883
    assert math.isclose(M.nec(cm05, 471.39), 19327 / (471.39 * 41 + 19327), abs_tol=1e-12)
    cmy = M.confusion_matrix(run["labels"], run["probs"], run["thr_youden"])
    assert (cmy["tp"], cmy["fp"], cmy["fn"], cmy["tn"]) == (37, 96, 4, 19231)
    assert math.isclose(M.nec(cmy, 1), 100 / 19368, abs_tol=1e-12)                    # (4 + 96)/(41 + 19327)     = 0.005163
    assert math.isclose(M.nec(cmy, 100), 496 / 23427, abs_tol=1e-12)                  # (400 + 96)/(4100 + 19327) = 0.021172
    assert math.isclose(M.nec(cmy, 100), 0.021172, abs_tol=5e-7)


def test_nec_extremes_synthetic():
    y = np.array([1, 1, 0, 0, 0])
    all_wrong = dict(tp=0, fp=3, fn=2, tn=0)
    all_neg = dict(tp=0, fp=0, fn=2, tn=3)
    perfect = dict(tp=2, fp=0, fn=0, tn=3)
    for r in (1, 7, 100):
        assert M.nec(all_wrong, r) == 1.0
        assert math.isclose(M.nec(all_neg, r), r * 2 / (r * 2 + 3))
        assert M.nec(perfect, r) == 0.0
    assert M.implied_r(y) == 1.5


def test_min_nec_finds_oracle_threshold():
    y = np.array([0, 0, 1, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    # r = 1: best is thr in (0.4, 0.8): FN 1, FP 0 -> 1/6 ; or thr in (0.2,0.3): FN 0, FP 1 -> 1/6 (tie, lowest thr wins)
    v, t = M.min_nec(y, p, 1)
    assert math.isclose(v, 1 / 6) and 0.2 < t < 0.3
    # r = 10: FN is expensive -> thr in (0.2, 0.3): (0 + 1)/(30 + 3)
    v, t = M.min_nec(y, p, 10)
    assert math.isclose(v, 1 / 33) and 0.2 < t < 0.3
    thr, vals = M.nec_curve(y, p, 1)
    assert np.all(np.diff(thr) > 0) and math.isclose(vals[-1], 3 / 6)        # all-negative rule last


# ---- Elkan threshold / implied r --------------------------------------------------------------------
def test_elkan_threshold_and_inverse():
    for r in (1, 2, 5, 10, 20, 50, 100, 200):
        t = M.elkan_threshold(r)
        assert math.isclose(t, 1 / (1 + r))
        assert math.isclose(M.cost_ratio_of_threshold(t), r, rel_tol=1e-12)
    assert M.elkan_threshold(1) == 0.5 and M.cost_ratio_of_threshold(0.5) == 1.0


# ---- partial AUC ------------------------------------------------------------------------------------
def test_partial_auc_perfect_and_worst():
    y = np.array([0] * 50 + [1] * 10)
    perfect = np.r_[np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 10)]
    assert math.isclose(M.partial_auc_high_recall(y, perfect)["pauc"], 1.0, abs_tol=1e-12)
    assert M.partial_auc_high_recall(y, perfect)["fpr_at_recall"] == 0.0
    worst = 1 - perfect
    assert math.isclose(M.partial_auc_high_recall(y, worst)["pauc"], 0.0, abs_tol=1e-12)


def test_partial_auc_random_is_about_0p025():
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.2).astype(int)
    p = rng.random(20000)
    v = M.partial_auc_high_recall(y, p)["pauc"]
    assert abs(v - 0.025) < 0.01


# ---- calibration ------------------------------------------------------------------------------------
def test_temperature_recovers_scale():
    rng = np.random.default_rng(1)
    z_true = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < M.sigmoid(z_true)).astype(int)
    fit = M.fit_temperature(z_true * 3.0, y)                    # logits over-confident by x3
    assert abs(fit["T"] - 3.0) < 0.15 and not fit["at_bound"]
    assert fit["nll_after"] < fit["nll_before"]
    fit1 = M.fit_temperature(z_true, y)
    assert abs(fit1["T"] - 1.0) < 0.05


def test_temperature_separable_hits_bound():
    z = np.r_[np.full(10, 2.0), np.full(10, -2.0)]
    y = np.r_[np.ones(10), np.zeros(10)]
    assert M.fit_temperature(z, y)["at_bound"]


def test_ece_and_reliability_bins():
    y = np.array([1, 0, 1, 1, 0, 0])
    p = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    rel = M.reliability(y, p, n_bins=10)
    # bin 9 holds three 0.9s with 2/3 positives; bin 1 holds three 0.1s with 1/3 positives
    assert rel["count"][9] == 3 and rel["count"][1] == 3
    assert math.isclose(rel["acc"][9], 2 / 3) and math.isclose(rel["acc"][1], 1 / 3)
    assert math.isclose(rel["ece"], 0.5 * abs(0.9 - 2 / 3) + 0.5 * abs(0.1 - 1 / 3), abs_tol=1e-12)
    assert math.isclose(M.ece(y, p, 10), rel["ece"])


# ---- McNemar -----------------------------------------------------------------------------------------
def test_mcnemar_exact():
    y = np.array([1] * 5 + [0] * 5)
    a = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])     # 1 FN, 1 FP
    b = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 1])     # 0 FN, 1 FP (same FP)
    m = M.mcnemar_exact(y, a, b)
    assert (m["b"], m["c"]) == (0, 1) and m["c_hs"] == 1 and m["p_value"] == 1.0   # one discordant: p = 2*0.5
    m2 = M.mcnemar_exact(y, b, a)
    assert (m2["b"], m2["c"]) == (1, 0)
    # ten discordants all one way: exact two-sided p = 2 * 0.5**10
    a = np.zeros(10, int); b = np.ones(10, int); y = np.ones(10, int)
    assert math.isclose(M.mcnemar_exact(y, a, b)["p_value"], 2 * 0.5 ** 10)
    assert M.mcnemar_exact(y, a, a)["p_value"] == 1.0
