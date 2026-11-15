"""Hand-computed metric, bootstrap and split tests."""

from __future__ import annotations

import numpy as np

from clinicalnoterag.eval.metrics import (
    code_counts,
    exact_match,
    normalize_answer,
    numeric_match,
    prf,
    reciprocal_rank,
    retrieval_recall,
    token_f1,
)
from clinicalnoterag.eval.stats import (
    bootstrap_ci,
    bootstrap_mean_ci,
    paired_bootstrap_delta,
)
from clinicalnoterag.train.dataset import split_patients


def test_token_f1_hand_computed():
    assert abs(token_f1("a b c", "a b") - 0.8) < 1e-9
    assert token_f1("a b", "a b") == 1.0
    assert token_f1("x", "y") == 0.0


def test_exact_match_ignores_citations_and_trailing_period():
    assert exact_match("Blood pressure 120 mmHg [1].", "blood pressure 120 mmHg")
    assert normalize_answer("A [1].  B") == "a. b"


def test_retrieval_recall_and_mrr_hand_computed():
    assert retrieval_recall(["a", "b"], ["b", "c"]) == 0.5
    assert retrieval_recall([], []) is None
    assert reciprocal_rank(["x", "b", "a"], ["a", "b"]) == 0.5
    assert reciprocal_rank(["x"], ["a"]) == 0.0


def test_numeric_match_handles_currency_and_patient_ids():
    assert numeric_match("Total $1,159.80 for PAT-0001", 1159.8) == 1.0
    assert numeric_match("There are 2 encounters", 2.0) == 1.0
    assert numeric_match("There are 2 encounters", 3.0) == 0.0


def test_code_counts_and_prf_hand_computed():
    tp, fp, fn = code_counts({("ICD-10", "I10"), ("CPT", "99213")}, {("ICD-10", "I10")})
    assert (tp, fp, fn) == (1, 1, 0)
    precision, recall, f1 = prf(1, 0, 1)
    assert precision == 1.0
    assert recall == 0.5
    assert abs(f1 - 2 / 3) < 1e-9


def test_bootstrap_ci_is_reproducible_under_seed():
    values = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    first = bootstrap_ci(values, n_resamples=500, seed=42)
    second = bootstrap_ci(values, n_resamples=500, seed=42)
    assert first == second
    low, high = first
    assert low <= np.mean(values) <= high


def test_bootstrap_mean_ci_empty():
    result = bootstrap_mean_ci([], n_resamples=100)
    assert result == {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}


def test_paired_bootstrap_identical_systems():
    values = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0])
    result = paired_bootstrap_delta(values, values, n_resamples=500, seed=42)
    assert result.delta == 0.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0
    assert result.p_value == 1.0


def test_paired_bootstrap_detects_a_real_shift():
    baseline = np.zeros(50)
    tuned = np.ones(50)
    result = paired_bootstrap_delta(tuned, baseline, n_resamples=500, seed=42)
    assert result.delta == 1.0
    assert result.p_value < 0.01


def test_patient_split_is_disjoint_and_complete():
    patients = [f"PAT-{i:04d}" for i in range(1, 35)]
    splits = split_patients(patients, seed=42)
    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])
    assert train and val and test
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert train | val | test == set(patients)
    assert len(train) > len(val)
