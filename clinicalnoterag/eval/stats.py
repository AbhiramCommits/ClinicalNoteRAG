"""Bootstrap confidence intervals and paired system comparisons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 42


def bootstrap_indices(
    n: int, n_resamples: int = DEFAULT_RESAMPLES, seed: int = DEFAULT_SEED
) -> np.ndarray:
    """Return a reproducible ``(n_resamples, n)`` matrix of item indices."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_resamples, n))


def bootstrap_ci(
    values: np.ndarray | list[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    statistic: Callable[[np.ndarray], np.ndarray] | None = None,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean (or a custom vectorized statistic).

    Args:
        values: Per-item metric values.
        n_resamples: Number of bootstrap resamples (10,000 by project contract).
        seed: Fixed seed (42).
        statistic: Optional vectorized function mapping a resampled array to a
            scalar. Defaults to the mean.
        alpha: Two-sided error rate.

    Returns:
        ``(low, high)`` percentile interval; ``(0, 0)`` for empty input.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return (0.0, 0.0)
    indices = bootstrap_indices(array.size, n_resamples, seed)
    samples = array[indices]
    if statistic is None:
        statistics = samples.mean(axis=1)
    else:
        statistics = np.asarray([statistic(sample) for sample in samples], dtype=float)
    low, high = np.percentile(statistics, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(low), float(high))


def bootstrap_mean_ci(
    values: np.ndarray | list[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Return point estimate and 95% CI for a mean metric."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    low, high = bootstrap_ci(array, n_resamples=n_resamples, seed=seed)
    return {
        "value": round(float(array.mean()), 6),
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
        "n": int(array.size),
    }


def bootstrap_ratio_ci(
    numerator: np.ndarray | list[float],
    denominator: np.ndarray | list[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Bootstrap a ratio of summed per-item counts (e.g. micro precision)."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if num.size == 0 or den.size != num.size:
        return {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    indices = bootstrap_indices(num.size, n_resamples, seed)
    samples_num = num[indices].sum(axis=1)
    samples_den = den[indices].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = np.where(samples_den > 0, samples_num / samples_den, 0.0)
    total_den = den.sum()
    value = float(num.sum() / total_den) if total_den > 0 else 0.0
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "value": round(value, 6),
        "ci_low": round(float(low), 6),
        "ci_high": round(float(high), 6),
        "n": int(num.size),
    }


@dataclass
class PairedResult:
    """Paired bootstrap comparison of two systems on the same items."""

    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serializable representation."""
        return {
            "delta": round(self.delta, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "p_value": round(self.p_value, 6),
            "n": self.n,
        }


def paired_bootstrap_delta(
    values_a: np.ndarray | list[float],
    values_b: np.ndarray | list[float],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> PairedResult:
    """Paired bootstrap of mean(a) - mean(b) with a two-sided p-value.

    ``values_a`` and ``values_b`` must be aligned per benchmark item (higher is
    better for both). The p-value is the twice the smaller one-sided bootstrap
    tail probability, capped at 1.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.size == 0 or a.size != b.size:
        raise ValueError("paired arrays must be non-empty and the same length")
    differences = a - b
    indices = bootstrap_indices(differences.size, n_resamples, seed)
    samples = differences[indices].mean(axis=1)
    delta = float(differences.mean())
    low, high = np.percentile(samples, [2.5, 97.5])
    p_value = 2.0 * min(float((samples <= 0).mean()), float((samples >= 0).mean()))
    p_value = min(1.0, p_value)
    return PairedResult(
        delta=delta, ci_low=float(low), ci_high=float(high), p_value=p_value, n=int(a.size)
    )


__all__ = [
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "PairedResult",
    "bootstrap_ci",
    "bootstrap_indices",
    "bootstrap_mean_ci",
    "bootstrap_ratio_ci",
    "paired_bootstrap_delta",
]
