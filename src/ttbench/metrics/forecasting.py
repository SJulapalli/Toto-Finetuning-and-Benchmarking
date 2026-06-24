"""Forecasting metrics.

All metrics are numpy-only and operate on a single (univariate) forecast window
so they can be unit-tested in isolation and aggregated by the eval runner.

Shapes
------
- ``y_true``: 1-D array of length ``H`` (the horizon / prediction length).
- ``y_pred``: 1-D array of length ``H`` (a point forecast, usually the median).
- ``samples``: 2-D array ``(n_samples, H)`` for probabilistic metrics. A
  deterministic forecast is just ``n_samples == 1``.

The headline metrics are MASE (point) and CRPS (probabilistic), matching the
methodology BOOM / Gift-Eval report, so numbers here are comparable to
Datadog's published Toto results.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error. Scale-dependent; sanity metric only."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric mean absolute percentage error, in [0, 200].

    Scale-free but unstable near zero, hence secondary. The EPS guard keeps it
    finite when both true and predicted values are ~0.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred) + EPS
    return float(200.0 * np.mean(np.abs(y_true - y_pred) / denom))


def seasonal_naive_scale(context: np.ndarray, season_length: int) -> float:
    """In-sample MAE of a seasonal-naive forecast over the context window.

    This is the denominator of MASE. We measure how well "just repeat the value
    from one season ago" does on the history the model was given, then judge the
    model relative to that. Falls back to season_length=1 (a random-walk / lag-1
    scale) when the context is too short to hold a full season.
    """
    context = np.asarray(context, float)
    m = season_length if context.shape[0] > season_length else 1
    diffs = np.abs(context[m:] - context[:-m])
    scale = float(np.mean(diffs)) if diffs.size > 0 else 0.0
    return max(scale, EPS)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    context: np.ndarray,
    season_length: int,
) -> float:
    """Mean Absolute Scaled Error.

    MASE = MAE(forecast) / seasonal_naive_scale(context). Scale-free, so it can
    be averaged across heterogeneous telemetry. <1 beats seasonal-naive.
    """
    scale = seasonal_naive_scale(context, season_length)
    return mae(y_true, y_pred) / scale


def crps_from_samples(y_true: np.ndarray, samples: np.ndarray) -> float:
    """Continuous Ranked Probability Score via the energy (sample) estimator.

    CRPS = E|X - y| - 0.5 * E|X - X'|, where X, X' are independent draws from the
    forecast distribution. We average the per-timestep CRPS over the horizon.

    - First term rewards forecasts whose mass sits near the realized value.
    - Second term rewards *spread* (a sharp, confident forecast is penalized less
      only if it is also accurate), which is why CRPS captures calibration that
      MASE cannot.

    A deterministic forecast (n_samples == 1) has a zero second term, so CRPS
    reduces exactly to MAE -- the correct degenerate behavior.
    """
    y_true = np.asarray(y_true, float)
    samples = np.asarray(samples, float)
    if samples.ndim == 1:
        samples = samples[None, :]
    n = samples.shape[0]

    # E|X - y|: mean over samples of |sample - truth|, per timestep.
    term1 = np.mean(np.abs(samples - y_true[None, :]), axis=0)

    # E|X - X'|: mean absolute difference between all sample pairs, per timestep.
    # Computed without forming the full (n, n, H) tensor for the n==1 case.
    if n == 1:
        term2 = np.zeros_like(term1)
    else:
        # pairwise abs diffs along the sample axis, per timestep
        diffs = np.abs(samples[:, None, :] - samples[None, :, :])  # (n, n, H)
        term2 = diffs.sum(axis=(0, 1)) / (n * n)

    crps_per_step = term1 - 0.5 * term2
    return float(np.mean(crps_per_step))


def all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    samples: np.ndarray,
    context: np.ndarray,
    season_length: int,
) -> dict[str, float]:
    """Compute every metric for one forecast window."""
    return {
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred, context, season_length),
        "crps": crps_from_samples(y_true, samples),
    }
