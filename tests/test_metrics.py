"""Unit tests for the forecasting metrics, pinned to hand-checkable cases."""

import numpy as np

from ttbench.metrics import (
    crps_from_samples,
    mae,
    mase,
    seasonal_naive_scale,
    smape,
)


def test_perfect_forecast_is_zero_everywhere():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    ctx = np.arange(48, dtype=float)
    assert mae(y, y) == 0.0
    assert smape(y, y) < 1e-6
    assert mase(y, y, ctx, season_length=24) == 0.0
    assert crps_from_samples(y, y[None, :]) == 0.0


def test_crps_equals_mae_for_deterministic_forecast():
    # With a single sample the second CRPS term vanishes, so CRPS == MAE exactly.
    rng = np.random.default_rng(0)
    y = rng.normal(size=20)
    pred = rng.normal(size=20)
    assert np.isclose(crps_from_samples(y, pred[None, :]), mae(y, pred))


def test_mase_is_one_for_seasonal_naive_forecast():
    # If the "forecast" is itself a seasonal-naive copy, MASE should be ~1:
    # numerator and denominator are the same kind of one-season-ago error.
    m = 12
    rng = np.random.default_rng(1)
    series = rng.normal(size=200).cumsum()  # a random walk
    ctx, horizon = series[:150], 24
    target = series[150:150 + horizon]
    # seasonal-naive point forecast: repeat the value from m steps before each step
    naive_pred = series[150 - m:150 - m + horizon]
    val = mase(target, naive_pred, ctx, season_length=m)
    assert 0.5 < val < 2.0  # close to 1, loose band for the random instance


def test_crps_rewards_sharp_calibrated_distribution():
    # A tight distribution centered on the truth must beat a diffuse one.
    rng = np.random.default_rng(2)
    y = np.zeros(10)
    sharp = rng.normal(0.0, 0.1, size=(200, 10))
    diffuse = rng.normal(0.0, 5.0, size=(200, 10))
    assert crps_from_samples(y, sharp) < crps_from_samples(y, diffuse)


def test_seasonal_naive_scale_floor_and_fallback():
    # Flat context -> scale floored to EPS (no division blow-up).
    flat = np.ones(48)
    assert seasonal_naive_scale(flat, season_length=24) > 0.0
    # Context shorter than a season -> falls back to lag-1 without error.
    short = np.array([1.0, 3.0, 2.0, 5.0])
    assert seasonal_naive_scale(short, season_length=24) > 0.0
