"""Benchmark regression gate.

Runs the full pipeline (synthetic generate -> windows -> run_benchmark) on a
fixed seed and asserts the aggregate metrics sit inside a pinned band. This is
the "automated regression test" that fails loudly if a change to the data
generator, windowing, metrics, or model silently shifts the numbers.

The bands below are captured from an actual run (seed=0). Re-pin intentionally
if you change the pipeline on purpose.
"""

import numpy as np

from ttbench.data import SyntheticConfig, generate, make_windows
from ttbench.eval import run_benchmark
from ttbench.models import SeasonalNaive

SEASON = 24

# Pinned expected bands (lo, hi) for the seasonal-naive baseline on seed=0.
# Captured from a real run: mase=1.7576, crps=8.6845 (mean across 256 windows).
# Bands are tight (~+/-1.5%) so a silent pipeline change trips them; re-pin
# intentionally if you change the pipeline on purpose.
EXPECTED = {
    "mase": (1.73, 1.78),
    "crps": (8.55, 8.82),
}


def _run():
    series, _ = generate(SyntheticConfig(n_series=64, length=1024, season_length=SEASON, seed=0))
    windows = make_windows(series, context_length=256, horizon=96, stride=96,
                           max_windows_per_series=4)
    model = SeasonalNaive(season_length=SEASON, residual_samples=100, seed=0)
    return run_benchmark(model, windows, season_length=SEASON)


def test_pipeline_runs_and_is_deterministic():
    r1 = _run()
    r2 = _run()
    # Same seed -> identical aggregates (proves end-to-end determinism).
    for m in ("mae", "smape", "mase", "crps"):
        assert np.isclose(r1.aggregate[m], r2.aggregate[m]), m
    assert r1.n_windows == 64 * 4


def test_metrics_within_pinned_bands():
    r = _run()
    for metric, (lo, hi) in EXPECTED.items():
        val = r.aggregate[metric]
        assert lo <= val <= hi, f"{metric}={val:.4f} outside [{lo}, {hi}]"
