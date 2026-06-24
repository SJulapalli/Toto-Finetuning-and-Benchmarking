"""Synthetic telemetry generator.

The goal is data that *looks like* observability metrics -- seasonal, trending,
noisy, and occasionally spiky -- so the benchmark runs end-to-end on a laptop
with no downloads and no GPU. Real BOOM data is loaded by ``data/boom.py``
(optional, behind the ``toto`` extra) and exposes the same interface.

Each series is the sum of:
  - a base level (per-series random),
  - a slow linear trend,
  - one or two seasonal sine components (e.g. daily + weekly),
  - Gaussian observation noise,
  - injected anomalies: additive spikes and persistent level-shifts.

Anomaly positions are returned as a boolean mask so the same generator can later
feed an anomaly-detection task, not just forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticConfig:
    n_series: int = 64
    length: int = 1024          # timesteps per series
    season_length: int = 24     # primary seasonality (e.g. 24 = daily on hourly data)
    add_weekly: bool = True     # add a secondary 7*season_length component
    noise_scale: float = 0.05   # noise as a fraction of the seasonal amplitude
    anomaly_rate: float = 0.01  # fraction of timesteps that get a spike
    seed: int = 0


def generate(cfg: SyntheticConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic telemetry dataset.

    Returns
    -------
    series : np.ndarray, shape (n_series, length), float, non-negative
    anomalies : np.ndarray, shape (n_series, length), bool
    """
    rng = np.random.default_rng(cfg.seed)
    t = np.arange(cfg.length)

    series = np.empty((cfg.n_series, cfg.length), dtype=float)
    anomalies = np.zeros((cfg.n_series, cfg.length), dtype=bool)

    for i in range(cfg.n_series):
        base = rng.uniform(10.0, 100.0)
        amp = rng.uniform(5.0, 30.0)
        trend = rng.normal(0.0, 0.01) * t  # gentle drift, can be +/-

        seasonal = amp * np.sin(2 * np.pi * t / cfg.season_length + rng.uniform(0, 2 * np.pi))
        if cfg.add_weekly:
            weekly_period = cfg.season_length * 7
            seasonal += 0.5 * amp * np.sin(2 * np.pi * t / weekly_period + rng.uniform(0, 2 * np.pi))

        noise = rng.normal(0.0, cfg.noise_scale * amp, size=cfg.length)
        y = base + trend + seasonal + noise

        # Additive spikes (transient anomalies).
        n_spikes = rng.binomial(cfg.length, cfg.anomaly_rate)
        if n_spikes > 0:
            idx = rng.choice(cfg.length, size=n_spikes, replace=False)
            y[idx] += rng.uniform(3, 8, size=n_spikes) * amp * rng.choice([-1, 1], size=n_spikes)
            anomalies[i, idx] = True

        # An occasional persistent level-shift (regime change).
        if rng.random() < 0.3:
            shift_at = rng.integers(cfg.length // 4, cfg.length * 3 // 4)
            y[shift_at:] += rng.uniform(1, 3) * amp * rng.choice([-1, 1])
            anomalies[i, shift_at] = True

        series[i] = np.clip(y, 0.0, None)  # telemetry is typically non-negative

    return series, anomalies
