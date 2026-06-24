"""Baseline forecasters.

SeasonalNaive is the scientific control. It is also definitionally tied to MASE
(whose denominator is the seasonal-naive error), so reporting it makes the MASE
numbers interpretable: a real model must beat this to earn MASE < 1.
"""

from __future__ import annotations

import numpy as np

from ttbench.models.base import Forecast


class SeasonalNaive:
    """Repeat the last observed season, tiled to cover the horizon.

    For season_length m, the forecast for step h is context[-m + (h mod m)] --
    i.e. "the same point one season ago." With residual_samples=0 it is purely
    deterministic (one sample). With residual_samples>0 we wrap the point
    forecast in a crude Gaussian drawn from in-sample residuals, which gives
    SeasonalNaive a non-degenerate CRPS for fairer comparison against
    probabilistic models.
    """

    def __init__(self, season_length: int, residual_samples: int = 0, seed: int = 0):
        self.season_length = season_length
        self.residual_samples = residual_samples
        self.name = f"seasonal_naive_m{season_length}"
        self._rng = np.random.default_rng(seed)

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        context = np.asarray(context, float)
        m = self.season_length if context.shape[0] >= self.season_length else 1

        last_season = context[-m:]
        reps = int(np.ceil(horizon / m))
        point = np.tile(last_season, reps)[:horizon]

        if self.residual_samples <= 0:
            return Forecast(point)

        # Spread estimated from one-season-ago residuals on the context.
        resid = context[m:] - context[:-m]
        sigma = float(np.std(resid)) if resid.size > 0 else 0.0
        noise = self._rng.normal(0.0, sigma, size=(self.residual_samples, horizon))
        samples = np.clip(point[None, :] + noise, 0.0, None)
        return Forecast(samples)
