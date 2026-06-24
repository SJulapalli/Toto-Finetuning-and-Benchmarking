"""Model interface shared by baselines, Toto zero-shot, and Toto fine-tuned.

The contract is deliberately minimal: given a 1-D context, return a ``Forecast``
holding samples of shape ``(n_samples, horizon)``. Point metrics read the median;
CRPS reads the full sample set. The eval runner is then model-agnostic, so adding
Chronos / Moirai / a fine-tuned checkpoint later is just a new ``predict``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Forecast:
    """A probabilistic forecast for a single window.

    samples : (n_samples, horizon). A deterministic forecast is n_samples == 1.
    """

    samples: np.ndarray

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, float)
        if self.samples.ndim == 1:
            self.samples = self.samples[None, :]

    @property
    def horizon(self) -> int:
        return self.samples.shape[1]

    @property
    def median(self) -> np.ndarray:
        """Point forecast: the per-timestep median across samples."""
        return np.median(self.samples, axis=0)

    @property
    def mean(self) -> np.ndarray:
        return np.mean(self.samples, axis=0)

    def quantile(self, q: float) -> np.ndarray:
        return np.quantile(self.samples, q, axis=0)


@runtime_checkable
class ForecastModel(Protocol):
    """Anything the runner can benchmark."""

    name: str

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        """Forecast ``horizon`` steps given a 1-D ``context`` history."""
        ...
