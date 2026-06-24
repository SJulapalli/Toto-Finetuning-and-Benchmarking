"""Toto 1.0 wrapper implementing the ForecastModel interface.

This is the translation layer between our harness (1-D context in, samples
(n_samples, horizon) out) and Toto's multivariate tensor API. It is imported
lazily (only when you actually request the Toto model) so the light core stays
torch-free.

Install the dependency group first:

    pip install -e ".[toto]"

Toto 1.0 is used (not 2.0) because it supports fine-tuning, so the same model
backs both the zero-shot benchmark here and the Project-2 fine-tuned comparison.
"""

from __future__ import annotations

import numpy as np

from ttbench.models.base import Forecast

try:
    import torch
    from toto.data.util.dataset import MaskedTimeseries
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "Toto support requires the 'toto' extra. Install it with:\n"
        '    pip install -e ".[toto]"'
    ) from e


def _select_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    # MPS is intentionally NOT auto-selected: Toto weights are bf16 and MPS
    # bf16 coverage is spotty. Pass device="mps" explicitly to override.
    return "cpu"


class TotoForecastModel:
    """Zero-shot (or fine-tuned) Toto forecaster behind the ForecastModel API."""

    def __init__(
        self,
        checkpoint: str = "Datadog/Toto-Open-Base-1.0",
        num_samples: int = 256,
        samples_per_batch: int = 256,
        time_interval_seconds: int = 3600,
        device: str | None = None,
    ):
        self.checkpoint = checkpoint
        self.num_samples = num_samples
        self.samples_per_batch = samples_per_batch
        self.time_interval_seconds = int(time_interval_seconds)
        self.device = _select_device(device)
        self.name = f"toto_{checkpoint.split('/')[-1]}"

        model = Toto.from_pretrained(checkpoint).to(self.device).eval()
        # On CPU, cast bf16 weights up to fp32 for correctness/speed.
        if self.device == "cpu":
            model = model.float()
        self.toto = model
        self.forecaster = TotoForecaster(self.toto.model)

    def predict(self, context: np.ndarray, horizon: int) -> Forecast:
        context = np.asarray(context, dtype=np.float32)
        L = context.shape[0]

        # Map 1-D univariate context -> Toto's (batch=1, variates=1, time=L).
        series = torch.from_numpy(context).to(self.device).reshape(1, 1, L)
        padding_mask = torch.ones((1, 1, L), dtype=torch.bool, device=self.device)
        id_mask = torch.zeros((1, 1, L), dtype=torch.long, device=self.device)
        timestamps = (
            torch.arange(L, dtype=torch.long, device=self.device)
            * self.time_interval_seconds
        ).reshape(1, 1, L)
        # time_interval_seconds has NO time dim: shape (batch, variates).
        intervals = torch.full(
            (1, 1), self.time_interval_seconds, dtype=torch.long, device=self.device
        )

        inputs = MaskedTimeseries(
            series=series,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamps,
            time_interval_seconds=intervals,
        )

        with torch.no_grad():
            fc = self.forecaster.forecast(
                inputs,
                prediction_length=horizon,
                num_samples=self.num_samples,
                samples_per_batch=self.samples_per_batch,
            )

        if fc.samples is None:
            raise RuntimeError(
                "Toto returned no samples; set num_samples > 0 for CRPS."
            )

        # samples: (batch, variate, time, samples) -> our (n_samples, horizon).
        samples = fc.samples[0, 0]          # (horizon, n_samples)
        samples = samples.permute(1, 0)     # (n_samples, horizon)
        return Forecast(samples.float().cpu().numpy())
