"""The benchmark loop.

``run_benchmark`` is the pure, model-agnostic core: windows in, aggregated
metrics out. ``main`` is the config-driven CLI wrapper (the console-script
target ``ttbench-benchmark``) that builds the dataset + model, runs the loop,
prints a table, and writes a reproducible JSON result.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ttbench.data import SyntheticConfig, generate, make_windows
from ttbench.metrics import all_metrics
from ttbench.models import SeasonalNaive
from ttbench.models.base import ForecastModel

METRIC_NAMES = ("mae", "smape", "mase", "crps")


@dataclass
class BenchmarkResult:
    """Everything needed to report and reproduce a benchmark run."""

    model_name: str
    n_windows: int
    season_length: int
    horizon: int
    context_length: int
    aggregate: dict[str, float]            # mean of each metric across windows
    aggregate_median: dict[str, float]     # median of each metric across windows
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_table(self) -> str:
        lines = [
            f"model={self.model_name}  windows={self.n_windows}  "
            f"H={self.horizon}  ctx={self.context_length}  m={self.season_length}",
            f"{'metric':<8}{'mean':>12}{'median':>12}",
            "-" * 32,
        ]
        for m in METRIC_NAMES:
            lines.append(f"{m:<8}{self.aggregate[m]:>12.4f}{self.aggregate_median[m]:>12.4f}")
        return "\n".join(lines)


def run_benchmark(
    model: ForecastModel,
    windows,
    season_length: int,
    config: dict[str, Any] | None = None,
) -> BenchmarkResult:
    """Score ``model`` over ``windows`` and aggregate.

    Per window: predict -> derive point forecast (median) + samples -> metrics.
    MASE needs the context (its denominator is the seasonal-naive error on the
    history the model saw), so it is threaded through here rather than computed
    in a separate pass.
    """
    if not windows:
        raise ValueError("no windows to evaluate")

    per_window: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    horizon = windows[0].target.shape[0]

    for w in windows:
        forecast = model.predict(w.context, horizon)
        scores = all_metrics(
            y_true=w.target,
            y_pred=forecast.median,
            samples=forecast.samples,
            context=w.context,
            season_length=season_length,
        )
        for m in METRIC_NAMES:
            per_window[m].append(scores[m])

    aggregate = {m: float(np.mean(per_window[m])) for m in METRIC_NAMES}
    aggregate_median = {m: float(statistics.median(per_window[m])) for m in METRIC_NAMES}

    return BenchmarkResult(
        model_name=getattr(model, "name", model.__class__.__name__),
        n_windows=len(windows),
        season_length=season_length,
        horizon=horizon,
        context_length=int(windows[0].context.shape[0]),
        aggregate=aggregate,
        aggregate_median=aggregate_median,
        config=config or {},
    )


def compare_table(results: list[BenchmarkResult]) -> str:
    """Side-by-side mean metrics for several models evaluated on the SAME windows.

    Also shows MASE/CRPS as a ratio vs the first model (the baseline), so the
    improvement reads directly: ratio < 1.0 means "better than baseline".
    """
    if not results:
        return "(no results)"
    base = results[0]
    header = f"{'model':<28}" + "".join(f"{m:>12}" for m in METRIC_NAMES)
    lines = [
        f"windows={base.n_windows}  H={base.horizon}  ctx={base.context_length}  "
        f"m={base.season_length}",
        header,
        "-" * len(header),
    ]
    for r in results:
        row = f"{r.model_name:<28}" + "".join(f"{r.aggregate[m]:>12.4f}" for m in METRIC_NAMES)
        lines.append(row)
    # relative improvement on the headline metrics
    lines.append("-" * len(header))
    for metric in ("mase", "crps"):
        ratios = " ".join(
            f"{r.model_name.split('_')[0]}={r.aggregate[metric] / base.aggregate[metric]:.3f}"
            for r in results[1:]
        )
        if ratios:
            lines.append(f"{metric} vs {base.model_name} (lower=better): {ratios}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load_config(path: str | None) -> dict[str, Any]:
    """Load a YAML config, or return defaults if no path is given."""
    defaults: dict[str, Any] = {
        "dataset": "synthetic",
        "synthetic": {"n_series": 64, "length": 1024, "season_length": 24, "seed": 0},
        "window": {"context_length": 256, "horizon": 96, "stride": 96,
                   "max_windows_per_series": 4},
        "model": {"name": "seasonal_naive", "residual_samples": 100},
    }
    if path is None:
        return defaults
    import yaml  # local import: keeps yaml optional at import time
    with open(path) as fh:
        loaded = yaml.safe_load(fh) or {}
    # shallow merge over defaults so partial configs work
    merged = {**defaults, **loaded}
    return merged


def _build_dataset(cfg: dict[str, Any]):
    if cfg["dataset"] != "synthetic":
        raise NotImplementedError(
            f"dataset '{cfg['dataset']}' not wired yet (BOOM loader is a later milestone)"
        )
    s = cfg["synthetic"]
    series, _anomalies = generate(SyntheticConfig(**s))
    w = cfg["window"]
    windows = make_windows(
        series,
        context_length=w["context_length"],
        horizon=w["horizon"],
        stride=w.get("stride"),
        max_windows_per_series=w.get("max_windows_per_series"),
    )
    return series, windows


def _build_model(cfg: dict[str, Any], season_length: int) -> ForecastModel:
    m = cfg["model"]
    if m["name"] == "seasonal_naive":
        return SeasonalNaive(
            season_length=season_length,
            residual_samples=m.get("residual_samples", 0),
            seed=m.get("seed", 0),
        )
    if m["name"] == "toto":
        # Lazy import: pulls in torch/toto only when actually requested.
        from ttbench.models.toto_model import TotoForecastModel

        return TotoForecastModel(
            checkpoint=m.get("checkpoint", "Datadog/Toto-Open-Base-1.0"),
            num_samples=m.get("num_samples", 256),
            samples_per_batch=m.get("samples_per_batch", 256),
            time_interval_seconds=m.get("time_interval_seconds", 3600),
            device=m.get("device"),
        )
    raise NotImplementedError(f"unknown model '{m['name']}'")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the ttbench forecasting benchmark.")
    parser.add_argument("--config", default=None, help="Path to a YAML config.")
    parser.add_argument("--out", default=None, help="Where to write the JSON result.")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    season_length = cfg["synthetic"]["season_length"]

    _series, windows = _build_dataset(cfg)
    model = _build_model(cfg, season_length)
    result = run_benchmark(model, windows, season_length, config=cfg)

    print(result.to_table())

    out = args.out or f"results/{result.model_name}.json"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(result), indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
