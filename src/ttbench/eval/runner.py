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
        # A window may carry its own seasonality (BOOM: derived per frequency);
        # otherwise fall back to the dataset-level default.
        m = w.season_length if w.season_length is not None else season_length
        scores = all_metrics(
            y_true=w.target,
            y_pred=forecast.median,
            samples=forecast.samples,
            context=w.context,
            season_length=m,
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
    """Side-by-side metrics for several models evaluated on the SAME windows.

    Reports BOTH mean and median. The median is the robust headline: on
    heavy-tailed telemetry, per-window MASE can blow up when a window's
    in-context seasonal scale is ~0, and the *mean* of those ratios is dominated
    by a few outliers. The median (and the gluonts-style aggregate-then-divide
    MASE, a tracked refinement) are not. Improvement ratios use the median.
    """
    if not results:
        return "(no results)"
    base = results[0]

    def block(title: str, key: str) -> list[str]:
        header = f"{'model':<28}" + "".join(f"{m:>12}" for m in METRIC_NAMES)
        out = [title, header, "-" * len(header)]
        for r in results:
            agg = getattr(r, key)
            out.append(f"{r.model_name:<28}" + "".join(f"{agg[m]:>12.4f}" for m in METRIC_NAMES))
        return out

    lines = [
        f"windows={base.n_windows}  H={base.horizon}  ctx={base.context_length}",
        "",
        *block("mean (note: mean MASE is unstable on heavy-tailed data)", "aggregate"),
        "",
        *block("median (robust headline)", "aggregate_median"),
        "",
    ]
    for metric in ("mase", "crps"):
        ratios = " ".join(
            f"{r.model_name.split('_')[0]}="
            f"{r.aggregate_median[metric] / base.aggregate_median[metric]:.3f}"
            for r in results[1:]
        )
        if ratios:
            lines.append(f"median {metric} vs {base.model_name} (lower=better): {ratios}")
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


def _global_season_length(cfg: dict[str, Any]) -> int:
    """Dataset-level fallback seasonality (BOOM windows override this per-series)."""
    return int(cfg.get("synthetic", {}).get("season_length", 1))


def _build_dataset(cfg: dict[str, Any]):
    w = cfg["window"]
    if cfg["dataset"] == "synthetic":
        series, _anomalies = generate(SyntheticConfig(**cfg["synthetic"]))
        windows = make_windows(
            series,
            context_length=w["context_length"],
            horizon=w["horizon"],
            stride=w.get("stride"),
            max_windows_per_series=w.get("max_windows_per_series"),
            season_length=cfg["synthetic"]["season_length"],
        )
        return series, windows

    if cfg["dataset"] == "boom":
        # Lazy import: needs the 'toto' extra (datasets + huggingface_hub + gluonts).
        from ttbench.data.boom import freq_to_seasonality, load_boom
        from ttbench.data.windows import make_windows_varlen

        b = cfg.get("boom", {})
        series = load_boom(
            n_series=b.get("n_series", 50),
            max_len=b.get("max_len", 4096),
            freqs=b.get("freqs"),
            seed=b.get("seed", 0),
        )
        targets = [s.target for s in series]
        seasons = [freq_to_seasonality(s.freq) for s in series]
        windows = make_windows_varlen(
            targets,
            context_length=w["context_length"],
            horizon=w["horizon"],
            stride=w.get("stride"),
            max_windows_per_series=w.get("max_windows_per_series"),
            season_lengths=seasons,
        )
        return series, windows

    raise NotImplementedError(f"unknown dataset '{cfg['dataset']}'")


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
    season_length = _global_season_length(cfg)

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
