"""Rolling-window slicing for forecasting evaluation.

Turns raw series into ``(context, target)`` pairs. Correctness invariant:
``target`` is always the ``horizon`` steps immediately following ``context``,
and ``context`` never contains future information. Future leakage here is the
classic silent benchmark bug, so this is deliberately tiny and explicit.

Two entry points:
- ``make_windows`` for a rectangular ``(n_series, length)`` array (synthetic).
- ``make_windows_varlen`` for a list of variable-length 1-D series (BOOM),
  where each series may carry its own seasonality (derived from its frequency).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class Window:
    series_id: int
    context: np.ndarray            # shape (context_length,)
    target: np.ndarray            # shape (horizon,)
    season_length: int | None = None  # per-window MASE seasonality; None -> use runner default


def _slice_one(
    series_1d: np.ndarray,
    series_id: int,
    context_length: int,
    horizon: int,
    stride: int,
    season_length: int | None,
    max_windows: int | None,
) -> list[Window]:
    """Slice a single 1-D series into windows (shared by both entry points)."""
    length = series_1d.shape[0]
    need = context_length + horizon
    windows: list[Window] = []
    starts = range(0, length - need + 1, stride)
    if max_windows is not None:
        starts = list(starts)[:max_windows]
    for start in starts:
        ctx_end = start + context_length
        windows.append(
            Window(
                series_id=series_id,
                context=series_1d[start:ctx_end].copy(),
                target=series_1d[ctx_end:ctx_end + horizon].copy(),
                season_length=season_length,
            )
        )
    return windows


def make_windows(
    series: np.ndarray,
    context_length: int,
    horizon: int,
    stride: int | None = None,
    max_windows_per_series: int | None = None,
    season_length: int | None = None,
) -> list[Window]:
    """Slice every row of a rectangular ``(n_series, length)`` array.

    stride defaults to ``horizon`` (non-overlapping targets). Raises if the
    series are too short -- a rectangular dataset is assumed uniform, so a short
    length is a config error, not something to silently skip.
    """
    if series.ndim != 2:
        raise ValueError(f"expected (n_series, length), got shape {series.shape}")
    stride = stride or horizon
    n_series, length = series.shape
    need = context_length + horizon
    if length < need:
        raise ValueError(f"series length {length} < context_length+horizon ({need})")

    windows: list[Window] = []
    for sid in range(n_series):
        windows.extend(
            _slice_one(series[sid], sid, context_length, horizon, stride,
                       season_length, max_windows_per_series)
        )
    return windows


def make_windows_varlen(
    series_list: Sequence[np.ndarray],
    context_length: int,
    horizon: int,
    stride: int | None = None,
    max_windows_per_series: int | None = None,
    season_lengths: Sequence[int] | None = None,
) -> list[Window]:
    """Slice a list of variable-length 1-D series.

    Series shorter than ``context_length + horizon`` are SKIPPED (real datasets
    are heterogeneous; a short series is expected, not an error). ``season_lengths``
    gives a per-series MASE seasonality (e.g. derived from each series' frequency).
    """
    stride = stride or horizon
    need = context_length + horizon
    windows: list[Window] = []
    for sid, s in enumerate(series_list):
        s = np.asarray(s, float)
        if s.shape[0] < need:
            continue
        m = season_lengths[sid] if season_lengths is not None else None
        windows.extend(
            _slice_one(s, sid, context_length, horizon, stride, m, max_windows_per_series)
        )
    return windows
