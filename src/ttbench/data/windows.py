"""Rolling-window slicing for forecasting evaluation.

Turns raw series into ``(context, target)`` pairs. Correctness invariant:
``target`` is always the ``horizon`` steps immediately following ``context``,
and ``context`` never contains future information. Future leakage here is the
classic silent benchmark bug, so this is deliberately tiny and explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Window:
    series_id: int
    context: np.ndarray  # shape (context_length,)
    target: np.ndarray   # shape (horizon,)


def make_windows(
    series: np.ndarray,
    context_length: int,
    horizon: int,
    stride: int | None = None,
    max_windows_per_series: int | None = None,
) -> list[Window]:
    """Slice every series into evaluation windows.

    Parameters
    ----------
    series : (n_series, length) array.
    context_length : timesteps of history handed to the model.
    horizon : timesteps to predict (held out).
    stride : gap between consecutive window starts. Defaults to ``horizon``
        (non-overlapping targets -- the usual rolling-origin setup).
    max_windows_per_series : cap windows per series to bound eval cost.
    """
    if series.ndim != 2:
        raise ValueError(f"expected (n_series, length), got shape {series.shape}")
    stride = stride or horizon
    n_series, length = series.shape
    need = context_length + horizon
    if length < need:
        raise ValueError(
            f"series length {length} < context_length+horizon ({need})"
        )

    windows: list[Window] = []
    for sid in range(n_series):
        starts = range(0, length - need + 1, stride)
        if max_windows_per_series is not None:
            starts = list(starts)[:max_windows_per_series]
        for start in starts:
            ctx_end = start + context_length
            windows.append(
                Window(
                    series_id=sid,
                    context=series[sid, start:ctx_end].copy(),
                    target=series[sid, ctx_end:ctx_end + horizon].copy(),
                )
            )
    return windows
