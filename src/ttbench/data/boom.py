"""Loader for Datadog's BOOM observability benchmark.

BOOM is the real evaluation target: ~350M observations of production telemetry,
stored on HuggingFace as one ``datasets.save_to_disk`` folder per series group
(GIFT-Eval format), named ``ds-<id>-<freq>`` with fields:

    start: ISO timestamp (str)   item_id: str   freq: pandas code (str)
    target: 1-D float array (or 2-D for multivariate series)

We never pull the whole thing (8k+ folders). We list the folders, sample a
deterministic subset, and download only those with ``allow_patterns``. Exposes
the same shape as the synthetic generator so the harness is dataset-agnostic.

Requires the 'toto' extra (datasets + huggingface_hub + gluonts):

    pip install -e ".[toto]"
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np

REPO = "Datadog/BOOM"


@dataclass
class BoomSeries:
    item_id: str
    freq: str
    target: np.ndarray  # 1-D


def _require_deps():
    try:
        from datasets import load_from_disk
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The BOOM loader needs the 'toto' extra (datasets + huggingface_hub). "
            'Install it with:\n    pip install -e ".[toto]"'
        ) from e
    return HfApi, snapshot_download, load_from_disk


def freq_to_seasonality(freq: str) -> int:
    """Canonical seasonality for a pandas frequency, via gluonts.

    Matches GIFT-Eval / BOOM methodology (e.g. minute -> 1440, 5-min -> 288,
    10-sec -> 360, hour -> 24). Falls back to 1 (no seasonality) on anything
    gluonts can't parse.
    """
    try:
        from gluonts.time_feature import get_seasonality

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence pandas freq-deprecation noise
            return int(get_seasonality(freq))
    except Exception:
        return 1


def list_series_dirs(repo: str = REPO) -> list[str]:
    """All ``ds-*`` series-group folder names in the BOOM repo."""
    HfApi, _, _ = _require_deps()
    info = HfApi().dataset_info(repo)
    dirs = {
        s.rfilename.split("/")[0]
        for s in (info.siblings or [])
        if s.rfilename.startswith("ds-")
    }
    return sorted(dirs)


def _freq_of(dirname: str) -> str:
    # "ds-1003-10S" -> "10S"
    return dirname.rsplit("-", 1)[1]


def load_boom(
    n_series: int = 50,
    max_len: int | None = 4096,
    freqs: list[str] | None = None,
    repo: str = REPO,
    seed: int = 0,
) -> list[BoomSeries]:
    """Download and load a deterministic subset of BOOM series.

    Parameters
    ----------
    n_series : how many series-group folders to sample.
    max_len : keep only the most recent ``max_len`` points per series (bounds
        memory + CPU inference cost). None keeps full length.
    freqs : optional allow-list of frequency codes (e.g. ["5T", "10S"]).
    seed : controls the deterministic subset selection.
    """
    _HfApi, snapshot_download, load_from_disk = _require_deps()

    dirs = list_series_dirs(repo)
    if freqs:
        wanted = set(freqs)
        dirs = [d for d in dirs if _freq_of(d) in wanted]
    if not dirs:
        raise ValueError(f"no BOOM series matched freqs={freqs}")

    rng = np.random.default_rng(seed)
    rng.shuffle(dirs)
    selected = dirs[:n_series]

    # Pull only the chosen folders.
    local = snapshot_download(
        repo, repo_type="dataset", allow_patterns=[f"{d}/*" for d in selected]
    )

    out: list[BoomSeries] = []
    for d in selected:
        ds = load_from_disk(os.path.join(local, d))
        for row in ds:
            target = np.asarray(row["target"], dtype=float)
            freq = row["freq"]
            item_id = str(row["item_id"])

            # Multivariate series store target as (variates, time): split each
            # variate into its own univariate series for our 1-D harness.
            variates = target if target.ndim > 1 else target[None, :]
            for v, vt in enumerate(variates):
                vt = vt[-max_len:] if max_len and vt.shape[0] > max_len else vt
                sid = item_id if variates.shape[0] == 1 else f"{item_id}:{v}"
                out.append(BoomSeries(item_id=sid, freq=freq, target=vt.copy()))

    return out
