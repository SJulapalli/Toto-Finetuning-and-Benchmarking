"""Visualize real BOOM telemetry and contrast its distribution with synthetic.

Two figures' worth of insight in one image:
  - top row: a few example BOOM series (what real telemetry looks like),
  - bottom: pooled distribution comparisons, synthetic vs BOOM, of
      (A) per-series z-scored VALUES   -> overall signal shape,
      (B) per-series-scaled INCREMENTS -> smoothness / tail heaviness (log y).

Because scales differ across series, everything is normalized per series before
pooling, so we compare distribution *shape*, not magnitude.

    python scripts/visualize_boom.py --n-series 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ttbench.data import SyntheticConfig, generate  # noqa: E402
from ttbench.data.boom import load_boom  # noqa: E402

MAX_POINTS = 60_000  # cap pooled samples for histogram speed


def _pool_zscored_values(series_list: list[np.ndarray], rng) -> np.ndarray:
    vals = []
    for s in series_list:
        s = np.asarray(s, float)
        sd = s.std()
        if sd > 0:
            vals.append((s - s.mean()) / sd)
    pooled = np.concatenate(vals) if vals else np.array([])
    if pooled.size > MAX_POINTS:
        pooled = rng.choice(pooled, MAX_POINTS, replace=False)
    return pooled


def _pool_scaled_increments(series_list: list[np.ndarray], rng) -> np.ndarray:
    """First differences scaled by the series' own level std.

    Units of "fraction of a typical level fluctuation" -> heavy-tailed jumps in
    BOOM show up as fat tails vs the near-Gaussian synthetic increments.
    """
    diffs = []
    for s in series_list:
        s = np.asarray(s, float)
        sd = s.std()
        if sd > 0:
            diffs.append(np.diff(s) / sd)
    pooled = np.concatenate(diffs) if diffs else np.array([])
    if pooled.size > MAX_POINTS:
        pooled = rng.choice(pooled, MAX_POINTS, replace=False)
    return pooled


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Compare synthetic vs BOOM distributions.")
    p.add_argument("--n-series", type=int, default=8)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--freqs", nargs="*", default=None, help="e.g. 5T 10S")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/boom_vs_synthetic.png")
    args = p.parse_args(argv)

    rng = np.random.default_rng(args.seed)

    print("loading BOOM subset ...", flush=True)
    boom = load_boom(n_series=args.n_series, max_len=args.max_len,
                     freqs=args.freqs, seed=args.seed)
    boom_targets = [b.target for b in boom]

    # Synthetic sample sized comparably.
    syn, _ = generate(SyntheticConfig(n_series=max(len(boom), 1),
                                      length=args.max_len, seed=args.seed))
    syn_targets = [syn[i] for i in range(syn.shape[0])]

    n_examples = min(4, len(boom_targets))
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, max(n_examples, 2))

    # Top row: example BOOM series.
    for i in range(n_examples):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(boom_targets[i], lw=0.7, color="#d62728")
        ax.set_title(f"BOOM {boom[i].item_id} ({boom[i].freq})", fontsize=8)
        ax.tick_params(labelsize=6)

    # Bottom-left: z-scored value distribution.
    axv = fig.add_subplot(gs[1, 0])
    bins = np.linspace(-4, 4, 80)
    axv.hist(_pool_zscored_values(syn_targets, rng), bins=bins, density=True,
             histtype="step", lw=1.5, color="#1f77b4", label="synthetic")
    axv.hist(_pool_zscored_values(boom_targets, rng), bins=bins, density=True,
             histtype="step", lw=1.5, color="#d62728", label="BOOM")
    axv.set_title("z-scored value distribution", fontsize=9)
    axv.set_xlabel("(x - mean) / std", fontsize=8)
    axv.legend(fontsize=8)

    # Bottom-right: scaled increment distribution (log y reveals tails).
    axd = fig.add_subplot(gs[1, 1])
    dbins = np.linspace(-3, 3, 100)
    axd.hist(_pool_scaled_increments(syn_targets, rng), bins=dbins, density=True,
             histtype="step", lw=1.5, color="#1f77b4", label="synthetic")
    axd.hist(_pool_scaled_increments(boom_targets, rng), bins=dbins, density=True,
             histtype="step", lw=1.5, color="#d62728", label="BOOM")
    axd.set_yscale("log")
    axd.set_title("scaled increment distribution (log y -> tails)", fontsize=9)
    axd.set_xlabel("diff(x) / std(x)", fontsize=8)
    axd.legend(fontsize=8)

    fig.suptitle("Synthetic vs BOOM telemetry", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
