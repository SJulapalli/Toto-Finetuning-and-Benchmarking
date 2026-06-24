"""Visualize the synthetic telemetry generator.

A sanity-check tool: plots a grid of generated series, marks the injected
anomalies (spikes + level-shifts), and optionally draws the context/horizon
boundary so you can see exactly what one forecasting window looks like.

    python scripts/visualize_synthetic.py
    python scripts/visualize_synthetic.py --n-series 6 --seed 1 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file backend by default; --show switches to interactive
import matplotlib.pyplot as plt  # noqa: E402

from ttbench.data import SyntheticConfig, generate  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Plot synthetic telemetry series.")
    p.add_argument("--n-series", type=int, default=6, help="how many series to plot")
    p.add_argument("--length", type=int, default=512)
    p.add_argument("--season-length", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--context-length", type=int, default=256,
                   help="draw the context/horizon split at this point (0 to disable)")
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--out", default="results/synthetic_preview.png")
    p.add_argument("--show", action="store_true", help="open an interactive window")
    args = p.parse_args(argv)

    # Generate enough series to cover the grid, then plot the first n.
    cfg = SyntheticConfig(
        n_series=max(args.n_series, 1),
        length=args.length,
        season_length=args.season_length,
        seed=args.seed,
    )
    series, anomalies = generate(cfg)

    n = min(args.n_series, series.shape[0])
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 2.4 * nrows), squeeze=False)
    t = range(args.length)

    for i in range(n):
        ax = axes[i // ncols][i % ncols]
        ax.plot(t, series[i], lw=0.9, color="#1f77b4", label="value")

        # Mark injected anomalies (spikes + level-shift onsets).
        anom_idx = anomalies[i].nonzero()[0]
        if anom_idx.size:
            ax.scatter(anom_idx, series[i][anom_idx], s=18, color="#d62728",
                       zorder=3, label="injected anomaly")

        # Show one forecasting window: context | horizon.
        if args.context_length > 0:
            c, h = args.context_length, args.horizon
            if c + h <= args.length:
                ax.axvline(c, color="gray", ls="--", lw=1)
                ax.axvspan(c, c + h, color="orange", alpha=0.12, label="forecast horizon")

        ax.set_title(f"series {i}", fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")

    # Hide any unused axes in the grid.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        f"Synthetic telemetry (seed={args.seed}, m={args.season_length})", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"wrote {out_path}")
    if args.show:
        matplotlib.use("MacOSX", force=True)
        plt.show()


if __name__ == "__main__":
    main()
