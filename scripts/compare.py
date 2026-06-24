"""Compare several models on the SAME windows and print a side-by-side table.

Fairness matters here: every model is scored on identical (context, target)
windows, so the MASE denominators are shared and the comparison is apples-to-
apples. Usage:

    python scripts/compare.py --config configs/synthetic_compare.yaml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from ttbench.eval.runner import (
    _build_dataset,
    _build_model,
    _global_season_length,
    compare_table,
    run_benchmark,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare models on identical windows.")
    parser.add_argument("--config", required=True, help="YAML config with a `models` list.")
    parser.add_argument("--out", default="results/comparison.json")
    args = parser.parse_args(argv)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    season_length = _global_season_length(cfg)
    _series, windows = _build_dataset(cfg)  # built ONCE, shared by all models

    model_specs = cfg.get("models") or [cfg["model"]]
    results = []
    for spec in model_specs:
        model = _build_model({**cfg, "model": spec}, season_length)
        print(f"running {model.name} over {len(windows)} windows ...", flush=True)
        results.append(run_benchmark(model, windows, season_length, config=cfg))

    print("\n" + compare_table(results))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
