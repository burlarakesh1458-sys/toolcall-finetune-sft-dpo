"""Download + build the processed dataset.

python -m toolcall_ft.prepare_data --config configs/cpu_small.yaml
"""

from __future__ import annotations

import argparse
import json

from .common import load_config, resolve
from .data import build_dataset, download_raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--raw", default=None, help="path to an already-downloaded raw json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    raw = resolve(args.raw) if args.raw else download_raw(resolve(d["raw_path"]))
    stats = build_dataset(
        raw,
        resolve(d["dir"]),
        total=d["total"],
        refusal_frac=d["refusal_frac"],
        val_size=d["val_size"],
        test_size=d["test_size"],
        max_tools_chars=d["max_tools_chars"],
        max_per_tool=d["max_per_tool"],
        synthetic_negative_frac=d["synthetic_negative_frac"],
        seed=cfg["seed"],
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
