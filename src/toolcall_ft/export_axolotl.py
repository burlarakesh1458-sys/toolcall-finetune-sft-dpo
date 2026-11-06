"""Export the processed splits / DPO pairs into the JSONL layouts the Axolotl YAMLs expect.

python -m toolcall_ft.export_axolotl [--dpo outputs/<run>/dpo_pairs.jsonl]
"""

from __future__ import annotations

import argparse
import json

from .common import resolve
from .data import load_jsonl
from .prompts import build_messages, target_str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="data/axolotl")
    ap.add_argument("--dpo", default=None)
    args = ap.parse_args()
    out = resolve(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, name in (("train", "sft_train"), ("val", "sft_val")):
        rows = load_jsonl(resolve(args.data_dir) / f"{split}.jsonl")
        with open(out / f"{name}.jsonl", "w") as f:
            for e in rows:
                msgs = build_messages(e) + [{"role": "assistant", "content": target_str(e["target"])}]
                f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} -> {out / (name + '.jsonl')}")
    if args.dpo:
        rows = load_jsonl(resolve(args.dpo))
        with open(out / "dpo_pairs.jsonl", "w") as f:
            for r in rows:
                f.write(
                    json.dumps(
                        {"messages": r["prompt"], "chosen": r["chosen"][0], "rejected": r["rejected"][0]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"wrote {len(rows)} DPO pairs -> {out / 'dpo_pairs.jsonl'}")


if __name__ == "__main__":
    main()
