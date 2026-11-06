"""Evaluate a stage (base few-shot / base zero-shot / sft / sft_dpo) on the held-out test set.

python -m toolcall_ft.evaluate --config configs/cpu_small.yaml --stage baseline_fewshot
"""

from __future__ import annotations

import argparse
import json
import time

from .common import generate, load_config, load_model, load_tokenizer, resolve, set_seed, write_json
from .data import load_jsonl
from .prompts import build_messages
from .scorer import score_all

STAGES = {
    # stage name: (adapter subdir or None, few_shot)
    "baseline_zeroshot": (None, False),
    "baseline_fewshot": (None, True),
    "sft": ("sft_adapter", False),
    "sft_dpo": ("dpo_adapter", False),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    adapter_sub, few_shot = STAGES[args.stage]
    out_dir = resolve(cfg["output_dir"])
    res_dir = resolve(cfg["results_dir"])
    limit = cfg["data"].get(f"{args.split}_limit")
    examples = load_jsonl(resolve(cfg["data"]["dir"]) / f"{args.split}.jsonl", limit)

    tok = load_tokenizer(cfg["model"]["base"])
    model = load_model(cfg, adapter=(out_dir / adapter_sub) if adapter_sub else None)
    ecfg = cfg["eval"]
    msgs = [build_messages(e, few_shot=few_shot) for e in examples]
    print(f"[eval] stage={args.stage} split={args.split} n={len(examples)}", flush=True)
    # Resumable, chunked generation: each finished chunk is appended to a cache file so an
    # interrupted run (OOM kill, preemption) continues where it stopped.
    cache = out_dir / "eval_cache" / f"{args.stage}_{args.split}.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    todo = [i for i, e in enumerate(examples) if e["id"] not in done]
    todo.sort(key=lambda i: len(json.dumps(msgs[i])))
    chunk = ecfg["batch_size"] * 2
    print(f"  {len(done)} cached, {len(todo)} to generate", flush=True)
    for c in range(0, len(todo), chunk):
        idx = todo[c : c + chunk]
        t0 = time.time()
        outs = generate(
            model,
            tok,
            [msgs[i] for i in idx],
            ecfg["max_new_tokens"],
            ecfg["batch_size"],
            template_kwargs=cfg["model"].get("chat_template_kwargs"),
        )
        dt = (time.time() - t0) / len(idx)
        with open(cache, "a") as f:
            for i, o in zip(idx, outs, strict=True):
                rec = {"id": examples[i]["id"], "pred": o[0], "sec": dt}
                done[rec["id"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  generated {len(done)}/{len(examples)}", flush=True)
    preds = [done[e["id"]]["pred"] for e in examples]
    elapsed = sum(done[e["id"]]["sec"] for e in examples)
    metrics, rows = score_all(preds, [e["target"] for e in examples])
    metrics["seconds"] = round(elapsed, 1)
    metrics["stage"] = args.stage
    metrics["model"] = cfg["model"]["base"]
    print(json.dumps(metrics, indent=2), flush=True)

    write_json(res_dir / f"metrics_{args.stage}.json", metrics)
    with open(res_dir / f"predictions_{args.stage}.jsonl", "w") as f:
        for e, p, r in zip(examples, preds, rows, strict=True):
            f.write(
                json.dumps(
                    {
                        "id": e["id"],
                        "query": e["query"],
                        "gold": e["target"],
                        "pred": p,
                        "exact_match": r["exact_match"],
                        "json_valid": r["json_valid"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
