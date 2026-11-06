"""Optional strong-model reference: the same few-shot prompt sent to a large hosted model
through Groq's OpenAI-compatible API, scored with the same scorer on the same test set.

    GROQ_API_KEY=... python -m toolcall_ft.groq_reference --config configs/cpu_small.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .common import load_config, resolve, write_json
from .data import load_jsonl
from .prompts import build_messages
from .scorer import score_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    ap.add_argument("--limit", type=int, default=None, help="default: the config's test_limit")
    ap.add_argument("--full", action="store_true", help="run on the whole test split")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from openai import OpenAI

    cfg = load_config(args.config)
    res_dir = resolve(cfg["results_dir"])
    limit = None if args.full else (args.limit or cfg["data"].get("test_limit"))
    examples = load_jsonl(resolve(cfg["data"]["dir"]) / "test.jsonl", limit)
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("GROQ_API_KEY", "proxy"))
    res_dir.mkdir(parents=True, exist_ok=True)
    pred_path = res_dir / "predictions_groq_reference.jsonl"
    done: dict[str, dict] = {}
    if pred_path.exists():  # resumable: keep predictions already written
        for line in pred_path.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    todo = [e for e in examples if e["id"] not in done]
    print(f"[groq] {len(done)} cached, {len(todo)} to run with {args.workers} workers", flush=True)

    def call(ex):
        for attempt in range(8):
            try:
                r = client.chat.completions.create(
                    model=args.model,
                    messages=build_messages(ex, few_shot=True),
                    temperature=0,
                    max_tokens=2048,
                )
                return (r.choices[0].message.content or "").strip()
            except Exception as e:  # rate limits etc.
                wait = 10 * (attempt + 1)
                print(f"  retry {attempt + 1} after error: {str(e)[:120]} (sleep {wait}s)", flush=True)
                time.sleep(wait)
        return None  # not recorded -> retried on the next (resumed) run

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(pred_path, "a") as f:
        futs = {pool.submit(call, ex): ex for ex in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            ex = futs[fut]
            out = fut.result()
            if out is None:
                continue
            rec = {"id": ex["id"], "gold": ex["target"], "pred": out}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            done[ex["id"]] = rec
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    missing = [e["id"] for e in examples if e["id"] not in done]
    if missing:
        print(f"[groq] {len(missing)} examples still missing after retries; rerun to resume", flush=True)
        return
    preds = [done[e["id"]]["pred"] for e in examples]
    metrics, rows = score_all(preds, [e["target"] for e in examples])
    metrics.update(stage="groq_reference", model=args.model)
    print(json.dumps(metrics, indent=2))
    write_json(res_dir / "metrics_groq_reference.json", metrics)


if __name__ == "__main__":
    main()
