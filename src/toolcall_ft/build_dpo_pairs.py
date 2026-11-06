"""Phase 2a: sample several outputs per training prompt from the SFT model, score each
with the task scorer, and build (chosen, rejected) preference pairs automatically.

Labelling rule (correctness-based, no human or LLM judge):
  - reward(): exact match 1.0 > right tool, partial args 0.5*F1 > wrong tool -0.25
              > wrong call/decline decision -0.5 > invalid JSON -1.0
  - chosen  = highest-reward sample; if none is an exact match and ``gold_anchor`` is
              on, chosen = the gold answer (so we never prefer a wrong answer).
  - rejected = lowest-reward sample, kept only if its reward < chosen's reward.
  Prompts where every sample is already exact produce no pair (no learning signal).

    python -m toolcall_ft.build_dpo_pairs --config configs/cpu_small.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter

from .common import generate, load_config, load_model, load_tokenizer, resolve, set_seed, write_json
from .data import load_jsonl
from .prompts import build_messages, target_str
from .scorer import reward


def build_pairs(examples, samples, gold_anchor: bool = True):
    pairs, stats = [], Counter()
    for ex, outs in zip(examples, samples, strict=True):
        scored = sorted(((reward(o, ex["target"]), o) for o in outs), key=lambda x: -x[0])
        best_r, best = scored[0]
        worst_r, worst = scored[-1]
        stats["all_exact" if worst_r == 1.0 else "has_error"] += 1
        if best_r < 1.0 and gold_anchor:
            best_r, best = 1.0, target_str(ex["target"])
            stats["gold_anchored"] += 1
        if worst_r >= best_r:
            stats["no_pair"] += 1
            continue
        pairs.append(
            {
                "id": ex["id"],
                "prompt": build_messages(ex, few_shot=False),
                "chosen": [{"role": "assistant", "content": best}],
                "rejected": [{"role": "assistant", "content": worst}],
                "chosen_reward": best_r,
                "rejected_reward": worst_r,
            }
        )
    stats["pairs"] = len(pairs)
    return pairs, dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    p = cfg["dpo"]
    out_dir, res_dir = resolve(cfg["output_dir"]), resolve(cfg["results_dir"])

    # Use training prompts only (never val/test). Take prompts not seen... SFT saw the
    # train subset; we sample from the train split beyond the SFT subset when available
    # so DPO prompts are fresh for the SFT model.
    all_train = load_jsonl(resolve(cfg["data"]["dir"]) / "train.jsonl")
    start = cfg["data"].get("train_limit") or 0
    pool = all_train[start:] if len(all_train) - start >= p["n_prompts"] else all_train
    rng = random.Random(cfg["seed"])
    examples = rng.sample(pool, p["n_prompts"])

    tok = load_tokenizer(cfg["model"]["base"])
    model = load_model(cfg, adapter=out_dir / "sft_adapter")
    msgs = [build_messages(e, few_shot=False) for e in examples]
    t0 = time.time()
    samples = generate(
        model,
        tok,
        msgs,
        max_new_tokens=cfg["eval"]["max_new_tokens"],
        batch_size=max(1, cfg["eval"]["batch_size"] // p["n_samples"]),
        do_sample=True,
        temperature=p["temperature"],
        top_p=p["top_p"],
        num_return_sequences=p["n_samples"],
        log_every=5,
        template_kwargs=cfg["model"].get("chat_template_kwargs"),
    )
    pairs, stats = build_pairs(examples, samples, p.get("gold_anchor", True))
    stats["sampling_seconds"] = round(time.time() - t0, 1)
    stats["prompts"] = len(examples)
    stats["samples_per_prompt"] = p["n_samples"]
    rewards = [reward(o, e["target"]) for e, outs in zip(examples, samples, strict=True) for o in outs]
    stats["sample_exact_rate"] = round(sum(r == 1.0 for r in rewards) / len(rewards), 4)
    stats["sample_invalid_json_rate"] = round(sum(r == -1.0 for r in rewards) / len(rewards), 4)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dpo_pairs.jsonl", "w") as f:
        for pr in pairs:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")
    # A small human-readable sample of pairs is committed for the write-up.
    with open(res_dir / "dpo_pairs_sample.jsonl", "w") as f:
        for pr in pairs[:15]:
            f.write(
                json.dumps(
                    {
                        "id": pr["id"],
                        "query": pr["prompt"][-1]["content"].split("User request:\n")[-1],
                        "chosen": pr["chosen"][0]["content"],
                        "rejected": pr["rejected"][0]["content"],
                        "chosen_reward": pr["chosen_reward"],
                        "rejected_reward": pr["rejected_reward"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    write_json(res_dir / "dpo_pairs_stats.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
