"""Phase 2b: DPO on top of the SFT adapter (stacked training).

The SFT adapter is loaded twice: once as the trainable policy ("train") and once,
frozen, as the reference ("reference"). This works identically for a full-precision
base (CPU run) and a 4-bit QLoRA base (GPU run) and needs no merge.

    python -m toolcall_ft.train_dpo --config configs/cpu_small.yaml
"""

from __future__ import annotations

import argparse
import json
import time

from datasets import Dataset

from .common import load_config, load_model, load_tokenizer, resolve, set_seed, write_json
from .data import load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    from peft import PeftModel, prepare_model_for_kbit_training
    from trl import DPOConfig, DPOTrainer

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    p = cfg["dpo"]
    out_dir, res_dir = resolve(cfg["output_dir"]), resolve(cfg["results_dir"])

    rows = load_jsonl(out_dir / "dpo_pairs.jsonl")
    tk = cfg["model"].get("chat_template_kwargs")
    recs = []
    for r in rows:
        rec = {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        if tk:
            rec["chat_template_kwargs"] = tk
        recs.append(rec)
    ds = Dataset.from_list(recs)

    tok = load_tokenizer(cfg["model"]["base"])
    base = load_model(cfg, trainable=True)
    if cfg["model"].get("load_in_4bit"):
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, str(out_dir / "sft_adapter"), adapter_name="train", is_trainable=True)
    model.load_adapter(str(out_dir / "sft_adapter"), adapter_name="reference")
    model.set_adapter("train")

    dpo_args = DPOConfig(
        output_dir=str(out_dir / "dpo_ckpt"),
        beta=p["beta"],
        learning_rate=p["learning_rate"],
        num_train_epochs=p["num_train_epochs"],
        per_device_train_batch_size=p["per_device_train_batch_size"],
        gradient_accumulation_steps=p["gradient_accumulation_steps"],
        max_length=p["max_length"],
        max_prompt_length=p["max_prompt_length"],
        logging_steps=p["logging_steps"],
        lr_scheduler_type="cosine",
        gradient_checkpointing=p.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.1,
        save_strategy="steps",
        save_steps=p.get("save_steps", 8),
        save_total_limit=1,
        bf16=p.get("bf16", False),
        fp16=p.get("fp16", False),
        model_adapter_name="train",
        ref_adapter_name="reference",
        report_to=cfg.get("report_to", ["tensorboard"]),
        logging_dir=str(out_dir / "tb" / "dpo"),
        seed=cfg["seed"],
        dataloader_num_workers=0,
    )
    trainer = DPOTrainer(model=model, args=dpo_args, train_dataset=ds, processing_class=tok)
    t0 = time.time()
    from transformers.trainer_utils import get_last_checkpoint

    last = get_last_checkpoint(str(out_dir / "dpo_ckpt")) if (out_dir / "dpo_ckpt").exists() else None
    if last:
        print(f"resuming from {last}", flush=True)
    trainer.train(resume_from_checkpoint=last)
    elapsed = time.time() - t0

    # Save only the trained policy adapter.
    trainer.model.save_pretrained(str(out_dir / "dpo_tmp"), selected_adapters=["train"])
    import shutil

    src = out_dir / "dpo_tmp" / "train"
    dst = out_dir / "dpo_adapter"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src if src.exists() else out_dir / "dpo_tmp"), str(dst))
    shutil.rmtree(out_dir / "dpo_tmp", ignore_errors=True)

    hist = trainer.state.log_history
    write_json(res_dir / "dpo_log_history.json", hist)
    keys = ["loss", "rewards/chosen", "rewards/rejected", "rewards/accuracies", "rewards/margins", "learning_rate"]
    with open(res_dir / "dpo_curve.csv", "w") as f:
        f.write("step,epoch," + ",".join(k.replace("/", "_") for k in keys) + "\n")
        for h in hist:
            if "loss" in h:
                f.write(f"{h.get('step')},{h.get('epoch', '')}," + ",".join(str(h.get(k, "")) for k in keys) + "\n")
    summary = {"pairs": len(ds), "train_seconds": round(elapsed, 1), "global_steps": trainer.state.global_step}
    write_json(res_dir / "dpo_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
