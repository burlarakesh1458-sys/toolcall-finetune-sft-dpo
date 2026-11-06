"""Phase 1: supervised fine-tuning with LoRA / QLoRA via TRL's SFTTrainer.

python -m toolcall_ft.train_sft --config configs/cpu_small.yaml
"""

from __future__ import annotations

import argparse
import json
import time

from datasets import Dataset

from .common import load_config, load_model, load_tokenizer, resolve, set_seed, write_json
from .data import load_jsonl
from .prompts import to_sft_record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    d, s = cfg["data"], cfg["sft"]
    out_dir = resolve(cfg["output_dir"])
    res_dir = resolve(cfg["results_dir"])

    train = load_jsonl(resolve(d["dir"]) / "train.jsonl", d.get("train_limit"))
    val = load_jsonl(resolve(d["dir"]) / "val.jsonl", d.get("val_limit"))
    tk = cfg["model"].get("chat_template_kwargs")
    train_ds = Dataset.from_list([to_sft_record(e, tk) for e in train])
    val_ds = Dataset.from_list([to_sft_record(e, tk) for e in val])

    tok = load_tokenizer(cfg["model"]["base"])
    tok.padding_side = "right"  # training pads on the right
    model = load_model(cfg, trainable=True)
    if cfg["model"].get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=s.get("gradient_checkpointing", True))

    lora = LoraConfig(
        r=s["lora_r"],
        lora_alpha=s["lora_alpha"],
        lora_dropout=s["lora_dropout"],
        target_modules=s["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    sft_args = SFTConfig(
        output_dir=str(out_dir / "sft_ckpt"),
        num_train_epochs=s["num_train_epochs"],
        learning_rate=s["learning_rate"],
        per_device_train_batch_size=s["per_device_train_batch_size"],
        per_device_eval_batch_size=s["per_device_train_batch_size"],
        gradient_accumulation_steps=s["gradient_accumulation_steps"],
        warmup_ratio=s["warmup_ratio"],
        lr_scheduler_type=s["lr_scheduler_type"],
        logging_steps=s["logging_steps"],
        eval_strategy="steps",
        eval_steps=s["eval_steps"],
        save_strategy="steps",  # small LoRA checkpoints -> resumable after interruption
        save_steps=s.get("save_steps", 10),
        save_total_limit=1,
        max_length=d["max_length"],
        completion_only_loss=True,  # loss only on the assistant JSON
        gradient_checkpointing=s.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=s.get("bf16", False),
        fp16=s.get("fp16", False),
        report_to=cfg.get("report_to", ["tensorboard"]),
        logging_dir=str(out_dir / "tb" / "sft"),
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        dataloader_num_workers=0,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        peft_config=lora,
    )
    trainer.model.print_trainable_parameters()
    t0 = time.time()
    from transformers.trainer_utils import get_last_checkpoint

    last = get_last_checkpoint(str(out_dir / "sft_ckpt")) if (out_dir / "sft_ckpt").exists() else None
    if last:
        print(f"resuming from {last}", flush=True)
    trainer.train(resume_from_checkpoint=last)
    elapsed = time.time() - t0
    trainer.model.save_pretrained(str(out_dir / "sft_adapter"))
    tok.save_pretrained(str(out_dir / "sft_adapter"))

    # Persist the training curve as CSV/JSON (TensorBoard logs are written too).
    hist = trainer.state.log_history
    write_json(res_dir / "sft_log_history.json", hist)
    with open(res_dir / "sft_curve.csv", "w") as f:
        f.write("step,epoch,train_loss,eval_loss,learning_rate,mean_token_accuracy\n")
        for h in hist:
            if "loss" in h or "eval_loss" in h:
                f.write(
                    f"{h.get('step')},{h.get('epoch', '')},{h.get('loss', '')},{h.get('eval_loss', '')},"
                    f"{h.get('learning_rate', '')},{h.get('mean_token_accuracy', h.get('eval_mean_token_accuracy', ''))}\n"
                )
    summary = {
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "train_seconds": round(elapsed, 1),
        "global_steps": trainer.state.global_step,
        "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in trainer.model.parameters()),
    }
    write_json(res_dir / "sft_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
