"""Config loading, seeding, model loading and batched generation."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def torch_dtype(name: str):
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(cfg: dict[str, Any], adapter: str | Path | None = None, trainable: bool = False):
    """Load the base model (optionally 4-bit) and optionally a LoRA adapter on top."""
    from transformers import AutoModelForCausalLM

    mcfg = cfg["model"]
    # ``train_dtype`` lets training hold the frozen base in a smaller dtype than evaluation
    # (the LoRA adapter itself is kept in fp32 by PEFT either way).
    dtype = mcfg.get("train_dtype", mcfg.get("dtype", "float32")) if trainable else mcfg.get("dtype", "float32")
    kwargs: dict[str, Any] = {"dtype": torch_dtype(dtype)}
    if mcfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype(mcfg.get("compute_dtype", "bfloat16")),
        )
        kwargs["device_map"] = "auto"
    if mcfg.get("attn_implementation"):
        kwargs["attn_implementation"] = mcfg["attn_implementation"]
    model = AutoModelForCausalLM.from_pretrained(mcfg["base"], **kwargs)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    if not trainable:
        model.eval()
    return model


def json_complete(text: str) -> bool:
    """True once ``text`` holds a complete top-level JSON object (string-aware brace
    matching). Used to stop decoding early instead of always running max_new_tokens."""
    s = text.lstrip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else ""
        s = s.lstrip()
    if not s.startswith("{"):
        return False
    depth, in_str, esc = 0, False, False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


class JsonStop:
    """transformers StoppingCriteria (duck-typed): per-row stop when the JSON closes."""

    def __init__(self, tok, prompt_len: int):
        self.tok, self.prompt_len = tok, prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        new = input_ids[:, self.prompt_len :]
        texts = self.tok.batch_decode(new, skip_special_tokens=True)
        return torch.tensor([json_complete(t) for t in texts], dtype=torch.bool, device=input_ids.device)


def generate(
    model,
    tok,
    batch_messages: list[list[dict[str, str]]],
    max_new_tokens: int = 96,
    batch_size: int = 8,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    log_every: int = 0,
    template_kwargs: dict[str, Any] | None = None,
) -> list[list[str]]:
    """Batched chat generation. Returns, per prompt, a list of ``num_return_sequences`` strings.
    Prompts are length-sorted to minimise padding; output order matches input order."""
    import torch

    tk = template_kwargs or {}
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **tk) for m in batch_messages]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out: list[list[str]] = [[] for _ in texts]
    device = next(model.parameters()).device
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": tok.pad_token_id,
        }
        from transformers import StoppingCriteriaList

        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([JsonStop(tok, enc["input_ids"].shape[1])])
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)
        with torch.no_grad():
            seqs = model.generate(**enc, **gen_kwargs)
        new = seqs[:, enc["input_ids"].shape[1] :]
        dec = tok.batch_decode(new, skip_special_tokens=True)
        for j, i in enumerate(idx):
            out[i] = [s.strip() for s in dec[j * num_return_sequences : (j + 1) * num_return_sequences]]
        if log_every and (b // batch_size) % log_every == 0:
            done = min(b + batch_size, len(order))
            print(f"  generated {done}/{len(order)}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
