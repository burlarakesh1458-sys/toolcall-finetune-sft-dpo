"""Plot training curves and write the results table (markdown) from saved artefacts.

python -m toolcall_ft.report --config configs/cpu_small.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import load_config, resolve, write_json  # noqa: E402
from .scorer import score_all  # noqa: E402

STAGE_ORDER = [
    ("baseline_zeroshot", "Base, zero-shot prompt"),
    ("baseline_fewshot", "Base, engineered few-shot prompt"),
    ("sft", "SFT (LoRA)"),
    ("sft_dpo", "SFT + DPO"),
    ("groq_reference", "Reference: gpt-oss-120b (Groq API), few-shot"),
]
METRICS = [
    ("json_valid", "JSON valid"),
    ("tool_acc", "Function-name acc"),
    ("exact_match", "Exact match"),
    ("arg_f1", "Argument F1"),
    ("refusal_acc", "Call/decline acc"),
    ("refusal_recall", "Decline recall"),
    ("false_refusal", "False-decline rate"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def plot_sft(res_dir: Path) -> Path | None:
    rows = read_csv(res_dir / "sft_curve.csv")
    if not rows:
        return None
    tr = [(int(r["step"]), fnum(r["train_loss"])) for r in rows if fnum(r["train_loss"]) is not None]
    ev = [(int(r["step"]), fnum(r["eval_loss"])) for r in rows if fnum(r["eval_loss"]) is not None]
    fig, ax = plt.subplots(figsize=(7, 4))
    if tr:
        ax.plot(*zip(*tr, strict=True), label="train loss", color="#1f77b4")
    if ev:
        ax.plot(*zip(*ev, strict=True), "o-", label="val loss", color="#d62728")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy (completion tokens)")
    ax.set_title("SFT (LoRA) training curve — CPU small-scale run")
    ax.grid(alpha=0.3)
    ax.legend()
    out = res_dir / "plots" / "sft_loss.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_dpo(res_dir: Path) -> Path | None:
    rows = read_csv(res_dir / "dpo_curve.csv")
    if not rows:
        return None
    steps = [int(r["step"]) for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(steps, [fnum(r["loss"]) for r in rows], label="DPO loss")
    a1.set_xlabel("optimizer step")
    a1.set_ylabel("loss")
    a1.set_title("DPO loss")
    a1.grid(alpha=0.3)
    a2.plot(steps, [fnum(r["rewards_chosen"]) for r in rows], label="reward (chosen)")
    a2.plot(steps, [fnum(r["rewards_rejected"]) for r in rows], label="reward (rejected)")
    a2.plot(steps, [fnum(r["rewards_margins"]) for r in rows], label="margin")
    a2.set_xlabel("optimizer step")
    a2.set_title("DPO implicit rewards")
    a2.grid(alpha=0.3)
    a2.legend()
    fig.suptitle("DPO on top of SFT adapter — CPU small-scale run")
    out = res_dir / "plots" / "dpo_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def plot_metrics(res_dir: Path, metrics: dict[str, dict]) -> Path | None:
    stages = [(k, lbl) for k, lbl in STAGE_ORDER if k in metrics and k != "groq_reference"]
    if not stages:
        return None
    keys = ["json_valid", "tool_acc", "exact_match", "arg_f1", "refusal_acc"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(stages)
    for i, (k, lbl) in enumerate(stages):
        vals = [metrics[k].get(m, 0) * 100 for m in keys]
        bars = ax.bar([x + i * w for x in range(len(keys))], vals, w, label=lbl)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=7)
    ax.set_xticks([x + w * (len(stages) - 1) / 2 for x in range(len(keys))])
    ax.set_xticklabels([dict(METRICS)[m] for m in keys])
    ax.set_ylabel("%")
    ax.set_ylim(0, 110)
    ax.set_title("Held-out test metrics by stage — CPU small-scale run")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    out = res_dir / "plots" / "metrics_by_stage.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def results_table(metrics: dict[str, dict]) -> str:
    head = "| Stage | n | " + " | ".join(lbl for _, lbl in METRICS) + " |"
    sep = "|---|---:|" + "---:|" * len(METRICS)
    lines = [head, sep]
    for k, lbl in STAGE_ORDER:
        if k not in metrics:
            continue
        m = metrics[k]
        cells = []
        for key, _ in METRICS:
            v = m.get(key)
            cells.append("–" if v is None or v != v else f"{100 * v:.1f}%")
        lines.append(f"| {lbl} | {m.get('n', '')} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def groq_on_same_subset(res_dir: Path) -> dict | None:
    """The Groq reference may have been run on the whole test split; re-score it on exactly
    the example ids the local stages were evaluated on so the table compares like with like."""
    gp = res_dir / "predictions_groq_reference.jsonl"
    local = next(
        (
            res_dir / f"predictions_{k}.jsonl"
            for k, _ in STAGE_ORDER
            if (res_dir / f"predictions_{k}.jsonl").exists() and k != "groq_reference"
        ),
        None,
    )
    if not gp.exists() or local is None:
        return None
    ids = [json.loads(line)["id"] for line in local.read_text().splitlines()]
    groq = {(r := json.loads(line))["id"]: r for line in gp.read_text().splitlines()}
    if not all(i in groq for i in ids):
        return None
    m, _ = score_all([groq[i]["pred"] for i in ids], [groq[i]["gold"] for i in ids])
    full = json.loads((res_dir / "metrics_groq_reference.json").read_text())
    m.update(
        stage="groq_reference",
        model=full.get("model"),
        full_split_n=full.get("n"),
        full_split_exact_match=full.get("exact_match"),
    )
    write_json(res_dir / "metrics_groq_reference_subset.json", m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    res_dir = resolve(cfg["results_dir"])
    metrics = {}
    for k, _ in STAGE_ORDER:
        p = res_dir / f"metrics_{k}.json"
        if p.exists():
            metrics[k] = json.loads(p.read_text())
    sub = groq_on_same_subset(res_dir)
    if sub:
        metrics["groq_reference"] = sub
    plots = [plot_sft(res_dir), plot_dpo(res_dir), plot_metrics(res_dir, metrics)]
    table = results_table(metrics)
    (res_dir / "results_table.md").write_text(table + "\n")
    print(table)
    print("plots:", [str(p) for p in plots if p])


if __name__ == "__main__":
    main()
