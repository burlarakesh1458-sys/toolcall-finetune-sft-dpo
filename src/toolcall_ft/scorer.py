"""Scoring of model outputs against gold tool calls.

Metrics (all in [0, 1]):
- json_valid      : output is a single JSON object with keys {"tool", "arguments"}
                    (``arguments`` an object, ``tool`` a string or null). Markdown code
                    fences are tolerated; any other surrounding text is not.
- tool_acc        : predicted tool name == gold tool name (null == null counts).
- exact_match     : tool name and the full argument dict match after normalisation.
- arg_f1          : micro F1 over (key, normalised value) pairs, computed on examples
                    whose gold label is a call.
- refusal_acc     : the call/decline decision is correct (over all examples).
- refusal_recall  : fraction of gold "no tool" examples that the model declined.
- false_refusal   : fraction of gold call examples the model wrongly declined.
"""

from __future__ import annotations

import json
import re
from typing import Any

FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_output(text: str) -> dict[str, Any] | None:
    """Strictly parse a model output into {"tool", "arguments"} or return None."""
    if text is None:
        return None
    s = text.strip()
    m = FENCE_RE.match(s)
    if m:
        s = m.group(1).strip()
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or set(obj.keys()) != {"tool", "arguments"}:
        return None
    if not (obj["tool"] is None or isinstance(obj["tool"], str)):
        return None
    if not isinstance(obj["arguments"], dict):
        return None
    if obj["tool"] is None and obj["arguments"]:
        return None
    return obj


def norm_value(v: Any) -> Any:
    """Canonical form for comparing argument values."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        f = float(v)
        return int(f) if f.is_integer() else round(f, 6)
    if isinstance(v, str):
        s = re.sub(r"\s+", " ", v.strip()).lower()
        # numeric strings compare equal to numbers ("5" == 5) only when purely numeric
        return s
    if isinstance(v, list):
        return tuple(norm_value(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, norm_value(x)) for k, x in v.items()))
    return v


def arg_pairs(args: dict[str, Any]) -> set[tuple[str, Any]]:
    return {(k, json.dumps(norm_value(v), sort_keys=True, default=str)) for k, v in args.items()}


def score_one(pred_text: str, gold: dict[str, Any]) -> dict[str, Any]:
    pred = parse_output(pred_text)
    gold_call = gold["tool"] is not None
    res: dict[str, Any] = {
        "json_valid": pred is not None,
        "gold_is_call": gold_call,
        "pred_is_call": bool(pred is not None and pred["tool"] is not None),
        "tool_correct": False,
        "exact_match": False,
        "tp": 0,
        "fp": 0,
        "fn": len(gold["arguments"]) if gold_call else 0,
    }
    if pred is None:
        # Invalid output: treated as "no decision" -> wrong for both calls and declines.
        res["decision_correct"] = False
        return res
    res["decision_correct"] = res["pred_is_call"] == gold_call
    res["tool_correct"] = pred["tool"] == gold["tool"]
    gp, pp = arg_pairs(gold["arguments"]), arg_pairs(pred["arguments"])
    res["exact_match"] = res["tool_correct"] and gp == pp
    if gold_call:
        if res["tool_correct"]:
            res["tp"] = len(gp & pp)
            res["fp"] = len(pp - gp)
            res["fn"] = len(gp - pp)
        else:
            res["fp"] = len(pp)
    return res


def reward(pred_text: str, gold: dict[str, Any]) -> float:
    """Scalar used to rank DPO samples: invalid JSON < wrong decision < partial < exact."""
    r = score_one(pred_text, gold)
    if not r["json_valid"]:
        return -1.0
    if r["exact_match"]:
        return 1.0
    if not r["decision_correct"]:
        return -0.5
    if not r["tool_correct"]:
        return -0.25
    tp, fp, fn = r["tp"], r["fp"], r["fn"]
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return 0.5 * f1


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    if n == 0:
        return {}
    calls = [r for r in rows if r["gold_is_call"]]
    refs = [r for r in rows if not r["gold_is_call"]]
    tp = sum(r["tp"] for r in calls)
    fp = sum(r["fp"] for r in calls)
    fn = sum(r["fn"] for r in calls)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": n,
        "json_valid": sum(r["json_valid"] for r in rows) / n,
        "exact_match": sum(r["exact_match"] for r in rows) / n,
        "tool_acc": sum(r["tool_correct"] for r in rows) / n,
        "arg_f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
        "refusal_acc": sum(r["decision_correct"] for r in rows) / n,
        "refusal_recall": (sum(r["json_valid"] and not r["pred_is_call"] for r in refs) / len(refs))
        if refs
        else float("nan"),
        "false_refusal": (sum(r["json_valid"] and not r["pred_is_call"] for r in calls) / len(calls))
        if calls
        else float("nan"),
        "exact_match_calls": (sum(r["exact_match"] for r in calls) / len(calls)) if calls else float("nan"),
    }


def score_all(preds: list[str], golds: list[dict[str, Any]]) -> tuple[dict[str, float], list[dict]]:
    rows = [score_one(p, g) for p, g in zip(preds, golds, strict=True)]
    return aggregate(rows), rows
