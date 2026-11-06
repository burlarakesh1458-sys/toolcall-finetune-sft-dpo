"""Download, parse, clean, validate, dedupe and split glaive-function-calling-v2.

Each raw record has a ``system`` string (zero or more JSON function schemas) and a
``chat`` transcript. We keep only the *first* user turn and the *first* assistant
turn and turn them into a single-step task:

    input : list of tool schemas + user request
    target: {"tool": "<name>", "arguments": {...}}   # a tool applies
            {"tool": null, "arguments": {}}         # no tool applies -> decline

Examples where the first assistant turn is a clarifying question (neither a call nor
a decline) are dropped because their label is ambiguous for this task.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests

DATASET_ID = "glaiveai/glaive-function-calling-v2"
DATASET_REVISION = "e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac"
DATASET_FILE = "glaive-function-calling-v2.json"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{DATASET_FILE}"

TOOLS_MARKER = "Use them if required -"
FUNCCALL_RE = re.compile(
    r"^<functioncall>\s*\{\s*\"name\"\s*:\s*\"(?P<name>[^\"]+)\"\s*,\s*\"arguments\"\s*:\s*'(?P<args>.*)'\s*\}\s*$",
    re.DOTALL,
)
REFUSAL_RE = re.compile(
    r"\b(sorry|apologi[sz]e|unable to|not able to|can't|cannot|can not|"
    r"don't have the (ability|capability|capabilities|functionality)|"
    r"do not have the (ability|capability|capabilities|functionality)|"
    r"outside (the scope|my capabilities)|beyond my capabilities|"
    r"my current (capabilities|functions?) (are|is|allows?) )",
    re.IGNORECASE,
)
EOT = "<\u200b|endoftext|>"  # glaive uses a zero-width-space-poisoned EOS marker

JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


# --------------------------------------------------------------------------- download
def download_raw(dest: Path, max_bytes: int | None = None, timeout: int = 300) -> Path:
    """Download the pinned raw JSON. ``max_bytes`` fetches only a prefix (HTTP Range);
    :func:`iter_raw_records` tolerates the truncated tail."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    headers = {"Range": f"bytes=0-{max_bytes - 1}"} if max_bytes else {}
    with requests.get(DATASET_URL, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return dest


def iter_raw_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream objects out of a (possibly truncated) top-level JSON array."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    dec = json.JSONDecoder()
    i = text.find("[") + 1
    n = len(text)
    while i < n:
        while i < n and text[i] in " \r\n\t,":
            i += 1
        if i >= n or text[i] == "]":
            break
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break  # truncated tail
        if isinstance(obj, dict):
            yield obj


# --------------------------------------------------------------------------- parsing
def parse_tools(system: str) -> list[dict[str, Any]] | None:
    """Return the list of function schemas in a glaive system prompt (None if absent)."""
    idx = system.find(TOOLS_MARKER)
    if idx < 0:
        return None
    blob = system[idx + len(TOOLS_MARKER) :]
    dec = json.JSONDecoder()
    tools: list[dict[str, Any]] = []
    i = 0
    while i < len(blob):
        while i < len(blob) and blob[i] in " \r\n\t,":
            i += 1
        if i >= len(blob):
            break
        try:
            obj, i = dec.raw_decode(blob, i)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        tools.append(obj)
    return tools or None


def split_turns(chat: str) -> list[tuple[str, str]]:
    """Split a glaive chat into (role, text) turns."""
    parts = re.split(r"(?:^|\n\n\n)(USER|ASSISTANT|FUNCTION RESPONSE):\s?", chat)
    turns = []
    for k in range(1, len(parts) - 1, 2):
        text = parts[k + 1].replace(EOT, "").replace("<|endoftext|>", "").strip()
        turns.append((parts[k], text))
    return turns


def parse_function_call(text: str) -> dict[str, Any] | None:
    m = FUNCCALL_RE.match(text.strip())
    if not m:
        return None
    try:
        args = json.loads(m.group("args"))
    except json.JSONDecodeError:
        return None
    if not isinstance(args, dict):
        return None
    return {"tool": m.group("name"), "arguments": args}


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_RE.search(text)) and "<functioncall>" not in text


# --------------------------------------------------------------------------- validation
def validate_schema(tool: dict[str, Any]) -> bool:
    if not isinstance(tool.get("name"), str) or not tool["name"]:
        return False
    params = tool.get("parameters", {"type": "object", "properties": {}})
    if not isinstance(params, dict) or not isinstance(params.get("properties", {}), dict):
        return False
    req = params.get("required", [])
    return isinstance(req, list) and all(r in params.get("properties", {}) for r in req)


def validate_call(call: dict[str, Any], tools: list[dict[str, Any]]) -> bool:
    """Gold-label validation: tool exists, required args present, no unknown args,
    primitive types match the JSON schema."""
    if call["tool"] is None:
        return call["arguments"] == {}
    by_name = {t["name"]: t for t in tools}
    tool = by_name.get(call["tool"])
    if tool is None:
        return False
    params = tool.get("parameters", {}) or {}
    props = params.get("properties", {}) or {}
    for r in params.get("required", []) or []:
        if r not in call["arguments"]:
            return False
    for k, v in call["arguments"].items():
        if k not in props:
            return False
        expected = props[k].get("type") if isinstance(props[k], dict) else None
        if expected in JSON_TYPES:
            ok = isinstance(v, JSON_TYPES[expected])
            if expected in ("number", "integer") and isinstance(v, bool):
                ok = False
            if not ok:
                return False
        if isinstance(props[k], dict) and "enum" in props[k] and v not in props[k]["enum"]:
            return False
    return True


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower()).strip(" ?.!")


def record_to_example(rec: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Convert one raw record. Returns (example, reason) where reason explains drops."""
    tools = parse_tools(rec.get("system", ""))
    if tools is None:
        return None, "no_tools"
    if not all(validate_schema(t) for t in tools):
        return None, "bad_schema"
    if len({t["name"] for t in tools}) != len(tools):
        return None, "dup_tool_names"
    turns = split_turns(rec.get("chat", ""))
    if len(turns) < 2 or turns[0][0] != "USER" or turns[1][0] != "ASSISTANT":
        return None, "bad_turns"
    query, answer = turns[0][1], turns[1][1]
    if not query or len(query) < 8:
        return None, "empty_query"
    if answer.startswith("<functioncall>"):
        call = parse_function_call(answer)
        if call is None:
            return None, "unparseable_call"
    elif is_refusal(answer):
        call = {"tool": None, "arguments": {}}
    else:
        return None, "clarification_or_chat"
    if not validate_call(call, tools):
        return None, "invalid_call"
    ex = {"tools": tools, "query": query, "target": call}
    ex["id"] = hashlib.sha1(
        (normalize_query(query) + "|" + ",".join(sorted(t["name"] for t in tools))).encode()
    ).hexdigest()[:12]
    return ex, "ok"


def target_to_str(target: dict[str, Any]) -> str:
    """Canonical serialized target used for SFT labels."""
    return json.dumps({"tool": target["tool"], "arguments": target["arguments"]}, ensure_ascii=False)


STOPWORDS = {
    "get",
    "the",
    "a",
    "an",
    "for",
    "of",
    "to",
    "and",
    "in",
    "on",
    "by",
    "with",
    "calculate",
    "create",
    "search",
    "find",
    "check",
    "generate",
    "convert",
    "info",
}


def content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower().replace("_", " "))
    return {w.rstrip("s") for w in words if len(w) > 2 and w not in STOPWORDS}


def make_synthetic_negatives(
    calls: list[dict[str, Any]], n: int, rng: random.Random, max_tries: int = 50
) -> list[dict[str, Any]]:
    """Tool-swap negatives: pair a real request with the tool list of an unrelated
    example, so the correct behaviour is to decline. A swap is rejected if the query
    shares any content word with the swapped tools' names/descriptions."""
    out: list[dict[str, Any]] = []
    used: set[str] = set()
    for ex in calls:
        if len(out) >= n:
            break
        if ex["id"] in used:
            continue
        q_words = content_words(ex["query"])
        own = {t["name"] for t in ex["tools"]}
        for _ in range(max_tries):
            other = calls[rng.randrange(len(calls))]
            names = {t["name"] for t in other["tools"]}
            if names & own:
                continue
            tool_words = set().union(
                *(content_words(t["name"] + " " + str(t.get("description", ""))) for t in other["tools"])
            )
            if q_words & tool_words:
                continue
            neg = {
                "tools": other["tools"],
                "query": ex["query"],
                "target": {"tool": None, "arguments": {}},
                "source": "synthetic_negative",
            }
            neg["id"] = (
                "neg-"
                + hashlib.sha1((normalize_query(ex["query"]) + "|" + ",".join(sorted(names))).encode()).hexdigest()[:10]
            )
            out.append(neg)
            used.add(ex["id"])
            break
    return out


def tools_char_len(tools: list[dict[str, Any]]) -> int:
    return len(json.dumps(tools, separators=(",", ":")))


# --------------------------------------------------------------------------- pipeline
def build_dataset(
    raw_path: Path,
    out_dir: Path,
    total: int = 4000,
    refusal_frac: float = 0.2,
    val_size: int = 400,
    test_size: int = 600,
    max_tools_chars: int = 1200,
    max_per_tool: int = 40,
    synthetic_negative_frac: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Clean -> validate -> dedupe -> cap per tool -> add negatives -> split.

    ``refusal_frac`` of the final set are "no tool applies" examples; at most
    ``synthetic_negative_frac`` of those are synthetic tool-swap negatives, the rest are
    natural glaive declines. Writes {train,val,test}.jsonl and stats.json.
    """
    reasons: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    calls, refusals = [], []
    n_raw = 0
    for rec in iter_raw_records(raw_path):
        n_raw += 1
        ex, why = record_to_example(rec)
        if ex is None:
            reasons[why] += 1
            continue
        if tools_char_len(ex["tools"]) > max_tools_chars:
            reasons["too_long"] += 1
            continue
        nq = normalize_query(ex["query"])
        if ex["id"] in seen_ids or nq in seen_queries:
            reasons["duplicate"] += 1
            continue
        seen_ids.add(ex["id"])
        seen_queries.add(nq)
        ex["source"] = "natural"
        reasons["kept"] += 1
        (refusals if ex["target"]["tool"] is None else calls).append(ex)

    rng = random.Random(seed)
    rng.shuffle(calls)
    rng.shuffle(refusals)
    # Cap examples per target tool so a few popular tools do not dominate.
    per_tool: Counter[str] = Counter()
    capped = []
    for ex in calls:
        per_tool[ex["target"]["tool"]] += 1
        if per_tool[ex["target"]["tool"]] <= max_per_tool:
            capped.append(ex)
        else:
            reasons["per_tool_cap"] += 1
    n_ref_total = int(total * refusal_frac)
    n_nat = min(len(refusals), n_ref_total - int(n_ref_total * synthetic_negative_frac))
    n_syn = n_ref_total - n_nat
    n_call = min(len(capped), total - n_ref_total)
    chosen_calls = capped[:n_call]
    # Negatives reuse queries of calls that are NOT in the final pool (no query reuse).
    negatives = make_synthetic_negatives(capped[n_call:] or capped, n_syn, rng)
    pool = chosen_calls + refusals[:n_nat] + negatives
    rng.shuffle(pool)

    # Stratified split so every split has the same refusal ratio.
    def strat(items: list[dict], k: int) -> tuple[list[dict], list[dict]]:
        ref = [e for e in items if e["target"]["tool"] is None]
        cal = [e for e in items if e["target"]["tool"] is not None]
        k_ref = round(k * len(ref) / max(1, len(items)))
        take = ref[:k_ref] + cal[: k - k_ref]
        ids = {e["id"] for e in take}
        rest = [e for e in items if e["id"] not in ids]
        rng.shuffle(take)
        return take, rest

    test, rest = strat(pool, test_size)
    val, train = strat(rest, val_size)
    splits = {"train": train, "val": val, "test": test}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for e in items:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    stats = {
        "source": f"{DATASET_ID}@{DATASET_REVISION}",
        "raw_records_scanned": n_raw,
        "filter_reasons": dict(reasons),
        "pool_calls_after_dedupe": len(calls),
        "pool_calls_after_tool_cap": len(capped),
        "pool_natural_refusals": len(refusals),
        "final_counts": {
            "calls": len(chosen_calls),
            "natural_refusals": n_nat,
            "synthetic_negatives": len(negatives),
        },
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_refusal_rate": {
            k: round(sum(e["target"]["tool"] is None for e in v) / max(1, len(v)), 4) for k, v in splits.items()
        },
        "unique_tool_names": len({t["name"] for e in pool for t in e["tools"]}),
        "avg_tools_per_example": round(sum(len(e["tools"]) for e in pool) / max(1, len(pool)), 3),
        "seed": seed,
    }
    assert not ({e["id"] for e in train} & {e["id"] for e in test}), "train/test leak"
    assert not ({e["id"] for e in val} & {e["id"] for e in test}), "val/test leak"
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
                if limit and len(out) >= limit:
                    break
    return out
