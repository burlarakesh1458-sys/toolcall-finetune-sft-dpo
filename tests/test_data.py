import json
import random

from toolcall_ft.data import (
    build_dataset,
    is_refusal,
    make_synthetic_negatives,
    normalize_query,
    parse_function_call,
    parse_tools,
    record_to_example,
    split_turns,
    validate_call,
)
from toolcall_ft.prompts import build_messages, to_sft_record

EOT = "<\u200b|endoftext|>"
WEATHER = {
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        "required": ["city"],
    },
}
NEWS = {
    "name": "get_news",
    "description": "Get news headlines",
    "parameters": {"type": "object", "properties": {"country": {"type": "string"}}, "required": ["country"]},
}


def sys_prompt(*tools):
    body = "\n".join(json.dumps(t, indent=4) for t in tools)
    return (
        f"SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n{body}\n"
    )


def call_chat(query, name, args):
    return (
        f'USER: {query}\n\n\nASSISTANT: <functioncall> {{"name": "{name}", "arguments": \'{json.dumps(args)}\'}} {EOT}\n\n\n'
        "FUNCTION RESPONSE: {}\n\n\nASSISTANT: Done. " + EOT + "\n\n\n"
    )


def text_chat(query, answer):
    return f"USER: {query}\n\n\nASSISTANT: {answer} {EOT}\n\n\n"


def test_parse_tools_multiple_and_absent():
    assert [t["name"] for t in parse_tools(sys_prompt(WEATHER, NEWS))] == ["get_weather", "get_news"]
    assert parse_tools("SYSTEM: You are a helpful assistant, with no access to external functions.") is None


def test_split_turns_strips_eot():
    turns = split_turns(call_chat("weather in Pune?", "get_weather", {"city": "Pune"}))
    assert [r for r, _ in turns] == ["USER", "ASSISTANT", "FUNCTION RESPONSE", "ASSISTANT"]
    assert EOT not in turns[1][1]


def test_parse_function_call_single_quoted_args():
    call = parse_function_call('<functioncall> {"name": "get_weather", "arguments": \'{"city": "Pune", "days": 2}\'}')
    assert call == {"tool": "get_weather", "arguments": {"city": "Pune", "days": 2}}
    assert parse_function_call('<functioncall> {"name": "x", "arguments": \'{bad json}\'}') is None


def test_refusal_detection():
    assert is_refusal("I'm sorry, but I don't have the capability to book flights.")
    assert not is_refusal("Sure, could you tell me the city?")


def test_validate_call_types_required_unknown():
    assert validate_call({"tool": "get_weather", "arguments": {"city": "Pune"}}, [WEATHER])
    assert not validate_call({"tool": "get_weather", "arguments": {}}, [WEATHER])  # missing required
    assert not validate_call({"tool": "get_weather", "arguments": {"city": "P", "x": 1}}, [WEATHER])  # unknown
    assert not validate_call({"tool": "get_weather", "arguments": {"city": "P", "days": "2"}}, [WEATHER])  # type
    assert not validate_call({"tool": "get_weather", "arguments": {"city": "P", "days": True}}, [WEATHER])  # bool!=int
    assert not validate_call({"tool": "nope", "arguments": {}}, [WEATHER])
    assert validate_call({"tool": None, "arguments": {}}, [WEATHER])


def test_record_to_example_call_refusal_and_clarification():
    ex, why = record_to_example(
        {
            "system": sys_prompt(WEATHER),
            "chat": call_chat("Weather in Pune for 2 days?", "get_weather", {"city": "Pune", "days": 2}),
        }
    )
    assert why == "ok" and ex["target"] == {"tool": "get_weather", "arguments": {"city": "Pune", "days": 2}}
    ex, why = record_to_example(
        {
            "system": sys_prompt(WEATHER),
            "chat": text_chat("Book me a flight to Delhi", "I'm sorry, I can't book flights."),
        }
    )
    assert why == "ok" and ex["target"] == {"tool": None, "arguments": {}}
    ex, why = record_to_example(
        {"system": sys_prompt(WEATHER), "chat": text_chat("What's the weather?", "Sure, which city?")}
    )
    assert ex is None and why == "clarification_or_chat"
    ex, why = record_to_example(
        {"system": sys_prompt(WEATHER), "chat": call_chat("Weather in Pune please", "get_weather", {"town": "Pune"})}
    )
    assert ex is None and why == "invalid_call"


def test_normalize_query():
    assert normalize_query("  What's the  WEATHER?? ") == "what's the weather"


def test_synthetic_negatives_do_not_overlap():
    calls = [
        {
            "id": "1",
            "tools": [WEATHER],
            "query": "What is the weather in Pune?",
            "target": {"tool": "get_weather", "arguments": {"city": "Pune"}},
        },
        {
            "id": "2",
            "tools": [NEWS],
            "query": "Show me headlines from India",
            "target": {"tool": "get_news", "arguments": {"country": "India"}},
        },
    ]
    negs = make_synthetic_negatives(calls, 2, random.Random(0))
    for n in negs:
        assert n["target"] == {"tool": None, "arguments": {}}
        assert n["source"] == "synthetic_negative"
    # weather query must never be paired with the weather tool
    for n in negs:
        if "weather" in n["query"].lower():
            assert all(t["name"] != "get_weather" for t in n["tools"])


def _write_raw(path, n_calls=60, n_ref=20):
    recs = []
    cities = [f"City{i}" for i in range(n_calls)]
    for i, c in enumerate(cities):
        recs.append(
            {
                "system": sys_prompt(WEATHER, NEWS),
                "chat": call_chat(f"What is the weather in {c} today?", "get_weather", {"city": c}),
            }
        )
        if i < 5:  # duplicates must be removed
            recs.append(
                {
                    "system": sys_prompt(WEATHER, NEWS),
                    "chat": call_chat(f"what is the weather in {c} today", "get_weather", {"city": c}),
                }
            )
    for i in range(n_ref):
        recs.append(
            {
                "system": sys_prompt(NEWS),
                "chat": text_chat(f"Please order pizza number {i}", "I'm sorry, I can't order food."),
            }
        )
    recs.append({"system": "SYSTEM: no tools", "chat": text_chat("hi there friend", "hello")})
    path.write_text(json.dumps(recs)[:-1] + ', {"system": "trunc')  # truncated tail like a ranged download


def test_build_dataset_end_to_end(tmp_path):
    raw = tmp_path / "raw.json"
    _write_raw(raw)
    stats = build_dataset(
        raw,
        tmp_path / "out",
        total=50,
        refusal_frac=0.2,
        val_size=10,
        test_size=10,
        max_per_tool=1000,
        synthetic_negative_frac=0.5,
        seed=1,
    )
    assert stats["filter_reasons"]["duplicate"] == 5
    assert stats["filter_reasons"]["no_tools"] == 1
    fc = stats["final_counts"]
    total = fc["calls"] + fc["natural_refusals"] + fc["synthetic_negatives"]
    assert fc["calls"] == 40 and fc["natural_refusals"] == 5
    # every call example shares tool names with every other one here, so no safe tool-swap
    # negative exists: the builder must produce none rather than a mislabeled one
    assert fc["synthetic_negatives"] == 0
    assert stats["split_sizes"] == {"train": total - 20, "val": 10, "test": 10}
    splits = {
        s: [json.loads(line) for line in (tmp_path / "out" / f"{s}.jsonl").read_text().splitlines()]
        for s in ("train", "val", "test")
    }
    ids = [e["id"] for s in splits.values() for e in s]
    assert len(ids) == len(set(ids)), "an example appears in two splits"
    queries = [
        normalize_query(e["query"]) for s in ("train", "test") for e in splits[s] if e.get("source") == "natural"
    ]
    assert len(queries) == len(set(queries))
    for s in splits.values():
        assert any(e["target"]["tool"] is None for e in s)


def test_deterministic_given_seed(tmp_path):
    raw = tmp_path / "raw.json"
    _write_raw(raw)
    a = build_dataset(raw, tmp_path / "a", total=50, val_size=10, test_size=10, max_per_tool=1000, seed=7)
    b = build_dataset(raw, tmp_path / "b", total=50, val_size=10, test_size=10, max_per_tool=1000, seed=7)
    assert a == b
    assert (tmp_path / "a" / "test.jsonl").read_text() == (tmp_path / "b" / "test.jsonl").read_text()


def test_prompt_and_sft_record():
    ex = {
        "tools": [WEATHER],
        "query": "Weather in Pune?",
        "target": {"tool": "get_weather", "arguments": {"city": "Pune"}},
    }
    msgs = build_messages(ex, few_shot=True)
    assert msgs[0]["role"] == "system" and msgs[-1]["role"] == "user"
    assert len(msgs) == 1 + 2 * 3 + 1
    rec = to_sft_record(ex)
    assert rec["prompt"][-1]["content"].endswith("Weather in Pune?")
    assert json.loads(rec["completion"][0]["content"]) == ex["target"]
