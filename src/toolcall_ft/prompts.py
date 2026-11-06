"""Prompt construction. The same system instruction is used for every stage; the
baseline additionally gets few-shot demonstrations (the "best prompt" condition)."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are a function-calling engine. You receive a list of available tools (JSON schemas) and a user request.
Decide whether ONE of the tools can fulfil the request.
- If a tool applies, respond with ONLY a JSON object: {"tool": "<tool name>", "arguments": {<argument name>: <value>, ...}}
  Use exact tool and argument names from the schema, include every required argument, use values stated by the user, and match the JSON types in the schema (numbers as numbers, not strings).
- If no tool can fulfil the request, respond with ONLY: {"tool": null, "arguments": {}}
Never add explanations, markdown or code fences. Output a single line of JSON."""

# Hand-written demonstrations (not drawn from the dataset, so no test leakage).
FEW_SHOT: list[dict[str, Any]] = [
    {
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["city"],
                },
            }
        ],
        "query": "What's the weather like in Hyderabad right now, in celsius please?",
        "target": {"tool": "get_weather", "arguments": {"city": "Hyderabad", "unit": "celsius"}},
    },
    {
        "tools": [
            {
                "name": "calculate_bmi",
                "description": "Calculate body mass index",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "weight_kg": {"type": "number"},
                        "height_m": {"type": "number"},
                    },
                    "required": ["weight_kg", "height_m"],
                },
            },
            {
                "name": "convert_currency",
                "description": "Convert an amount between currencies",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number"},
                        "from_currency": {"type": "string"},
                        "to_currency": {"type": "string"},
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                },
            },
        ],
        "query": "Can you convert 250 US dollars to Indian rupees?",
        "target": {
            "tool": "convert_currency",
            "arguments": {"amount": 250, "from_currency": "USD", "to_currency": "INR"},
        },
    },
    {
        "tools": [
            {
                "name": "search_recipes",
                "description": "Search recipes by ingredient",
                "parameters": {
                    "type": "object",
                    "properties": {"ingredients": {"type": "array", "items": {"type": "string"}}},
                    "required": ["ingredients"],
                },
            }
        ],
        "query": "Please book me a cab to the airport for 6 am tomorrow.",
        "target": {"tool": None, "arguments": {}},
    },
]


def format_user_message(tools: list[dict[str, Any]], query: str) -> str:
    tools_json = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    return f"Available tools:\n{tools_json}\n\nUser request:\n{query}"


def target_str(target: dict[str, Any]) -> str:
    return json.dumps({"tool": target["tool"], "arguments": target["arguments"]}, ensure_ascii=False)


def build_messages(ex: dict[str, Any], few_shot: bool = False) -> list[dict[str, str]]:
    """Chat messages for one example (without the assistant answer)."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for demo in FEW_SHOT:
            msgs.append({"role": "user", "content": format_user_message(demo["tools"], demo["query"])})
            msgs.append({"role": "assistant", "content": target_str(demo["target"])})
    msgs.append({"role": "user", "content": format_user_message(ex["tools"], ex["query"])})
    return msgs


def to_sft_record(ex: dict[str, Any], chat_template_kwargs: dict[str, Any] | None = None) -> dict[str, Any]:
    """TRL conversational prompt/completion record (loss on completion only).
    ``chat_template_kwargs`` (e.g. {"enable_thinking": False} for Qwen3) is forwarded by TRL."""
    rec: dict[str, Any] = {
        "prompt": build_messages(ex, few_shot=False),
        "completion": [{"role": "assistant", "content": target_str(ex["target"])}],
    }
    if chat_template_kwargs:
        rec["chat_template_kwargs"] = chat_template_kwargs
    return rec
