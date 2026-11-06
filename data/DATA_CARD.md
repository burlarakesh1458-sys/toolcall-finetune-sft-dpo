# Data card — single-step tool calling (derived from glaive-function-calling-v2)

## Source
- **Dataset:** [`glaiveai/glaive-function-calling-v2`](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2), pinned to revision `e7f4b6456019f5d8bcb991ef0dd67d8ff23221ac` (file `glaive-function-calling-v2.json`, 271 MB, 112,960 records).
- **License:** Apache-2.0 (upstream). The processed splits in `data/processed/` are a derivative and keep that license.
- **Why this one:** ungated and directly downloadable. `Salesforce/xlam-function-calling-60k` was the first choice but it is gated (HTTP 401 without accepting terms), so it cannot be fetched by a reproducible script.

## Task framing
Each raw record is a system prompt with 0–N JSON function schemas plus a multi-turn chat. Only the **first user turn** and the **first assistant turn** are kept, giving one decision per example:

| Input | Target (single-line JSON) |
|---|---|
| tool schemas + user request | `{"tool": "<name>", "arguments": {...}}` when a tool applies |
| tool schemas + user request | `{"tool": null, "arguments": {}}` when no tool applies (decline) |

## Cleaning pipeline (`src/toolcall_ft/data.py`)
Counts are from the actual build (`data/processed/stats.json`).

| Step | Rule | Records removed |
|---|---|---:|
| no tools | system prompt has no function schemas | 34,598 |
| exact / normalised duplicates | same lower-cased, whitespace-collapsed query, or same (query, tool set) | 44,314 |
| clarification / chit-chat | first assistant turn asks a follow-up question instead of calling or declining (ambiguous label for this task) | 25,520 |
| unparseable call | `<functioncall>` whose single-quoted `arguments` string is not valid JSON | 1,663 |
| duplicate tool names | two schemas with the same name in one prompt | 351 |
| invalid gold call | called tool not in the list, missing required arg, unknown arg, JSON-type mismatch (bool≠int, "5"≠number) or enum violation | 39 |
| bad schema / bad turns / too long | malformed schema, broken transcript, tools JSON > 1,200 chars | 41 |
| per-tool cap | at most 40 examples per target tool so a few popular tools (loan / BMI / password generators) don't dominate | 2,315 |

After cleaning: 6,217 unique calls (3,902 after the cap) and 217 natural declines.

### Negatives ("no tool applies")
Natural declines are rare (217), so the 20% decline share is filled with **synthetic tool-swap negatives** (583): a real request is paired with the tool list of an unrelated example. A swap is rejected if the query shares any content word with the swapped tools' names or descriptions, and negatives reuse only queries that are *not* used elsewhere in the dataset. This is a heuristic, so a small number of negatives may be arguable (e.g. "places of interest near me" vs a restaurant finder); it is documented as a limitation.

## Splits (seed 42, stratified on call vs decline)

| Split | Size | Declines | Natural / synthetic | 1 tool / 2 tools in prompt |
|---|---:|---:|---|---|
| train | 3,000 | 20.0% | 2,562 / 438 | 2,203 / 797 |
| val | 400 | 20.0% | 346 / 54 | 308 / 92 |
| test (held out) | 600 | 20.0% | 509 / 91 | 437 / 163 |

- 610 unique tool names; 1.26 tools per prompt on average; calls carry 1–6 arguments (mostly 2).
- No example id or normalised query appears in more than one split (asserted at build time and in `tests/test_data.py`).
- DPO prompts are drawn from the train split only, from examples *after* the SFT subset, so they are fresh for the SFT model and never touch val/test.

### Subsets used by each config
| Config | train | val | test |
|---|---:|---:|---:|
| `cpu_small.yaml` (executed) | first 240 | first 24 | first 100 (79 calls / 21 declines) |
| `gpu_qwen3_8b.yaml` (not executed) | 3,000 | 200 | 600 |

## Known issues
- glaive is synthetic (LLM-generated); some argument values are paraphrased relative to the user text, and a few natural "declines" may be refusals a stronger model would handle.
- Scoring normalises case and whitespace of string arguments and numeric types (5 == 5.0); it does **not** accept semantic equivalents ("NYC" vs "New York").
- Only single-call, first-turn behaviour is covered: no parallel calls, no multi-turn argument filling.
