import math

from toolcall_ft.scorer import aggregate, parse_output, reward, score_all, score_one

GOLD_CALL = {"tool": "get_weather", "arguments": {"city": "Hyderabad", "days": 3}}
GOLD_NONE = {"tool": None, "arguments": {}}


def test_parse_valid_and_fenced():
    assert parse_output('{"tool": "a", "arguments": {}}') == {"tool": "a", "arguments": {}}
    assert parse_output('```json\n{"tool": null, "arguments": {}}\n```') == {"tool": None, "arguments": {}}


def test_parse_rejects_bad_shapes():
    assert parse_output("") is None
    assert parse_output('Sure! {"tool": "a", "arguments": {}}') is None  # extra prose
    assert parse_output('{"tool": "a"}') is None  # missing key
    assert parse_output('{"tool": "a", "arguments": {}, "x": 1}') is None  # extra key
    assert parse_output('{"tool": 3, "arguments": {}}') is None
    assert parse_output('{"tool": "a", "arguments": "{}"}') is None  # stringified args
    assert parse_output('{"tool": null, "arguments": {"a": 1}}') is None  # decline with args
    assert parse_output("[1, 2]") is None


def test_exact_match_normalises_numbers_case_and_key_order():
    pred = '{"arguments": {"days": 3.0, "city": "  hyderabad "}, "tool": "get_weather"}'
    r = score_one(pred, GOLD_CALL)
    assert r["json_valid"] and r["tool_correct"] and r["exact_match"]
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)


def test_partial_arguments_counted_in_f1():
    r = score_one('{"tool": "get_weather", "arguments": {"city": "Hyderabad", "days": 5}}', GOLD_CALL)
    assert r["tool_correct"] and not r["exact_match"]
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)


def test_wrong_tool_gets_no_true_positives():
    r = score_one('{"tool": "get_news", "arguments": {"city": "Hyderabad"}}', GOLD_CALL)
    assert not r["tool_correct"] and r["tp"] == 0 and r["fp"] == 1 and r["fn"] == 2


def test_refusal_logic():
    ok = score_one('{"tool": null, "arguments": {}}', GOLD_NONE)
    assert ok["exact_match"] and ok["decision_correct"]
    wrong = score_one('{"tool": "get_weather", "arguments": {"city": "x"}}', GOLD_NONE)
    assert not wrong["decision_correct"] and not wrong["exact_match"]
    false_decline = score_one('{"tool": null, "arguments": {}}', GOLD_CALL)
    assert not false_decline["decision_correct"]


def test_invalid_json_is_wrong_everywhere():
    r = score_one("I think you should call get_weather", GOLD_CALL)
    assert not r["json_valid"] and not r["decision_correct"] and r["fn"] == 2


def test_aggregate_metrics():
    preds = [
        '{"tool": "get_weather", "arguments": {"city": "Hyderabad", "days": 3}}',  # exact
        '{"tool": null, "arguments": {}}',  # false decline
        '{"tool": null, "arguments": {}}',  # correct decline
        "garbage",  # invalid on a decline example
    ]
    golds = [GOLD_CALL, GOLD_CALL, GOLD_NONE, GOLD_NONE]
    m, _ = score_all(preds, golds)
    assert m["n"] == 4
    assert m["json_valid"] == 0.75
    assert m["exact_match"] == 0.5
    assert m["refusal_acc"] == 0.5
    assert m["refusal_recall"] == 0.5
    assert m["false_refusal"] == 0.5
    # args: tp=2, fp=0, fn=2 -> P=1, R=0.5, F1=2/3
    assert math.isclose(m["arg_f1"], 2 / 3)


def test_aggregate_empty():
    assert aggregate([]) == {}


def test_reward_ordering():
    exact = reward('{"tool": "get_weather", "arguments": {"city": "Hyderabad", "days": 3}}', GOLD_CALL)
    partial = reward('{"tool": "get_weather", "arguments": {"city": "Hyderabad"}}', GOLD_CALL)
    wrong_tool = reward('{"tool": "other", "arguments": {}}', GOLD_CALL)
    wrong_decision = reward('{"tool": null, "arguments": {}}', GOLD_CALL)
    invalid = reward("nope", GOLD_CALL)
    assert exact > partial > wrong_tool > wrong_decision > invalid
