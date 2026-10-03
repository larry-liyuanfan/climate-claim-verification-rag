"""Synthetic checks for the read-only result audit, not model evaluation."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "closeout_audit", Path(__file__).parents[1] / "scripts/audit_scifact_train_closeout.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def attempt(raw, **updates):
    data = raw.encode()
    receipt = dict(attempted_bytes=len(data), stored_bytes=len(data), dropped_bytes=0,
                   truncated=False, io_failed=False, sha256=audit.sha(data),
                   stored_prefix_sha256=audit.sha(data))
    receipt.update(updates)
    return {"diagnostics": {"private_attachment": receipt}}, {audit.sha(data): data}


@pytest.mark.parametrize("action", ["answer", "abstain", "read", "rewrite", "rerank"])
def test_raw_action_not_strict_validity(action):
    a, index = attempt(json.dumps({"documents": [], "action": action, "extra": "private"}))
    assert audit.wire_action(a, index) == action
    assert audit.wire_order(a, index) == ("documents", "2")


@pytest.mark.parametrize("raw", ['{"action":"read","action":"answer"}',
                                 '{"action":"private-unknown-value"}', '{}'])
def test_unknown_and_duplicate_actions_are_not_exported(raw):
    a, index = attempt(raw)
    assert audit.wire_action(a, index) == "unknown_ambiguous_json"
    assert audit.wire_order(a, index) == ("unknown", "unknown")


def test_truncation_missing_and_malformed_not_zero_intent():
    a, index = attempt('{"action":"read"}', truncated=True, dropped_bytes=3)
    assert audit.wire_action(a, index) == "unknown_unavailable"
    a, index = attempt('{"action":')
    assert audit.wire_action(a, index) == "unknown_unparseable_json"
    assert audit.wire_action(a, {}) == "unknown_unavailable"
    a, index = attempt('[["action","read"]]')
    assert audit.wire_action(a, index) == "unknown_non_object_json"


def test_receipts_count_duplicate_payload_attempts_separately():
    a, _ = attempt('{"action":"abstain"}')
    report = audit.receipt_totals([a, a])
    assert report["private_attachment"]["present"] == 2
    assert report["grammar_log"]["missing"] == 2
    assert report["affected_attempts_union"] == 2
    assert report["private_attachment"]["attempted_bytes"] == 2 * len('{"action":"abstain"}')


def test_independent_counts_first_three_and_alternatives():
    gold = {1: {"evidence": {"10": [{"label": "SUPPORT", "sentences": [1, 2]},
                                    {"label": "SUPPORT", "sentences": [4]}]}},
            2: {"evidence": {}}}
    pred = {1: {"evidence": {"10": {"label": "SUPPORT", "sentences": [0, 3, 5, 4]}}},
            2: {"evidence": {"11": {"label": "SUPPORT", "sentences": [0]}}}}
    metrics = audit.rescore(gold, pred)
    assert metrics["abstract_label_only"] == audit.metric(1, 2, 1)
    assert metrics["abstract_rationalized"] == audit.metric(0, 2, 1)
    assert metrics["sentence_selection"] == audit.metric(1, 5, 3)
    assert audit.opportunities(gold[1], {"10": [1, 4]}) == 1
    assert audit.opportunities(gold[1], {"10": [1]}) == 0
    with pytest.raises(ValueError, match="private values suppressed"):
        audit.rescore(gold, {1: pred[1]})


def test_cost_ledger_includes_failed_unknown_attempts():
    result = {"model_calls": 2, "unknown_usage_attempts": 1,
              "usage": {"input_tokens": 13, "output_tokens": 2},
              "generation_attempts": [
                  {"usage_known": True, "usage": {"input_tokens": 10, "output_tokens": 2}},
                  {"usage_known": False, "usage": {"input_tokens": 3}}]}
    audit.verify_cost(result)
    result["usage"]["input_tokens"] = 10
    with pytest.raises(ValueError):
        audit.verify_cost(result)


@pytest.mark.parametrize("rats", [[{"label": "SUPPORT", "sentences": [1]},
                                   {"label": "SUPPORT", "sentences": [1]}],
                                  [{"label": "SUPPORT", "sentences": [1]},
                                   {"label": "CONTRADICT", "sentences": [2]}]])
def test_unpredicted_invalid_gold_fails_closed(rats):
    with pytest.raises(ValueError):
        audit.rescore({1: {"evidence": {"10": rats}}}, {1: {"evidence": {}}})


def test_duplicate_document_content_never_silently_resolves():
    with pytest.raises(ValueError):
        audit.unique_used_hashes({1: "same", 2: "same"}, {"c0": "same"})
    assert audit.unique_used_hashes({1: "same", 2: "same", 3: "unique"},
                                    {"c0": "unique"}) == {"c0": "3"}


def test_identical_private_responses_require_duplicate_physical_files(tmp_path):
    a, index = attempt('{"action":"abstain"}')
    a["diagnostics"]["grammar_log"] = {"stored_bytes": 0}
    raw = next(iter(index.values()))
    (tmp_path / "first-response.txt").write_bytes(raw)
    with pytest.raises(ValueError):
        audit.reconcile_private_files(tmp_path, [a, a])
    (tmp_path / "second-response.txt").write_bytes(raw)
    recovered, count, size = audit.reconcile_private_files(tmp_path, [a, a])
    assert recovered == index and count == 2 and size == len(raw) * 2
    with pytest.raises(ValueError):
        audit.reconcile_private_files(tmp_path, [a])
