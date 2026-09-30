"""Synthetic contracts only; no real train/test/model interpretation."""

import copy
import hashlib
import types
import io
import tarfile
from pathlib import Path

import pytest

from climate_rag.evidence_gap_candidate import (
    GapProviderAdapter,
    render_gap_prompt,
    system_prompt_gap,
)
from climate_rag.scifact_consumption import (
    consumption_ledger,
    review_selected,
    select_unconsumed,
    verify_bytes,
)
from climate_rag.scifact_grounding import Abstract, GoldClaim
from climate_rag.scifact_semantic_policy import (
    OLD_G,
    SEMANTIC_G,
    POLICIES,
    LocalQwenSemanticGapProvider,
    policy_prompt,
    policy_sha,
    render_policy_prompt,
)
from climate_rag.scifact_semantic_runtime import (
    PolicyCounter,
    SemanticCommonPacking,
    SemanticPackingFixture,
    run_semantic_slot,
    semantic_packing_snapshot,
    validate_pair_rows,
)
from climate_rag.scifact_terminal import action_schema, source_from_abstract
from climate_rag.scifact_train_diagnostic import STRATA, select_component_distinct
from test_bounded_scifact_runtime import ABSTAIN, provider_fixture, wire
from test_scifact_train_diagnostic import TinyTokenizer

SHA = "a" * 64
GIT = "b" * 40


def population():
    rows = [{"id": i, "stratum": STRATA[i % 4]} for i in range(20)]
    assignment = {
        "eligible_train_ids": list(range(20)),
        "claim_component": {str(i): str(i) for i in range(20)},
    }
    assignment["claim_component"]["1"] = "0"  # same component, different stratum
    return rows, assignment


def record(i, result=None):
    value = {"claim_id": i, "source_path": "private/frozen-run", "source_sha256": SHA}
    if result is not None:
        value["result"] = result
    return value


def test_exclusion_before_options_stable_and_legacy_default_unchanged():
    audit, assignment = population()
    components = {int(i): c for i, c in assignment["claim_component"].items()}
    baseline = select_component_distinct(audit, components)
    assert baseline == select_component_distinct(
        audit, components, excluded_components=()
    )
    picked, summary = select_component_distinct(
        audit, components, excluded_components=["0"]
    )
    assert all(r["id"] not in (0, 1) for r in picked)
    assert len(picked) == 12 and len({r["component"] for r in picked}) == 12
    assert (picked, summary) == select_component_distinct(
        list(reversed(audit)), components, excluded_components=["0"]
    )
    assert (
        select_component_distinct(
            audit, components, excluded_components=set(components.values())
        )[0]
        == []
    )


def test_failures_abstention_unknown_usage_consumed_and_planned_uncertain():
    audit, assignment = population()
    actual = [
        record(
            0, {"generation_attempts": [{"status": "model_exception", "usage": {}}]}
        ),
        record(
            0,
            {
                "generation_attempts": [{"status": "validation_failed"}],
                "outcome": "export_error",
            },
        ),
        record(2, {"model_calls": 1, "outcome": "model_abstention"}),
        record(3, {"model_calls": 0}),
    ]
    ledger = consumption_ledger(assignment, audit, [record(0), record(4)], actual)
    assert ledger["model_consumed_ids"] == [0, 2]
    assert ledger["uncertain_ids"] == [3, 4]
    assert 1 in ledger["component_excluded_eligible_ids"]
    assert len(ledger["records"][0]["evidence"]) == 3
    selected, summary = select_unconsumed(audit, assignment, ledger)
    assert all(r["component"] not in ledger["component_excluded"] for r in selected)
    assert all(set(r) == {"id", "component", "legacy_stratum"} for r in selected)
    assert summary["shortfall_policy"].startswith("stop")


@pytest.mark.parametrize(
    "case", ["missing_audit", "duplicate", "foreign", "mapping", "nested", "sha"]
)
def test_ledger_fails_closed_on_identity(case):
    audit, assignment = population()
    actual = [record(0, {"claim_id": 0, "model_calls": 1})]
    if case == "missing_audit":
        audit.pop()
    if case == "duplicate":
        audit.append(audit[0])
    if case == "foreign":
        actual[0]["claim_id"] = 999
    if case == "mapping":
        assignment["claim_component"].pop("19")
    if case == "nested":
        actual[0]["result"]["claim_id"] = 2
    if case == "sha":
        actual[0]["source_sha256"] = "not-a-sha"
    with pytest.raises(ValueError):
        consumption_ledger(assignment, audit, [], actual)
    with pytest.raises(ValueError):
        verify_bytes(b"wrong", SHA)
    verify_bytes(b"right", hashlib.sha256(b"right").hexdigest())


def test_old_prompt_bytes_schema_and_invalid_policy():
    tokenizer = TinyTokenizer()
    observation = {
        "immutable_claim": "Fixture",
        "current_citable": [],
        "preview_only": [],
    }
    schema = action_schema(["answer", "abstain"], ["c0"], ["c0:0"], 5)
    assert policy_prompt(OLD_G) == system_prompt_gap()
    assert render_policy_prompt(
        tokenizer, observation, schema, OLD_G
    ) == render_gap_prompt(tokenizer, observation, schema)
    assert policy_sha(OLD_G) != policy_sha(SEMANTIC_G)
    assert "SUPPORTS" in policy_prompt(SEMANTIC_G) and "REFUTES" in policy_prompt(
        SEMANTIC_G
    )
    assert "no minimum number of tools" in policy_prompt(SEMANTIC_G)
    assert GapProviderAdapter(PolicyCounter(tokenizer, OLD_G)).prepare(
        observation, schema
    ) == GapProviderAdapter(PolicyCounter(tokenizer, SEMANTIC_G)).prepare(
        observation, schema
    )
    with pytest.raises(ValueError):
        policy_prompt("unknown")


def inputs(large=False):
    corpus = {
        i: Abstract(
            i,
            "Fixture",
            tuple(
                f"Sentence{j} " + "x" * (220 if large else 2)
                for j in range(30 if large else 3)
            ),
            False,
        )
        for i in range(100, 120)
    }
    sources = [source_from_abstract(a) for a in corpus.values()]
    return corpus, lambda q, k: sources[:k]


def backend(tokenizer, policy, ids=()):
    b = SemanticPackingFixture(tokenizer, policy, ids)
    b.base = types.SimpleNamespace(tokenizer=tokenizer)
    return b


@pytest.mark.parametrize("policy", POLICIES)
def test_count_generate_same_prompt_full_wire_and_greedy(tmp_path, monkeypatch, policy):
    old, calls, parsers = provider_fixture(
        tmp_path, monkeypatch, [wire(ABSTAIN)], gap=True
    )
    b = LocalQwenSemanticGapProvider.__new__(LocalQwenSemanticGapProvider)
    b.__dict__.update(old.__dict__)
    b.policy = policy
    adapter = GapProviderAdapter(b)
    obs = {"immutable_claim": "Fixture", "current_citable": [], "preview_only": []}
    schema = action_schema(["abstain"], [], [], 5)
    count = adapter.count_prompt(obs, schema)
    result = adapter.generate(obs, schema, 512, 45)
    assert result["usage"]["input_tokens"] == count == len(calls[0]["input_ids"].ids)
    assert result["usage"]["output_tokens"] > len(result["raw"])
    assert calls[0]["do_sample"] is False and calls[0]["num_beams"] == 1
    assert parsers[0][2] is not None


def test_both_policy_history_max_pure_and_actual_cost():
    t = TinyTokenizer()
    adapter = GapProviderAdapter(backend(t, OLD_G))
    obs = {"immutable_claim": "Fixture", "current_citable": [], "preview_only": []}
    schema = action_schema(["abstain"], [], [], 5)
    packing = SemanticCommonPacking(adapter, t)
    before = packing(obs, schema)
    adapter._previous_gap = {"synthetic_history": "x" * 1200}
    expected, envelope = adapter.prepare(obs, schema)
    old_state = copy.deepcopy(adapter.__dict__)
    after = packing(obs, schema)
    assert after > before and after >= adapter.count_prompt(obs, schema)
    assert after >= PolicyCounter(t, SEMANTIC_G).count_prompt(expected, envelope)
    assert (
        adapter.records == old_state["records"]
        and adapter._previous_gap == old_state["_previous_gap"]
    )


def test_boundary_fixture_matches_runtime_rerank_schema_and_pair_identity():
    corpus, retrieve = inputs(large=True)
    t, rows = TinyTokenizer(), []
    for policy in POLICIES:
        b = backend(t, policy)
        row = run_semantic_slot(
            42, "Fixture claim", "adaptive", b, retrieve, lambda q, c: c, corpus, GIT
        )
        probe = semantic_packing_snapshot("Fixture claim", retrieve, t, policy=policy)
        assert probe["initial_identity"] == row["result"]["initial_context_identity"]
        assert (
            probe["prompt_tokens"]
            == row["result"]["generation_attempts"][0]["input_prompt_tokens"]
        )
        rows.append(row)
    validate_pair_rows(rows, GIT, [42], ["adaptive"])
    assert (
        rows[0]["result"]["initial_context_identity"]
        == rows[1]["result"]["initial_context_identity"]
    )
    assert (
        rows[0]["result"]["usage"]["input_tokens"]
        != rows[1]["result"]["usage"]["input_tokens"]
    )
    rows[0]["comparison_protocol"] = "scifact-bounded-gap-paired-v1"
    with pytest.raises(ValueError):
        validate_pair_rows(rows, GIT, [42], ["adaptive"])


@pytest.mark.parametrize("policy", POLICIES)
def test_explicit_read_history_execution_audit_and_cost(policy):
    corpus, retrieve = inputs()
    b = backend(TinyTokenizer(), policy, ["c5"])
    row = run_semantic_slot(
        42, "Fixture claim", "adaptive", b, retrieve, lambda q, c: c, corpus, GIT
    )
    result = row["result"]
    assert result["claim_id"] == 42 and result["arm"] == policy
    assert result["decision_execution_audit"][0]["actual_event_status"] == "completed"
    assert result["model_tool_links"][0]["subsequent_model_attempt"] == 1
    assert not result["trace_unlinked_events"]
    assert (
        b.observations[1]["evidence_gap_feedback"]["previous_valid_envelope"][
            "proposed_action"
        ]
        == "read"
    )
    probe = semantic_packing_snapshot(
        "Fixture claim", retrieve, b.tokenizer, ["c5"], policy
    )
    assert probe["fixture_history"]["read_ids"] == ["c5"] and probe["model_calls"] == 0
    assert (
        probe["prompt_tokens"]
        == result["generation_attempts"][-1]["input_prompt_tokens"]
    )
    with pytest.raises(ValueError):
        semantic_packing_snapshot(
            "Fixture claim",
            retrieve,
            b.tokenizer,
            ["c0", "c1", "c2", "c3", "c4"],
            policy,
        )


def test_export_failure_retains_durable_cost(monkeypatch):
    def fail(*args):
        raise ValueError("fixture export failure")

    monkeypatch.setattr(
        "climate_rag.scifact_bounded_runtime.to_original_prediction", fail
    )
    corpus, retrieve = inputs()
    saved = []
    row = run_semantic_slot(
        42,
        "Fixture",
        "adaptive",
        backend(TinyTokenizer(), OLD_G),
        retrieve,
        lambda q, c: c,
        corpus,
        GIT,
        lambda result: saved.append(copy.deepcopy(result)),
    )
    assert row["prediction"] is None and row["export_error"] == "ValueError"
    assert saved[0]["usage"] == row["result"]["usage"]
    assert saved[0]["source_git"] == GIT and saved[0]["prompt_sha256"] == policy_sha(
        OLD_G
    )


def test_only_selected_queries_probed_no_reselection_legacy_separate(monkeypatch):
    calls = []

    def snapshot(claim, retrieve, tokenizer, read_ids, policy):
        calls.append((claim, policy))
        return {
            "candidate_doc_ids": [],
            "visible": {},
            "initial_identity": {"visible": [], "previews": []},
        }

    monkeypatch.setattr(
        "climate_rag.scifact_consumption.semantic_packing_snapshot", snapshot
    )
    selected = [{"id": 1, "component": "one", "legacy_stratum": "gold_absent_top20"}]
    gold = {1: GoldClaim(1, "only selected", {}, ())}
    rows, compact = review_selected(selected, gold, None, None)
    assert len(calls) == 2 and {c[0] for c in calls} == {"only selected"}
    assert rows[0]["legacy_stratum"] == "gold_absent_top20"
    assert rows[0]["current_opportunity"][OLD_G]["stratum"] == "nei"
    assert compact["model_calls"] == 0 and not compact["reselection_after_probe"]
    with pytest.raises(ValueError):
        review_selected(selected, gold | {2: gold[1]}, None, None)


@pytest.mark.parametrize(
    "field",
    ["claim_id", "route", "arm", "comparison_protocol", "source_git", "prompt_sha256"],
)
def test_nested_result_provenance_cannot_be_relabelled(field):
    corpus, retrieve = inputs()
    rows = [
        run_semantic_slot(
            42,
            "Fixture",
            "adaptive",
            backend(TinyTokenizer(), policy),
            retrieve,
            lambda q, c: c,
            corpus,
            GIT,
        )
        for policy in POLICIES
    ]
    rows[0]["result"][field] = "old_or_wrong"
    with pytest.raises(ValueError):
        validate_pair_rows(rows, GIT, [42], ["adaptive"])


def test_source_marker_alone_does_not_freeze_changed_code(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from prepare_scifact_semantic_pair import verify_source_tree

    source = tmp_path / "source"
    source.mkdir()
    payloads = {"SOURCE_REVISION": (GIT + "\n").encode(), "entry.py": b"# frozen\n"}
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as bundle:
        for name, payload in payloads.items():
            (source / name).write_bytes(payload)
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    verify_source_tree(source, archive, sha, GIT)
    (source / "entry.py").write_bytes(b"# modified after marker\n")
    with pytest.raises(ValueError):
        verify_source_tree(source, archive, sha, GIT)


def test_checked_input_bytes_remain_pinned_when_backing_file_changes(
    tmp_path, monkeypatch
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from prepare_scifact_semantic_pair import checked

    path = tmp_path / "private-fixture.json"
    original = b'{"fixture":true}'
    path.write_bytes(original)
    sha = hashlib.sha256(original).hexdigest()
    payload = checked(path, sha)
    path.write_bytes(b'{"fixture":"changed"}')
    assert payload == original
    with pytest.raises(ValueError):
        checked(path, sha)
