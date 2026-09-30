"""Synthetic seams only. No frozen train rows, model weights or real inference."""
import copy
import importlib
import inspect
import io
import json
import tarfile
import types
from pathlib import Path

import pytest

from climate_rag import scifact_semantic_contract as c
from climate_rag import scifact_semantic_execution as ex
from climate_rag.scifact_semantic_receipts import previous_arm, read_completed_run
from climate_rag.scifact_semantic_policy import LocalQwenSemanticGapProvider, policy_sha
from test_scifact_semantic_pair import backend, inputs
from test_scifact_train_diagnostic import TinyTokenizer

GIT = "a" * 40
SOURCE = "b" * 64
ARCHIVE = "c" * 64


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("package_scifact_semantic_pair")


def protocol(monkeypatch):
    claims = [{"id": i, "claim": f"Synthetic fixture {i}"} for i in range(12)]
    monkeypatch.setattr(c, "SELECTED_INFERENCE_SHA", c.sha(c.encoded(claims)))
    ids = list(range(12))
    p = {"release_id": c.RELEASE, "comparison_protocol": c.PAIR,
         "preparation_source_git": c.PREPARATION_GIT, "execution_source_git": GIT,
         "preparation_compact_sha256": c.COMPACT_SHA, "policy_prompt_sha256": c.PROMPTS,
         "policies": list(c.POLICIES), "routes": list(c.ROUTES), "budget": c.BUDGET,
         "model_manifest_sha256": c.MODEL_SHA, "reranker_manifest_sha256": c.RERANKER_SHA,
         "tokenizer_sha256": c.TOKENIZER_SHA,
         "ordered_claim_ids": ids, "corpus_documents": 5183, "model_calls": 0,
         "official_dev_read": False, "gpu_authorized": False, "fresh_baseline_required": True,
         "scope": c.SCOPE, "decoding": c.DECODING, "packing": c.PACKING,
         "release_requires_external_coordinator_authorization": True, "preflight_calls_per_arm": 4,
         "old_G_results_reusable": False,
         "slots_per_arm": [{"claim_id": i, "route": r} for i in ids for r in c.ROUTES],
         "paired_slots": [{"claim_id": i, "route": r, "policy": a} for a in c.POLICIES for i in ids for r in c.ROUTES],
         "inference_file_sha256": {"corpus.jsonl": c.CORPUS_SHA, "claims.jsonl": c.sha(c.jsonl(claims))}}
    return p, claims


@pytest.mark.parametrize("mutation", ["claims", "prompt", "budget", "source", "arm", "order", "fresh", "gold_field", "protocol_gold"])
def test_protocol_rejects_replacement_even_self_consistent(monkeypatch, mutation):
    p, claims = protocol(monkeypatch)
    c.validate_protocol(p, claims, GIT)
    p = copy.deepcopy(p)
    if mutation == "claims":
        claims[0]["claim"] = "Replacement despite internally consistent matrix"
        p["inference_file_sha256"]["claims.jsonl"] = c.sha(c.jsonl(claims))
    elif mutation == "prompt":
        p["policy_prompt_sha256"][c.POLICIES[0]] = "d" * 64
    elif mutation == "budget":
        p["budget"]["max_calls"] = 4
    elif mutation == "source":
        p["execution_source_git"] = c.PREPARATION_GIT
    elif mutation == "arm":
        p["policies"] = ["format_repaired", "format_repaired_gap"]
    elif mutation == "order":
        p["slots_per_arm"].reverse()
    elif mutation == "fresh":
        p["old_G_results_reusable"] = True
    elif mutation == "gold_field":
        claims[0]["evidence"] = {}
    else:
        p["gold"] = {}
    with pytest.raises(ValueError, match="protocol"):
        c.validate_protocol(p, claims, GIT)


def test_inference_exact_allowlist_hash_and_no_gold(tmp_path, monkeypatch):
    p, claims = protocol(monkeypatch)
    corpus = b"synthetic corpus\n"
    monkeypatch.setattr(c, "CORPUS_SHA", c.sha(corpus))
    p["inference_file_sha256"]["corpus.jsonl"] = c.CORPUS_SHA
    c.write_once(tmp_path / "protocol.json", p)
    (tmp_path / "corpus.jsonl").write_bytes(corpus)
    (tmp_path / "claims.jsonl").write_bytes(c.jsonl(claims))
    digest = c.sha((tmp_path / "protocol.json").read_bytes())
    assert c.load_inference(tmp_path, digest, GIT)[1] == claims
    (tmp_path / "gold.json").write_text("{}")
    with pytest.raises(ValueError, match="allowlist"):
        c.load_inference(tmp_path, digest, GIT)


@pytest.mark.parametrize("policy", c.POLICIES)
def test_policy_preflight_bound_and_durable_case_journal(tmp_path, monkeypatch, policy):
    b = types.SimpleNamespace(policy=policy, gap=True, wire_protocol=ex.CANDIDATE_PROTOCOL, kind="fixture")
    def smoke(backend, *, persist_case):
        records = []
        for i in range(4):
            record = {"passed": True, "usage": {"input_tokens": 10, "output_tokens": 20}}
            persist_case(i, "started", {"provider_generate_attempted": True})
            persist_case(i, "completed", record)
            records.append(record)
        return {"status": "passed", "attempted_calls": 4, "records": records}
    monkeypatch.setattr(ex, "bounded_runtime_smoke", smoke)
    report = ex.preflight(b, policy, tmp_path, GIT, SOURCE)
    assert report["gap"] and report["prompt_sha256"] == c.PROMPTS[policy] == policy_sha(policy)
    assert len(list(tmp_path.glob("preflight-*.json"))) == 8
    with pytest.raises(FileExistsError):
        ex.preflight(b, policy, tmp_path, GIT, SOURCE)
    b.gap = False
    with pytest.raises(ValueError, match="binding"):
        ex.provider_binding(b, policy)


def test_unknown_cost_and_export_failure_saved_before_stop(tmp_path, monkeypatch):
    b = types.SimpleNamespace(policy=c.POLICIES[0], gap=True, wire_protocol=ex.CANDIDATE_PROTOCOL,
                              start_slot=lambda p: p.mkdir())
    (tmp_path / "private-responses").mkdir()
    p = {"execution_source_git": GIT, "slots_per_arm": [{"claim_id": 1, "route": "adaptive"}]}
    def slot(*args, persist_raw):
        raw = {"claim_id": 1, "route": "adaptive", "arm": b.policy, "source_git": GIT,
               "prompt_sha256": c.PROMPTS[b.policy], "comparison_protocol": c.PAIR,
               "generation_attempts": [{"usage_known": False, "usage": {"input_tokens": 37}}]}
        persist_raw(raw)
        raise RuntimeError("synthetic auxiliary export failure")
    monkeypatch.setattr(ex, "run_semantic_slot", slot)
    with pytest.raises(ValueError, match="durable_stop"):
        ex.execute_slots(p, [{"id": 1, "claim": "Fixture"}], b, None, None, None, tmp_path)
    row = json.loads((tmp_path / "slot-01.json").read_bytes())
    assert row["result"]["generation_attempts"][0]["usage"] == {"input_tokens": 37}
    assert row["export_error"] == "RuntimeError" and row["prompt_sha256"] == c.PROMPTS[b.policy]


def test_preflight_journal_failure_stops_before_any_model_call():
    from climate_rag.scifact_bounded_smoke import bounded_runtime_smoke
    calls = []
    b = types.SimpleNamespace(gap=False, count_prompt=lambda *a: 10,
                              generate=lambda *a: calls.append(a))
    def persist(index, phase, record):
        if phase == "started" and index == 0:
            assert record["provider_generate_attempted"] is False
            raise FileExistsError("synthetic partial preflight")
    with pytest.raises(FileExistsError):
        bounded_runtime_smoke(b, persist_case=persist)
    assert calls == []


def completed_fixture(root, monkeypatch):
    p, _ = protocol(monkeypatch)
    policy = c.POLICIES[0]
    directory = root / "runs" / (c.RELEASE + "-" + policy)
    out = directory / "inference"
    out.mkdir(parents=True)
    flight = {"policy": policy, "prompt_sha256": c.PROMPTS[policy], "gap": True,
              "source_git": GIT, "protocol_sha256": SOURCE, "status": "passed", "attempted_calls": 4,
              "records": [{}] * 4, "execution_kind": "local_model"}
    c.write_once(out / "runtime-preflight.json", flight)
    run = {"release_id": c.RELEASE, "arm": policy, "comparison_protocol": c.PAIR,
           "source_git": GIT, "source_archive_sha256": SOURCE, "preparation_source_git": c.PREPARATION_GIT,
           "preparation_compact_sha256": c.COMPACT_SHA, "protocol_sha256": SOURCE,
           "inference_archive_sha256": ARCHIVE, "inference_file_sha256": p["inference_file_sha256"],
           "model_sha256": c.MODEL_SHA, "reranker_sha256": c.RERANKER_SHA,
           "prompt_sha256": c.PROMPTS[policy], "gold_loaded": False,
           "preflight_sha256": c.sha((out / "runtime-preflight.json").read_bytes()), "runs": []}
    for index, spec in enumerate(p["slots_per_arm"], 1):
        raw = spec | {"arm": policy, "source_git": GIT, "comparison_protocol": c.PAIR, "prompt_sha256": c.PROMPTS[policy],
                      "usage": {"input_tokens": 123, "output_tokens": 17}, "generation_attempts": []}
        row = raw | {"result": raw}
        c.write_once(out / f"slot-{index:02d}-raw.json", raw)
        c.write_once(out / f"slot-{index:02d}.json", row)
        run["runs"].append(row)
    c.write_once(out / "run.json", run)
    state = {"status": "complete", "source_git": GIT, "source_archive_sha256": SOURCE,
             "protocol_sha256": SOURCE, "arm": policy, "inference_sha256": c.sha((out / "run.json").read_bytes())}
    c.write_once(directory / "operator-status.json", state)
    return p, out, run


@pytest.mark.parametrize("field", ["source_git", "prompt_sha256", "arm", "model_sha256", "gold_loaded"])
def test_predecessor_fresh_full_binding_and_later_recheck(tmp_path, monkeypatch, field):
    p, out, run = completed_fixture(tmp_path, monkeypatch)
    expected = previous_arm(tmp_path, p, SOURCE, SOURCE, ARCHIVE)
    assert expected == c.sha((out / "run.json").read_bytes())
    run[field] = "changed"
    (out / "run.json").write_bytes(c.encoded(run))
    with pytest.raises(ValueError):
        read_completed_run(out / "run.json", p, SOURCE, SOURCE, ARCHIVE, c.POLICIES[0])
    with pytest.raises(ValueError, match="previous"):
        previous_arm(tmp_path, p, SOURCE, SOURCE, ARCHIVE)


def test_model_archive_filter_does_not_extract_legacy_data(tmp_path, scripts):
    from run_scifact_semantic_operator import model_assets_only
    path = tmp_path / "assets.tar"
    with tarfile.open(path, "w") as arc:
        for name in ("models/generator/model/config.json", "models/reranker/model/config.json",
                     "gold.json", "evidence.jsonl", "args-validation.json"):
            info = tarfile.TarInfo(name)
            info.size = 2
            arc.addfile(info, io.BytesIO(b"{}"))
    model_assets_only(path, tmp_path / "filtered", c.sha(path.read_bytes()))
    assert {p.relative_to(tmp_path / "filtered").as_posix() for p in (tmp_path / "filtered").rglob("*") if p.is_file()} == {
        "models/generator/model/config.json", "models/reranker/model/config.json"}


@pytest.mark.parametrize("field", ["all", "usage", "generation_attempts", "prompt_sha256"])
def test_durable_raw_projection_must_be_complete(tmp_path, monkeypatch, field):
    p, out, _ = completed_fixture(tmp_path, monkeypatch)
    path = out / "slot-01-raw.json"
    raw = json.loads(path.read_bytes())
    if field == "all":
        raw = {}
    else:
        raw.pop(field)
    path.write_bytes(c.encoded(raw))
    with pytest.raises(ValueError, match="durable_raw"):
        read_completed_run(out / "run.json", p, SOURCE, SOURCE, ARCHIVE, c.POLICIES[0])


def test_operator_gold_is_after_both_exits_and_no_old_baseline(scripts):
    import run_scifact_semantic_operator as op
    import run_scifact_semantic_arm as arm
    import score_scifact_semantic_pair as scoring
    text = inspect.getsource(op.main)
    assert text.index("execute(command") < text.index('if policy == POLICIES[1]:') < text.index("extract_scoring_after_both_inference_exits")
    assert text.count("previous_arm(") == 2
    assert "--r2-run" not in inspect.getsource(scoring)
    assert "--gold" not in inspect.getsource(arm)
    assert "LocalQwenSemanticGapProvider" in inspect.getsource(arm)
    assert "gap=True" in inspect.getsource(scoring)
    assert LocalQwenSemanticGapProvider.kind == "local_model"


def test_nei_empty_failure_is_not_correct_abstention(scripts):
    from score_scifact_semantic_pair import terminal_counts
    from climate_rag.scifact_grounding import GoldClaim
    def row(i, outcome, action, status):
        return {"claim_id": i, "prediction": {"evidence": {}}, "result": {"outcome": outcome,
            "generation_attempts": [{"action": action, "status": status}]}}
    rows = [row(1, "model_abstention:insufficient_evidence", "abstain", "valid_decision"),
            row(2, "deadline", None, "terminal_failure")]
    report = terminal_counts([GoldClaim(1, "a", {}, ()), GoldClaim(2, "b", {}, ())], rows)
    assert report["nei_valid_model_abstention"] == report["nei_failure_empty_prediction"] == 1
    assert report["failure_or_budget_termination"] == 1 and report["nonempty_answer_coverage"] == 0
    assert report["claim_verdict_accuracy"] is None


@pytest.mark.parametrize("policy", c.POLICIES)
def test_semantic_runtime_roundtrip_both_are_gap_envelopes(tmp_path, monkeypatch, policy):
    corpus, retrieve = inputs()
    b = backend(TinyTokenizer(), policy, ["c5"])
    b.start_slot = lambda p: p.mkdir()
    (tmp_path / "private-responses").mkdir()
    p = {"execution_source_git": GIT, "slots_per_arm": [{"claim_id": 42, "route": "adaptive"}]}
    rows = ex.execute_slots(p, [{"id": 42, "claim": "Fixture claim"}], b, retrieve, lambda q, s: s, corpus, tmp_path)
    assert rows[0]["result"]["decision_execution_audit"][0]["raw_wire_action"] == "read"
    assert rows[0]["result"]["candidate_wire_audit"]
    assert rows[0]["result"]["model_tool_links"][0]["subsequent_model_attempt"] == 1


def test_package_mechanical_conversion_exact_bundles_and_no_reselection(tmp_path, monkeypatch, scripts):
    p, claims = protocol(monkeypatch)
    selection = [{"id": r["id"], "component": str(r["id"]), "legacy_stratum": "nei"} for r in claims]
    original = tmp_path / "prepared"
    private = original / "private"
    private.mkdir(parents=True)
    payloads = {
        "selected-before-probe.json": {"selection": selection},
        "selected-inference.json": claims,
        "selected-gold.json": [dict(r, evidence={}, cited_doc_ids=[]) for r in claims],
        "current-opportunity.json": selection,
        "consumption-ledger.json": {"component_excluded_eligible_ids": [100], "component_excluded": ["100"]},
        "future-pair-protocol-draft.json": {"ordered_claim_ids": list(range(12)), "prompt_sha256": c.PROMPTS,
             "budget": c.BUDGET, "scope": c.SCOPE, "decoding": c.DECODING, "packing": c.PACKING},
    }
    for name, value in payloads.items():
        c.write_once(private / name, value)
    compact = {"source_git": c.PREPARATION_GIT, "prompt_sha256": c.PROMPTS, "tokenizer_sha256": c.TOKENIZER_SHA,
               "private_output_sha256": {n: c.sha((private / n).read_bytes()) for n in payloads}}
    c.write_once(original / "compact.json", compact)
    compact_sha = c.sha((original / "compact.json").read_bytes())
    monkeypatch.setattr(scripts, "COMPACT_SHA", compact_sha)
    monkeypatch.setattr(c, "COMPACT_SHA", compact_sha)
    corpus = b'{"id":1,"abstract":["Fixture"]}\n'
    archive_path = tmp_path / "input.tar"
    with tarfile.open(archive_path, "w") as archive:
        info = tarfile.TarInfo(c.CORPUS_MEMBER)
        info.size = len(corpus)
        archive.addfile(info, io.BytesIO(corpus))
    monkeypatch.setattr(scripts, "INPUT_SHA", c.sha(archive_path.read_bytes()))
    for module in (scripts, c):
        monkeypatch.setattr(module, "CORPUS_SHA", c.sha(corpus))
    receipt = scripts.package(original, archive_path, tmp_path / "output", GIT)
    assert receipt["model_calls"] == 0 and not receipt["gpu_authorized"] and not receipt["reselected"]
    with tarfile.open(tmp_path / "output/inference.tar") as archive:
        assert set(archive.getnames()) == c.INFERENCE_NAMES
        packaged = json.loads(archive.extractfile("protocol.json").read())
        assert len(packaged["paired_slots"]) == 96
        assert "legacy_stratum" not in json.dumps(packaged)
    with tarfile.open(tmp_path / "output/scoring.tar") as archive:
        assert set(archive.getnames()) == c.SCORING_NAMES
    with pytest.raises(FileExistsError):
        scripts.package(original, archive_path, tmp_path / "output", GIT)
    (private / "selected-gold.json").write_text("[]")
    with pytest.raises(ValueError, match="sha"):
        scripts.package(original, archive_path, tmp_path / "changed", GIT)


@pytest.mark.parametrize("mutation", ["gold", "selection", "review", "manifest"])
def test_scoring_cannot_substitute_self_consistent_labels_or_strata(scripts, monkeypatch, mutation):
    import score_scifact_semantic_pair as scoring
    selection = [{"id": 1, "component": "one", "legacy_stratum": "nei"}]
    gold = [{"id": 1, "claim": "Fixture", "evidence": {}}]
    payload = {"gold.jsonl": c.jsonl(gold), "selected-strata.json": c.encoded(selection),
        "selection-freeze.json": c.encoded({"selection": selection}), "current-opportunity.json": b"[]\n",
        "consumption-ledger.json": b"{}\n", "preparation-draft.json": b"{}\n"}
    hashes = {"selected-gold.json": c.sha(c.encoded(gold)),
        "selected-before-probe.json": c.sha(payload["selection-freeze.json"]),
        "current-opportunity.json": c.sha(payload["current-opportunity.json"]),
        "consumption-ledger.json": c.sha(payload["consumption-ledger.json"]),
        "future-pair-protocol-draft.json": c.sha(payload["preparation-draft.json"])}
    monkeypatch.setattr(scoring, "PRIVATE_SHA", hashes)
    manifest = {"preparation_private_sha256": dict(hashes)}
    scoring.validate_prepared_scoring(payload, manifest)
    if mutation == "gold":
        gold[0]["evidence"] = {"123": []}
        payload["gold.jsonl"] = c.jsonl(gold)
    elif mutation == "selection":
        selection[0]["legacy_stratum"] = "replacement"
        payload["selected-strata.json"] = c.encoded(selection)
    elif mutation == "review":
        payload["current-opportunity.json"] = b"[1]\n"
    else:
        manifest["preparation_private_sha256"] = {}
    # Updating caller-controlled file hashes does not authorize different frozen evidence.
    manifest["scoring_file_sha256"] = {n: c.sha(b) for n, b in payload.items()}
    with pytest.raises(ValueError, match="semantic_prepar"):
        scoring.validate_prepared_scoring(payload, manifest)
