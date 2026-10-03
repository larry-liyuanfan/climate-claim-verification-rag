"""Read-only posthoc evidence/cost audit. Never runs models or changes a run.

Only compact aggregates/hashes are written, exclusively. The execution commit
and this audit's commit are separate identities. Call only after both jobs end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, cast

from pydantic import ValidationError

from climate_rag.agent_protocol import ModelResponseValidationError
from climate_rag.agent_v3 import valid_search_query
from climate_rag.evidence_gap_candidate import CANDIDATE_PROTOCOL, GapProviderAdapter
from climate_rag.public_v2 import file_sha256
from climate_rag.scifact_bounded_comparison import score_bounded_pair
from climate_rag.scifact_bounded_runtime import ARMS
from climate_rag.scifact_grounding import Abstract, GoldClaim, parse_abstract, parse_gold
from climate_rag.scifact_scoring import ClaimPrediction, parse_prediction, score_original
from climate_rag.scifact_terminal import action_schema, parse_action, render_answer, source_from_abstract
from climate_rag.verification import normalise_claim
from run_scifact_bounded_arm import MODEL_SHA, PAIR_SHA, RELEASE, RERANKER_SHA
from score_scifact_bounded_pair import audit_private_wire
from audit_scifact_train_closeout import rescore

TOOLS = {"read", "rewrite", "rerank"}
COMPLETE_FEEDBACK = "tool_completed: inspect current_citable; no semantic conclusion implied"
REPAIR_SUFFIX = "; choose a new legal action; read known preview candidates before citation"
SACCT_FIELDS = ("JobIDRaw", "State", "ExitCode", "ElapsedRaw", "TotalCPU", "CPUTimeRAW", "MaxRSS", "ReqTRES", "AllocTRES")


def require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)  # never include raw/gold content in errors


def read_json(path: Path) -> Any:
    require(not path.is_symlink(), "symlink_input")
    return json.loads(path.read_text(encoding="utf-8"))


def tariff(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Known fields are lower bounds if absent; never fill absent timing with 0."""
    totals = {k: 0 for k in ("input_tokens", "output_tokens")}
    missing: Counter[str] = Counter()
    elapsed: list[float] = []
    for a in records:
        for k in totals:
            value = a.get("usage", {}).get(k)
            if value is None:
                missing[k] += 1
            else:
                require(type(value) is int and value >= 0, "invalid_usage")
                totals[k] += value
        ms = a.get("diagnostics", {}).get("generation_elapsed_ms")
        if ms is not None:
            require(isinstance(ms, (int, float)) and math.isfinite(ms) and ms >= 0, "invalid_generation_time")
            elapsed.append(ms)
    unknown = sum(a.get("usage_known") is False or a.get("diagnostics", {}).get("output_usage_unknown", False)
                  or any(k not in a.get("usage", {}) for k in totals) for a in records)
    return {"calls": len(records), "known_tokens_including_failures": totals,
            "unknown_usage_attempts": unknown, "missing_fields": dict(missing),
            "token_totals_are_lower_bounds": bool(unknown),
            "generation_ms_known_sum": sum(elapsed) if elapsed else None,
            "generation_ms_missing": len(records) - len(elapsed)}


def physical(directory: Path, records: list[dict[str, Any]], gap: bool) -> tuple[dict[str, Any], dict[str, str]]:
    checker = cast(Callable[[Path, list[dict[str, Any]], bool], dict[str, Any]], audit_private_wire)
    response = checker(directory, records, gap)  # existing multiplicity audit
    raw: dict[str, str] = {}
    grammar: Counter[tuple[str, int]] = Counter()
    for p in directory.iterdir():
        require(p.name.endswith(("-response.txt", "-grammar.txt")), "unexpected_private_file")
        payload = p.read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        if p.name.endswith("-response.txt"):
            raw[sha] = payload.decode("utf-8")
        else:
            grammar[sha, len(payload)] += 1
    expected: Counter[tuple[str, int]] = Counter()
    empty = 0
    for a in records:
        d = a["diagnostics"]
        r = d["grammar_log"]
        require(r["truncated"] is False and r["io_failed"] is False and r["dropped_bytes"] == 0
                and r["attempted_bytes"] == r["stored_bytes"] and r["sha256"] == r["stored_prefix_sha256"],
                "incomplete_grammar_receipt")
        if r["stored_bytes"]:
            expected[r["sha256"], r["stored_bytes"]] += 1
        else:
            require(r["sha256"] == hashlib.sha256(b"").hexdigest(), "empty_grammar_hash")
            empty += 1  # lazy sink correctly creates no physical file for an empty log
        require(d["output_sha256"] == d["private_attachment"]["sha256"], "wire_output_hash")
        if "output_tokens" in a.get("usage", {}):
            require(d["output_tokens"] == a["usage"]["output_tokens"], "wire_output_cost")
    require(grammar == expected, "grammar_receipt_multiset")
    return {**response, "grammar_logs": sum(grammar.values()), "empty_grammar_receipts": empty}, raw


def visible_view(trace: dict[str, Any], attempt: dict[str, Any], corpus: dict[int, Abstract],
                 aliases: dict[str, int], source_hashes: dict[str, str]) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    visible: dict[str, Any] = {}
    canonical: set[tuple[int, int]] = set()
    for v in trace["visible"]:
        doc = corpus[v["doc_id"]]
        source = source_from_abstract(doc)
        sid = f'{v["alias"]}:{v["sentence_index"]}'
        require(sid not in visible and 0 <= v["sentence_index"] < len(doc.sentences), "visible_id")
        text = doc.sentences[v["sentence_index"]]
        sha = hashlib.sha256(text.encode()).hexdigest()
        require(v["text_sha256"] == sha and v["source_text_sha256"] == source.text_sha256
                and source_hashes.get(v["alias"]) == source.text_sha256, "visible_corpus_hash")
        require(v["alias"] not in aliases or aliases[v["alias"]] == doc.doc_id, "alias_mutation")
        aliases[v["alias"]] = doc.doc_id
        visible[sid] = {"source_id": str(doc.doc_id), "sentence_index": v["sentence_index"], "text": text,
                        "text_sha256": sha, "source_text_sha256": source.text_sha256}
        canonical.add((doc.doc_id, v["sentence_index"]))
    require(attempt["visible_sentence_sha256"] == {s: v["text_sha256"] for s, v in visible.items()}, "attempt_visible_hash")
    identity = trace["actual_observation"]
    require(identity["visible"] == [{"sentence_id": s, "sha256": v["text_sha256"]} for s, v in visible.items()], "actual_visible_order")
    require(all(p["citable"] is False for p in identity["previews"]), "preview_not_evidence")
    require(attempt["diagnostics"]["actual_observation"] == identity, "provider_observation_identity")
    return visible, canonical


class RecordedBackend:
    """Replay only stored wire through existing GapProviderAdapter, never infer."""
    name, kind, wire_protocol = "posthoc-recorded", "fixture", CANDIDATE_PROTOCOL

    def __init__(self) -> None:
        self.raw = ""
        self.attempt: dict[str, Any] = {}

    def generate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        a = self.attempt
        d = {k: v for k, v in a["diagnostics"].items() if k != "gap_mapping"}
        if a["status"] == "provider_response_invalid" and d.get("category") not in {"gap_span_invalid", "gap_envelope_invalid"}:
            raise ModelResponseValidationError(a["usage"], d)
        # Adapter-generated categories did not exist in the backend response.
        if d.get("category") in {"gap_span_invalid", "gap_envelope_invalid"}:
            d.pop("category")
        return {"raw": self.raw, "usage": a["usage"], "diagnostics": d}


def audit_slot(row: dict[str, Any], gold: GoldClaim, corpus: dict[int, Abstract], raw: dict[str, str], *, gap: bool | None = None) -> dict[str, Any]:
    result = row["result"]
    gap = row["arm"] == ARMS[1] if gap is None else gap
    attempts, traces, events = result["generation_attempts"], result["visible_attempts"], result["events"]
    require(len(attempts) == len(traces) == result["model_calls"], "attempt_trace_count")
    backend = RecordedBackend()
    adapter = GapProviderAdapter(cast(Any, backend)) if gap else None
    cursor = 0
    candidates: list[str] = []
    selected: list[str] = []
    source_hashes: dict[str, str] = {}
    aliases: dict[str, int] = {}
    reads: set[tuple[str, ...]] = set()
    sets: list[set[tuple[int, int]]] = []
    audit: list[dict[str, Any]] = []
    fixed: Counter[str] = Counter()
    links: list[tuple[int, int]] = []
    statuses: Counter[str] = Counter()
    raw_actions: Counter[str] = Counter()
    last_answer = None
    rewritten = reranked = False
    tool_count = 0

    def consume(event: dict[str, Any]) -> None:
        nonlocal candidates, selected, source_hashes, rewritten, reranked, tool_count
        if event.get("status") == "skipped":
            return
        require(event["before_context"] == selected, "event_before_context")
        tool_count += 1
        if event["status"] == "completed":
            if event["tool"] == "rerank":
                require(set(event["candidate_ids"]) == set(candidates), "rerank_candidate_universe")
                reranked = True
            if event["tool"] == "rewrite":
                rewritten = True
            candidates = event["candidate_ids"]
            selected = event["requested_context"]
            require(len(candidates) == len(set(candidates)) and set(selected) <= set(candidates), "event_candidate_shape")
            source_hashes = event["source_sha256"]
            require(set(source_hashes) == set(candidates), "event_source_hash_keys")

    while cursor < len(events) and not events[cursor].get("model_selected") and "tool" in events[cursor]:
        event = events[cursor]
        fixed[event["tool"] + ":" + event["status"]] += 1
        consume(event)
        cursor += 1
    for i, (a, trace) in enumerate(zip(attempts, traces, strict=True)):
        require(trace["attempt_index"] == i and a["requested_context"] == selected, "attempt_context")
        visible, current = visible_view(trace, a, corpus, aliases, source_hashes)
        require(all(s.rsplit(":", 1)[0] in selected for s in visible), "visible_outside_requested_context")
        expected_actions = ["abstain"] + (["answer"] if visible else [])
        if row["route"] == "adaptive" and i + 1 < result["budget"]["max_calls"] and tool_count < result["budget"]["max_tools"]:
            if candidates:
                expected_actions.append("read")
            if not rewritten:
                expected_actions.append("rewrite")
            if candidates and not reranked:
                expected_actions.append("rerank")
        require(a["allowed_actions"] == expected_actions, "allowed_actions_reconstruction")
        sets.append(current)
        schema = action_schema(a["allowed_actions"], candidates, list(visible), result["budget"]["context_k"])
        original = raw[a["diagnostics"]["private_attachment"]["sha256"]]
        mapped = original
        provider_invalid = a["status"] == "provider_response_invalid"
        if adapter is not None:
            backend.raw, backend.attempt = original, a
            observation = {"immutable_claim": normalise_claim(gold.claim), "feedback": trace["feedback"],
                           "current_citable": [{"sentence_id": s, "text": v["text"]} for s, v in visible.items()]}
            try:
                mapped = adapter.generate(observation, schema, 512, 120)["raw"]
                require(not provider_invalid, "mapping_unexpected_success")
            except ModelResponseValidationError:
                require(provider_invalid, "mapping_unexpected_failure")
            require(adapter.records[-1] == result["candidate_wire_audit"][i]
                    == a["diagnostics"]["gap_mapping"], "mapping_history_or_cost")
        decision = None
        error = None
        if not provider_invalid and a["status"] != "terminal_failure":
            try:
                decision = parse_action(json.loads(mapped), a["allowed_actions"], visible, candidates, result["budget"]["context_k"])
                if decision["action"] == "read" and (decision["source_ids"] == selected or tuple(decision["source_ids"]) in reads):
                    error = "read_loop"
                if decision["action"] == "rewrite" and not valid_search_query(normalise_claim(gold.claim), normalise_claim(decision["query"])):
                    error = "rewrite_constraint_or_loop"
            except ValidationError:
                error = "invalid_schema"
            except json.JSONDecodeError:
                error = "invalid_json"
            except ValueError as exc:
                error = str(exc)
            require((a["status"] == "validation_failed") == (error is not None), "strict_status_mismatch")
            if error:
                require(a["error_code"] == error, "strict_error_mismatch")
            if decision is not None:
                require(a.get("action") == decision["action"], "strict_action_mismatch")
        event_index = None
        if a["status"] == "valid_decision":
            require(decision is not None, "valid_decision_missing")
            assert decision is not None
            if decision["action"] in TOOLS and cursor < len(events) and events[cursor].get("model_selected"):
                event_index = cursor
                event = events[cursor]
                require(event["tool"] == decision["action"], "event_action_mismatch")
                if event["status"] == "completed" and decision["action"] == "read":
                    require(event["requested_context"] == decision["source_ids"], "read_arguments")
                    reads.add(tuple(decision["source_ids"]))
                if decision["action"] == "rewrite":
                    require(event["query_sha256"] == hashlib.sha256(normalise_claim(decision["query"]).encode()).hexdigest(), "rewrite_arguments")
                consume(event)
                links.append((i, event_index))
                cursor += 1
            elif decision["action"] == "answer":
                last_answer = render_answer(decision, visible)
                require(i == len(attempts) - 1, "answer_not_terminal")
            elif decision["action"] == "abstain":
                require(i == len(attempts) - 1, "abstain_not_terminal")
        if i + 1 < len(attempts):
            expected = COMPLETE_FEEDBACK if event_index is not None and events[event_index]["status"] == "completed" else a.get("error_code", "") + REPAIR_SUFFIX
            require(traces[i + 1]["feedback"] == expected, "next_feedback_mismatch")
        next_a = attempts[i + 1] if i + 1 < len(attempts) else None
        audit.append({"attempt_index": i, "raw_wire_action": a["diagnostics"]["raw_wire_action"],
                      "strict_action": a.get("action"), "strict_status": a["status"], "actual_event_index": event_index,
                      "actual_event_status": events[event_index]["status"] if event_index is not None else None,
                      "next_attempt_index": i + 1 if next_a is not None else None,
                      "next_feedback": traces[i + 1]["feedback"] if next_a is not None else None,
                      "next_strict_action": next_a.get("action") if next_a is not None else None,
                      "next_strict_status": next_a["status"] if next_a is not None else None})
        statuses[a["status"]] += 1
        observed_action = a["diagnostics"].get("raw_wire_action")
        raw_actions[observed_action if observed_action in TOOLS | {"answer", "abstain"} else "unparsed_or_other_redacted"] += 1
    require(not any(e.get("model_selected") for e in events[cursor:]), "unlinked_model_event")
    require(tool_count == result["tool_calls"], "tool_count")
    require(result["decision_execution_audit"] == audit and not result["trace_unlinked_events"], "recorded_execution_audit")
    require(result["answer"] == last_answer, "final_answer_not_from_terminal_wire")
    expected_links = [{"attempt_index": x["attempt_index"], "event_index": x["actual_event_index"],
                       "proposal_status": x["strict_status"], "subsequent_model_attempt": x["next_attempt_index"]}
                      for x in audit if x["strict_action"] in TOOLS]
    require(result["model_tool_links"] == expected_links, "model_tool_links")
    require(not gap or len(result["candidate_wire_audit"]) == len(attempts), "gap_record_count")
    require(not traces or result["initial_context_identity"] == traces[0]["actual_observation"], "initial_context_identity")
    prediction = parse_prediction(row["prediction"], corpus)
    require(result["validation_repairs"] == min(sum(a["status"] in {"validation_failed", "provider_response_invalid"} for a in attempts), result["budget"]["max_repairs"]), "repair_cost_count")
    chain = evidence_chain(prediction, gold, sets, attempts, events, links, result["outcome"])
    return {"fixed_tools": dict(fixed), "attempt_statuses": dict(statuses), "raw_action_proposals": dict(raw_actions),
            "chain": chain, "validation_repairs": result["validation_repairs"], "cost": tariff(attempts)}


def evidence_chain(prediction: ClaimPrediction, gold: GoldClaim, views: list[set[tuple[int, int]]],
                   attempts: list[dict[str, Any]], events: list[dict[str, Any]], links: list[tuple[int, int]],
                   outcome: str | None = None) -> dict[str, int]:
    keys = ("model_events", "execution_success", "next_observation_available", "added_citable_evidence",
            "then_legal_decision", "then_final_cites_added", "then_gold_label_complete_rationale",
            "success_without_observable_next_context", "successful_event_with_next_legal_decision",
            "newly_seen_evidence", "newly_seen_then_legal_decision", "newly_seen_then_final_cites",
            "newly_seen_then_gold_label_complete_rationale", "then_strict_whole_final_answer_correct",
            "newly_seen_then_strict_whole_final_answer_correct")
    counts: Counter[str] = Counter({k: 0 for k in keys})
    credited_docs: set[int] = set()
    newly_credited_docs: set[int] = set()
    structural_match = strict_whole_answer(prediction, gold)
    last = attempts[-1] if attempts else {}
    valid_terminal = (last.get("status") == "valid_decision" and (
        last.get("action") == "answer" and outcome == "ids_validated_semantics_unmeasured"
        or last.get("action") == "abstain" and bool(outcome and outcome.startswith("model_abstention:"))))
    whole_correct = structural_match and valid_terminal
    cited = {(d, i) for d, p in prediction.evidence.items() for i in p.sentences}
    for attempt_index, event_index in links:
        counts["model_events"] += 1
        if events[event_index]["status"] != "completed":
            continue
        counts["execution_success"] += 1
        if attempt_index + 1 >= len(views):
            counts["success_without_observable_next_context"] += 1
            continue
        counts["next_observation_available"] += 1
        added = views[attempt_index + 1] - views[attempt_index]
        seen_before = set().union(*views[:attempt_index + 1])
        newly_seen = views[attempt_index + 1] - seen_before
        legal = attempts[attempt_index + 1]["status"] == "valid_decision"
        counts["successful_event_with_next_legal_decision"] += legal
        counts["newly_seen_evidence"] += bool(newly_seen)
        counts["newly_seen_then_legal_decision"] += bool(newly_seen) and legal
        counts["newly_seen_then_final_cites"] += bool(newly_seen & cited) and legal
        if not added:
            continue
        counts["added_citable_evidence"] += 1
        if not legal:
            continue
        counts["then_legal_decision"] += 1
        if not added & cited:
            continue
        counts["then_final_cites_added"] += 1
        closed: set[int] = set()
        newly_closed: set[int] = set()
        for doc_id, pred in prediction.evidence.items():
            single = ClaimPrediction(prediction.claim_id, {doc_id: pred})
            official = score_original([gold], [single])["metrics"]["abstract_rationalized"]["correct"]
            if official and any(set(r.sentences) <= set(pred.sentences[:3])
                                and any((doc_id, i) in added for i in r.sentences)
                                for r in gold.evidence[doc_id]):
                closed.add(doc_id)
            if official and any(set(r.sentences) <= set(pred.sentences[:3])
                                and any((doc_id, i) in newly_seen for i in r.sentences)
                                for r in gold.evidence[doc_id]):
                newly_closed.add(doc_id)
        counts["then_gold_label_complete_rationale"] += bool(closed)
        counts["newly_seen_then_gold_label_complete_rationale"] += bool(newly_closed)
        counts["then_strict_whole_final_answer_correct"] += bool(closed) and whole_correct
        counts["newly_seen_then_strict_whole_final_answer_correct"] += bool(newly_closed) and whole_correct
        credited_docs.update(closed)
        newly_credited_docs.update(newly_closed)
    counts["unique_final_correct_docs_associated_with_chain"] = len(credited_docs)
    counts["unique_final_correct_docs_associated_with_newly_seen_chain"] = len(newly_credited_docs)
    counts["strict_whole_final_answer_correct_slots"] = int(whole_correct)
    counts["structural_whole_gold_match_slots"] = int(structural_match)
    return dict(counts)


def strict_whole_answer(prediction: ClaimPrediction, gold: GoldClaim) -> bool:
    """Conservative custom diagnostic, NOT a new official accuracy metric."""
    if set(prediction.evidence) != set(gold.evidence):
        return False
    for doc_id, pred in prediction.evidence.items():
        rats = gold.evidence[doc_id]
        all_gold_sentences = {i for r in rats for i in r.sentences}
        if (pred.label != rats[0].label or not set(pred.sentences) <= all_gold_sentences
                or not any(set(r.sentences) <= set(pred.sentences[:3]) for r in rats)):
            return False
    return True


def sum_reports(reports: list[dict[str, Any]], field: str) -> dict[str, int]:
    result: Counter[str] = Counter()
    for r in reports:
        result.update(r[field])
    return dict(result)


def members(path: Path, expected: str, allowed: set[str]) -> dict[str, bytes]:
    require(file_sha256(path) == expected, "data_archive_hash")
    with tarfile.open(path) as bundle:
        entries = bundle.getmembers()
        require(len(entries) == len(allowed) and {m.name for m in entries} == allowed
                and all(m.isfile() and m.size <= 64 * 1024 * 1024 for m in entries), "data_archive_members")
        result = {}
        for m in entries:
            handle = bundle.extractfile(m)
            require(handle is not None, "missing_archive_member")
            assert handle is not None
            result[m.name] = handle.read()
        return result


def slurm_snapshot(job_ids: list[int]) -> list[dict[str, Any]]:
    fields = ",".join(k + ("%256" if k.endswith("TRES") else "") for k in SACCT_FIELDS)
    done = subprocess.run(["sacct", "-j", ",".join(map(str, job_ids)), "-nP", "-o", fields],
                          check=True, capture_output=True, text=True, timeout=20)
    result = []
    for line in done.stdout.splitlines():
        values = line.split("|")
        require(len(values) == len(SACCT_FIELDS), "sacct_field_count")
        result.append(dict(zip(SACCT_FIELDS, (v or None for v in values), strict=True)))
    return result


def closeout(root: Path, submission: dict[str, Any], protocol: dict[str, Any],
             slurm: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [j["job_id"] for j in submission["jobs"]]
    top = {r["JobIDRaw"]: r for r in slurm}
    require(all(str(j) in top for j in ids), "missing_slurm_identity")
    # No run/score/gold read until BOTH scheduler records say completed success.
    require(all(top[str(j)]["State"] == "COMPLETED" and top[str(j)]["ExitCode"] == "0:0" for j in ids), "pair_not_completed")
    require(protocol["release_id"] == submission["release_id"] == RELEASE and protocol["arms"] == list(ARMS)
            and submission["jobs"][1]["dependency"] == f"afterok:{ids[0]}", "release_identity")
    require(file_sha256(Path(submission["source_archive"])) == submission["source_archive_sha256"]
            and file_sha256(Path(submission["wrapper"])) == submission["wrapper_sha256"], "execution_archive_wrapper_hash")
    with tarfile.open(submission["source_archive"]) as source:
        marker = [m for m in source.getmembers() if m.name == "SOURCE_REVISION"]
        require(len(marker) == 1 and marker[0].isfile() and marker[0].size == 41, "execution_source_marker")
        handle = source.extractfile(marker[0])
        require(handle is not None, "execution_source_marker_missing")
        assert handle is not None
        require(handle.read() == (submission["execution_source_git"] + "\n").encode(), "execution_source_revision")
    paths = [root / "runs" / (RELEASE + "-" + arm) for arm in ARMS]
    operators = [read_json(p / "operator-status.json") for p in paths]
    runs = [read_json(p / "inference/run.json") for p in paths]
    score = read_json(paths[1] / "score.json")
    for j, arm, p, op, run in zip(ids, ARMS, paths, operators, runs, strict=True):
        require(op["status"] == "complete" and op["stage"] == "complete" and op["job_id"] == str(j)
                and op["arm"] == run["arm"] == arm and op["release_id"] == run["release_id"] == RELEASE
                and op["source_git"] == run["source_git"] == submission["execution_source_git"]
                and op["source_archive_sha256"] == submission["source_archive_sha256"]
                and op["pair_protocol_sha256"] == run["pair_protocol_sha256"] == PAIR_SHA
                and op["inference_sha256"] == file_sha256(p / "inference/run.json")
                and op["official_dev_read"] is False and run["gold_loaded"] is False
                and op["train_gold_extracted"] == (arm == ARMS[1])
                and run["model_sha256"] == MODEL_SHA and run["reranker_sha256"] == RERANKER_SHA
                and run["protocol_sha256"] == protocol["parent_protocol_sha256"]
                and run["inference_file_sha256"] == protocol["inference_file_sha256"], "operator_run_join")
        require([(r["claim_id"], r["route"], r["arm"]) for r in run["runs"]]
                == [(c, route, arm) for c in protocol["ordered_claim_ids"] for route in protocol["routes"]], "ordered_48_slots")
        require(all(op["archives"][name]["sha256"] == submission["asset_sha256"][key]
                    for name, key in (("input", "model_input_archive"), ("runtime", "runtime_archive"),
                                      ("overlay", "overlay_archive"), ("grammar", "grammar"))), "operator_asset_hashes")
    require(score["source_git"] == submission["execution_source_git"] and score["pair_protocol_sha256"] == PAIR_SHA
            and operators[1]["score_sha256"] == file_sha256(paths[1] / "score.json")
            and score["run_sha256"] == {a: file_sha256(p / "inference/run.json") for a, p in zip(ARMS, paths, strict=True)}, "score_join")
    infer = members(root / "envs" / f'scifact-train-inference-{protocol["inference_tar_sha256"]}.tar', protocol["inference_tar_sha256"], {"claims.jsonl", "corpus.jsonl", "protocol.json"})
    private = members(root / "envs" / f'scifact-train-scoring-{protocol["scoring_tar_sha256"]}.tar', protocol["scoring_tar_sha256"], {"gold.jsonl", "selected-strata.json", "manifest.json"})
    manifest = json.loads(private["manifest.json"])
    for name, payload in infer.items():
        require(hashlib.sha256(payload).hexdigest() == manifest["output_sha256"]["inference/" + name], "inference_member_hash")
    for name in ("gold.jsonl", "selected-strata.json"):
        require(hashlib.sha256(private[name]).hexdigest() == manifest["output_sha256"]["private/" + name], "gold_member_hash")
    corpus = {d.doc_id: d for d in (parse_abstract(json.loads(line)) for line in infer["corpus.jsonl"].splitlines())}
    gold = [parse_gold(json.loads(line), corpus) for line in private["gold.jsonl"].splitlines()]
    by_id = {g.claim_id: g for g in gold}
    selection = json.loads(private["selected-strata.json"])
    strata = {r["id"]: r["stratum"] for r in selection}
    recomputed = score_bounded_pair(gold, corpus, runs[0]["runs"] + runs[1]["runs"], selection)
    require(all(score[k] == v for k, v in recomputed.items()), "existing_official_score_changed")
    arms = {}
    for arm, run, op, path in zip(ARMS, runs, operators, paths, strict=True):
        pre = read_json(path / "inference/runtime-preflight.json")
        require(run["preflight_sha256"] == file_sha256(path / "inference/runtime-preflight.json")
                and pre["status"] == "passed" and len(pre["records"]) == pre["attempted_calls"] == 4
                and pre["official_data_read"] is False, "preflight_identity")
        prefix = path / "inference/private-responses"
        pre_physical, _ = physical(prefix / "preflight", pre["records"], arm == ARMS[1])
        reports, wires = [], []
        for i, row in enumerate(run["runs"], 1):
            require(read_json(path / f"inference/slot-{i:02d}.json") == row, "durable_slot_identity")
            durable = read_json(path / f"inference/slot-{i:02d}-raw.json")
            require(all(row["result"].get(k) == v for k, v in durable.items())
                    and set(row["result"]) - set(durable) == {"decision_execution_audit", "trace_unlinked_events", "model_tool_links", "initial_context_identity"},
                    "durable_cost_trace_identity")
            wire, raws = physical(prefix / f"slot-{i:02d}", row["result"]["generation_attempts"], arm == ARMS[1])
            reports.append(audit_slot(row, by_id[row["claim_id"]], corpus, raws))
            wires.append(wire)
        routes: dict[str, Any] = {}
        for route in protocol["routes"]:
            rr = [r for r, row in zip(reports, run["runs"], strict=True) if row["route"] == route]
            routes[route] = {k: sum_reports(rr, k) for k in ("fixed_tools", "attempt_statuses", "raw_action_proposals", "chain")}
            quality = recomputed["paired_quality"]
            routes[route]["official_point_score"] = quality[arm]["routes"][route]["official_point_score"] if quality else None
            if quality:
                independent = cast(Callable[[Any, Any], dict[str, Any]], rescore)(
                    {g.claim_id: g.gold_row() for g in gold},
                    {r["claim_id"]: r["prediction"] for r in run["runs"] if r["route"] == route})
                original_metrics = routes[route]["official_point_score"]["metrics"]
                require(all(independent[m][k] == original_metrics[m][k] for m in independent
                            for k in ("correct", "predicted", "relevant")), "independent_official_counts")
        attempts = [a for r in run["runs"] for a in r["result"]["generation_attempts"]]
        self_reports: Counter[str] = Counter()
        if arm == ARMS[1]:
            for row in run["runs"]:
                records = row["result"]["candidate_wire_audit"]
                state = records[0].get("model_reported_evidence_state") if records else None
                if state:
                    key = (row["route"], strata[row["claim_id"]], state["retrieval_need"],
                           state["relevance"], state["support"], records[0]["mapped_action"]["action"])
                    self_reports["|".join(key)] += 1
        arms[arm] = {"slots": len(run["runs"]), "routes": routes, "train_cost": tariff(attempts),
                     "initial_fallible_self_report_counts_route_stratum_need_relevance_support_action": dict(self_reports),
                     "synthetic_preflight_cost": tariff(pre["records"]), "preflight_wire_receipts": pre_physical,
                     "train_wire_receipts": {k: sum(w[k] for w in wires) for k in wires[0]},
                     "repairs": sum(r["validation_repairs"] for r in reports),
                     "timing_ms_overlapping_not_additive": {"operator": op["elapsed_seconds"] * 1000,
                         "model_load": run["model_load_ms"], "reranker_load": run["reranker_load_ms"],
                         "bm25_build": run["bm25_build_ms"], "synthetic_preflight": pre["elapsed_ms"],
                         "train_questions_sum": sum(r["result"]["elapsed_ms"] for r in run["runs"]),
                         "tool_events_sum": sum(e["elapsed_ms"] for r in run["runs"] for e in r["result"]["events"] if "elapsed_ms" in e)}}
    return {"schema_version": "scifact-bounded-posthoc-closeout-v1", "status": "audited_compact",
            "execution_source_git": submission["execution_source_git"], "source_archive_sha256": submission["source_archive_sha256"],
            "pair_protocol_sha256": PAIR_SHA, "job_ids": ids, "slurm": slurm,
            "identity_checks_passed": True, "score_recomputed_equal": True,
            "independent_official_counts_equal": True if recomputed["paired_quality"] else None,
            "initial_contexts_comparable": recomputed["comparability"]["all_comparable"],
            "run_sha256": score["run_sha256"], "score_sha256": file_sha256(paths[1] / "score.json"), "arms": arms,
            "boundaries": ["Biased eligible-train diagnostic; not dev/test or causal proof of tool benefit.",
                "then_* counts form event-level funnels; successful_event_with_next_legal_decision is independent of new-evidence discovery. Unique credited documents are deduplicated within each slot.",
                "Immediate added evidence differs from newly seen evidence (outside all prior citable views); rereading A after A->B->A is not a discovery.",
                "Observed evidence-use closure is not proof that the tool caused correctness; post-tool decisions or self-reports alone do not count.",
                "Strict whole-answer correctness additionally requires a legal final answer/abstain with matching termination outcome: exact gold document set, correct labels, complete alternative rationale within first three sentences per document, and no nongold rationale sentences. Failed empty predictions are not successful NEI. Structural matching and partial official document credit are separate.",
                "Preview hashes and self-reports are never citable evidence.",
                "Physical full-wire receipts bind reported token ledgers; no independent tokenizer retokenization is claimed.",
                "Generation/tool/question/operator/Slurm times overlap and must not be added.",
                "Blank Slurm values remain null, not zero; GPU allocation time is not GPU utilization.",
                "Raw wire, gold, row-level evidence and IDs remain on Spartan."]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--audit-git", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    require(not args.output.exists(), "no_audit_overwrite")
    require(args.output.resolve().is_relative_to(args.root.resolve() / "posthoc"), "output_must_be_separate_posthoc_directory")
    require(len(args.audit_git) == 40 and all(c in "0123456789abcdef" for c in args.audit_git), "audit_git_required")
    require(file_sha256(args.protocol) == PAIR_SHA, "paired_protocol_sha")
    submission = read_json(args.submission)
    slurm = slurm_snapshot([j["job_id"] for j in submission["jobs"]])
    result = closeout(args.root, submission, read_json(args.protocol), slurm)
    result.update(posthoc_audit_git=args.audit_git, posthoc_script_sha256=file_sha256(Path(__file__)))
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "output_sha256": file_sha256(args.output)}))


if __name__ == "__main__":
    main()
