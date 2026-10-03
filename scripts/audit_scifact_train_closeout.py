"""Read-only, stdlib-only audit of the completed r2 train diagnostic.

Run on Spartan, never copy its inputs locally. This does not import the scorer,
run a model, reconstruct prompts, change artifacts, or open dev/test data.
Stdout contains only fixed-key counts, booleans and provenance hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from collections import Counter
from pathlib import Path

ROUTES = ("fixed_retrieval", "fixed_rerank", "deterministic_extra", "adaptive")
STRATA = ("initial_doc_opportunity", "top20_doc_replenishable", "gold_absent_top20", "nei")
TOOLS = {"read", "rewrite", "rerank"}
ACTIONS = TOOLS | {"answer", "abstain"}
OUTCOMES = {"ids_validated_semantics_unmeasured", "validation_repair_exhausted",
            "model_abstention:insufficient_evidence"}
ERRORS = {"invalid_schema", "invalid_json", "invalid_model_output", "read_loop",
          "duplicate_or_excess_documents", "total_sentence_budget",
          "duplicate_sentence_reference", "sentence_not_currently_visible",
          "cross_document_reference", "disallowed_action"}
METRICS = ("abstract_label_only", "abstract_rationalized", "sentence_selection", "sentence_label")
SOURCE = "80fcd07faed28890b70096386c5897f0343d3104"
INFERENCE = "92846f0904992e7b7137e550c46cd0c76b2f77b4f808a6d6be25482f90bf1cca"
SCORING = "174ea64c476fc3abdd5575d53e9a1808c8fd5e402e07cf327e58295346d41444"
FROZEN = {
    "operator-status.json": "caae8c0367ff898872ca54d5ed7d747d6276c8413465ceeded761dda51be2a42",
    "score.json": "535460dfef5430444fedeb46f980ed5fa090d4299ac885deee5c8b964ea4d844",
    "inference/run.json": "23566957ed986dde054b61a74529350984b1f018a27f69ec771bc4223dcd1c2b",
    "inference/runtime-preflight.json": "ee9c3eda11fd0194536bb704a3faad863eb4de253d615fc04632784fd9e19121",
}


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def require(condition):
    if not condition:
        raise ValueError("audit contract failed; private values suppressed")


def checked(path, expected):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size < 50_000_000)
    payload = path.read_bytes()
    require(sha(payload) == expected)
    return payload


def archive(path, expected, names):
    checked(path, expected)
    with tarfile.open(path) as bundle:
        members = bundle.getmembers()
        require(len(members) == len(names) and {m.name for m in members} == set(names))
        require(all(m.isfile() and m.size < 40_000_000 for m in members))
        return {name: bundle.extractfile(name).read() for name in names}


def metric(correct, predicted, relevant):
    precision, recall = correct / predicted if predicted else 0, correct / relevant if relevant else 0
    return dict(correct=correct, predicted=predicted, relevant=relevant,
                precision=precision, recall=recall,
                f1=2 * correct / (predicted + relevant) if predicted + relevant else 0)


def rescore(gold, predictions):
    """Independent count arithmetic; original first-three/alternative rules."""
    require(set(gold) == set(predictions))
    gd = gs = pd = ps = label = rationalized = selected = selected_label = 0
    for key, g in gold.items():
        evidence = g["evidence"]
        gd += len(evidence)
        gs += sum(len(r["sentences"]) for rats in evidence.values() for r in rats)
        for rats in evidence.values():
            require(len({r["label"] for r in rats}) == 1)
            require(sum(len(r["sentences"]) for r in rats)
                    == len({s for r in rats for s in r["sentences"]}))
        for doc, p in predictions[key]["evidence"].items():
            pd += 1
            ps += len(p["sentences"])
            rats = evidence.get(doc, [])
            if not rats:
                continue
            ok = p["label"] == rats[0]["label"]
            label += ok
            rationalized += ok and any(set(r["sentences"]) <= set(p["sentences"][:3]) for r in rats)
            credited = {s for r in rats if set(r["sentences"]) <= set(p["sentences"])
                        for s in r["sentences"]}
            selected += len(credited)
            selected_label += len(credited) * ok
    return dict(zip(METRICS, [metric(label, pd, gd), metric(rationalized, pd, gd),
                             metric(selected, ps, gs), metric(selected_label, ps, gs)]))


def opportunities(gold, visible):
    return sum(any(len(r["sentences"]) <= 3 and set(r["sentences"]) <= set(visible.get(doc, []))
                   for r in rats) for doc, rats in gold["evidence"].items())


class ObjectPairs(list):
    """Preserve JSON object identity and duplicate keys, unlike an ordinary list."""


def wire_action(attempt, response_index):
    """Hash joins recover bytes only remotely; every attempt counted separately.

    No arbitrary raw substring is exported. A missing/truncated payload is unknown,
    not evidence of no tool intent. Top-level action is intent, not schema validity.
    """
    receipt = attempt.get("diagnostics", {}).get("private_attachment", {})
    if (not isinstance(receipt, dict) or receipt.get("io_failed")
            or receipt.get("truncated") or receipt.get("dropped_bytes") != 0):
        return "unknown_unavailable"
    raw = response_index.get(receipt.get("sha256"))
    if raw is None or len(raw) != receipt.get("attempted_bytes"):
        return "unknown_unavailable"
    try:
        pairs = json.loads(raw, object_pairs_hook=ObjectPairs)
        if not isinstance(pairs, ObjectPairs):
            return "unknown_non_object_json"
        actions = [v for k, v in pairs if k == "action"]
        if len(actions) == 1 and isinstance(actions[0], str) and actions[0] in ACTIONS:
            return actions[0]
        return "unknown_ambiguous_json"
    except (ValueError, TypeError):
        return "unknown_unparseable_json"


def wire_order(attempt, response_index):
    action = wire_action(attempt, response_index)
    if action.startswith("unknown_"):
        return "unknown", "unknown"
    receipt = attempt["diagnostics"]["private_attachment"]
    pairs = json.loads(response_index[receipt["sha256"]], object_pairs_hook=ObjectPairs)
    fields = [key for key, _ in pairs]
    first = fields[0] if fields[0] in {"action", "documents", "source_ids", "query", "reason"} else "other_redacted"
    return first, str(fields.index("action") + 1)


def receipt_totals(attempts):
    output = {}
    affected = set()
    for kind in ("private_attachment", "grammar_log"):
        c = Counter()
        for index, attempt in enumerate(attempts):
            r = attempt.get("diagnostics", {}).get(kind)
            if not isinstance(r, dict):
                c["missing"] += 1
                affected.add(index)
                continue
            c["present"] += 1
            for k in ("attempted_bytes", "stored_bytes", "dropped_bytes"):
                require(type(r[k]) is int and r[k] >= 0)
                c[k] += r[k]
            require(r["attempted_bytes"] == r["stored_bytes"] + r["dropped_bytes"])
            require(r["truncated"] == (r["dropped_bytes"] > 0))
            for k in ("truncated", "io_failed"):
                require(type(r[k]) is bool)
                c[k] += r[k]
            if r["truncated"] or r["io_failed"]:
                affected.add(index)
        output[kind] = {k: c[k] for k in ("present", "missing", "attempted_bytes", "stored_bytes",
                                         "dropped_bytes", "truncated", "io_failed")}
    output["affected_attempts_union"] = len(affected)
    return output


def verify_cost(result):
    attempts = result["generation_attempts"]
    require(result["model_calls"] == len(attempts))
    require(all(type(a["usage_known"]) is bool for a in attempts))
    require(result["unknown_usage_attempts"] == sum(not a["usage_known"] for a in attempts))
    for k in ("input_tokens", "output_tokens"):
        require(all(type(a["usage"].get(k, 0)) is int and a["usage"].get(k, 0) >= 0 for a in attempts))
        require(sum(a["usage"].get(k, 0) for a in attempts) == result["usage"][k])


def quantile(values, q):
    ordered = sorted(values)
    require(ordered and all(math.isfinite(v) and v >= 0 for v in ordered))
    index = (len(ordered) - 1) * q
    lower = int(index)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (index - lower)


def unique_used_hashes(doc_hashes, event_hashes):
    by_hash = {}
    for doc_id, digest in doc_hashes.items():
        by_hash.setdefault(digest, []).append(str(doc_id))
    require(all(len(by_hash.get(digest, [])) == 1 for digest in event_hashes.values()))
    return {alias: by_hash[digest][0] for alias, digest in event_hashes.items()}


def reconcile_private_files(directory, attempts):
    """Multiplicity matters even when several calls emitted identical bytes."""
    expected, actual, responses = Counter(), Counter(), {}
    for attempt in attempts:
        for field, kind in (("private_attachment", "response"), ("grammar_log", "grammar")):
            receipt = attempt.get("diagnostics", {}).get(field)
            require(isinstance(receipt, dict))
            if receipt["stored_bytes"] > 0:
                expected[kind, receipt["stored_prefix_sha256"], receipt["stored_bytes"]] += 1
    for path in directory.iterdir():
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1048576)
        kind = "response" if path.name.endswith("-response.txt") else "grammar"
        require(path.name.endswith(f"-{kind}.txt"))
        raw = path.read_bytes()
        digest = sha(raw)
        actual[kind, digest, len(raw)] += 1
        if kind == "response":
            responses[digest] = raw
    require(expected == actual)
    return responses, sum(actual.values()), sum(k[2] * count for k, count in actual.items())


def audit(root):
    run_dir = root / "runs/climate-scifact-train-diagnostic-20260930-r2"
    files = {k: json.loads(checked(run_dir / k, v)) for k, v in FROZEN.items()}
    run, score = files["inference/run.json"], files["score.json"]
    pre, op = files["inference/runtime-preflight.json"], files["operator-status.json"]
    require(run["source_git"] == score["source_git"] == op["source_git"] == SOURCE)
    require(op["status"] == "complete" and op["official_dev_read"] is False)
    require(run["gold_loaded"] is False and score["independent_test"] is False)
    require(score["run_sha256"] == FROZEN["inference/run.json"])
    require(op["score_sha256"] == FROZEN["score.json"])
    require(run["preflight_sha256"] == FROZEN["inference/runtime-preflight.json"])
    require(pre["execution_kind"] == "real_model" and pre["status"] == "passed")
    inf = archive(root / f"envs/scifact-train-inference-{INFERENCE}.tar", INFERENCE,
                  ["claims.jsonl", "corpus.jsonl", "protocol.json"])
    private = archive(root / f"envs/scifact-train-scoring-{SCORING}.tar", SCORING,
                      ["gold.jsonl", "selected-strata.json", "manifest.json"])
    manifest = json.loads(private["manifest.json"])
    for name, raw in inf.items():
        require(sha(raw) == manifest["output_sha256"]["inference/" + name])
    for name in ("gold.jsonl", "selected-strata.json"):
        require(sha(private[name]) == manifest["output_sha256"]["private/" + name])
    protocol = json.loads(inf["protocol.json"])
    require(sha(inf["protocol.json"]) == run["protocol_sha256"])
    require(run["inference_file_sha256"] == {n: sha(inf[n]) for n in ("claims.jsonl", "corpus.jsonl")})
    gold = {r["id"]: r for r in map(json.loads, private["gold.jsonl"].splitlines())}
    corpus = {r["doc_id"]: r for r in map(json.loads, inf["corpus.jsonl"].splitlines())}
    selection = json.loads(private["selected-strata.json"])
    sel = {r["id"]: r for r in selection}
    rows = run["runs"]
    require(len(gold) == len(selection) == len(sel) == 12 and len(corpus) == 5183)
    require(set(gold) == set(sel) and len({r["component"] for r in selection}) == 12)
    require(Counter(r["stratum"] for r in selection) == {s: 3 for s in STRATA})
    require([{k: r[k] for k in ("claim_id", "route")} for r in rows] == protocol["slots"])
    require(len(rows) == 48 and len({(r["claim_id"], r["route"]) for r in rows}) == 48)
    private_dir = run_dir / "inference/private-responses"
    expected_attempts = [a for row in rows for a in row["result"]["generation_attempts"]] + pre["records"]
    response_index, private_files, private_bytes = reconcile_private_files(private_dir, expected_attempts)
    require(response_index)
    output = {"schema_version": "scifact-train-r2-posthoc-audit-v1", "source_git": SOURCE,
              "audit_script_sha256": sha(Path(__file__).read_bytes()) if __file__ != "<stdin>"
              else globals().get("AUDIT_SCRIPT_SHA256"),
              "input_sha256": FROZEN, "inference_tar_sha256": INFERENCE,
              "scoring_tar_sha256": SCORING, "claims": 12, "slots": 48,
              "components": 12, "corpus_documents": 5183,
              "protocol_sha256": run["protocol_sha256"],
              "model_sha256": run["model_sha256"], "reranker_sha256": run["reranker_sha256"],
              "independent_test": False, "bootstrap_replicates": 0,
              "official_dev_read": False, "routes": {}}
    corpus_hash = {k: sha(json.dumps([d["title"], d["abstract"]], ensure_ascii=False,
                                    separators=(",", ":")).encode())
                   for k, d in corpus.items()}
    all_attempts = []
    for route in ROUTES:
        rr = [r for r in rows if r["route"] == route]
        predictions = {r["claim_id"]: r["prediction"] for r in rr}
        metrics = rescore(gold, predictions)
        frozen = score["routes"][route]
        for m in METRICS:
            for k, value in metrics[m].items():
                require(math.isclose(value, frozen["official_point_score"]["metrics"][m][k], abs_tol=1e-14))
        tokens = Counter()
        counts = Counter()
        errors, wire, invalid_wire, executed, outcomes, statuses = (Counter() for _ in range(6))
        first_fields, action_positions = Counter(), Counter()
        repair_transitions = Counter()
        strata = {s: Counter() for s in STRATA}
        for row in rr:
            result = row["result"]
            require(result["route"] == route and result["claim_id"] == row["claim_id"])
            require(result["budget"] == protocol["budget"] and not result["trace_unlinked_events"])
            require(not row.get("export_error"))
            verify_cost(result)
            attempts = result["generation_attempts"]
            all_attempts.extend(attempts)
            counts["model_calls"] += len(attempts)
            counts["unknown_usage_attempts"] += result["unknown_usage_attempts"]
            counts["rerank_pairs"] += result["rerank_pairs"]
            counts["validation_repairs"] += result["validation_repairs"]
            counts["model_tool_links"] += len(result["model_tool_links"])
            tokens.update(result["usage"])
            outcome = result["outcome"]
            require(outcome in OUTCOMES)
            outcomes[outcome] += 1
            g, selected = gold[row["claim_id"]], sel[row["claim_id"]]
            st = strata[selected["stratum"]]
            st["slots"] += 1
            st[outcome] += 1
            require(len(result["visible_attempts"]) == len(attempts))
            first_visible = None
            for i, a in enumerate(attempts):
                provenance = result["visible_attempts"][i]
                require(provenance["attempt_index"] == i)
                visible, mapped = {}, {}
                for v in provenance["visible"]:
                    doc = corpus[v["doc_id"]]
                    text = doc["abstract"][v["sentence_index"]]
                    require(sha(text.encode()) == v["text_sha256"])
                    require(sha(json.dumps([doc["title"], doc["abstract"]], ensure_ascii=False,
                                           separators=(",", ":")).encode()) == v["source_text_sha256"])
                    visible.setdefault(str(v["doc_id"]), []).append(v["sentence_index"])
                    mapped[f"{v['alias']}:{v['sentence_index']}"] = v["text_sha256"]
                require(mapped == a["visible_sentence_sha256"])
                if i == 0:
                    first_visible = visible
                    st["first_visible_gold_doc_opportunities"] += opportunities(g, visible)
                    st["first_prompt_tokens"] += a["input_prompt_tokens"]
                    st["first_preview_payload_tokens"] += a["preview_payload_tokens"]
                require(a["status"] in {"valid_decision", "validation_failed", "provider_response_invalid"})
                statuses[a["status"]] += 1
                if "error_code" in a:
                    error = a["error_code"]
                    errors[error if error in ERRORS else "other_redacted"] += 1
                if i and attempts[i - 1]["status"] in {"validation_failed", "provider_response_invalid"}:
                    previous = attempts[i - 1]
                    repair_transitions["subsequent_attempts"] += 1
                    repair_transitions["next_valid_decision"] += a["status"] == "valid_decision"
                    if a["status"] == "valid_decision":
                        require(a.get("action") in ACTIONS)
                        repair_transitions["next_valid_" + a["action"]] += 1
                    repair_transitions["same_error_code"] += bool(a.get("error_code")) and a.get("error_code") == previous.get("error_code")
                    repair_transitions["identical_full_output_hash"] += a.get("diagnostics", {}).get("output_sha256") == previous.get("diagnostics", {}).get("output_sha256")
                action = wire_action(a, response_index)
                wire[action] += 1
                first_field, action_position = wire_order(a, response_index)
                first_fields[first_field] += 1
                action_positions[action_position] += 1
                if a["status"] != "valid_decision":
                    invalid_wire[action] += 1
                counts["parsed_tool_proposals"] += a.get("action") in TOOLS
                counts["successful_parsed_tool_decisions"] += a.get("action") in TOOLS and a["status"] == "valid_decision"
                counts["tool_offered_attempts"] += bool(TOOLS & set(a["allowed_actions"]))
            require(first_visible is not None)
            # No model tool ran in this frozen result, so all completed fixed
            # events precede attempt zero. Do not apply this to a changed policy.
            require(not any(e.get("model_selected") for e in result["events"]))
            last_tool = [e for e in result["events"] if e.get("status") == "completed"][-1]
            alias_docs = unique_used_hashes(corpus_hash, last_tool["source_sha256"])
            final_candidates = {alias_docs[a] for a in last_tool["candidate_ids"]}
            requested_docs = {alias_docs[a] for a in attempts[0]["requested_context"]}
            for doc, rats in g["evidence"].items():
                if doc not in final_candidates:
                    st["gold_docs_outside_candidates"] += 1
                elif doc not in requested_docs:
                    st["gold_docs_candidate_not_requested"] += 1
                else:
                    complete = all(i in first_visible.get(doc, []) for i in range(len(corpus[int(doc)]["abstract"])))
                    st["gold_docs_requested_all_sentences_visible"] += complete
                    st["gold_docs_requested_incomplete_sentence_visibility"] += not complete
                    if not any(len(r["sentences"]) <= 3 and set(r["sentences"]) <= set(first_visible.get(doc, [])) for r in rats):
                        st["gold_docs_requested_without_first3_rationale_opportunity"] += 1
            if opportunities(g, first_visible):
                st["visible_opportunity_slots"] += 1
                st["visible_opportunity_" + outcome] += 1
            if route == "adaptive":
                require(first_visible == selected["initial"]["visible"])
                first = result["events"][0]
                candidates = selected["initial"]["candidate_doc_ids"]
                require(first["candidate_ids"] == [f"c{i}" for i in range(len(candidates))])
                for i, doc_id in enumerate(candidates):
                    doc = corpus[doc_id]
                    require(first["source_sha256"][f"c{i}"] == sha(json.dumps(
                        [doc["title"], doc["abstract"]], ensure_ascii=False, separators=(",", ":")).encode()))
                require(attempts[0]["input_prompt_tokens"] == selected["initial"]["prompt_tokens"])
                if selected["stratum"] == "top20_doc_replenishable":
                    witness = selected["witness"]
                    witness_ids = [f"c{candidates.index(d)}" for d in selected["witness_read_doc_ids"]]
                    require("read" in attempts[0]["allowed_actions"] and len(witness_ids) == 1)
                    require(witness_ids != attempts[0]["requested_context"])
                    require(witness["candidate_doc_ids"] == candidates and witness["probe_calls"] == 2)
                    require(witness["prompt_tokens"] <= protocol["budget"]["max_input_tokens"])
                    require(opportunities(g, witness["visible"]) > 0)
                    st["preparation_legal_read_witness_matches_initial_state"] += 1
            for event in result["events"]:
                if "tool" in event:
                    kind = event["tool"]
                    require(kind in TOOLS | {"retrieve", "deterministic_extra_query"})
                    require(event["status"] in {"completed", "failed", "skipped"})
                    executed[("model_" if event.get("model_selected") else "fixed_") + kind + "_" + event["status"]] += 1
        for k in ("input_tokens", "output_tokens"):
            require(tokens[k] == frozen["known_tokens_including_failures"][k])
        stratum_metrics = {}
        for stratum in STRATA:
            sg = {k: g for k, g in gold.items() if sel[k]["stratum"] == stratum}
            stratum_metrics[stratum] = rescore(sg, {k: predictions[k] for k in sg})
            for m in METRICS:
                for k, value in stratum_metrics[stratum][m].items():
                    require(math.isclose(value, frozen["by_stratum"][stratum]["metrics"][m][k], abs_tol=1e-14))
        nei_false = sum(bool(predictions[k]["evidence"]) for k, g in gold.items() if not g["evidence"])
        require(nei_false == frozen["official_point_score"]["separate_diagnostics_not_official_f1"]["nei_claims_with_false_evidence"])
        for q, key in ((.5, "whole_question_p50_ms"), (.95, "whole_question_p95_ms")):
            require(math.isclose(quantile([r["result"]["elapsed_ms"] for r in rr], q), frozen[key], abs_tol=1e-8))
        output["routes"][route] = {
            "metrics": metrics, "outcomes": dict(outcomes), "counts": dict(counts),
            "known_tokens_including_failures": dict(tokens), "attempt_status": dict(statuses),
            "error_counts": dict(errors), "wire_top_level_action": dict(wire),
            "wire_first_field": dict(first_fields), "wire_action_position_1based": dict(action_positions),
            "repair_feedback_transitions": dict(repair_transitions),
            "nei_claims": 3, "nei_false_evidence_claims": nei_false,
            "metrics_by_stratum": stratum_metrics,
            "unsuccessful_attempt_wire_action": dict(invalid_wire), "actual_tool_events": dict(executed),
            "by_stratum": {s: dict(v) for s, v in strata.items()},
            "whole_question_p50_ms": frozen["whole_question_p50_ms"],
            "whole_question_p95_ms": frozen["whole_question_p95_ms"],
        }
    pre_attempts = pre["records"]
    require(pre["attempted_calls"] == len(pre_attempts) == 4)
    output["diagnostics_receipts"] = {"train": receipt_totals(all_attempts),
                                       "synthetic_preflight": receipt_totals(pre_attempts)}
    # Synthetic preflight exercises more than the train action envelope; verify
    # its bytes, not an incorrectly imposed train-only top-level action schema.
    for attempt in pre_attempts:
        receipt = attempt["diagnostics"]["private_attachment"]
        require(not receipt["truncated"] and not receipt["io_failed"])
        require(len(response_index.get(receipt["sha256"], b"")) == receipt["attempted_bytes"])
    output["private_storage_audit"] = {"physical_files": private_files, "physical_bytes": private_bytes,
                                       "receipt_file_hash_multiplicity_match": True,
                                       "preflight_response_recovery_verified": 4}
    output["separate_preflight"] = {"model_calls": 4, "elapsed_ms": pre["elapsed_ms"],
                                    "input_tokens": sum(a["usage"]["input_tokens"] for a in pre_attempts),
                                    "output_tokens": sum(a["usage"]["output_tokens"] for a in pre_attempts)}
    output["separate_load_ms"] = {k: run[k] for k in ("model_load_ms", "reranker_load_ms", "bm25_build_ms")}
    output["audit"] = {"complete_matrix": True, "independent_score_counts_match": True,
                       "all_actual_visible_sentence_hashes_match": True, "cost_ledger_match": True,
                       "adaptive_first_observation_matches_frozen_sampling": True,
                       "raw_preview_id_trace_available": False,
                       "useful_post_tool_feedback_behavior_measurable": False,
                       "new_inference_or_prompt_replay": False}
    require(all(r["counts"]["model_tool_links"] == 0 for r in output["routes"].values()))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.root), sort_keys=True, indent=2, allow_nan=False))
    except Exception as exc:
        locations = []
        trace = exc.__traceback__
        while trace is not None:
            locations.append(str(trace.tb_lineno))
            trace = trace.tb_next
        raise SystemExit("AUDIT_FAILED at code lines " + ",".join(locations)
                         + "; private exception values suppressed") from None
