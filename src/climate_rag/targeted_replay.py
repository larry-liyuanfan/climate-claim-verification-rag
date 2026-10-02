"""Old consumed CLIMATE32 replay: new search policy, never independent test.

This module contains no gold loader and no model loader. Production identities
are checked before constructing a provider. Synthetic tests inject tiny inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .agent_v3 import SentenceAgentV3, V3Budget
from .scifact_read_continuation import ordered_write
from .scifact_utility_runtime import JournalProvider, ledger_cost
from .targeted_query import PROTOCOL, ROUTES
from .verification import normalise_claim
from . import stop_acquire

ROOT = Path(
    "/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2"
)
INPUT_NAME = "budget-agent-inputs-20260929-563738f0be1f7bf7b99b8de20bcdeab552f7e893166dec260b8f7f1e7951c3c1.tar"
INPUT_SHA = "563738f0be1f7bf7b99b8de20bcdeab552f7e893166dec260b8f7f1e7951c3c1"
OLD_PROTOCOL_SHA = "abfcb61e9e6a54641f025101a4011fa88e07e3697a36f0acd308fa8e86052674"
CORPUS_SHA = "c14315aee9feecbbbbc3b0c7101d978b52b47cee9a613336304bcaa736460c71"
GOLD_SHA = "d2dd28422bffacf87ded2153b3bfac4ca9e1edc903e1d4e7a88d3a40f5d2fd8b"
SELECTION_SHA = "988b6682034a70966c8bbad5ff3c42202933fe85791b97a5f39e4a1b68071bc6"
ORDER_SHA = "6b197778061b1e3cbe9e250a962bc34883c0d5224ce980ccb2313fd134195f8c"
MODEL_SHA = "d1dd9783afdf4e0fbd21eee824834d71b86982f5a5d5f6f371fe07f2f76f3cf6"
RERANKER_SHA = "de1d4ac39101816774439e68881e2308c5e5f1bd94d0b0dc4c492a56c2681052"
MANIFEST_BYTES = {
    "generator": "8bff1d6532f4c1e29eb87c3d2230da9b718a353b6d8039cdec93ca9a566bf917",
    "reranker": "e1f56457935dd69b67e3249cbd81fd7aabaf5e2f0b870aa173b6fcb573b564ed",
}
STUDY = "public_v2_repeatedly_used_validation_replay_not_independent_test"
BUDGET = V3Budget().model_dump()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def policy(protocol: str = PROTOCOL) -> dict[str, Any]:
    route_protocol(protocol, "adaptive")
    result = {
        "protocol": protocol,
        "routes": list(ROUTES),
        "study_kind": STUDY,
        "tasks": 32,
        "decisive": 24,
        "nondecisive": 8,
        "planned_slots": 160,
        "ordered_ids_sha256": ORDER_SHA,
        "old_protocol_sha256": OLD_PROTOCOL_SHA,
        "selection_sha256": SELECTION_SHA,
        "gold_sha256": GOLD_SHA,
        "input_archive_sha256": INPUT_SHA,
        "corpus_sha256": CORPUS_SHA,
        "documents": 5240,
        "model_sha256": MODEL_SHA,
        "reranker_sha256": RERANKER_SHA,
        "model_manifest_file_sha256": MANIFEST_BYTES,
        "budget": BUDGET,
        "max_generator_calls": 800,
        "max_rerank_requests": 128,
        "max_queries": 2,
        "warmup_calls": 0,
        "dense_enabled": False,
        "rerank_width": 20,
        "bm25": "existing BM25Index defaults; same 5240 documents in file order",
        "fusion": "RRF k=60 over raw per-query Top20; stable source-ID ties; truncate20",
        "packing": "retain all previously displayed full sentences; fail if they no longer fit",
        "query_control": "same query-item schema; fixed upfront list vs adaptive feedback actions",
        "same_caps_not_equal_actual_calls": True,
        "bootstrap_samples": 5000,
        "seed": 20260929,
        "primary_metrics": [
            "recall@5",
            "mrr@10",
            "ndcg@10",
            "evidence_f1",
            "official_decisive_label_accuracy",
        ],
        "comparators": ["fixed_rerank", "fixed_multiquery"],
        "quality_gate": "adaptive-minus-each-comparator bootstrap lower CI >0 for Evidence F1 and decisive accuracy; no negative mean Recall/MRR/nDCG",
        "cost_gate": "complete generator and reranker physical accounting, no more failures/pairs/reranker tokens, no higher mean total generator tokens or mean elapsed_ms against either comparator",
        "interpretation": "quality without cost gate is extra-compute tradeoff, not efficiency win; no independent/generalisation/production claim",
        "citation_metrics": "official gold-ID match; provenance is checked, semantic entailment unmeasured",
        "failure_denominator": "all32 per route; crashes/unknown usage retained; missing matrix unscored, never dropped",
        "split_access": "old consumed CLIMATE validation only; no SciFact/dev300/frozen test",
        "training": False,
        "automatic_retry": False,
    }
    if protocol == stop_acquire.PROTOCOL:
        result.update(
            controller_by_route={r: route_protocol(protocol, r) for r in ROUTES},
            acquisition="stop or acquire unseen preview/targeted query, then separate verdict; stop always legal",
            final_capacity="acquire only with >=3 remaining combined generations; repairs consume same reserve",
            query_control="four fixed controls unchanged; only adaptive gate/verdict changes",
            comparison_kind="whole_policy_not_causal_feedback_isolation",
            retrieval_denominator=24, binary_label_denominator=23, cost_denominator=32,
            legacy_adaptive_protocol=PROTOCOL,
        )
    return result


def route_protocol(protocol: str, route: str) -> str:
    if protocol not in {PROTOCOL, stop_acquire.PROTOCOL} or route not in ROUTES:
        raise ValueError("unknown_matrix_protocol_or_route")
    return protocol if route == "adaptive" else PROTOCOL


def frozen_tasks(raw: bytes, selection_raw: bytes) -> list[dict[str, str]]:
    if sha(raw) != OLD_PROTOCOL_SHA or sha(selection_raw) != SELECTION_SHA:
        raise ValueError("frozen_input_metadata_hash")
    protocol, selection = json.loads(raw), json.loads(selection_raw)
    tasks = protocol["validation"]
    ids = [t["id"] for t in tasks]
    if (
        len(tasks) != 32
        or len(set(ids)) != 32
        or ids != selection["selected_ids"]
        or sha(json.dumps(ids, separators=(",", ":")).encode()) != ORDER_SHA
        or protocol["corpus_sha256"] != CORPUS_SHA
    ):
        raise ValueError("old32_order_corpus_identity")
    for task in tasks:
        if (
            set(task) != {"id", "claim_text"}
            or not isinstance(task["id"], str)
            or not isinstance(task["claim_text"], str)
            or not 1 <= len(normalise_claim(task["claim_text"])) <= 2000
        ):
            raise ValueError("invalid_frozen_claim")
    return list(tasks)


class TargetedJournal(JournalProvider):
    capacity_ceiling = 800

    def render(self, observation: Any, schema: Any) -> str:
        return str(self.backend.render(observation, schema))


def run_matrix(
    tasks: list[dict[str, str]],
    backend: Any,
    retrieve: Any,
    rerank: Any,
    output: Path,
    *,
    physical_guard: Any = None,
    protocol: str = PROTOCOL,
) -> dict[str, Any]:
    route_protocol(protocol, "adaptive")
    if not tasks or len(tasks) > 32 or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("invalid_matrix_tasks")
    output.mkdir(mode=0o700)
    slots = [{"task_id": t["id"], "route": route} for t in tasks for route in ROUTES]
    ordered_write(
        output / "planned.json",
        {
            "protocol": protocol,
            "slots": slots,
            "max_generations": len(slots) * 5,
            "warmups": 0,
            "retry": False,
        },
    )
    journal = TargetedJournal(
        backend,
        output / "ledger",
        max_generations=len(slots) * 5,
        protocol=protocol,
        physical_guard=physical_guard,
    )
    agents = {r: SentenceAgentV3(journal, retrieve, rerank=rerank,
                               protocol=route_protocol(protocol, r)) for r in ROUTES}
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for route in ROUTES:
            slot = f"slot-{len(rows):03d}"
            journal.slot = slot
            directory = output / slot
            directory.mkdir(mode=0o700)
            ordered_write(
                directory / "reserved.json",
                {
                    "task_id": task["id"],
                    "route": route,
                    "started_unix": time.time(),
                    "deadline_seconds": BUDGET["timeout_seconds"],
                },
            )
            backend.start_slot(directory / "private")
            if hasattr(rerank, "start_slot"):
                rerank.start_slot(slot)
            row = {
                "task_id": task["id"],
                "slot": slot,
                **agents[route].run(task["claim_text"], route),
            }
            ids = [
                p.name.removesuffix(".reserved.json")
                for p in journal.directory.glob("g*.reserved.json")
                if json.loads(p.read_bytes())["slot"] == slot
            ]
            row["physical_cost"] = ledger_cost(journal.directory, ids)
            row["reranker_cost"] = (
                rerank.cost_for_slot(slot) if hasattr(rerank, "cost_for_slot") else None
            )
            ordered_write(directory / "finished.json", row)
            rows.append(row)
    run = {"protocol": protocol, "study_kind": STUDY, "budget": BUDGET, "runs": rows}
    ordered_write(output / "run.json", run)
    return run


def slot_watchdog(output: Path) -> None:
    for p in output.glob("slot-*/reserved.json"):
        if (p.parent / "finished.json").exists():
            continue
        try:
            stamp = json.loads(p.read_bytes())["started_unix"]
        except json.JSONDecodeError:
            stamp = p.stat().st_mtime
        if time.time() - stamp >= BUDGET["timeout_seconds"]:
            raise TimeoutError("targeted_slot_deadline")
