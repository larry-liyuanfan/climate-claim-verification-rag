"""Post-exit automatic NLI proxy, never a human or official correctness label.

No gold, dataset loader or acquisition policy is imported here. Incomplete
matrices fail closed. Long pairs are unscored, never silently truncated.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any

import numpy as np

from .fair_acquisition import COVERAGE_PROTOCOL, PROTOCOL, ROUTES
from .metrics import paired_bootstrap

MODEL = "cross-encoder/nli-deberta-v3-small"
REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"
LABELS = ("contradiction", "entailment", "neutral")
THRESHOLD = 0.80  # frozen before any evaluation; no calibration claim
CONTRACT = "saved-fair-citation-nli-proxy-v1"


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def matrix(run: Mapping[str, Any], *, expected_tasks: int | None = None,
           expected_protocol: str = PROTOCOL) -> list[dict[str, Any]]:
    if expected_protocol not in {PROTOCOL, COVERAGE_PROTOCOL} or run.get("protocol") != expected_protocol:
        raise ValueError("unsupported_saved_protocol")
    rows = run["runs"]
    ids = [r["task_id"] for r in rows if r["route"] == ROUTES[0]]
    if not ids or len(set(ids)) != len(ids) or len(rows) != 3 * len(ids):
        raise ValueError("incomplete_matrix")
    if expected_tasks is not None and len(ids) != expected_tasks:
        raise ValueError("changed_frozen_denominator")
    if any([r["task_id"] for r in rows if r["route"] == arm] != ids for arm in ROUTES):
        raise ValueError("route_order_mismatch")
    claims: dict[str, str] = {}
    for row in rows:
        fingerprint = row["immutable_claim_sha256"]
        if claims.setdefault(row["task_id"], fingerprint) != fingerprint:
            raise ValueError("cross_route_claim_mismatch")
    return list(rows)


def cited_pair(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Verify byte identity and actual final-prompt delivery, without gold."""
    answer = row["answer"]
    if answer is None:
        return None
    claim = row["initial_frame"]["immutable_claim"]
    if text_sha(claim) != row["immutable_claim_sha256"]:
        raise ValueError("immutable_claim_mismatch")
    last = row["generation_attempts"][-1]
    if last["observation"]["immutable_claim"] != claim:
        raise ValueError("final_prompt_claim_mismatch")
    decision = last["decision"]
    if (last["stage"] != "verdict" or last["status"] != "valid_decision"
            or decision["action"] != "answer" or decision["label"] != answer["label"]
            or answer["label"] not in {"SUPPORTS", "REFUTES"}):
        raise ValueError("answer_not_valid_terminal")
    visible = {s["sentence_id"]: s["text"] for s in last["observation"]["current_citable"]}
    chosen = decision["sentence_ids"]
    citations = answer["citations"]
    if not 1 <= len(chosen) <= 3 or len(set(chosen)) != len(chosen) or len(citations) != len(chosen):
        raise ValueError("invalid_citation_set")
    sentences = []
    for sid, citation in zip(chosen, citations, strict=True):
        alias, ordinal = sid.split(":")
        text = citation["text"]
        if (visible.get(sid) != text or text_sha(text) != citation["text_sha256"]
                or last["visible_sentence_sha256"].get(sid) != text_sha(text)
                or row["visible_source_ids"][alias] != citation["source_id"]
                or int(ordinal) != citation["sentence_index"]):
            raise ValueError("citation_not_actual_delivered_bytes")
        sentences.append(text)
    return {"premise": "\n".join(sentences), "hypothesis": claim,
            "sentences": sentences, "target": "entailment" if answer["label"] == "SUPPORTS" else "contradiction"}


def classify(probabilities: Sequence[float], target: str) -> dict[str, Any]:
    values = [float(p) for p in probabilities]
    if (len(values) != 3 or any(not math.isfinite(p) or not 0 <= p <= 1 for p in values)
            or abs(sum(values) - 1) > 1e-5 or target not in LABELS[:2]):
        raise ValueError("invalid_nli_probabilities")
    label = LABELS[max(range(3), key=lambda i: values[i])]
    return {"probabilities": dict(zip(LABELS, values, strict=True)),
            "target": target, "predicted_relation": label,
            "passes_proxy": values[LABELS.index(target)] >= THRESHOLD}


def aggregate(records: Sequence[Mapping[str, Any]], *, tasks: int) -> dict[str, Any]:
    if tasks < 1 or len(records) != tasks * 3:
        raise ValueError("incomplete_semantic_proxy_matrix")
    result: dict[str, Any] = {"contract": CONTRACT, "model": MODEL, "revision": REVISION,
        "threshold": THRESHOLD, "all_tasks_per_arm": tasks, "routes": {},
        "scope": "posthoc automatic NLI proxy on previously consumed development outputs",
        "human_annotations": False, "official_scores_changed": False,
        "gold_read": False, "causal_feedback_gain": False,
        "abstention_semantic_correctness": "unmeasured; no answer != correct NEI",
        "limitation": "general SNLI/MNLI model; uncalibrated climate entailment, numbers, time and multi-evidence errors possible; not a truth oracle"}
    vectors = {}
    roster = None
    for arm in ROUTES:
        selected = [r for r in records if r["route"] == arm]
        ids = [r["slot_position"] for r in selected]
        if len(selected) != tasks or ids != list(range(tasks)) or (roster is not None and ids != roster):
            raise ValueError("semantic_proxy_roster_mismatch")
        roster = ids
        status = Counter(r["status"] for r in selected)
        if not set(status) <= {"scored", "abstained", "execution_failure", "overlength", "model_error"}:
            raise ValueError("unknown_proxy_status")
        passed = sum(bool(r.get("joint_citations", {}).get("passes_proxy", False)) for r in selected)
        per_sentence = [s for r in selected for s in r.get("individual_citations", [])]
        result["routes"][arm] = {"denominator_all_tasks": tasks, "status": dict(status),
            "joint_citation_proxy_pass_count": passed, "joint_citation_proxy_pass_all_tasks": passed / tasks,
            "joint_citation_proxy_pass_scored_answers": passed / status["scored"] if status["scored"] else None,
            "individual_citations_scored": len(per_sentence),
            "individual_citation_proxy_pass_count": sum(bool(s["passes_proxy"]) for s in per_sentence)}
        # Errors and abstentions remain 0 in all-task proxy; not relabelled NEI.
        vectors[arm] = [float(bool(r.get("joint_citations", {}).get("passes_proxy", False))) for r in selected]
    result["paired_bootstrap"] = {"autonomous-minus-" + arm: paired_bootstrap(
        vectors[arm], vectors["autonomous"], samples=5000, seed=20261003)
        for arm in ROUTES[:2]}
    return result


class LocalNLI:
    """Pinned local CPU execution; no external API, remote code, or truncation."""

    def __init__(self, model_path: str, *, threads: int = 4) -> None:
        import torch
        from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
        from transformers.models.auto.tokenization_auto import AutoTokenizer
        if not 1 <= threads <= 8:
            raise ValueError("bounded_cpu_threads")
        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()  # type: ignore[no-untyped-call]
        labels = tuple(str(self.model.config.id2label[i]).casefold() for i in range(3))
        if labels != LABELS:
            raise ValueError("model_label_mapping_mismatch")
        self.max_length = min(int(self.tokenizer.model_max_length), 512)

    def score(self, premise: str, hypothesis: str) -> tuple[list[float], int]:
        features = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=False)
        tokens = int(features["input_ids"].shape[-1])
        if tokens > self.max_length:
            raise OverflowError("pair_exceeds_model_context_no_truncation")
        with self.torch.inference_mode():
            scores = self.model(**features).logits.softmax(-1)[0].detach().cpu().numpy()
        return np.asarray(scores, dtype=float).tolist(), tokens
