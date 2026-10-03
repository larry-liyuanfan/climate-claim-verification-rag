"""CPU-only paired component bootstrap over a complete original-SciFact matrix.

No data loader or inference runner. Counts come from the frozen scorer; failures
are retained. Never average per-claim or per-group F1, and never call sign mass p.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .scifact_grounding import Abstract, GoldClaim, parse_gold
from .scifact_scoring import parse_prediction, score_original

ROUTES = ("fixed_retrieval", "fixed_rerank", "deterministic_extra", "adaptive")
METRICS = (
    "abstract_label_only",
    "abstract_rationalized",
    "sentence_selection",
    "sentence_label",
)
REPLICATES, SEED = 5000, 20260930


@dataclass(frozen=True)
class RunRecord:
    route: str
    prediction: Mapping[str, Any]
    termination_reason: str
    whole_question_elapsed_ms: float
    input_tokens_lower_bound: int
    output_tokens_lower_bound: int
    unknown_usage_attempts: int
    cost_scope: str = "whole_question_including_repairs_and_failures"


def nonnegative_number(value: Any) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError("cost must be finite and nonnegative, not bool/text")
    return float(value)


def nonnegative_int(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("cost count must be a nonnegative integer")
    return value


def component_weights(groups: int) -> NDArray[np.int64]:
    if groups < 1:
        raise ValueError("no nonempty target groups")
    draws = np.random.Generator(np.random.PCG64(SEED)).integers(
        0, groups, size=(REPLICATES, groups), dtype=np.int64
    )
    weights = np.zeros((REPLICATES, groups), dtype=np.int64)
    for i, row in enumerate(draws):
        weights[i] = np.bincount(row, minlength=groups)
    return weights


def micro_f1(counts: NDArray[Any]) -> NDArray[np.float64]:
    denominator = counts[..., 1] + counts[..., 2]
    return np.divide(
        2.0 * counts[..., 0],
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0,
    )


def interval(samples: NDArray[Any], groups: int) -> dict[str, Any]:
    if groups < 2:
        return {"ci95": None, "reason": "only_one_nonempty_target_group"}
    return {
        "ci95": np.quantile(samples, [0.025, 0.975], method="linear").tolist(),
        "method": "paired_component_percentile",
        "p_value": None,
    }


def evaluate_matrix(
    gold: Sequence[GoldClaim],
    records: Sequence[RunRecord],
    corpus: Mapping[int, Abstract],
    claim_components: Mapping[int, str],
    *,
    separate_phase_costs: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """All four routes × exact gold IDs required; only nonempty target groups drawn.

    claim_components may include training-only entries from the frozen original
    grouping; those entries are deliberately excluded from the bootstrap universe.
    Input token values are charged known-use lower bounds, not invented zeros.
    """
    if not gold or len({g.claim_id for g in gold}) != len(gold):
        raise ValueError("empty/duplicate gold matrix")
    ordered_gold = sorted(gold, key=lambda g: g.claim_id)
    for g in ordered_gold:
        parse_gold(g.gold_row(), corpus)
    ids = [g.claim_id for g in ordered_gold]
    if any(
        i not in claim_components
        or not isinstance(claim_components[i], str)
        or not claim_components[i].strip()
        for i in ids
    ):
        raise ValueError("missing/invalid target component identity")
    group_names = sorted({claim_components[i] for i in ids})
    group_index = {name: i for i, name in enumerate(group_names)}
    membership = np.array(
        [group_index[claim_components[i]] for i in ids], dtype=np.int64
    )
    group_sizes = np.bincount(membership, minlength=len(group_names))
    slots: dict[tuple[str, int], RunRecord] = {}
    parsed = {}
    for record in records:
        prediction = parse_prediction(record.prediction, corpus)
        key = record.route, prediction.claim_id
        if record.route not in ROUTES or prediction.claim_id not in ids or key in slots:
            raise ValueError("extra/duplicate/unknown matrix slot")
        if record.cost_scope != "whole_question_including_repairs_and_failures":
            raise ValueError("wrong cost scope")
        nonnegative_number(record.whole_question_elapsed_ms)
        for n in (
            record.input_tokens_lower_bound,
            record.output_tokens_lower_bound,
            record.unknown_usage_attempts,
        ):
            nonnegative_int(n)
        reason = record.termination_reason
        if reason == "ids_validated_semantics_unmeasured":
            if not prediction.evidence or any(
                not d.sentences for d in prediction.evidence.values()
            ):
                raise ValueError("accepted answer requires nonempty document evidence")
        elif isinstance(reason, str) and (
            reason
            in {
                "model_abstention:insufficient_evidence",
                "model_abstention:conflicting_evidence",
                "model_abstention:budget",
                "generation_budget_exhausted",
                "deadline",
                "deadline_during_prompt_assembly",
                "validation_repair_exhausted",
            }
            or (
                reason.startswith("controller_failure:")
                and len(reason) > len("controller_failure:")
            )
        ):
            if prediction.evidence:
                raise ValueError("failure/abstain cannot contain accepted evidence")
        else:
            raise ValueError("unknown termination reason")
        slots[key], parsed[key] = record, prediction
    if set(slots) != {(r, i) for r in ROUTES for i in ids}:
        raise ValueError("incomplete matrix: missing predictions are not synthesized")
    phase_costs = None
    if separate_phase_costs is not None:
        if set(separate_phase_costs) != {"model_load_ms", "runtime_preflight_ms"}:
            raise ValueError("separate phase costs require load and preflight")
        phase_costs = {
            k: nonnegative_number(v) for k, v in separate_phase_costs.items()
        }

    weights = component_weights(len(group_names))
    denominators = weights @ group_sizes
    reports: dict[str, Any] = {}
    boot_quality: dict[str, NDArray[Any]] = {}
    boot_cost: dict[str, NDArray[Any]] = {}
    for route in ROUTES:
        full = score_original(ordered_gold, [parsed[route, i] for i in ids])
        counts = np.zeros((len(group_names), len(METRICS), 3), dtype=np.int64)
        costs = np.zeros((len(group_names), 3), dtype=np.float64)
        unknown = failures = 0
        latencies = []
        for g, group in zip(ordered_gold, membership, strict=True):
            one = score_original([g], [parsed[route, g.claim_id]])
            for m, metric in enumerate(METRICS):
                counts[group, m] += [
                    one["metrics"][metric][name]
                    for name in ("correct", "predicted", "relevant")
                ]
            row = slots[route, g.claim_id]
            costs[group] += [
                row.whole_question_elapsed_ms,
                row.input_tokens_lower_bound,
                row.output_tokens_lower_bound,
            ]
            latencies.append(row.whole_question_elapsed_ms)
            unknown += row.unknown_usage_attempts
            failures += (
                row.termination_reason != "ids_validated_semantics_unmeasured"
                and not row.termination_reason.startswith("model_abstention:")
            )
        total = counts.sum(axis=0)
        for m, metric in enumerate(METRICS):
            if not np.array_equal(
                total[m],
                [
                    full["metrics"][metric][k]
                    for k in ("correct", "predicted", "relevant")
                ],
            ):
                raise ValueError(
                    "per-claim sufficient statistics disagree with full scorer"
                )
        sampled_counts = (weights @ counts.reshape(len(group_names), -1)).reshape(
            REPLICATES, len(METRICS), 3
        )
        boot_quality[route] = micro_f1(sampled_counts)
        boot_cost[route] = (weights @ costs) / denominators[:, None]
        means = costs.sum(axis=0) / len(ids)
        reports[route] = {
            "official_score": full,
            "failures_retained": failures,
            "cost": {
                "mean_whole_question_elapsed_ms": float(means[0]),
                "whole_question_p50_ms": float(
                    np.quantile(latencies, 0.5, method="linear")
                ),
                "whole_question_p95_ms": float(
                    np.quantile(latencies, 0.95, method="linear")
                ),
                "mean_input_tokens_lower_bound": float(means[1]),
                "mean_output_tokens_lower_bound": float(means[2]),
                "unknown_usage_attempts": unknown,
                "token_totals_exact": unknown == 0,
                "latency_quantile_ci": None,
                "latency_quantile_ci_reason": "not_computed; point values use whole-question durations, not per-call or paired differences",
            },
        }
    comparisons = {}
    for comparator in ("fixed_rerank", "deterministic_extra"):
        left, right = reports["adaptive"], reports[comparator]
        comparisons["adaptive-minus-" + comparator] = {
            "quality": {
                metric: {
                    "point_delta_f1": left["official_score"]["metrics"][metric]["f1"]
                    - right["official_score"]["metrics"][metric]["f1"],
                    **interval(
                        boot_quality["adaptive"][:, m] - boot_quality[comparator][:, m],
                        len(group_names),
                    ),
                }
                for m, metric in enumerate(METRICS)
            },
            "mean_whole_question_elapsed_ms": {
                "point_delta": left["cost"]["mean_whole_question_elapsed_ms"]
                - right["cost"]["mean_whole_question_elapsed_ms"],
                **interval(
                    boot_cost["adaptive"][:, 0] - boot_cost[comparator][:, 0],
                    len(group_names),
                ),
            },
            "mean_token_delta": None,
            "token_delta_reason": "unknown usage: lower-bound differences cannot establish exact savings",
        }
        if left["cost"]["token_totals_exact"] and right["cost"]["token_totals_exact"]:
            comparisons["adaptive-minus-" + comparator]["mean_token_delta"] = {
                name: {
                    "point_delta": left["cost"]["mean_" + name + "_tokens_lower_bound"]
                    - right["cost"]["mean_" + name + "_tokens_lower_bound"],
                    **interval(
                        boot_cost["adaptive"][:, col] - boot_cost[comparator][:, col],
                        len(group_names),
                    ),
                }
                for name, col in (("input", 1), ("output", 2))
            }
            comparisons["adaptive-minus-" + comparator]["token_delta_reason"] = (
                "all recorded usage known"
            )
    group_payload = json.dumps(
        [[i, claim_components[i]] for i in ids], separators=(",", ":")
    ).encode()
    return {
        "schema_version": "scifact-paired-component-statistics-v1",
        "primary": "adaptive-minus-fixed_rerank / abstract_rationalized / micro F1",
        "attribution_control": "adaptive-minus-deterministic_extra / abstract_rationalized / micro F1",
        "other_quality_metrics": "prespecified secondary, not selected posthoc",
        "routes": reports,
        "paired_comparisons": comparisons,
        "bootstrap": {
            "replicates": REPLICATES,
            "seed": SEED,
            "rng": "numpy.PCG64",
            "numpy_version": np.__version__,
            "sampling": "G whole nonempty target components with replacement, paired across routes",
            "quantile": "linear",
            "interval": "95% percentile",
            "p_values_computed": False,
            "target_claim_count": len(ids),
            "nonempty_target_groups": len(group_names),
            "group_size_counts": sorted(group_sizes.tolist()),
            "excluded_non_target_mapping_entries": len(
                set(claim_components) - set(ids)
            ),
            "target_membership_sha256": hashlib.sha256(group_payload).hexdigest(),
            "integer_weights_sha256": hashlib.sha256(
                weights.astype("<i8").tobytes(order="C")
            ).hexdigest(),
            "weighted_claim_count_range": [
                int(denominators.min()),
                int(denominators.max()),
            ],
            "duplicate_group_draw_replicates": int(np.any(weights > 1, axis=1).sum()),
            "empty_or_single_group_ci": "no CI when G=1; empty matrix rejected",
        },
        "separate_load_and_preflight_costs": phase_costs,
        "nei_diagnostic_ci": None,
        "nei_diagnostic_ci_reason": "not_computed; absent NEI denominator is null in frozen scorer",
        "claim_accuracy": None,
        "model_evaluation_performed_by_this_function": False,
        "limitations": [
            "Component grouping does not certify independence or foundation-model exposure.",
            "Known token sums include supplied repair/failure charges; completeness is an upstream ledger obligation.",
            "Bootstrap resamples components and recomputes pooled micro counts, not per-claim F1.",
            "No latency-quantile intervals or bootstrap sign-mass p-values are computed.",
        ],
    }
