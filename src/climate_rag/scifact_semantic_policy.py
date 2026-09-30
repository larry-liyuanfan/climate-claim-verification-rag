"""One frozen semantic instruction candidate; no learned critic or extra calls.

Original G prompt bytes/defaults and its schema/mapper remain unchanged. This
module selects an explicit prompt version for both counting and generation.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .evidence_gap_candidate import render_gap_prompt, system_prompt_gap
from .local_bounded_scifact_provider import LocalQwenBoundedSciFactProvider

OLD_G = "scifact-gap-original-v1"
SEMANTIC_G = "scifact-gap-semantic-policy-v1"
POLICIES = (OLD_G, SEMANTIC_G)
PAIR = "scifact-semantic-policy-paired-v1"
SEMANTIC_POLICY = (
    " Scientific evidence decision policy: sharing a topic, entity, or background "
    "with the claim is not SUPPORTS. For each document, check the actual assertion "
    "against the claim's subject or population, relation and direction, quantities "
    "and conditions. SUPPORTS requires the cited sentences to establish that "
    "specific assertion, not merely mention related facts. REFUTES requires "
    "affirmative evidence incompatible with the claim; absence of mention, lack "
    "of proof, or a different study alone is not a contradiction. Do not emit "
    "every retrieved document by default: omit documents without a complete "
    "supporting or contradicting rationale. support=sufficient can justify either "
    "label; it does not mean only positive support. Evaluate support for the concrete "
    "label using only current citable sentences. retrieval_need describes an "
    "evidence gap, not a mandatory action. Consider read or rewrite only when "
    "available candidates or a specific search could address that gap. You may "
    "answer or abstain immediately, with no minimum number of tools. Never "
    "treat a preview, your self-report, or this policy as evidence."
)


def policy_prompt(policy: str) -> str:
    if policy not in POLICIES:
        raise ValueError("unknown_semantic_policy_version")
    return system_prompt_gap() + ("" if policy == OLD_G else SEMANTIC_POLICY)


def policy_sha(policy: str) -> str:
    return hashlib.sha256(policy_prompt(policy).encode()).hexdigest()


def render_policy_prompt(tokenizer: Any, observation: Mapping[str, Any],
                         schema: Mapping[str, Any], policy: str) -> str:
    if policy == OLD_G:
        return render_gap_prompt(tokenizer, observation, schema)
    return str(tokenizer.apply_chat_template([
        {"role": "system", "content": policy_prompt(policy) + "\n" + json.dumps(
            schema, ensure_ascii=False, separators=(",", ":"))},
        {"role": "user", "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":"))},
    ], tokenize=False, add_generation_prompt=True, enable_thinking=False))


class LocalQwenSemanticGapProvider(LocalQwenBoundedSciFactProvider):
    """Same model/grammar/greedy decoding and full-wire cost as bounded G."""

    def __init__(self, model_dir: Path, manifest: dict[str, str], *, private_dir: Path,
                 policy: str, device: str = "cuda") -> None:
        policy_prompt(policy)  # reject before any model load
        self.policy = policy
        super().__init__(model_dir, manifest, private_dir=private_dir, device=device, gap=True)
        self.name += ":" + policy

    def render(self, observation: Mapping[str, Any], schema: Mapping[str, Any]) -> str:
        # Inherited count_prompt and generate both call this exact method.
        return render_policy_prompt(self.base.tokenizer, observation, schema, self.policy)
