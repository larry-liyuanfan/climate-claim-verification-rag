"""Candidate-only ordered callback; no frozen config, environment or HF changes."""

from __future__ import annotations

import importlib.metadata
from typing import Any

from .local_scifact_provider import require_clean_grammar_environment


class OrderedGapPrefix:
    """Same callable signature as HF prefix_allowed_tokens_fn, without Torch.

    CPU tests exercise this actual callback with cached tokenizer prefixes. That
    is not a model.generate() or GPU compatibility/result claim.
    """

    def __init__(self, enforcer: Any):
        self.token_enforcer = enforcer

    def __call__(self, batch_id: int, sent: Any) -> list[int]:
        if batch_id != 0:
            raise ValueError("candidate_callback_is_serial_single_sequence")
        result: list[int] = self.token_enforcer.get_allowed_tokens(
            sent.tolist()
        ).allowed_tokens
        return result


def build_ordered_gap_prefix(
    tokenizer_data: Any, schema: dict[str, Any]
) -> OrderedGapPrefix:
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig

    require_clean_grammar_environment()
    if importlib.metadata.version("lm-format-enforcer") != "0.11.3":
        raise ValueError("candidate_requires_lmfe_0_11_3")

    def config() -> Any:
        return CharacterLevelParserConfig(
            alphabet=tokenizer_data.tokenizer_alphabet,
            max_consecutive_whitespaces=12,
            force_json_field_order=True,
            max_json_array_length=20,
        )

    parser = JsonSchemaParser(schema, config=config())
    enforcer = TokenEnforcer(tokenizer_data, parser)
    # LMFE0.11.3 constructor overwrites parser.config. Reapply independently,
    # before ANY prefix/cache state exists, preserving the tokenizer alphabet.
    if (
        enforcer.root_parser is not parser
        or enforcer.prefix_states
        or enforcer.allowed_token_cache
    ):
        raise ValueError("candidate_prefix_must_be_fresh")
    # Mutate only the ordering flag of the effective config: never replace the
    # tokenizer alphabet/config after construction. Other explicit bounds must
    # have survived as the pinned library's clean-environment defaults.
    parser.config.force_json_field_order = True
    if (
        parser.config.alphabet != tokenizer_data.tokenizer_alphabet
        or parser.config.max_consecutive_whitespaces != 12
        or parser.config.max_json_array_length != 20
    ):
        raise ValueError("unexpected_effective_candidate_grammar_config")
    if not enforcer.root_parser.config.force_json_field_order:
        raise ValueError("ordered_candidate_config_not_effective")
    return OrderedGapPrefix(enforcer)
