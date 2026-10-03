"""Versioned generation-time semantic prefix constraints; frozen v1 is unchanged.

JSON syntax still comes from pinned LMFE. This immutable structural tracker
intersects its next characters with uniqueness and cross-document cardinality.
No completed output is rewritten, deduplicated, truncated, or turned into NEI.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, NoReturn

from .local_scifact_provider import require_clean_grammar_environment

PROTOCOL = "scifact-bounded-prefix-v1"
Path = tuple[str | int, ...]


@lru_cache(maxsize=128)
def _without_raw_controls(characters: str) -> str:
    return characters.translate(dict.fromkeys(range(32)))


@dataclass(frozen=True)
class Frame:
    kind: str
    path: Path
    phase: str
    key: str = ""
    index: int = 0


@dataclass(frozen=True)
class Document:
    alias: str | None = None
    sentences: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticState:
    frames: tuple[Frame, ...] = ()
    root_started: bool = False
    string_path: Path | None = None
    string_is_key: bool = False
    string_raw: str = ""
    escape_pending: bool = False
    documents: tuple[Document, ...] = ()
    read_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Contract:
    document_sentences: tuple[tuple[str, tuple[str, ...]], ...]
    read_candidates: tuple[str, ...]
    envelope: bool
    schema_sha256: str

    def local(self, path: Path) -> Path:
        if not self.envelope:
            return path
        return path[1:] if path and path[0] == "decision" else ("outer", *path)

    def choices(self, state: SemanticState, path: Path) -> tuple[str, ...] | None:
        p = self.local(path)
        groups = dict(self.document_sentences)
        if len(p) == 2 and p[0] == "source_ids" and isinstance(p[1], int):
            return tuple(s for s in self.read_candidates if s not in state.read_ids)
        if len(p) < 3 or p[0] != "documents" or not isinstance(p[1], int):
            return None
        index = p[1]
        doc = state.documents[index]
        used = {d.alias for i, d in enumerate(state.documents) if i != index}
        aliases = tuple(a for a in groups if a not in used and doc.alias in (None, a))
        if p[2:] == ("source_id",):
            return aliases
        if len(p) == 4 and p[2] == "sentence_ids" and isinstance(p[3], int):
            if len(doc.sentences) >= 8 or sum(len(d.sentences) for d in state.documents) >= 20:
                return ()
            return tuple(s for a in aliases for s in groups[a] if s not in doc.sentences)
        return None

    def next_array_item_possible(self, state: SemanticState, frame: Frame) -> bool:
        path = self.local(frame.path)
        if path == ("documents",):
            used = {d.alias for d in state.documents}
            return (len(state.documents) < 5
                    and sum(len(d.sentences) for d in state.documents) < 20
                    and any(a not in used for a, _ in self.document_sentences))
        options = self.choices(state, (*frame.path, frame.index))
        return options is None or bool(options)


def _fail() -> NoReturn:
    raise ValueError("bounded_prefix_semantic_rejection")


def _replace_top(state: SemanticState, frame: Frame) -> SemanticState:
    return replace(state, frames=(*state.frames[:-1], frame))


def _value_path(state: SemanticState) -> Path:
    if not state.frames:
        if state.root_started:
            _fail()
        return ()
    frame = state.frames[-1]
    if frame.phase not in {"value", "value_or_end"}:
        _fail()
    return (*frame.path, frame.key if frame.kind == "object" else frame.index)


def _consume_value(state: SemanticState) -> SemanticState:
    if not state.frames:
        return replace(state, root_started=True)
    frame = state.frames[-1]
    return _replace_top(state, replace(frame, phase="separator", index=frame.index + 1))


def _store_string(state: SemanticState, contract: Contract, value: str) -> SemanticState:
    assert state.string_path is not None
    if state.string_is_key:
        frame = state.frames[-1]
        return _replace_top(state, replace(frame, key=value, phase="colon"))
    options = contract.choices(state, state.string_path)
    if options is not None and value not in options:
        _fail()
    path = contract.local(state.string_path)
    if len(path) >= 3 and path[0] == "documents" and isinstance(path[1], int):
        index = path[1]
        doc = state.documents[index]
        if path[2:] == ("source_id",):
            doc = replace(doc, alias=value)
        elif len(path) == 4 and path[2] == "sentence_ids":
            doc = replace(doc, alias=value.rsplit(":", 1)[0], sentences=(*doc.sentences, value))
        state = replace(state, documents=(*state.documents[:index], doc, *state.documents[index + 1:]))
    elif len(path) == 2 and path[0] == "source_ids":
        state = replace(state, read_ids=(*state.read_ids, value))
    return _consume_value(state)


def advance(state: SemanticState, contract: Contract, char: str) -> SemanticState:
    """One character, no mutation or model/gold access. Syntax is intersected by LMFE."""
    if len(char) != 1:
        _fail()
    string_path = state.string_path
    if string_path is not None:
        if ord(char) < 32:
            _fail()  # No completion can repair a literal JSON-string control.
        if char == '"' and not state.escape_pending:
            value = json.loads('"' + state.string_raw + '"')
            state = _store_string(state, contract, value)
            return replace(state, string_path=None, string_is_key=False, string_raw="", escape_pending=False)
        raw = state.string_raw + char
        options = None if state.string_is_key else contract.choices(state, string_path)
        # ID enums in pinned LMFE use literal ASCII continuations, not escaped
        # aliases. Free query/gap values remain JSON-escaped/Unicode capable.
        if options is not None and not any(s.startswith(raw) for s in options):
            _fail()
        escaped = not state.escape_pending and char == "\\"
        return replace(state, string_raw=raw, escape_pending=escaped)
    if char in " \r\n\t":
        return state
    if char == '"':
        is_key = bool(state.frames and state.frames[-1].kind == "object"
                      and state.frames[-1].phase in {"key", "key_or_end"})
        path = state.frames[-1].path if is_key else _value_path(state)
        if not is_key:
            options = contract.choices(state, path)
            if options is not None and not options:
                _fail()
        return replace(state, string_path=path, string_is_key=is_key)
    if char in "{[":
        path = _value_path(state)
        local = contract.local(path)
        if char == "{" and len(local) == 2 and local[0] == "documents":
            if (local[1] != len(state.documents) or not contract.next_array_item_possible(
                    state, Frame("array", path[:-1], "value", index=len(state.documents)))):
                _fail()
            state = replace(state, documents=(*state.documents, Document()))
        state = _consume_value(state)
        frame = Frame("object" if char == "{" else "array", path,
                      "key_or_end" if char == "{" else "value_or_end")
        return replace(state, frames=(*state.frames, frame))
    if not state.frames:
        _fail()
    frame = state.frames[-1]
    if char in "}]":
        if ((char == "}" and frame.kind != "object") or (char == "]" and frame.kind != "array")
                or frame.phase not in {"separator", "key_or_end", "value_or_end"}):
            _fail()
        return replace(state, frames=state.frames[:-1])
    if char == ":" and frame.kind == "object" and frame.phase == "colon":
        return _replace_top(state, replace(frame, phase="value"))
    if char == "," and frame.phase == "separator":
        if frame.kind == "array" and not contract.next_array_item_possible(state, frame):
            _fail()  # Prune the comma, not just a later 21st completed sentence.
        return _replace_top(state, replace(frame, phase="key" if frame.kind == "object" else "value"))
    _fail()


def contract_from_schema(schema: dict[str, Any], *, envelope: bool) -> Contract:
    groups: list[tuple[str, tuple[str, ...]]] = []
    read: tuple[str, ...] = ()
    for branch in schema["anyOf"]:
        props = branch["properties"]
        action = props["action"]["enum"][0]
        if action == "read":
            read = tuple(props["source_ids"]["items"]["enum"])
        if action == "answer":
            for item in props["documents"]["items"]["anyOf"]:
                fields = item["properties"]
                alias = fields["source_id"]["enum"][0]
                sids = tuple(fields["sentence_ids"]["items"]["enum"])
                if not sids or any(s.rsplit(":", 1)[0] != alias for s in sids):
                    raise ValueError("invalid_source_sentence_contract")
                groups.append((alias, sids))
    if len({a for a, _ in groups}) != len(groups) or len(set(read)) != len(read):
        raise ValueError("duplicate_schema_identity")
    return Contract(tuple(groups), read, envelope, hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()).hexdigest())


class BoundedSciFactParser:
    """LMFE CharacterLevelParser duck type; immutable logical state, local memo only."""

    def __init__(self, base: Any, contract: Contract, config: Any,
                 state: SemanticState | None = None, prefix: str = "") -> None:
        self.base, self.contract, self._config = base, contract, config
        self.state, self.prefix = state or SemanticState(), prefix
        self._transitions: dict[str, SemanticState] | None = None
        self._allowed: str | None = None

    @property
    def config(self) -> Any:
        return self._config

    @config.setter
    def config(self, value: Any) -> None:
        self._config = value
        self.base.config = value

    def get_allowed_characters(self) -> str:
        if self._allowed is not None:
            return self._allowed
        if (self.state.string_path is not None and not self.state.string_is_key
                and self.contract.choices(self.state, self.state.string_path) is None):
            # Semantics impose no extra next-character restriction on a free
            # value string. Avoid scanning the entire tokenizer alphabet per
            # trie node; add_character STILL advances the outer state for every
            # explored character. This is not LMFE's token-level shortcut.
            allowed = _without_raw_controls(str(self.base.get_allowed_characters()))
            if '"' in allowed and not self.state.escape_pending:
                try:
                    advance(self.state, self.contract, '"')
                except (ValueError, IndexError):
                    allowed = allowed.replace('"', '')
            self._allowed = allowed
            return allowed
        if self._transitions is None:
            transitions = {}
            for char in set(self.base.get_allowed_characters()):
                try:
                    transitions[char] = advance(self.state, self.contract, char)
                except (ValueError, IndexError):
                    continue
            self._transitions = transitions
        self._allowed = "".join(sorted(self._transitions))
        return self._allowed

    def add_character(self, char: str) -> BoundedSciFactParser:
        if char not in self.get_allowed_characters():
            _fail()
        state = (self._transitions[char] if self._transitions is not None
                 else advance(self.state, self.contract, char))
        return BoundedSciFactParser(self.base.add_character(char), self.contract,
                                    self.config, state, self.prefix + char)

    def can_end(self) -> bool:
        return (self.state.root_started and not self.state.frames
                and self.state.string_path is None and bool(self.base.can_end()))

    def cache_key(self) -> None:
        # First release deliberately has no cross-state token-set cache. Any
        # future optimization must prove its key covers syntax + all semantics.
        return None

    def shortcut_key(self) -> None:
        # Never forward LMFE's freetext fast path: every token-trie character
        # must pass the outer semantic tracker, including boundary-crossing tokens.
        return None


class CheckedBoundedPrefix:
    """Guard LMFE's silent EOS/ForceStop fallback on the actual HF callback path."""

    def __init__(self, callback: Any) -> None:
        self.callback = callback
        self.token_enforcer = callback.token_enforcer
        self._prompt: tuple[int, ...] | None = None

    def __call__(self, batch_id: int, sent: Any) -> list[int]:
        if batch_id != 0:
            raise ValueError("bounded_prefix_is_single_sequence")
        sequence = tuple(sent.tolist())
        if self._prompt is None:
            self._prompt = sequence
        elif (sequence[:len(self._prompt)] != self._prompt
              or (sequence != self._prompt and sequence not in self.token_enforcer.prefix_states
                  and sequence[:-1] not in self.token_enforcer.prefix_states)):
            raise ValueError("bounded_prefix_history_not_incremental")
        allowed: list[int] = self.callback(batch_id, sent)
        state = self.token_enforcer.prefix_states[sequence]
        if not isinstance(state.parser, BoundedSciFactParser):
            raise ValueError("bounded_prefix_backend_state_replaced")
        eos = self.token_enforcer.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        if set(eos_ids).intersection(allowed) and not state.parser.can_end():
            raise ValueError("bounded_prefix_backend_premature_eos")
        return allowed


def build_bounded_scifact_prefix(tokenizer_data: Any, action_schema: dict[str, Any],
                                 envelope_schema: dict[str, Any] | None = None,
                                 *, plain_partition: Any = None) -> CheckedBoundedPrefix:
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    from lmformatenforcer.integrations.transformers import TransformersPrefixAllowedTokensFn
    from .bounded_token_traversal import ChildFirstTokenEnforcer

    require_clean_grammar_environment()
    if importlib.metadata.version("lm-format-enforcer") != "0.11.3":
        raise ValueError("bounded_prefix_requires_lmfe_0_11_3")
    wrapped = envelope_schema is not None
    config = CharacterLevelParserConfig(alphabet=tokenizer_data.tokenizer_alphabet,
                                       max_consecutive_whitespaces=12,
                                       force_json_field_order=wrapped, max_json_array_length=20)
    parser = BoundedSciFactParser(JsonSchemaParser(envelope_schema or action_schema, config=config),
                                  contract_from_schema(action_schema, envelope=wrapped), config)
    callback = TransformersPrefixAllowedTokensFn(ChildFirstTokenEnforcer(tokenizer_data, parser, plain_partition))
    enforcer = callback.token_enforcer
    if enforcer.prefix_states or enforcer.allowed_token_cache:
        raise ValueError("bounded_prefix_must_start_fresh")
    # TokenEnforcer overwrites root config; reapply before any token is observed.
    parser.config.force_json_field_order = wrapped
    if (parser.config.alphabet != tokenizer_data.tokenizer_alphabet
            or parser.config.max_consecutive_whitespaces != 12
            or parser.config.max_json_array_length != 20):
        raise ValueError("bounded_prefix_effective_config_mismatch")
    return CheckedBoundedPrefix(callback)
