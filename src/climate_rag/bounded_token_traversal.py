"""Exact plain-string trie partition for the pinned LMFE 0.11.3 parser.

Unlike LMFE's json_freetext cache this never bulk-accepts quotes, escapes, raw
control characters, or boundary-crossing tokens. All such tokens traverse the
complete outer parser. This file does not patch the installed dependency.
"""
from __future__ import annotations

from typing import Any

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.tokenenforcer import TokenEnforcer
from lmformatenforcer.tokenizerprefixtree import TokenizerPrefixTreeNode


class PlainStringPartition:
    def __init__(self, data: Any) -> None:
        self.source_tree = data.tokenizer_tree
        self.alphabet = data.tokenizer_alphabet
        alphabet = set(self.alphabet)
        self.exceptional = TokenizerPrefixTreeNode()
        self.plain: list[tuple[int, int, int, int]] = []
        self.max_length = 0
        for token, text, _ in data.regular_tokens:
            if text and all(ord(c) >= 32 and c not in '"\\' and c in alphabet for c in text):
                leading = len(text) - len(text.lstrip(' '))
                run = peak = 0
                for c in text:
                    run = run + 1 if c == ' ' else 0
                    peak = max(peak, run)
                self.plain.append((token, len(text), leading, peak))
                self.max_length = max(self.max_length, len(text))
            else:
                node = self.exceptional
                for char in text:
                    node = node.children.setdefault(char, TokenizerPrefixTreeNode())
                node.tokens.append(token)
        self._cache: dict[tuple[int, int, bool, int], tuple[int, ...]] = {}

    def tokens(self, current_length: int, maximum: int, whitespace: int, whitespace_limit: int) -> tuple[int, ...]:
        remaining = min(self.max_length, maximum - current_length)
        key = (remaining, whitespace, current_length == 0, whitespace_limit)
        if key not in self._cache:
            # Match StringParsingState: leading spaces do not increment its
            # parsed_string while empty; JsonSchemaParser still counts them.
            selected = tuple(t for t, n, leading, peak in self.plain
                             if n - (leading if current_length == 0 else 0) <= remaining
                             and max(whitespace + leading, peak) <= whitespace_limit)
            if len(self._cache) >= 128:
                self._cache.clear()  # memory bound, never a semantic approximation
            self._cache[key] = selected
        return self._cache[key]


class ChildFirstTokenEnforcer(TokenEnforcer):  # type: ignore[misc]  # pinned dependency has no typed exports
    def __init__(self, data: Any, parser: Any, partition: PlainStringPartition | None = None):
        if partition is not None and partition.source_tree is not data.tokenizer_tree:
            raise ValueError("plain_partition_tokenizer_identity_mismatch")
        super().__init__(data, parser)
        self.plain_partition = partition

    def _plain_state(self, parser: Any) -> tuple[int, int] | None:
        state = parser.state
        if (self.plain_partition is None or state.string_path is None or state.string_is_key
                or state.escape_pending or parser.contract.choices(state, state.string_path) is not None
                or type(parser.base) is not JsonSchemaParser
                or parser.config.alphabet != self.plain_partition.alphabet):
            return None
        key = parser.base.shortcut_key()
        if not isinstance(key, tuple) or len(key) != 4 or key[0] != 'json_freetext':
            return None
        return key[1], key[3]

    def _collect_allowed_tokens(self, parser: Any, tree_node: Any,
                                allowed_tokens: Any, shortcut_key: Any) -> None:
        if shortcut_key is not None:
            raise ValueError("bounded_prefix_does_not_use_token_shortcuts")
        # Only partition at vocabulary root. The exceptional trie is complete
        # for every nonplain token, including strings crossing a closing quote.
        plain = self._plain_state(parser) if tree_node is self.tokenizer_tree.root else None
        if plain is not None:
            assert self.plain_partition is not None
            allowed_tokens.extend(self.plain_partition.tokens(
                *plain, parser.base.num_consecutive_whitespaces, parser.config.max_consecutive_whitespaces))
            tree_node = self.plain_partition.exceptional
        allowed_tokens.extend(tree_node.tokens)
        allowed = parser.get_allowed_characters()
        for char, child in tree_node.children.items():
            if char in allowed:
                self._collect_allowed_tokens(parser.add_character(char), child, allowed_tokens, None)
