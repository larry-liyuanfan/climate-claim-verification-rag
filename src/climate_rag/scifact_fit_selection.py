"""ID-first selection from a hash-locked JSONL member; no non-FIT gold decoding."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, BinaryIO

from .scifact_grounding import Abstract, GoldClaim, parse_gold
from .scifact_state_supervision import require

NUMBER = re.compile(rb'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?')
INTEGER = re.compile(rb'(?:0|[1-9][0-9]*)')
MAX_LINE = 1024 * 1024


class IdScanner:
    """Scan syntax, decode top-level keys only; never materialise non-ID values."""
    def __init__(self, raw: bytes) -> None:
        require(0 < len(raw) <= MAX_LINE, 'jsonl_line_bound')
        self.raw, self.pos = raw, 0

    def whitespace(self) -> None:
        while self.pos < len(self.raw) and self.raw[self.pos] in b' \t\r\n':
            self.pos += 1

    def take(self, value: int) -> bool:
        self.whitespace()
        if self.pos < len(self.raw) and self.raw[self.pos] == value:
            self.pos += 1
            return True
        return False

    def string(self) -> tuple[int, int]:
        self.whitespace()
        start = self.pos
        require(self.take(34), 'expected_json_string')
        while self.pos < len(self.raw):
            ch = self.raw[self.pos]
            self.pos += 1
            if ch == 34:
                return start, self.pos
            require(ch >= 32, 'json_control_character')
            if ch == 92:
                require(self.pos < len(self.raw), 'json_escape_end')
                esc = self.raw[self.pos]
                self.pos += 1
                if esc == 117:
                    code = self.raw[self.pos:self.pos + 4]
                    require(len(code) == 4 and all(c in b'0123456789abcdefABCDEF' for c in code), 'json_unicode_escape')
                    self.pos += 4
                else:
                    require(esc in b'"\\/bfnrt', 'json_escape')
        raise ValueError('json_unterminated_string')

    def skip(self, depth: int = 0) -> None:
        require(depth < 64, 'json_depth_bound')
        self.whitespace()
        require(self.pos < len(self.raw), 'json_truncated')
        ch = self.raw[self.pos]
        if ch == 34:
            self.string()
        elif ch in (123, 91):
            self.pos += 1
            end = 125 if ch == 123 else 93
            if self.take(end):
                return
            while True:
                if ch == 123:
                    self.string()
                    require(self.take(58), 'json_colon')
                self.skip(depth + 1)
                if self.take(end):
                    return
                require(self.take(44), 'json_comma')
        else:
            for literal in (b'true', b'false', b'null'):
                if self.raw.startswith(literal, self.pos):
                    self.pos += len(literal)
                    return
            match = NUMBER.match(self.raw, self.pos)
            require(match is not None, 'json_primitive')
            if match:
                self.pos = match.end()

    def identifier(self) -> int:
        require(self.take(123), 'top_level_object_required')
        keys: set[str] = set()
        found = None
        while True:
            start, end = self.string()
            key = json.loads(self.raw[start:end])  # keys only, never non-FIT values
            require(key not in keys, 'duplicate_top_level_key')
            keys.add(key)
            require(self.take(58), 'json_colon')
            self.whitespace()
            start = self.pos
            self.skip()
            if key == 'id':
                value = self.raw[start:self.pos]
                require(len(value) <= 20 and INTEGER.fullmatch(value) is not None, 'top_level_id_integer')
                found = int(value)
            if self.take(125):
                break
            require(self.take(44), 'json_comma')
        self.whitespace()
        require(self.pos == len(self.raw) and found is not None, 'missing_id_or_trailing_json')
        if found is None:
            raise ValueError('missing_id')
        return found


def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, 'selected_duplicate_json_key')
        result[key] = value
    return result


def select_complete_fit(stream: BinaryIO, expected_sha: str, fit_ids: Sequence[int],
                        corpus: Mapping[int, Abstract]) -> dict[int, GoldClaim]:
    require(bool(fit_ids) and len(set(fit_ids)) == len(fit_ids)
            and all(type(i) is int for i in fit_ids), 'fit_whitelist')
    wanted, seen, selected = set(fit_ids), set(), {}
    digest = hashlib.sha256()
    while line := stream.readline(MAX_LINE + 1):
        digest.update(line)
        claim_id = IdScanner(line).identifier()
        require(claim_id not in seen, 'duplicate_train_row_id')
        seen.add(claim_id)
        if claim_id in wanted:
            selected[claim_id] = line
    require(digest.hexdigest() == expected_sha, 'train_member_sha')
    require(set(selected) == wanted, 'fit_whitelist_coverage')
    # No unselected row is ever passed to a JSON deserialiser. Nothing is
    # persisted before complete member SHA and whitelist coverage are verified.
    return {i: parse_gold(json.loads(selected[i], object_pairs_hook=no_duplicate_keys), corpus) for i in fit_ids}
