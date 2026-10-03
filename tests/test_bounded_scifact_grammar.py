"""Synthetic generation-contract checks; never a model-quality result."""
import json
import string

import pytest

pytest.importorskip("lmformatenforcer")
from lmformatenforcer import JsonSchemaParser, TokenEnforcerTokenizerData
from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig

from climate_rag.bounded_scifact_grammar import (
    BoundedSciFactParser, build_bounded_scifact_prefix, contract_from_schema,
)
from climate_rag.scifact_terminal import action_schema

ALIASES = ["c0", "c1", "c10", "c2", "c3"]
SIDS = [f"{a}:{i}" for a in ALIASES for i in range(10)]
SCHEMA = action_schema(["abstain", "answer", "read", "rewrite", "rerank"], ALIASES, SIDS, 5)


def root(schema=SCHEMA):
    config = CharacterLevelParserConfig()
    return BoundedSciFactParser(JsonSchemaParser(schema, config=config),
                                contract_from_schema(schema, envelope=False), config)


def feed(parser, text):
    for i, char in enumerate(text):
        if char not in parser.get_allowed_characters():
            return False, parser, i
        parser = parser.add_character(char)
    return parser.can_end(), parser, len(text)


def dump(value):
    return json.dumps(value, separators=(",", ":"))


def answer(counts):
    return {"action": "answer", "documents": [
        {"source_id": a, "label": "SUPPORTS" if j % 2 else "REFUTES",
         "sentence_ids": [f"{a}:{i}" for i in range(n - 1, -1, -1)]}
        for j, (a, n) in enumerate(zip(ALIASES, counts))]}


@pytest.mark.parametrize("counts", [(8, 8, 4), (8, 7, 5), (8,), (4, 4, 4, 4, 4), (1,)])
def test_full_old_expressive_capacity_and_order(counts):
    assert feed(root(), dump(answer(counts)))[0]


@pytest.mark.parametrize("value", [
    answer((8, 8, 5)), answer((9,)),
    {"action": "read", "source_ids": ["c1", "c10", "c1"]},
    {"action": "answer", "documents": [answer((1,))["documents"][0]] * 2},
    {"action": "answer", "documents": [{"source_id": "c0", "label": "SUPPORTS",
                                         "sentence_ids": ["c0:0", "c0:1", "c0:0"]}]},
])
def test_rejects_before_invalid_complete_output(value):
    ok, parser, index = feed(root(), dump(value))
    assert not ok and index < len(dump(value))
    assert not parser.can_end()


def test_total_budget_rejects_impossible_comma_not_only_twenty_first_item():
    text = dump(answer((8, 8, 4)))
    prefix = text[:text.rfind("]")]
    _, parser, index = feed(root(), prefix)
    assert index == len(prefix)
    assert "," not in parser.get_allowed_characters()  # no new document after total20
    text = dump(answer((8, 8, 4)))
    end_sid = text.rfind('"]') + 1
    _, parser, index = feed(root(), text[:end_sid])
    assert index == end_sid
    assert "," not in parser.get_allowed_characters()  # no 21st SID in current document
    assert "]" in parser.get_allowed_characters()


def test_any_field_order_preserves_shared_prefix_aliases_and_mixed_labels():
    value = answer((1, 1, 1))
    for d in value["documents"]:
        ordered = {"sentence_ids": d["sentence_ids"], "label": d["label"], "source_id": d["source_id"]}
        d.clear()
        d.update(ordered)
    reordered = {"documents": value["documents"], "action": "answer"}
    assert feed(root(), dump(reordered))[0]
    assert feed(root(), dump({"action": "read", "source_ids": ["c1", "c10"]}))[0]


@pytest.mark.parametrize("value", [
    {"action": "read", "source_ids": ["c1", "c10"]},
    {"action": "rewrite", "query": 'A "quoted" unicode α query'},
    {"action": "rerank"},
    {"action": "abstain", "reason": "insufficient_evidence"},
    answer((1,)),
])
def test_five_actions_remain_reachable(value):
    assert feed(root(), dump(value))[0]


def test_semantic_state_and_old_parser_are_immutable_for_branches():
    start = root()
    initial_key = start.cache_key()
    a = dump(answer((8, 8, 4)))
    b = dump({"action": "read", "source_ids": ["c1", "c10"]})
    assert feed(start, a)[0] and feed(start, b)[0] and feed(start, a)[0]
    assert start.cache_key() == initial_key and not start.state.documents


def toy_data():
    vocab = list(dict.fromkeys(string.printable + "α")) + ['{"', '"}', '","', "c1", "c10", "0:0", ']}']
    decoder = lambda ids: "".join(vocab[i] for i in ids)  # noqa: E731
    eos = len(vocab)
    data = TokenEnforcerTokenizerData([(i, s, False) for i, s in enumerate(vocab)],
                                     decoder, eos, False, eos + 1)
    return data, vocab, eos


def test_real_hf_prefix_callback_with_multicharacter_toy_tokens():
    torch = pytest.importorskip("torch")
    data, vocab, eos = toy_data()
    def check(text):
        callback = build_bounded_scifact_prefix(data, SCHEMA)
        sent = [eos]
        # Greedy longest match deliberately exercises punctuation/ID-crossing tokens.
        rest = text
        while rest:
            token = max((i for i, s in enumerate(vocab) if rest.startswith(s)), key=lambda i: len(vocab[i]))
            if token not in callback(0, torch.tensor(sent)):
                return False
            sent.append(token)
            rest = rest[len(vocab[token]):]
        return eos in callback(0, torch.tensor(sent))
    assert check(dump(answer((8, 8, 4))))
    assert check(dump({"action": "read", "source_ids": ["c1", "c10"]}))
    assert check(dump(answer((8, 7, 5))))
    assert not check(dump(answer((8, 8, 5))))
    assert not check(dump({"action": "read", "source_ids": ["c1", "c1"]}))


@pytest.mark.parametrize("failure", ["force_stop", "premature_eos", "missing_predecessor"])
def test_actual_callback_silent_backend_fallback_guards(failure):
    torch = pytest.importorskip("torch")
    from lmformatenforcer.characterlevelparser import ForceStopParser
    data, _, eos = toy_data()
    callback = build_bounded_scifact_prefix(data, SCHEMA)
    prompt = torch.tensor([eos])
    callback(0, prompt)
    state = callback.token_enforcer.prefix_states[(eos,)]
    if failure == "force_stop":
        state.parser = ForceStopParser()
    elif failure == "premature_eos":
        state.allowed_tokens.append(eos)
    else:
        prompt = torch.tensor([eos, 1, 2])
    with pytest.raises(ValueError, match={"force_stop": "state_replaced", "premature_eos": "premature_eos",
                                        "missing_predecessor": "not_incremental"}[failure]):
        callback(0, prompt)


def test_effective_outer_inner_configuration_and_envelope_order():
    pytest.importorskip("torch")
    from climate_rag.evidence_gap_candidate import gap_schema
    data, _, _ = toy_data()
    for envelope in (None, gap_schema(SCHEMA)):
        callback = build_bounded_scifact_prefix(data, SCHEMA, envelope)
        parser = callback.token_enforcer.root_parser
        assert parser.config is parser.base.config
        assert parser.config.alphabet == data.tokenizer_alphabet
        assert parser.base.context.alphabet_without_quotes == data.tokenizer_alphabet.replace('"', '')
        assert parser.config.force_json_field_order == (envelope is not None)
        assert parser.config.max_consecutive_whitespaces == 12
        assert parser.cache_key() is None and parser.shortcut_key() is None


def test_whitespace_escapes_and_terminal_completion():
    text = dump({"action": "rewrite", "query": 'space α \\ " [ ], { }'})
    assert feed(root(), text)[0]
    good = dump({"action": "abstain", "reason": "budget"})
    assert feed(root(), good + " " * 12)[0]
    accepted, _, index = feed(root(), good + " " * 13)
    assert not accepted and index == len(good) + 12


@pytest.mark.parametrize("fragment", ['a\t', 'a\x01'])
def test_raw_control_character_cannot_close_free_string(fragment):
    _, parser, index = feed(root(), '{"action":"rewrite","query":"' + fragment)
    assert index == len('{"action":"rewrite","query":"' + fragment) - 1
    assert fragment[-1] not in parser.get_allowed_characters()


@pytest.mark.parametrize("fragment", [r'a\t', r'a\u0009', r'a\"b', r'\u03b1'])
def test_valid_escaped_free_strings_still_close(fragment):
    assert feed(root(), '{"action":"rewrite","query":"' + fragment + '"}')[0]


def test_child_first_traversal_matches_pinned_original_on_toy_vocab():
    pytest.importorskip("torch")
    from lmformatenforcer.tokenenforcer import TokenEnforcer
    from lmformatenforcer.tokenlist import TokenList
    data, _, _ = toy_data()
    optimized = build_bounded_scifact_prefix(data, SCHEMA).token_enforcer
    original = TokenEnforcer(data, root())
    for prefix in ['', '{"action":"', '{"action":"rewrite","query":"ab',
                   '{"action":"rewrite","query":"a\\',
                   dump(answer((8, 8, 4)))[:-3]]:
        _, parser, consumed = feed(root(), prefix)
        assert consumed == len(prefix)
        a, b = TokenList(False, data.vocab_size), TokenList(False, data.vocab_size)
        original._collect_allowed_tokens(parser, data.tokenizer_tree.root, a, None)
        optimized._collect_allowed_tokens(parser, data.tokenizer_tree.root, b, None)
        assert set(a.allowed_tokens) == set(b.allowed_tokens)


def test_plain_partition_exact_allowed_set_matches_unpartitioned_hf():
    torch = pytest.importorskip("torch")
    from climate_rag.bounded_token_traversal import PlainStringPartition
    from climate_rag.evidence_gap_candidate import gap_schema
    vocab = list(dict.fromkeys(string.printable + "α中\x00\x01")) + [
        "abc", "abc ", "   a", " " * 12, " " * 13, 'a"', 'a"}',
        'a\t"', 'a\t', r'\t', r'\u03b1', r'a\"', "abc\\", "c1", "c10", '","']
    eos = len(vocab)
    data = TokenEnforcerTokenizerData([(i, s, False) for i, s in enumerate(vocab)],
                                     lambda ids: ''.join(vocab[i] for i in ids), eos, False, eos + 1)
    partition = PlainStringPartition(data)
    action = json.loads(json.dumps(SCHEMA))
    rewrite = next(b for b in action['anyOf'] if b['properties']['action']['enum'] == ['rewrite'])
    rewrite['properties']['query'].update(minLength=2, maxLength=7)
    for text, envelope in [
        ('{"action":"rewrite","query":"α中 ab"}', None),
        ('{"action":"rewrite","query":"  ab  "}', None),
        (r'{"action":"rewrite","query":"a\t b"}', None),
        (r'{"action":"rewrite","query":"\u03b1b"}', None),
        ('{"action":"rewrite","query":"' + ' ' * 12 + 'ab"}', None),
        ('{"action":"read","source_ids":["c1","c10"]}', None),
        (dump(answer((8, 8, 4))), None),
        ('{"evidence_state":{"retrieval_need":"uncertain","relevance":"unknown",'
         '"support":"unknown"},"gap_claim_span":"α中","decision":{"action":"rerank"}}', gap_schema(action)),
    ]:
        slow = build_bounded_scifact_prefix(data, action, envelope)
        fast = build_bounded_scifact_prefix(data, action, envelope, plain_partition=partition)
        sent = [eos]
        for char in text:
            a, b = set(slow(0, torch.tensor(sent))), set(fast(0, torch.tensor(sent)))
            assert a == b, (text, len(sent), a ^ b)
            token = vocab.index(char)
            if token not in a:
                break  # Pinned LMFE string-length accounting may reject long escapes.
            sent.append(token)
        assert set(slow(0, torch.tensor(sent))) == set(fast(0, torch.tensor(sent)))


def test_same_alphabet_different_token_identity_partition_is_rejected():
    pytest.importorskip("torch")
    from climate_rag.bounded_token_traversal import PlainStringPartition
    a, _, _ = toy_data()
    b, _, _ = toy_data()
    assert a.tokenizer_alphabet == b.tokenizer_alphabet
    # Even equal vocab contents require the exact source tree; swapping an ID
    # or passing a partition from another tokenizer must never bulk-accept it.
    with pytest.raises(ValueError, match="tokenizer_identity"):
        build_bounded_scifact_prefix(b, SCHEMA, plain_partition=PlainStringPartition(a))
