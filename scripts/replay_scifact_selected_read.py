"""Selected-only CPU opportunity replay; no selection, model, reranker or scheduler."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import tarfile
import time
from types import SimpleNamespace
from typing import Any, Callable

from climate_rag.agent_v3 import V3Budget
from climate_rag.local_bounded_scifact_provider import observation_identity
from climate_rag.scifact_bounded_runtime import ARMS, run_bounded_slot
from climate_rag.scifact_grounding import parse_abstract, parse_gold
from climate_rag.scifact_retrieval import SciFactBM25
from climate_rag.scifact_semantic_contract import (
    CORPUS_MEMBER, CORPUS_SHA, INPUT_SHA, PRIVATE_SHA, TOKENIZER_SHA, encoded, sha, write_once,
)
from climate_rag.scifact_semantic_policy import OLD_G, POLICIES
from climate_rag.scifact_terminal import render_scifact_prompt
from climate_rag.scifact_train_diagnostic import PackingProbe, document_opportunities, unused_reranker
from prepare_scifact_semantic_pair import checked, verify_source_tree

ROOT = Path('/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2')
PACKING = 'existing_CommonPacking_max_bare_empty_G_active_not_pure_bare_counting'


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


class ReadReplay:
    gap = False
    name, kind, terminal_protocol = PackingProbe.name, PackingProbe.kind, PackingProbe.terminal_protocol

    def __init__(self, tokenizer: Any, read_ids: list[str]) -> None:
        self.probe = PackingProbe(tokenizer, read_ids)
        self.observations = self.probe.observations
        self.base = SimpleNamespace(tokenizer=tokenizer)
        self.schemas: list[dict[str, Any]] = []

    def count_text(self, text: str) -> int:
        return int(self.probe.count_text(text))

    def count_prompt(self, observation: dict[str, Any], schema: dict[str, Any]) -> int:
        return int(self.probe.count_prompt(observation, schema))

    def generate(self, observation: dict[str, Any], schema: dict[str, Any],
                 max_output_tokens: int, remaining_seconds: float) -> dict[str, Any]:
        self.schemas.append(copy.deepcopy(schema))
        return dict(self.probe.generate(observation, schema, max_output_tokens, remaining_seconds))


def fixed_read(review: dict[str, Any], candidates: list[int]) -> list[str]:
    old = review['current_opportunity'][OLD_G]
    if review['legacy_stratum'] != 'top20_doc_replenishable':
        require('witness' not in old, 'unexpected_saved_witness')
        return []
    witnesses = [review['current_opportunity'][p] for p in POLICIES]
    read_ids = list(old['witness']['fixture_history']['read_ids'])
    require(len(read_ids) == 1, 'one_original_read_only')
    require(all(w['witness']['fixture_history']['read_ids'] == read_ids
                and w['witness_read_doc_ids'] == old['witness_read_doc_ids'] for w in witnesses),
            'original_policy_witness_disagreement')
    for alias, doc in zip(read_ids, old['witness_read_doc_ids'], strict=True):
        require(alias.startswith('c') and alias[1:].isdigit(), 'original_alias_format')
        index = int(alias[1:])
        require(index < len(candidates) and candidates[index] == doc
                == old['initial']['candidate_doc_ids'][index], 'witness_alias_target_changed')
    return read_ids


def replay(claim: dict[str, Any], review: dict[str, Any], retrieve: Any,
           corpus: Any, tokenizer: Any,
           capture: Callable[[list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
           ) -> dict[str, Any]:
    candidates = list(retrieve(claim['claim'], 20))
    candidate_ids = [int(s.source_id) for s in candidates]
    read_ids = fixed_read(review, candidate_ids)
    provider = ReadReplay(tokenizer, read_ids)
    def same_query(query: str, width: int) -> Any:
        from climate_rag.verification import normalise_claim
        require(query == normalise_claim(claim['claim']) and width == 20, 'no_rewrite_or_width_change')
        return candidates
    row = run_bounded_slot(claim['id'], claim['claim'], 'adaptive', ARMS[0],
        provider, same_query, unused_reranker, corpus)
    result = row['result']
    reads = [e for e in result['events'] if e['tool'] == 'read' and e['status'] == 'completed']
    require(len(provider.observations) == 1 + bool(read_ids)
            and len(reads) == bool(read_ids)
            and result['outcome'] == 'model_abstention:insufficient_evidence', 'fixed_scripted_history_failed')
    states = []
    for observation, schema in zip(provider.observations, provider.schemas, strict=True):
        visible: dict[int, list[int]] = {}
        for entry in observation['current_citable']:
            alias, index = entry['sentence_id'].split(':')
            visible.setdefault(candidate_ids[int(alias[1:])], []).append(int(index))
        prompt = render_scifact_prompt(tokenizer, observation, schema)
        states.append({'observation_sha256':sha(encoded(observation)), 'schema_sha256':sha(encoded(schema)),
            'state_sha256':sha(encoded({'observation':observation,'schema':schema})),
            'prompt_sha256':sha(prompt.encode()), 'prompt_tokens':provider.count_prompt(observation,schema),
            'identity':observation_identity(observation),'visible':visible})
    old = review['current_opportunity'][OLD_G]
    if capture is not None:
        # Preserve nested insertion order: the renderer does NOT sort JSON keys.
        capture(copy.deepcopy(provider.observations), copy.deepcopy(provider.schemas))
    return {'claim_id':claim['id'],'legacy_stratum':review['legacy_stratum'],
        'candidate_doc_ids':candidate_ids,
        'ordered_candidates': [{'doc_id':int(s.source_id),'source_sha256':s.text_sha256} for s in candidates],
        'candidate_order_sha256':sha(encoded(candidate_ids)),
        'candidate_order_unchanged':candidate_ids == old['initial']['candidate_doc_ids'],
        'initial_identity_unchanged':states[0]['identity'] == old['initial']['initial_identity'],
        'frozen_read_ids':read_ids,'scripted_completed_reads':len(reads),
        'fixture_responses':len(states),'model_calls':0,'reranker_calls':0,'states':states}


def summarise(states: list[dict[str, Any]], gold: Any) -> dict[str, Any]:
    changes: Counter[str] = Counter()
    initial_docs = final_docs = initial_claims = post_read_claims = new_read_claims = 0
    private_diagnosis = []
    for row in states:
        g = gold[row['claim_id']]
        initial = document_opportunities(g,row['states'][0]['visible'])
        final = document_opportunities(g,row['states'][-1]['visible'])
        initial_docs += len(initial)
        final_docs += len(final)
        initial_claims += bool(initial)
        if row['frozen_read_ids']:
            post_read_claims += bool(final)
            new_read_claims += bool(set(final)-set(initial))
        if not g.evidence:
            current = 'nei'
        elif initial:
            current = 'initial_doc_opportunity'
        elif row['frozen_read_ids'] and final:
            current = 'top20_doc_replenishable_by_original_witness'
        elif not set(g.evidence).intersection(row['candidate_doc_ids']):
            current = 'gold_absent_top20'
        else:
            current = 'no_complete_first3_rationale_under_original_witness_only'
        changes[row['legacy_stratum']+'->'+current] += 1
        private_diagnosis.append({'claim_id':g.claim_id,'initial_eligible_gold_doc_ids':initial,
            'final_eligible_gold_doc_ids':final,'current_class':current})
    return {'selected_queries':len(states),'fixture_responses':sum(r['fixture_responses'] for r in states),
        'candidate_orders_unchanged':sum(r['candidate_order_unchanged'] for r in states),
        'initial_visible_preview_identities_unchanged':sum(r['initial_identity_unchanged'] for r in states),
        'initial_complete_first3_eligible_claims':initial_claims,'initial_eligible_gold_docs':initial_docs,
        'final_eligible_gold_docs':final_docs,'scripted_read_events':sum(r['scripted_completed_reads'] for r in states),
        'original_read_witnesses_with_complete_first3_rationale':post_read_claims,
        'original_read_witnesses_newly_reachable':new_read_claims,'opportunity_changes':dict(changes),
        'private_diagnosis':private_diagnosis}


def main() -> None:
    import resource
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-git',required=True)
    parser.add_argument('--source-archive',type=Path,required=True)
    parser.add_argument('--source-sha',required=True)
    args = parser.parse_args()
    require(os.name == 'posix' and os.environ.get('PYTHONHASHSEED') == '0'
            and os.environ.get('USE_TORCH') == '0', 'cpu_tokenizer_only_environment')
    resource.setrlimit(resource.RLIMIT_CPU,(60,60))
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    started = time.monotonic()
    source = Path(__file__).resolve().parents[1]
    verify_source_tree(source,args.source_archive,args.source_sha,args.source_git)
    output = ROOT/'posthoc'/('scifact-selected-read-opportunity-'+args.source_git[:12])
    output.mkdir(mode=0o700)
    private = output/'private'
    private.mkdir(mode=0o700)
    prepared = ROOT/'posthoc/scifact-semantic-preparation-779e49883570/private'
    def read(name: str) -> Any:
        return json.loads(checked(prepared/name,PRIVATE_SHA[name]))
    selected = read('selected-before-probe.json')['selection']
    claims = read('selected-inference.json')
    reviews = read('current-opportunity.json')
    ids = [r['id'] for r in selected]
    require(len(ids) == len(set(ids)) == 12 and ids == [r['id'] for r in claims]
            == [r['id'] for r in reviews], 'unchanged_twelve_selected_order')
    require(Counter(r['legacy_stratum'] for r in reviews) == dict.fromkeys(
        ('initial_doc_opportunity','top20_doc_replenishable','gold_absent_top20','nei'),3), 'frozen_three_each')
    archive_path = ROOT/'envs/scifact-semantic-inputs-c4907db6.tar'
    checked(archive_path,INPUT_SHA)
    tokenizer_dir = output/'tokenizer'
    tokenizer_dir.mkdir(mode=0o700)
    with tarfile.open(archive_path) as archive:
        stream = archive.extractfile(CORPUS_MEMBER)
        require(stream is not None,'missing_corpus')
        raw = stream.read() if stream is not None else b''
        require(sha(raw) == CORPUS_SHA,'fixed_corpus')
        corpus = {d.doc_id:d for d in (parse_abstract(json.loads(line)) for line in raw.splitlines())}
        require(len(corpus) == 5183,'corpus_count')
        for name,digest in TOKENIZER_SHA.items():
            stream = archive.extractfile('data/qwen3-4b-tokenizer-v3/'+name)
            require(stream is not None,'missing_tokenizer')
            raw = stream.read() if stream is not None else b''
            require(sha(raw) == digest,'tokenizer_hash')
            with (tokenizer_dir/name).open('xb') as saved:
                saved.write(raw)
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,local_files_only=True,trust_remote_code=False)
    retrieve = SciFactBM25(corpus)
    states = [replay(c,r,retrieve,corpus,tokenizer) for c,r in zip(claims,reviews,strict=True)]
    require(sum(r['fixture_responses'] for r in states) == 15,'exact_fifteen_scripted_responses')
    write_once(private/'states-before-gold.json',states)
    # Inspect only the already selected twelve gold rows AFTER state freezing.
    gold_rows = read('selected-gold.json')
    gold = {g.claim_id:g for g in (parse_gold(r,corpus) for r in gold_rows)}
    require([r['id'] for r in gold_rows] == ids,'selected_gold_order')
    summary = summarise(states,gold)
    write_once(private/'diagnosis.json',summary.pop('private_diagnosis'))
    compact = dict(summary, source_git=args.source_git,source_archive_sha256=args.source_sha,
        input_archive_sha256=INPUT_SHA,selected_inputs_sha256={n:PRIVATE_SHA[n] for n in
            ('selected-before-probe.json','selected-inference.json','current-opportunity.json','selected-gold.json')},
        corpus_sha256=CORPUS_SHA,tokenizer_sha256=TOKENIZER_SHA,budget=V3Budget().model_dump(),
        actual_generation_envelope='bare_gap_false_no_G_adapter',packing=PACKING,
        private_artifact_sha256={p.name:sha(p.read_bytes()) for p in private.iterdir()},
        model_calls=0,reranker_calls=0,rewrite_opportunity='unknown_not_probed',
        selection_or_witness_search=False,full_531_probe=False,training=False,official_dev_test_read=False,
        fixture_history_not_model_trajectory=True,
        elapsed_seconds=time.monotonic()-started,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        boundary='gold_preparation_seen_selected_TRAIN_oracle_read_opportunity_not_Agent_or_test_gain')
    write_once(output/'compact.json',compact)
    print(json.dumps(compact,sort_keys=True))


if __name__ == '__main__':
    main()
