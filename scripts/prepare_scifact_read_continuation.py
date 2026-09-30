"""CPU-only exact second-state reconstruction; no gold reading or model loading."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tarfile
from typing import Any

from climate_rag.scifact_adapter_regression import require
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_read_continuation import OLD_STATES_SHA, ordered_write, validate_state
from climate_rag.scifact_retrieval import SciFactBM25
from climate_rag.scifact_semantic_contract import (
    CORPUS_MEMBER, CORPUS_SHA, INPUT_SHA, PRIVATE_SHA, TOKENIZER_SHA, checked, encoded, sha, write_once,
)
from climate_rag.scifact_terminal import render_scifact_prompt
from prepare_scifact_semantic_pair import verify_source_tree
from replay_scifact_selected_read import ROOT, replay


def reconstruct(claim: Any, review: Any, expected: Any, retrieve: Any,
                corpus: Any, tokenizer: Any) -> dict[str, Any]:
    frames: list[tuple[Any, Any]] = []
    def capture(obs: Any, schemas: Any) -> None:
        frames.extend(zip(obs, schemas, strict=True))
    current = replay(claim, review, retrieve, corpus, tokenizer, capture=capture)
    require(json.loads(encoded(current)) == json.loads(encoded(expected)), 'original_replay_metadata_changed')
    require(len(frames) == 2 and len(current['frozen_read_ids']) == 1, 'exact_one_saved_read')
    obs, schema = frames[1]
    prompt = render_scifact_prompt(tokenizer, obs, schema)
    row = {'claim_id':claim['id'], 'candidate_doc_ids':current['candidate_doc_ids'],
        'ordered_candidates':current['ordered_candidates'], 'frozen_read_ids':current['frozen_read_ids'],
        'observation':obs, 'schema':schema, 'expected_state':expected['states'][1],
        'new_token_ids_sha256':sha(encoded(tokenizer.encode(prompt, add_special_tokens=False))),
        'token_ids_hash_has_no_old_reference':True, 'scripted_read_not_model_action':True}
    validate_state(row, tokenizer)
    return row


def main() -> None:
    import resource
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-git', required=True)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    require(os.name == 'posix' and os.environ.get('PYTHONHASHSEED') == '0'
            and os.environ.get('USE_TORCH') == '0', 'cpu_tokenizer_only')
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    source = Path(__file__).resolve().parents[1]
    verify_source_tree(source, args.source_archive, args.source_sha, args.source_git)
    prior = ROOT/'posthoc/scifact-selected-read-opportunity-dcbe368c1c02/private/states-before-gold.json'
    old = json.loads(checked(prior, OLD_STATES_SHA))
    prepared = ROOT/'posthoc/scifact-semantic-preparation-779e49883570/private'
    def read(name: str) -> Any:
        return json.loads(checked(prepared/name, PRIVATE_SHA[name]))
    selection = read('selected-before-probe.json')['selection']
    claims, reviews = read('selected-inference.json'), read('current-opportunity.json')
    require([r['id'] for r in selection] == [r['id'] for r in claims] == [r['id'] for r in reviews]
            == [r['claim_id'] for r in old] and len(old) == 12, 'old_order')
    chosen = [i for i, r in enumerate(old) if r['frozen_read_ids']]
    require(len(chosen) == 3 and all(old[i]['legacy_stratum'] == 'top20_doc_replenishable' for i in chosen),
            'only_original_three_no_reselection')
    output = ROOT/'posthoc'/('scifact-read-continuation-'+args.source_git[:12])
    output.mkdir(mode=0o700)
    token_dir = output/'tokenizer'
    token_dir.mkdir(mode=0o700)
    archive_path = ROOT/'envs/scifact-semantic-inputs-c4907db6.tar'
    checked(archive_path, INPUT_SHA)
    with tarfile.open(archive_path) as archive:
        stream = archive.extractfile(CORPUS_MEMBER)
        require(stream is not None, 'corpus_missing')
        raw = stream.read() if stream else b''
        require(sha(raw) == CORPUS_SHA, 'corpus_changed')
        corpus = {d.doc_id:d for d in (parse_abstract(json.loads(line)) for line in raw.splitlines())}
        require(len(corpus) == 5183, 'corpus_shape')
        for name, digest in TOKENIZER_SHA.items():
            stream = archive.extractfile('data/qwen3-4b-tokenizer-v3/'+name)
            raw = stream.read() if stream else b''
            require(sha(raw) == digest, 'tokenizer_changed')
            with (token_dir/name).open('xb') as handle:
                handle.write(raw)
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(token_dir, local_files_only=True, trust_remote_code=False)
    retrieve = SciFactBM25(corpus)
    rows = [reconstruct(claims[i], reviews[i], old[i], retrieve, corpus, tokenizer) for i in chosen]
    # FIT membership is reporting metadata, never an input field or a selection rule.
    split_sha = '578f57ec2955e9d51c1da7d4d219c15c339be7aa0790f283e3872221f3fb432a'
    split = json.loads(checked(ROOT/'posthoc/scifact-grounding-candidate-40d84a377bd1/private/selection-before-packing.json', split_sha))
    ids = set(split['fit'])
    memberships = [selection[i]['id'] in ids for i in chosen]
    require(len(ids) == 48 and sum(memberships) == 1, 'one_fit_two_no_direct_fit')
    bundle = output/'conditional-inputs.json'
    ordered_write(bundle, {'rows':rows})
    reloaded = json.loads(bundle.read_bytes())
    for row in reloaded['rows']:
        validate_state(row, tokenizer)
    write_once(output/'private-membership.json', {'fit_overlap':memberships})
    compact = {'source_git':args.source_git, 'source_archive_sha256':args.source_sha,
        'old_states_sha256':OLD_STATES_SHA, 'conditional_inputs_sha256':sha(bundle.read_bytes()),
        'fit_membership_metadata_sha256':split_sha,
        'private_membership_sha256':sha((output/'private-membership.json').read_bytes()),
        'states':[{k:r['expected_state'][k] for k in ('observation_sha256','schema_sha256','state_sha256','prompt_sha256','prompt_tokens')}
                  | {'new_token_ids_sha256':r['new_token_ids_sha256']} for r in rows],
        'exact_second_states':3, 'all_original_metadata_matched':True,
        'preserved_nested_key_order_roundtrip':True, 'scripted_fixture_responses':6,
        'scripted_reads':3, 'model_calls':0, 'reranker_calls':0, 'gold_loaded':False,
        'fit_overlap':1, 'no_direct_fit_overlap':2, 'validation_read':False,
        'token_id_hashes_are_new_not_historical_proofs':True,
        'boundary':'oracle_read_conditional_opportunity_not_autonomous_success_or_heldout',
        'maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write_once(output/'compact.json', compact)
    print(json.dumps(compact, sort_keys=True))


if __name__ == '__main__':
    main()
