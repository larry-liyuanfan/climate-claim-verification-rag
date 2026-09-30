"""One FIT-only CPU preparation; complete gold only for frozen original 48 IDs."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import sys
import tarfile
import time
from typing import Any

from climate_rag.scifact_fit_selection import select_complete_fit
from climate_rag.scifact_grounding import Abstract, GoldClaim, LABEL_TO_PROJECT, parse_abstract
from climate_rag.scifact_grounding_sft import canonical_context
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_retrieval import SciFactBM25
from climate_rag.scifact_semantic_contract import CORPUS_SHA, INPUT_SHA, TOKENIZER_SHA, checked, encoded, sha, write_once
from climate_rag.scifact_state_supervision import (
    CONFIG, VERSION, build_claim, candidates_from_rows, config_sha, require,
    tokenize_target, validate_tokenized, validate_weights,
)
from prepare_scifact_semantic_pair import PREP, PREP_SHA, verify_source_tree

ROOT = Path('/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2')
OLD = 'posthoc/scifact-grounding-candidate-40d84a377bd1'
MANIFEST_SHA = '28de2c5d1d531aabdb757ccb45db0b91232a54ac7089ac7dc5ebf42e65ba3b2f'
FILES = {
    'fit/records.json': '70e40db85b0e51292ee1235be6bb708272846a0fcd4be2d123cfb4f2238d15a5',
    'private/selection-before-packing.json': '578f57ec2955e9d51c1da7d4d219c15c339be7aa0790f283e3872221f3fb432a',
    'private/families.json': '8b151a3a031b3cf7937c6394632a26dc64d9e7284bf976fffcdffc50d74e70cc',
    'inference/corpus.jsonl': CORPUS_SHA,
}


def narrow_inputs(directory: Path, files: dict[str, str] = FILES,
                  manifest_sha: str = MANIFEST_SHA) -> tuple[list[int], list[dict[str, Any]], dict[str, Any], dict[int, Abstract]]:
    manifest = json.loads(checked(directory/'manifest.json', manifest_sha))
    require(all(manifest['files'][n] == h for n, h in files.items()), 'old_manifest_allowlist')
    raw = {n: checked(directory/n, h) for n, h in files.items()}
    fit = json.loads(raw['private/selection-before-packing.json'])['fit']
    rows = json.loads(raw['fit/records.json'])
    require(len(fit) == 48 and len(set(fit)) == 48 and all(type(i) is int for i in fit)
            and {r['claim_id'] for r in rows} == set(fit), 'exact_original_48_fit')
    families = json.loads(raw['private/families.json'])
    require(families['corpus_sha256'] == CORPUS_SHA and families['claim_grouping_recomputed'] is False,
            'family_identity')
    corpus = {d.doc_id: d for d in (parse_abstract(json.loads(line))
                                  for line in raw['inference/corpus.jsonl'].splitlines())}
    return fit, rows, families, corpus


def member(bundle: tarfile.TarFile, name: str) -> Any:
    matches = [m for m in bundle.getmembers() if m.name == name]
    require(len(matches) == 1 and matches[0].isfile(), 'unique_regular_archive_member')
    stream = bundle.extractfile(matches[0])
    require(stream is not None, 'archive_member_unavailable')
    return stream


def validate_old_rows(rows: list[dict[str, Any]], claims: dict[int, GoldClaim],
                      families: dict[str, Any], corpus: dict[int, Abstract]) -> dict[int, str]:
    components: dict[int, str] = {}
    identities = set()
    for row in rows:
        claim, doc = claims[row['claim_id']], row['document_id']
        require(row['provenance'] == 'official_annotated_document_alternative' and row['weight'] == 1,
                'v1_target_provenance')
        identity = (claim.claim_id, doc, row['alternative_index'])
        require(identity not in identities, 'duplicate_v1_alternative')
        identities.add(identity)
        ctx = canonical_context(row['context'], corpus)
        require(ctx['document_ids'] == [doc] and ctx['observation']['immutable_claim'] == claim.claim,
                'original_claim_or_document_changed')
        alt = claim.evidence[doc][row['alternative_index']]
        require(row['target'] == {'action': 'answer', 'documents': [{'source_id': 'c1',
                'label': LABEL_TO_PROJECT[alt.label], 'sentence_ids': [f'c1:{i}' for i in alt.sentences]}]},
                'v1_not_subset_of_original_annotations')
        component = str(row['component'])
        require(claim.claim_id not in components or components[claim.claim_id] == component, 'claim_component_conflict')
        components[claim.claim_id] = component
    require(len(set(components.values())) == len(claims), 'fit_component_duplicate')
    for i, claim in claims.items():
        require(claim.label_scope in {'SUPPORTS', 'REFUTES'}, 'no_new_fit_nei')
        require(families['component_partition'][components[i]] == 'fit', 'fit_partition')
        for doc in set(claim.evidence) | set(claim.cited_doc_ids):
            family = str(families['document_family'][str(doc)])
            require(families['family_component'][family] == components[i], 'original_document_family_component')
    return components


def histogram(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(v) for v in values).items()))


def prepare(source_git: str, source_archive: Path, source_sha: str) -> dict[str, Any]:
    started = time.perf_counter()
    source = Path(__file__).resolve().parents[1]
    verify_source_tree(source, source_archive, source_sha, source_git)
    fit_ids, old_rows, families, corpus = narrow_inputs(ROOT/OLD)
    archive = ROOT/'envs/scifact-semantic-inputs-c4907db6.tar'
    checked(archive, INPUT_SHA)
    with tarfile.open(archive) as bundle:
        raw_manifest = member(bundle, PREP+'/preparation-manifest.json').read()
        require(sha(raw_manifest) == PREP_SHA, 'original_manifest_sha')
        original = json.loads(raw_manifest)
        train_sha = original['output_file_sha256']['gold/claims_train.jsonl']
        claims = select_complete_fit(member(bundle, PREP+'/gold/claims_train.jsonl'), train_sha, fit_ids, corpus)
    components = validate_old_rows(old_rows, claims, families, corpus)
    def in_fit(doc: int) -> bool:
        family = str(families['document_family'][str(doc)])
        owner = families['family_component'].get(family)
        return owner is not None and families['component_partition'][owner] == 'fit'
    pool = {i: d for i, d in corpus.items() if in_fit(i)}
    require(bool(pool) and all(d in pool for c in claims.values() for d in c.evidence), 'fit_family_pool')
    token_dir = ROOT/'posthoc/scifact-read-continuation-fc4ffd61a761/tokenizer'
    for name, digest in TOKENIZER_SHA.items():
        checked(token_dir/name, digest)
    require('torch' not in sys.modules, 'no_torch_in_cpu_preparation')
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(token_dir, local_files_only=True, trust_remote_code=False)
    retrieve = SciFactBM25(pool)
    output = ROOT/'posthoc'/('scifact-state-supervision-fit-v2-'+source_git[:12])
    output.mkdir(mode=0o700)
    records, reports, frames = [], [], []
    for i in fit_ids:
        require(time.perf_counter() - started < 580, 'cpu_deadline_before_next_claim')
        built = build_claim(claims[i], components[i], retrieve(claims[i].claim, 20), corpus, tokenizer)
        records.extend(built['records'])
        reports.append(built['report'])
        frames.append({'claim_id': i, 'frames': built['captured_frames']})
    require(len(records) <= 192 and 'torch' not in sys.modules, 'record_or_runtime_bound')
    # Persist physical observation/schema key order, not merely canonical hashes.
    ordered_write(output/'fit-records.json', records)
    ordered_write(output/'captured-frames.json', frames)
    write_once(output/'claim-reports.json', reports)
    write_once(output/'complete-fit-gold.json', [claims[i].gold_row() for i in fit_ids])
    restored = json.loads((output/'fit-records.json').read_bytes())
    for row in restored:
        candidates = candidates_from_rows(row['candidates'], corpus)
        validate_tokenized(row, tokenize_target(tokenizer, row['frame'], row['target'], candidates))
    training_ready = all(r['training_ready'] for r in reports)
    if training_ready:
        validate_weights(restored, fit_ids)
    old_counts = Counter(r['claim_id'] for r in old_rows)
    targets = [r['target'] for r in records if r['target']['action'] == 'answer']
    docs = [d for target in targets for d in target['documents']]
    sentences = [s for doc in docs for s in doc['sentence_ids']]
    import resource
    compact = {
        'version': VERSION, 'source_git': source_git, 'source_archive_sha256': source_sha,
        'config': CONFIG, 'config_sha256': config_sha(),
        'original_manifest_sha256': PREP_SHA, 'input_archive_sha256': INPUT_SHA,
        'train_member_sha256': train_sha, 'old_manifest_sha256': MANIFEST_SHA,
        'allowlisted_file_sha256': FILES, 'tokenizer_sha256': TOKENIZER_SHA,
        'fit_ids_sha256': sha(encoded(fit_ids)), 'fit_components_sha256': sha(encoded(components)),
        'fit_pool_doc_ids_sha256': sha(encoded(sorted(pool))),
        'corpus_documents': len(corpus), 'fit_pool_documents': len(pool),
        'claims': len(claims), 'components': len(set(components.values())), 'decision_records': len(records),
        'old_records': len(old_rows), 'old_records_per_claim': histogram(old_counts.values()),
        'old_rationale_lengths': histogram(len(r['target']['documents'][0]['sentence_ids']) for r in old_rows),
        'claims_with_more_original_alternatives_than_old_records': sum(
            sum(len(a) for a in c.evidence.values()) > old_counts[i] for i, c in claims.items()),
        'official_documents_per_claim': histogram(len(c.evidence) for c in claims.values()),
        'official_alternatives_per_claim': histogram(sum(len(a) for a in c.evidence.values()) for c in claims.values()),
        'records_per_claim': histogram(r['retained_records'] for r in reports),
        'retained_claim_masses': histogram(f"{r['retained_weight_numerator']}/{r['retained_weight_denominator']}" for r in reports),
        'action_counts': histogram(r['target']['action'] for r in records),
        'answer_document_counts': histogram(len(t['documents']) for t in targets),
        'rationale_lengths': histogram(len(d['sentence_ids']) for d in docs),
        'supervised_input_tokens_sum': sum(r['packing']['input_tokens'] for r in records),
        'supervised_target_tokens_sum': sum(r['packing']['target_tokens'] for r in records),
        'supervised_input_tokens_max': max((r['packing']['input_tokens'] for r in records), default=0),
        'supervised_target_tokens_max': max((r['packing']['target_tokens'] for r in records), default=0),
        'old_input_tokens_sum': sum(r['packing']['input_tokens'] for r in old_rows),
        'old_target_tokens_sum': sum(r['packing']['target_tokens'] for r in old_rows),
        'cited_original_sentence_positions': histogram(int(s.split(':')[1]) for s in sentences),
        'cited_retrieval_ranks': histogram(int(d['source_id'][1:]) + 1 for d in docs),
        'initial_complete_witness_claims': sum(r['initial_complete'] for r in reports),
        'post_read_complete_witness_claims': sum(r['post_read_complete'] is True for r in reports),
        'whole_gold_covered_by_target_claims': sum(r['whole_gold_covered_by_target'] for r in reports),
        'initial_complete_documents_per_claim': histogram(r['initial_complete_document_count'] for r in reports),
        'final_complete_documents_per_claim': histogram(r['final_complete_document_count'] for r in reports),
        'available_target_combinations_per_claim': histogram(r['available_complete_target_combinations'] for r in reports),
        'claims_with_alternative_cap': sum(r['target_alternatives_capped'] for r in reports),
        'gap_categories': histogram(g for r in reports for g in r['gaps']),
        'claims_with_gaps': sum(bool(r['gaps']) for r in reports),
        'program_teacher_reads': sum(r['program_teacher_reads'] for r in reports),
        'scripted_responses': sum(r['scripted_responses'] for r in reports),
        'model_calls': 0, 'model_tool_executions': 0, 'optimizer_steps_executed': 0,
        'training_data_ready': training_ready, 'training_authorized': False,
        'planned_claim_group_steps_if_released': 12 if training_ready else None,
        'old_record_steps': 24, 'optimization_trajectory_equivalence_claimed': False,
        'non_fit_bytes_streamed_not_deserialized_or_persisted': True,
        'non_fit_gold_parsed': False, 'new_claims': 0, 'tune_validation_dev_test_gold_parsed': False,
        'boundary': 'program_teacher_supervision_preparation_not_model_quality_or_Agent_success',
        'elapsed_seconds': time.perf_counter() - started,
        'maxrss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'cpu_seconds': resource.getrusage(resource.RUSAGE_SELF).ru_utime + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
        'private_file_sha256': {p.name: sha(p.read_bytes()) for p in output.iterdir() if p.is_file()},
    }
    write_once(output/'compact.json', compact)
    print(json.dumps(compact, sort_keys=True))
    return compact


def main() -> None:
    import resource
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-git', required=True)
    parser.add_argument('--source-archive', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    require(os.name == 'posix' and os.environ.get('PYTHONHASHSEED') == '0'
            and os.environ.get('USE_TORCH') == '0', 'posix_cpu_tokenizer_only')
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    signal.alarm(600)
    prepare(args.source_git, args.source_archive, args.source_sha)


if __name__ == '__main__':
    main()
