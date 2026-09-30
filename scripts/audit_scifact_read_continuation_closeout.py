"""Three consumed cases only; physical audit precedes already-consumed gold."""
from collections import Counter
import json
from pathlib import Path
import resource

from transformers.models.auto.tokenization_auto import AutoTokenizer

from audit_scifact_bounded_closeout import audit_slot, members, physical, strict_whole_answer, tariff
from climate_rag.scifact_adapter_regression import ADAPTER_SHA, INFERENCE_SHA, SCORING_MANIFEST, SCORING_SHA
from climate_rag.scifact_grounding import parse_abstract, parse_gold
from climate_rag.scifact_read_continuation import OLD_STATES_SHA, finalise, validate_state, visible_contract
from climate_rag.scifact_scoring import parse_prediction, score_original
from climate_rag.scifact_semantic_contract import CORPUS_SHA, INFERENCE_NAMES, MODEL_SHA, SCORING_NAMES, TOKENIZER_SHA, checked, encoded, sha
from climate_rag.scifact_train_diagnostic import document_opportunities
from run_scifact_read_continuation_operator import validate_release

ROOT = Path('/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2')
resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
STAGE = ROOT / 'envs/read-continuation-source-fc4ffd61a761'
OUT = ROOT / 'runs/scifact-read-conditional-20261001-v1'
OLD = ROOT / 'runs/scifact-adapter-bare-regression-20261001-v1'
release_sha = '6638761114eb51b98b5f8db95121f13ae05c873fdd2c69e9abb5896382ea8d69'
release = json.loads(checked(STAGE / 'release-authorized.json', release_sha))
validate_release(release, STAGE / 'source')
assert sha((STAGE / 'source.tar').read_bytes()) == release['source_archive_sha256']
assert sha((STAGE / 'wrapper.sbatch').read_bytes()) == release['wrapper_sha256']
score = json.loads(checked(OUT / 'score.json', '1dea7d5dcec2bbd2d766d4b378f5ff3f887879202f5ca2d2c7470dfc403ecb1b'))
run = json.loads(checked(OUT / 'inference/run.json', 'b7ea2e1c50e3bdc8a965477284e205007f668c9039806403bc53df4e00e71281'))
proof = json.loads((OUT / 'allocation/worker-exit.json').read_bytes())
ended = json.loads((OUT / 'allocation/ended.json').read_bytes())
assert proof['child_reaped'] and proof['returncode'] == 0 and not proof['parent_wait_interrupted']
assert proof['release_sha256'] == release_sha and proof['run_sha256'] == sha((OUT / 'inference/run.json').read_bytes())
assert ended['score_returncode'] == ended['inference_returncode'] == 0
assert proof['ended_unix'] <= (OUT / 'allocation/worker-exit.json').stat().st_mtime
assert (OUT / 'allocation/worker-exit.json').stat().st_mtime <= (OUT / 'cost-before-quality.json').stat().st_mtime <= (OUT / 'score.json').stat().st_mtime
index = json.loads((OUT / 'cost-before-quality.json').read_bytes())
assert index['worker_exit_sha256'] == sha((OUT / 'allocation/worker-exit.json').read_bytes())
assert {k: v for k, v in index.items() if k not in ('worker_exit_sha256', 'physical_sha256')} == score['costs']
for name, digest in index['physical_sha256'].items():
    assert sha((OUT / 'inference' / name).read_bytes()) == digest
assert set(index['physical_sha256']) == {p.relative_to(OUT / 'inference').as_posix() for p in (OUT / 'inference').rglob('*') if p.is_file()}
records = run['rows']
assert len(records) == 3
assert all(value == score['costs'][key] for key, value in tariff([r['attempt'] for r in records]).items())
state = {'active_adapters': ['default'], 'lora_layers': 72, 'enabled': True, 'merged': False, 'trainable_parameters': 0}
integrity = json.loads(checked(OUT / 'inference/adapter-integrity.json', run['adapter_integrity_sha256']))
assert integrity['active_state'] == state and integrity['tensor_count'] == 144 and integrity['all_checkpoint_values_equal']
bundle_path = Path(release['conditional_inputs'])
bundle = json.loads(checked(bundle_path, release['conditional_inputs_sha256']))['rows']
assert [r['claim_id'] for r in records] == [r['claim_id'] for r in bundle]
membership = json.loads(checked(bundle_path.parent / 'private-membership.json', release['private_membership_sha256']))['fit_overlap']
assert len(membership) == 3 and sum(membership) == 1
prior_states_path = ROOT / 'posthoc/scifact-selected-read-opportunity-dcbe368c1c02/private/states-before-gold.json'
prior_states = {r['claim_id']: r for r in json.loads(checked(prior_states_path, OLD_STATES_SHA))}
data = members(ROOT / 'envs/scifact-semantic-bundles-99cd9ff/inference.tar', INFERENCE_SHA, set(INFERENCE_NAMES))
assert sha(data['corpus.jsonl']) == CORPUS_SHA
corpus = {d.doc_id: d for d in (parse_abstract(json.loads(line)) for line in data['corpus.jsonl'].splitlines())}
for name, digest in TOKENIZER_SHA.items():
    checked(bundle_path.parent / 'tokenizer' / name, digest)
tokenizer = AutoTokenizer.from_pretrained(bundle_path.parent / 'tokenizer', local_files_only=True, trust_remote_code=False)
new_raw = []
for n, (row, saved) in enumerate(zip(bundle, records, strict=True), 1):
    validate_state(row, tokenizer)
    assert row['expected_state'] == prior_states[row['claim_id']]['states'][1]
    assert saved == json.loads((OUT / f'inference/slot-{n:02d}.json').read_bytes())
    reservation = json.loads((OUT / f'inference/slot-{n:02d}-reserved.json').read_bytes())
    assert reservation['identity'] == saved['identity'] and reservation['call_cap'] == 1
    d = saved['attempt']['diagnostics']
    assert d['base_model_sha256'] == MODEL_SHA and d['adapter_sha256'] == ADAPTER_SHA
    assert d['adapter_state'] == state and d['regression_policy_sha256'] == run['provider_binding_sha256']
    assert d['actual_prompt_sha256'] == row['expected_state']['prompt_sha256']
    assert d['actual_schema_sha256'] == row['expected_state']['schema_sha256']
    assert saved['attempt']['usage']['input_tokens'] == row['expected_state']['prompt_tokens']
    audit, raw = physical(OUT / f'inference/private-responses/slot-{n:02d}', [saved['attempt']], False)
    assert audit == score['wire_audits'][n - 1]
    assert finalise(raw[d['output_sha256']], row, corpus) == saved['terminal']
    new_raw.append(raw)
old_run = json.loads(checked(OLD / 'inference/run.json', '7ebffe6edf98e1dbbf255dee5e47461f884a43c4562d54d3a85b1e737d9daade'))
old_score = json.loads(checked(OLD / 'score.json', '1d75e03a26f7a0fd107a7d082895497726f21a71aee3dd448f8342235e6c84b7'))
old_rows, old_raw = [], []
for row in bundle:
    matches = [(i, r) for i, r in enumerate(old_run['runs'], 1) if r['claim_id'] == row['claim_id'] and r['route'] == 'adaptive']
    assert len(matches) == 1
    n, old = matches[0]
    attempts = old['result']['generation_attempts']
    assert len(attempts) == 1 and not old['result']['model_tool_links']
    d = attempts[0]['diagnostics']
    assert d['actual_prompt_sha256'] == prior_states[row['claim_id']]['states'][0]['prompt_sha256']
    assert d['base_model_sha256'] == MODEL_SHA and d['adapter_sha256'] == ADAPTER_SHA and d['adapter_state'] == state
    audit, raw = physical(OLD / f'inference/private-responses/slot-{n:02d}', attempts, False)
    assert audit == old_score['wire_audits'][n - 1]
    old_rows.append(old)
    old_raw.append(raw)
# Only after cost, exit, raw, prompt/schema, adapter and old-row checks: consumed gold.
gold_bytes = members(ROOT / 'envs/scifact-semantic-bundles-99cd9ff/scoring.tar', SCORING_SHA, set(SCORING_NAMES))
assert sha(gold_bytes['manifest.json']) == SCORING_MANIFEST
manifest = json.loads(gold_bytes['manifest.json'])
assert all(sha(gold_bytes[name]) == digest for name, digest in manifest['scoring_file_sha256'].items())
ids = {r['claim_id'] for r in bundle}
gold = {r['id']: parse_gold(r, corpus) for line in gold_bytes['gold.jsonl'].splitlines() if (r := json.loads(line))['id'] in ids}
assert set(gold) == ids
predictions = [parse_prediction(r['terminal']['prediction'], corpus) for r in records]
old_predictions = [parse_prediction(r['prediction'], corpus) for r in old_rows]
ordered_gold = [gold[r['claim_id']] for r in bundle]
assert score_original(ordered_gold, predictions) == score['quality']
cases = []
for n, (row, saved, p, old, old_p, raw, fit) in enumerate(zip(bundle, records, predictions, old_rows, old_predictions, old_raw, membership, strict=True), 1):
    g = gold[row['claim_id']]
    audit_slot(old, g, corpus, raw, gap=False)
    shown = {}
    for entry in visible_contract(row, corpus).values():
        shown.setdefault(int(entry['source_id']), []).append(entry['sentence_index'])
    before = {int(k): v for k, v in prior_states[g.claim_id]['states'][0]['visible'].items()}
    assert shown == {int(k): v for k, v in prior_states[g.claim_id]['states'][1]['visible'].items()}
    initial_opportunity = document_opportunities(g, before)
    conditional_opportunity = document_opportunities(g, shown)
    assert not initial_opportunity and conditional_opportunity
    assert len(p.evidence) == len(g.evidence) == 1
    doc_id, predicted = next(iter(p.evidence.items()))
    alternatives = g.evidence.get(doc_id, ())
    correct_doc = bool(alternatives)
    correct_label = correct_doc and predicted.label == alternatives[0].label
    union = {i for alt in alternatives for i in alt.sentences}
    first3 = set(predicted.sentences[:3])
    complete = any(set(alt.sentences) <= first3 for alt in alternatives)
    overlap = set(predicted.sentences) & union
    mechanism = ('wrong_document' if not correct_doc else 'wrong_document_label' if not correct_label
                 else 'complete_alternative' if complete else 'no_annotated_rationale_sentence_selected' if not overlap
                 else 'partial_alternative_missing_required_sentence')
    cases.append({'anonymous_case': f'case-{n}', 'fit_overlap': fit,
        'initial_complete_rationale_reachable': bool(initial_opportunity),
        'post_scripted_read_complete_rationale_reachable': bool(conditional_opportunity),
        'predicted_document_is_scripted_read_document': doc_id in [row['candidate_doc_ids'][int(a[1:])] for a in row['frozen_read_ids']],
        'correct_document': correct_doc, 'correct_document_label': correct_label,
        'predicted_sentences': len(predicted.sentences), 'complete_rationale': bool(correct_label and complete),
        'gold_sentence_overlap_count': len(overlap), 'minimum_complete_alternative_length': min(len(a.sentences) for a in alternatives),
        'minimum_missing_sentences_for_an_alternative': min(len(set(a.sentences) - set(predicted.sentences)) for a in alternatives),
        'mechanism': mechanism, 'first_three_truncation_explains_failure': False,
        'strict_whole_answer': strict_whole_answer(p, g),
        'original_adaptive_correct_label': score_original([g], [old_p])['metrics']['abstract_label_only']['correct'],
        'original_adaptive_complete_rationale': score_original([g], [old_p])['metrics']['abstract_rationalized']['correct'],
        'model_tool_executions': 0, 'scripted_read_is_not_model_behavior': True})
groups = {}
for flag in (True, False):
    name = 'fit_overlap' if flag else 'no_direct_fit_overlap'
    indices = [i for i, m in enumerate(membership) if m is flag]
    groups[name] = {'conditional': score_original([ordered_gold[i] for i in indices], [predictions[i] for i in indices]),
        'original_adaptive': score_original([ordered_gold[i] for i in indices], [old_predictions[i] for i in indices])}
    assert groups[name]['conditional'] == score['fit_subgroups'][name]
compact = {'job_id': '31781591', 'status': 'completed_limited_conditional_grounding_not_autonomous_success',
    'source_git': release['source_git'], 'release_sha256': release_sha,
    'hashes': {name: sha((OUT / name).read_bytes()) for name in ('score.json', 'inference/run.json', 'inference/adapter-integrity.json', 'cost-before-quality.json', 'allocation/worker-exit.json', 'allocation/ended.json')},
    'audit_script_sha256': sha(Path(__file__).read_bytes()),
    'physical_index_files_rehashed': len(index['physical_sha256']), 'conditional_raw_audits': 3, 'original_adaptive_raw_audits': 3,
    'actual_prompt_schema_and_token_count_reverified': True, 'active_adapter': state, 'checkpoint_tensors_verified': 144,
    'costs': score['costs'], 'original_adaptive_subset_costs': tariff([r['result']['generation_attempts'][0] for r in old_rows]),
    'conditional_quality': score['quality'], 'original_adaptive_quality': score_original(ordered_gold, old_predictions),
    'fit_subgroups': groups, 'anonymous_mechanisms': cases, 'mechanism_counts': dict(Counter(c['mechanism'] for c in cases)),
    'post_read_complete_first3_rationale_reachable': 3, 'initial_complete_rationale_reachable': 0,
    'worker_exit_before_cost_before_quality': True, 'this_audit_model_calls': 0,
    'scored_only_previously_consumed_claims': 3, 'new_gold_or_validation_read': False, 'old_artifacts_modified': False,
    'slurm': {'state': 'COMPLETED', 'exit_code': '0:0', 'elapsed_seconds': 76, 'total_cpu_seconds': 64.313, 'maxrss_kib': 17164576},
    'gpu_allocation_seconds_not_kernel_time': 76, 'independent_test': False, 'validation_gate': False,
    'boundary': 'Oracle/scripted read conditional answering, not retrieval capability, autonomous Agent gain or heldout performance.',
    'followup_decision': 'limited_conditional_capacity_but_grounding_weak; CPU_state_supervision_proposal_only_no_new_training'}
destination = ROOT / 'posthoc/scifact-read-conditional-closeout-31781591'
destination.mkdir(mode=0o700)
with (destination / 'compact.json').open('xb') as stream:
    stream.write(encoded(compact) + b'\n')
print(json.dumps({'compact_sha256': sha((destination / 'compact.json').read_bytes()),
    'physical_files': compact['physical_index_files_rehashed'], 'cases': cases,
    'old_adaptive': compact['original_adaptive_quality']['metrics'],
    'conditional': compact['conditional_quality']['metrics'], 'costs': compact['costs']}, sort_keys=True))
