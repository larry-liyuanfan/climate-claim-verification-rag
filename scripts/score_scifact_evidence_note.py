"""Physical accounting first; private old-TRAIN quality only after a reaped worker."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from climate_rag.scifact_adapter_regression import SCORING_MANIFEST, load_frozen, require
from climate_rag.scifact_evidence_note import NOTE_SCHEMA, note_input, terminal_input, validated_note
from climate_rag.scifact_grounding import parse_abstract, parse_gold
from climate_rag.scifact_read_continuation import ordered_write as write_once
from climate_rag.scifact_scoring import parse_prediction, score_original
from climate_rag.scifact_semantic_contract import MODEL_SHA, SCORING_NAMES, checked, encoded, sha
from climate_rag.scifact_terminal import parse_action
from audit_scifact_bounded_closeout import physical, strict_whole_answer, tariff
from run_scifact_evidence_note import prediction
from run_scifact_evidence_note_operator import CONTROL, CONTROL_HASHES, validate_release
from score_scifact_semantic_pair import validate_prepared_scoring


def collect_costs(output: Path, release: dict[str, Any], release_sha: str) -> dict[str, Any]:
    proof_path = output / 'allocation/worker-exit.json'
    proof = json.loads(proof_path.read_bytes())
    require(proof['release_sha256'] == release_sha, 'exit_identity')
    directory = output / 'inference'
    paths = sorted(directory.glob('case-*/*/reserved.json'))
    require(len(paths) <= 24 and all(p.parent.name in {'note', 'terminal'} for p in paths), 'stage_reservation_bound')
    records: list[dict[str, Any]] = []
    by_stage: dict[str, list[dict[str, Any]]] = {'note': [], 'terminal': []}
    wall_ms: dict[str, list[float]] = {'note': [], 'terminal': []}
    issues = []
    for path in paths:
        stage = path.parent.name
        try:
            request = json.loads(path.read_bytes())
            require(request['identity']['release_sha256'] == release_sha
                    and request['stage'] == stage and request['max_output_tokens'] == 512
                    and request['max_seconds'] == 120 and 0 < request['input_tokens'] <= 8192, 'physical_request_bound')
        except (ValueError, OSError, KeyError, TypeError):
            unresolved = {'usage': {}, 'diagnostics': {}, 'usage_known': False}
            records.append(unresolved)
            by_stage[stage].append(unresolved)
            issues.append(path.parent.relative_to(directory).as_posix())
            continue
        finished = path.parent / 'finished.json'
        try:
            value = json.loads(finished.read_bytes())
        except (OSError, ValueError):
            value = {'usage': {}, 'diagnostics': {}}
        usage, diagnostic = value.get('usage') or {}, value.get('diagnostics') or {}
        if isinstance(value.get('elapsed_ms'), (float, int)):
            wall_ms[stage].append(float(value['elapsed_ms']))
        record = {'usage': usage, 'diagnostics': diagnostic,
            'usage_known': all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('input_tokens', 'output_tokens'))
                and not diagnostic.get('output_usage_unknown', False)}
        records.append(record)
        by_stage[stage].append(record)
        if 'response' in value:
            try:
                require(diagnostic['actual_prompt_sha256'] == request['prompt_sha256']
                        and diagnostic['actual_schema_sha256'] == request['schema_sha256']
                        and diagnostic['regression_policy_sha256'] == sha(encoded(release['policy']))
                        and diagnostic['adapter_sha256'] == release['policy']['adapter_model_sha256']
                        and diagnostic['base_model_sha256'] == MODEL_SHA
                        and diagnostic['adapter_state'] == {'active_adapters': ['default'], 'lora_layers': 72,
                            'enabled': True, 'merged': False, 'trainable_parameters': 0}, 'physical_binding')
                physical(path.parent / 'private', [record], False)
                require(sha(value['response']['raw'].encode()) == diagnostic['output_sha256'], 'physical_raw_binding')
            except (ValueError, KeyError, OSError):
                issues.append(path.parent.relative_to(directory).as_posix())
        else:
            issues.append(path.parent.relative_to(directory).as_posix())
    costs = tariff(records)
    run_path = directory / 'run.json'
    run_ok = run_path.exists() and sha(run_path.read_bytes()) == proof['run_sha256']
    if run_ok:
        saved_run = json.loads(run_path.read_bytes())
        run_ok = (len(saved_run.get('results', [])) == 12
            and len({r['claim_id'] for r in saved_run['results']}) == 12
            and len(list(directory.glob('case-*/result.json'))) == 12)
    # Do not turn an unreserved stage or an unknown response into a zero-cost success.
    index = {p.relative_to(directory).as_posix(): sha(p.read_bytes()) for p in directory.rglob('*') if p.is_file()}
    return {'costs': costs, 'by_stage': {s: tariff(v) for s, v in by_stage.items()},
        'stage_wall_ms_including_response_io_known_sum': {s: sum(v) if v else None for s, v in wall_ms.items()},
        'stage_wall_ms_missing': {s: len(by_stage[s]) - len(v) for s, v in wall_ms.items()},
        'planned_claims': 12, 'planned_stage_slots': 24, 'reserved_stage_slots': len(paths),
        'unreserved_stage_slots': 24 - len(paths), 'new_control_calls': 0,
        'physical_issues': issues, 'physical_sha256': index, 'worker_exit_sha256': sha(proof_path.read_bytes()),
        'quality_ready': bool(proof['child_reaped'] and proof['returncode'] == 0 and not proof['interrupted']
            and run_ok and not costs['token_totals_are_lower_bounds'] and not issues),
        'gold_loaded': False, 'baseline_cost_excluded_from_new_calls': True}


def verify_new_rows(directory: Path, release: dict[str, Any], release_sha: str,
                    claims: list[dict[str, Any]], corpus: Any) -> list[dict[str, Any]]:
    run = json.loads((directory / 'run.json').read_bytes())
    require(run['release_sha256'] == release_sha and run['gold_loaded'] is False
            and [r['claim_id'] for r in run['results']] == [c['id'] for c in claims], 'full12_identity')
    frames = json.loads(checked(directory / 'frames.json', run['frames_sha256']))
    require(len(frames) == 12, 'all12_frames')
    integrity = json.loads(checked(directory / 'adapter-integrity.json', run['adapter_integrity_sha256']))
    require(integrity['tensor_count'] == 144 and integrity['all_checkpoint_values_equal'] is True
            and integrity['release_sha256'] == release_sha, 'adapter_reload')
    for i, (row, frame) in enumerate(zip(run['results'], frames, strict=True), 1):
        case = directory / f'case-{i:02d}'
        outcome = json.loads((case / 'result.json').read_bytes())
        require(all(row[k] == v for k, v in outcome.items()), 'case_summary_mismatch')
        request = json.loads((case / 'note/request.json').read_bytes())
        require(request['observation'] == note_input(frame['observation']) and request['schema'] == NOTE_SCHEMA, 'note_input_only')
        terminal = case / 'terminal/request.json'
        if terminal.exists():
            note_raw = json.loads((case / 'note/finished.json').read_bytes())['response']['raw']
            note = validated_note(note_raw, [])  # Physical worker separately rejects tokenizer control strings.
            final = json.loads(terminal.read_bytes())
            require(final['observation'] == terminal_input(frame['observation'], note)
                    and final['schema'] == frame['schema'] and final['note_sha256'] == sha(note.encode()), 'unchanged_stage2_context')
        else:
            require(outcome['terminal_attempted'] is False and (case / 'terminal-not-attempted.json').exists(), 'explicit_skip')
        if outcome['status'] == 'completed':
            response = json.loads((case / 'terminal/finished.json').read_bytes())['response']
            decision = parse_action(json.loads(response['raw']), frame['observation']['allowed_actions'],
                frame['visible'], list(frame['alias_to_source']), 5)
            require(decision == row['decision'], 'terminal_wire_decision')
        require(prediction(row['claim_id'], outcome, frame, corpus)['prediction'] == row['prediction'], 'canonical_prediction')
    return list(run['results'])


def score(args: Any) -> dict[str, Any]:
    source = Path(__file__).resolve().parents[1]
    release = json.loads(checked(args.release, args.release_sha))
    validate_release(release, source)
    require(args.output.as_posix() == release['output'], 'fixed_output')
    cost_path = args.output / 'cost-before-quality.json'
    costs = json.loads(cost_path.read_bytes())
    require(costs['quality_ready'] and costs == collect_costs(args.output, release, args.release_sha), 'durable_costs_before_gold')
    claims, raw_corpus = load_frozen(args.inference_dir)
    corpus = {d.doc_id: d for d in (parse_abstract(json.loads(line)) for line in raw_corpus.splitlines())}
    rows = verify_new_rows(args.output / 'inference', release, args.release_sha, claims, corpus)
    # Only after costs + physical rows pass: open original gold and old control scores.
    require({p.name for p in args.scoring_dir.iterdir()} == SCORING_NAMES, 'scoring_allowlist')
    manifest = json.loads(checked(args.scoring_dir / 'manifest.json', SCORING_MANIFEST))
    payload = {n: checked(args.scoring_dir / n, h) for n, h in manifest['scoring_file_sha256'].items()}
    validate_prepared_scoring(payload, manifest)
    gold = [parse_gold(json.loads(line), corpus) for line in payload['gold.jsonl'].splitlines()]
    require([g.claim_id for g in gold] == [r['claim_id'] for r in rows], 'gold_full12')
    for name, digest in CONTROL_HASHES.items():
        checked(CONTROL / name, digest)
    control_run = json.loads((CONTROL / 'inference/run.json').read_bytes())
    control_rows = [r for r in control_run['runs'] if r['route'] == 'fixed_rerank']
    predicted = [parse_prediction(r['prediction'], corpus) for r in rows]
    control = [parse_prediction(r['prediction'], corpus) for r in control_rows]
    pairs = [{'claim_id': g.claim_id, 'official_nei': not g.evidence,
        'new_status': r['status'], 'old_strict_correct': strict_whole_answer(c, g),
        'new_strict_correct': r['status'] == 'completed' and strict_whole_answer(p, g)}
        for g, r, p, c in zip(gold, rows, predicted, control, strict=True)]
    old_score = json.loads((CONTROL / 'score.json').read_bytes())['quality']['routes']['fixed_rerank']
    require(score_original(gold, control) == old_score['official_point_score'], 'old_control_not_rerun_or_changed')
    return {'scope': 'old12_exposed_TRAIN_not_independent_test_not_gate', 'claims': 12,
        'cost_before_quality_sha256': sha(cost_path.read_bytes()), 'costs': costs,
        'status_counts': dict(Counter(r['status'] for r in rows)), 'paired_private_diagnostics': pairs,
        'new_point_score': score_original(gold, predicted), 'old_control': old_score,
        'old_strict_correct': sum(p['old_strict_correct'] for p in pairs),
        'new_strict_correct': sum(p['new_strict_correct'] for p in pairs),
        'recovered': sum(not p['old_strict_correct'] and p['new_strict_correct'] for p in pairs),
        'regressed': sum(p['old_strict_correct'] and not p['new_strict_correct'] for p in pairs),
        'failure_empty_predictions_are_not_successful_NEI': True,
        'causal_attribution': 'two_calls_more_compute_not_pure_prompt_or_thinking_effect',
        'validation_gate': False, 'further_execution_authorized': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'inference-dir', 'scoring-dir', 'release'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--release-sha', required=True)
    args = parser.parse_args()
    write_once(args.output / 'score.json', score(args))


if __name__ == '__main__':
    main()
