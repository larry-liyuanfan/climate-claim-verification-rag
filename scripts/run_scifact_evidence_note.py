"""Reconstruct all old12 fixed-rerank frames; at most note+terminal, no gold."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, cast

from climate_rag.scifact_adapter_regression import checkpoint_metadata, load_frozen, require
from climate_rag.scifact_bounded_runtime import ARMS, run_bounded_slot
from climate_rag.scifact_evidence_note import EvidenceNoteProvider, run_frame
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_semantic_contract import MODEL_SHA, TOKENIZER_SHA, checked, encoded, sha
from climate_rag.scifact_read_continuation import ordered_write as write_once
from climate_rag.scifact_state_supervision import CaptureTeacher
from climate_rag.scifact_terminal import PROTOCOL, render_answer, render_scifact_prompt, source_from_abstract, to_original_prediction
from audit_scifact_bounded_closeout import visible_view
from run_scifact_bounded_arm import verify_manifest
from run_scifact_grounding_candidate import restore_causal_adapter


def capture_frame(claim: dict[str, Any], metadata: dict[str, Any], row: dict[str, Any],
                  corpus: Any, tokenizer: Any) -> dict[str, Any]:
    """Replay recorded ORDER only, never rerun retrieval/reranker/model decisions."""
    require(claim['id'] == metadata['claim_id'] == row['claim_id'] and row['route'] == 'fixed_rerank', 'old_claim_matrix')
    candidates = [source_from_abstract(corpus[i]) for i in metadata['candidate_doc_ids']]
    aliases = {f'c{i}': source for i, source in enumerate(candidates)}
    events = row['result']['events']
    require(len(events) == 2 and [e['tool'] for e in events] == ['retrieve', 'rerank']
            and all(e['status'] == 'completed' and not e['model_selected'] for e in events), 'recorded_fixed_events')
    require(events[0]['candidate_ids'] == list(aliases), 'old_candidate_order')
    for event in events:
        require(event['source_sha256'] == {a: aliases[a].text_sha256 for a in event['candidate_ids']}, 'old_source_hashes')
    order = events[1]['candidate_ids']
    require(set(order) == set(aliases) and len(order) == len(aliases), 'old_rerank_order')
    teacher = CaptureTeacher(tokenizer)
    run_bounded_slot(claim['id'], claim['claim'], 'fixed_rerank', ARMS[0], teacher,
        lambda q, width: candidates, lambda q, rows: [aliases[a] for a in order], corpus)
    require(len(teacher.frames) == 1, 'single_scripted_capture')
    frame = teacher.frames[0]
    attempt = row['result']['generation_attempts'][0]
    diagnostic = attempt['diagnostics']
    visible, _ = visible_view(row['result']['visible_attempts'][0], attempt, corpus,
        {a: int(s.source_id) for a, s in aliases.items()}, events[1]['source_sha256'])
    require(frame['observation']['current_citable'] == [
        {'sentence_id': sid, 'text': v['text']} for sid, v in visible.items()], 'same_ordered_original_sentences')
    frame.update(visible=visible, alias_to_source={a: s.source_id for a, s in aliases.items()},
                 prompt_tokens=teacher.count_prompt(frame['observation'], frame['schema']))
    prompt = render_scifact_prompt(tokenizer, frame['observation'], frame['schema'])
    require(sha(prompt.encode()) == diagnostic['actual_prompt_sha256']
            and sha(encoded(frame['schema'])) == diagnostic['actual_schema_sha256']
            and frame['prompt_tokens'] == attempt['usage']['input_tokens']
            and len(tokenizer.encode(prompt, add_special_tokens=False)) == frame['prompt_tokens'], 'original_actual_prompt')
    return frame


def prediction(claim_id: int, outcome: dict[str, Any], frame: dict[str, Any], corpus: Any) -> dict[str, Any]:
    decision = outcome['decision']
    answer = render_answer(decision, frame['visible']) if decision and decision['action'] == 'answer' else None
    reason = ('ids_validated_semantics_unmeasured' if answer else
        'model_abstention:' + decision['reason'] if decision else 'controller_failure:' + outcome['status'])
    return dict(to_original_prediction(claim_id, {'protocol': PROTOCOL, 'answer': answer, 'outcome': reason}, corpus))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inference-dir', 'model-dir', 'model-manifest', 'output', 'release'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--release-sha', required=True)
    args = parser.parse_args()
    from run_scifact_evidence_note_operator import CONTROL, STATES, validate_release
    source = Path(__file__).resolve().parents[1]
    release = json.loads(checked(args.release, args.release_sha))
    validate_release(release, source)
    require(os.name == 'posix' and bool(os.environ.get('SLURM_JOB_ID'))
            and bool(os.environ.get('CUDA_VISIBLE_DEVICES')), 'allocated_gpu_only')
    require((source / 'SOURCE_REVISION').read_text().strip() == release['source_git'], 'source_identity')
    require(str(args.output) == release['output'] + '/inference' and not args.output.exists(), 'unused_output')
    claims, corpus_bytes = load_frozen(args.inference_dir)
    corpus = {d.doc_id: d for d in (parse_abstract(json.loads(line)) for line in corpus_bytes.splitlines())}
    control = json.loads(checked(CONTROL / 'inference/run.json', release['control']['inference/run.json']))
    rows = [r for r in control['runs'] if r['route'] == 'fixed_rerank']
    metadata = json.loads(checked(STATES, release['states_sha256']))
    require(len(claims) == len(rows) == len(metadata) == 12
            and [c['id'] for c in claims] == [r['claim_id'] for r in rows] == [m['claim_id'] for m in metadata], 'all12_fixed_order')
    binding = release['checkpoint_binding']
    adapter = Path(binding['directory'])
    checkpoint_metadata(adapter, binding)
    manifest = cast(Callable[[Path, Path, str], dict[str, str]], verify_manifest)(args.model_manifest, args.model_dir, MODEL_SHA)
    for name, digest in TOKENIZER_SHA.items():
        checked(args.model_dir / name, digest)
    args.output.mkdir(mode=0o700)
    identity = {'source_git': release['source_git'], 'release_sha256': args.release_sha,
        'policy_sha256': sha(encoded(release['policy'])), 'gold_loaded': False}
    write_once(args.output / 'consumed.json', identity | {'claims': 12, 'planned_stage_slots': 24})
    private = args.output / 'load-private'
    private.mkdir(mode=0o700)
    provider = EvidenceNoteProvider(args.model_dir, manifest, private_dir=private, gap=False)
    provider.base.model, integrity = restore_causal_adapter(provider.base.model, adapter / 'final')
    require(integrity['tensor_count'] == 144 and integrity['all_checkpoint_values_equal'], 'exact_adapter_reload')
    integrity['active_state'] = provider.bind(release['policy'])
    write_once(args.output / 'adapter-integrity.json', integrity | identity)
    # All frames must pass equality before ANY physical generation is reserved.
    frames = [capture_frame(c, m, r, corpus, provider.base.tokenizer)
              for c, m, r in zip(claims, metadata, rows, strict=True)]
    write_once(args.output / 'frames.json', frames)
    results = []
    for i, (claim, frame) in enumerate(zip(claims, frames, strict=True), 1):
        result = run_frame(provider, frame, args.output / f'case-{i:02d}', identity)
        results.append({'claim_id': claim['id'], **result, **prediction(claim['id'], result, frame, corpus)})
    write_once(args.output / 'run.json', identity | {'results': results, 'claims': 12, 'planned_stage_slots': 24,
        'frames_sha256': sha((args.output / 'frames.json').read_bytes()),
        'adapter_integrity_sha256': sha((args.output / 'adapter-integrity.json').read_bytes()),
        'new_baseline_calls': 0, 'retrieval_or_reranker_calls': 0, 'warmups': 0, 'retries': 0})


if __name__ == '__main__':
    main()
