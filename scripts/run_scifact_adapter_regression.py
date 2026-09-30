"""One old-TRAIN four-route regression. No gold import in the inference process."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, cast

from climate_rag.scifact_adapter_regression import (
    ActiveAdapterProvider, PROTOCOL, active_state, checkpoint_metadata, load_frozen, policy_identity, require,
)
from climate_rag.scifact_bounded_runtime import ARMS, run_bounded_slot
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_retrieval import SciFactBM25, rerank_sources
from climate_rag.scifact_semantic_contract import MODEL_SHA, RERANKER_SHA, ROUTES, TOKENIZER_SHA, checked, encoded, sha, write_once
from climate_rag.rerank import Qwen3CausalLMReranker
from run_scifact_bounded_arm import verify_manifest
from run_scifact_grounding_candidate import restore_causal_adapter


def execute_slots(claims: list[dict[str, Any]], provider: Any, retrieve: Any, rerank: Any,
                  corpus: Any, output: Path, identity: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (claim, route) in enumerate(((c, r) for c in claims for r in ROUTES), 1):
        require(active_state(provider.base.model)['enabled'], 'adapter_before_slot')
        raw_path = output / f'slot-{index:02d}-raw.json'
        write_once(output / f'slot-{index:02d}-reserved.json',
            {'claim_id': claim['id'], 'route': route, 'identity': identity, 'started_unix': time.time()})
        provider.start_slot(output / 'private-responses' / f'slot-{index:02d}')

        def persist(value: dict[str, Any]) -> None:
            value.update(regression_protocol=PROTOCOL, regression_identity=identity)
            write_once(raw_path, value)

        row = run_bounded_slot(claim['id'], claim['claim'], route, ARMS[0], provider,
                               retrieve, rerank, corpus, persist_raw=persist)
        row['regression_identity'] = identity
        write_once(output / f'slot-{index:02d}.json', row)
        rows.append(row)
        require(row.get('export_error') is None and not row['result']['trace_unlinked_events'],
                'export_or_event_link_failure')
        calls = row['result']['generation_attempts']
        require(len(calls) <= (5 if route == 'adaptive' else 3), 'route_call_bound')
        for attempt in calls:
            receipt = attempt.get('diagnostics', {}).get('private_attachment', {})
            require(attempt['usage_known'] and receipt.get('truncated') is False
                    and receipt.get('io_failed') is False
                    and receipt.get('stored_bytes') == receipt.get('attempted_bytes'),
                    'incomplete_wire_or_unknown_usage_stop')
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inference-dir', 'model-dir', 'model-manifest', 'reranker-dir', 'reranker-manifest',
                 'adapter', 'output', 'release'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--release-sha', required=True)
    args = parser.parse_args()
    from run_scifact_adapter_regression_operator import validate_release
    source = Path(__file__).resolve().parents[1]
    release = json.loads(checked(args.release, args.release_sha))
    validate_release(release, source)
    require(os.name == 'posix' and bool(os.environ.get('SLURM_JOB_ID'))
            and bool(os.environ.get('CUDA_VISIBLE_DEVICES')), 'allocated_gpu_only')
    require((source / 'SOURCE_REVISION').read_text().strip() == release['source_git'], 'frozen_source')
    require(not args.output.exists() and str(args.output) == release['output'] + '/inference', 'fresh_fixed_inference')
    claims, corpus_bytes = load_frozen(args.inference_dir)
    abstracts = [parse_abstract(json.loads(line)) for line in corpus_bytes.splitlines()]
    corpus = {d.doc_id: d for d in abstracts}
    require(len(corpus) == len(abstracts) == 5183, 'frozen_corpus_shape')
    verifier = cast(Callable[[Path, Path, str], dict[str, str]], verify_manifest)
    manifest = verifier(args.model_manifest, args.model_dir, MODEL_SHA)
    verifier(args.reranker_manifest, args.reranker_dir, RERANKER_SHA)
    for name, digest in TOKENIZER_SHA.items():
        checked(args.model_dir / name, digest)
    checkpoint_metadata(args.adapter)
    args.output.mkdir(mode=0o700)
    policy = policy_identity(source)
    identity = {'source_git': release['source_git'], 'source_archive_sha256': release['source_archive_sha256'],
        'release_sha256': args.release_sha, 'policy': policy, 'policy_sha256': sha(encoded(policy)),
        'inference_archive_sha256': release['inference_archive_sha256'], 'gold_loaded': False}
    write_once(args.output / 'consumed.json', identity | {'planned_slots': 48, 'all_slots_consumed': False})
    (args.output / 'private-responses').mkdir(mode=0o700)
    began = time.monotonic()
    provider = ActiveAdapterProvider(args.model_dir, manifest, private_dir=args.output / 'private-responses', gap=False)
    provider.base.model, integrity = restore_causal_adapter(provider.base.model, args.adapter / 'final')
    require(integrity['tensor_count'] == 144 and integrity['all_checkpoint_values_equal'], 'fixed_tensor_restore')
    integrity['active_state'] = provider.bind(policy)
    write_once(args.output / 'adapter-integrity.json', integrity | identity)
    model_ms = (time.monotonic() - began) * 1000
    began = time.monotonic()
    reranker = Qwen3CausalLMReranker(str(args.reranker_dir), device='cuda', dtype='bfloat16', max_length=2048, batch_size=1)
    rerank_ms = (time.monotonic() - began) * 1000
    retrieve = SciFactBM25(corpus)
    rows = execute_slots(claims, provider, retrieve, lambda q, c: rerank_sources(reranker, q, c), corpus, args.output, identity)
    write_once(args.output / 'run.json', identity | {'runs': rows, 'planned_slots': 48,
        'status': 'inference_completed_not_quality_acceptance', 'model_load_ms': model_ms,
        'reranker_load_ms': rerank_ms, 'warmup_or_preflight_calls': 0,
        'adapter_integrity_sha256': sha((args.output / 'adapter-integrity.json').read_bytes())})


if __name__ == '__main__':
    main()
