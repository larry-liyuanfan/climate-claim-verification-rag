"""A separately released, three-call conditional probe; not an Agent episode."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, cast

from climate_rag.agent_protocol import ModelResponseValidationError
from climate_rag.scifact_adapter_regression import (
    ActiveAdapterProvider, active_state, checkpoint_metadata, load_frozen, require,
)
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_read_continuation import CALL_SECONDS, finalise, policy, validate_state
from climate_rag.scifact_semantic_contract import MODEL_SHA, TOKENIZER_SHA, checked, encoded, sha, write_once
from run_scifact_bounded_arm import verify_manifest
from run_scifact_grounding_candidate import restore_causal_adapter


def call_once(provider: Any, row: dict[str, Any], corpus: Any, output: Path,
              index: int, identity: dict[str, Any]) -> dict[str, Any]:
    validate_state(row, provider.base.tokenizer)
    require(active_state(provider.base.model)['enabled'], 'active_before_call')
    write_once(output/f'slot-{index:02d}-reserved.json', {'identity':identity,
        'claim_id':row['claim_id'], 'state_sha256':row['expected_state']['state_sha256'],
        'call_cap':1, 'new_call_seconds':CALL_SECONDS, 'reserved_unix':time.time()})
    provider.start_slot(output/f'private-responses/slot-{index:02d}')
    attempt: dict[str, Any] = {'usage':{}, 'usage_known':False, 'status':'reserved', 'diagnostics':{}}
    terminal: dict[str, Any] = {'status':'not_observed', 'prediction':None, 'autonomous_success':False}
    try:
        response = provider.generate(row['observation'], row['schema'], 512, CALL_SECONDS)
        attempt.update(usage=response['usage'], usage_known=True, diagnostics=response['diagnostics'], status='response_received')
        require(response['usage']['input_tokens'] == row['expected_state']['prompt_tokens']
                and type(response['usage']['output_tokens']) is int
                and 0 <= response['usage']['output_tokens'] <= 512, 'actual_token_budget_mismatch')
        try:
            terminal = finalise(response['raw'], row, corpus)
            attempt.update(status='schema_valid_proposal' if terminal['status'] == 'proposed_not_executed'
                           else 'valid_decision', action=terminal['decision']['action'])
        except ValueError as exc:
            attempt.update(status='validation_failed', exception_type=type(exc).__name__)
            terminal['status'] = 'invalid_no_repair'
    except ModelResponseValidationError as exc:
        attempt.update(status='provider_response_invalid', usage=exc.usage, diagnostics=exc.diagnostics,
            usage_known=all(k in exc.usage for k in ('input_tokens','output_tokens'))
            and not exc.diagnostics.get('output_usage_unknown', False))
        terminal['status'] = 'provider_failure_no_retry'
    except Exception as exc:
        attempt.update(status='exception_no_retry', exception_type=type(exc).__name__)
        terminal['status'] = 'exception_no_retry'
    record = {'identity':identity, 'claim_id':row['claim_id'], 'expected_state':row['expected_state'],
        'attempt':attempt, 'terminal':terminal, 'model_tool_executions':0,
        'scripted_read_not_model_call':True, 'new_call_seconds_not_episode_remainder':CALL_SECONDS}
    write_once(output/f'slot-{index:02d}.json', record)
    # Preserve an unknown/partial paid attempt and stop; never retry it or add repair.
    require(attempt['usage_known'] and attempt['status'] in {'valid_decision','schema_valid_proposal','validation_failed'}, 'partial_call_stop')
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('release','inference-dir','model-dir','model-manifest','adapter','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--release-sha', required=True)
    args = parser.parse_args()
    from run_scifact_read_continuation_operator import validate_release
    source = Path(__file__).resolve().parents[1]
    release = json.loads(checked(args.release, args.release_sha))
    validate_release(release, source)
    require(os.name == 'posix' and bool(os.environ.get('SLURM_JOB_ID'))
            and bool(os.environ.get('CUDA_VISIBLE_DEVICES')), 'allocated_gpu_only')
    require((source/'SOURCE_REVISION').read_text().strip() == release['source_git'], 'frozen_source')
    require(str(args.output) == release['output']+'/inference' and not args.output.exists(), 'fresh_output')
    bundle = json.loads(checked(Path(release['conditional_inputs']), release['conditional_inputs_sha256']))
    claims, raw = load_frozen(args.inference_dir)
    corpus = {d.doc_id:d for d in (parse_abstract(json.loads(line)) for line in raw.splitlines())}
    rows = bundle['rows']
    require(len(rows) == len({r['claim_id'] for r in rows}) == 3 and len(corpus) == 5183
            and set(r['claim_id'] for r in rows) <= {c['id'] for c in claims}, 'frozen_three_shape')
    verifier = cast(Callable[[Path, Path, str], dict[str, str]], verify_manifest)
    manifest = verifier(args.model_manifest, args.model_dir, MODEL_SHA)
    for name, digest in TOKENIZER_SHA.items():
        checked(args.model_dir/name, digest)
    checkpoint_metadata(args.adapter)
    args.output.mkdir(mode=0o700)
    (args.output/'private-responses').mkdir(mode=0o700)
    identity = {'source_git':release['source_git'], 'source_archive_sha256':release['source_archive_sha256'],
        'release_sha256':args.release_sha, 'conditional_inputs_sha256':release['conditional_inputs_sha256'],
        'policy':policy(source), 'policy_sha256':sha(encoded(policy(source))), 'gold_loaded':False}
    write_once(args.output/'consumed.json', identity | {'planned_slots':3})
    provider = ActiveAdapterProvider(args.model_dir, manifest, private_dir=args.output/'private-responses', gap=False)
    provider.base.model, integrity = restore_causal_adapter(provider.base.model, args.adapter/'final')
    require(integrity['tensor_count'] == 144 and integrity['all_checkpoint_values_equal'], 'tensor_restoration')
    # Provider only needs active=True; the rest is a separately bound probe policy.
    binding = identity['policy'] | {'adapter_active':True}
    integrity['active_state'] = provider.bind(binding)
    identity['provider_binding_sha256'] = sha(encoded(binding))
    write_once(args.output/'adapter-integrity.json', integrity | identity)
    for row in rows:
        validate_state(row, provider.base.tokenizer)
    results = [call_once(provider, row, corpus, args.output, i, identity) for i,row in enumerate(rows,1)]
    write_once(args.output/'run.json', identity | {'rows':results, 'planned_slots':3,
        'adapter_integrity_sha256':sha((args.output/'adapter-integrity.json').read_bytes()),
        'status':'conditional_inference_completed_not_Agent_success'})


if __name__ == '__main__':
    main()
