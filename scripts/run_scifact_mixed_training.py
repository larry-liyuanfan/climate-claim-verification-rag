"""Train-only, separately versioned release entry. No prepare/evaluate/resume.

The release and tokenized preparation do not exist yet. This implementation is
not permission to load a model, prepare real inputs or submit a GPU job.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from climate_rag.scifact_mixed_inputs import SCOPE, input_config_sha, make_plan, validate_inputs
from climate_rag.scifact_mixed_training import CONFIG, VERSION, _run_epoch, config_sha
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_semantic_contract import MODEL_SHA, checked, sha
from climate_rag.scifact_state_supervision import require

ROOT = Path('/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2')


def release_inputs(path: Path, digest: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(os.name == 'posix' and bool(os.environ.get('SLURM_JOB_ID'))
            and bool(os.environ.get('CUDA_VISIBLE_DEVICES')), 'allocated_GPU_required')
    release = json.loads(checked(path, digest))
    require(release['authorization'] == 'coordinator_exact_hash_release'
            and release['purpose'] == VERSION and release['config_sha256'] == config_sha(), 'new_exact_release_required')
    require(type(release['max_runtime_seconds']) is int and 60 <= release['max_runtime_seconds'] <= 86400,
            'separately_frozen_runtime_required')
    source = Path(__file__).resolve().parents[1]
    require((source/'SOURCE_REVISION').read_text().strip() == release['source_git'], 'mixed_source_revision')
    files = {p.relative_to(source).as_posix() for p in (source/'src').rglob('*.py')}
    files |= {'scripts/run_scifact_mixed_training.py', 'scripts/run_scifact_grounding_candidate.py',
              'scripts/run_scifact_grounding_train_operator.py'}
    require(set(release['source_files_sha256']) == files, 'complete_executable_source_manifest')
    for name in files:
        checked(source/name, release['source_files_sha256'][name])
    output = Path(release['output'])
    require(output.parent == ROOT/'runs' and output.name.startswith('scifact-mixed-training-')
            and not output.exists(), 'exclusive_mixed_output')
    bundle = Path(release['prepared_directory'])
    require(bundle.resolve().is_relative_to(ROOT/'posthoc'), 'private_prepared_directory')
    preparation = json.loads(checked(bundle/'complete.json', release['preparation_receipt_sha256']))
    require(preparation['status'] == 'complete' and preparation['scope'] == SCOPE
            and preparation['source_bridge_validated'] is True
            and preparation['input_config_sha256'] == release['input_config_sha256'] == input_config_sha(),
            'released_source_preparation_required')
    prepared = json.loads(checked(bundle/'prepared.json', release['prepared_file_sha256']))
    plan = json.loads(checked(bundle/'plan.json', release['plan_file_sha256']))
    roster = json.loads(checked(bundle/'roster.json', release['roster_file_sha256']))
    require(preparation['files'] == {'prepared.json': release['prepared_file_sha256'],
            'plan.json': release['plan_file_sha256'], 'roster.json': release['roster_file_sha256']}, 'preparation_file_binding')
    data = validate_inputs(prepared)
    require(data['scope'] == SCOPE and data['roster'] == roster and plan == make_plan(prepared)
            and plan['payload']['optimizer_steps'] == 36, 'frozen_real_144_223_36')
    for tok in data['tokenized']:
        require(0 < tok['input_tokens'] <= 8192 and 0 < tok['target_tokens'] <= 512
                and all(type(i) is int and i >= 0 for i in tok['input_ids']), 'frozen_token_limits')
    require(release['model_manifest_sha256'] == MODEL_SHA, 'fresh_frozen_base_required')
    return dict(release), prepared, plan


def train(release: dict[str, Any], prepared: dict[str, Any], plan: dict[str, Any]) -> None:
    from run_scifact_grounding_train_operator import runtime_check
    from run_scifact_grounding_candidate import model_manifest
    runtime = runtime_check(release)
    import torch
    from peft import LoraConfig, get_peft_model
    from climate_rag.local_agent_model import LocalQwenDecisionProvider
    require(torch.cuda.is_available(), 'GPU_runtime_unavailable')
    torch.manual_seed(CONFIG['seed'])
    provider = LocalQwenDecisionProvider(Path(release['model_directory']), model_manifest(Path(release['model_manifest'])))
    builder: Any = get_peft_model
    model = builder(provider.model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM', bias='none'))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    trainable = [{'name': n, 'elements': p.numel(), 'dtype': str(p.dtype)}
                 for n, p in model.named_parameters() if p.requires_grad]
    require(bool(trainable) and all('lora_' in p['name'] and any(t in p['name'] for t in CONFIG['target_modules'])
                                    for p in trainable), 'only_requested_fresh_LoRA_trainable')
    require(all(p.dtype == torch.bfloat16 for p in model.parameters() if not p.requires_grad), 'frozen_base_bfloat16')
    vocabulary_size = model.get_input_embeddings().num_embeddings
    require(all(max(t['input_ids']) < vocabulary_size for t in prepared['payload']['tokenized']), 'token_vocabulary_bound')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    def resources(started: float) -> dict[str, Any]:
        import resource
        torch.cuda.synchronize()
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {'elapsed_seconds': time.perf_counter()-started,
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'host_maxrss_kib_linux': usage.ru_maxrss, 'host_cpu_seconds': usage.ru_utime+usage.ru_stime}
    def publish(output: Path) -> dict[str, Any]:
        from safetensors.torch import load_file
        final = output/'final'
        model.save_pretrained(final, safe_serialization=True)
        expected = {'adapter_config.json', 'adapter_model.safetensors', 'README.md'}
        require({p.name for p in final.iterdir()} == expected and not any(p.is_symlink() for p in final.iterdir()),
                'final_adapter_files')
        tensors = load_file(str(final/'adapter_model.safetensors'), device='cpu')
        require(bool(tensors) and all(bool(torch.isfinite(t).all()) for t in tensors.values()), 'saved_adapter_finite')
        ordered_write(output/'training-runtime.json', {'imports': runtime, 'trainable': trainable,
            'config': CONFIG, 'resources': resources(started), 'fresh_base_and_adapter': True})
        return {p.name: sha(p.read_bytes()) for p in sorted(final.iterdir())}
    started = time.perf_counter()
    _run_epoch(prepared, plan, model, Path(release['output']), device=model.device,
               publish=publish, first_update=resources)


def claim_worker(execution: Path, release_sha: str) -> None:
    """Consume exactly one worker attempt, BEFORE imports/model loading."""
    reservation = json.loads((execution/'reserved.json').read_bytes())
    parent = os.getppid()
    require(reservation['release_sha256'] == release_sha
            and reservation['parent_pid'] == parent
            and os.environ.get('CLIMATE_MIXED_PARENT_PID') == str(parent)
            and not (execution/'worker-exit.json').exists(), 'bounded_parent_required')
    ordered_write(execution/'worker-started.json', {'release_sha256': release_sha,
        'parent_pid': parent, 'worker_pid': os.getpid(), 'automatic_retry': False})


def bounded_worker(command: list[str], execution: Path, seconds: int) -> int:
    """Any timeout, interruption or exception terminates/reaps the sole child."""
    child: subprocess.Popen[bytes] | None = None
    code: int | None = None
    interrupted: str | None = None
    launching = True
    pending_signal: int | None = None
    old_handlers: dict[signal.Signals, Any] = {}
    def stop(signum: int, frame: Any) -> None:
        nonlocal pending_signal
        if launching:
            # Popen may have created the child before returning its handle.
            # Defer raising until the caller can reliably kill/reap that PID.
            pending_signal = signum
            return
        raise InterruptedError(f'parent_signal_{signum}')
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, stop)
    try:
        with (execution/'stdout.log').open('xb') as stdout, (execution/'stderr.log').open('xb') as stderr:
            env = {**os.environ, 'CLIMATE_MIXED_PARENT_PID': str(os.getpid())}
            if pending_signal is not None:
                raise InterruptedError(f'parent_signal_{pending_signal}')
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env)
            launching = False
            if pending_signal is not None:
                raise InterruptedError(f'parent_signal_{pending_signal}')
            code = child.wait(timeout=seconds)
        return code
    except BaseException as exc:
        interrupted = type(exc).__name__
        raise
    finally:
        # Ignore repeated signals while obtaining a real termination receipt.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                code = child.wait()
            ordered_write(execution/'worker-exit.json', {'returncode': code,
                'interrupted': interrupted, 'child_reaped': child is not None and code is not None,
                'child_started': child is not None, 'automatic_retry': False})
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    release, prepared, plan = release_inputs(args.release, args.release_sha)
    execution = Path(release['output']+'-execution')
    if args.worker:
        claim_worker(execution, args.release_sha)
        train(release, prepared, plan)
        return
    execution.mkdir(mode=0o700, exist_ok=False)
    ordered_write(execution/'reserved.json', {'version': VERSION, 'release_sha256': args.release_sha,
        'parent_pid': os.getpid(), 'max_runtime_seconds': release['max_runtime_seconds'], 'automatic_retry': False})
    code = bounded_worker([sys.executable, str(Path(__file__).resolve()), '--release', str(args.release),
        '--release-sha', args.release_sha, '--worker'], execution, release['max_runtime_seconds'])
    require(code == 0, 'bounded_training_worker_failed_no_retry')


if __name__ == '__main__':
    main()
