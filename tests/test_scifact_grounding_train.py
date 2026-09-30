from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_scifact_grounding_candidate as entry
import run_scifact_grounding_train_operator as operator


def test_draft_release_is_never_execution_authority(tmp_path: Path) -> None:
    path = tmp_path / 'draft.json'
    path.write_text(json.dumps({'authorization': 'DRAFT_NOT_AUTHORIZED'}))
    args = SimpleNamespace(release=path, release_sha=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match='draft_not_authorized'):
        entry.release_check(args, 'train')


def test_runtime_file_tampering_fails_before_model_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    deps = tmp_path / 'envs/deps'
    site = deps / 'site'
    site.mkdir(parents=True)
    (site / 'package.py').write_bytes(b'verified')
    manifest = deps / 'runtime-files.json'
    manifest.write_text(json.dumps({'envs/deps/site/package.py': operator.sha(site / 'package.py')}))
    info = {'status': 'imports_verified', 'stderr_empty': True, 'dependency_errors': [],
            'runtime_files_sha256': operator.sha(manifest)}
    receipt = deps / 'final-receipt.json'
    receipt.write_text(json.dumps(info))
    release = {'runtime_receipt_sha256': operator.sha(receipt), 'runtime_files_sha256': operator.sha(manifest)}
    monkeypatch.setattr(operator, 'ROOT', tmp_path)
    monkeypatch.setattr(operator, 'DEPS', deps)
    monkeypatch.setattr(operator, 'SITES', [site])
    operator.runtime_check(release)
    (site / 'package.py').write_bytes(b'changed')
    with pytest.raises(ValueError, match='runtime_files_changed'):
        operator.runtime_check(release)


def test_nonfinite_gradient_and_final_adapter_are_rejected() -> None:
    import torch
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, float('nan'))
    # The actual entrypoint uses this explicit rejecting flag, not the default.
    assert 'error_if_nonfinite=True' in inspect.getsource(entry.train)
    with pytest.raises(RuntimeError, match='non-finite'):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    with torch.no_grad():
        model.weight.fill_(float('nan'))
    with pytest.raises(ValueError, match='nonfinite_final_adapter'):
        entry.check_finite_adapter(model)
    with torch.no_grad():
        model.weight.fill_(1)
    entry.check_finite_adapter(model)


def test_single_epoch_contains_pilot_once_without_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import torch
    import peft
    import climate_rag.local_agent_model as local
    model = torch.nn.Linear(1, 1, bias=False)
    model.config = SimpleNamespace(use_cache=True)
    model.device = 'cpu'
    model.gradient_checkpointing_enable = lambda **kwargs: None
    model.enable_input_require_grads = lambda: None
    model.save_pretrained = lambda path, **kwargs: (path.mkdir(), (path / 'adapter.safetensors').write_bytes(b'fixture'))
    seen = []
    def forward(**values):
        seen.append(values['input_ids'].item())
        return SimpleNamespace(loss=(model.weight ** 2).mean())
    model.forward = forward
    rows = [{'provenance': 'official_annotated_document_alternative', 'weight': 1.0,
             'context': i, 'target': {}, 'packing': {}} for i in range(96)]
    monkeypatch.setattr(entry, 'release_check', lambda *args: {})
    monkeypatch.setattr(entry, 'load_bundle', lambda *args: ({'files': {'fit/records.json': 'fixture'}}, {}))
    monkeypatch.setattr(entry, 'checked', lambda *args: json.dumps(rows).encode())
    monkeypatch.setattr(entry, 'model_manifest', lambda *args: {})
    monkeypatch.setattr(entry, 'canonical_context', lambda ctx, corpus: ctx)
    monkeypatch.setattr(entry, 'token_record', lambda tokenizer, ctx, target: {
        'input_ids': [ctx], 'labels': [ctx], 'attention_mask': [1]})
    monkeypatch.setattr(local, 'LocalQwenDecisionProvider', lambda *args: SimpleNamespace(model=model, tokenizer=None))
    monkeypatch.setattr(peft, 'get_peft_model', lambda base, config: base)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(name='CPU fixture', total_memory=0))
    monkeypatch.setitem(sys.modules, 'resource', SimpleNamespace(RUSAGE_SELF=0,
        getrusage=lambda _: SimpleNamespace(ru_maxrss=0)))
    args = SimpleNamespace(bundle=tmp_path, data_sha='fixture', output=tmp_path / 'output',
                           model=tmp_path, model_manifest=tmp_path)
    entry.train(args)
    pilot = json.loads((args.output / 'first-step-pilot.json').read_bytes())
    complete = json.loads((args.output / 'complete.json').read_bytes())
    assert pilot['optimizer_steps'] == 1 and pilot['records_seen'] == 4
    assert pilot['remaining_optimizer_steps'] == 23 and pilot['optimizer_reinitialized'] is False
    assert len(seen) == len(set(seen)) == 96 and sorted(seen) == list(range(96))
    assert complete['optimizer_steps'] == 24 and complete['records_seen_once'] == 96
    assert complete['checkpoint_policy'] == 'final_only'
