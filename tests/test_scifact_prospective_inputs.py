"""Synthetic whole consumer chain; no real query/gold/model or scheduler."""
import hashlib
import io
import json
import tarfile

import pytest

from climate_rag import scifact_prospective_inputs as module
from climate_rag.scifact_evidence_commit import ARMS, CALL_CAPS, PROTOCOL
from climate_rag.scifact_evidence_commit_runtime import CommitJournal, run_episode
from climate_rag.scifact_grounding import Abstract
from climate_rag.scifact_natural_contract import base_state
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_retrieval import SciFactBM25
from climate_rag.scifact_semantic_contract import encoded, sha, write_once
import scifact_evidence_input as adapter
import run_scifact_evidence_commit_operator as operator
import score_scifact_document_verifier as shared
import score_scifact_evidence_commit as scoring
from test_scifact_evidence_commit import Backend, RUN, choose


def selection(monkeypatch):
    pool = [{'id':i, 'component':f'fixture-{i % 64:02}'} for i in range(1, 98)]
    monkeypatch.setattr(module, 'POOL_COMPONENTS', (64, sha(encoded(sorted({r['component'] for r in pool})))))
    monkeypatch.setattr(module, 'POOL_IDS', (97, sha(encoded(sorted(r['id'] for r in pool)))))
    selected = module.select_metadata(pool, {'fixture-metadata':'f'*64})
    assert module.select_metadata(list(reversed(pool)), {'fixture-metadata':'f'*64}) == selected
    return selected, pool


def prepared_fixture(out, monkeypatch):
    selected, _ = selection(monkeypatch)
    hashes = module.freeze_selection(out, selected)
    corpus = {i:Abstract(i, 'Fixture', ('Fixture claim sentence.', 'Other sentence.'), False) for i in range(100, 120)}
    tokenizer = Backend([]).base.tokenizer
    claims = [{'id':r['id'], 'claim':'Fixture claim sentence.'} for r in selected['selected']]
    retrieve = SciFactBM25(corpus)
    pairs = [module.prepare_frame(c, retrieve, tokenizer) for c in claims]
    frames = [p[0] for p in pairs]
    module.validate_frames(claims, frames, corpus, tokenizer)
    (out/'inference').mkdir()
    ordered_write(out/'inference/claims.json', claims)
    ordered_write(out/'inference/initial-frames.json', frames)
    raw = b''.join((json.dumps({'doc_id':d.doc_id,'title':d.title,'abstract':list(d.sentences),'structured':False})+'\n').encode() for d in corpus.values())
    (out/'inference/corpus.jsonl').write_bytes(raw)
    monkeypatch.setattr(adapter, 'CORPUS_SHA', sha(raw))
    monkeypatch.setattr(shared, 'CORPUS_SHA', sha(raw))
    receipt = {'protocol':module.PROTOCOL,'scope':module.SCOPE,
        'selection_sha256':hashes['selection.json'], 'component_reservations_sha256':hashes['component-reservations.json'],
        'claims_sha256':sha((out/'inference/claims.json').read_bytes()),
        'frames_sha256':sha((out/'inference/initial-frames.json').read_bytes()),
        'ordered_ids_sha256':selected['ordered_ids_sha256'],'model_calls':0,
        'official_gold_decoded':False,'protected_split_read':False,'reselection':False,
        'shared_preparation_cost':{'initial_retrieval_calls':24, 'shared_among_arms':3, 'meaning':'synthetic_timing_only'}}
    write_once(out/'preparation.json', receipt)
    release = dict(adapter.frozen_fields(operator.FROZEN_FIELDS), **adapter.release_inputs(out),
        authorization='coordinator_exact_hash_release', source_git='a'*40,
        source_archive_sha256='b'*64, wrapper_sha256='c'*64)
    operator.validate_release(release)
    return release, claims, frames, corpus, tokenizer, pairs


def test_once_metadata_only_selection_rejects_drift_and_replacement(tmp_path, monkeypatch):
    selected, pool = selection(monkeypatch)
    assert len(selected['selected']) == len({r['component'] for r in selected['selected']}) == 24
    with pytest.raises(ValueError, match='projection_only'):
        module.select_metadata([dict(pool[0], label='SUPPORTS')]+pool[1:], {})
    with pytest.raises(ValueError, match='drift'):
        module.select_metadata(pool[:-1], {})
    out = tmp_path/'frozen'
    hashes = module.freeze_selection(out, selected)
    assert module.reserved_selection(out, hashes['selection.json'], hashes['component-reservations.json']) == selected
    with pytest.raises(FileExistsError):
        module.freeze_selection(out, selected)


def test_true_initial_frame_and_release_entry_no_old_prefix(tmp_path, monkeypatch):
    out = tmp_path/'prepared'
    release, claims, frames, corpus, tokenizer, pairs = prepared_fixture(out, monkeypatch)
    assert all(p[1]['model_calls'] == 0 and p[1]['candidate_k'] == 20 and not p[1]['physical_generation_prefix_created'] for p in pairs)
    assert all(f['observation']['remaining_calls'] == 5 and f['observation']['remaining_tools'] == 4 for f in frames)
    assert all(f['document_order'] == [f'c{i}' for i in range(5)] for f in frames)
    assert not list(out.rglob('*prefix*'))
    assert adapter.load_frames(out, claims, corpus, tokenizer, release) == frames
    assert release['planned_episodes'] == 72 and release['max_generator_calls'] == 240 and release['call_caps'] == CALL_CAPS
    assert 'initial_inventory_sha256' not in release
    for changed in ({'authorization':'DRAFT_CPU_READY_NOT_AUTHORIZED'}, {'call_caps':dict(CALL_CAPS,adaptive=6)},
                    {'input_protocol':'typo'}, {'initial_inventory_sha256':'a'*64}):
        with pytest.raises(ValueError):
            operator.validate_release(dict(release, **changed))
    monkeypatch.setattr(adapter, 'PREPARED', out)
    copied = tmp_path/'copied'
    adapter.copy_prepared(copied, release)
    assert adapter.load_frames(copied, claims, corpus, tokenizer, release) == frames
    run = tmp_path/'future-run'
    monkeypatch.setattr(adapter, 'OUTPUT', run)
    monkeypatch.setattr(operator, 'verify_runtime_receipt', lambda r: {'generation_calls':0})
    release.update(prepared=out.as_posix(), output=run.as_posix())
    operator.start_attempt(release, RUN, 'synthetic_no_job')
    assert json.loads((run/'reserved.json').read_bytes())['attempt_id'] == 'prospective24-confirmation-v1'


def test_preflight_cli_uses_new_frames_not_old_prefix(tmp_path, monkeypatch):
    import sys
    import types
    import preflight_scifact_evidence_commit as preflight
    out = tmp_path/'prepared'
    release, _, _, _, tokenizer, _ = prepared_fixture(out, monkeypatch)
    write_once(tmp_path/'draft.json', release)
    monkeypatch.setattr(preflight, 'TOKENIZER_SHA', {})
    import climate_rag.scifact_semantic_contract as contract
    monkeypatch.setattr(contract, 'CORPUS_SHA', adapter.CORPUS_SHA)
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(
        from_pretrained=lambda *a, **k:tokenizer)))
    monkeypatch.setattr(sys, 'argv', ['preflight', '--prepared',str(out),'--release',str(tmp_path/'draft.json'),
        '--tokenizer',str(tmp_path/'unused'),'--output',str(tmp_path/'preflight.json')])
    preflight.main()
    report = json.loads((tmp_path/'preflight.json').read_bytes())
    assert report['claims'] == 24 and report['model_calls'] == 0 and not report['gold_read']


@pytest.mark.parametrize('fault', ['sha', 'ordered_claims', 'reservation', 'frames'])
def test_input_mismatch_fails_before_generation_or_gold(tmp_path, monkeypatch, fault):
    out = tmp_path/'prepared'
    release, claims, frames, corpus, tokenizer, _ = prepared_fixture(out, monkeypatch)
    if fault == 'sha':
        release['frames_sha256'] = '0'*64
    elif fault == 'ordered_claims':
        claims.reverse()
    elif fault == 'reservation':
        (out/'component-reservations.json').write_text('{}')
    else:
        frames[0]['document_order'].reverse()
        (out/'inference/initial-frames.json').write_text(json.dumps(frames))
    with pytest.raises(ValueError):
        adapter.load_frames(out, claims, corpus, tokenizer, release)


def test_new_input_three_arm_scoring_adapter_synthetic_only(tmp_path, monkeypatch):
    output = tmp_path/'run'
    output.mkdir()
    release, claims, frames, corpus, tokenizer, _ = prepared_fixture(output/'prepared', monkeypatch)
    ids = [c['id'] for c in claims]
    ordered_write(output/'reserved.json', {'release_sha256':RUN})
    proof = {'child_reaped':False,'returncode':0,'interrupted':None,'release_sha256':RUN}
    ordered_write(output/'worker-exit.json', proof)
    def forbidden(*args, **kwargs):
        pytest.fail('no old frame or early gold access')
    monkeypatch.setattr(shared, 'load_frames', forbidden)
    with pytest.raises(ValueError, match='exit_before'):
        scoring.score_after_exit(output, forbidden, release, release_sha=RUN)
    proof['child_reaped'] = True
    (output/'worker-exit.json').write_text(json.dumps(proof))
    # Missing all72: cost still has full denominator, zero gold reads.
    with monkeypatch.context() as patch:
        patch.setattr(shared.tarfile, 'open', forbidden)
        report = scoring.score_after_exit(output, forbidden, release, release_sha=RUN)
    assert report['planned_slots'] == 72 and report['status'] == 'no_quality'
    (output/'cost-before-gold.json').unlink()  # fixture-only second path in a fresh logical evaluation
    (output/'no-quality.json').unlink()
    actions = []
    def verdict(doc):
        return {'source_id':doc,'label':'SUPPORTS','sentence_ids':[doc+':0']}
    for _ in ids:
        actions += [verdict('c0'), *[verdict(f'c{i}') for i in range(4)],
                    {'action':'verify','source_id':'c0'}, verdict('c0'), choose]
    backend = Backend(actions)
    inference = output/'inference'
    inference.mkdir()
    ordered_write(inference/'initial-frames.json', frames)
    journal = CommitJournal(backend, inference/'ledger', max_generations=240,
                            protocol=PROTOCOL, physical_guard=base_state, run_identity=RUN)
    for claim, frame in zip(claims, frames, strict=True):
        for arm in ARMS:
            row = run_episode(claim['id'], arm, frame, journal, corpus, inference/f"{claim['id']}-{arm}", run_identity=RUN)
            assert row['initial_retrieval']['status'] == 'new_cpu_preparation_shared_three_arms'
    root = tmp_path/'synthetic-root'
    (root/'envs').mkdir(parents=True)
    archive = root/'envs'/shared.ARCHIVE
    first = frames[0]['alias_to_source']['c0']
    gold_bytes = b''.join((json.dumps(dict(c, evidence={first:[{'label':'SUPPORT','sentences':[0]}]}, cited_doc_ids=[int(first)]))+'\n').encode() for c in claims)
    info = tarfile.TarInfo(shared.PREP+'/gold/claims_train.jsonl')
    info.size = len(gold_bytes)
    with tarfile.open(archive, 'w') as bundle:
        bundle.addfile(info, io.BytesIO(gold_bytes))
    monkeypatch.setattr(shared, 'ARCHIVE_SHA', sha(archive.read_bytes()))
    monkeypatch.setattr(shared, 'TRAIN_SHA', hashlib.sha256(gold_bytes).hexdigest())
    original, audits, opened = scoring.audit_episode, [], []
    def audited(*args, **kwargs):
        audits.append(args[0]['claim_id'])
        return original(*args, **kwargs)
    original_open = tarfile.open
    def late_gold(*args, **kwargs):
        assert len(audits) == 72
        opened.append(True)
        return original_open(*args, **kwargs)
    monkeypatch.setattr(scoring, 'audit_episode', audited)
    monkeypatch.setattr(shared.tarfile, 'open', late_gold)
    report = scoring.score_after_exit(output, lambda:tokenizer, release, root, release_sha=RUN)
    assert report['status'] == 'scored' and opened == [True] and len(audits) == 72
    assert report['selection_sha256'] == release['selection_sha256'] and report['limits'] == module.SCOPE
    assert backend.calls == 192 and [report['arms'][a]['planned'] for a in ARMS] == [24]*3
    assert report['initial_retrieval_cost']['shared_among_arms'] == 3


@pytest.mark.parametrize('fault', ['missing_frame', 'wrong_frame_sha', 'selection'])
def test_complete_ledger_survives_input_failure_without_gold(tmp_path, monkeypatch, fault):
    output = tmp_path/'run'
    output.mkdir()
    release, claims, _, _, _, _ = prepared_fixture(output/'prepared', monkeypatch)
    ordered_write(output/'reserved.json', {'release_sha256':RUN})
    ordered_write(output/'worker-exit.json', {'child_reaped':True, 'returncode':0,
        'interrupted':None, 'release_sha256':RUN})
    ledger = output/'inference/ledger'
    ledger.mkdir(parents=True)
    for n, (claim, arm) in enumerate((c, a) for c in claims for a in ARMS):
        slot = f"{claim['id']}-{arm}"
        (output/'inference'/slot).mkdir()
        ordered_write(output/'inference'/slot/'result.json', {})
        ordered_write(ledger/f'g{n:02}.reserved.json', {'slot':slot})
        ordered_write(ledger/f'g{n:02}.finished.json', {'usage_known':True,'elapsed_ms':1,
            'usage':{'input_tokens':10,'output_tokens':2},
            'response':{'diagnostics':{'generation_elapsed_ms':1}}})
    if fault == 'missing_frame':
        (output/'prepared/inference/initial-frames.json').unlink()
    elif fault == 'wrong_frame_sha':
        release['frames_sha256'] = '0'*64
    else:
        (output/'prepared/selection.json').write_text('{}')
    def forbidden(*args, **kwargs):
        pytest.fail('no gold/model/tokenizer on damaged input')
    monkeypatch.setattr(shared.tarfile, 'open', forbidden)
    result = scoring.score_after_exit(output, forbidden, release, release_sha=RUN)
    assert result['status'] == 'no_quality' and result['planned_slots'] == 72 and not result['gold_read']
    cost = json.loads((output/'cost-before-gold.json').read_bytes())
    assert cost['physical_generation_cost']['unique_physical_calls'] == 72
    assert cost['physical_generation_cost']['unknown_usage_attempts'] == 0
    assert cost['initial_retrieval_cost']['status'] == 'unknown_input_binding_failed'
    assert cost['initial_retrieval_cost']['seconds'] is None
