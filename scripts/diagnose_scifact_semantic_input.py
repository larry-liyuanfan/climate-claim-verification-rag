"""CPU-only, selected consumed TRAIN24 diagnosis; no generation or score rewrite."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import tarfile
from typing import Any

from audit_scifact_bounded_closeout import strict_whole_answer
from climate_rag.scifact_evidence_commit import ARMS, CommitState, ISOLATED_PROTOCOL, PROTOCOL, render_prompt
from climate_rag.scifact_evidence_commit_runtime import audit_episode
from climate_rag.scifact_fit_selection import select_complete_fit
from climate_rag.scifact_grounding import GoldClaim, LABEL_TO_PROJECT, parse_abstract
from climate_rag.scifact_natural_contract import require
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_scoring import ClaimPrediction, parse_prediction
from climate_rag.scifact_semantic_contract import CORPUS_SHA, TOKENIZER_SHA, checked, sha
from prepare_scifact_state_supervision import member
from prepare_scifact_utility8 import ARCHIVE, ARCHIVE_SHA, PREP, TRAIN_SHA
from run_scifact_grounding_train_operator import ROOT
import scifact_evidence_input as inputs

RUN = ROOT / 'runs/scifact-evidence-commit-prospective24-v1-20261002-confirmation-v1'
QUALITY_SHA = '247fcdd764d5de4d39469786c1ef8b770e87a29fcf03033ab90fe47ed8d29eae'
RELEASE_SHA = 'db794da36ea061b5c0f537dd83829c7bd69e6763663410f407c6f5a5f72123ad'
RELEASE = ROOT / 'envs/prospective24-source-da243036871f/release.authorized.json'
TOKENIZER = ROOT / 'posthoc/scifact-read-continuation-fc4ffd61a761/tokenizer'


def invariant_probe(frame: Any, tokenizer: Any, *, protocol: str = ISOLATED_PROTOCOL) -> dict[str, Any]:
    """Exercise actual renderer/tokenizer, never generate or infer a verdict."""
    counts = []
    comparisons = 0
    for doc in frame['document_order']:
        expected = None
        for arm in ARMS:
            for calls, tools in ((0, 1), (1, 1), (3, 2), (4, 4)):
                state = CommitState(1, arm, frame, 'synthetic-invariance', protocol=protocol)
                # Counters are probe inputs, not a simulated physical receipt.
                # Include real-shaped synthetic feedback, unrelated to scientific gold.
                state.accept({'status':'failed', 'physical_attempt_id':'synthetic',
                              'failure':'TimeoutError', 'usage':None}, doc, '0'*64)
                state.calls, state.tools = calls, tools
                obs, schema = state.inputs('verify', doc, [])
                prompt = render_prompt(tokenizer, obs, schema)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                value = (obs, schema, prompt, ids)
                if expected is None:
                    expected = value
                    counts.append(len(ids))
                require(value == expected, 'semantic_input_not_invariant')
                require('remaining' not in obs and 'verification_feedback' not in obs,
                        'control_state_leaked')
                comparisons += 1
        clean = CommitState(2, 'adaptive', frame, 'different-episode', protocol=protocol)
        obs, schema = clean.inputs('verify', doc, [])
        prompt = render_prompt(tokenizer, obs, schema)
        require((obs, schema, prompt, tokenizer.encode(prompt, add_special_tokens=False)) == expected,
                'semantic_input_history_or_episode_leaked')
    old = CommitState(1, 'adaptive', frame, 'legacy')
    old_obs, old_schema = old.inputs('verify', frame['document_order'][0], [])
    old.calls = 1
    changed, _ = old.inputs('verify', frame['document_order'][0], [])
    require(render_prompt(tokenizer, old_obs, old_schema) != render_prompt(tokenizer, changed, old_schema),
            'legacy_behavior_must_remain_reproducible')
    planner = CommitState(1, 'adaptive', frame, 'planner', protocol=protocol)
    p0, _ = planner.inputs('plan', None, planner.order)
    planner.calls, planner.tools = 3, 3
    p1, _ = planner.inputs('plan', None, [])
    require(p0['remaining'] == {'physical_generations':5, 'tools':4}
            and p1['remaining'] == {'physical_generations':2, 'tools':2}, 'planner_budget_hidden')
    return {'documents':len(counts), 'cross_state_comparisons':comparisons,
            'maximum_verify_tokens':max(counts), 'all_equal':True,
            'legacy_control_sensitivity_preserved':True, 'planner_real_budget_preserved':True}


def document_match(label: Any, indices: list[int], rats: Any) -> tuple[bool, bool, bool]:
    label_ok = label == LABEL_TO_PROJECT[rats[0].label]
    first3 = any(set(r.sentences) <= set(indices[:3]) for r in rats)
    annotated = {i for r in rats for i in r.sentences}
    strict = label_ok and first3 and set(indices) <= annotated
    return label_ok, first3, strict


def diagnose_claim(gold: GoldClaim, frame: Any, rows: Any, candidates: list[str]) -> dict[str, Any]:
    visible: dict[int, set[int]] = {}
    for sid, value in frame['visible'].items():
        visible.setdefault(int(value['source_id']), set()).add(int(sid.rsplit(':', 1)[1]))
    retrieved = {int(d) for d in candidates}
    scope: Counter[str] = Counter()
    for doc, rats in gold.evidence.items():
        scope['gold_documents'] += 1
        scope['not_in_initial_top20'] += doc not in retrieved
        scope['retrieved_but_document_not_visible'] += doc in retrieved and doc not in visible
        scope['visible_document_missing_complete_rationale'] += doc in visible and not any(
            set(r.sentences) <= visible[doc] for r in rats)
        scope['first3_scoring_unreachable'] += not any(len(r.sentences) <= 3 for r in rats)
    arms = {}
    for arm in ARMS:
        row = rows[arm]
        require(row['claim_id'] == gold.claim_id and row['arm'] == arm and row['protocol'] == PROTOCOL,
                'existing_slot_identity')
        counters: Counter[str] = Counter()
        judgments = {}
        attempted = set()
        for feedback in row['verification_feedback']:
            doc = int(feedback['provenance']['original_source_id'])
            attempted.add(doc)
            if feedback['status'] == 'valid':
                j = feedback['judgment']
                require(doc not in judgments, 'repeated_document_judgment')
                judgments[doc] = (j['label'], [int(s.rsplit(':', 1)[1]) for s in j['sentence_ids']])
            else:
                counters['failed_verification'] += 1
        final = row['prediction']['evidence'] if row['state'] == 'valid_terminal' else {}
        correct_refs = []
        for doc, rats in gold.evidence.items():
            reachable = any(len(r.sentences) <= 3 and set(r.sentences) <= visible.get(doc, set()) for r in rats)
            counters['visible_reachable_gold_not_attempted'] += reachable and doc not in attempted
            counters['visible_reachable_gold_without_valid_verdict'] += reachable and doc not in judgments
            if doc not in judgments:
                continue
            label, indices = judgments[doc]
            label_ok, first3, strict = document_match(label, indices, rats)
            counters['visited_annotated_documents'] += 1
            counters['visited_label_disagreement'] += not label_ok
            counters['visited_label_insufficient'] += label == 'INSUFFICIENT'
            counters['visited_first3_rationale_missing'] += not first3
            counters['visited_full_selected_rationale_missing'] += not any(set(r.sentences) <= set(indices) for r in rats)
            counters['visited_selected_unannotated_sentences'] += not set(indices) <= {i for r in rats for i in r.sentences}
            counters['strict_correct_annotated_verdicts'] += strict
            if strict:
                correct_refs.append(indices)
                p = final.get(str(doc), {})
                _, _, kept = document_match(LABEL_TO_PROJECT.get(p.get('label')), p.get('sentences', []), rats)
                counters['correct_verdict_not_preserved_in_final'] += not kept
        counters['positive_verdict_on_unannotated_document'] = sum(
            d not in gold.evidence and label != 'INSUFFICIENT' for d, (label, _) in judgments.items())
        counters['final_unannotated_documents'] = sum(int(d) not in gold.evidence for d in final)
        arms[arm] = {'counts':dict(counters), 'terminal_state':row['state'],
            'all_gold_strict_verdict_subset_available': bool(gold.evidence) and len(correct_refs) == len(gold.evidence)
                and len(correct_refs) <= 5 and sum(map(len, correct_refs)) <= 20}
    return {'positive':bool(gold.evidence), 'coverage':dict(scope), 'arms':arms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(os.name == 'posix' and bool(os.environ.get('SLURM_JOB_ID'))
            and not os.environ.get('CUDA_VISIBLE_DEVICES'), 'allocated_cpu_only')
    quality = json.loads(checked(RUN/'quality.json', QUALITY_SHA))
    release = json.loads(checked(RELEASE, RELEASE_SHA))
    complete = json.loads(checked(RUN/'complete.json', '359e176f2a525c7f2a5a4ee8892671fd2e5a9044ee4c79c62f62730a65157959'))
    require(complete['status'] == 'scored' and complete['quality_sha256'] == QUALITY_SHA, 'completed_existing_run')
    _, claims = inputs.check_prepared(RUN/'prepared', release)
    ids = [c['id'] for c in claims]
    for name, digest in TOKENIZER_SHA.items():
        checked(TOKENIZER/name, digest)
    import transformers
    tokenizer = getattr(transformers, 'AutoTokenizer').from_pretrained(TOKENIZER, local_files_only=True)
    corpus = {d.doc_id:d for d in (parse_abstract(json.loads(line)) for line in
        checked(RUN/'prepared/inference/corpus.jsonl', CORPUS_SHA).splitlines())}
    frames = inputs.load_frames(RUN/'prepared', claims, corpus, tokenizer, release)
    probes = [invariant_probe(frame, tokenizer) for frame in frames]
    # The authenticated frame preserves aliases for all initial BM25 candidates,
    # including preview-only documents. Do not trust an unpinned timing sidecar.
    retrieval = [list(f['alias_to_source'].values()) for f in frames]
    require(all(len(r) == len(set(r)) == 20 for r in retrieval), 'authenticated_initial_top20')
    audited = []
    for claim_id, frame in zip(ids, frames, strict=True):
        rows = {}
        for arm in ARMS:
            directory = RUN/f'inference/{claim_id}-{arm}'
            row = json.loads((directory/'result.json').read_bytes())
            rows[arm] = audit_episode(row, frame, RUN/'inference/ledger', directory/'private-responses',
                                      tokenizer, corpus, run_identity=RELEASE_SHA, protocol=PROTOCOL)
        audited.append(rows)
    # Only selected consumed TRAIN IDs deserialized as gold; no dev/test read.
    checked(ROOT/'envs'/ARCHIVE, ARCHIVE_SHA)
    with tarfile.open(ROOT/'envs'/ARCHIVE) as bundle:
        with member(bundle, PREP+'/gold/claims_train.jsonl') as stream:
            golds = select_complete_fit(stream, TRAIN_SHA, ids, corpus)
    cases = []
    for index, (claim_id, frame, retrieved, rows) in enumerate(zip(ids, frames, retrieval, audited, strict=True)):
        case = diagnose_claim(golds[claim_id], frame, rows, retrieved)
        case['case'] = f'case_{index+1:02d}'
        for arm in ARMS:
            scored = quality['arms'][arm]['cases'][index]
            prediction = parse_prediction(rows[arm]['prediction'], corpus) if rows[arm]['state'] == 'valid_terminal' else ClaimPrediction(claim_id, {})
            correct = rows[arm]['state'] == 'valid_terminal' and strict_whole_answer(prediction, golds[claim_id])
            require(scored['claim_id'] == claim_id and correct == scored['strict_whole_answer'], 'old_score_not_rewritten')
            case['arms'][arm]['strict_whole_answer'] = correct
        case['top1_adaptive_common_error'] = case['positive'] and not any(
            case['arms'][a]['strict_whole_answer'] for a in ('fixed_top1', 'adaptive'))
        cases.append(case)
    report = {'protocol':ISOLATED_PROTOCOL, 'old_job':'31956320', 'old_quality_sha256':QUALITY_SHA,
        'script_sha256':sha(Path(__file__).read_bytes()),
        'source_git':(Path(__file__).resolve().parents[1]/'SOURCE_REVISION').read_text().strip(),
        'selection_sha256':release['selection_sha256'], 'frames_sha256':release['frames_sha256'],
        'retrieval_source':'all20_aliases_in_authenticated_initial_frames', 'tokenizer_sha256':TOKENIZER_SHA,
        'physical_replay_episodes':24*len(ARMS),
        'invariance_probes':probes, 'cases':cases, 'selected_count':24,
        'positive_count':sum(c['positive'] for c in cases),
        'common_positive_errors':sum(c['top1_adaptive_common_error'] for c in cases),
        'model_calls':0, 'training':False, 'protected_split_read':False, 'new_sampling':False,
        'scope':inputs.SCOPE, 'raw_gold_exported':False,
        'limits':['Annotation mismatch is not proof unannotated text is semantically wrong.',
                  'Counter overlap is intentional; no counterfactual model recovery is inferred.',
                  'CPU proves input invariance, not improved outputs; legacy decoding config/seed remain unknown.']}
    ordered_write(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases', 'invariance_probes')}))


if __name__ == '__main__':
    main()
