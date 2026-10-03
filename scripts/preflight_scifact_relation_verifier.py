"""CPU token/contract probes on pinned consumed frames, never gold or generation."""
from __future__ import annotations

import argparse
import hashlib
from itertools import permutations, product
import json
import os
from pathlib import Path
from typing import Any

from climate_rag.scifact_evidence_commit import CommitState, ISOLATED_PROTOCOL, render_prompt
from climate_rag.scifact_grounding import parse_abstract
from climate_rag.scifact_natural_contract import require
from climate_rag.scifact_read_continuation import ordered_write
from climate_rag.scifact_relation_verifier import RELATION_PROTOCOL, QUALIFIERS, parse_assessment
from climate_rag.scifact_semantic_contract import CORPUS_SHA, TOKENIZER_SHA, checked
from diagnose_scifact_semantic_input import invariant_probe
from run_scifact_evidence_commit_operator import release_fields
from run_scifact_grounding_train_operator import ROOT
import scifact_evidence_input as inputs


def synthetic_assessment(frame: Any, doc: str, label: str = "REFUTES") -> dict[str, Any]:
    """Max-cardinality ID envelope, not a claim about the visible science."""
    ids = sorted((s for s in frame['visible'] if s.rsplit(':',1)[0] == doc), key=lambda s:(len(s),s), reverse=True)
    positive = label != 'INSUFFICIENT'
    return {'source_id':doc, 'relation':label,
        'qualifiers':{k:'not_applicable' if positive else 'not_established' for k in QUALIFIERS},
        'direct_sentence_ids':ids[:8], 'background_sentence_ids':ids[8:12],
        'minimal_sentence_ids':ids[:8] if positive else [],
        'uncertainty':'none' if positive else 'missing_direct_evidence'}


def probe_relation(frame: Any, tokenizer: Any) -> dict[str, Any]:
    invariant = invariant_probe(frame, tokenizer, protocol=RELATION_PROTOCOL)
    maxima: dict[str, int] = {}
    calls: dict[str, int] = {}
    max_response = 0
    for protocol in (ISOLATED_PROTOCOL, RELATION_PROTOCOL):
        counts = []

        def measure(state: CommitState, stage: str, doc: str | None, available: list[str]) -> None:
            obs, schema = state.inputs(stage, doc, available)
            counts.append(len(tokenizer.encode(render_prompt(tokenizer, obs, schema), add_special_tokens=False)))

        initial = CommitState(1,'adaptive',frame,'synthetic-preflight:'+'a'*64,protocol=protocol)
        measure(initial,'plan',None,initial.order)
        for doc in initial.order:
            measure(initial,'verify',doc,[])
            for label in ('SUPPORTS','REFUTES','INSUFFICIENT'):
                raw = synthetic_assessment(frame,doc,label)
                parse_assessment(raw,frame,doc)
                if protocol == RELATION_PROTOCOL:
                    # Include one EOS token in the same 512-token physical ceiling.
                    max_response = max(max_response, len(tokenizer.encode(
                        json.dumps(raw,separators=(',',':')),add_special_tokens=False))+1)
        # Every ordered pair of visible documents; positive, insufficient and
        # failure feedback combinations. No scientific labels/predictions.
        pairs: list[tuple[str,...]] = list(permutations(initial.order,2)) or [(initial.order[0],)]
        for docs in pairs:
            for modes in product(('positive','insufficient','failed'),repeat=len(docs)):
                state = CommitState(1,'adaptive',frame,'synthetic-preflight:'+'a'*64,protocol=protocol)
                for position,(doc,mode) in enumerate(zip(docs,modes,strict=True)):
                    raw = synthetic_assessment(frame,doc,'INSUFFICIENT' if mode == 'insufficient' else 'REFUTES')
                    decision = raw if protocol == RELATION_PROTOCOL else {
                        'source_id':doc,'label':raw['relation'],'sentence_ids':raw['minimal_sentence_ids']}
                    state.accept({'status':'failed' if mode == 'failed' else 'valid',
                        'physical_attempt_id':f'synthetic-{position}', 'decision':decision,
                        'failure':'SyntheticTimeout' if mode == 'failed' else None,
                        'usage':None},doc,'b'*64)
                    state.calls = 2*(position+1)
                    available = [d for d in state.order if d not in docs[:position+1]] if position == 0 else []
                    measure(state,'plan',None,available)
        maxima[protocol],calls[protocol] = max(counts),len(counts)
    return {'invariance':invariant, 'maximum_prompt_tokens':maxima, 'prompt_probe_counts':calls,
        'maximum_synthetic_response_tokens_with_eos':max_response,
        'prompt_overflow':any(n>8192 for n in maxima.values()), 'response_overflow':max_response>512,
        'scope':'all_ordered_visible_pairs_three_feedback_modes_max_cardinality_not_all_outputs'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    require(os.name == 'posix' and bool(os.environ.get('SLURM_JOB_ID'))
            and not os.environ.get('CUDA_VISIBLE_DEVICES'), 'allocated_cpu_only')
    prepared = ROOT/'runs/scifact-evidence-commit-prospective24-v1-20261002-confirmation-v1/prepared'
    fields = release_fields({'protocol':RELATION_PROTOCOL,'input_protocol':inputs.PROTOCOL})
    _,claims = inputs.check_prepared(prepared,fields)
    directory = ROOT/'posthoc/scifact-read-continuation-fc4ffd61a761/tokenizer'
    for name,digest in TOKENIZER_SHA.items():
        checked(directory/name,digest)
    import transformers
    tokenizer = getattr(transformers,'AutoTokenizer').from_pretrained(directory,local_files_only=True)
    corpus = {d.doc_id:d for d in (parse_abstract(json.loads(line)) for line in
        checked(prepared/'inference/corpus.jsonl',CORPUS_SHA).splitlines())}
    frames = inputs.load_frames(prepared,claims,corpus,tokenizer,fields)
    rows = [probe_relation(frame,tokenizer) for frame in frames]
    result = {'protocol':RELATION_PROTOCOL,'control_protocol':ISOLATED_PROTOCOL,
        'source_git':(Path(__file__).resolve().parents[1]/'SOURCE_REVISION').read_text().strip(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'tokenizer_sha256':TOKENIZER_SHA,'selection_sha256':fields['selection_sha256'],
        'frames_sha256':fields['frames_sha256'],'claims':len(rows),'rows':rows,
        'maximum_prompt_tokens':{p:max(r['maximum_prompt_tokens'][p] for r in rows)
            for p in (ISOLATED_PROTOCOL,RELATION_PROTOCOL)},
        'maximum_synthetic_response_tokens_with_eos':max(r['maximum_synthetic_response_tokens_with_eos'] for r in rows),
        'overflow_count':sum(r['prompt_overflow'] or r['response_overflow'] for r in rows),
        'synthetic_feedback_not_predictions':True,'model_calls':0,'training':False,
        'gold_read':False,'new_sampling':False,'protected_split_read':False,
        'scorer_changed':False,'planner_changed':False,
        'adaptive_max_verified_documents_under_five_calls':2,
        'three_gold_document_strict_case_remains_structurally_unreachable':True}
    ordered_write(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}))
    require(result['overflow_count'] == 0,'preflight_overflow_no_silent_truncation')


if __name__ == '__main__':
    main()
