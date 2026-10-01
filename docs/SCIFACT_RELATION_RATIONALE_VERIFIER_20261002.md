# Single-generation relation/rationale verifier (CPU implementation)

## Hypothesis and scope

The [consumed TRAIN24 diagnosis](SCIFACT_SEMANTIC_INPUT_ISOLATION_20261002.md)
found available evidence but label disagreements and over-selected rationale.
This package changes the verifier's semantic interface, not retrieval, planner
policy, model weights, evaluation scoring or call limits. It does not yet show
better scientific judgments or autonomous Agent value.

The versioned candidate is `scifact-evidence-commit-relation-v3-20261002`.
The control `scifact-evidence-commit-semantic-v2-20261002` remains implemented and
unchanged; v1 and job 31956320 remain historical evidence, not a substitute for
a measured v2 control. Current authorization is CPU-only: no new model calls,
training, sampling, gold reading or protected-split evaluation.

## What one response now represents

| Field | Meaning / constraint |
|---|---|
| `source_id` | One original visible document, unchanged source binding |
| `relation` | SUPPORTS / REFUTES / INSUFFICIENT for the complete claim |
| `qualifiers` | Entity/population, conditions and comparison-context alignment |
| `direct_sentence_ids` | Up to eight original sentences in the joint evidential chain |
| `background_sentence_ids` | Up to four dispensable topic/context sentences, disjoint from direct evidence |
| `minimal_sentence_ids` | Model-selected ordered sufficient subset of direct evidence; up to eight, no automatic trimming |
| `uncertainty` | None, missing direct evidence, incomparable scope, conflict or unclear relation |

Scope alignment concerns **comparability**, not agreement with the claimed
outcome. A different number, direction or asserted event time within a comparable
population/experiment can directly REFUTE the claim. An unrelated population or
incomparable measurement is not automatically a refutation. Necessary antecedent,
experimental-condition and qualifying sentences belong in the joint rationale;
they need not independently entail the claim. Only dispensable context is background.

For SUPPORTS/REFUTES, all material comparison axes must be established or not
applicable, uncertainty is `none`, and minimal evidence is nonempty. INSUFFICIENT
has an empty minimal list and a typed uncertainty reason; its partial direct or
background evidence can remain visible as fallible feedback. Inconsistent objects,
cross-document citations, duplicate IDs and contradictory role lists are rejected
as charged failures, never silently rewritten to INSUFFICIENT.

These are model assertions, not a formal entailment checker. Structural validation
cannot prove truth, qualifier correctness, evidential necessity or minimality.
No chain-of-thought/free scientific explanation is requested or treated as evidence.

## End-to-end integration

Verifier inputs retain v2's semantic allowlist, with no remaining-budget/history
leak. Only the v3 verifier uses the richer schema and prompt. Planner instructions,
allowed actions and next-call policy remain unchanged; the planner sees additional
assessment feedback, not a new tool or a new source.

The physical response is parsed and kept intact as `assessment`. An explicit
mechanical projection maps `relation` → old `label` and `minimal_sentence_ids` →
old `sentence_ids`, preserving every selected ID and its order. The existing
immutable registry, source/claim/frame hashes and original scorer consume only
that projection. The full assessment remains in feedback beside the compatible
old `judgment`; its physical-response hash participates in verifier identity.
The auditor rebuilds both from the recorded raw response. Editing a saved
projection or assessment cannot bypass the physical replay.

Unknown versions fail closed. Release/packager/worker/journal/scorer carry the
selected version. Preparation and execution identities remain separate; v3 uses
its own attempt/output directory. Draft releases are not executable permission.
Output cap remains **512 tokens**, prompt cap **8,192**, episode timeout **120 s**;
top1/all/adaptive call caps remain **1/4/5** and the run ceiling remains **240**.

## Independent controller limit retained

With five calls, adaptive can execute only
`plan → verify → plan → verify → commit`: at most **two documents**. `case_24`
needs three annotated documents for the unchanged complete strict answer, so it
remains structurally unreachable even with perfect single-document judgments.
This is a separate controller limitation, not a verifier failure. It is tested,
not repaired here; the case stays in every denominator. Fixed top1's one-document
limit likewise remains unchanged. Do not claim this package solves multi-evidence
control or compare partial-document success with complete-answer correctness.

## Review and CPU validation

Integrated review traces input projection → grammar → raw parse → original-score
projection → registry → feedback → physical replay → release/package → preflight.
It explicitly checks numeric contradiction versus incomparable scope and necessary
joint context. Local affected suites: **111 passed**, including old v1/v2 paths,
v3 three-arm execution, relation/scope failure, unknown cost, deadline, physical
audit tampering, source identity, no first-three trimming and the two-document cap.
Three real inherited-generate/LMFE integration tests use prescribed synthetic tokens,
not model weights or scientific quality labels. Ruff and affected strict type
checks pass.

The pinned real tokenizer's synthetic five-document probe passed 732 prompt
probes across v2/v3, 60 arm/counter-state checks (including reference checks), plus
five clean-history checks. Maximum input was 2,351 tokens for v2 and 2,546 for v3;
maximum synthetic response was 80 tokens including EOS. These small synthetic
sentences are not a substitute for the real 24-frame preflight.

`preflight_scifact_relation_verifier.py` checks both protocols on the existing
24 frames and every ordered visible document pair, with positive/INSUFFICIENT/
failed synthetic feedback combinations. It checks maximum-cardinality response
envelopes and refuses input/output overflow rather than truncating or repacking.
This covers explicit probe configurations, **not every possible legal generated
object**. Actual runtime prompt/output/deadline guards still apply to all calls.
The CPU wrapper binds exact source archive, extracted tree and wrapper hash;
it loads tokenizer assets only and refuses login-node execution.

Execution-source identity, real preflight receipt and clean-archive results will
be appended after the single authorized CPU run; no pending result is claimed here.

## Falsifiable same-contract comparison proposal (not run authorization)

1. Freeze candidate source/model/tokenizer/prompt/schema/scorer/data identities.
   Use the same consumed 24 claims, the same first visible document per claim and
   the same original frame for both isolated v2 and structured v3. Do not sample
   by error type or remove `case_24`/unanswerable claims. No old v1 output is used
   as if it were a measured v2 output.
2. First compare **fixed-top1 v2 versus fixed-top1 v3**: 24 × 2 = **48** physical
   generations, one per episode, within the unchanged 240 ceiling. Same model,
   deterministic decode settings, source visibility and 512-token output limit;
   lock the complete effective generation configuration and seed. Pair order is
   predetermined, not chosen after seeing outputs. This isolates the verifier
   from controller behavior and does not constitute an Agent comparison.
3. Keep all failures and costs. Report paired wins/losses on original strict
   positive whole-answer correctness, all four original official evidence F1
   measures, label errors, rationale misses/over-selection, and fixed unresolved
   outputs separately from correct NEI. Report every physical call, input/output
   token, latency and unknown-cost item. Rationale annotation mismatch is not
   independently adjudicated semantic falsehood.
4. The hypothesis passes this diagnostic only if v3 has a positive net strict
   positive recovery, does not lower official abstract-rationalized F1, and has
   complete schema/source/cost accounting. Otherwise retain the negative result;
   do not rerun, trim using gold, move thresholds, hide unrecoverable cases or
   call formatting success a scientific gain. Cost increases remain visible even
   when the quality criterion passes.
5. This is already consumed TRAIN development evidence, not an independent test,
   generalization proof or resume-ready quality improvement. A later all/adaptive
   comparison is a separate authorization and must retain the two-document limit.

The proposed paired 48-slot execution roster needs a separately reviewed release
and runner binding before any real model call. The currently implemented generic
three-arm operator is still 72 slots / 240 calls per protocol; do not launch it
twice and pretend the combined ceiling is unchanged. No GPU run is authorized
by this document or by the CPU-ready candidate.

## Reproduction entry points

```bash
python -m pytest -q tests/test_scifact_relation_verifier.py tests/test_scifact_semantic_input.py tests/test_scifact_evidence_commit.py tests/test_scifact_evidence_commit_entry.py tests/test_scifact_document_verifier.py
python -m ruff check src scripts tests
python -m mypy src/climate_rag
bash -n hpc/scifact_relation_verifier_cpu.sbatch
python scripts/package_scifact_evidence_commit.py --help
python scripts/preflight_scifact_relation_verifier.py --help
```

Select v3 explicitly using the packager's `--protocol`; old defaults remain v1.
The frozen prospective input compact and accepted runtime receipts are required.
Raw scientific data and physical responses remain on Spartan. Shared career
materials/current resume and Trip/Energy/FLARE are outside this package.
