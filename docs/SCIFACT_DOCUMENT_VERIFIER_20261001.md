# Single-document verification diagnostic — 2026-10-01

## Status and falsifiable question

The original CPU package was released once as job **31914601** (source
`496b584cdb93041fa53a5f8eb662fdad0c49a088`). It failed before model loading on
2026-10-01, 22:16:20–22:16:40 +10, `FAILED/1:0`: the release omitted
`python_executable`, required by the reused runtime consumer. Allocated resources
were one A100, 8 CPUs and 32 GiB; batch MaxRSS was 304436 KiB. Only preparation
and reservation files exist: **zero inference ledger entries/results, no quality
denominator**. This was a startup-contract defect, not missing Torch or POSIX.

The first infrastructure retry **r2** is CPU-only pending a new exact-hash
coordinator release. The protocol, original 24×2 matrix, model, prompts, scoring
and all budgets are unchanged. The old stage, lock and run remain preserved;
the new run is `runs/scifact-document-verifier-v1-20261001-r2`. No automatic
submission, training, validation or frozen-test evaluation is authorized here.

### Startup-contract repair

The tracked release constructor now binds both the accepted inventory receipt
and the actual successful job 31895661 runtime observation (SHA
`6153e39d11dbf44b57b311359cf2b5846dcf6a45ef70cd9069f69e4c8c68d455`).
The inventory receipt does not itself contain the interpreter field; the
observation supplies the actual executable, with matching Python, POSIX, Torch,
package versions and module locations. The validator checks every downstream
release key before reservation, and the operator calls the real runtime consumer
before preparing data or loading a model. Remote output paths are canonical POSIX
strings even when the candidate is packaged on Windows.

CPU regression covers constructor → validator → actual runtime receipt/hash
consumer → exclusive reservation; only the import observation is stubbed.
Missing keys, wrong interpreter/type/path, receipt tampering, attempt drift and
duplicate output are rejected. The failed r1 package is a retained negative case;
its previously passing validator-only fixture did not exercise the consumer.
The new producer emits a draft, not execution permission. A distinct r2 stage,
submission lock and newly approved exact hashes are required for any later run.
Affected CPU tests: **74 passed** (startup and existing verifier semantics), plus
**32 passed** source-package regressions. Ruff, strict affected-script mypy and
shell syntax passed. Prior unchanged full-suite/model-semantic results are
retained rather than rerun locally to validate an unchanged model protocol.

Hypothesis: the same unadapted Qwen3-4B can recover positive grounding by judging
the entire immutable claim against a single document's actually visible original
sentences, reducing interference in joint document/label/sentence selection.
This does not assume a verifier judgment is true, or that an Agent will use it.

The original source `723c6a8fc4f0956ba91497b278d5f8fe65dc9fee`, job 31895661,
its raw outputs and its scoring are unchanged. Its A/B/C results are only a
**compute-different historical reference**, not a matched verifier baseline.

## Frozen inputs and control variables

- All original FIT24, in original order, paired across `fixed` and `adaptive`:
  48 planned episodes, including unresolved cases. No reselection from successes.
- Selection SHA `107c50a8ecc1fd4d1b352cdc7e2f86b2997d6abc027d5afeb176bab1d09f7acb`.
- Actual-input inventory SHA (insertion-order JSON identity)
  `0c4b663184acabc0a4b0421f37f92182a5aea2ab414dcad7d3a5840e3b57c028`.
  This binds 98 original files: original run reservation, selection and each of
  24 original initial frames, sealed prefixes and first physical-call receipts.
- Before any generation, authenticate all 24 original frames against their
  actual tokenizer/prompt hashes, original physical response, immutable claim,
  corpus text, original sentence index and source SHA. New prompts have their
  own hashes; they are not represented as unchanged old prompts.
- No repacking, truncation, alias renumbering or new sentences. The initial
  retrieval artifact is replayed and consumes one logical tool debit in each
  route; its historical compute is not re-executed or claimed free online work.
- Same base model/manifest/tokenizer and no adapters, new retrieval, reranker,
  free claim decomposition, evidence notes or SFT. Existing restricted climate,
  public CLIMATE-FEVER tests and SciFact dev300 are untouched.

## Tool and routes

`verify(source_id)` receives the full immutable claim and **only** that document's
currently visible original sentences. It returns a schema-constrained
`SUPPORTS | REFUTES | INSUFFICIENT` and ordered same-document sentence IDs.
SUPPORTS/REFUTES can use multiple jointly supporting sentences; INSUFFICIENT
requires no IDs. Original IDs, text and ordering remain intact. Hidden,
cross-document, duplicate and invented references are rejected.

Both routes use the same model object for planner, verifier and terminal:

| Route | Policy | Maximum trace |
|---|---|---|
| fixed | Verify visible documents in frozen retrieval order, reserving terminal | 4 verify + 1 terminal |
| adaptive | Model chooses whether/which visible document to verify, or terminates directly | plan → verify → plan → verify → terminal |

Each episode shares **5 physical generations / 5 tools including initial
retrieval / 120 seconds / 512 output tokens per generation / 8192 input tokens**.
The adaptive schema offers verify only when at least three generations remain
before its planner call, leaving verifier and terminal capacity. Failed calls
consume their generation and tool debit. No hidden retries, warmups or batching.
Repeated verification of the same document is disallowed. A direct model
answer/abstention ends the episode without free synthesis.

Full provenance, ordered sentence hashes, actual physical call ID, usage and
failure status remain in the private trace. The next planner/terminal receives
the exact judgment (or failure), source ID/SHA and explicit fallible/non-citable
marking. Duplicate per-sentence hashes and billing fields are not duplicated in
the model prompt. The entire original visible evidence set is still supplied.
Document-level INSUFFICIENT is not a claim-level NEI decision; parse errors,
truncation, timeouts and unknown costs cannot be converted to INSUFFICIENT/NEI.

## Prompt readiness: discovered overflow, repaired without dropping evidence

The first CPU stress probe on all actual initial frames found five claims over
8192 tokens after four feedback envelopes (maximum **9202**). Repeated sentence
hashes in feedback, not the original evidence, caused this overhead.
Keeping those full fields in trace and using the sourced compact judgment in
the prompt yielded **maximum 5511 tokens, 0/24 overflow** in the same probes.

The probe covers every document projection, the initial optional-tool schema,
and terminal/planner prompts with accumulated maximum-citation or failed
verification feedback. Its feedback labels are explicitly **synthetic CPU
envelopes**, not model predictions or semantic evidence. It does not certify
every possible future JSON permutation. Runtime overflow still fails explicitly
without generation or evidence removal. No scientific quality is inferred.

Private probe artifacts are under
`E:/Project/_climate_transfer/document-verifier-cpu-inputs-20261001/`.
Frozen source input and restricted/large artifacts remain in their original
Spartan locations; no original claim text or model responses are added to Git.

## Exit, cost and scoring

The parent confines and reaps one child process; each episode also has the
existing external 120-second watchdog. Only after exit is `cost-before-gold`
written. It retains total and per-arm physical calls, input/output tokens,
backend/ledger elapsed time, missing receipts and unknown lower bounds.
Truncated or unassigned reservations remain charged as `unassigned`, cause
`no_quality`, and cannot be dropped or filled with zero.

Before opening selected TRAIN gold, reconstruct every model judgment and final
prediction from its physical raw response. Check actual stage prompt/token
hashes, unadapted model guard, visibility, feedback transitions, policy,
shared budget and cumulative physical elapsed time. A controller row cannot
claim a shorter episode than its physical ledger. No hidden calls may remain.

Use the original per-document SciFact scorer plus strict whole-answer correctness,
separating positives, NEI and unresolved, retaining all 24 cases per arm. The
fixed and adaptive routes have identical scoring. Missing/failed/unknown-cost
runs have no quality output. No verdict-model placeholder zero is used.

Decision rules:

1. Both routes have zero correct positives: the hypothesis is not supported,
   even if NEI improves.
2. Fixed recovers positives: verifier efficacy is observed; Agent benefit still
   requires correct adaptive proposal → execution → observed feedback → action.
3. Adaptive alone recovers positives: report that separately, never falsely
   state positives remain zero; audit actual trajectories before any Agent claim.
4. Compare quality against **per-arm actual costs**, not only each best answer.
   Consumed TRAIN24 cannot establish population generalization or test gains.
   No automatic SFT, new sample, test opening or retry follows a negative result.

## Entrypoints and resource proposal

- `prepare_scifact_document_verifier.input_inventory/load_frames`: frozen actual
  inputs, no gold or new retrieval.
- `preflight_scifact_document_verifier.py`: tokenizer-only context stress probes.
- `run_scifact_document_verifier.py`: same-provider 48-episode worker.
- `run_scifact_document_verifier_operator.py`: isolated allocation, runtime
  identity, generator-only extraction, child exit, cost-first scoring.
- `score_scifact_document_verifier.py`: raw/contract reconstruction and TRAIN24
  quality; old scorer modules are unchanged.
- `package_scifact_document_verifier.py`: reproducible draft release constructor,
  bound to the two accepted runtime records; no submission or model calls.
- `hpc/scifact_document_verifier.sbatch`: **DRAFT r2 candidate**, single A100/8 CPU/32 GiB RAM/
  30 GiB scratch/2 h, no requeue. Worker cap 6360 s comprises 48×120 s ceilings
  plus 600 s loading/preflight; the remainder covers extraction/scoring/exit.
  Previous comparable run used 666 s for 72 generations, but this is not a
  runtime guarantee. Only `sbatch --test-only` is allowed in this CPU package.

The shared legacy journal only gains an overridable renderer and capacity
ceiling; its old renderer, 168 hard maximum and default 56 are unchanged.
The new journal alone has 240 global/5 per-episode caps. Legacy regressions are
included with new synthetic tests for feedback, multisentence evidence,
failure-not-NEI, physical time, unknown cost and release refusal.

Local full regression: **1209 passed / 1 Windows POSIX-invariant skip** in
216.19 s. Ruff over `src/scripts/tests`, strict affected-module/script mypy,
shell syntax and tracked secret/PII scan passed. The skip is a platform-specific
permission assertion, not a missing Torch/Linux dependency. No new quality
claim follows from these mechanical checks. Clean-source reproduction and the
exact source/wrapper/release hashes accompany the coordinator handoff.

No resume, career evidence, Trip, Energy or FLARE files are edited by this package.
