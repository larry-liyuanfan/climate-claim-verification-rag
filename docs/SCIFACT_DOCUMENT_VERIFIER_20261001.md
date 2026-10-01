# Single-document verification diagnostic — 2026-10-01

## Status and falsifiable question

**R2 completed with a negative result.** Job 31918065 ran 2026-10-01
22:33:56–22:41:16 +10, `COMPLETED/0:0`, on execution source
`86414e39d4bd9516c9f8cc2d20c10b184c56f2b7`. All 48 episode records and 144
physical call receipts are present; no unknown/unassigned cost remains. See the
[compact closeout](verified-runs/scifact-document-verifier-closeout-31918065.json).

| Metric | Fixed verification | Optional adaptive verification |
|---|---:|---:|
| Strict complete positive answers | 0/15 | 0/15 |
| Correct NEI | 3/9 | 0/9 |
| Valid terminal / unresolved | 15 / 9 | 8 / 16 |
| Model-selected verify proposals / executions | 0 / 0 (scripted route) | 0 / 0 |
| Scripted verify executions | 96 | 0 |
| Physical generations | 120 | 24 |
| Input / output tokens | 254,677 / 6,815 | 86,066 / 5,076 |
| Model backend time | 215.506 s | 150.486 s |
| Abstract rationalized micro-F1 | 0.135593 | 0.083333 |
| Sentence-label micro-F1 | 0.060606 | 0.052632 |

Every adaptive first wire response proposed `answer`, including the 16 rejected
ones: there was no hidden/rejected verify request. Thus there is **no autonomous
proposal → verification → feedback → subsequent action trajectory** to claim.
Fixed verification yielded 96 structurally valid judgments, but that is not
gold-label accuracy; the downstream system still recovered no strict positive
answer. It does not prove that each individual judgment was wrong. Its three
correct NEI abstentions do not establish positive grounding or Agent benefit.
Failures remain in the 24-per-route denominator with their actual costs.

Physical totals independently sum to 144 generations, 340,743 input and 11,891
output tokens. Allocation elapsed was 440 s; MaxRSS was 9,263,156 KiB (8.83 GiB).
These are offline diagnostic costs, not a billed saving, GPU utilization or an
online SLA. Original TRAIN24 was already consumed; no generalization, independent
test improvement or new resume result follows. This package is closed; any next
experiment needs a separately justified hypothesis and authorization.

## Preserved startup failure and repair

The original CPU package was released once as job **31914601** (source
`496b584cdb93041fa53a5f8eb662fdad0c49a088`). It failed before model loading on
2026-10-01, 22:16:20–22:16:40 +10, `FAILED/1:0`: the release omitted
`python_executable`, required by the reused runtime consumer. Allocated resources
were one A100, 8 CPUs and 32 GiB; batch MaxRSS was 304436 KiB. Only preparation
and reservation files exist: **zero inference ledger entries/results, no quality
denominator**. This was a startup-contract defect, not missing Torch or POSIX.

The first infrastructure retry **r2** received a separate exact-hash release
`7efcde3be0a066c504bfe876df961a7eebf53f56b7af93a411dcf1020ad25877`, used
once for 31918065. The protocol, original 24×2 matrix, model, prompts, scoring
and budgets were unchanged. The old stage, lock and run remain preserved;
the new run is `runs/scifact-document-verifier-v1-20261001-r2`. No automatic
resubmission, training, validation or frozen-test evaluation followed.

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
shell syntax passed; clean-source reproduction passed 74 tests. Linux CI
36861788720 completed **1254 passed / zero skipped**, with source mypy and secret
scan passing. None of these mechanical checks is substituted for the negative
quality results above.

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
- `hpc/scifact_document_verifier.sbatch`: resource ceiling for the **DRAFT bounded-decoder-v1 comparison**, single A100/8 CPU/32 GiB RAM/
  30 GiB scratch/2 h, no requeue. Worker cap 6360 s comprises 48×120 s ceilings
  plus 600 s loading/preflight; the remainder covers extraction/scoring/exit.
  Previous comparable run used 666 s for 72 generations, but this is not a
  runtime guarantee. Only `sbatch --test-only` is allowed in this CPU package.

The shared legacy journal only gains an overridable renderer and capacity
ceiling; its old renderer, 168 hard maximum and default 56 are unchanged.
The new journal alone has 240 global/5 per-episode caps. Legacy regressions are
included with new synthetic tests for feedback, multisentence evidence,
failure-not-NEI, physical time, unknown cost and release refusal.

Initial implementation's historical local full regression: **1209 passed / 1 Windows POSIX-invariant skip** in
216.19 s. Ruff over `src/scripts/tests`, strict affected-module/script mypy,
shell syntax and tracked secret/PII scan passed. The skip is a platform-specific
permission assertion, not a missing Torch/Linux dependency. No new quality
claim follows from these mechanical checks. Clean-source reproduction and the
exact source/wrapper/release hashes accompany the coordinator handoff.

No resume, career evidence, Trip, Energy or FLARE files are edited by this package.

## Whole-chain review before submission

The review follows release producer → actual runtime consumer → worker/provider
→ generation callback → tool feedback → physical replay → cost → scorer. A
collection of passing component tests is not a substitute for these seams.

The r2 raw audit found 144/144 complete wires. Both adaptive schemas and inputs
offered `verify`, but all 24 adaptive first responses chose `answer`. The fixed
route generated 96 valid document judgments (20 SUPPORTS, 15 REFUTES and 61
INSUFFICIENT). These are predictions, not accuracy counts. All 25 unresolved
terminals failed the existing parser: fixed 7 total-sentence-budget + 2 duplicate
document failures; adaptive 13 + 3. Duplicate first failures can also exceed the
sentence budget. The existing semantic tracker rejects those exact 25 raw
outputs and accepts the other 23. It does not predict a guarded rerun's quality.

Five reproducible cases in the preserved r2 private results are:

| TRAIN case | Fixed route | Adaptive route |
|---|---|---|
| 403 | Four INSUFFICIENT judgments, then scored-correct NEI abstention | 36 citations; budget rejection |
| 304 | Four INSUFFICIENT judgments, then scored-correct NEI abstention | Valid 5-document/20-citation answer, incorrect under NEI scoring |
| 853 | Terminal reverses one judgment's label and repeats its document | Also repeats a document; both have 31 citations |
| 1237 | Valid terminal does not consistently retain positive/negative feedback | 22 citations; budget rejection |
| 341 | Valid terminal follows both positive judgments yet fails strict scoring | 23 citations; budget rejection |

These cases distinguish structural rejection, feedback use and gold correctness.
Four document-level INSUFFICIENT judgments do not prove claim-level NEI. The
CPU-only `audit_scifact_document_gold.py` diagnoses local label/first-three
rationale coverage separately from full-selection coverage and strict annotation
agreement. A gold `[0,1]` versus judgment `[0,1,2]` is **not** called a strict
match. Unannotated citations are not proven semantic errors. No projection is
counted as a generated final answer, and no old score is rewritten.

The minimal implementation uses `LocalQwenSciFactProvider.build_decoder()`:

- Default provider callback construction and generation defaults are unchanged.
- Document `plan`/`terminal` use `build_bounded_scifact_prefix`; `verify` retains
  ordinary LMFE. Every call receives fresh parser state; no fake `preview_only`
  observation or inherited bounded-provider initializer is introduced.
- Actual parser/config and `scifact-document-stage-bounded-v1-20261001` identity
  are recorded. Prefix construction remains charged to the same deadline.
- Model-facing observations, prompts, schemas, model, data, scorer and budgets
  are unchanged. Old r2 authorization is rejected. A new draft attempt/output
  `bounded-decoder-v1` is a scientific implementation comparison, not an
  infrastructure retry; no new GPU execution is authorized by CPU checks.

Acceptance includes the real inherited `generate()` with a prescribed-token
fake model invoking the real LMFE callback, then Journal → episode → physical
audit. Tests cover optional verify, stage routing, repeated callbacks, invalid
IDs, duplicate documents, 20/21 citations, time accounting and cost preservation.
Scorer-entry tests cover four no-quality gates and 48 synthetic slots with
costs/physical audit before synthetic gold access. Existing startup consumer
tests protect release fields and exclusive output reservation.

Real cached-Qwen-tokenizer CPU preflight also exercised the actual generate seam:
plan/verify/terminal and INSUFFICIENT accepted; duplicate terminal rejected.
It loaded no model weights, made no model inference calls, read no gold and is
not a latency/quality benchmark. Script SHA:
`c85ecb1026cda198f88535774901eaa50670aed6e1112f2c4af63d05318d819b`.

Before any queue submission: finish independent diff review, lint/type/full
regression and clean-source integration reproduction; freeze exact source,
wrapper and release identity together. The coordinator decides any new GPU run
once from that cohesive package, rather than approving individual files.

Current bounded-decoder candidate: independent actual-diff review found no
remaining blocker; local full regression **1280 passed / 1 Windows-only POSIX
permission skip** in 142.53 s. Ruff passed for all source/scripts/tests. The
cached real-tokenizer preflight and 48-slot synthetic scorer entry tests passed.
These mechanical checks do not change r2 quality or authorize GPU execution.
