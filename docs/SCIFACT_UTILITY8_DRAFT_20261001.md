# utility8: bounded real-tool-value diagnostic — DRAFT

No GPU job, model inference, training or new data inspection was performed when
preparing this implementation. It requires the coordinator's exact-hash release
after source/interface/scoring review. It is not evidence of model improvement.

## Fixed scope

Eight **already exposed official TRAIN** examples, outcome-selected from the
hash-locked original48 and supplemental96 reports, in this order:

1. Original48's unique natural-read case.
2. Original48's first two initial-sufficient cases, preserving report order.
3. Supplemental96's first two initial-sufficient cases, preserving report order.
4. Supplemental96's first context-abstain case.
5. Supplemental96's first two official NEI cases.

The selection freezes claim IDs, components, roles, source hashes and branch
order before decoding query strings. No substitution, new sampling, packing
change, SFT, adapter, protected/tune/validation/dev300/frozen-test access or
claim of an independent evaluation. Only the shared public 5,183-document
corpus and the eight gold-free query rows reach inference. The actual eight
IDs have **not yet been materialized by this draft package**.

## Branch and budget semantics

- **A:** frozen base Qwen3-4B, `run_bounded_slot`, adaptive route, unchanged
  CommonPacking and V3Budget: 5 calls / 5 tools / 8,192 input / 512 output /
  120 seconds. Capture after attempt0 parsing but before its tool execution.
- **B:** resume A's exact initial state and paid attempt0 prefix; execute one
  scripted **real read**, then allow at most one new real model decision.
  Select the first reference from A's original, lawful attempt0; otherwise
  the first initial preview. An invalid attempt0, original read loop or invalid
  reference is `not_run`, never repaired by selecting a later response/reference.
- **C:** resume the same prefix; execute one real Qwen3-4B rerank of the current
  candidate set; then at most one new real model decision.

B/C deduct the complete A prefix elapsed (initial retrieval + packing + actual
first generation), retain its raw response/usage, and use 4 remaining calls and
3 remaining tools after the scripted tool. State reconstruction overhead is
also charged by the controller clock. Their one-new-decision restriction does
not change the offered schema or original budget. New tool proposals are
`proposed_not_executed`; schema validity and controller legality remain separate.
No repair or further tool follows the continuation. Terminal A0 answers/abstains
may be forked for diagnosis, but those branches are not autonomous policy choices.

Stable `cN → original source` mappings survive reranking. Full ordered
observation/schema, rendered prompt and token-ID identities are checked; ordered
exclusive JSON persistence preserves physical prompt order on disk round trips.
This is a strategy diagnostic, **not** an equal-compute causal comparison.

## Physical cost and model runtime

- At most **56 actual generator invocations** (`8 × (5 + 1 + 1)`), including
  failed invocations. No old four-call runtime smoke or uncounted warmup.
- Persist each unique physical reservation before the provider runs, then its
  response/failure and known/unknown usage before another invocation. Reusing
  A0 costs no new physical call; logical branch totals include that shared ID.
  Equal response text does not merge independent physical calls.
- At most **16 actual rerank requests / 320 requested pairs** across A and C.
  Forward hooks record real padded/nonpadding input tokens, token hashes,
  start/completion and timing. Unknown tokens stay `null`. Reranker forward
  classification is not counted as text generation.
- Load reranker on CPU. Before each real rerank, move generator to CPU, release
  cached GPU memory, move reranker to GPU, run it, move it back, restore generator.
  Swap latency is included. This is serial GPU residency, not merely serial calls
  with both models left on GPU. Actual peak memory remains to be measured.
- One A100, 8 CPU, 32 GiB host RAM, 30 GiB scratch, Slurm cap 60 minutes;
  no automatic restart/requeue/replacement. One lifetime output reservation.

## Cost before quality

Inference is a separate bounded/reaped child. First persist physical generator
and reranker cost plus planned/missing slots. Unknown cost, missing expected
receipts, nonzero worker exit or unresolved identity produces `no-quality.json`
without reading new official gold or publishing quality numbers.

For a complete auditable run, decode official TRAIN annotations for only the
eight selected IDs after exit. Reconstruct terminal predictions from real raw
responses and frozen visible sources. Reuse the existing four official micro
metrics and raw counts, alongside valid-terminal, strict-whole-answer and NEI
false-evidence/valid-abstention/failed-empty counts. Every arm retains eight
planned slots; empty failure fallbacks are never called correct NEI.

A0 is a direct baseline only when its first decision is a lawful answer or
abstention; never substitute A's final answer. Report complete per-document
alternative rationales, wrong document/label/sentence additions and losses.
Candidate trajectories require a legal terminal, gained complete rationale,
no added erroneous evidence **and no lost complete rationale**. No automatic
training follows. Same quality at extra cost is not positive utility.

The tool-change trace includes scripted/model origin, before/after requested
context, new/lost visible sentences, visibility order, next-frame feedback and
elapsed time. Rerank may change order/packing without introducing a new source;
new IDs are not a prerequisite for usefulness.

## Reproducible entry points

- `scripts/prepare_scifact_utility8.py`: selector, immutable reservations and
  ID-first query-only adapter. It does not call models or open full gold rows.
- `scripts/run_scifact_utility8.py`: base model + real reranker worker, physical
  ledger and A/B/C matrix. Requires allocation and exact release.
- `scripts/score_scifact_utility8.py`: post-exit costs, physical audit, fixed TRAIN
  scoring; never an independently runnable held-out evaluation.
- `scripts/run_scifact_utility8_operator.py` and `hpc/scifact_utility8.sbatch`:
  allocated orchestration, resource limits and no-retry lifetime reservation.
- `scripts/package_scifact_utility8.py`: clean exact Git archive plus **the actual
  utility8 wrapper** guard, including wrong revision/archive/wrapper rejection.
  A legacy wrapper's passing receipt is not used as utility8 proof.

Release fields: authorization=`coordinator_exact_hash_release`, protocol,
source_git, source_archive_sha256, wrapper_sha256, runtime_receipt_sha256,
runtime_files_sha256, python_executable, model_archive_sha256, fixed output,
resource_cap, max_generator_calls=56, max_rerank_requests=16,
warmup_generation_calls=0, training_authorized=false,
protected_split_read=false, automatic_retry=false. No executable release file
is supplied by this draft. Existing runtime inventory receipts and actual import
identity are verified; this does not claim a new full 11k-file cache rehash.

Synthetic tests exercise round-trip resume, original alias identity, budget/time
deduction, no free shared call, invalid/loop proposals, attempt57 rejection,
fixed selection, query-only decoding, 24-slot cost/audit, rationale tradeoffs and
cost-first no-quality behavior. Synthetic counts are infrastructure evidence,
not scientific model results. The normal controller's optional hook defaults
to `None`; no production policy, prompt or original experiment is replaced.

## Local draft validation

On 2026-10-01, the utility8 synthetic suite plus the bounded-runtime, terminal
and read-continuation regression suites passed **90 tests** in the existing
Windows Torch validation environment. This includes 23 utility8 cases. Ruff
passed for the modified/new Python files, and targeted strict mypy passed on
nine source files with `--follow-imports=silent --platform linux`; this is not
a claim that all legacy script imports pass strict typing. The exact-source
packager emits a separate post-commit receipt for the actual wrapper guard.
No model call, new dataset read or remote operation is part of these checks.
