# Frozen adapter, old twelve-query four-route regression — CPU package

## Decision and scientific boundary

Status: **DRAFT; no GPU submission or real model invocation authorized by this
package**. Preserve validation entrypoint/source `ce34e16` without calling its
twelve frozen validation queries. The tune count gate remains unchanged; it is
not an autonomous-Agent acceptance gate.

The [posthoc physical audit](verified-runs/scifact-grounding-structure-20261001.json)
confirms 96 single-document FIT records, all `answer/c1`; adapted tune outputs
also use one `c1` document each. Sentence indices vary. One old selected query
and component occurs in FIT, within the three saved read-opportunity queries.
Report this group as **one FIT-overlap plus two not directly FIT-overlapping**;
both are previously exposed regression inputs, never held-out samples.
Keep all twelve and the NEI / gold-absent-Top20 failures. Do not use saved read
witnesses as actual model actions or force reads.

The original bare controller starts aliases at `c0`; FIT/tune contexts started
at `c1`. This transfer limitation is recorded, **not silently normalized**.
Only within-new-run comparisons use the same adapter in every route. Comparing
against old G-policy results cannot isolate a causal LoRA benefit.

## Reuse, not a new controller

`LocalQwenBoundedSciFactProvider(gap=False)` → real CausalLM adapter restoration →
`run_bounded_slot(ARMS[0])` → existing `CommonPacking` → original route controller.
No `GapProviderAdapter` / new G policy / prompt change. Existing CommonPacking is
`max(bare, empty-G, active)` token fitting, **not pure bare token counting**.
Actual token costs remain the provider's actual prompt usage.

The loader verifies all checkpoint files, exact 144 tensor keys/values, fixed
base model/tokenizer hashes, 72 active unmerged LoRA layers, `default` adapter,
eval mode and no trainable parameters before any slot. Active state is checked
again before each slot and physical generation. Release/slot/raw/run receipts
bind base + adapter + training hashes, active status, prompt, schema implementation,
dynamic prompt/schema hashes, decoder, packing and budget. A base-only name/hash
is not proof of adaptation.

Frozen evidence:

- Adapter training receipt: `f5e6a865ec09cb67a520e36ba4646fb297a381383d4d0357208488ef6e9168a0`.
- Adapter tensors: `dd1974a26549b3337824252a9c38773873acce72427f92e656835bf9d71935d4`.
- Selected old input: `0ac620a9b3e10ed80aac3724689e4d23830450b440d2dc8d35f2ed7f79657e44`.
- Reused inference archive: `def10d4030d402ef5294680e0426ba1ed05ac42cb9b1d7bf4e8daa0a328961c9`.
- Reused scoring archive: `7e2303364b69be920de017179485bbb6f9d0e5d48367717576b3411c92c8ca8e`.

The old bundle/protocol validates original bytes and old source `99cd9ff`; the
new source/release/policy are separate execution identities, not an assertion
that the old semantic-G protocol executed the new bare adapter.

## Calls and resource bounds

| Route (12 slots each) | Generator call cap, including repair | Rerank operation cap |
|---|---:|---:|
| fixed_retrieval | 36 | 0 |
| fixed_rerank | 36 | 12 |
| deterministic_extra | 36 | 12 |
| adaptive | 60 | 12 |
| Total | **168** | **36** |

48 denotes slots, not model calls. At most 96 repair calls are already included
in 168. There are **zero extra smoke/warmup calls**. The first real slot uses the
same audited path; no preflight consumes another query. Rerank is batch size 1,
at most 720 **requested** query-document pairs, not evidence of 720 completed
forwards after a failure. No reranker token measurement is available (`null`,
not zero). Per-route completed/failed-or-uncompleted states, known elapsed time
and missing timing are aggregated from existing raw events.

Unchanged per-slot budget: K20/context5, 5 calls, 2 repairs, 5 tools, 8192 input /
512 output tokens, shared 120s **cooperative slot deadline**, not 120s per call.
Synchronous rerank can exceed that check, so the isolated worker has a **6300s
hard wait bound**; timeout or interrupted wait kills/reaps only its child group.
The shell envelope covers prechecks/extraction/scoring with GNU timeout **6900s
plus a 30s kill grace**. Slurm **7200s** is the final cgroup-wide deadline.
Time reserved for post-exit scoring: 180s. Existing shape: 1 A100 / 8 CPU /
32 GiB host RAM / 30 GiB scratch / no-requeue. 48×120s=5760s accounts for the
96-minute worst-case checked-slot budget; tune's 25/30-minute caps cannot apply.

## Exit-first auditing and costs

The worker receives only old claims/corpus and model assets, not a gold path.
After it exits and is reaped, the operator may extract the frozen scoring bundle.
This is process/dataflow separation under the same account, not an OS sandbox.

Reuse `physical(..., gap=False)` → `audit_slot(..., gap=False)` →
`score_diagnostic(...)`, including exact
`result.answer → to_original_prediction → row.prediction` equality. The audit
links raw proposal → validated decision → actual event → next visible input →
final citation → official complete-rationale/label credit. Mechanical schema
validity or a fixture `model_selected` flag alone never demonstrates Agent value.

Costs are persisted **before** any quality/gold loading. Every slot is reserved
durably; a missing raw record yields unknown calls/cost and preserves all 48
planned slots, rather than zero cost or a dropped denominator. Failures and
repair calls remain charged. Partial runs retain physical hashes and costs but
do not produce a quality gate. Complete runs report full twelve-query results
plus the predefined 1+2 read-opportunity strata, without suppressing failures.

No validation gate, automatic retry, next training or other model launch is
created. After one separately released regression, the coordinator decides
whether missing tool-trajectory/NEI supervision needs a new bounded proposal.

## CPU validation and reproduction

New/affected fixture suites pass: 70 tests, one pre-existing PEFT tiny-model
vocabulary warning. This includes an actual tiny CausalLM restoration/disabled
adapter rejection (no model download/generation), active-state tampering,
wrong release/DRAFT rejection, interrupted child cleanup, full four-route
physical/event/prediction scoring, a changed prediction rejection, and unknown
partial cost preservation. The real shell prologue also terminates a sleeping
CPU fixture at its shortened one-second outer deadline. Ruff and strict mypy
pass for the four new source files.

Entrypoints are `run_scifact_adapter_regression_operator.py`,
`run_scifact_adapter_regression.py`, `score_scifact_adapter_regression.py`.
Packaging must explicitly choose `hpc/scifact_adapter_regression.sbatch` in
`package_scifact_source.py`; its legacy default wrapper is not this release.
Exact export/import/test-only receipts will be appended after CPU preparation.
No public/private test, resume or shared career material is changed.
