# Frozen adapter, old twelve-query four-route regression — CPU package

## Decision and scientific boundary

Status: **Job 31773502 completed and physically audited; negative Agent result**.
The original CPU package and DRAFT below remain unchanged historical receipts;
the separate submission record is at the end of this document. Preserve
validation entrypoint/source `ce34e16` without calling its twelve frozen
validation queries. The tune count gate remains unchanged; it is not an
autonomous-Agent acceptance gate.

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

### Exact CPU handoff (completed, still DRAFT)

- Execution source: `bd7af701fcca654383f1907520612f79d5516e3b`.
- Source tar SHA: `a99051e0ece0e9fa7e0a103285f35645a3cf546b2081ebcf7d2be044a98c6f91`.
- Correct wrapper SHA: `bbc5df196509695b0391583ea5e663a21a234ae1e5147ba605a4aaee6eda885a`.
- DRAFT release SHA: `d66594ef975b6491d0c78f70107f38799a8051eca2be76030f5aa88cd827ff4d`.
- Readiness receipt SHA: `d6dc111e9fe63f2f21f992b58ff43a70cdc47173d0af0957e5c6cda3c556abbe`.

[Exact-source receipt](verified-runs/scifact-adapter-regression-source-bd7af70.json)
verifies 486 regular files, a unique 41-byte SOURCE_REVISION and the actual
wrapper guard. **Clean exported source** passed 70 related tests in 8.87s,
Ruff and strict mypy; one tiny-PEFT vocabulary warning remains expected.

[Spartan readiness](verified-runs/scifact-adapter-regression-readiness-bd7af70.json)
verifies actual Python 3.10.4 / POSIX / Torch 2.1.2 + isolated PEFT runtime imports,
the frozen checkpoint files and original twelve-query input archive. No gold or
weights were loaded. Model archive header-selected payload is 16,120,463,404
bytes; full base/reranker payload hashing remains mandatory inside allocation
before loading, not represented as already rehashed during this CPU check.

`sbatch --test-only` returned zero. Its simulated Job 31771616 / estimated start
text is **not an actual submitted job or a guaranteed schedule**. Output was
unused. The staged files are under:

`/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/envs/adapter-regression-source-bd7af701fcca/`

The proposed output remains:

`/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/runs/scifact-adapter-bare-regression-20261001-v1`

The source tar, `wrapper.sbatch`, `release-draft.json`, CPU probe and readiness
receipt are available there. No actual sbatch, warmup, training or generation
occurred in this package. A later documentation-only commit does not replace
the frozen execution source. Next execution still requires coordinator review
and a distinct exact-hash authorization receipt.

### Single actual submission (2026-10-01)

The coordinator independently released **only** the old12 four-route regression.
The sole DRAFT token was replaced in its original bytes, preserving the DRAFT:
activated release **4532 bytes**, SHA
`b435fa82002e3dbb96511c73e02339ed5a875e56ad97414dfc03fc315598c6ae`.
Execution source remains `bd7af701fcca654383f1907520612f79d5516e3b`;
later receipt/documentation commits are not substituted as execution source.

An exclusive, durable `single-submission/` reservation precedes the only
`sbatch` invocation. It preserves raw stdout/stderr, response, and receipt.
Actual `sbatch --parsable` returned **31773502**, exit zero. Receipt SHA:
`e5df5789d1d1dcf98c4e3a9be2a85b29b448779c936d6338361ab2167d2688f1`.
Any unknown submission outcome requires reconciliation, not another submission.

First scheduler snapshot: `PENDING (Resources)`, elapsed zero, no estimated
start (`N/A`). Priority 13408 = fairshare 13407 + job size 1, with age/site/QoS
zero, in both requested partitions. This is neither a running allocation nor
an evaluation result. Do not confuse the earlier test-only ID with this job.

The matching job queue was empty and output/authorization/reservation paths
were unused at precheck. Filesystem available space was 294,440,140,800 bytes;
this is **not a user quota measurement**. `quota -s` separately reported the
home NFS volume at its 50 GiB limit. Sources and receipts use project GPFS;
model/cache/temp use allocation scratch, not home. No home cleanup or global
environment changes were made. The CPU receipt already verifies working
Torch 2.1.2 and POSIX; no reinstall was needed.

All original 48-slot, 168-generator, 36-rerank and 720-requested-pair limits
remain, including zero extra smoke/warmup. No new training, validation, dev/test,
automatic retry, or scheduler override is authorized. Quality and cost closeout
remain pending; no resume metric is added merely because submission succeeded.

### Completed physical closeout

Job 31773502 completed `0:0`, elapsed 201 seconds, TotalCPU 183.559 seconds,
MaxRSS 17,338,364 KiB. [Compact physical evidence](verified-runs/scifact-adapter-regression-closeout-31773502.json)
binds the original score `1d75e03a...c84b7` and run `7ebffe6e...daade`.
195 indexed physical files were rehashed; all 48 raw slot records and physical
wire audits match. All 144 restored adapter tensors and 72 active unmerged
layers were verified. Exit/reap precedes cost persistence and scoring. The
closeout did not load gold, rescore, call a model, or alter the original results.
Remote compact SHA: `73cd7561391c090f5ddb1569b12e0db9f45554d73fd37c767aa3e5f352e4e8d4`.

| Route (12 slots each) | Correct rationalized docs / 9 relevant | Document F1 | Strict whole-answer slots |
|---|---:|---:|---:|
| Fixed retrieval | 1 | 0.09524 | 1 |
| Fixed rerank | 4 | 0.38095 | 4 |
| Deterministic extra | 3 | 0.28571 | 3 |
| Adaptive | 1 | 0.10000 | 2 |

Adaptive's two strict slots include **one valid NEI abstention**, not two correct
documents. Its twelve `slots_with_tool_opportunity` mean **menus offered tools**;
only three frozen cases had independently established read-replenishable evidence.
Adaptive made zero tool proposals, executions or new-evidence closure chains.
It returned eleven answers and one abstention. Its three read-opportunity cases
all failed; fixed rerank was correct for 1/1 FIT-overlap and 1/2 not-direct-FIT
cases. Both groups are exposed TRAIN-internal regression inputs.

All 48 calls have known usage: 172,572 input / 1,376 output tokens; 24 completed
rerank operations, 480 requested pairs. Reranker tokens remain unmeasured, not
zero. No warmup or repair calls occurred. Timing is this bounded sequential
experiment, not an online SLA.

A separate read-only aggregation of the already stored physical responses
found **47 nonempty outputs, each one document and one sentence**. Fixed
retrieval chose visible rank one nine times and rank two three times; adaptive
did so eight and three times. Fixed rerank and deterministic extra each chose
the first visible document 12/12 times, with multiple original aliases
(`c0/c1/c2/c6/c12/c17`, depending on route). Thus the earlier tune's literal
`c1`-only behavior must not be generalized to this regression. The observed
behavior is narrow output structure and visible-order dependence, not proof of
a universal alias lock or a causal explanation.

Conclusion: fixed rerank improves this exposed diagnostic's evidence output;
the adapter still has not demonstrated autonomous tool selection. Preserve
negative NEI/gold-absent/read-opportunity cases. No validation launch, retraining,
new model call, or resume claim is authorized by this closeout.
