# Evidence and claim boundaries

## 2026-10-02 evidence bottleneck supervised execution preparation

[Execution entry and reproduction](SCIFACT_BOTTLENECK_EXECUTION_20261002.md)
adds the parent/runner/exit/scorer/compact connection without changing A/B/C,
input, decoding, cost or scoring contracts. Fixed 72 planned slots and physical
cost survive preparation, model-load, worker and scorer failures. The model
path is bound to actual scratch extraction rather than a rewritten release.
Only synthetic CPU/CI validation and metadata packaging are authorized here;
no cluster submission, model weight loading, real gold or new quality result.
The prior tokenizer receipt is reused only after component/probe equality checks;
31996384 remains the same historical failed job, not a rerun.

## 2026-10-02 evidence bottleneck CPU implementation

[Frozen mechanism and reproduction](SCIFACT_EVIDENCE_BOTTLENECK_CPU_20261002.md):
same consumed TRAIN24/top1, one-response A and shared-selector B/C, only actual
full-document visibility changes between B/C. Maximum 96 physical calls / 72
route results is a future-run draft ceiling, **not inference authorization**.
No new quality/resume number, no real gold or protected split reading in this
CPU package. Preserve v2/v3 results below; component changes are not Agent gains.

Single CPU attempt **31996384** completed 582 tokenizer probes on the frozen 24
claims, zero overflow (input max 3,836/8,192; synthetic output max 170/512), then
failed during Torch-dependent draft metadata export. Slurm **FAILED/1:0** is
preserved. Draft export was separated into a local CPU helper; no second job or
model call. [Execution receipt](verified-runs/scifact-evidence-bottleneck-cpu-31996384.json)
keeps the exact executed source distinct from the later local-only repair.

## 2026-10-02 paired verifier result and CPU-only scoring recovery

[Result/decision report](SCIFACT_PAIRED_VERIFIER_RESULT_20261002.md) and
[redacted compact](verified-runs/scifact-paired-31980221-cpu-replay.json), SHA
`d68fa9ace3ff7749c2f9ca5a86775d09ca7eb413214127425f109f3a0c4cb6e9`:
same consumed conditional TRAIN24 × two protocols × three arms, all 144 slots,
392 physical calls, 830,597 input / 20,737 output tokens, zero unknown usage.
No new sample, training, protected split or model call was used by closeout.

Original GPU job **31980221** is still FAILED/2:0, 762 s, MaxRSS 9,269,536K;
both workers exited 0 and were reaped. Original no-quality/gold-unread reports
remain untouched. The exact failure was `paired_effective_generation_binding`:
authorized JSON stored seven integer-valued floats as integers. Recorded contract
identity matched the released contract, but not the scorer's rebuilt representation.
Repair **1f6f920** retains frozen semantic checks and all model/physical/feedback
audits while checking the trusted release's exact `identity()` serialization.
Separate CPU replay **31988504** completed in 49 s / 868,868K MaxRSS, restored both
scores with unchanged original-tree manifests and zero model calls. A 684-file
`verify_source_tree` check passed **after** this replay; it is not a pre-execution
attestation. Future replay entrypoints now perform that same check before scoring.
Exact-repair [Linux CI](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36907952703)
passed **1,400 tests**, Ruff, strict typing and tracked secret/PII scanning.

v2 → v3 fixed-top1 strict positives **5/15 → 4/15**, official abstract-rationalized
F1 **0.5882 → 0.3500**; fixed-all **3/15 → 2/15**, F1 **0.4906 → 0.2750**.
Adaptive strict positives remain **5/15**, but NEI correct **6/9 → 2/9** and F1
**0.5556 → 0.4000**. v3 adaptive beats its own weaker top1 by one positive, not the
v2 top1 quality/cost baseline. Across-version adaptive has three positive wins and
three losses, not recovery of the same five cases. Reject v3 promotion; no resume
quality gain. Fixed unresolved and adaptive explicit abstention remain asymmetric;
do not convert these into a fair NEI/classification improvement claim.

## 2026-10-02 paired verifier preparation (CPU only; not a result)

[Whole-chain report](SCIFACT_PAIRED_VERIFIER_PREPARATION_20261002.md): prepares the
same consumed 24 frames across isolated v2 / structured v3 and three arms;
144 planned slots, independent 240-call ledgers, aggregate ceiling 480. Complete
generation defaults, effective overrides, seed and LMFE binding are recorded.
This supersedes the earlier 48-call top1-only proposal, not its measured results
(there were none). Reuses 31969505 without a new Slurm or model call. Fixed-policy
cross-version and within-version Agent comparisons remain separate; no resume gain.
Execution source `5a12fd9f398afb1af772111253600e68410e885a`: clean archive 60 passed
(one Windows POSIX skip); exact Linux CI 36900289223 **1,393 passed**; Ruff, strict
source typing and secret scan passed. Metadata-only Spartan runtime/configuration
preflight passed. [Compact receipt](verified-runs/scifact-paired-verifier-preparation-5a12fd9.json)
binds exact archive, parent/child drafts and runtime hashes; all remain non-executable
drafts until separate exact-release acceptance.

## 2026-10-02 structured relation/rationale verifier (CPU readiness only)

[Implementation/comparison proposal](SCIFACT_RELATION_RATIONALE_VERIFIER_20261002.md)
and [real-frame preflight](verified-runs/scifact-relation-verifier-cpu-31969505.json):
source `bc79ad927a1fea357c4cd1f0d0b38c2caf25e01d`, CPU **31969505 COMPLETED / 0:0**,
162 s, TotalCPU 148.698 s, MaxRSS 235,400 K. Versioned v3 produces one typed
relation/qualifier/direct-versus-background/minimal-evidence/uncertainty assessment;
projection preserves its ordered evidence under the original scorer and source
binding. Necessary joint-context sentences remain eligible; comparable numeric
contradictions can REFUTE and are not forced into incomparable-scope INSUFFICIENT.

Exact-source Linux CI **1,373 passed**, clean-archive **54 passed**. The same 24
frames produced **8,784 prompt probes per protocol**; maximum v2/v3 input tokens
**6,811 / 7,181**, against 8,192. New assessment's max-cardinality synthetic output
was 174 including EOS, against 512; zero observed overflows. New input isolation
passed 1,440 state checks including references plus 120 clean-history comparisons.
These are explicit probe configurations, not a proof that every legal response
fits or that any scientific label improved. No inference/training/new sampling/
gold/protected split access; no resume promotion. The unchanged five-call adaptive
controller still visits at most two documents; the three-document case stays in
the denominator. The original 48-call proposal was superseded by the later paired
three-arm preparation linked above; no separate 48-call run is planned or authorized.
Compact SHA `7e21b603c225374723122763f700f73dacc04f4005cde7784e23e837814cb388`.

## 2026-10-02 semantic-input isolation and error diagnosis (CPU only)

[Report](SCIFACT_SEMANTIC_INPUT_ISOLATION_20261002.md) and
[compact](verified-runs/scifact-semantic-input-cpu-31963271.json): job **31963271**,
COMPLETED / `0:0`, 16 s, 244,588 K MaxRSS; source
`58975a914b3bb0f224db366b519fe267feb0fbbd`. New explicitly versioned verifier input
is independent of arm/prior-call/remaining-budget state; planner budgets, physical
limits and old protocol remain intact. **1,440 arm/counter-state checks across
120 document probes** (including reference states), 120 clean-history checks
and **72 audited physical-record replays** passed; exact-source CI **1,342 passed**.
No inference, training, new sampling or protected split access occurred.

All **18 claim-document annotations / 15 positives** were initially retrieved and their
complete rationales visible. In the ten top1/adaptive common errors, adaptive
has four label-disagreement cases, eight with unannotated selected sentences,
and two with three reachable gold documents left unattempted; categories overlap.
No locally strict-correct verdict was omitted from final answers. Prioritize
verifier label/minimal-rationale selection, with multi-document control secondary.
Annotation mismatch is not proof of semantic falsehood. This consumed TRAIN
diagnosis proves input invariance, not improved verdicts, independent-test quality
or Agent value; original quality scores and current resume remain unchanged.
Compact SHA `5864c9912ad8ada0e868392aa18c883409d3ea43d6910988339f747fe1c38cce`.

## 2026-10-02 conditional TRAIN24 confirmation: no top1 promotion

[Final report](SCIFACT_PROSPECTIVE24_CONFIRMATION_20261002.md) and
[compact](verified-runs/scifact-prospective24-confirmation-31956320.json):
job **31956320**, COMPLETED / `0:0`, 300 s, 72/72 slots, 200 physical calls;
execution source `da243036871f61eef2e039a1618b8a3e1e1a00ac`.
Top1/all/adaptive strict evidence-bearing answers **5/15, 3/15, 5/15**;
abstract-rationalized F1 **0.6061, 0.5000, 0.5714**; tokens
**36,941 / 146,242 / 190,689**. Adaptive has no additional positive recovery
over top1 and costs 5.16 times the tokens. Six correct model NEI decisions are
not equal-interface gains over fixed unresolved responses. All 531 eligible
TRAIN rows had earlier gold-preparation exposure: not independent test or
source-family-unseen evidence. Three real paired examples, all failures/costs
and once-only shared preparation are preserved. No promotion, automatic rerun
or resume change. Compact SHA
`7315a387e66f60dde5aaa11840aca2a2c715168d9c7a043f575b2d65ebc46515`.

## 2026-10-02 prospective input preparation (CPU only)

[Whole-chain review and input contract](SCIFACT_PROSPECTIVE24_CPU_20261002.md):
conditional remaining 64 components / 97 IDs, fixed 24-component metadata
selection, no-generation original retrieval/packing and shared input adapter.
Zero model calls and no new real-gold access; no independent test or resume
improvement is claimed. Existing exposed TRAIN24 results remain separate.
CPU job **31953981** completed in 38 s / exit `0:0`; all 24 input frames are
prepared. The [compact](verified-runs/scifact-prospective24-cpu-31953981.json)
records source/data hashes and shared preparation costs; future GPU draft is
not executable authorization. Exact-candidate Linux CI: 1,329 tests passed.

## 2026-10-01 separately released component diagnostic submission

[Receipt](verified-runs/scifact-component-submission-31729507.json): exactly one
job **31729507**, source `79f069d`, archive `b46a1866...`, unchanged Stage A
protocol/33 inputs/targets/model identities. Scheduling test passed; first real
snapshot is **PENDING (Resources)** with no confirmed start. Requested one A100,
8 CPU, 32 GiB RAM, 30 GiB scratch, 100 minutes, normal QoS/Nice 0/no-requeue.
The fixed release was absent before submission and now has a one-attempt
submission journal. No new monitor, cancellation, re-submission or other-project
mutation. This is a submission receipt, **not model results or resume evidence
of improved quality**. Raw data/gold/predictions must remain on Spartan.

## 2026-10-01 component provider/operator CPU package

[Implementation and boundaries](SCIFACT_COMPONENT_EXECUTION_CPU_20261001.md):
real provider adapter and separate worker/scorer/operator are implemented but
not executed with model weights. The frozen 426ff73 preparation and 33 inputs
remain unchanged. Numeric-enum decoding is relaxed in a separately hashed copy;
the original output parser still rejects invalid IDs/duplicates/nonempty abstain.
Costs, unknown attempts and stopped planned slots remain in all denominators.
Synthetic grammar/token paths and software fixtures are not quality results.
The whole-input 6/9 candidate coverage must not be replaced by a valid-output
subset's pool coverage. No resume promotion or new model/data job follows.

## 2026-10-01 Stage A CPU component contract

[Implementation/limits](SCIFACT_COMPONENT_PREPARATION_20261001.md): separate
screening/relation/oracle-rationale inputs and schemas; actual-visible source
reconstruction; full-abstract budget gate; OR-alternative/first3/overselection
scoring; metadata-preserving failure denominators and single-attempt cost journal.
The targeted fixture suite and scripted CLI examples establish software contracts,
not grounding quality. No real model calls, new TRAIN partition, official dev/test
consumption, model training, Slurm submission or resume promotion is included.
Real preparation/clean reproduction status belongs to the versioned receipt;
passing synthetic cases does not establish real token-budget coverage.

The [real-tokenizer compact](verified-runs/scifact-component-preparation-426ff73.json)
now verifies 33/33 full-abstract inputs below the 8192-token limit: screening
1844–3019, relation 401–1060, rationale 452–1150. Candidate gold coverage is 6/9,
not model recall; all three cited-context NEI controls have usable sources.
[Receipt](verified-runs/scifact-component-validation-20261001.json): exact source
`426ff73`, 52 targeted tests in a clean archive, four scripted CLI fixtures,
Ruff/type/secret checks; zero real model calls. No weights or new dependency
download/install was needed for tokenizer-only preparation.

## 2026-09-30 semantic comparison: completed negative Agent result

[Report](SCIFACT_SEMANTIC_CLOSEOUT_20260930.md) and
[compact](verified-runs/scifact-semantic-pair-closeout-31706518-31706520.json)
bind execution `99cd9ff`, both completed jobs, 96 selected TRAIN slots and a
separate `2182f61` stdlib collector. All 32 official metric groups match
independent count/P/R/F1 arithmetic; durable rows, physical wire multiplicities
and token sums agree. Frozen scoring executed `physical/audit_slot(gap=True)`;
the CPU collector did not rerun a model or the full Pydantic scorer.

Adaptive old/new rationalized credit is **1/36/9 → 1/30/9** (correct/predicted/
relevant), F1 .044444 → .051282, with two label-correct documents in both.
Strict whole-answer 1 → 2 comes from valid NEI abstention 1 → 2, not more correct
evidence. Raw tools and actual model tool events are zero for both. Fixed-rerank
rationalized 2/34/9 → 3/34/9 occurs with label-correct 4 → 3: retain that tradeoff,
do not advertise it as Agent discovery. Complete first3-eligible rationales are
actually visible for three adaptive / six fixed-rerank gold documents per policy;
wrong relations, missed documents and late rationale placement remain.

Both jobs completed `0:0` in 629/627 s, batch MaxRSS 18,541,472/18,489,180 KiB;
each made 48 TRAIN + four real synthetic preflight calls, with zero unknown cost,
repair or termination failure. Token ledgers and offline timing scopes remain in
the report. No API financial saving, online SLA, independent test, causal Agent
benefit or resume improvement is supported.

Cumulative exposure leaves **491 claims / 301 components** after excluding
24 consumed components and 40 eligible member claims, out of 531/325. All 531
already had gold-aware preparation exposure. No new selection was made. The
[CPU-only feasibility plan](SCIFACT_GROUNDING_FEASIBILITY_20260930.md) proposes
33 oracle-conditioned diagnostic calls on already-consumed claims, then at most
one grounding candidate if separately released and justified; no action-SFT or
prompt-sweep escalation is authorized. Original score/run/archive/locks and
earlier restricted/public/fixture evidence remain unchanged.
## 2026-09-30 semantic-policy CPU preparation: no model-quality result

At frozen source `779e49883570371ce6223cb2b4a8619df5924d3b`, the sole CPU
job31698106 completed in 123 s (105.408 CPU s, 1,929,304 KiB batch MaxRSS).
The [preparation report](SCIFACT_SEMANTIC_POLICY_CPU_20260930.md),
[compact](verified-runs/scifact-semantic-preparation-31698106.json) and
[resource/package receipt](verified-runs/scifact-semantic-preparation-resources-31698106.json)
retain exact source/input/prompt hashes. Twelve consumed components exclude
23 eligible IDs before fixed matching; twelve selected questions fill three
per legacy stratum. Thirty selected-only fixture packing probes confirm equal
initial contexts and unchanged opportunity strata; no reselection occurred.

All 531 eligible queries were already exposed to legacy gold-aware preparation.
The twelve selected train questions are not an independent test. Model calls
were zero and dev/test was unopened in this CPU preparation; the fresh two-arm
96-slot model comparison was then unreleased. Sixty-four targeted tests and clean-archive reproduction,
Ruff and six-file strict mypy passed. Neither fixture tool use nor schema success
demonstrates real model behavior. The frozen F/G negative result below remains
unchanged; no current-resume or Agent-quality promotion is supported.

The [follow-on execution freeze](SCIFACT_SEMANTIC_EXECUTION_CPU_20260930.md) at
`44b0424abcfed65e4efdeeaf793f0d47ad92f734` adds new policy-specific runner/operator/
scorer entrypoints and exact private inference/scoring bundles. Eighty targeted
tests pass, including clean-source reproduction; strict mypy checks nine source
files. New inference is bound to the already-fixed selected-claim hash; scoring
independently binds original private gold/strata hashes. Both policies use gap
envelopes, separate preflight costs and full raw-wire audits. Genuine abstentions
and failure-empty predictions remain distinct. This implementation/package step
submitted no Slurm job, made no model call and supplies no quality gain or GPU release.

The release candidate is superseded by the minimal completion-journal repair
`99cd9ff707697ea395a9bc067f3ce91395071cdb`, with unchanged preparation, selected
claims, prompts and budgets. Five focused tests also pass from its exact-byte
clean archive. Known usage survives in a failed report, and one summary write
is attempted without another model call. If both writes fail, the partial report
exists only in process memory; after process exit durable cost is unavailable/
unknown, never zero. The [repair handoff](verified-runs/scifact-semantic-execution-repair-99cd9ff.json)
pins the replacement source and private bundle identities. Old artifacts remain
historical CPU evidence, not an alternative release candidate.

After the coordinator's separate exact-hash release, the sole submission created
jobs **31706518 / 31706520**, with the second `afterok:31706518`. The
[submission receipt](verified-runs/scifact-semantic-pair-submission-31706518-31706520.json)
records both real IDs, the retained exclusive lock, test-only admission and actual
resource requests. At the bounded snapshot the first was running on A100-short,
the second pending dependency; both request one A100 / eight CPU / 32 GiB / two
hours without requeue. No model-quality, completed-preflight or completion claim
follows from submission. No dev/test, automatic retry or new monitor was released.

## 2026-09-30 bounded grammar and evidence-gap pair: audited negative result

[Versioned preparation](SCIFACT_BOUNDED_GAP_CPU_20260930.md) preserves the frozen
r2 negative outcome and repairs invalid generation prefixes without rewriting
answers. It records actual full-sentence/preview hashes, separates proposals
from executed tools, and refuses incomparable initial contexts. The 96-slot
F/F+G protocol is frozen as two serial batches. Full local CPU suite: 587 passed/
one Windows POSIX skip; final Linux CI with CPU Torch: 588 passed/zero skips.
Seventeen pinned-tokenizer synthetic cases and two exact-token-set differential
probes passed. After separate coordinator release, [jobs 31686182 and 31686183](verified-runs/scifact-bounded-pair-submission-31686182-31686183.json)
completed 48 slots per arm with identical initial contexts. The [CPU-only
posthoc audit](SCIFACT_BOUNDED_POSTHOC_CLOSEOUT_20260930.md) and
[compact](verified-runs/scifact-bounded-pair-posthoc-31692980.json) verify hashes,
durable slots, full wire, score recomputation and independent official integers.
All 96 decisions were format-valid, with zero repairs; neither adaptive arm
proposed or executed a model-selected tool. Adaptive abstract-rationalized F1
.1000/.1081 reflects the same 2/9 correct gold documents, not a tool benefit;
gap-aware fixed retrieval scored .1212. Strict whole-answer diagnostics remain
2/12 per adaptive arm, distinct from official document credit. The biased train
sample is not independent dev/test. No Agent-quality promotion, online SLA or
resume improvement is supported. One import-path infrastructure failure and
its single successful CPU retry remain explicit in the cost ledger.

## 2026-09-30 SciFact train diagnostic: no Agent-quality promotion

The [r2 closeout](SCIFACT_TRAIN_DIAGNOSTIC_CLOSEOUT_20260930.md) verifies48/48 slots
on12 gold-stratified eligible-train components, not independent dev/test. Original
abstract-rationalized F1 is .1053/.0833/.1053/.0952 for fixed retrieval/rerank/
deterministic-extra/adaptive. All routes correctly rationalized1/9 gold documents;
adaptive produced no raw or successfully parsed tool intent. Of107 generation
attempts,87 failed strict validation, mainly cross-document sentence-budget
violations. These failures and all tokens remain in the accounting. Full gold
text was visible in several failed slots, so this is not merely a recall miss.

The [compact](verified-runs/scifact-train-r2-closeout-31645005.json) contains only
counts, hashes and content-free diagnostics, independently checked against the
frozen score. Gold, raw responses and traces stay on Spartan. Post-tool feedback
behavior is unobservable, not proven incapable or zero-error; no significance,
generalization, online SLA or resume benefit is claimed. Job31620529 remains a
separate pre-inference startup failure. Original300-dev remains unreleased.

## 2026-09-30 CPU-only identity and terminal compatibility

The [frozen cross-dataset identity audit](CROSS_DATASET_IDENTITY_AUDIT_20260930.md)
reports no target matches/connections under its fixed lexical/source rules;
missing mappings prevent an independence claim. The separate [SciFact document
terminal](SCIFACT_DOCUMENT_TERMINAL_CPU_20260930.md) supports per-document labels
and exact original sentence IDs with 24 synthetic contract fixtures. Neither
package runs a model, releases SciFact dev, changes the frozen official-compatible
scorer or supplies a new resume-quality metric. V3 job31601753 remains separate.
The [CPU statistical preparation](SCIFACT_PAIRED_STATISTICS_CPU_20260930.md)
adds complete-matrix checks and 5,000 paired component resamples with charged
failures and unknown-usage boundaries; 23 synthetic fixtures are not dev scores.

## 2026-09-30 frozen Agent comparison: completed, negative quality result

Job31543304 completed the frozen96-slot validation and24-slot authored-vNext
matrices using the5,240-document public corpus, BM25 and optional fixed4B
reranking; dense retrieval was disabled. This is neither the1,208,827-document
LoRA experiment nor a new frozen-test evaluation. All120 generated responses
have recorded token usage:0 mechanically accepted answers,105 schema failures,
14 genuine model abstentions and1 exact-quote rejection. There were no legal
model-requested rewrite/rerank actions or adaptive extra tool executions.

The frozen scorer's failure-gated delivered-evidence metrics use24 evidence
claims; label delivery uses23 non-DISPUTED decisive labels. Fixed-rerank versus
fixed-retrieval Recall@5 delta is0.0694 with a5,000-sample95% interval containing
zero; adaptive delta0.0417 also includes zero. Semantic supportability remains
unmeasured, not100% because zero answers were emitted. Authored-task quality
metrics remain null. Posthoc context diagnostics cannot replace these scores.

All192,026 input and21,361 output generation tokens, including failures, count.
The local artifact is content-free; original model responses and per-row
predictions remain private on Spartan. See the [full closeout](BUDGET_AGENT_FULL_CLOSEOUT_20260930.md),
[compact aggregate](verified-runs/budget-agent-full-31543304.json) and
[resource/audit receipt](verified-runs/budget-agent-full-resources-20260930.json).
No current resume was edited or independent-test/online-Agent benefit inferred.

## 2026-09-27 correction: ranking depth, integrity and timing

This notice supersedes historical field labels, not the immutable run artifacts.
Public-v2 and restricted five-stage downstream saved Top-5 predictions while
calling their ranking fields MRR@10/nDCG@10 (and sometimes Recall@10/50).
Their Recall@5 and F1@5 remain usable for valid base/ranking runs; MRR can be
interpreted as MRR@5, but nDCG cannot be renamed because IDCG still used ten.
These old fields/intervals are not complete Top-10 evidence. Exact source
commits and corrected commands are recorded in [D9](DECISIONS.md#d9--audit-ranking-depth-and-checkpoint-values-before-model-selection).
The separate restricted 20-step LoRA full gate retained Top-50, so its MRR@10,
nDCG@10 and F1@50 remain correctly scoped. Never compare its F1@50 directly
with downstream F1@5. Fixture classifier scores are not live verdict quality.

- Real public checkpoint integrity passed in CPU job `31364439`: 392 tensors
  restored exactly and fixed query/document embeddings changed repeatably;
  [compact record](verified-runs/adapter-integrity-20260927.json). This is not
  an adapter quality improvement or proof of every historical pilot's cause.
- Clean code reproduction at `97b2dc8`: 106 passed, one Torch test skipped
  locally; source mypy/Ruff passed. Seven real PEFT tests passed on Spartan.
  See [runtime record](verified-runs/search-tradeoffs-reproduction-20260927.json).
- Same-query full-ranking profiling job `31364586` completed in 1,334 s,
  exit 0:0. Public validation LTR / Top-20 / Top-100 R@5 is
  .6054 / .5948 / .6275, F1@5 .3828 / .3829 / .3969, measured E2E P95
  77.8 ms / 1.91 s / 9.30 s. All R@5/F1@5 paired intervals cross zero;
  low-latency LTR is a practical offline recommendation, not a noninferiority
  result or deployment. [Full record](verified-runs/search-tradeoffs-20260927.json).
  Old sums of stage P95 values remain cost proxies, not request P95.


## 2026-09-03 representation-evaluation closeout

| Record | Status | Boundary |
|---|---|---|
| Public frozen-test policy | `verified-consumed-test` | CLIMATE-FEVER test was consumed on 2026-08-25 by `bm25-lexical-baseline-v1`; this closeout produced no new candidate test score and the validation gate was not run |
| Historical split post-hoc audit | `strict-warning / decisive-pass` | exact source SHA `8a4b9032...`; zero claim/shared-ID/exact-text leaks, one 0.913-Jaccard train/test document variant; both annotations NEI, so decisive-evidence variant count is zero |
| New v2 grouped split | `verified-public-data-preparation` | 1,535 claims/5,240 unique documents, 1,075/230/230 split; shared IDs, normalised/near claims and normalised/near evidence variants all have zero cross-partition leaks; no test model was run |
| Base/adapted comparison contract | `verified-code` | exact query/corpus/candidate-universe/width/cutoff/data hashes plus >=5,000 paired bootstrap; fixture tests are not quality evidence |
| LTR Top-K reachability correction | `verified-code; quality-pending` | feature rows now equal the serving-width RRF pool; unreachable positives are counted and excluded. Historical job `29504398` is not relabelled as a result of this code change |
| Query taxonomy | `verified-diagnostic-code` | entity, numeric/year, geographic, lexical mismatch, semantic inference, multi-evidence and unanswerable; deterministic heuristic slices, not human labels |
| Historical search Pareto record | `retired-component-cost-proxy` | Original values remain for audit, but sums of component P95 values are not request P95; the CLI now refuses this config for new Pareto decisions. The 2026-09-27 public comparison supplies independently timed E2E instead |
| Iris isolated CPU preflight | `verified` | job `29926197`, exact SHA `09d2524`, 15 targeted tests passed in 7 s, exit `0:0`, batch MaxRSS `53,244 KiB`; public code/fixtures only, no GPU or frozen-test evaluation |

Compact closeout: `docs/verified-runs/representation-evaluation-closeout-20260903.json`.

## Current repository

| Evidence | Status | Verification |
|---|---|---|
| Package and CLI workflows | `verified-software` | Clean GitHub clone at `97b2dc8`: 106 passed, 1 Torch-dependent skip, Ruff and strict mypy (31 source modules); historical test counts below identify old runs, not the current total |
| Same-candidate LTR / bounded 4B profiles | `verified-public-validation-only` | Job `31364586`, inference SHA `b797f66`, 126 queries/5,240 docs and identical Top-100 pools; all Top-5 paired intervals cross zero. LTR recommended for latency; Top-100 retained as optional offline quality candidate. No public adapter promotion, test reopening, verdict score or online SLA |
| BM25, hash dense exact search, RRF, hard negatives | `verified` | deterministic unit tests |
| Claim-grouped Qwen3-Embedding-0.6B InfoNCE/LoRA adaptation | `verified-offline-dev-promotion` | data job `29460405`, 20-step training `29462754`, injection preflight `29463845`, sampled screen `29463846`, full-corpus gate `29465819`; all 154 official-dev claims/1,208,827 documents; Recall@5 `0.2793→0.2970`, MRR `0.3633→0.3869`, nDCG@10 `0.2994→0.3203`, F1@50 `0.07253→0.07544`; all 5,000-sample paired intervals positive; offline dev, not independent test/online A/B |
| Pairwise LTR fallback and LightGBM persistence | `verified-code` | 48-test project environment; commit `636e915` fixes persisted LightGBM feature metadata; effectiveness requires a valid candidate-supported run |
| Qwen3 encoder and FAISS FlatIP adapter | `verified-build` | full 1,208,827-vector Spartan build plus fixed-dev effectiveness run completed |
| FAISS HNSW/IVF-PQ | `verified` | jobs `29418470`/`29418595`; fixed 154-query FlatIP-grounded quality-speed comparison |
| Retrieval/end-to-end metrics | `verified` | synthetic fixtures with exact expected values |
| Bootstrap and calibration utilities | `verified` | seeded unit tests |
| Five-stage orchestration | `verified-smoke` | synthetic fixture with deterministic reranker |
| Public CLIMATE-FEVER adapter and fixed split | `verified-public-data-contract` | upstream SHA `8a4b9032...e0b`; 1,535 claims/7,675 annotations/5,240 unique evidence; seed `20260825`; train/validation/test 1,075/230/230; zero shared-evidence and normalised-claim cross-split leakage |
| Public frozen-test BM25 baseline | `verified-public-external-retrieval-baseline` | 129 test claims with decisive evidence: Recall@5/10/50 `0.4571/0.5490/0.7182`, MRR@10 `0.4567`, nDCG@10 `0.4221`; compact record `docs/verified-runs/climate-fever-public-bm25-test-20260825.json` | Retrieval only; no cross-encoder or verdict model, so do not report claim accuracy or end-to-end RAG quality |
| Public retrieval v2 adapter matrix | `historical-inconclusive-adapter/no-promotion` | Array `30005221` observed six 64-query ties; diagnostic `30007124` failed integrity, so no valid full adapter result exists. Base-only `30007546` measured 126 decisive validation claims: RRF/4B Recall@5/F1@5 `0.6275/0.3969`. Historical Top-10 fields are censored. The original compact JSON is retained unchanged; fixed-probe restoration passed on 2026-09-27 without a quality rerun. Frozen test and SciFact remain closed |
| Grounded FastAPI service | `verified-code-and-fixture` | `/api/search`, `/api/verify`, trace, metrics and health tests; invalid citation IDs/quotes, provider failure and missing verifier all abstain | Model Studio credentials are not configured in the local environment; live Qwen3.7-Plus external-test verdict metrics are not yet verified |
| Full 1,208,827-document BM25 index | `verified` | Spartan job `29360715`; commit `a7b110e`; 40.333 s total, 126,334,728-byte artifact, Slurm MaxRSS 2,630,496 K |
| Full Qwen3 embeddings + FAISS FlatIP reference index | `verified-build` | Spartan job `29382416`; commit `2cd75e3`; 1,208,827 × 1,024-d vectors; 1,696.767 s total; 5,133,839,551-byte dense artifact; MaxRSS 22,583,288 K |
| Qwen3-Embedding-4B sampled size gate | `verified-negative-resource-gate` | job `29458425`, commit `ced5e32`: evidence-preserving 5,000-document/eight-claim screen at 1,024 dimensions; Recall@5 `0.925` vs 0.6B `0.950`, F1/MRR tied; `7.21` vs `50.95 docs/s`; peak Torch GPU bytes `17.42 GB` vs `3.17 GB`; no full rebuild |
| HNSW quality-speed result | `verified` | Recall@5 vs Flat `0.9961`; batch QPS `3,060.64`; P50/P95 `12.88/15.41 ms`; 5,280,336,294-byte index |
| IVF-PQ quality-speed result | `verified-negative` | 66,211,820-byte index and batch QPS `8,436.04`, but Recall@5 vs Flat only `0.3688`; rejected by quality gate |
| Fixed-dev RRF retrieval result | `verified` | job `29435589`, 154 claims, `final_k=5`: Recall@5 `0.2709` vs BM25 `0.1721`; Evidence F1 `0.1785` vs `0.1168`; 5,000-sample paired intervals exclude zero; compact hashes/metrics in `docs/verified-runs/five-stage-fixed-dev-20260819.json` |
| Qwen3-Reranker-0.6B pure-replacement result | `verified-negative` | job `29448904`, commit `01a571f`: 7,700 RRF Top-50 pairs; Recall@5/F1 `0.2438/0.1573` vs RRF `0.2709/0.1785`; paired intervals vs RRF cross zero; P50/P95 `4.88/5.88 s` per query |
| RRF + Qwen3-0.6B weighted-rank fusion | `verified-dev-selection/cutoff-corrected` | job `29452723`, commit `18faf6a`: selected 4:1 rank fusion; Recall@5 `0.2890` vs `0.2709` (delta interval `0.0005–0.0389`); F1@5 interval crosses zero; old Top-10 fields are not full-depth evidence; measured stage P50/P95 `5.57/6.52 s` |
| Qwen3-Reranker-4B pure replacement | `verified-inconclusive` | job `29453918`, commit `c04e39c`: 154 claims/7,700 pairs; Recall@5/F1 `0.3054/0.1997`, but paired intervals versus RRF cross zero; P50/P95 `4.20/4.82 s` |
| RRF + Qwen3-4B balanced weighted-rank fusion | `verified-dev-selection/cutoff-corrected` | jobs `29453918`/`29455049`: Recall@5/F1@5 `0.3153/0.2131`; both 5,000-sample paired intervals versus RRF above zero. Legacy MRR@10/nDCG@10 used Top-5 lists; not complete Top-10 results. Weights and size selected on the same dev, not independent test |
| Qwen3-Reranker-8B pilot Pareto gate | `verified-negative-pareto-gate` | job `29456898`, commit `53a3782`: same 8 claims/400 pairs as 4B pilot; tied F1/Recall@5 `0.3016/0.4688`, MRR `0.5042` vs 4B `0.5104`, P95 `8.25 s` vs `5.13 s`; full 8B intentionally not run |
| Candidate-agreement cost-aware 4B router | `verified-partial-dev-gate/not-selected` | job `29479185`, commit `95f72a5`: five-fold cross-fit, 43.51% strong calls, estimated mean `1.865 s/query`, Recall@5/F1 `0.2878/0.1909`; paired improvements vs RRF but only 38.05%/36.00% of always-4B gain preserved, below the 80% gate |
| Text-aware cost router | `verified-negative-dev-gate` | job `29479261`, commit `2ff8cf4`: five-fold cross-fit, 14.29% strong calls, Recall@5/F1 `0.2718/0.1788`; paired intervals vs RRF cross zero; rejected |
| Legacy LightGBM LambdaMART fixed-dev result | `invalidated-training-set` | Recall@5 `0.0029`, but the training builder injected unretrieved gold evidence with zero retrieval features; retain only as failure forensics, not a model-quality result |
| Legacy LTR + deterministic reranker result | `invalidated-upstream` | Recall@5 `0.0127`; downstream of the invalid legacy LTR candidates and explicitly not Qwen3 |
| Candidate-supported LambdaMART correction | `verified-negative-dev-gate` | Job `29484697`, commit `023ed9b`: 1,169 train groups/26,626 rows and 3,246 reachable positives; training pairwise accuracy `0.9529`, but fixed-dev Recall@5/F1 collapsed to `0.0075/0.0059` versus RRF `0.2709/0.1785`; both paired intervals versus BM25 were below zero. This is a valid negative result, not the earlier invalidated training set |
| RRF-prior, serving-width LTR correction | `verified-top5-rank-position-dev/cutoff-corrected` | Job `29504398`, commit `b47e437`: 1,169 groups/120,146 rows; RRF→4:1 RRF/LambdaMART MRR@5 `0.3446→0.3648`, paired interval `[+0.0032,+0.0390]`. Legacy field named MRR@10 used only five ranks; censored nDCG@10 is not a full-depth result. Recall@5/F1@5 `0.2709/0.1785→0.2801/0.1824` but intervals cross zero. CPU scoring P95 `7.80 ms` is one component, not E2E. Original compact record remains in `verified-runs/rrf-prior-ltr-fusion-gate-20260822.json` |

## Historical Group 045 records

| Record | Status | Boundary |
|---|---|---|
| 1,208,827 evidence passages | `verified` | counted by full BM25 index artifact; data not redistributed |
| 1,228 train claims | `project-record-only` | final notebook output; not re-counted by the BM25 build |
| BGE/BM25-top-1000 Recall@5 0.223 | `project-record-only` | separate experiment, not reproduced here |
| Notebook dev F 0.1763 / Accuracy 0.6234 / H-mean 0.2749 | `project-record-only` | exact local notebook output |
| Rounded F 0.19 / Accuracy 0.61 / H-mean about 0.29 | `project-record-only` | training document; different/rounded run |
| Public rank 5 and snapshot metrics | `unsupported` | no stable official artifact located; removed |

## Resume rule

Before reporting an improvement, retain the data/split hash, metric definition, model/index parameters, hardware/runtime/cost, Git commit, paired comparison, error cases, and team-versus-individual boundary. Fixture scores only prove scorer behavior.

The `29382416` artifact manifest correctly records the job, Git SHA, environment, and restricted input hash, but its start and finish timestamps are identical because the old writer created both at artifact-write time. Runtime claims therefore use `metrics.json` and Slurm accounting. The follow-up code records command start time explicitly.

The dense encoder size gate is recorded in `docs/verified-runs/qwen3-embedding-4b-pilot-20260820.json`. Job `29458425` completed in `14 min 16 s` with exit `0:0` and Slurm MaxRSS `27,017,012 K`. It retains all gold evidence in the sampled corpus but evaluates only eight claims, so it supports the decision not to spend resources on a full 4B rebuild; it does not prove full-corpus 4B effectiveness.

The retained-0.6B adaptation chain is recorded in `docs/verified-runs/qwen3-embedding-lora-sampled-gate-20260821.json` and `docs/verified-runs/qwen3-embedding-lora-full-gate-20260821.json`. Data job `29460405` resolved 13,354/13,354 evidence IDs and produced a claim-grouped split; training job `29462754` completed 20 LoRA/InfoNCE steps; preflight `29463845` verified 5,046,272 injected adapter parameters. Sampled gate `29463846` used 126 held-out claims, 5,000 of 1,208,827 documents and all 368 labelled positives; Recall@5, MRR and nDCG intervals were positive, while Evidence F1 crossed zero, so no sampled-only promotion was made. The first full-corpus job `29464119` finished encoding/search but failed while persisting a rebuildable 4.95 GB adapted FlatIP index and produced no usable metrics. Commit `c815070` made index persistence opt-in. Replacement `29465819` completed at commit `c81507084f3b5355c29b0ed0351dc85672a0c619` in `49 min 37 s` with exit `0:0`, Slurm MaxRSS `22,890,736 K`, and `0.827 L40S-hours`. It evaluated all 154 official-dev claims, 463 required evidence rows and 1,208,827 documents with 5,000 paired samples. Recall@5 changed `0.2793→0.2970` (95% interval `0.0014–0.0350`), MRR `0.3633→0.3869` (`0.0060–0.0432`), nDCG `0.2994→0.3203` (`0.0090–0.0341`), and F1@50 `0.07253→0.07544` (`0.00120–0.00482`). The adapter passes the offline official-dev promotion gate. The successful run saved zero adapted-index bytes; restricted predictions, checkpoint, corpus and large base artifacts remain on Spartan. This is not independent test generalisation or an online A/B result.

The fixed-dev run manifest records job `29435589`, commit `636e9159a14a59248ce8c4e93c396370c4af508e`, dev-claim hash `ea9976e8...`, config hash `8ae39fb2...`, `final_k=5`, 5,000 bootstrap samples, and `deterministic-feature-fallback`. It is a retrieval-only experiment: the classifier is unconfigured, and Recall@10/50 are not separately interpretable because only five final documents were emitted.

The local Qwen3 reranker run manifest records job `29448904`, commit `01a571fbee549e4ecff7fec7eaf7fa6c076bd623`, the same dev-claim hash, config hash `4d2fbab...`, RRF as the reranker base, 154 queries, 7,700 pairs, and `final_k=5`. The paired RRF-vs-Qwen comparison is job `29449093` at commit `2de8293`. Restricted predictions and model cache remain on Spartan.

The corrected fusion run manifest records job `29452723`, commit `18faf6aa59b593ae7e362c6decdc031d6ef96575`, 154 queries and 7,700 pairs. Comparison job `29453474` bootstrapped every Qwen/fusion prediction file against the same RRF predictions. The selected `base4` profile is a rank-level fusion, not a learned score calibration: its 4:1 weights were evaluated alongside balanced and 2:1 profiles on the fixed dev split, so they are not an independent test-set hyperparameter claim.

The 4B run manifest records job `29453918`, commit `c04e39c9fb4983f010623ba14d0c8cf9ac371edd`, dev-claim hash `ea9976e8...`, config hash `64aa8c1d...`, BF16 inference, 154 queries and 7,700 pairs. Comparison job `29455049` evaluated pure, balanced, 2:1 and 4:1 outputs against the same RRF predictions with 5,000 paired samples. Balanced fusion was best on this dev split and therefore carries a `verified-dev-selection` boundary rather than an independent test label. Restricted predictions and the 4B model cache remain on Spartan.

The 8B resource gate is recorded in `docs/verified-runs/qwen3-reranker-8b-pilot-20260820.json`. Job `29456898` used node-local model caching after a storage-only failed attempt, finished with exit `0:0`, and did not beat the 4B pilot on quality/latency. No full-dev 8B result exists or is claimed.

