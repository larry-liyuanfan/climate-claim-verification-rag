# Climate Evidence Retrieval and Grounded Verification

The [verifier-input isolation and CPU diagnosis](docs/SCIFACT_SEMANTIC_INPUT_ISOLATION_20261002.md)
separates scientific evidence from controller budgets without changing the old
protocol. Whole-chain review preceded CPU job **31963271**: **1,440 arm/counter-state
checks** (including reference states), 120 clean-history checks and 72
recorded-episode replays passed in 16 s, with
zero new model calls. All 18 claim-document annotations for the consumed cohort's 15
positives were retrieved and visible; label/rationale selection, then incomplete
multi-document reading, are the observed next repair targets. This is input
invariance and diagnostic evidence, **not a demonstrated model-quality gain**.
Old negative results and original scoring remain unchanged.

The [once-frozen conditional TRAIN24 confirmation](docs/SCIFACT_PROSPECTIVE24_CONFIRMATION_20261002.md)
completed **72/72 slots in 300 s** after integrated pre-submission review.
Complete evidence-bearing answers were **5/15 top1, 3/15 fixed-all, 5/15 adaptive**.
Adaptive matched top1's positive answers but had lower abstract-rationalized
F1 (**0.5714 vs 0.6061**) and **5.16 times its tokens**. Its six correct explicit
NEI decisions have a different interface from fixed unresolved outputs and are
not proof of a fair reasoning advantage. Keep this negative promotion decision:
no tuning/resampling rerun, independent-test claim, deployment or resume gain.
The report preserves all costs and three explanatory paired cases. The
[whole-chain submission checklist](docs/SCIFACT_PROSPECTIVE24_CPU_20261002.md#whole-chain-review-before-submission)
requires integrated review and exact-archive preflight before queue submission,
not serial component repairs discovered after each queue wait.

The [prospective conditional TRAIN24 CPU package](docs/SCIFACT_PROSPECTIVE24_CPU_20261002.md)
adapts once-frozen metadata and gold-free initial frames to the unchanged policy;
CPU job **31953981 completed in 38 s**, with zero model calls or new real-gold
access. It adds no model result, independent-test claim or GPU authorization.

The earlier, separately authorized [evidence-commit confirmation](docs/SCIFACT_EVIDENCE_COMMIT_CONFIRMATION_20261002.md)
completed **72/72 slots in 312 s** after whole-chain CPU review. Immutable verifier
references and deterministic assembly recovered strict evidence-bearing answers:
**3/15 fixed-top1, 3/15 fixed-all, 4/15 adaptive** on the same exposed TRAIN24.
The model proposed and executed 30 verifications; six episodes continued after
INSUFFICIENT feedback. One paired case correctly chose `c1` instead of `c0` or
all positive documents. This is exercised document selection, not universal
Agent gain: adaptive abstract-rationalized F1 **0.4444 < top1 0.4848**, and its
222,945 tokens cost **5.30× top1**. It used 12.5% fewer calls than fixed-all but
39.3% more tokens. Verification is mandatory before commit; every actual commit
had only one legal positive subset. Correct model NEI, fixed unresolved states,
wrong evidence and costs are reported separately. No independent test, resume
gain, automatic rerun or deployment is claimed. The
[CPU protocol and receipts](docs/SCIFACT_EVIDENCE_COMMIT_20261002.md) remain preserved.

The [whole-chain-reviewed bounded-decoder confirmation](docs/SCIFACT_DOCUMENT_BOUNDED_CONFIRMATION_20261001.md)
completed as [job 31930176](docs/verified-runs/scifact-document-bounded-closeout-31930176.json)
in 391 s: **all 25 prior structural rejections were removed**, but strict positive
whole-answer correctness remained **0/15 in both routes**. Fixed correct NEI
remained 3/9; adaptive remained 0/9 and chose no verification in 24 opportunities.
All 96 fixed verifier responses are byte-identical to the previous run, and all
144 paired inputs/prompts/schemas match. Correct rationalized documents increased
4→10 (fixed) and 2→6 (adaptive), alongside many more predicted documents; this
is not recovered whole-answer grounding or autonomous Agent benefit. Actual
cost was 144 generations / 340,743 input / 10,031 output tokens. Keep the proven
output-constraint repair; the remaining issue is evidence selection/integration,
not JSON formatting. These are the same exposed TRAIN24 diagnostics, not a new
validation/test result; no current resume metric is changed.

The [single-document verifier diagnostic](docs/SCIFACT_DOCUMENT_VERIFIER_20261001.md)
completed as [job 31918065](docs/verified-runs/scifact-document-verifier-closeout-31918065.json)
with **48/48 recorded episodes and a negative result**: strict positive grounding
was 0/15 in both fixed and adaptive routes; correct NEI was 3/9 versus 0/9.
The adaptive model made no verify proposal or call in its 24 actual first
responses. Fixed verification consumed 120 generations versus 24 adaptive
generations, with every token accounted for; lower cost without correct answers
is not Agent efficiency. The original visible evidence, unadapted model, scorer
and shared limits were frozen. These are already-exposed TRAIN24 diagnostics,
not validation, independent test or online improvement. R1's separate pre-model
startup failure is preserved; no automatic next retry, training or resume gain
is authorized by this result.

Whole-chain review traced **25 rejected terminals** to missing generation-time
document uniqueness/20-sentence constraints (20 sentence-budget failures and
5 duplicate-document first failures). All 96 verifier judgments were structurally
valid, which does not establish correctness. The stage-aware decoder candidate
reuses the existing bounded prefix only for planning/final answers; single-document
judgments retain ordinary LMFE. The confirmation above is a separate implementation
comparison, not an infrastructure retry or a demonstrated whole-answer gain.
See the [integrated review and
release gate](docs/SCIFACT_DOCUMENT_VERIFIER_20261001.md#whole-chain-review-before-submission).

The [utility8 closeout](docs/verified-runs/scifact-utility8-closeout-31834687.json)
preserves a negative A/B/C diagnostic on eight already exposed TRAIN cases.
Job 31834687 completed 24/24 valid terminal slots, but strict correctness was
**0/1/0 of eight**; the single success was an NEI abstention. Scripted read/rerank
routes did not improve complete-rationale grounding. Actual unique cost was
24 generator calls / 81,509 input / 3,007 output tokens, not 40 calls after
double-counting the shared prefix. **This is not autonomous Agent success.**

A [mixed program-artifact bridge and separate trainer entry](docs/SCIFACT_MIXED_TRAINING_BRIDGE_20261001.md)
now preserve old48/supp49 source records and distinguish NEI47 from five context
abstentions. [CPU preparation job 31854794](docs/verified-runs/scifact-mixed-cpu-closeout-31854794.json)
completed **144/144 ready / 223 rows / 36 planned updates**, with no failed or
unknown claims. Raw-byte factories and physical readback preserve the fixed
records, token packing and equal claim mass. All three output hashes and the
atomic completion marker were rechecked; records remain private on Spartan.
The separately released [training job 31858295](docs/verified-runs/scifact-mixed-training-closeout-31858295.json)
then completed all **144 claims / 223 rows / 36 updates exactly once** in 394 s.
The final 144 FP32 LoRA tensors passed hash, shape and finite-value checks;
observed training loss 0.284 is **not retrieval or grounding quality**.
The [paired tune job 31865294](docs/verified-runs/scifact-mixed-tune-closeout-31865294.json)
completed **24/24 valid calls**, with all 144 saved LoRA tensors reloaded and
physical costs audited before gold. On the same already-exposed 12 TRAIN-internal
terminal cases, rationalized-document F1 rose **0.040 → 0.333** (only 1 → 2 correct
documents), while sentence-selection recall fell **9/17 → 2/17**. NEI false-evidence
cases fell 4/4 → 0/4, but the adapter also returned no evidence for **5/8 evidence-bearing
claims**. This is a precision/coverage trade-off, not an independent-test or Agent
gain. No validation promotion follows automatically. After this closeout, the
coordinator separately released the unchanged four-route candidate.
[Job 31871386 completed](docs/verified-runs/scifact-mixed-four-route-closeout-31871386.json)
48/48 slots in 189 s: fixed retrieval / fixed rerank / deterministic extra / adaptive
strict correctness was **3/5/4/3 of twelve** (each includes three correct NEI abstentions).
Adaptive proposed **zero tools**, and sufficient-evidence grounding failures remain.
This is a negative autonomous-policy result, not Agent success; utility8 remains unchanged.
A [bounded evidence-note closeout](docs/SCIFACT_EVIDENCE_NOTE_20261001.md) reuses all
twelve fixed-rerank frames, with **no new baseline calls**. Job 31884581 made
12 note + 11 terminal calls; one capped note without EOS correctly skipped its
terminal call. Strict correctness remained **5/12**, while generation time grew
**11.729 → 93.081 s**. Rationalized F1 rose only through fewer positive predictions,
not more correct evidence. This negative quality/cost result is not promoted.

The [natural FIT24 protocol](docs/SCIFACT_NATURAL_FIT24_20261001.md) freezes 24
components once from 136 eligible exposed FIT144 components. Execution job
31895661 has now completed. Original scoring and the separate read-only audit report
72/72 valid terminal slots: A/B/C strict correctness **0/7/3 of 24**; all ten B/C
successes are NEI abstentions, not recovered positive evidence or autonomous tool
use. A selected no tools. The [completed CPU posthoc](docs/verified-runs/scifact-natural-posthoc-closeout-31904484.json)
finds that **14 of 15 evidence-bearing claims already had complete initial visible
evidence, but none was answered strictly correctly**. Neither scripted tool made
an incomplete claim complete. The next implementation decision is terminal
evidence-selection/label/abstention improvement under the same visible evidence,
not more retrieval or tool-policy training. See [provenance and limitations](docs/SCIFACT_NATURAL_POSTHOC_20261001.md).
No training or validation promotion follows automatically from this exposed-TRAIN
negative result.

A [CPU-only captured-state training seam](docs/SCIFACT_CAPTURED_STATE_TRAINING_DRIVER_20261001.md)
now preserves stable rerank aliases and captured prompts, validates an external
roster, and tests complete-claim mean updates with the actual cohort size.
That original seam still has no real-data loader and retains its CPU <=1M
synthetic-model guard. The separate mixed-input entry does not weaken it.

A [synthetic physical-receipt adapter](docs/SCIFACT_OBSERVATION_RECEIPTS_20261001.md)
audits utility8 attempt/frame/tool-event lineage and deduplicates shared calls
without dropping incomplete slots. Accepted observations are not supervision
targets or evidence of Agent improvement; no real outputs were imported.

An [annotation-bound terminal candidate seam](docs/SCIFACT_TERMINAL_SUPERVISION_20261001.md)
now distinguishes official-format NEI from missing positive context, checks full
OR rationales and preserves captured prompts. It is validated only on synthetic
annotations; the subsequent program-channel NEI47 preparation is reported below.
No model training is implied.

The [NEI47 CPU preparation implementation](docs/SCIFACT_NEI47_PREPARATION_20261001.md)
adds a separate **program_capture** channel for the 47 already exposed official
NEI cases inside frozen supplemental96. It uses actual BM25/CommonPacking with
zero real model calls, not fabricated physical receipts. The fixed-input entry,
failure accounting and exact-source packaging are synthetic-tested; **no real
model or training is invoked by that path**. The subsequent authorized
[CPU job 31843230](docs/verified-runs/scifact-nei47-cpu-closeout-31843230.json)
completed **47/47 ready** with no gaps/failed/unknown cases. This is preparation
of already exposed TRAIN examples, **not a trained model or quality gain**.

## Grounding adaptation: controlled tune signal, not yet Agent improvement

[FIT-only production-state supervision](docs/SCIFACT_FIT_STATE_SUPERVISION_20261001.md)
repairs the training-data contract using the original 48 FIT claims, complete
official alternative sets, real controller observations and equal per-claim
loss mass. This CPU preparation is **not a new trained model or quality gain**;
private annotations remain on Spartan and reserved validation stays untouched.
Single CPU job 31789062 produced **93 answer records / 48 equal-weight claims**,
with no gaps. All claims already had an initial complete witness; **zero natural
read examples** arose. It is grounding-data readiness, not Agent-policy
readiness, and no subsequent training or held-out evaluation was launched.

The [metadata-only pool audit](docs/SCIFACT_POOL_METADATA_AUDIT_20261001.md)
shows why this is not a model improvement: the one matched saved read case moves
from rank 7 / preview-only in the full corpus to rank 1 / citable in the FIT
pool, with identical source/sentence hashes. Pool membership and BM25 statistics
changed together. Unowned documents are not proven safe background; no pool
expansion, extra TRAIN cohort or new training was released by that audit.

A separately approved [shared-corpus CPU protocol](docs/SCIFACT_SHARED_CORPUS_PROTOCOL_20261001.md)
now freezes the original 48 FIT claims and changes only the retrieval pool to
all 5,183 public documents. It explicitly drops source-family-unseen claims and
separates indexing, retrieved candidates, prepared-prompt text and supervision
targets. The one CPU comparison completed as
[job 31831831](docs/verified-runs/scifact-shared-cpu-closeout-31831831.json):
45 initial complete witnesses, **one natural read adding 19 citable sentences**
and a complete post-read rationale, plus two context-limited abstains. The 93
prepared records have equal per-claim mass and no gaps. This limited teacher
coverage is **not a trained Agent result**; no model training or held-out
evaluation followed, and all negative/no-read cases remain in the private audit.

The prospective [supplemental FIT package](docs/SCIFACT_SUPPLEMENTAL_FIT_20261001.md)
selected 96 new components using metadata-only SHA ordering, before reading their
annotations. CPU job **31832344** completed in 108 s: 47 official NEI claims were
retained as unsupported, while 49 supported claims produced 46 direct answers
and three context abstains—**zero additional natural reads**. Its 83 records do
not mean 83 independent claims. No replacement sampling was performed, and the
original 48 control is unchanged. This negative action-coverage result is not a
reason to claim Agent improvement or automatically train the enlarged cohort.

A separate [claim-group-mean optimizer contract](docs/SCIFACT_CLAIM_GROUP_MEAN_20261001.md)
passes synthetic CPU gradient/AdamW checks while preserving historical v2 `/48`.
It is an interface validation, not a new trainer, adapter or Agent-quality result.

[Stage B](docs/SCIFACT_GROUNDING_CANDIDATE_CPU_20261001.md) implements a
protocol-bound consumption ledger, whole-component/source-family separation,
official alternative-aware targets, and a single-config LoRA training / paired
non-oracle evaluation entrypoint using the production citation contract.
Real-tokenizer CPU preparation passed with frozen 48/12/12 TRAIN components,
96 official alternative-aware fit records and no packing gaps. Both evaluation
pools contain only 58 source-partitioned documents; this is a controlled grounding
experiment, **not a full-corpus retrieval result or an independent test**.
One separately authorized run, **31757970**, completed 96 records / 24 updates
and one final LoRA checkpoint; its 144 tensors passed finite/shape/hash audits.
The separately released 12-query tune pair completed: correctly rationalized
documents increased **1 to 4**, but NEI false evidence stayed **4/4**. A
[physical posthoc audit](docs/verified-runs/scifact-grounding-structure-20261001.json)
found single-document `c1` targets in every FIT record and `c1`-only adapted tune
answers. This controlled TRAIN-internal signal does **not** establish multi-document
selection, retrieval or autonomous tool use. The original gate and negative
findings remain unchanged; the frozen validation has **not** been called.

[The four-route regression](docs/SCIFACT_ADAPTER_REGRESSION_CPU_20261001.md)
completed job **31773502**: same active adapter, 48 slots / 48 calls over twelve
previously exposed TRAIN queries, including one FIT-overlap query. Correctly
rationalized documents were **1 / 4 / 3 / 1** for fixed retrieval / fixed rerank /
deterministic extra / adaptive. Adaptive offered tools in all twelve slots but
made **zero tool proposals or executions**. One valid NEI abstention is separate
from the one correct document. All 47 nonempty outputs selected one document
and one sentence; this regression is **not** literal-`c1`-only. Fixed rerank and
deterministic extra always selected the first actually visible document.
This is a retained negative Agent result, not an independent-test improvement.

[A three-state conditional continuation](docs/SCIFACT_READ_CONTINUATION_CPU_20261001.md)
reconstructed all three original scripted-read states exactly, including physical
prompt hashes and **1,868 / 1,648 / 2,528** input tokens. CPU readiness passed;
Torch and POSIX imports were verified on Spartan. This is preparation, not model
quality: **zero model calls during CPU preparation**. A separate exact release
then authorized three calls only; job **31781591 completed**. Its
[physical closeout](docs/SCIFACT_READ_CONDITIONAL_CLOSEOUT_31781591.md) found correct
document labels in 3/3 answers but complete rationales in only **1/3** (FIT 0/1,
not-direct-FIT 1/2). Both failures selected no annotated rationale sentence,
despite a complete single-sentence alternative being visible. These documents
were selected by the scripted/oracle read, **not by model tool use**. This is
limited conditional grounding capacity, not retrieval or autonomous Agent gain;
the original negative regression remains unchanged. No validation was consumed.

## Latest component diagnostic: complete, with grounding errors

[Job 31743877 closeout](docs/SCIFACT_COMPONENT_CLOSEOUT_31743877.md): **36 new + one
carried invocation**, all **33 formal outputs contract-valid**, zero unknown
costs. Screening selected **54/60** candidate occurrences (all candidates in
**9/12** inputs), so selecting all six available annotated golds does not establish
effective discrimination. Gold-document relation was **5/9** correct; rationale
first3 coverage was **7/9 only when given the correct relation**. The twelve claims
were already-consumed TRAIN diagnostics, not an independent test or Agent gain.
Raw/gold remain on Spartan, the old negative preflight is retained, and no new
training or model calls were performed during physical closeout.

### Earlier r2 diagnostic: stopped at semantic preflight

Job **31734906** completed at the scheduler level, but the application stopped:
one synthetic response was technically valid `abstain`, whereas the fixture
expected `select [1]`. All **33 formal diagnostics were unattempted**, not model
failures or zero accuracy. The [terminal report](docs/SCIFACT_COMPONENT_CLOSEOUT_31734906.md)
preserves physical hashes, **277 input / 15 output tokens**, and both jobs' total
**177 allocated GPU-job seconds**. Real-tokenizer CPU checks show the tested
active and abstention paths are reachable; the cause of model abstention is not
established. No model/Agent improvement or independent-test result is claimed.

A [separate CPU-prepared continuation](docs/SCIFACT_COMPONENT_CONTINUATION_CPU_20261001.md)
keeps that negative result immutable, separates technical readiness from semantic
measurement and references the one paid response without repeating it. It allows
at most 36 new calls only after a separate release; no new model result is claimed.

Subsequent authorized submission: [job 31743877](docs/verified-runs/scifact-component-continuation-submission-31743877.json)
was submitted once on **2026-10-01 02:26:52 Australia/Sydney**, using execution
source `a104115`, after a successful scheduler test-only check. At **02:27:27** it
was `PENDING (Resources)`; the then-estimated 02:41:52 start was not guaranteed.
No new model result is available in this submission snapshot. The one old call
stays paid and semantically negative; no repeated submission or monitor was created.

### Predecessor failure and validated CPU repair

Job **31729507 failed before provider construction**: canonical JSON persistence
reordered input object fields, so runtime prompt hashes no longer matched Stage A.
The original [submission snapshot](docs/verified-runs/scifact-component-submission-31729507.json)
is historical; the [failure/repair report](docs/SCIFACT_COMPONENT_PACKING_REPAIR_20261001.md)
records **zero generation calls but 81 allocated GPU-job seconds**.

Source `81917f2` restores the original authored field order, without changing
values, document order, frozen expected hashes or budgets. The existing real
tokenizer verifies **33/33 complete packing identities** on both preparation and
runtime tokenizer asset paths; four synthetic persistence seams also pass.
Local and exact-archive clean-source regression each pass **151 tests**, with
one Windows-only POSIX skip. These are infrastructure checks, not model quality.
The CPU repair itself submitted no GPU job. The subsequent separately authorized
`r2` submission above preserves the failed v1 provenance and the same 37-call ceiling.

### Validated CPU preparation (historical checkpoint)

The [versioned provider/operator](docs/SCIFACT_COMPONENT_EXECUTION_CPU_20261001.md)
now connects these frozen inputs to a single-attempt 4+33 call plan, private
wire/cost journaling and a separate post-exit scorer. A local decoder compatibility
fix restores tested legal numeric-array paths without changing the original
prompt/schema; invalid IDs still fail the canonical contract, with no retry.
Those checks establish **CPU preparation**, not a component model result. The
subsequent single-job release above does not turn software tests into accuracy.

The [Stage A preparation contract](docs/SCIFACT_COMPONENT_PREPARATION_20261001.md)
separates document screening, document relation and oracle-conditioned rationale
selection on the same twelve already-consumed claims. Full abstracts, independent
schemas, alternative-aware scoring, source/prompt identities and durable failure
costs are implemented: 52 targeted tests pass in an exact-source clean checkout.
Real-tokenizer preparation fits all 33 inputs; frozen screening candidates cover
6/9 gold documents, with the remaining three recorded as candidate gaps.
No real component model evaluation,
new training/split or Agent improvement is claimed. The real provider is only
enabled within the separately released single-attempt job above.

## Latest bounded diagnosis: completed, not Agent improvement

The [semantic-policy closeout](docs/SCIFACT_SEMANTIC_CLOSEOUT_20260930.md) verifies
both completed jobs and all 96 TRAIN slots. Both adaptive policies made zero
tool proposals/executions. Correctly rationalized adaptive documents stayed at
one; F1 .0444 → .0513 reflects fewer predictions, not newly discovered evidence.
The [compact](docs/verified-runs/scifact-semantic-pair-closeout-31706518-31706520.json)
retains 32 independently matched official metric groups, real token/resource
cost and visible-evidence failure layers. No independent-test or Agent benefit
is claimed. The [next-step proposal](docs/SCIFACT_GROUNDING_FEASIBILITY_20260930.md)
prioritizes document/relation/rationale grounding diagnosis; no new model job,
training or resume change is authorized by that document.

### Earlier preparation/submission checkpoints (historical)

The [semantic-policy CPU preparation](docs/SCIFACT_SEMANTIC_POLICY_CPU_20260930.md)
completed in job **31698106** with zero model calls. It excludes all 23 eligible
IDs in the 12 previously consumed components, then fixes twelve train questions
for a fresh original-G/semantic-policy comparison with equal initial contexts.
This verifies exposure accounting and packing contracts, **not model improvement**;
all 531 eligible questions already had gold-aware preparation exposure. CPU
success alone did not authorize GPU or dev/test execution; the negative F/G result remains.

The subsequent [semantic execution handoff](docs/SCIFACT_SEMANTIC_EXECUTION_CPU_20260930.md)
freezes separate inference/scoring bundles and fresh two-policy entrypoints with
80 passing CPU seam/regression tests. Gold isolation, cost journaling,
policy-bound preflights and serial predecessor checks are implemented. The
`99cd9ff` pre-release repair also passes five focused clean-source tests: a
completion-journal failure retains known usage in a failed summary and stops;
if summary persistence also fails, durable cost is unavailable/unknown, not zero. This is
**CPU preparation only**: zero model calls, zero submitted jobs and no new quality result.

After a separate exact-hash release, the [serial comparison submission](docs/verified-runs/scifact-semantic-pair-submission-31706518-31706520.json)
created jobs **31706518 → afterok → 31706520**. Its historical snapshot showed the
first running and the second waiting on its dependency. Both subsequently
completed as recorded in the closeout above; no dev/test execution was authorized.

The [bounded generation / evidence-gap CPU preparation](docs/SCIFACT_BOUNDED_GAP_CPU_20260930.md)
adds generation-time uniqueness/total-budget enforcement, actual preview hashes,
charged error feedback and a preregistered two-arm comparison. Synthetic CPU
checks passed. The released pair **31686182 → afterok → 31686183** completed;
the [read-only posthoc audit](docs/SCIFACT_BOUNDED_POSTHOC_CLOSEOUT_20260930.md)
verified all 96 slots, source/wire hashes and independent official score counts.
All decisions were format-valid, but both adaptive arms selected **zero tools**.
Adaptive abstract-rationalized F1 was .1000/.1081 (both only 2/9 correct gold
documents); the gap-aware fixed-retrieval baseline scored .1212. This is a
**biased 12-claim train diagnostic, not evidence of Agent benefit or a holdout
improvement**. Tokens and the failed/successful CPU audit allocations are retained;
no current-resume change or new model run follows.

The [SciFact train diagnostic closeout](docs/SCIFACT_TRAIN_DIAGNOSTIC_CLOSEOUT_20260930.md)
records completed job **31645005**, all48 slots and independently checked score
counts. This is a **biased12-claim train diagnostic, not a holdout**. All four
routes correctly rationalized only1/9 gold documents; adaptive selected zero tools
even after inspecting its unparsed wire outputs. Most failures exceeded the
cross-document sentence budget. Full evidence was visible in several failed
cases, so retrieval hits alone do not establish answer quality. No Agent-quality
promotion, dev/test release or current-resume change follows. The failed startup
job31620529 and its [source-archive repair](docs/SCIFACT_STARTUP_REPAIR_20260930.md)
are preserved separately.

2026-09-30 follow-up: a separately authorized [feedback-v2 development pilot](docs/AGENT_FEEDBACK_V2_DEVELOPMENT_20260930.md)
adds action-discriminated schemas and charged validation feedback; it is not a
reinterpretation of the frozen result below. CPU validation and CI passed; the
single [pilot job31587302](docs/BUDGET_AGENT_FEEDBACK_V2_CLOSEOUT_20260930.md)
completed18 slots:15 final abstentions,3 repair exhaustions,0 accepted answers
and0 model-selected tools. This is a negative development result. No new holdout run,
training, grammar-decoding deployment or current-resume change is implied.

Separately, the [original SciFact CPU preparation](docs/SCIFACT_GROUNDED_PROTOCOL_PROPOSAL_20260930.md)
preserves sentence rationales, quarantines train/dev source-family overlap and
cross-checks a new scorer against the official reference. The retry above is
train-only; its300 dev claims remain unreleased, not a new test claim.
The [cross-dataset CPU identity audit](docs/CROSS_DATASET_IDENTITY_AUDIT_20260930.md)
found no matches or graph connections to consumed Climate tracks under fixed
lexical/source rules. Missing source mappings remain unknown; this is not proof
of independence, and no old test text or labels were reopened.
The separate [SciFact document-terminal CPU contract](docs/SCIFACT_DOCUMENT_TERMINAL_CPU_20260930.md)
supports document-specific labels and ordered original-sentence references.
The [local Qwen/LMFE provider](docs/SCIFACT_LOCAL_PROVIDER_CPU_20260930.md) is
implemented with CPU/mocked-HF validation and subsequently exercised in the
train-only diagnostic above. Dev evaluation remains unreleased; synthetic tests
are not model-quality evidence.
The [paired-component statistical preparation](docs/SCIFACT_PAIRED_STATISTICS_CPU_20260930.md)
recomputes micro F1 and mean costs with 5,000 paired group resamples; it has only
synthetic fixtures, no real-data quality result or external-dev release.

A separately versioned [sentence-ID v3 CPU package](docs/AGENT_SENTENCE_ID_V3_CPU_20260930.md)
adds explicit candidate reading, actual-visible-sentence citation constraints,
budgeted prompt assembly and opt-in grammar decoding. Fixtures/tokenizer smoke
are not model-quality evidence. The [frozen job31601753 closeout](docs/AGENT_V3_PILOT_CLOSEOUT_20260930.md)
contains 24 complete slots on six already-consumed authored prompts. Mechanical
answer acceptance was 4/6 retrieval, 5/6 rerank, 6/6 deterministic-extra and 4/6
adaptive; **adaptive selected zero tools**, and duplicate-reference repairs made
no progress. An irrelevant empty-query rewrite also produced a mechanically valid
answer, showing why these counts are not scientific accuracy or Agent gains.
External-dev evaluation remains unreleased; raw responses stay on Spartan.

2026-09-30: the [frozen full Agent comparison](docs/BUDGET_AGENT_FULL_CLOSEOUT_20260930.md)
completed in job **31543304**, but did **not** demonstrate Agent quality: all
120 task/route slots ran; **0 answers passed mechanical checks**, 105 outputs
failed schema validation, 14 were model-requested abstentions and one was
rejected for an inexact quote. No model-directed rewrite/rerank executed.
Validation retrieval-effect intervals include zero. The original frozen scores,
failed-output token costs and five sanitized cases are retained in the
[content-free compact](docs/verified-runs/budget-agent-full-31543304.json).
This repeated validation/authored-task comparison is separate from the earlier
restricted LoRA development gains and public search tradeoffs below. It does
not reopen frozen test, justify deployment or support an autonomous-Agent claim.

2026-09-29: [budgeted evidence-driven Agent CPU package](docs/BUDGET_AGENT_CPU_HANDOFF_20260929.md)
adds a source/constraint ledger, bounded rewrite/rerank decisions, cited-answer
validation and fixed-chain comparison runners. The [first real-model pilot](docs/BUDGET_AGENT_GPU_PILOT_20260929.md)
loaded both4B models and generated9 responses, but **all9 failed structured-response
validation** before any accepted action. It is a negative integration result, not
a RAG quality gain. A separately released diagnostic canary also rejected9/9;
in-place replay of the original strings through the frozen validator confirmed
non-rewrite actions illegally carrying `query`. The minimal model-visible
cross-field contract repair has now been checked in
[confirmation31520350](docs/BUDGET_AGENT_CONFIRMATION_20260929.md): **2/9 responses
passed mechanical answer checks, while7/9 still violated the action/query
contract**. None was a valid model-requested abstention; no adaptive rewrite or
rerank executed. This is a negative protocol-compliance result, not semantic
accuracy or Agent-quality improvement. A frozen-protocol offline comparison is
technically interpretable with failures retained, but remains separately gated
on coordinator release. That full comparison has since completed with the
negative result above; the pilot's 2/9 is not its success rate. The CPU heuristic
control remains negative.

The [offline scoring audit](docs/BUDGET_AGENT_SCORING_AUDIT_20260929.md) now requires
complete frozen task/route matrices and gold identities, preserves partial token
accounting, and separates model abstention from controller rejection/failure.
The confirmation was scored separately on CPU using frozen208ff93; inference
remains72eaa90. Original decoded responses stay on Spartan; only the reviewed
[compact receipt](docs/verified-runs/budget-agent-protocol-confirm-31520350.json)
is published. Official pilot retrieval/verdict quality is null, not zero.

The [serial full operator](docs/BUDGET_AGENT_FULL_OPERATOR_20260929.md) was separately
released on2026-09-30 and submitted once as **31542525 (initially PENDING/Priority)**;
see the [submission receipt](docs/verified-runs/budget-agent-full-submission-20260930.json).
It failed during runtime preparation after39s; both inference phases remained
not started. The [CPU diagnosis and minimal repair](docs/BUDGET_AGENT_FULL_FAILURE_20260930.md)
identify a `./lib/...` tar-prefix mismatch, not a model/scoring failure.
A subsequent explicit coordinator release authorized exactly one repair rerun:
**31543304**, initially PENDING/Resources with no scheduled start; see the
[r2 submission receipt](docs/verified-runs/budget-agent-full-r2-submission-20260930.json).
No automatic retry or extra experiment is authorized. Job31543304 subsequently
completed in701s, exit0:0; final results and resource accounting are in the
[closeout](docs/BUDGET_AGENT_FULL_CLOSEOUT_20260930.md). Initial submission
receipts remain historical snapshots. The operator shares one read-only
input/model extraction across separate frozen validation and authored-vNext
processes, with phase-specific receipts and fail-closed scoring.

> 2026-09-27 audit: public-v2 and restricted five-stage Top-10 metrics were censored by Top-5 prediction
> storage; base-only downstream Recall@5 and Evidence F1@5 are unaffected. The public adapter
> loader is repaired and real checkpoint/output-effect checks passed on Spartan.
> This proves restoration, not better retrieval. The three-route validation
> profile is complete: LambdaMART is the recommended low-latency candidate;
> Top-100 4B retains the highest Top-5 point estimates at a much higher cost.
> See [measured tradeoffs, uncertainty and reproduction](docs/SEARCH_TRADEOFFS_20260927.md).

A reproducible search-and-ranking extension of the **2026 COMP90042 Group 045 team project**. The course system used BM25 candidate retrieval, BGE bi-encoder reranking, and a LoRA-tuned claim classifier. This repository now separates two evidence tracks: a restricted 1.21M-document scale/selection track on Spartan and a public CLIMATE-FEVER external benchmark that can be reproduced without course data.

```text
claim normalisation + entity/year constraints
  ├─ BM25 lexical recall
  └─ Qwen3 dense/HNSW recall
        ↓
      RRF → Qwen3-4B cross-encoder
        ↓
      evidence sufficiency / one bounded re-retrieval
        ↓
      structured verdict provider
        ↓
      validated citations or explicit abstention
```

The repository does **not** claim an official leaderboard rank. Restricted course data, raw predictions, and private checkpoints are not redistributed.

## What is implemented

| Layer | Implementation | Truth boundary |
|---|---|---|
| Lexical retrieval | Deterministic inverted-index BM25 with the course tokenizer and trusted-artifact persistence | Full 1,208,827-document Spartan build verified; retrieval quality evaluation remains separate |
| Dense retrieval | Deterministic hash smoke encoder; Sentence Transformers adapter with reusable, ID-hashed embeddings | Full 1,208,827-document Qwen3 build verified; a 20-step hard-negative LoRA adapter passed the full-corpus offline official-dev promotion gate; hash mode is not a semantic model |
| ANN | NumPy exact IP plus FAISS FlatIP, HNSW, and IVF-PQ adapters | Full-corpus fixed-query comparison verified; HNSW retained as the quality-speed default, IVF-PQ rejected by the quality gate |
| Fusion/LTR | RRF; LightGBM LambdaMART when installed; deterministic linear pairwise fallback | Historical restricted LTR failures remain recorded; the separately trained public Top-100 LTR is the measured low-latency candidate, not a production rollout |
| Reranking | Configurable 0.6B/4B/8B local Qwen3 model, Alibaba Model Studio adapter, deterministic feature fallback | 0.6B exposed first-stage replacement failure; 4B plus balanced rank fusion improved fixed-dev Recall@5 and Evidence F1@5; an 8B pilot failed the latency/quality Pareto gate, so 4B remains the offline quality profile |
| Public benchmark | CLIMATE-FEVER adapter, evidence-aware near-duplicate split and frozen-test BM25 baseline | 1,535 claims/7,675 annotations; final test is not used for model selection |
| Evaluation | Recall@K, hit rate, MRR@10, nDCG@10, evidence P/R/F1, verdict Macro-F1/Accuracy, citation quality, ECE/Brier and paired bootstrap | Retrieval and verification results remain separately labelled |
| Confidence | Temperature scaling, coverage-risk and selective abstention utilities | Requires provider or classifier confidence |
| Serving | FastAPI search/verify/trace/metrics endpoints with bounded re-retrieval | Verifier failure, invalid citation IDs or quote mismatch fail closed to `NOT_ENOUGH_INFO` |
| Provenance | Input hashes, Git SHA, environment, metrics, predictions, error cases, report | Generated for every CLI run |

## Quick start

### Agent-consumable evidence, without claiming an autonomous Agent

The optional **LangChain Core 1.6.5** integration uses a real `BaseRetriever`,
source-bearing `Document` objects, an LCEL sequence and a Pydantic `StructuredTool`.
Its manual CPU demo ran against the existing **5,240-document public corpus**;
it does not repeat frozen-test evaluation or load a verdict model. Source IDs,
corpus/text hashes and exact text spans survive the adapter; an empty result
remains `no_evidence`, never a fabricated refutation.
See [the reproducible application handoff](docs/AGENT_APPLICATION_HANDOFF_20260927.md)
for commands, cases, framework boundaries and the separately measured search tradeoff.

### Existing retrieval CLI

The basic install below is for the retrieval CLI, not the complete optional-model
test suite. Use the explicit Windows Torch or Linux/POSIX interpreter in
[local validation environments](docs/LOCAL_VALIDATION_ENVIRONMENTS.md) for tests;
`.[test]` alone does not install Torch/PEFT.

```bash
python -m pip install -e ".[test]"

climate-rag index \
  --evidence fixtures/evidence.json \
  --backend both \
  --output-dir artifacts/smoke-index

climate-rag evaluate \
  --claims fixtures/claims.json \
  --predictions fixtures/predictions_candidate.json \
  --baseline-predictions fixtures/predictions_baseline.json \
  --output-dir runs/smoke-eval

climate-rag prepare-public \
  --output-dir data/climate-fever-public \
  --seed 20260825

climate-rag audit-public-split \
  --prepared-dir data/climate-fever-public \
  --output-dir runs/climate-fever-split-audit
```

## Public retrieval v2

The public-v2 cycle was pre-registered in
[`configs/public_retrieval_v2.json`](configs/public_retrieval_v2.json). It uses
only the 1,075-claim train partition for hard-negative LoRA/InfoNCE training and
the 230-claim validation partition for pilot/full selection. The old consumed
test is permanently sealed. Six fixed Qwen3-Embedding-0.6B adapters cover
100/300 steps, rank 8/16, 4/8 hard negatives and temperatures 0.03/0.05; no more
than two can reach full validation.

All six 64-query pilots historically tied the base (Recall@5 `0.53203125`).
Those ties do not establish adapter ineffectiveness: weight restoration was not
verified, and historical Top-10 metrics used Top-5 prediction lists. The one
pre-registered full diagnostic then exposed missing adapter keys and did not
produce a valid promotion result. No adapter was promoted, no quality retry was
run, and the SciFact event was not authorised; its qrels were never opened.
The 2026-09-27 repair subsequently proved checkpoint restoration and output
changes on fixed label-free probes, not a new quality gain.

The validation-only base closeout compared BM25, base dense Flat/HNSW, fixed
RRF, Top-100 LambdaMART and the fixed 1:1 RRF/Qwen3-4B fusion. The fusion reached
Recall@5/Evidence F1@5 `0.6275/0.3969` over 126 decisive validation claims.
Historical MRR@10/nDCG@10 and Recall@10 were censored by the Top-5 output and
are not complete Top-10 estimates. These are same-validation selection metrics,
not adapter gains or independent-test evidence. The complete method, no-promotion decision,
archive hashes and truth boundaries are in
[`docs/PUBLIC_RETRIEVAL_V2.md`](docs/PUBLIC_RETRIEVAL_V2.md).
The compact record is
[`docs/verified-runs/climate-public-retrieval-v2-20260904.json`](docs/verified-runs/climate-public-retrieval-v2-20260904.json).

### Measured search-model choice, not just a component list

On the same 5,240 documents and 126 decisive **public validation** queries,
each route used the same ordered Top-100 BM25/HNSW/RRF pool. Full rankings now
support genuine Top-10 metrics; evidence F1 is explicitly Top-5.

| Route | Recall@5 | F1@5 | Warm serial E2E P95 | Peak Torch allocated |
|---|---:|---:|---:|---:|
| LambdaMART | .6054 | .3828 | 77.8 ms | 2.25 GiB |
| RRF + 4B Top-20 | .5948 | .3829 | 1.91 s | 10.54 GiB |
| RRF + 4B Top-100 | .6275 | .3969 | 9.30 s | 11.07 GiB |

**Recommendation:** LTR for latency-sensitive use; keep Top-100 4B as an
optional offline quality profile. Top-20 did not establish a useful benefit
over LTR. The 5,000-draw paired Recall@5/F1@5 intervals cross zero for all
three comparisons; no equivalence or significant reranker win is claimed.
LTR scoring is CPU-based, but this complete path still encodes queries on GPU.
These are single-process warmed measurements, not HTTP SLA, billed cost, an
independent test or an adapter-quality gain. Details, taxonomy, stage timings
and hashes: [decision report](docs/SEARCH_TRADEOFFS_20260927.md) and
[compact evidence](docs/verified-runs/search-tradeoffs-20260927.json).

The candidate fixture is intentionally perfect and tests only the scorer: Recall@5, Evidence F1, Accuracy, and H-mean are `1.0`. These are **not** climate fact-checking quality metrics. The deliberately flawed fixture baseline has Recall@5 `0.50`, Evidence F1 `0.50`, Accuracy `0.75`, and H-mean `0.60`. The full representation-training case and two evidence-grounded resume bullets are in [`docs/REPRESENTATION_TRAINING_CASE.md`](docs/REPRESENTATION_TRAINING_CASE.md).

## Main commands

Build BM25:

```bash
climate-rag index --evidence /data/evidence.json --backend bm25 --output-dir /artifacts/bm25
```

Build Qwen3 embeddings and FAISS FlatIP, retaining a validated reusable vector cache:

```bash
climate-rag index \
  --evidence /data/evidence.json \
  --backend dense \
  --encoder sentence-transformer \
  --model Qwen/Qwen3-Embedding-0.6B \
  --ann flat \
  --embeddings /artifacts/qwen3-0.6b.npy \
  --output-dir /artifacts/dense-flat
```

The cache sidecar stores encoder, dimension, document count, and ordered document-ID hash. A mismatch fails closed. HNSW and IVF-PQ reuse the same embeddings through the example configs.

Mine hard negatives and train fusion:

```bash
climate-rag mine-negatives \
  --claims /data/train-claims.json \
  --rankings /artifacts/train_rankings.jsonl \
  --limit 100 \
  --ltr-candidate-width 100 \
  --output-dir /artifacts/hard-negatives

climate-rag train-fusion \
  --features /artifacts/ltr_features.jsonl \
  --algorithm auto \
  --output-dir /artifacts/ltr
```

The retained 0.6B dense encoder is improved through task adaptation rather than
blind parameter scaling. `scripts/prepare_embedding_training.py` converts mined
training negatives into the current ms-swift Qwen3-Embedding InfoNCE format,
keeps every claim wholly in train or validation, removes gold/duplicate-text
false negatives, and emits hashed run artifacts. Spartan job `29462754`
completed a bounded 20-step LoRA/InfoNCE run, and preflight `29463845` verified
that 5,046,272 adapter parameters were injected into the serving encoder.

The claim-grouped sampled gate (`29463846`) retained all 368 labelled positives
for 126 held-out claims in a 5,000/1,208,827-document corpus. Recall@5 changed
from `0.6090` to `0.6303` (paired 95% interval `0.0048–0.0429`), while MRR and
nDCG intervals were also positive. Evidence F1 changed from `0.1087` to `0.1095`
with an interval crossing zero (`-0.0003–0.0026`), so that run remained only a
sampled screen.

The complete official-dev replacement gate (`29465819`) then evaluated all 154
claims and all 1,208,827 evidence passages. Recall@5 changed from `0.2793` to
`0.2970` (paired 95% interval `0.0014–0.0350`), MRR@10 from `0.3633` to
`0.3869`, nDCG@10 from `0.2994` to `0.3203`, and Evidence F1 from `0.07253` to
`0.07544`; all four 5,000-sample paired intervals were above zero. The adapter
therefore passes the pre-registered offline promotion gate. It is an
official-dev result, not independent test generalisation or an online A/B test.
Compact sampled and full-corpus records are published in
[`docs/verified-runs/qwen3-embedding-lora-sampled-gate-20260821.json`](docs/verified-runs/qwen3-embedding-lora-sampled-gate-20260821.json)
and
[`docs/verified-runs/qwen3-embedding-lora-full-gate-20260821.json`](docs/verified-runs/qwen3-embedding-lora-full-gate-20260821.json).

`auto` uses LightGBM LambdaMART when present. Otherwise it persists `linear_pairwise_ranknet_fallback`; it is never renamed LambdaMART. Candidate rows must be split by claim before held-out evaluation. LTR rows are generated from the exact RRF Top-K used by serving; positives outside that set are counted as unreachable instead of being injected with zero retrieval features.

Paired base/adapted comparison and train/serve consistency checks:

```bash
climate-rag evaluate-representation \
  --claims /data/dev-claims.json \
  --evidence /data/evidence.jsonl \
  --baseline-predictions /artifacts/base.json \
  --candidate-predictions /artifacts/adapted.json \
  --baseline-contract /artifacts/base-contract.json \
  --candidate-contract /artifacts/adapted-contract.json \
  --bootstrap-samples 5000 \
  --output-dir /artifacts/representation-pair

climate-rag audit-stage-contract \
  --training-contract /artifacts/training-contract.json \
  --serving-contract /artifacts/serving-contract.json \
  --output-dir /artifacts/stage-contract-audit

climate-rag build-pareto \
  --profiles /artifacts/same-query-measured-profiles.json \
  --output-dir /artifacts/search-pareto
```

The `configs/stage_contract.*.example.json` files contain labelled fixture
values only. Promotion requires contracts generated from measured artifacts.
The old `configs/search_profiles.verified.json` is a retired historical proxy,
not a measured request-P95 table; the CLI refuses it for new Pareto decisions.

Run the fixed five-stage comparison:

```bash
climate-rag evaluate \
  --claims /data/dev-claims.json \
  --experiment-config configs/five_stage.example.yaml \
  --output-dir /artifacts/five-stage
```

It evaluates `bm25`, `dense`, `rrf`, `ltr`, and `ltr_reranker` with one claim split and one `final_k`, then bootstraps each stage against BM25. The configured reranker name is recorded. Deterministic fallback results cannot be described as Qwen3 results.

Serve multi-stage retrieval and grounded verification:

```bash
climate-rag serve \
  --bm25-index /artifacts/bm25/bm25.pkl.gz \
  --dense-index /artifacts/dense-hnsw \
  --reranker qwen-local \
  --reranker-model Qwen/Qwen3-Reranker-4B \
  --verifier model-studio \
  --verifier-model qwen3.7-plus \
  --max-queries 2 \
  --host 0.0.0.0 --port 8000
```

The service exposes `POST /api/search`, `POST /api/verify`, `GET /api/traces/{trace_id}`, `GET /metrics`, and `GET /health`; legacy `POST /retrieve` remains for compatibility. Model Studio credentials are read only from environment variables. Without a configured verifier the service returns an explicit abstention rather than fabricating a verdict.

## Public external baseline

The public adapter pins the upstream CLIMATE-FEVER source hash, deduplicates 5,240 evidence passages, and keeps claims connected by shared evidence, normalised claim duplicates, high-similarity claim text, and exact/near-duplicate evidence text in the same partition. The v2 `70/15/15` split (seed `20260825`) contains 1,075/230/230 claims and passes the strict cross-partition claim/document-variant audit. It is a new split protocol, not a way to re-open the historical frozen test.

The first frozen-test run is deliberately only a lexical baseline. Among the 129 test claims with SUPPORTS/REFUTES evidence, BM25 reached Recall@5/10/50 `0.4571/0.5490/0.7182`, MRR@10 `0.4567`, and nDCG@10 `0.4221`. It does **not** establish verdict quality or cross-encoder gains. Exact hashes and boundaries are in [`docs/verified-runs/climate-fever-public-bm25-test-20260825.json`](docs/verified-runs/climate-fever-public-bm25-test-20260825.json).

That historical frozen test has already been consumed. A post-hoc strict audit
found one cross-partition near-document pair at 0.90 Jaccard; both annotations
are `NOT_ENOUGH_INFO`, and the decisive-evidence audit remains clean. The result
is retained as a historical lexical baseline, but every new candidate test is
blocked by `configs/public_evaluation_policy.json`. Model selection is
validation-only; this project cycle no longer has an unused public test for a
new independent claim.

## Data and artifacts

Claims use the course-compatible keyed JSON schema with `claim_text`, optional `claim_label`, and `evidences`. Evidence may be a keyed JSON object, list records, or JSONL. Installing `ijson` streams a keyed object; otherwise JSON is loaded into memory.

Every command writes `run_manifest.json`, `metrics.json`, `predictions.jsonl`, `error_cases.jsonl`, and `report.md`. Non-secret arguments, Git SHA, input hashes, environment, and Slurm job ID are recorded.

The restricted full corpus has been located on Spartan at:

```text
/data/gpfs/projects/punim2936/nlp/COMP90042_2026-main/data
```

It contains the 167 MB `evidence.json` and train/dev/test claim files. It is referenced only by Slurm/Apptainer configuration and must not be copied into GitHub. See [`hpc/README.md`](hpc/README.md).

### Verified Spartan BM25 build

The native-module run on 2026-08-18 produced a full-corpus lexical index. These are engineering measurements, not retrieval-quality scores:

| Field | Verified value |
|---|---:|
| Slurm job | `29360715` (`COMPLETED`, exit `0:0`) |
| Git commit | `a7b110e8d647e9e6f51272d02c2436d4a346a27c` |
| Evidence documents | `1,208,827` |
| Vocabulary | `531,996` |
| BM25 build time | `36.886 s` |
| Total command time | `40.333 s` |
| Slurm elapsed time | `51 s` |
| Serialized index | `126,334,728 bytes` |
| Slurm MaxRSS | `2,630,496 K` |

The restricted artifact remains under project storage at `climate-artifacts/bm25`; only its non-sensitive metrics and provenance are published.

### Verified Spartan Qwen3 dense and FlatIP build

The replacement job completed on 2026-08-19 after commit `2cd75e3` moved Hugging Face caches from the quota-limited home directory to project storage. These are full-corpus **index-build measurements**, not retrieval-effectiveness scores:

| Field | Verified value |
|---|---:|
| Slurm job | `29382416` (`COMPLETED`, exit `0:0`) |
| Git commit | `2cd75e3b2cc2af059a9880e9482d8e814c94cdc5` |
| Encoder | `Qwen/Qwen3-Embedding-0.6B` |
| Evidence vectors | `1,208,827` |
| Vector dimension | `1,024` |
| Dense/FlatIP build time | `1,694.743 s` |
| Total command time | `1,696.767 s` |
| Slurm elapsed time | `28 min 25 s` |
| Embedding cache | `4,951,355,520 bytes` |
| FAISS FlatIP index | `4,951,355,437 bytes` |
| Dense artifact total | `5,133,839,551 bytes` |
| Slurm MaxRSS | `22,583,288 K` (`21.54 GB` via `seff`) |
| Allocation | `1× H100`, `8 CPU`, `96 GB`; `0.474 H100-hours` wall allocation |

The cache sidecar records the model, dimension, document count, and ordered document-ID hash. The run manifest records job `29382416`, Git SHA, environment, and the restricted input hash. Its start/finish timestamps were identical because the original writer generated both at artifact-write time; the follow-up code fixes that provenance defect, so the verified duration above comes from `metrics.json` and Slurm accounting. The 4.95 GB embeddings, 4.95 GB index, and restricted corpus remain on Spartan.

The dense/FlatIP build itself produced no effectiveness result. The separately completed CPU-only ANN benchmark and fixed-dev run below provide the quality-speed and retrieval comparisons.

### Verified dense encoder size gate

Job `29458425` compared `Qwen3-Embedding-0.6B` with `Qwen3-Embedding-4B` at the same 1,024-dimensional output on an evidence-preserving 5,000-document sample. All 27 gold-evidence rows for the same eight claims were forced into the sample; this is a resource screen, not full-corpus retrieval evidence. The 4B candidate did not pass: Recall@5 changed from `0.950` to `0.925` (paired 95% interval `-0.075–0.000`), while MRR@10 and Evidence F1 tied. Document encoding fell from `50.95` to `7.21 docs/s`, and peak Torch GPU allocation rose from `3.17 GB` to `17.42 GB`. The retained full-corpus/latency-profile 0.6B index is therefore kept, and a full 4B rebuild was deliberately not submitted. This sampled gate supports that resource decision; it does not prove that 4B would be worse on the complete corpus. The compact record is in [`docs/verified-runs/qwen3-embedding-4b-pilot-20260820.json`](docs/verified-runs/qwen3-embedding-4b-pilot-20260820.json).

### Verified full-corpus embedding-adapter promotion gate

The retained 0.6B encoder was adapted instead of replaced. Job `29465819`
evaluated the 20-step hard-negative InfoNCE/LoRA checkpoint on all 154 untouched
official-dev claims, all 463 required evidence rows and the complete 1,208,827-
document corpus at 1,024 dimensions. It used commit `c815070`, one L40S, eight
CPUs and a 64 GB request, completed in `49 min 37 s` with exit `0:0`, reached
Slurm MaxRSS `22,890,736 K`, and consumed `0.827 L40S-hours` of wall allocation.

| Metric | Base 0.6B | Adapted 0.6B | Mean delta | Paired 95% interval |
|---|---:|---:|---:|---:|
| Recall@5 | `0.2793` | **`0.2970`** | `+0.0176` | `0.0014–0.0350` |
| MRR@10 | `0.3633` | **`0.3869`** | `+0.0236` | `0.0060–0.0432` |
| nDCG@10 | `0.2994` | **`0.3203`** | `+0.0210` | `0.0090–0.0341` |
| Evidence F1 | `0.07253` | **`0.07544`** | `+0.00291` | `0.00120–0.00482` |

The adapted corpus encoded in `2,888.48 s` (`418.50 docs/s`), built the in-memory
FlatIP reference in `1.86 s`, and recorded peak Torch GPU allocation
`25,243,138,560 bytes`. The first attempt had completed encoding/search but then
failed while persisting a rebuildable 4.95 GB adapted index under project quota.
The replacement saved `0` index bytes and retained only the small manifest,
metrics, report and restricted prediction artifact on Spartan. This operational
fix did not change the evaluation or its pre-registered gate.

### Verified Spartan ANN quality-speed comparison

Jobs `29418470` and `29418595` reused the same 1,208,827-row, 1,024-dimensional Qwen3 embedding cache. The benchmark used 154 fixed dev queries, 32 FAISS CPU threads, three batch-search repeats, and FlatIP as the ANN ground truth.

| Index | Recall@5 vs Flat | Batch QPS | Single-query P50 / P95 | FAISS index bytes | Decision |
|---|---:|---:|---:|---:|---|
| FlatIP | `1.0000` | `5.15` | `411.84 / 432.68 ms` | `4,951,355,437` | exact reference |
| HNSW (`M=32`, `efSearch=64`) | `0.9961` | `3,060.64` | `12.88 / 15.41 ms` | `5,280,336,294` | retained quality-speed default |
| IVF-PQ (`nlist=4096`, `nprobe=32`, `m=32`) | `0.3688` | `8,436.04` | `1.52 / 1.69 ms` | `66,211,820` | rejected: excessive recall loss |

The QPS values are batched, in-memory measurements on this fixed 154-query/32-thread run; they are not an online-service SLA. HNSW build time was `374.750 s` with batch MaxRSS `19,080 M`; IVF-PQ build time was `385.315 s` with batch MaxRSS `17,073,512 K`. Restricted indexes remain on Spartan.

### Verified fixed-dev retrieval comparison

Job `29435589` evaluated 154 restricted dev claims at `final_k=5` using commit `636e915`, the same BM25/HNSW candidate stores, and 5,000 paired bootstrap samples. The run was CPU-only (`32 CPU`, `32 GB` request), completed in `1 min 42 s`, and reached batch MaxRSS `11,596,892 K`.

| Stage | Recall@5 | MRR@10 | nDCG@10 | Evidence F1 |
|---|---:|---:|---:|---:|
| BM25 | `0.1721` | `0.2513` | `0.1644` | `0.1168` |
| Qwen3 dense/HNSW | `0.2696` | `0.3308` | `0.2487` | `0.1768` |
| BM25+dense RRF | `0.2709` | `0.3446` | `0.2495` | `0.1785` |
| Pure Qwen3-Reranker-0.6B replacement (first run) | `0.2438` | `0.3053` | `0.2180` | `0.1573` |
| RRF + Qwen3-0.6B weighted rank fusion (4:1) | **`0.2890`** | **`0.3801`** | **`0.2739`** | **`0.1905`** |
| Pure Qwen3-Reranker-4B | `0.3054` | `0.3763` | `0.2738` | `0.1997` |
| RRF + Qwen3-4B weighted rank fusion (1:1) | **`0.3153`** | **`0.3961`** | **`0.2849`** | **`0.2131`** |
| Legacy LambdaMART (invalidated training set) | `0.0029` | `0.0065` | `0.0030` | `0.0027` |
| Legacy LambdaMART + deterministic reranker (invalidated upstream) | `0.0127` | `0.0359` | `0.0149` | `0.0111` |

Against BM25, RRF improved Recall@5 by `0.0988` (paired-bootstrap 95% interval `0.0543–0.1452`) and Evidence F1 by `0.0616` (`0.0360–0.0893`). The first pure Qwen replacement run (`29448904`) regressed, exposing an architectural error: it discarded a strong first-stage order. Job `29452723` preserved that order with 0.6B weighted-rank fusion; its selected 4:1 profile improved Recall@5, MRR and nDCG, but not Evidence F1 with a stable interval.

The aggregate five-stage metrics and exact input/artifact hashes are published in [`docs/verified-runs/five-stage-fixed-dev-20260819.json`](docs/verified-runs/five-stage-fixed-dev-20260819.json). A later audit found that the legacy LTR training builder injected unretrieved gold evidence with zero retrieval features, so the two LTR rows above are retained as failure-forensics evidence, not model-quality evidence. The corrected candidate-supported job `29484697` removed that defect but remained a negative gate: 1,169 train groups/26,626 rows reached `0.9529` train pairwise accuracy, while fixed-dev Recall@5/F1 collapsed to `0.0075/0.0059` versus RRF `0.2709/0.1785`. The compact record is in [`docs/verified-runs/candidate-supported-ltr-gate-20260822.json`](docs/verified-runs/candidate-supported-ltr-gate-20260822.json). Commit `b47e437` then recorded the RRF prior, matched the 100-candidate training and serving widths and predeclared a 4:1 rank-preserving fusion. CPU job `29504398` completed 1,169 groups/120,146 rows and raised fixed-dev MRR@5 from RRF `0.3446` to `0.3648`, with a 5,000-sample paired interval `[+0.0032,+0.0390]`. The old `mrr@10` field used only five saved ranks and is reinterpreted as MRR@5; old nDCG@10 was also censored and is not retained as a complete Top-10 result. Recall@5/F1 rose to `0.2801/0.1824`, but their intervals versus RRF crossed zero. CPU feature scoring took P95 `7.80 ms/query`. This is retained as a low-latency rank-position profile, not the main quality profile; see the [compact record](docs/verified-runs/rrf-prior-ltr-fusion-gate-20260822.json). Restricted predictions and candidate lists remain on Spartan.

The model-size gate then ran Qwen3-Reranker-4B in BF16 on the identical 154-claim/7,700-pair split. Full job `29453918` completed in `12 min 48 s` on one A100 `1g.20gb` MIG slice (`8 CPU`, `32 GB` request; batch MaxRSS `20,372,008 K`). It recorded P50/P95 `4.20/4.82 s` per query. Pure 4B aggregate metrics rose but paired intervals versus RRF crossed zero. The selected 1:1 RRF/4B rank fusion reached Recall@5 `0.3153` and Evidence F1@5 `0.2131`. Its legacy Top-10 fields used Top-5 predictions; MRR can be interpreted only as MRR@5, while nDCG cannot be relabelled because its ideal denominator used ten. Comparison job `29455049` measured deltas versus RRF of `+0.0444` Recall@5 (95% interval `0.0163–0.0733`, `p=0.0024`), `+0.0515` MRR@5 (`0.0149–0.0883`), and `+0.0347` Evidence F1@5 (`0.0165–0.0535`). All values come from 5,000 paired bootstrap samples. Because the fusion weights and model size were selected on this same fixed dev split, these are dev-set model-selection results, not an independent test claim.

The optional 8B gate was also executed rather than left as a configuration claim. The first attempt (`29456746`) failed before inference because the shared project filesystem lacked room for the weight shards; four incomplete files totalling about 3.2 GB were removed, and commit `53a3782` moved the one-off cache to node-local ephemeral storage. Replacement pilot `29456898` completed in `2 min 19 s` on the same A100 `1g.20gb` MIG shape (batch MaxRSS `29,380,852 K`). On the same eight claims/400 pairs, the best 8B fusion tied 4B on Evidence F1 (`0.3016`) and Recall@5 (`0.4688`), was slightly lower on the archived truncated reciprocal-rank metric (`0.5042` vs `0.5104`; not claimed as complete MRR@10), and raised P95 latency from `5.13 s` to `8.25 s` (`+60.8%`). The full 8B run was therefore deliberately not submitted. This is a resource-selection gate, not a full-dev 8B quality result; the derived record is in [`docs/verified-runs/qwen3-reranker-8b-pilot-20260820.json`](docs/verified-runs/qwen3-reranker-8b-pilot-20260820.json).

Two RouteLLM-inspired cost-aware gates then tested whether the 4B path could be
called selectively. Both used deterministic five-fold hash cross-fitting, so a
claim's labels never trained its own route. The candidate-list-agreement router
called 4B on `43.51%` of queries and reduced the analytical mean latency estimate
to `1.865 s/query`; Recall@5/F1 were `0.2878/0.1909`, significantly above RRF,
but this preserved only `38.05%/36.00%` of the always-4B gain and was
significantly below the strong path. Adding inference-safe hashed claim-text
features avoided `85.71%` of calls but collapsed to RRF-level quality. Both fail
the predeclared 80% gain-preservation target and are not selected. The compact
record is in
[`docs/verified-runs/rerank-router-gates-20260821.json`](docs/verified-runs/rerank-router-gates-20260821.json).

HNSW+RRF remains the latency-oriented default; balanced 4B fusion is the measured offline quality profile. The RRF-prior LambdaMART fusion is an optional CPU rank-position profile because it improved MRR/nDCG but not Recall/F1 conclusively and remains below the 4B fusion's absolute quality. IVF-PQ and both cost routers remain rejected. The legacy LTR rows are invalidated; candidate-supported job `29484697` is a separate valid negative gate, while `29504398` records the bounded RRF-prior recovery. Because `final_k=5`, recorded Recall@10/50 equals Recall@5 and is not presented as a wider-cutoff result. Claim classification was not configured, so zero claim accuracy/H-mean values are not classifier findings.

The [paper-to-hiring map](docs/PAPER_TO_HIRING.md) connects Qwen3 model sizing
and adaptation, rank-preserving reranking, cost-aware routing and candidate-
supported learning-to-rank to exact code, jobs, evidence and stop conditions.

## Historical result boundary

Local Group 045 records contain non-equivalent evaluations:

- final notebook development output: Evidence F `0.1763`, Accuracy `0.6234`, H-mean `0.2749`;
- separate BGE/BM25-top-1000 experiment record: Recall@5 `0.223`;
- rounded training document: approximately Evidence F `0.19`, Accuracy `0.61`, H-mean `0.29`.

These are **historical project records**, not reproduced package results, and must not be mixed. An earlier README claimed a public rank and snapshot metrics without a stable official artifact; those claims have been removed.

## Attribution and publication boundary

- The 2026 COMP90042 submission was a **Group 045 team project**. Do not present the course pipeline, data, or team output as one person's independent work.
- This repository is a post-course portfolio engineering extension. Git history and the evidence ledger identify later additions.
- Course data, teammate identifiers, raw private predictions, checkpoints, and credentials are excluded.
- No open-source license is granted at present. Public visibility does not itself grant reuse rights.

## Technical references

- [Qwen3 Embedding technical report](https://arxiv.org/abs/2506.05176) and [official implementation](https://github.com/QwenLM/Qwen3-Embedding)
- [FAISS index families](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [LightGBM learning-to-rank parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html#learning-to-rank-parameters)
- [Spartan job submission](https://dashboard.hpc.unimelb.edu.au/job_submission/) and [container guidance](https://dashboard.hpc.unimelb.edu.au/software/containers/)

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/DECISIONS.md`](docs/DECISIONS.md), [`docs/EVIDENCE.md`](docs/EVIDENCE.md), and the [interview defence/code map](docs/INTERVIEW.md).
