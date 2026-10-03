# Follow-up proposal: supervise the actual evidence state

**Proposal only.** This is a code-review/documentation package, not a new data
preparation, training run or model evaluation. Job **31781591**, its source
`fc4ffd61a7615682911556838c66f47aa33d0293`, inputs, policy and scorer are unchanged.
No unused validation, official dev/test or additional gold was read for this
proposal. Any implementation/data selection/model execution needs its own
bounded scope and exact release. Do not retrofit the current three-case score.

## Diagnosis grounded in code and existing receipts

- `prepare_scifact_grounding_candidate.py:prepare` builds FIT records using
  `fit_records`; retrieval is used separately for the tune/validation contexts.
  Its existing preparation routine reads eligible TRAIN annotations and builds
  both evaluation partitions: **do not rerun it for this proposal**. A future
  FIT-only preparation mode must avoid touching reserved evaluation content.
- `scifact_grounding_sft.py:context` shows complete documents, numbers aliases
  from `c1`, permits only `answer/abstain`, and has no preview/tool budget.
  `fit_records` creates one annotated document per record, always alias `c1`,
  one complete official rationale alternative and an `answer` target.
- `CONFIG/QUOTAS` currently select 24 SUPPORTS + 24 REFUTES FIT components;
  they do not select NEI for FIT. The verified output was **96 records / 24
  updates**, not evidence that rejection or tool policy was trained.
- Production `run_bounded_slot` assigns candidates from `c0` and uses
  `SciFactBoundedAgent` plus `CommonPacking`: current citable sentences,
  non-citable previews, remaining budget, feedback and state-dependent schema.
  A syntactically compatible terminal JSON does not eliminate this input shift.
- The retained 48-slot regression produced 47 nonempty single-document,
  single-sentence answers and zero model tool proposals. It was **not** globally
  `c1`-only. The [physical closeout](SCIFACT_ADAPTER_REGRESSION_CPU_20261001.md)
  reports this as a negative result, not proof that the mismatch is the sole cause.

## Minimal change map — no new framework

| Existing code | Proposed bounded extension |
| --- | --- |
| `scripts/prepare_scifact_grounding_candidate.py` | Add a separately versioned **FIT-only** path. Reuse already-frozen FIT component/document-family membership and consumption ledger; generate candidates without consulting labels. Save candidate order and state before attaching teacher targets. Do not regenerate any held-out partition. |
| `src/climate_rag/scifact_bounded_runtime.py:run_bounded_slot`, `scifact_bounded_agent.py:pack` | Capture real provider-bound observation/schema/visible map via a scripted provider, as already demonstrated by `ReadReplay` and the conditional preparer. Reuse controller alias assignment, budgets, feedback and `CommonPacking`; do not reimplement a parallel training packer. Capture is a future audited hook, not an unreviewed change to job 31781591. |
| `src/climate_rag/scifact_grounding_sft.py:fit_records` | Add a new versioned state-to-target builder. Targets depend on what is **actually citable**, not simply which annotated document exists in the corpus. Retain official per-document labels and complete alternative rationale sets; store policy-teacher provenance separately. |
| `scifact_grounding_sft.py:token_record`, `canonical_context`, `prediction` | Keep the original controlled experiment intact. Add production-state equivalents that validate the actual schema/alias map and physical prompt order, rather than reconstructing every state as complete-document `context()`. Keep one assistant target per captured state and the existing prefix/EOS/loss-mask checks. |
| `scripts/run_scifact_grounding_candidate.py:train` | Accept only a separately frozen, versioned record contract with explicit semantic-label and action provenance. It currently accepts only `official_annotated_document_alternative`; do not weaken this to an arbitrary provenance string. Recompute state/prompt/token/loss-mask identities and audit distribution before the first optimizer step. Preserve final-only checkpoint and exact adapter restoration. |

## Three kinds of supervision and their limits

1. **Answer now:** the real packed visible set already contains a complete
   official alternative. Supervise the per-document relation and **all** sentence
   IDs in that alternative. Do not force a `read` merely to manufacture a tool
   trajectory. Preserve genuine multiple sentences/documents when licensed by
   the annotation; do not union mutually alternative rationale sets or flatten
   different document relations into an invented single claim label.
2. **Read then answer:** no complete alternative is currently citable, but an
   already-retrieved, actually displayed preview candidate can be read legally
   within the existing budget. A deterministic FIT-only teacher proposes that
   read; execute it through the real controller, then capture a second state.
   Include the terminal target only when the new packed visible set contains
   the full rationale. No gold-document backfill, fabricated tool result,
   alias relabeling or forced evidence retention. If packing still hides required
   sentences, retain a packing failure rather than teaching a partial answer.
3. **Abstain:** distinguish two cases. An official claim-level NEI annotation
   may license no-evidence rejection; a SUPPORT/REFUTE claim whose usable
   evidence is absent or budget-inaccessible licenses a **context-insufficient
   policy action**, not an invented NEI ground-truth label. Unannotated documents
   are not document-level NEI negatives. Empty retrieval, timeout, invalid JSON
   and physical execution failures stay failures, not successful abstentions.

Each record needs separate `semantic_target_provenance` (official annotation,
alternative ID/hash, or no semantic target) and `action_target_provenance`
(versioned program teacher + decision reason). Program-selected reads using
FIT annotations are oracle teacher supervision; neither those reads nor their
replays are autonomous model behavior. Ambiguous legal reads are not uniquely
correct actions. Such a teacher can be useful without being an official label.

The current FIT has no NEI components. Any new NEI pool is a **future proposal**:
select and freeze eligible TRAIN components using the existing component/family
and exposure rules under a separate data authorization. Do not borrow reserved
validation NEI, inspect it for examples, or first consume candidate gold and
then declare it independent. This document selects no new IDs and creates no
new supervision records.

## Assistant-only loss and complete rationale preservation

Use one record per actual decision point, linked by trajectory ID; the read
state and post-read answer state are not two independent claims. Render each
with the same production tokenizer/chat template/schema and physical nested
key order. The already implemented `token_record` invariant is essential:
`encode(prompt + target + EOS)` must start with the exact inference prefix;
all prefix positions have label `-100`, only assistant JSON and its real EOS
contribute loss. Tool observations, previews, system/schema text and feedback
must remain masked. Do not add synthetic conversation history that production
does not consume; current controller history appears through its real state.

Reject complete targets exceeding the frozen output limit; never truncate to
the first rationale sentence to make a sample fit. Track retained/dropped
records by component, target action, visible rank, number of documents and
rationale length before training. Derive aliases from actual retrieved order;
do not fix bias by replacing all `c1` targets with `c0`, permuting only labels,
or silently moving a gold document to first position. Any future candidate-order
augmentation would need its own frozen replay and alias/source consistency test.

## Decision after the existing three-call job

**Observed update, without new implementation authorization:** the
[completed three-call audit](SCIFACT_READ_CONDITIONAL_CLOSEOUT_31781591.md)
falls into the limited-success branch: one complete answer, two wrong-rationale
sentences. Every case had a reachable singleton alternative. Therefore the two
failures are not demonstrated multi-sentence truncation or merely missing the
rest of a partially correct rationale. Prioritize sentence/evidence discrimination
and justified abstention in the actual packed state; retain complete-rationale
tests as safeguards, not as an asserted explanation for this particular failure.
This does not authorize preparing new data, modifying the teacher or training.

- **All three conditional answers fail:** prioritize relation/rationale grounding
  and justified abstention in real packed contexts before action imitation.
  `read` supervision cannot repair an inability to use the evidence once visible.
- **At least one conditional answer succeeds:** it establishes only a conditional
  capacity on these previously exposed inputs. Propose state-conditioned read
  trajectories **alongside** already-sufficient answer and warranted-abstain
  controls; do not optimize number of tool calls. Preserve one FIT-overlap and
  two not-direct-FIT results separately; three examples are not a generalization
  estimate or proof of causal sufficiency.
- Unknown costs, missing slots or invalid/nonterminal proposals cannot satisfy
  either success claim. Keep the original three-case denominator and exact
  original adaptive comparison. Do not choose a new threshold after seeing it.

## Required tests before an implementation release

Extend the existing grounding-candidate/train, bounded-agent and read-continuation
tests rather than start another test framework:

1. Captured state equals real serving observation/schema/prompt/tokens after
   physical JSON roundtrip; canonical hashes alone must not hide order changes.
2. The same claim/candidate list in sufficient, preview-only and exhausted-budget
   states yields the declared answer/read/abstain policy without gold fields
   leaking into the observation. Preview-only citation and read-loop are rejected.
3. Full multi-sentence alternatives survive target construction and official
   export. Alternative sets are never unioned; absent sentences, duplicate
   aliases, mixed source revisions and over-budget targets fail explicitly.
   Preserve the proposed sentence order through the official first-three
   rationale scorer; sorting/truncation must not masquerade as equivalent output.
4. Assistant-only labels, real EOS and Unicode/BPE prefix equality survive every
   action. Tokens from a tool result or prior state never receive target loss.
   Run the boundary fixture with the frozen real tokenizer, not just a character
   stand-in; an EOS-equals-PAD setting must not accidentally mask the target EOS.
5. Current-candidate sources and all replay turns share the original component
   partition; sibling/near-duplicate documents and reused trajectories cannot
   inflate independent sample counts or enter held-out pools.
6. Official NEI and context-insufficient teacher actions have distinct provenance.
   Unannotated negatives, timeouts and parse errors cannot become official NEI.
7. Whole-claim weighting/record caps prevent two-turn examples from silently
   doubling that claim's influence. Distribution and dropped-record denominators
   are frozen before training; weights and gradients are checked explicitly.
8. Existing final-checkpoint/tensor audits, single-execution cost accounting,
   unknown-cost handling and worker-exit-before-gold scoring remain enforced.

## Proposed resource envelope — not an execution request

First implement/test on tiny synthetic fixtures and previously authorized
FIT-only CPU replay, with tokenizer-only imports (no model weights). Candidate
CPU preparation ceiling: one process, 4 GiB, a bounded timeout; measure peak
RSS/token lengths before deriving any larger request. No new cluster submission
belongs to this documentation package.

A later reviewed candidate should retain the existing base model and small
q/v LoRA configuration, one epoch/final checkpoint and **at most 192 supervised
decision records** (at most 48 accumulation steps at width 4). Count turns,
claims and components separately. This ceiling is a cost bound, not a target
to fill with weak examples. Record-family weights and real sample counts must
be preregistered; no automatic sweep, multiple checkpoints or extra warmup.

Provisional hardware is one A100 / 4 CPU / 32 GiB RAM / 30 GiB scratch, based
on the existing model's verified training run (92 allocation seconds, 30.32
fit-loop seconds, 9,227,040 KiB batch MaxRSS). Longer production contexts may
change memory/time: a released first representative update must count toward
the single epoch, and its observed memory and step time determine the final
walltime/budget proposal with explicit startup and safety allowance. Do not
reuse the old 92 seconds as a forecast, promise an 80-GiB device, or automatically
resubmit after OOM. No GPU execution is requested or authorized by this proposal.

Subsequent base/adapted comparisons must use the same frozen production inputs,
candidate order and budgets, separately report grounding, unnecessary reads,
read-then-correct completion, justified abstention, failure and actual cost.
No desired positive result or target tool-use count is prescribed. Unused
validation remains unopened until a separately preregistered gate and release.
