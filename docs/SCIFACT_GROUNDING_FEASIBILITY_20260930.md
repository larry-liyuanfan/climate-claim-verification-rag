# Next grounding experiment: CPU-only feasibility, not execution release

## Recommended next step

First run a small **oracle-context fault-localization diagnostic on already
consumed TRAIN components**, only after a new explicit model release. Do not
consume new validation components to tune another prompt, and do not directly
start action SFT. The completed semantic pair shows relation/selection failures
even with a complete rationale visible; it does not prove that tool selection
alone is the limiting capability. [Measured closeout](SCIFACT_SEMANTIC_CLOSEOUT_20260930.md).

No runner, dataset split, training adapter or oracle output is claimed delivered
by this proposal. This document authorizes no GPU, model call, CPU Slurm job,
dev300/frozen-test evaluation, deployment or resume edit.

## Method and label provenance

SciFact's official pipeline separates retrieval, rationale selection and label
prediction and uses oracle components to localize errors. We borrow that
diagnostic separation, not its examples' dev/test commands.
[Official model documentation](https://github.com/allenai/scifact/blob/master/doc/model.md).

| Source | Allowed use | Important restriction |
|---|---|---|
| Official evidence document, relation and alternative sentence sets | Annotation-derived supervised relation and rationale targets | Preserve each alternative as an OR-set and keep all variants in one component partition |
| Official rationale trainer | Sentence-level annotation-derived membership within an annotated evidence document | It uses the union for binary sentence membership; that union is not a mandatory combined generative rationale |
| Official label trainer's non-rationale or original-NEI cited-context samples | Explicitly tagged `official-code-derived` weak/constructed insufficient-evidence samples | These are not new human judgments for each sampled context |
| Arbitrary retrieved but unannotated documents | Unjudged candidates for error analysis, or separately reviewed weak labels | Never silently label them verified NEI/false semantic negatives |

The sentence trainer creates binary membership in the union of annotated
rationales within evidence documents. The relation trainer uses individual
rationales and a same-document union, and constructs insufficient-evidence
contexts from non-rationale sentences or cited contexts of NEI claims.
[Rationale training code](https://github.com/allenai/scifact/blob/master/verisci/training/rationale_selection/transformer_scifact.py),
[relation training code](https://github.com/allenai/scifact/blob/master/verisci/training/label_prediction/transformer_scifact.py).
Historical dependencies, default epochs and dev-checkpoint searches are not
copied into the proposed implementation.

MultiVerS jointly learns document relation and rationale selection with full
claim/document context. That supports testing grounding before action policy;
a Qwen LoRA adapter borrowing its supervision is **not a reproduction of its
architecture**. Retain full abstract context and alternatives; do not gold-crop
the input, automatically gold-trim predictions, or replace the official first3
scorer. [MultiVerS paper](https://aclanthology.org/2022.findings-naacl.6/).
Its training notes discuss overfitting with many weak negatives; candidate count
does not justify treating weak labels as certain.
[Official training notes](https://github.com/dwadden/multivers/blob/main/doc/training.md).
Do not use a released SciFact-TRAIN-tuned checkpoint to claim generalization on
our own TRAIN-derived validation.

## Data feasibility and fixed partitions

The compact inventory verifies 491 model-unconsumed eligible claims in 301
remaining components, after all 24 consumed components and their 40 member
claims are excluded. All 531 eligible claims were already exposed to legacy
gold-aware preparation. Therefore a future validation set is **component-disjoint
within TRAIN**, not independent unseen test. Original dev-overlap quarantine,
restricted retrieval data and CLIMATE-FEVER frozen test remain untouched.

Reuse the frozen assignment rather than ID-based embedding-training splitting.
Its lexical/source-family grouping joins claim token-Jaccard ≥0.8, document
variants ≥0.9 and shared cited/evidence families; it cannot certify absence of
semantic paraphrases or foundation-model pretraining exposure. Do not rerun this
group audit against dev for the next package.

The existing preparation script hardcodes the old r2/F/G IDs; appending the new
semantic runs to that list is incorrect. A future ledger entrypoint must bind
each run to its own frozen protocol, then union attempts, failed/unknown calls
and uncertain planned exposure before whole-component exclusion. Our new compact
performs that cumulative accounting without selecting new claims.

Before any next training, freeze all component assignments, source hashes, label
provenance, candidate widths, decoder limits and reporting rules. All same-claim
documents, alternatives, derived weak negatives and paraphrase variants stay in
the same partition. A negative document may not silently cross a source-family
boundary; filter cross-partition source families or rebuild the train-only
component union before the freeze. No gold-based backfill after model outputs.

The 199 SUPPORT / 94 CONTRADICT / 198 NEI counts are claim counts, not a guaranteed
number of independently balanced components. Stratum shortages or oversize
contexts must be reported, not repaired by relaxing exclusion or changing
validation membership. Do not force the seven remaining replenishable claims
to supply a large balanced action-training set.

## Stage A — minimal oracle diagnostic, proposed only

Use the **already consumed current twelve claims**; original full abstracts and
fixed candidate identities stay private. Declare oracle conditioning explicitly.
There is one frozen prompt/schema per subtask, no prompt or decoding sweep.

| Subtask | Input / controlled condition | Output and diagnosis | Maximum calls |
|---|---|---|---:|
| Document screening | Claim + frozen candidate full abstracts, no gold marks | Select relevant evidence documents or abstain; official-set P/R plus unjudged extras | 12 |
| Relation | Claim + one gold full abstract for nine evidence claims; predeclared cited-context controls for the three official NEI claims | SUPPORT/CONTRADICT/insufficient; report positive relation confusion separately from NEI-context controls | 12 |
| Rationale | Claim + gold full abstract + explicitly supplied correct relation for the nine evidence claims | Choose a complete alternative, order deliberately, measure first3 credit and extra sentences | 9 |

Total **33 single-turn diagnostic calls plus four real synthetic preflight
calls**. The correct relation in the last row is an oracle intervention, not
production input or an end-to-end result. NEI controls never become arbitrary
retrieved-document negatives. Preserve original sentence order/IDs in context;
do not shorten the abstract to its gold sentences. If the fixed token limit
cannot hold a full context, fail/mark coverage for that case without replacement.

No search/read/rewrite is available in this diagnostic: that is intentional
isolation, not an Agent demonstration. Interpretation is conditional:

- Failed relation with a complete gold abstract: evidence-relation learning is a candidate bottleneck.
- Correct relation but incomplete/reordered rationale: focus on alternative-aware selection and constrained output planning.
- Gold-context success but candidate-screening failure: investigate document discrimination and unjudged/weak negative quality.
- No stable diagnosis: stop; do not automatically escalate to action SFT or more prompts.

Proposed resource ceiling for the future inference-only diagnostic: one A100 /
8 CPU / 32 GiB / one hour, serial, no requeue. The previous 52 calls including
preflight fit in ~10.5 minutes, but full-abstract contexts may differ; first verify
token/coverage counts and use the existing per-call timeout. This is a ceiling,
not an ETA or current scheduling authorization.

## Stage B — one grounding candidate, conditional and later

Only if Stage A supports a trainable grounding failure, propose **one** adapter
against the frozen base generator: full-context document-relation + selected
sentence supervision, not action trajectories. Reuse data parsers, typed output
contracts, cost journals and original scoring. Label-only ablation is a later
separate request, not an automatic matrix.

Initial cap: 48 fit components, 12 tuning components, 12 frozen TRAIN-validation
components; one deterministic claim per component initially. Fit may reuse
already-consumed components; tuning/validation must come from the 301-component
remainder and remain mutually/source-family disjoint. Up to four derived records
per fit claim gives **at most 192 records**. Keep the official vs constructed
supervision provenance, alternative IDs and masks private. Weight annotation-
derived and weak examples separately; the minimal first candidate can omit weak
negatives instead of assuming their reliability.

One configuration, one seed, one final checkpoint; at most one epoch or 64
optimizer steps, whichever comes first. This is a small task-adaptation pilot,
not pretraining. Architecture/rank/optimizer and a resource estimate require a
separate freeze and measured shape/pilot before release, not guessed walltime.
No official SciFact-finetuned checkpoint may serve as an allegedly untouched base.

Compare base/adapter using the **same claim and identical fixed candidate full
texts**, without supplying the gold relation, answer or gold evidence markers.
Both models must independently output document relations and cited sentence IDs;
gold is available only to the separate evaluator after inference. This is a
non-oracle joint grounding comparison. Stage A's correct-relation-conditioned
rationale scores remain diagnostic only and cannot substitute for this outcome.
Tuning may select one predeclared abstention threshold; only then run both on the twelve
frozen TRAIN-validation claims once (24 calls). Including tuning, the comparison
cap is 48 calls. Report document relation Macro-F1/confusion, official document
and sentence counts, alternative completion/first3 credit, unjudged extras, NEI
false evidence/coverage and token/latency cost. Do not describe zero-output
failures as abstention or use tool-call rate as the objective.

Any separately authorized later four-route Agent comparison must use the **same
frozen generator and adapter state across all four routes**. Do not compare an
adaptive route using the new adapter against historical base-generator baselines;
that would confound grounding adaptation with Agent policy. This clarification
does not authorize that comparison, Stage B data preparation, a split, training
or new evaluation, and does not change the running Stage A release or its source.

Preregister count-based advancement before running: more correctly rationalized
documents with no extra NEI false evidence, no additional invalid/failed outputs,
and unchanged input/candidate/token-budget contracts on tuning. If it fails, do
not open frozen validation. On the tiny validation report paired counts without
claiming population significance; a later meaningful-sized independent protocol
would be a separate decision. No automatic dev/test release follows any win.

## Minimum implementation and stops

Future CPU preparation would add a cumulative-ledger command, component split
manifest, provenance-tagged grounding records and an oracle-context fixture
runner; none exists as a claimed deliverable of this proposal. Reuse
`scifact_grounding`, `scifact_consumption`, semantic bundle isolation and the
frozen independent scoring helpers. Do not reuse the claim-ID-only split of
`embedding_training.py` for this component-level claim.

Stop on contaminated partitions, missing provenance, unreachable complete gold
under the declared diagnostic context, output-contract failures, unknown costs
or failure to meet the predeclared tuning gate. No silent gold clipping, scorer
changes, sample replacement, automatic retry, new polling or resume promotion.
This CPU/document package ends at the report and evidence handoff.
