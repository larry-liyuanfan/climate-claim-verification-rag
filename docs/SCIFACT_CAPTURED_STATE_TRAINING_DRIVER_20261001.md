# Captured-state training seam: CPU-only, synthetic validation

This package is an interface and mathematical validation, not a new trained
model, selected training cohort or empirical quality result. It adds no real
training CLI, model loader, data reader, GPU/Slurm entry or evaluation. utility8
job 31834687 continues to use immutable `e22143c`; its uploaded source, release,
eight-case selection, budget and model are unchanged.

## Three library entry points

`src/climate_rag/scifact_state_training_driver.py` provides:

1. `prepare_training_inputs(records, roster, tokenizer, corpus, provenance)`
   accepts an **externally frozen** ordered roster and accepted supervision.
   It validates every claim/state before any forward, directly projects the
   captured `observation` and `schema`, and calls existing `tokenize_target`,
   `validate_tokenized` and `validate_weights`. It does not call the old
   full-context teacher, select targets, fill gaps, drop records or renormalize.
2. `make_epoch_plan(prepared, seed)` shuffles claim IDs once, not records.
   All records belonging to a claim remain together; each group has at most
   four complete claims. The sealed plan records actual N, roster identity,
   record hashes/order, one epoch and exactly `ceil(N/4)` optimizer steps.
3. `run_one_epoch(prepared, plan, model, output)` injects an existing **CPU tiny
   model**, constructs the explicit AdamW, and reuses the shared claim-group
   mathematical kernel. Parameters and buffers must be on CPU and the model
   may have at most one million parameters. Execution accepts only
   `synthetic_fixture` provenance, not real training data. No weights are
   downloaded or saved. A future real-data/training release is still required.

New input contract: `scifact-captured-state-training-input-v1-20261001`.
Old state-supervision v2 defaults and its `/48` behavior remain unchanged.
`WeightContract` is an explicit per-call specification, not a mutation of old
global CONFIG. The new kernel `claim_group_mean_update` has no cohort/epoch
divisor; the legacy wrapper retains its old metadata and outputs.

## Frozen input contract and state audit boundary

The roster is `{payload, sha256}` with an ordered identity hash. Its payload
contains version, ordered `claim_ids`, actual `claim_count`, ordered
`record_sha256` list and the external provenance hash. It is not inferred from
whatever rows happen to survive. Provenance binds the supplied corpus,
tokenizer artifact identifier, capture run and supervision selection.
The tokenizer artifact identifier is supplied by the external audit; this
library does not open model/tokenizer files. Re-tokenization verifies actual
prompt/token/mask identities for every accepted state.

Records retain the captured five-field frame, corresponding completed tool
event, externally chosen target, complete trajectory/alternative/state indices
and counts, rational record weights, packing hashes, record/capture/event/
provenance hashes and explicit `ready`/no-gap state. Corpus identity is computed
from the supplied abstracts. No annotation loader is present. The preparation
interface can validate an `exposed_train_pending_release` package but the
runner rejects it; validating provenance does not authorize real training.

**Required upstream acceptance audit, not implemented or asserted by this
CPU package:**

- For utility8 A, associate `decision_execution_audit.actual_event_index` with
  `next_attempt_index`. Replay completed `event.candidate_ids`; a repair round
  without a tool inherits the previous state.
- For B/C, the scripted intervention applies to logical attempt 1, physical
  `frame-0`; `initial-frame` belongs to shared attempt 0.
- Join `generation_attempts[].diagnostics.physical_attempt_id` to the physical
  reserved/finished ledger; verify slot and actual observation/schema, and
  reject duplicate, missing or unfinished receipts. A frame is saved before
  the deadline check, so a dangling tail frame cannot be paired with a target
  merely by ordinal position.
- The shared A0 is one physical sample, not three independent examples.
  Alternative supervision may deliberately share an accepted frame with
  fractional weights; that is not a new independent model observation.
- Freeze all accepted targets, state lineage, annotation scope and roster
  before invoking this interface. The current final-answer scorer does not
  replace intermediate-state acceptance or prove target semantic correctness.

The utility8 output has not been read or converted by this CPU package.

## Stable aliases are not ranking positions

`frame_contract` and `tokenize_target` now accept keyword-only
`alias_to_source` and ordered `candidate_aliases`. Both must be supplied
together; their default path is unchanged. For example:

```text
registry: c0 → D0, c1 → D1
current order after rerank: [c1, c0]
reference c1:0 → D1 sentence 0 (never D0)
```

The exact captured schema is checked against the **current order**, never
reordered or rewritten. The original observation, rendered prompt, full token
sequence, EOS and assistant-only loss mask are retained and revalidated.
For old utility8 frames without candidate order, `captured_state` takes it
only from the corresponding completed tool event, never registry insertion
order or sorted cN names. Event source hashes, selected context, visible
document order and the preview prefix must agree. Within a trajectory,
registries may only append aliases: old aliases cannot change or disappear.

This validates a supplied frame/event pair; it does not independently replay
the physical journal or prove that the caller selected the correct event.
Legacy teacher functions such as `read_plan` still use the old unsorted
candidate contract; the new driver never invokes them for captured states.

## Objective and failure semantics

Each claim has unit mass, split over its complete declared
trajectory/alternative/state hierarchy. Checks require consistent counts and
complete index coverage at each level, not just per-row arithmetic plus total
mass one. Missing, duplicate, empty, gapped or inconsistent input fails before
any model forward, optimizer zeroing or step, including bad rows in the tail.

For a group G, the kernel computes
`sum(record_weight * assistant_token_mean_loss) / len(G)`.
Each whole group has exactly one clip and AdamW step, including the tail.
The new epoch metric is `sum(group_mean * len(G)) / N`, not `/48`, a record
average, or an unweighted average of group losses. It is the loss observed
**along the training path**, not a fresh evaluation of the final model.
LR=1e-4, betas=(0.9, 0.999), epsilon=1e-8, weight decay=0, clip L2 max norm=1;
the remaining AdamW switches are explicit and unchanged.

An output directory is a lifetime reservation. Per-group and step-start/
step-completion receipts record consumed work; exceptions or nonfinite losses,
gradients, parameters or optimizer state stop without retry. An exception
inside `optimizer.step` reports a started step with unknown completion, never
claims that parameters were unchanged. Returned steps and fully verified groups
are counted separately. There are no intermediate model checkpoints.

Only **`complete.json`** is a success marker, atomically published after all
groups pass. `complete.pending.json` is not success. `failed.json` is best
effort during hard termination or persistent storage failure; missing terminal
receipts mean incomplete, not resumable or automatically retryable.

## Bounded validation

Six synthetic test groups cover:

1. Actual controller read feedback and deterministic synthetic rerank,
   stable c1→D1, exact prompt/tokens/mask, wrong alias/order and terminal-frame
   wrong-event rejection even when the schema no longer offers read.
2. Heterogeneous claims with different legal labels/sentence sets, different
   alternative/state counts and assistant lengths. Correct claim-mean loss
   **and gradients are demonstrably different from incorrect record averaging**;
   two updates, Adam moments, clip/step counts and split-state invariance match
   an independent gather/log-softmax/manual-clipping reference.
3. N=1/3/5/7, deterministic seeds, complete groups and tail semantics.
4. Missing/duplicate/gapped records, inconsistent but mass-one hierarchy,
   and individually valid captures spliced with a rebound registry.
5. Provenance, frame, packing, token/mask and prepared-identity tampering.
6. Mid-forward, nonfinite loss/gradient, step-after-mutation, post-step NaN
   and final-write faults, honest consumed-work receipts, and no retry.

New six groups and the two affected legacy suites: **60 passed**. Invoke with
`PYTHONPATH=src;scripts` on Windows; the old preparation-script tests require
the scripts path. The first narrow invocation omitted it (four import failures),
then the correctly configured run passed; no source dependency was missing.
Ruff and targeted strict mypy apply to the three affected source modules.
These results prove local CPU contract/math behavior only, not autonomous
Agent benefit, accepted real supervision, trained-model quality or an SLA.
