# Bounded paired-run posthoc closeout

This is a read-only CPU audit, not a new model experiment. Execution remains
`b248fe7f43b175b15b90ec6143538d5dab8e2f35`; the audit code commit is recorded
separately. It writes a new compact file under the Climate `posthoc` directory,
never edits run/operator/score/wire files, and refuses overwrite. Both Slurm jobs
must be `COMPLETED 0:0` before any run, score or gold content is read.

`scripts/audit_scifact_bounded_closeout.py` reuses the existing paired scorer,
original SciFact scorer, strict terminal parser, G mapping adapter and physical
wire multiplicity checker. The older `audit_scifact_train_closeout.rescore`
pure function independently checks the four official metrics' integer counts;
its historical hard-coded main is never executed.

The joins cover Slurm IDs, operator and run status, source/tar/protocol/model and
asset hashes, ordered 48+48 slots, individual durable raw/converted slots, score
hash, and full raw-response/grammar physical receipts. Empty grammar sinks create
no file and are distinguished from a lost nonempty log. The actual visible
sentence hashes are checked against original corpus text and source hashes.

## What the counters mean

- Fixed retrieval/rerank events are separate from model-selected tools.
- Stored raw wire is replayed through the same G mapper and strict parser, not
  sent to a model. Full wire usage/history/feedback must match. Candidate state
  is reconstructed in the frozen serial controller's event order; this is not
  independent proof of events outside the recorded controller.
- Immediate added evidence is next citable minus previous citable. Newly seen
  evidence is next citable minus **all** earlier citable views in the slot.
  A→B→A can add A back but cannot discover A twice. Preview is never evidence.
- The `then_*` counters are event-level funnels: successful execution → observed
  added evidence → legal immediate subsequent decision → final citation of added
  evidence → official label plus complete first-three rationale credit.
  `successful_event_with_next_legal_decision` is counted independently of discovery.
- Final credit always belongs to the same claim/arm/route. Unique credited
  documents are deduplicated within the slot, not counted again for each tool.
- Partial official document credit is not strict whole-answer success. The
  conservative whole-answer diagnostic requires the exact gold document set,
  correct labels, a complete alternative rationale within each document's first
  three citations, and no nongold rationale sentences. A legal final answer or
  abstain with consistent termination is also required: a failed empty output
  is **not** a successful NEI. Structural gold matching is reported separately.
- All these are **observed evidence-use closure**, not proof that a tool caused
  correctness. Model self-reports remain fallible categorical observations.

## Cost and boundaries

All train calls, failures, repairs and unknown-usage flags remain in denominators.
The eight synthetic preflight calls are separate from train costs. G costs use
the original full wire, not its shorter mapped action. Missing fields are marked
unknown/lower bounds, never silently filled with measured zero. Generation,
tool, question, operator and Slurm times overlap and are not summed as one wall
time. Load/index/preflight times are explicit; blank MaxRSS remains null.

Only reviewed aggregate counts, booleans and hashes are exported. Gold, sentence
text, individual predictions and raw wire stay on Spartan. The biased 12-claim
train diagnostic remains neither a holdout nor an online A/B test. No current
resume change, new GPU run, automatic retry or polling automation follows.

Targeted checks: 25 new synthetic tests plus five existing release tests pass;
Ruff and strict mypy for the new script (`--follow-imports=silent` avoids
rechecking historical untyped entry points) pass. No unchanged model/training/
real-tokenizer suites were rerun.
