# Bounded paired-run posthoc closeout

## Verified result and decision

F (`format_repaired`, job 31686182) and F+G (`format_repaired_gap`, job 31686183)
both completed with exit `0:0`. The unchanged read-only audit at
`a347233b88c103cb23c76845527278f960a48406` completed in CPU allocation 31692980.
The [compact](verified-runs/scifact-bounded-pair-posthoc-31692980.json) SHA is
`94ca7cb2167decab8ae51d8dd9dd07bdc4d3028ac003a79700d62032d9e80abe`.
Identity joins, full-wire/grammar receipts, durable 48+48 slots, initial-context
comparability, score recomputation and 32 independent official metric integer
triples all passed. Inference source remains `b248fe7`; it was not modified by
the audit.

All 96 train generation attempts returned valid decisions with no repair. This
is format compliance, **not semantic correctness**. Both adaptive arms produced
only answer/abstain decisions: zero raw tool proposals, zero model-selected tool
events, and consequently zero observed new-evidence-to-correct-answer chains.
Their fixed initial retrieval events are counted separately, not attributed to
the model. A zero tool funnel here is not an estimate of success conditional on
tool execution, because no such execution occurred.

| Route (12 claims each) | F correct / predicted / relevant docs | F abstract-rationalized F1 | F+G correct / predicted / relevant docs | F+G abstract-rationalized F1 |
|---|---:|---:|---:|---:|
| Fixed retrieval | 2 / 31 / 9 | .1000 | 2 / 24 / 9 | .1212 |
| Fixed rerank | 2 / 44 / 9 | .0755 | 2 / 44 / 9 | .0755 |
| Deterministic extra | 1 / 43 / 9 | .0385 | 2 / 46 / 9 | .0727 |
| Adaptive | 2 / 31 / 9 | .1000 | 2 / 28 / 9 | .1081 |

The adaptive F1 difference comes from fewer predicted documents, not more
correctly rationalized gold documents. The conservative strict whole-final-answer
diagnostic is 2/12 for both adaptive arms (including legal NEI abstention where
appropriate); it is not official claim accuracy. Neither aggregate is a
generalization estimate on these biased eligible-train examples.

In F+G adaptive, all 12 initial self-reports said retrieval was not needed;
nine reported relevant/sufficient and answered, while three reported
irrelevant/missing and abstained. All three gold-absent-Top-20 cases answered,
and no top-20-replenishable case triggered a model tool. These are posthoc
diagnostic cross-tabs, not calibrated evidence-sufficiency labels or causal
proof. They explain why merely adding a self-report field did not establish
adaptive retrieval behavior.

Decision: retain this negative result. Do not promote the Agent, release a new
holdout, claim an online benefit, or change the current resume from this run.
No further GPU run or policy change is part of this closeout.

## Complete cost accounting

| Scope | Calls | Input tokens | Output tokens | Known generation time |
|---|---:|---:|---:|---:|
| F train, all four routes | 48 | 179,780 | 6,867 | 207.583 s |
| F+G train, all four routes | 48 | 198,260 | 10,401 | 378.244 s |
| F synthetic preflight | 4 | 2,720 | 238 | 7.083 s |
| F+G synthetic preflight | 4 | 4,260 | 535 | 22.172 s |

All 104 calls have recorded usage and generation timing; G uses full original
wire rather than mapped-action length. These are verified stored ledgers, not
an independent tokenizer recount or invoiced API cost. F and F+G Slurm elapsed
times are 364/567 seconds, with batch MaxRSS 18,534,620/18,567,812 KiB. GPU
allocation time is not utilization; overlapping generation, loading, tool,
question, operator and Slurm durations must not be added together.

CPU audit attempt 31691393 failed before any run/gold read because its wrapper
discarded module PYTHONPATH (`numpy` became invisible). Preserve its
[failure receipt](verified-runs/scifact-posthoc-environment-failure-31691393.json),
9 seconds elapsed, 2.318 CPU seconds and 117,960 KiB step MaxRSS. The separately
authorized single infrastructure retry 31692980 passed: 30 seconds elapsed,
22.720 CPU seconds, 1,408,252 KiB step MaxRSS, 1 CPU/4 GiB/5-minute cap, no GPU
or model calls. No failed compute is omitted from the ledger.

The [r2 wrapper](verified-runs/scifact-posthoc-a347233-r2.sbatch) SHA is
`b2238a9cd5250723819bd31a4ff21c3c8c97b4654abaf7c84becf1f6c055ac01`.
It reuses frozen runtime archive `2423a755...` and the existing
`v3_runtime_environment` ordering: overlay, runtime site-packages, source, then
module PYTHONPATH. An import-only check before allocation and again inside it
confirmed Python 3.10.4, NumPy 1.26.4 and jsonschema 4.23.0 from the runtime,
Pydantic 2.13.5 from the overlay, and the unchanged audit entry. Local/remote
`bash -n` and `sbatch --test-only` passed. The latter was only a scheduling
simulation, not an additional allocation. The old log and empty output directory
are preserved; retry output is exclusively `posthoc/scifact-pair-audit-a347233-r2`.

## Audit design and reproduction

The wrapper's `--imports-only` mode checks dependencies without reading runs,
gold or model weights. Its default mode requires a CPU allocation and invokes
the hash-pinned source archive at `a347233`; the reviewed compact is exported
only after the audit completes. Reproduction requires the private authorized
Spartan assets, not fixtures substituted for real runs. The existing output
directory deliberately prevents a repeated execution from overwriting evidence.
No archive, raw response, gold or per-row record is published with this report.

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
