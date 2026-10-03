# Same-evidence note → terminal diagnostic (closed: no strict gain)

## Measured closeout

[Job 31884581](verified-runs/scifact-evidence-note-closeout-31884581.json) completed
once, exit `0:0`, in 178 s. Execution source is
`c76fb71a4c265637422ee251f35adeb12df589a7`; documentation commits are not a new
execution. The same-source Linux quality run passed all 1114 tests without skips.

| Measure, same exposed TRAIN twelve | Existing direct control | Note + terminal |
|---|---:|---:|
| Strict whole answers correct | 5/12 | 5/12 |
| Correct NEI abstentions | 3/3 | 3/3 |
| Rationalized documents: correct / predicted / relevant | 2 / 5 / 9 | 2 / 2 / 9 |
| Rationalized-document F1 | 0.286 | 0.364 |
| Sentence-selection F1 | 0.261 | 0.200 |
| Actual generator calls | 12 | 23 |
| Input / output tokens | 42,914 / 253 | 66,554 / 1,761 |
| Sum of generation time, excluding model loading | 11.729 s | 93.081 s |

There were **zero recovered and zero regressed strict answers**. Precision rose
because fewer documents were predicted, not because more correct evidence was
found. The new route returned two answers, nine abstentions and one failure.
That failed note reached 512 tokens without EOS; its known cost remains counted,
its terminal call was not attempted, and the denominator stays twelve. Thus
**12 note + 11 terminal = 23 physical calls**, not 24 successful calls. The process
completed correctly while preserving this model-level failure.

Readback checked 132 file hashes, all 23 full raw response receipts, their 23
empty grammar receipts, 144 restored adapter tensors and all 11 unchanged
terminal contexts. There were no unknown token totals or physical receipt
issues. All twelve original frames had passed the frozen worker's equality
checks before generation; the note supplied no new citable evidence. Costs were
written after reaping the worker and before the original scoring stage.
Readback loaded no gold, ran no model and did not recompute quality.

**Decision: do not promote this note stage.** This is a negative quality/cost
diagnostic on old exposed TRAIN, not an independent test, online SLA, autonomous
Agent gain or a pure prompt-effect estimate. It does not authorize retry,
prompt tuning, a new baseline or consumption of validation12/dev300/retired test.
Private notes, frames, per-case outputs and gold remain on Spartan; only the
redacted aggregate and hashes are published. The current resume is unchanged.

## Frozen experiment design

**Question:** can the accepted mixed adapter recover rationale/label correctness
on unchanged evidence, without regressing NEI, and at what extra cost?
[31871386](verified-runs/scifact-mixed-four-route-closeout-31871386.json) completed
48 calls, but adaptive made no tool proposals. Fixed rerank was correct on 5/12
whole answers (three NEI); sufficient-evidence errors also occurred. Neither a
missing-read-only explanation nor an autonomous Agent improvement is justified.

The control is the existing **all12** fixed-rerank output, never a new baseline.
Recorded BM25 candidate order and rerank order reconstruct its first actual frame.
Before any generation, original prompt/schema hashes, token count, original
sentence text/order and stable aliases must match; c7 ranked first stays c7.
No gold, predicted-success filter, new retrieval or reranker execution is used.

1. Same mixed adapter writes a short JSON evidence-check note from only the
   original claim and visible original sentences. No correct-label hints.
2. Same adapter receives the complete unchanged original observation plus the
   note as explicitly untrusted/non-citable user data, and the original terminal
   schema. It cannot add evidence IDs, execute tools or override the claim.

Budget is **12 × (512 + 512)** output caps, not the old single-512 budget;
each physical input is at most 8192 tokens. Greedy/nonthinking, no training,
warmup, retry, repair, downloads or seed search. Stage1 failure skips stage2.
Overflow fails explicitly without repacking/removing sentences. EOS at token512,
no EOS at the cap, and truncated private files are different states. Per-stage
reservation/response costs survive later failures; unknown cost is not zero.
Cost comparison uses old one-call versus new two-call generation tokens/time;
old end-to-end latency includes retrieval/rerank and is not a matched comparator.
One isolated worker has a 120s stage watchdog and a total bound; it is reaped
before durable costs, and only then can a separate scorer load gold. Failures
remain in the twelve-question denominator and cannot count as successful NEI.

Entries: `run_scifact_evidence_note.py`, `run_scifact_evidence_note_operator.py`,
`score_scifact_evidence_note.py`; freeze with `package_scifact_evidence_note.py`
and `hpc/scifact_evidence_note.sbatch`. The candidate needs a separate exact-hash
coordinator release and `sbatch --test-only` before any actual submission.
One A100/8CPU/32GiB/30GiB scratch, 70min allocation bounds 24×120s maximum calls
plus model/runtime/audit overhead; the previous real four-route run took189s.

Synthetic tests cover aliases/frame equality, notes-as-data, physical two-call
policy/usage joins, failures/overflow/EOS and cost-before-gold. A stdlib-only WSL
test exercises the exact POSIX watchdog function, including Popen-signal cleanup.
The measured result is recorded above. Raw outputs stay on Spartan. Old12 are exposed
TRAIN (one current FIT component overlaps); validation12/dev300/retired test stay
sealed. Any gain also changes compute and cannot be attributed purely to prompt
or thinking. Future fixed/adaptive comparison must use the same module before
making a tool-loop claim; this package does not authorize that comparison.
