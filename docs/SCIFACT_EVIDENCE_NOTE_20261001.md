# Same-evidence note → terminal diagnostic (CPU candidate)

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
No real new result is claimed. Raw outputs stay on Spartan. Old12 are exposed
TRAIN (one current FIT component overlaps); validation12/dev300/retired test stay
sealed. Any gain also changes compute and cannot be attributed purely to prompt
or thinking. Future fixed/adaptive comparison must use the same module before
making a tool-loop claim; this package does not authorize that comparison.
