# Frozen scripted-read conditional continuation — CPU DRAFT

## Question and boundary

After the unchanged controller is scripted to perform its **three already
frozen read witnesses**, can the existing active adapter produce a correct
document answer from the exact second observation? This isolates a conditional
answering failure from choosing to read. The script chose the reads using prior
gold-aware opportunity preparation; success would **not** show autonomous
Agent behavior, generalization, or an independent test. Keep one FIT-overlap
case plus two not-direct-FIT cases. Do not select new cases or witnesses.

Status: CPU preparation only. No GPU submission, model call, repair, warmup,
training or validation execution is permitted without a separate exact release.
The prior 48-slot regression and its negative score remain unchanged.

## Reconstruct the actual input, not a similar prompt

`prepare_scifact_read_continuation.py` reuses `ReadReplay`, the original
candidate order, aliases, tokenizer, `run_bounded_slot(ARMS[0])`, `CommonPacking`
and scripted read→abstain history. A capture callback preserves the second
observation/schema in memory without changing normal replay behavior.
All original metadata and observation/schema/state/prompt SHA values plus prompt
token count must match `states-before-gold.json` SHA
`5a49b53aa6ab7790231b224b5966f3e8e6f2fff1856219c1c177af16ded714b8`.

Canonical state hashing alone is insufficient: the renderer respects nested
JSON insertion order. The new exclusive input file therefore uses **ordered
JSON**, then is reloaded and revalidated against the old actual prompt hash.
Token-ID hashes are newly recorded; the historical file only recorded token
counts, so there is no claim of a pre-existing token-ID hash comparison.
Disk JSON integer/string document keys are normalized only for metadata
equality, never for model prompt reconstruction.

The preparer reads only the fixed claims/corpus, prior selected witness
metadata, tokenizer and ID-only FIT split. It does not read gold targets or
`fit/records.json`, and does not examine unused validation queries. Prior
selection was gold-aware: this is process/dataflow discipline, not evidence of
unseen inputs or an account-level sandbox.

## Proposed execution (not released)

Reuse `ActiveAdapterProvider`, exact checkpoint/tensor restoration, physical
wire receipts, and a separately bound conditional policy. Each original case
has an exclusive reservation and **one** `generate` call. Total cap is **3**,
zero repairs/retries/warmups/reranker calls. It is not run through another
complete Agent episode. The observation's remaining-call/tool fields and
original schema stay unchanged; read/rewrite/rerank remain offered.

An answer uses the original `parse_action → render_answer →
to_original_prediction`. Another tool proposal is recorded but never executed.
Schema validity is separated from controller legality: repeat read and invalid
rewrite use the original loop/constraint rules without side effects. Invalid
wire, nonterminal proposals and missing records retain the three-case denominator;
they are not relabeled as valid abstentions. No failed call is retried.

Each call has a **new separately declared 120-second cap**, not an invented
remainder of the historical episode. Proposed resources: 1 A100, 8 CPU, 32 GiB
RAM, 30 GiB scratch, 15-minute Slurm limit; worker hard cap 480 seconds, outer
870 seconds plus 15-second kill grace, scorer 90 seconds. The previous full
48-call/two-model allocation took 201 seconds and 16.54 GiB MaxRSS. The new
shape retains safe memory/staging headroom but shortens walltime. Reusing the
verified extractor stages both model directories; **only the generator is loaded**.

The worker exits/is reaped before frozen scoring inputs become available.
Costs and physical-file hashes are written before gold inspection. Complete
records pass exact raw→parse→render→prediction equality and active-adapter /
actual-prompt / schema binding. Truncated slot JSON yields unknown/lower-bound
costs and blocks quality. Invalid raw cannot carry a stale nonempty prediction
into scoring. Official document/rationale metrics and the predefined 1+2
subgroups are reported separately from scripted read and autonomous success.

## Entry points and CPU checks

- Preparation: `scripts/prepare_scifact_read_continuation.py`.
- Inference: `scripts/run_scifact_read_continuation.py`.
- Exit-bound orchestration: `scripts/run_scifact_read_continuation_operator.py`.
- Cost-first scoring: `scripts/score_scifact_read_continuation.py`.
- Wrapper: `hpc/scifact_read_continuation.sbatch` (DRAFT only).

Fixtures cover disk roundtrips and prompt-order rejection, every original hash,
single-call reservations, all three tool proposals without execution, original
ordered citations, unknown/torn records, and injected stale predictions. The
real-tokenizer three-state receipt, exact source and release hashes follow only
after CPU reconstruction succeeds. A mismatch stops preparation rather than
substituting a near-match input. Current resume and shared career state are not edited.
