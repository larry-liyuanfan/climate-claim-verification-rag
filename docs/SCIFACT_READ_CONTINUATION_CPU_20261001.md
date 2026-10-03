# Frozen scripted-read conditional continuation — CPU receipt and bounded submission

## Question and boundary

After the unchanged controller is scripted to perform its **three already
frozen read witnesses**, can the existing active adapter produce a correct
document answer from the exact second observation? This isolates a conditional
answering failure from choosing to read. The script chose the reads using prior
gold-aware opportunity preparation; success would **not** show autonomous
Agent behavior, generalization, or an independent test. Keep one FIT-overlap
case plus two not-direct-FIT cases. Do not select new cases or witnesses.

CPU preparation passed. A subsequent separate exact release authorized one
three-call job, **31781591**, submitted once and now completed. The
[terminal closeout](SCIFACT_READ_CONDITIONAL_CLOSEOUT_31781591.md) reports 3/3
document labels but only 1/3 complete rationales after scripted read, with
zero model tool execution. Repairs, warmup, training and validation execution
remain unauthorized. The prior 48-slot regression and its negative score are
unchanged. The DRAFT/CPU/submission sections below preserve their historical
checkpoint status; they are not the current scheduler state.

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
ordered citations, unknown/torn records, and injected stale predictions.
A mismatch stops preparation rather than substituting a near-match input.
Current resume and shared career state are not edited.

## Verified CPU delivery (2026-10-01)

The real-tokenizer reconstruction succeeded for all **three** frozen second
states: observation/schema/state/physical-prompt hashes and historical token
counts match; ordered JSON survives a physical write/read roundtrip. Token
counts are **1,868 / 1,648 / 2,528**. The predefined split is **one FIT-overlap +
two not-direct-FIT** cases, all previously exposed TRAIN. Preparation used six
scripted fixture responses and three scripted reads, **zero model/reranker
calls**, no gold targets or validation input. MaxRSS was 276,532 KiB for this
small CPU preparer; it is not a model-serving memory benchmark.

Exact execution source remains `fc4ffd61a7615682911556838c66f47aa33d0293`,
even when subsequent evidence-only commits update this document. The archive
was checked against all 498 Git blobs and 13 directories, with one 41-byte
source marker. Affected tests passed **33/33** locally and in a clean export;
Ruff, Linux-targeted strict mypy, shell syntax and targeted secret checks passed.
No unchanged full suite was rerun merely for this evidence-only handoff.

The [redacted readiness receipt](verified-runs/scifact-read-continuation-cpu-20261001.json)
binds source/archive/wrapper, prepared inputs, ID-only membership, runtime and
DRAFT hashes. CPU validation accepts the DRAFT as a proposal and **rejects it
at the execution entrypoint**; the fixed output was unused at that check. No
`sbatch` submission or model call is part of this CPU delivery.

| Artifact | SHA-256 |
| --- | --- |
| Source archive | `727c596ae4a808d9bded5fce9a096a766cc8887280650771b93472f99adfb56e` |
| Wrapper | `da6de837ce4567c76e408245fcfeff86a002271e76b6d4ea411cde1a8ec3a931` |
| Preparation compact | `8951722f3bf505fd322a2d886178229138f168b59b3e03a53a6bdaf52a1136ab` |
| Ordered private inputs | `281313c1910f4775c69d78c6df6582d314a6854c8abb4f8d7c4196c74335a0a4` |
| Private membership | `02372ea0570c176e3fb1dbcccae02ff189407342d2e684aefc520f7f21ddbf95` |
| DRAFT release (5,745 bytes) | `2ccab922b735b84f0400edd737e0b05bca5b6d39be5b3698ff797f1e39cfec8e` |
| Remote CPU readiness | `db7152249404c4da0e63e746fe75db6ee5638ab9f4b86c12edf30f582ea368a4` |

The DRAFT and runtime check live under project-owned
`envs/read-continuation-source-fc4ffd61a761`; private ordered inputs and their
compact manifest live under `posthoc/scifact-read-continuation-fc4ffd61a761`.
Only redacted evidence is committed here, not claim text, IDs, gold or weights.

### Torch versus POSIX: no missing production dependency

Local Windows validation uses the existing `.venv-validation` with
**Torch 2.7.1+cpu**; a tensor smoke check passed. POSIX `resource` is an
operating-system interface, not a pip dependency to install on Windows. The
Linux operator is validated and executed on Spartan, where actual imports
matched the frozen receipt: **Python 3.10.4, Torch 2.1.2, `os.name=posix`**,
and `resource.RLIMIT_AS` is available. No model was loaded for this check.

Two distinct diagnostics must not be conflated. Tokenizer-only preparation
deliberately sets `USE_TORCH=0`, so Transformers can warn that no model backend
is enabled. Separately, an ad hoc readiness command accidentally replaced the
module-provided `PYTHONPATH`, making Torch unresolvable. Repeating only that
failed import check with the module path preserved passed. The frozen Slurm
wrapper already preserves `${PYTHONPATH:+:$PYTHONPATH}`; no installation,
production-source change or new runtime version was needed. The readiness
check verifies actual imports and existing manifest hashes, without claiming
to have rehashed every runtime file again.

## Separate exact release and single submission

The coordinator released only the source/input/resource contract above. The
original DRAFT was retained. Replacing its unique authorization literal, with
all other bytes unchanged, produced **5,755 bytes** and SHA
`6638761114eb51b98b5f8db95121f13ae05c873fdd2c69e9abb5896382ea8d69`.
No source archive was rebuilt after the evidence-only documentation commit.

The [submission receipt](verified-runs/scifact-read-continuation-submission-31781591.json)
records the actual job ID **31781591**, distinct from the test-only simulation.
An exclusive reservation, unused-output check and empty matching-job queue
preceded `sbatch --test-only`; it passed before a single actual `sbatch` call.
Inherited `SBATCH_*` overrides were removed (none were present). Slurm confirmed
normal QoS, Nice 0, no requeue and the exact 1 A100 / 8 CPU / 32 GiB / 30 GiB
scratch / 900-second resource contract.

The project fileset had 192.8 GiB used against 466/467 GiB soft/hard quota,
525,019 of 1,000,000 files. This is a fileset quota observation, not filesystem
free space. The account's home quota was full; the frozen wrapper already
routes model caches/temporary files to allocated scratch and outputs to the
project fileset, and no home data was deleted or modified.

Initial scheduler state was `PENDING (Resources)`, priority 13,418. The raw
estimated start was `2026-10-01T06:45:29`; its scheduler timezone was not
independently established, and it is not a start guarantee. No pending-job
cancellation, duplicate submission, new monitoring automation or additional
model experiment was created. Terminal scoring and physical audit were pending
at that submission checkpoint; both have since completed in the terminal report.

The bounded queue-time [observation-supervision proposal](SCIFACT_OBSERVATION_SUPERVISION_PROPOSAL_20261001.md)
maps the existing training-context mismatch to specific modules and tests. It
is documentation only, not a change to this job or permission for another run.
