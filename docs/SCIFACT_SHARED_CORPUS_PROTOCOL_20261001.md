# Precommitted shared-corpus FIT supervision protocol

This approved CPU-only experiment asks whether the same original 48 FIT claims,
retrieved against the production 5,183-document public SciFact corpus, naturally
yield read → controller feedback → answer supervision. This is preparation by a
program teacher, not model tool success or training. One CPU attempt, no automatic
retry or search over corpus/context/teacher configurations; every negative/gap
is retained. No GPU/model generation, adapter training or held-out evaluation.

## Fixed versus changed

Only the candidate pool changes from 256 FIT-family documents to all 5,183 public
corpus documents (and consequently the BM25 corpus statistics). The original
ordered 48 FIT IDs/hash, claim/component assignment, full gold/rationales,
tokenizer, BM25 Top-20, CommonPacking, action schema, budgets, program teacher,
record cap and record weights remain fixed. No context shrinking, positive
hiding, additional claims, seed change, reranker/model calls or quota fishing.
The existing 256-pool 93-answer bundle remains byte-for-byte unchanged.

The new effective protocol is `scifact-shared-corpus-fit-state-v1-20261001`.
The existing v2 implementation and **record schema** are reused, not presented as
the new data origin. Each record has explicit shared-corpus `data_provenance`,
protocol hash, corpus hash and pool identity; its full record hash is recomputed.
These fields never enter `frame/observation` or prompt tokens. The wrapper checks
each record's protocol against the effective configuration in addition to the
existing token/weight validators. Historical v2 config/artifacts/helper stay intact.
The future claim-group-mean optimizer is not executed by this experiment.

## Data access and exposure accounting

Public corpus SHA is `5df96a227847260d2877931d62e58288472902db605c5eacb31d9d9703165f76`.
Full original FIT annotations are **reused**, not resampled/reannotated, from
the prior verified `complete-fit-gold.json`:
`8daf2e74e74ec7f656be7c0b8234d7e36b7581f969c35b20a52554c5fb78e72f`.
The previous compact (`3cfc8acf…`) binds its source train-member hash, original
FIT IDs (`22fab53…`) and component identities. The old row-to-complete-gold/family
join is rechecked. No non-FIT gold archive scan is necessary. Tune, unused
validation, official dev/test questions/labels are not opened, selected or scored.

This protocol **does not claim source-family-unseen**. Existing fit/tune/validation/
unowned document mapping is descriptive only, never a safe/negative label.
The following levels have private identity hashes and separate public aggregates:

1. All indexed public documents and source hashes.
2. Retrieved candidate identities/order, distinct from actual prompt text.
3. Prepared-prompt citable sentences and preview strings as actually rendered,
   plus unique prompt/sentence counts and retained-record occurrences.
4. Supervised target documents/actions (including read targets), kept separate
   from indexed/retrieved/visible documents. Context-limited abstain is not NEI.

Captured runtime states, including gap-only states, are stored separately from
retained prompt records. Raw observations preserve feedback and remaining budgets.
Record trajectory/alternative/state joins and capture counts preserve the real
`run_bounded_slot/CommonPacking` read transition. These are captured observations
and supervision targets, **not a separately persisted full controller event log**.
No CPU exposure count means that a model has actually trained on that text.

## Predeclared comparison and reporting

All 48 reports are aligned by exact claim order and component before aggregate
comparison. Report initial complete witnesses, natural reads/post-read witnesses,
context abstain, all gaps and retained mass, record/action counts, largest joint
sequence and total tokens. The unique old full-pool scripted-read FIT-overlap case
is the fixed anchor: compare candidate order/rank, canonical states, actual rendered
prompt hashes/token count and read IDs. Post-read equivalence requires the same
read IDs. A matched anchor is not evidence for all claims or a model improvement.

Raw frames/gold/records/three-layer identities stay on Spartan. Git receives only
code/protocol, compact anonymous summaries and execution/source receipts. Any gap
or absence of reads is reported, not repaired by resampling or changing budgets.

## One execution gate

After narrow synthetic tests, targeted Ruff/mypy and coordinator review, freeze
and push source before execution. Reuse the exact Git source packager with
`core.autocrlf=false`, `core.eol=lf`, `tar.umask=0022` and one expanded
`SOURCE_REVISION`. Bind/verify the **new CPU wrapper**, separately from the
packager's legacy wrapper guard. Check duplicate jobs; `sbatch --test-only` then
one actual job: 1 CPU, 4 GiB, at most 10 minutes, no GPU, no requeue. The previous
256-pool run took 112 seconds; the larger-pool single attempt uses the approved
bounded 600-second ceiling, not an unmeasured throughput claim. Unique source-SHA
output directory, no overwrite or automatic replacement attempt. Infrastructure
failure requires diagnosis and a new decision before another submission.
