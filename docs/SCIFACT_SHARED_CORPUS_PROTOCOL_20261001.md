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

## Measured closeout: the single approved attempt

Source was frozen/pushed as `6fd92eb0ac91fab206222d321c86448046dd2879` before
execution. Six new synthetic tests passed (2.55 s); targeted Ruff and strict mypy
with `--follow-imports=silent --platform linux` passed for the three explicitly
checked new Python files. This is not a claim that historically untyped imported
test/smoke modules pass strict checking. The new wrapper passed syntax and its
own exact-source guard positive/incorrect-wrapper-hash negative checks; the
packager's separate legacy guard was not used as proof of the new wrapper.

[Submission](verified-runs/scifact-shared-cpu-submission-31831831.json) and
[closeout](verified-runs/scifact-shared-cpu-closeout-31831831.json) record actual
job **31831831**, **COMPLETED / exit 0:0**, **115 s**, 106.570 CPU-seconds,
Slurm batch MaxRSS **305,392 KiB**. The process's 313,476 KiB peak is a distinct
measurement. Initial Priority cleared without cancellation/resubmission. The
precheck ID 31831827 was not a submitted job, and its start estimate was not a
guarantee. Exactly one 1-CPU/4-GiB allocation, zero GPU/model calls or updates.

The [byte-preserved compact result](verified-runs/scifact-shared-cpu-31831831.json)
has SHA `27ad7cec3bb9aa1d9c12cd8618d9f9546956ed8b171e0483950333a3959c686d`.
All five private artifact hashes were rechecked on Spartan; the final log JSON
equals the compact. Gold SHA and original 48 ID/component hashes are unchanged.
Only compact/protocol/receipts were copied; private frames/targets remain remote.

| Same 48 previously exposed FIT claims | Old 256-document pool | Shared 5,183-document pool |
|---|---:|---:|
| Initial complete document witness | 48 | 45 |
| Read → complete post-read witness | 0 | 1 |
| Context-limited abstain claims | 0 | 2 |
| Decision records | 93 answer | 90 answer + 1 read + 2 abstain |
| Retained mass / gaps | 48 at 1 / 0 gaps | 48 at 1 / 0 gaps |
| Whole-gold coverage in teacher target | 47/48 | 45/48 |
| Total sequence tokens | 421,453 | 397,041 |
| Largest joint sequence | 6,482 | 5,877 |

The executed-read count is **one claim**, not a count inflated by alternative
records: here it also produces exactly **one read decision record**. It adds
**19 citable sentences**, followed by a complete authorized rationale; no read
has zero increment or a gap. There are 50 scripted capture responses because
the unchanged teacher probes initial state then replays the two-state read path;
these are not 50 model calls or independent examples. All no-read/abstain reports
and frames are retained. The two context-limited abstains do not assert official
NEI or make unannotated retrieved documents negative.

The predeclared matched anchor returns to rank **7**. Candidate order, initial
and post-read canonical state hashes, actual rendered prompt hashes, token counts
and read IDs all match the older full-corpus capture. This supports compatibility
for that anchor, not a claim that this mechanism alone explains every old/new
case or that a model improved. Pool membership and BM25 statistics changed together.

Exposure aggregates are deliberately separate:

- **5,183 indexed documents**; **836 unique retrieved documents**.
- Prepared-prompt text also contains **836 documents**: known owners FIT 86,
  tune 7, validation 5, unowned 738. This is shared public text; corresponding
  protected questions and annotations were not opened. Do not call these
  source-family-unseen samples or globally decontaminated evaluation.
- **232** documents appear as citable text and **647** as previews, with overlap;
  they must not be added as disjoint counts. There are **49 distinct prompt
  hashes** and **2,207 distinct citable sentences** across 93 prepared records.
- Supervised answer targets cover **49 documents**, all known FIT-owned; the
  one read target is one of them. A target annotation is not inferred for every
  indexed, retrieved, visible or unowned document.

Decision: technical supervision readiness is true under this explicitly shared
protocol, but **one natural read among 48 claims is very limited policy coverage**.
This is not a trained adapter, Agent success rate, independent-test result or
generalization gain. No resampling, altered context, new GPU/optimizer run,
held-out evaluation or resume update followed. Any next training/data decision
requires a separate bounded authorization; the future group-mean interface stays
unexecuted on these real records.
