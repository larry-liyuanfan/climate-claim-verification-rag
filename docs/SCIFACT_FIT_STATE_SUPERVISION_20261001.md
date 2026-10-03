# FIT-only production-state supervision (CPU preparation)

## Status and scope

**Completed CPU job 31789062, exit 0:0.** The original 48 FIT claims yielded
93 records with mass 1 per claim and no gaps. However, **all 93 targets are
answers: zero read/abstain supervision arose naturally**. This bundle repairs
grounding context/weighting but does not teach or demonstrate adaptive tool
selection. No training or evaluation followed. See the measured closeout below.

This is a **new supervised-data contract**, not a trained model or quality
improvement. No generation, GPU allocation, training, tune/validation/dev/test
evaluation or new claim selection is authorised by this entry point. The
prior negative adapter regression and three-state conditional result remain
unchanged. No current resume or other project's files are changed.

The original 48 FIT claims and family/component partition are fixed. Their
old 96 records contain at most four document/rationale alternatives per claim;
that cap is not a complete annotation view. In the old data, 19/17/5/7 claims
had respectively 1/2/3/4 equally weighted records, and 90/96 targets had one
sentence. These distributions describe supervision, not the cause of any
particular prediction error.

## Narrow data access and identities

`prepare_scifact_state_supervision.py` first checks the old manifest and only
four allowlisted files: FIT records, split ID metadata, frozen document-family
ownership, and gold-free corpus. It does not run the old preparer, regenerate
splits, rebuild family assignments, open scoring files or load reserved queries.

The locked archive `scifact-semantic-inputs-c4907db6.tar` contains original
TRAIN annotations. Its outer SHA and original preparation-manifest SHA are
checked; the exact gold member must be unique and a regular file. The
`IdScanner` scans JSONL syntax and top-level ID/key tokens, including escaped
keys. Non-FIT values are not deserialised or retained. Only the fixed 48 raw
lines are retained, and **the complete member SHA plus unique exact ID coverage
must pass before those lines are deserialised**. Thus non-FIT container bytes
are streamed/hashed; this is not a claim that no non-FIT bytes were read.
The outer archive hash currently reads the archive as bytes, within the 4 GiB
limit; the JSONL selection itself is streaming and bounded.

All original alternatives for those 48 claims survive. Every old claim,
document, alternative index, sentence order and label is joined to that view.
All evidence/cited document families must belong to the same original FIT
component. The candidate pool restores the entire frozen FIT family partition,
including background documents; it is **not** a gold-document-only pool.

## Production observations and teacher decisions

BM25 top-20 order feeds the unchanged `run_bounded_slot`, `CommonPacking`,
production schemas and real tokenizer. A scripted provider captures actual
observations rather than inventing a training-only context. Distractors,
aliases, previews, feedback and remaining budgets are preserved. The program
teacher's labels/reasons/provenance never enter the model input.

- A complete visible rationale for any annotated document permits a direct
  answer, even if another gold document was not retrieved. Visible multi-doc
  decisions are retained; alternative rationales remain OR alternatives, never
  a sentence union. Whole-claim coverage is reported separately.
- Otherwise, an available preview or partially citable document can be read
  through the actual controller. A subset read may reveal later sentences of
  a selected-but-truncated document; identical-selected/read-loop actions are
  not used. The resulting feedback and budgets are captured before answering.
- No complete rationale after that read, over-budget targets, or unresolved
  read opportunity produce explicit gaps, not partial fabricated evidence.
- If no annotated positive is in the frozen candidate set, a context-limited
  abstain target has **no official NEI label**. This does not label unannotated
  documents negative or establish factual insufficiency.

Teacher calls are program operations, not model tool use. A demonstrated read
is one possible action, not the only correct read. This is oracle-supervised
state construction, not DAgger or evidence of autonomous useful tool selection.

## Weights, loss and caps

Historical v2 contract is preserved below. A separately versioned
[prospective claim-group-mean interface](SCIFACT_CLAIM_GROUP_MEAN_20261001.md)
now has synthetic CPU AdamW/gradient validation; it does not retroactively change
these artifacts or authorize training.

Maximum 192 decision records, four per claim; those are ceilings, not targets.
One claim has mass 1, divided before tokenisation over its trajectory,
alternative targets and decision states. Duplicating the read frame for two
alternatives or splitting read/answer cannot double the claim's mass. Dropped
records do not donate mass to survivors; any missing mass blocks training.

The tested backward implementation is assistant-token mean causal CE times
record weight **divided by fixed global denominator 48**. Prompt and tool
feedback positions are -100; only current assistant JSON and EOS are trained.
EOS equal to PAD is handled positionally. Actual prompt-prefix token equality,
BPE boundary, EOS, full target length and token/mask hashes are checked; no
target truncation. Physical JSON preserves observation/schema key order.

A future trainer must accumulate complete claim groups of at most four and
use the same denominator for a short last group, without per-chunk weight-sum
normalisation. All 48 ready claims imply 12 optimizer steps, versus the old
96-record/4 accumulation's 24. The objective weighting, gradient scale and
update trajectory therefore differ; equal claim mass is **not** optimization
equivalence and does not justify copying a learning rate without review.

## Reproduction and resource bounds

- Entry: `scripts/prepare_scifact_state_supervision.py --source-git FULL_SHA
  --source-archive SOURCE.tar --source-sha ARCHIVE_SHA`.
- CPU wrapper: `hpc/scifact_state_supervision_cpu.sbatch`, one CPU, 4 GiB,
  10 minutes, no GPU, no requeue; run `sbatch --test-only` before one submission.
- Exact source archive/tree verification precedes data processing. A new
  exclusive output under the Climate `posthoc` directory prevents overwrite.
- Full FIT gold, actual frames, records and per-claim gaps remain on Spartan.
  Publish only compact counts/hashes, resources and boundaries.
- Local Torch tests use `.venv-validation`; the lightweight lint `.venv` has
  no Torch. POSIX resource limits run on Spartan Linux, not native Windows.
  `USE_TORCH=0` intentionally disables model imports for tokenizer-only prep.

Synthetic tests cover actual selected-document rereads, missing other gold
documents, alternative OR semantics, inaccessible rationale, policy abstention,
input/schema contamination, EOS=PAD, BPE-prefix/length rejection, safe ID-first
selection, narrow file reads, archive-member guards, and actual Torch gradients
for alternative/state duplication and a short last group. Old adapter and
read-continuation contracts are regression-tested without using real held-out
examples.

## Future training proposal — not released

After the single CPU preparation, report actual records, total/max sequence
tokens, gaps and claim masses. Only a complete, reviewed bundle may support one
new bounded checkpoint proposal. Reuse the established q/v LoRA architecture;
do not sweep ranks, seeds or models. Token totals and recent measured training
Elapsed/MaxRSS must size a separate allocation before any submission. There is
no GPU release in this document.

Do not automatically run old terminal-only validation: it allows only one
answer/abstain call and cannot establish Agent benefit. A later independently
reviewed production-agent comparison must distinguish grounding improvement,
more tool calls, and genuine same-model policy benefit. Reserved IDs and source
families remain unchanged and unconsumed here.

## Measured single-preparation closeout

Execution source `9bf5fdad2ec9b010cb6a305b3e3462e0c7ac0ab1`; exact archive
`f4079a80d4e74a8d8f988cc04a7bbb787de7b6b4eb6b6c6c2e3282ab7e78dab9`.
Source/package verification and test evidence are in
[the source receipt](verified-runs/scifact-fit-state-source-9bf5fdad2ec9.json).
The first non-exact archive was rejected before extraction/submission. The
replacement came from the existing exact-source packager: original Git bytes
and modes, 510 files, 13 directories and one 41-byte revision marker. The
legacy packager shell-guard check is not represented as CPU-wrapper execution;
the CPU wrapper separately passed syntax and hash checks, test-only scheduling,
and real source-tree verification in the single successful job.

The [byte-preserved compact artifact](verified-runs/scifact-fit-state-31789062.json)
has SHA `3cfc8acffcb001488ce841310349fa05a806a2216d66336afe331d9cd04e1bde`.
Its four private artifact hashes were rechecked on Spartan; its content equals
the successful job's final log record. The
[closeout receipt](verified-runs/scifact-fit-state-closeout-31789062.json)
records exact remote paths, accounting and log hash. Neither original gold nor
actual observation/record text was copied to GitHub or the local checkout.

| Quantity | Measured value and boundary |
|---|---|
| Original FIT claims / components | 48 / 48, identities unchanged |
| Corpus / frozen FIT-family pool | 5,183 / 256 documents; not full-corpus retrieval |
| Initial complete document witness | 48/48 claims, program audit not model accuracy |
| Whole-gold coverage in teacher target | 47/48 claims |
| Records / action distribution | 93 answers; 0 reads; 0 abstentions |
| Record count per claim | 20/17/5/6 claims have 1/2/3/4 records |
| Answer document counts | 83 one-doc, 4 two-doc, 6 three-doc records |
| Rationale lengths | 103 single-sentence and 6 two-sentence document targets |
| Complete annotations beyond old record cap | 3 claims, not every four-record claim |
| Alternative-combination cap | 3 claims; complete per-doc rationales not truncated |
| Claim mass / preparation gaps | 48 at exactly 1; zero gaps |
| Sequence tokens | 418,425 prompt + 3,028 target = 421,453 total; maximum joint length 6,482 |
| Historical sequence tokens | 106,927 + 2,791 = 109,718; different contexts/loss and step count |
| Scripted capture / model calls / updates | 48 fixture responses / 0 / 0 |
| Slurm Elapsed / TotalCPU | 112 s / 102.249 CPU-s; one CPU, no GPU |
| Slurm MaxRSS / process ru_maxrss | 249,132 KiB / 257,076 KiB; distinct measurement sources |

Actual cited retrieval ranks are 1/2/3/5, with 89/12/6/2 document targets;
sentence positions range from 0 to 17. This improves representation of real
candidate order and multiple documents, but still has substantial top-rank and
single-sentence concentration. Do not relabel it balanced action supervision.

**Decision:** stop after CPU closeout. Technical data readiness is true, but
Agent-policy readiness is false. Do not perturb retrieval, hide positives,
resample claims, add NEI or consume reserved validation to manufacture missing
read examples. A separately authorised decision is required to change that
training question. If proceeding strictly as a grounding-only experiment,
the 421,453 tokens are approximately 3.84 times the old token count and imply
12 whole-claim optimizer steps rather than 24 record steps. The old measured
training allocation took 92 seconds / 9,227,040 KiB MaxRSS. Linear token scaling
alone is not a runtime or memory guarantee because sequence length also changes.

A conditional future budget proposal is one A100 / 4 CPU / 32 GiB / at most
15 minutes, with a longest-record forward/backward resource preflight inside
that same allocation and no automatic retry on failure; the preflight update
must be accounted for, not silently added to the 12-step budget. This is a
proposal only, pending review of the changed learning-rate/normalisation scale
and a clear choice to pursue grounding rather than claim unsupported Agent
benefit. **No GPU job or new checkpoint has been launched.**
