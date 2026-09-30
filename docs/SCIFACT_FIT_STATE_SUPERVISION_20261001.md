# FIT-only production-state supervision (CPU preparation)

## Status and scope

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
