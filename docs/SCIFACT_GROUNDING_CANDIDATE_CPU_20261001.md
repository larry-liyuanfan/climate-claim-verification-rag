# One generator-grounding candidate — CPU implementation, not execution release

## Scope

The completed component diagnostic supports inspecting document relations and
rationales. It does not establish an Agent gain. This package adds one small
joint grounding SFT candidate (including the terminal `answer` field), **not tool
action trajectories**, and does not train or call the real generator.

Existing Windows Torch 2.7.1+cpu and Spartan Linux/POSIX Torch 2.1.2 are available;
no system Python, WSL or shared Torch replacement is needed. `USE_TORCH=0` is
intentional for tokenizer-only preparation. Optional candidate dependencies are
pinned in `requirements-grounding-candidate.txt`; this is not permission to
replace the Spartan module or shared runtime.

## Frozen candidate design

- Reuse the prior SHA-bound eligible TRAIN claim components. All 531 eligible
  claims had earlier gold-preparation exposure; these are not an external test.
  Accumulate actual, failed and uncertain attempts; carried synthetic calls are
  deduplicated by physical identity, not counted as real claims.
- Select one deterministic representative per component before packing: 48 fit
  (24 support / 24 refute), 12 tuning and 12 TRAIN-validation (4/4/4 label strata).
  Tune/validation exclude consumed/uncertain components and each other. Fit can
  include consumed TRAIN. Insufficient strata stop rather than change the seed.
- Restore the missing document-family map from the frozen corpus using the
  original token-Jaccard 0.9 function. Do not call the train/dev grouping routine,
  read official dev/test, or change claim components. Cross-component family
  ownership disagreement fails closed. Unowned corpus families are excluded;
  unused eligible components provide deterministically partitioned background
  candidates. This is a restricted TRAIN retrieval pool, not the full-corpus
  retrieval benchmark. Consumed background families go to fit only.
- Fit uses full officially annotated documents, at most four document/alternative
  records per claim, at most 192 overall. A target is one complete official
  alternative, never the union of alternatives. All source sentences remain in
  context. No arbitrary unannotated candidates or non-rationale sentences become
  human negative labels. Weak/constructed NEI training examples are omitted.
  Consequently the candidate has no NEI-target training; NEI evaluation measures
  the possible cost of this choice rather than presuming improved abstention.
- Evaluation freezes BM25 top five from the appropriate family partition, full
  contexts, claim and schema **before gold availability statistics**. No relation,
  selected-gold marker or oracle target enters model inputs; no gold backfill.
  Over-budget context/target is a recorded gap, not cropped or replaced.

## Production seam and one-config training

Both target and output use `scifact_terminal.parse_action → render_answer →
to_original_prediction`. Labels are `SUPPORTS/REFUTES`; sentences are `cN:original
index`; NEI is legitimate empty-evidence abstention, not a document NEI label.
Limits remain five documents, eight sentences/document, twenty total, with
ordered first-three / complete-alternative scoring. The wire is the existing
bare terminal **gap=False**, not the semantic-G envelope or its sufficiency prompt.
No gold-derived retrieval-need or sufficiency decisions are manufactured.

Training masks the entire system/user/generation prefix; only canonical assistant
JSON and EOS carry loss. Full tokenization must retain the exact inference token
prefix. Target bytes, token IDs and loss mask are hashed after canonicalization,
so JSON serialization order cannot silently alter targets. Batch size one avoids
padding/EOS ambiguity. CausalLM restoration verifies every adapter tensor value;
the dense bare-model loader is not reused.

One seed `20261001`, LoRA rank 8 / alpha 16 / dropout 0, q/v projections, AdamW
1e-4, batch one / accumulation four, one epoch or 64 optimizer steps whichever
comes first, one final checkpoint. At 192 records this is at most 48 optimizer
steps. No best-checkpoint selection, model/seed sweep or automatic retry.

## Paired evaluation and costs

Tuning is twelve identical inputs × base/adapter = 24 calls. Compare using the
same loaded CausalLM and grammar, disabling the adapter for base. No retries or
real warmup/preflight calls are outside the cap. Only a preregistered tuning
count gain in correctly rationalized documents, no additional NEI false evidence,
no additional invalid/failed outputs, known costs and unchanged inputs can open
the separate 24-call TRAIN-validation release. Total maximum is 48 calls.

Persist reservations before calls, retain raw responses / usage on failure,
stop on unknown costs and infrastructure errors. Independent post-exit scoring
checks raw parsing, original identities and physical files. Partial arms retain
all 24 planned slots, unattempted counts and known cost lower bounds; failures
never become valid NEI abstentions and cannot open validation. A missing durable
summary requires reconciliation, not replay or an invented zero-cost completion.

Any later four-route Agent comparison must use **one shared frozen generator and
adapter state across all four arms**. This fixed-input grounding comparison alone
cannot establish tool-policy improvement or independent generalization.

## Reproduction entrypoints

On Spartan, an exact Git-LF source archive and unchanged tokenizer files:

```bash
python scripts/prepare_scifact_grounding_candidate.py \
  --output "$CLIMATE_ROOT/posthoc/scifact-grounding-candidate-<source12>" \
  --tokenizer-dir "$CLIMATE_TOKENIZER" --source-git <40-char-SHA>
```

The CPU package ends after this command, tests, hash/shape/dependency checks and
handoff. The commands below are implemented but **must not be executed without
a new exact-hash coordinator release and GPU allocation**:

```bash
python scripts/run_scifact_grounding_candidate.py train \
  --bundle <private-preparation> --data-sha <manifest-SHA> --output <unique-train-run> \
  --model <frozen-Qwen3-4B> --model-manifest <model-manifest.json> \
  --release <separately-approved-train-release.json> --release-sha <SHA>
python scripts/run_scifact_grounding_candidate.py evaluate --partition tune \
  --bundle <private-preparation> --data-sha <manifest-SHA> --output <unique-tune-run> \
  --model <frozen-Qwen3-4B> --model-manifest <model-manifest.json> --adapter <unique-train-run> \
  --release <separately-approved-eval-release.json> --release-sha <SHA>
python scripts/run_scifact_grounding_candidate.py score --partition tune \
  --bundle <private-preparation> --data-sha <manifest-SHA> --output <unique-tune-run>
```

Validation uses the same entrypoints with `--partition validation` and the
hash-bound tuning gate. Release binds source, config, data, trained checkpoint,
output directory and reviewed runtime bound. No runtime estimate is inferred
from inference-only GPU usage; training requires an allocated shape/memory pilot
and a separately reviewed walltime. Private targets, IDs, manifests, cache and
weights remain on Spartan; publish only compact aggregates and hashes.

## Validation status

Implementation checks before real-data preparation: 150 related CPU tests pass,
including a tiny synthetic CausalLM/PEFT checkpoint test (no downloaded weights,
no training or generation); Ruff and strict mypy on the five new source files
pass. Final exact-source clean checkout and real-tokenizer preparation receipts
are to be recorded below after execution; they are not yet quality results.
