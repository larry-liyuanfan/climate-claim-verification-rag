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
`planned_unsuccessful` includes both failed and unattempted planned slots; it is
not an actual model-error rate. A persistence-failed truncated response retains
its physical hash and durable failure/usage but contributes no quality evidence.
Normal/valid responses must still parse and pass the raw-output audit.

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

Preparation source `40d84a377bd1223de5817a69818ac5f5aa63b384` passed 150 related
tests in its exact Git-LF clean source export (7.76 seconds), using the existing
isolated dependency environment. This includes a tiny synthetic CausalLM/PEFT
checkpoint test with no downloaded weights, training or generation. It is a
clean-source reproduction, not a claim of freshly resolved dependencies.

The [CPU preparation receipt](verified-runs/scifact-grounding-cpu-preparation-40d84a3.json)
and [source receipt](verified-runs/scifact-grounding-cpu-source-40d84a3.json) bind
the real tokenizer/data preparation separately from the later persistence-audit
fix. No data were reselected or rerun for that execution-only fix.

| Frozen property | Verified preparation result |
|---|---|
| TRAIN claim components | 48 fit / 12 tune / 12 validation; no gaps |
| Official alternative-aware fit records | 96: 45 SUPPORTS / 51 REFUTES; no weak NEI |
| Fit input / target tokens per record | 698–1,853 / 28–35; totals 106,927 / 2,791 |
| One-epoch optimizer steps | 24 derived from 96 records / accumulation 4; not executed |
| Tune / validation inputs | 12 / 12; 838–4,783 / 2,453–5,175 tokens |
| Tune / validation candidate occurrences | 56 / 59, at most five documents per query |
| Source-partition pools | Fit 256; tune 58; validation 58 documents |
| Unowned document exclusion | 4,811 of 5,183 documents excluded; 314 other-partition docs excluded from each evaluation pool |
| Historical exposure ledger | 9 runs, 356 deduplicated physical calls, one carried reference; 24 consumed claims / excluded components, zero unresolved claim IDs |
| This preparation | 23.63 seconds; zero new calls, training or Slurm submissions |

Each evaluation partition contains eight SUPPORTS/REFUTES and four official NEI
queries. All eight answerable queries in each have all annotated documents and
at least one complete first-three-eligible alternative visible: tune 9/9 gold
document occurrences, validation 8/8. Zero/partial gold coverage counts are both
zero, not omitted failures. These are **post-selection availability checks**, not
retrieval or model quality gains. Selection was frozen before statistics, without
gold backfill. The small TRAIN-only, source-owned pools and absence of missing-gold
answerable cases materially limit conclusions about open-corpus retrieval and
abstention. All 531 eligible TRAIN claims had earlier gold-preparation exposure;
official dev/test were not opened. No external generalization claim is permitted.

Private bundle manifest SHA:
`28de2c5d1d531aabdb757ccb45db0b91232a54ac7089ac7dc5ebf42e65ba3b2f`.
Compact bytes SHA:
`ad98fa9f1eec792f4bed02af886deaf73660871d8ddcd1900205bf708198318e`.
Ledger, split, source-family, fit and config hashes are in the compact receipt;
raw IDs, contexts, targets, per-row ledgers and corpus remain on Spartan.

## Isolated environment and next resource proposal (not a release)

The [environment receipt](verified-runs/scifact-grounding-cpu-environment-40d84a3.json)
records five offline, lock-hash-checked wheels installed only into the new
`grounding-cpu-40d84a3/site`: Pydantic 2.13.5 / core 2.46.5 and its three typing
dependencies. Existing tokenizer packages and tokenizer hashes were reused.
Neither shared Torch nor home-directory caches were modified. The Transformers
"None of PyTorch ..." message is expected because `USE_TORCH=0` intentionally
disables model loading for CPU preparation, not because the Linux module lacks
Torch or POSIX.

The optional training overlay remains a **separate readiness item**: PEFT,
Accelerate and grammar dependencies must be installed and hash-verified in the
dedicated execution environment before a release. The reused tokenizer-only site
has safetensors 0.8.0 whereas the candidate training pin is 0.5.3; do not silently
treat those environments as identical or replace shared packages.

Proposed one-allocation shape for review: one A100 80 GiB, four CPUs and 24 GiB
host RAM, batch one, gradient checkpointing, 96 records / 24 optimizer updates,
one final checkpoint. This is a conservative **capacity proposal**, not a measured
training footprint. The previous inference-only 10,305,912 KiB MaxRSS and
8,802,492,928-byte Torch peak cannot establish backward-pass memory or walltime.
A separately authorized allocation-based shape/timing pilot must establish those
bounds and account for every real training step; formal walltime should then use
measured time plus margin. No pilot, job, `sbatch --test-only`, generation or training
was run by this CPU package. No automatic duplicate training/replay is authorized.
