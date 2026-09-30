# One generator-grounding candidate — prepared, trained, not quality-evaluated

Latest: training job 31757970 completed and passed artifact integrity checks.
The sections below preserve the earlier CPU-preparation/release boundaries;
the final section records the separately authorized execution. No evaluation
or Agent/generalization improvement has been demonstrated.

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

## Follow-up execution readiness (CPU only)

The separately bounded readiness package has now installed and imported the
Linux overlay on the existing Torch 2.1.2 module. See the
[installation receipt](verified-runs/scifact-grounding-linux-install-20261001.json),
[successful import receipt](verified-runs/scifact-grounding-linux-imports-20261001.json)
and [`hpc/grounding_training_deps.lock`](../hpc/grounding_training_deps.lock).
PEFT 0.15.2, Accelerate 1.6.0, Transformers 4.51.3, tokenizers 0.21.4,
safetensors 0.5.3, LMFE 0.11.3 and interegular 0.3.3 import successfully.
An isolated NumPy 1.24.4 layer resolves the inherited SciPy 1.8.1 warning;
the final Linux probe has no stderr and no dependency errors. Torch and all
shared packages are untouched. Qwen3ForCausalLM was imported, not instantiated.

The private final receipt bytes SHA is
`9878d3e4d45735828b4260d87656fd2827481663266a875725bfe7abab7d390a`.
Its complete 11,186-file overlay/inherited-site manifest SHA is
`8d231f4980e0bf94fe26273074588a0f6c0ea67646117871fd97443ad1142697`.
The actual directory is `envs/grounding-training-deps-20261001` beneath the
Climate project root. Its ordered `numeric-site` / `site` plus the two reused
dependency sites are checked file-for-file before allocated model loading.
The system Torch/SciPy modules are identified by module/version, not claimed
as fully hashed operating-system images. CPU token preparation remains unchanged.

The new training-only wrapper requests **one A100, four CPUs, 32 GiB host RAM,
30 GiB job scratch and a 30-minute Slurm cap**; the worker cap is 25 minutes.
`sinfo` exposes `gpu:A100:4`, 32 CPUs and 514,917 MiB host memory per node in
the selected partitions, but not GPU memory. No 80-GiB GPU constraint or verified
GPU size is claimed. These bounds are reviewed resource caps, not measured
training time. The first real optimizer step on the fixed first four shuffled
records doubles as the in-allocation pilot: time, GPU identity/capacity/peak,
host peak and finite losses are recorded, then the SAME optimizer/model state
continues the remaining 23 steps. There is no separate/repeated pilot, reset,
extra sample, best-checkpoint selection or evaluation call.

Gradients are clipped with `error_if_nonfinite=True`; all trainable adapter
parameters must be finite before final saving. Failed/OOM/timeout runs retain
attempt markers, worker logs and whatever costs/progress are durable; they do
not produce a success checkpoint/complete receipt or automatically retry. A
NaN tensor fixture and a 96-record synthetic one-parameter fixture verify refusal
and the exact once-only 24-step schedule. They are not real-model training.

`hpc/scifact_grounding_train.sbatch` binds source archive/revision/wrapper and
the release hash. `scripts/run_scifact_grounding_train_operator.py` rejects
drafts, verifies the runtime tree, reserves one fixed allocation directory,
extracts only the frozen generator in allocated scratch and launches `train`.
Before loading weights it imports the actual runtime after the Torch pytree
shim, compares Python executable/version, package versions and module source
paths with the frozen receipt, and persists the observation. A fixture rejects
changed import origins even when dependency files remain unchanged. Torch, CUDA,
extension and Triton caches point into allocated scratch, not home.
It never invokes tune, validation or an answer generator. The draft uses
`authorization: DRAFT_NOT_AUTHORIZED`; exact hashes require coordinator review
and a separate release. CPU `sbatch --test-only` is scheduling validation only,
not submission or proof that a job has started. No GPU job is submitted here.

Final execution source `631aae70000ca4940c8225e046669dc85b1731e4` passed
**154 related tests** in its clean Git-LF export (8.03 s), Ruff, strict mypy,
Bash syntax, tracked secret scan and the **new training wrapper's own** archive
guard. The source tar hash is `4c9386edbd965a2e88629932cdaad11e9c8b43e90af6f9383b6d7d90a3ca650a`;
the actual training wrapper hash is `d88db6806c1d725866df303a63dcdd4493de0a74b8c7fc2de148a79e1de6af9a`.
The generic packager's legacy default wrapper was not used for scheduling:
`WRAPPER='hpc/scifact_grounding_train.sbatch'` was explicitly selected for the
[final source receipt](verified-runs/scifact-grounding-training-source-631aae7.json).

The same source's actual Linux `runtime_check` passed before any model loading.
See [runtime/dry-run receipt](verified-runs/scifact-grounding-training-readiness-631aae7.json)
and [final validation](verified-runs/scifact-grounding-training-validation-20261001.json).
The first dry run warned about the inherited login home working directory; a
project-directory-only scheduling check with explicit `--chdir` then passed
without that warning. Both were `--test-only`; the displayed reservation numbers
are **not submitted jobs**. Scheduler estimated 2026-10-05 07:44:16 in its own
clock, not a guaranteed start or a training ETA. Final draft release hash is
`dff04c8a53c8f527bcc939f68963b9527aa467b03866c77e012ec5b8377bde88`;
its authorization remains `DRAFT_NOT_AUTHORIZED`. No real training or generation
was performed. Documentation-only follow-up commits do not replace this frozen
execution source or change the earlier prepared data.

## Separately authorized training submission

After exact-hash coordinator review, the original draft's sole
`DRAFT_NOT_AUTHORIZED` string was replaced by `coordinator_exact_hash_release`,
preserving all other bytes. The separately saved activated file is 1,482 bytes,
SHA `e00c4f7db57d3c135d7fe74f126d208541d89cda02034118978d812d3393ad81`.
The original draft remains intact. Job **31757970** was submitted exactly once
from the project staging directory with explicit `--chdir`; see its
[submission receipt](verified-runs/scifact-grounding-training-submission-31757970.json).
First `squeue/sacct` verification confirmed RUNNING on `spartan-gpgpu126`,
one A100 / four CPUs / 32 GiB RAM, with the original 30-minute cap. This actual
start supersedes the earlier nonbinding test-only estimate; neither status nor
a future loss/checkpoint proves improved grounding quality. No evaluation or
answer-generation calls are authorized by this submission. Prior CPU-only
receipts remain truthful records of their earlier preparation stage.

## Training closeout: integrity passed, quality unmeasured

[Physical closeout](verified-runs/scifact-grounding-training-closeout-31757970.json)
and [adapter shape audit](verified-runs/scifact-grounding-adapter-shapes-31757970.json)
confirm job 31757970 completed with exit `0:0`. All 24 step markers match the
fixed shuffled once-only 96-record schedule, including the first four-record
pilot; there is exactly one final checkpoint. Worker exit was reaped with code
zero, the allocation/runtime/source/release/data/config identities agree, and
all 96 observed losses and saved adapter tensors are finite.

| Measured item | Result |
|---|---|
| Slurm elapsed / allocated GPU-job seconds | 92 s / 92 s (not active GPU kernel time) |
| Training worker / fit loop elapsed | 46.24 s / 30.32 s |
| Slurm TotalCPU / batch MaxRSS | 71.958 CPU-seconds / 9,227,040 KiB |
| Actual GPU | NVIDIA A100 80GB PCIe; 85,095,874,560 reported bytes |
| Peak Torch allocated memory | 11,987,743,232 bytes |
| Adapter | 2,949,120 parameters; 144 F32 tensors; 11,815,504-byte safetensors |
| Architecture checks | 36 layers × q/v × A/B; rank 8; all nonempty 2D pairs |
| Generation / evaluation / external dev-test | 0 calls / not run / not read |

The adapter safetensors hash is
`dd1974a26549b3337824252a9c38773873acce72427f92e656835bf9d71935d4`;
config hash `8a6619efd6a3de8e545953d1e3b5a65d500eaca65110ba446feb51b159a37823`;
training complete receipt hash
`f5e6a865ec09cb67a520e36ba4646fb297a381383d4d0357208488ef6e9168a0`.
All 35 private output files are indexed by manifest hash
`8c68150f16f7b103705ef4d4a4f88a0dce995a5350c65c712dc6b9550e14bb04`.
Weights, per-record losses, targets, full logs and physical manifests stay on
Spartan. Only compact aggregates/hashes are published. Mean training loss is
not a retrieval/verdict score; this small task-adaptation run is not pretraining,
an independent test, online A/B evidence or a reason to change resume claims.

### Next bounded proposal — tune-only, not released

Reuse this exact checkpoint with the implemented evaluator: twelve frozen tune
inputs paired as base-disabled-adapter versus the restored final adapter,
24 calls total with the same generator, tokenizer, grammar and input packing.
Restore adapter tensor values exactly before inference; no new training,
checkpoint selection, data replacement or extra warmup calls. Preserve every
attempt and cost, including failures. Score only after process exit, reconcile
physical responses, and apply the existing preregistered count gate. The
58-document TRAIN-only pool and all-gold-visible answerable queries remain
explicit limitations. This is a CPU planning proposal only: **no tune/validation
calls or additional GPU submission are authorized or performed here**. Validation
would require both the gate and another separate exact-hash release.

## Tune-only execution package (CPU preparation, not a model result)

`hpc/scifact_grounding_tune.sbatch` and
`scripts/run_scifact_grounding_tune_operator.py` package the existing evaluator,
not another evaluation framework. The wrapper is explicitly selected when
packaging; the generic packager's legacy diagnostic wrapper is not valid for
this package. Draft authorization remains `DRAFT_NOT_AUTHORIZED`.

The fixed training receipt and adapter hashes are
`f5e6a865ec09cb67a520e36ba4646fb297a381383d4d0357208488ef6e9168a0` and
`dd1974a26549b3337824252a9c38773873acce72427f92e656835bf9d71935d4`.
The [supplementary binding](verified-runs/scifact-grounding-shape-source-binding-31757970.json)
links that adapter to the original remote shape-audit bytes without replacing
the original receipt or rerunning training. Prepared data/config/runtime remain
unchanged. The fixed unused output is `runs/scifact-grounding-tune-20261001-v1`.

The resource ceiling remains one A100, four CPUs, 32 GiB host RAM, 30 GiB local
scratch and 30 minutes allocation, with a 25-minute inference worker cap. These
are bounded evaluation caps, **not measured inference latency**. Model extraction
and loading are allocation-only; runtime imports/hash checks do not load weights.
Twelve frozen inputs are evaluated once with the same restored CausalLM in
adapter-disabled and adapter-enabled states: at most 24 real calls, no warmup,
new training, sample substitution, checkpoint selection or validation call.

The operator first waits for/reaps the inference parent and verifies its durable
child-exit receipt. Only then does it launch the existing separate CPU scorer,
which reconciles physical responses and costs before applying the count gate.
Failed/unattempted slots remain in the two twelve-slot planned denominators.
Both arms must have zero `stop_required` records: even an otherwise improving
24-attempt/exit-zero run cannot pass if its final adapted response is incomplete,
over budget or has a non-parse failure. The original parse-failure comparison
rule is unchanged; paired fixtures distinguish these two cases.

A hard kill with no termination receipt, or a reserved call without a complete
arm summary, produces `cost-audit-pending.json`: physical file hashes, durable
reservation count, full 24-slot denominator, unknown costs and null cost totals.
This is **not a completed cost or quality audit**, never a zero-cost assertion,
and does not run the scorer, open validation or authorize replay. Complete
partial summaries still use the existing lower-bound cost audit. Any remaining
unreconciled failure requires a separate manual closeout, not automatic retry.

This twelve-query, 58-document TRAIN-internal tune experiment has gold visible
during preparation and all gold visible for its answerable queries. It cannot
establish independent test generalization, full-corpus retrieval or autonomous
Agent benefit. No resume or shared career materials are changed by this package.

Exact source `d9924d2e9b8db5cce5949569a494c258434770aa` passed **340 related
tests** in a clean Git source export (31.52 s), changed/reused entrypoint Ruff,
strict mypy on the tune operator and grounding eval/SFT modules, the actual
tune wrapper's own archive guard, Bash syntax and tracked secret scan.
Three existing fixture/deprecation warnings are recorded, not suppressed.
Actual Linux imports and all frozen dependency-file hashes passed with zero
stderr; training receipt and physical adapter hashes still match.
See [exact readiness receipt](verified-runs/scifact-grounding-tune-readiness-d9924d2.json).

The project-directory `sbatch --test-only` returned zero. Its displayed
reservation 31762198 is **not a submitted job**; the scheduler-clock start
estimate is not a guaranteed ETA. The remote release remains a draft with SHA
`274dbe6f38098a7bae19badf213470dfe350c4f75be9c5b89008b13f5a6da390`.
This preparation used no model calls, official dev/test data, training replay,
or actual GPU submission. Later documentation-only commits do not replace the
frozen execution source or activate that draft.

## Separately authorized tune submission

After the coordinator's exact-source review, a separate activated release was
saved by replacing only `DRAFT_NOT_AUTHORIZED` with
`coordinator_exact_hash_release`; the original draft is unchanged. Activated
bytes (1,657) match the coordinator's expected SHA
`562bf8282f2eb6b7eced4edc31e6b4be2b615480614213863ec3666f14a7ca34`.
One real job, **31763176**, was submitted with explicit project `--chdir` and
durable exclusive reservation/response/receipt. See the
[submission and first observation](verified-runs/scifact-grounding-tune-submission-31763176.json).

Pre-submit checks found no same-name job and no occupied output/allocation/
execution directory. Available project filesystem space was 287,588,352 KiB.
The account permits the requested GPU QoS; blank displayed resource-limit fields
were not treated as unlimited quota. No QoS/Nice or other users' jobs changed.
The first and only bounded observation was **PENDING (Resources)**, start `N/A`;
the pending `0:0` accounting field is not a successful completion. This is not
the earlier test-only reservation 31762198. The coordinator owns subsequent
status follow-up; no new monitor or automatic retry was created.

The authorization covers only the same twelve tune inputs paired across base
and adapted states, at most 24 model calls within the frozen resource ceiling.
Training, warmup, validation, official dev/test reads and resume changes remain
unauthorized. Submission/pending state does not establish grounding improvement;
quality and cost must be audited only after both inference processes exit.

## Tune closeout: positive grounding signal, unresolved NEI failure

Job **31763176** completed `0:0` in 142 allocation seconds, with batch MaxRSS
9,318,380 KiB and Slurm TotalCPU 123.542 seconds. The CPU closeout verified frozen
source/release/data/checkpoint/runtime identities, 144 equal restored adapter
tensors, both twelve-call denominators and all 49 physical files per arm against
the original arm summaries. Existing frozen `audit_arm`/scorer results were
reused; no model, gold, training or scorer rerun occurred. Worker exit/reap,
parent exit/reap, before-scoring receipt, physical score mtime and final allocation
receipt are ordered consistently. [Compact closeout](verified-runs/scifact-grounding-tune-closeout-31763176.json)
has physical SHA `1a2c0b0f678958b4bbd3418dbe833f23f537d8d017673e51d6b4983102250ea5`.

| Fixed twelve-query tune result | Base | Adapted |
|---|---:|---:|
| Correct first3-rationalized documents / relevant documents | 1 / 9 | 4 / 9 |
| Predicted documents | 41 | 12 |
| Abstract-rationalized F1 | 0.0400 | 0.3810 |
| Correct label-only documents | 5 | 7 |
| Complete rationale at any output position | 3 | 4 |
| NEI claims with false evidence / NEI claims | 4 / 4 | 4 / 4 |
| Valid NEI abstentions | 0 | 0 |
| Input / output tokens, including every attempt | 39,400 / 1,844 | 39,400 / 347 |
| Sum of call elapsed times | 57.945 s | 15.220 s |
| Unknown / stop-required / planned-unsuccessful | 0 / 0 / 0 | 0 / 0 / 0 |

The preregistered **count** gate passed, but this is not evidence that all F1
change comes from reasoning: F1 is `2*1/(41+9)` versus `2*4/(12+9)`, so reducing
excess predicted documents matters. Any-position complete rationale increased
3→4 whereas first3 credit increased 1→4; placement/selection matters too.
Label-only 5→7 mixes document selection and relation classification. All four
NEI cases still hallucinate evidence under the benchmark annotation: abstention
has not improved. Unjudged extra documents are not independently human-verified
false statements. Base ran first, output lengths differ, and call sums exclude
some staging/load overhead; these are not online SLA or causal latency results.

Consumption entry: **12 distinct frozen tune inputs, 24 physical calls**, not
24 distinct claims. Shared arm input identity is
`4e05451cc152ef5adb04b2662539d0e9ffc3cb32572aa29caa6503a5a223d169`.
No additional warmup, validation, official dev/test, sample substitution or retry
was consumed. Physical score/gate SHA are respectively
`232255d657d35e546f3d64480724f02aa5914c528176b998c8f4465377119803` /
`60bcc9ec34a1d5f33f6f665d3cf57d5c81c80ab8cdf2785f8a1b443680ae3d7a`.

Scope remains gold-preparation-seen TRAIN-internal grounding on the restricted
58-document pool, with all answerable gold already visible. It is not a
restricted-1.2M retrieval gain, independent test, external generalization,
tool-use/Agent benefit or resume-ready headline. The separately frozen validation
pair requires another exact-hash release; gate success alone does not execute it.

## Validation-only CPU package

The new `run_scifact_grounding_validation_operator.py` is a thin entrypoint over
the existing evaluator, adapter restoration, runtime checks, parent/child reap
proof and post-exit scorer. Tune still hard-codes `partition=tune`; validation
hard-codes `partition=validation`. The direct evaluator also rejects a release
whose partition disagrees with its command. The old submitted source and assets
remain unchanged.

Before extraction or model work, validation binds the physical tune gate and
score hashes above, their internal score link, passed/data/training identities,
24-call handoff and the original count condition including zero stop-required
and unknown records. It uses the already frozen twelve validation inputs and
same checkpoint/tokenizer/config/runtime. There is no new label inspection,
selection, prompt, budget, model change or checkpoint choice.

Only the output and partition are new: `runs/scifact-grounding-validation-20261001-v1`.
The proposed cap remains 24 calls (12 base then 12 adapted), no warmup/training,
one A100 / four CPUs / 32 GiB RAM / 30 GiB scratch / 30-minute allocation and
25-minute worker. Complete partial summaries retain failures and costs; missing
termination/summary remains cost-audit-pending. Validation writes a final score,
**not a new tune gate**, and no path automatically authorizes another stage.
The fixture checks include cross-partition release, incorrect physical gate or
score, changed link/next-call count, final incomplete response, unknown partial
costs and the absence of a new gate after validation scoring.

This section is CPU implementation/preparation only. The new release stays
`DRAFT_NOT_AUTHORIZED`; actual validation submission or model calls require the
coordinator's next exact-source/hash release. Its eventual result can only test
whether this small frozen TRAIN-internal signal repeats, not establish Agent
tool-use benefit or independent external-test generalization.

## Posthoc structure and exposure correction (2026-10-01)

The original tune gate and score are retained without modification. The
[physical metadata-only audit](verified-runs/scifact-grounding-structure-20261001.json)
(`b81dc4cc50f0b390a2ea08d68bd49daccbb95f4520a2ecab4cb65c3c8d7bcd1e`)
finds that all 96 FIT records contain one document and target alias `c1`.
All 12 adapted tune outputs are single-document `c1` answers (6 SUPPORTS,
6 REFUTES); sentence indices vary, so this is not fixed first-sentence copying.
This suggests alias/top-1 convergence, not demonstrated multi-document selection
or autonomous tool use. The original correctness/F1 improvement remains valid
under its original restricted-pool tune definition, not a broader Agent claim.

Contrary to an earlier coordination-summary assumption, FIT selection permits
historically consumed TRAIN components. Physical FIT/old-selection intersection
is **one exact claim and one component**, within the three historic read
opportunities: one trained, two not directly trained. All twelve were previously
exposed regression queries; neither subgroup is held out. Existing tune and
validation isolation is unchanged. No gold/scorer/model was rerun for this audit.

Validation-only source `ce34e16` is preserved as DRAFT, with 69 related tests
passing (one existing tiny-PEFT fixture warning), Ruff and strict mypy passing.
No remote validation test-only, submission or inference was performed. Its
unconsumed twelve-query validation is reserved. Next authorized work is CPU-only
packaging of the old twelve-query, four-route regression with the same active
adapter in every route; actual execution requires a subsequent exact release.
