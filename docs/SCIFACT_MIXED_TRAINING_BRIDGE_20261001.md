# Frozen program-artifact bridge and claim-mean training entry

Status: **real CPU preparation complete; GPU training not released**.
[Job 31854794](verified-runs/scifact-mixed-cpu-closeout-31854794.json) prepared
144/144 claims and 223 rows with 36 planned updates, zero failed/unknown slots.
Slurm elapsed was 82 s, TotalCPU 69.512 s and batch MaxRSS 621,280 K. Three
physical output hashes and the atomic complete marker were independently
rechecked. No model was loaded, optimizer stepped or quality gain measured.
The earlier CPU NEI47 run is complete, not a
missing-Torch/POSIX failure. Existing Windows Torch 2.7.1+cpu and Linux runtime
are reused; no dependency or operating-system installation is needed.

## Utility8 conclusion, not Agent success

The [redacted closeout](verified-runs/scifact-utility8-closeout-31834687.json)
records the coordinator/independent audit of job 31834687 (source `e22143cc`).
This package confirmed only the two small remote artifact hashes; it did not
reread gold or individual predictions. All 24 terminal slots were valid, but
strict correctness for A/B/C was **0/1/0 of eight**. B's single strict success
was official NEI abstention. Complete-rationale documents were zero throughout.

Eight scripted B reads removed 276 citable sentences and added none. Eight
scripted C reranks changed context without complete-rationale gain. They are
not autonomous model tool selections. The unique physical cost was 24 generator
calls, 81,509 input / 3,007 output tokens and 93.618682 seconds; eight reranks
used 160 pairs / 70,177 actual tokens / 77.249925 seconds. Shared A0 calls are
not charged twice as 40 physical calls. This is a negative diagnostic on exposed
TRAIN cases, not an independent test and not a pure-tool-benefit candidate.

## Narrow source bridge

`scifact_mixed_inputs.py` has only two source channels:

| Channel | Preserved evidence | Derived supervision origin |
|---|---|---|
| `legacy_program_artifact` old48 / supp49 | Exact original file and row hash, ordered candidates, frame, packing, hierarchy, Fraction, provenance, historical `/48` | `visible_OR_rationale`, `program_read_witness`, or `context_insufficient` |
| `program_capture` NEI47 | Existing initial-capture validator, inventory/file hash, sealed annotation provenance and exact tokens | `official_annotation_NEI` |

No old physical tool events are invented. Original five context-limited
abstentions are **not** relabelled official NEI. The bridge never calls the
NEI-only semantic matcher for them. It does not read annotations, run a teacher,
retrieve again, reselect alternatives, truncate tokens or renormalize weights.
The original 48/49 rows remain nested unchanged. Only the new optimizer envelope
declares actual N. No unused utility8 physical-trajectory loader was added.

An external roster is mandatory. Existing ordered ID hashes, old48 integer-key
component hash, original96 ordered component hash, original row indices and
47/49 partition are checked. Cohorts and excluded components/claims cannot
overlap. A valid count alone never makes arbitrary JSON a trusted data source.

Expected released package (not yet constructed): **144 claims, 223 decisions**:

| Quantity | Answer | Read | Abstain | Total |
|---|---:|---:|---:|---:|
| Decision rows | 170 | 1 | 52 | 223 |
| Effective claim mass | 91.5 | 0.5 | 52 | 144 |

Each claim retains mass 1 and at most four original decisions. The future
preparation release must actually call `legacy_envelopes` and `nei_envelope`
against pinned physical source bytes, then `build_inputs` / `make_plan`, and
freeze `prepared.json`, `roster.json`, `plan.json` plus a completion receipt.
Directly sealing JSON that merely *declares* old source hashes is not a valid
preparation. Trainer-side validation does not reopen those source files.
The subsequent bounded package now implements that producer in
`scripts/prepare_scifact_mixed_inputs.py`, with the actual
`hpc/scifact_mixed_prepare_cpu.sbatch` and exact-source packager
`scripts/package_scifact_mixed_preparation.py`. It was subsequently executed
once on the fixed exposed training sources as job 31854794; source `cb1e7646`
and the content-free receipt hashes are retained in the closeout above.

The producer reads only fixed metadata, old48/supp49 records, the accepted NEI47
inventory/47 artifacts and the existing public corpus/tokenizer. Original48 IDs
come from the frozen split, components from the independent claim reports;
supp49/NEI47 reuse the fixed original96 selection, not the rows that happen to
survive. Exclusions reuse split/component metadata, without protected question
or annotation reads. No original gold, teacher call, retrieval or model call is
performed. On failure, all 144 denominator slots remain ready/failed/unknown;
there is no replacement, dropping, resume or success marker for a partial run.

Writes preserve nested key order. Readback compares physical token IDs/masks,
packing and full source envelopes against the in-memory **factory outputs**,
not merely new self-hashes. Only complete success atomically publishes the
exact three-file receipt consumed by the trainer. Compact output contains
counts, hashes and resource usage; raw examples stay private on Spartan.
Failures additionally retain the stage, original exception type and a bounded
message in owner-only `private/failure-diagnostic.json`. This diagnostic is
not part of the public compact receipt or exported artifacts; it lets a
source-contract failure be investigated without repeating real preparation.

The thin wrapper requests **1 CPU / 4 GiB / 15 minutes / sapphire / no requeue /
zero GPUs**, reuses the existing offline tokenizer environment with
`USE_TORCH=0`, and binds actual source/archive/wrapper/release hashes. The
packager validates this wrapper's real shell guard and rejects wrong revision,
wrapper and archive; a generic old-wrapper result is not used as its proof.
Four new synthetic preparation tests passed in **2.62 s** after the private
failure-diagnostic addition; Ruff and strict
Linux-platform mypy passed on the three new source files. No old mathematical
suite was rerun for this producer addition. The coordinator's subsequent
single exact-source CPU release completed successfully; GPU training remains
separate and is not authorized by that CPU completion.

## Separate real trainer, no execution release

`scripts/run_scifact_mixed_training.py --release FILE --release-sha SHA` is the
new train-only entry. It has **no** prepare, evaluation, resume or retry option.
It binds the new purpose/config, exact executable source manifest/revision,
runtime receipt/files/interpreter, frozen input files/receipt, fresh model
manifest and unused output path. An allocated Linux GPU and separate reviewed
release are required. A worker consumes an exclusive pre-model-load marker and
must match its bounded parent's reservation. Interruptions/timeouts kill and
reap that child; the release cannot silently retry a startup failure.

Frozen training settings: fresh Qwen3-4B base, fresh q/v LoRA r8/alpha16/dropout0,
seed 20261001, LR 1e-4, explicit existing AdamW settings, base BF16,
gradient-checkpointing `use_reentrant=False`, cache off, one epoch. Actual LoRA
parameter names/counts/dtypes are recorded (PEFT parameters need not be BF16).

The shared mathematical kernel consumes **four complete claims per update**,
weights causal assistant-token means by exact retained Fractions, then divides
by the number of complete claims in that group. 144 claims produce **36 updates**.
The generic tail group uses its actual size; the epoch metric is
`sum(group_mean * group_claim_count) / actual_N`, not `/223`, `/48` or constant
`/4`. Each group clips/steps once. The first update's CUDA peak/host resource
pilot is update **1/36**, not an extra pilot or optimizer reinitialization.
Vocabulary-sized temporary logits are released after each backward; accumulated
gradients remain intact. No per-row `empty_cache` is used.

Only after all 36 updates / 144 claims / 223 rows are seen exactly once does the
runner save a final-only safetensors adapter, check files/tensor finiteness and
publish an atomic complete marker. Failure cannot publish success or auto-resume.
The old synthetic driver CPU <=1M guard and old trainer 192-row ceiling are
unchanged. Training loss is not final-model quality evidence.

## Five synthetic CPU groups and continuation boundaries

`tests/test_scifact_mixed_training.py` covers:

1. Original legacy row/hash/packing/order preservation and fake-event rejection.
2. Official NEI versus context-limited origins, using the existing capture validator.
3. Synthetic 144/223 shape, exact rational mass, fixed shuffle and 36-step plan.
4. N=5 tail-group actual mean and two AdamW updates versus an independent reference.
5. Missing/duplicate/contaminated data before forward; checkpoint failure without
   success; exclusive worker claim; timeout/interruption kill/reap receipts.

The existing six pre-clip-gradient/AdamW equivalence cases additionally verify
the temporary-tensor lifetime change. No real corpus or model weights are used.
Final new-group check: **5 passed in 2.68 s** in the existing Windows Torch
validation environment. The six existing equivalence cases passed in the
earlier targeted run; no unchanged full suite was rerun. Ruff passed on the
five affected Python files; strict Linux-platform mypy passed on four source
files. Independent read-only review closed the one-time worker reservation,
signal-during-child-creation and kill/reap findings. The fifth test includes a
real short CPU child timeout and synthetic interrupt/creation-race checks.
Initial fixture failures came from sorted JSON changing prompt/key order;
fixtures now preserve physical serialization and negative cases assert the
intended validation error. Production prompt/hash checks were not weakened.

Preparation, actual training and evaluation remain separate releases. If a
candidate is later trained, first freeze the exact decoder/budget/version and
metrics for **one base-versus-candidate exposed tune12 comparison**. Old12 and
utility8 remain exposed regression, not independent generalization; reserved
validation12, dev300 and sealed test are not accessed by this package.

The eventual required Agent check uses the **same adapter** on the original
frozen12 four routes: fixed retrieval, fixed rerank, deterministic extra and
adaptive. Retain all three original legal read opportunities. Assess proposal →
execution → feedback → correct terminal update across queries, and quality/cost
against fixed routes. Utility8 B adding no sentences does not negate those
opportunities; adding sentences is not the only possible rerank value. If fixed
post-read answers work but adaptive never reads, investigate conditional action
supervision later; if the evidence is in the prompt yet the answer stays wrong,
the failure is feedback use. Scripted conditional continuation, if separately
needed for attribution, never becomes autonomous evidence. No repeated epochs
or prompt tuning to chase a positive result and no 48-slot execution are released.
