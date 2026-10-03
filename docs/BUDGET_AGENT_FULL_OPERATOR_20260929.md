# Serial full operator preparation — NOT a GPU release

**Subsequent explicit release,2026-09-30:** coordinator authorized
`climate-full-72eaa90-20260930-r1` and clarified that unrelated shared-account
GPU jobs do not consume this career work package's single-job slot. After
repeat guards and test-only, the original owner submitted exactly one job:
**31542525**, initially PENDING/Priority. The
[immutable submission receipt](verified-runs/budget-agent-full-submission-20260930.json)
supersedes the preparation-only status below, not the frozen operator or inputs.
The job subsequently FAILED1:0 after39s in runtime preparation; neither inference
phase started. See the [preserved failure and preparation-only repair](BUDGET_AGENT_FULL_FAILURE_20260930.md).
No full results yet; do not resubmit this consumed release or alter other jobs.

The coordinator subsequently extended the same job's partition candidates to
`gpu-a100,gpu-a100-short` in place. A read-only snapshot at01:41:14+10:00
recorded PENDING/Priority and a nonbinding04:10 start estimate on2026-09-30.
Submit/eligible time remains01:31:13, resources and requeue/restart0 are unchanged;
observed AccrueTime01:40:43 and AGE factor0 do not establish preservation of all
age credit. The submission receipt appends this revision without overwriting
the original allocation request. No project-owner scheduling action or retry.

Status: operator prepared,18 targeted synthetic tests passed locally and from a
clean Git archive; Ruff, secret/PII and diff checks passed. Remote bash syntax
and `sbatch --test-only` passed. **No GPU or CPU job was submitted in this package.**
The test-only response31539025 is not an allocated/queued workload and is not
an authorization. Do not submit automatically when a queue becomes empty.

## Immutable identities

| Component | Frozen source |
|---|---|
| Inference and prompt/schema | `72eaa90567e3603a3b940edffbb21b684bf1d1ba` |
| Offline scorer and32/24/8 selection | `208ff931badff270cbbc9593c8ab54f5c79aec9e` |
| New operator only | `0fa11b580b6ac7d7af1d1861da179acda1f7d9fc` |

Operator archive SHA:
`0965c471b2f13197ae0eaffa30ada17745734fc5cf77e4ff06f853830da73cef`.
Original working-file Python SHA (LF, not the executed archive-member bytes):
`f442ec5c638c14c1a4ede7ef55d1cc300f88536c07a91397e42cac1990e39796`.
Actual original archive member/executed Python SHA (314 CRLF line endings):
`09e438a6496499b0da47d8b2b08ccd404bbb6d1105722febe12af31857e4a654`.
The archived bytes normalize exactly to the working-file bytes; this is an
archive line-ending difference, not the runtime-prefix root cause. The repair
pins the operator Python file to LF for subsequent archives, without rewriting r1.
Wrapper SHA:
`c266835b8dc614f221a66269e6f89224f7d7630bc550bbaa3c6f3d774bb56bf9`.

The wrapper verifies its own bytes against the pinned operator archive and
checks the exported operator source revision. The operator independently checks
the inference/scorer archives,16,122,255,360-byte input bundle, runtime/wheels,
validation gold, selection manifest, model manifests and protocols. All prior
input/model/prompt/budget identities remain unchanged; this is not a new runner,
training run, test split, data-selection exercise or third pilot.

## Execution and disk reuse

```text
unique persistent0700 release directory (atomic reservation; no overwrite)
  → verify and unpack each archive once in a unique node scratch directory
  → read-only shared evidence/protocol/model files; install pinned offline overlay once
  → frozen validation inference (32×3) → frozen validation score → identity/96-slot gate
  → frozen vNext inference (8×3) → frozen vNext score → identity/24-slot gate
  → complete only when BOTH phase gates pass
```

The two phases are separate child processes executing the unchanged72eaa90
entrypoint; they share the same on-disk weights, not resident CUDA models.
Models are loaded/hashed again at each phase and that work counts in walltime.
This avoids expanding the16GB bundle twice or copying a second set of weights
into30GiB scratch. Archive extraction rejects traversal, links and duplicates,
checks available scratch before writing and retains1GiB headroom per extraction.
No model weights were unpacked to validate this preparation; disk sizing is
still a runtime guard, not a completed full-environment measurement.

validation receives its existing protocol,32 query tasks and separate frozen
gold only in the scoring subprocess;24 evidence-bearing tasks are the retrieval
denominator. vNext receives its existing8 authored tasks, without gold.
Gold is never in the inference input bundle or inference command. Filesystem
separation is a data-flow boundary, not an adversarial OS sandbox for same-UID code.
The scorer checks the exact unique matrices before reporting results; model
failures remain rows, not grounds for switching queries or retuning prompts.

Private durable layout (all under the Climate project, not downloaded):

```text
runs/<separately-authorized-release-id>/
  operator-status.json
  runtime-install.log / dependency-check.log
  validation/run.consumed.json / run.json / score.json
  validation/inference.log / scoring.log / private-responses/
  vnext/run.consumed.json / run.json / score.json
  vnext/inference.log / scoring.log / private-responses/
```

The operator records independent runtime/scorer/operator identities, archive
and input-member hashes, phase transitions, model-manifest hashes, output file
hashes/sizes, elapsed time and status. No raw/row file is placed in Git or exported
automatically. Later public reporting requires a reviewed compact-only audit.

## Failure, signal and evidence boundaries

- A reused release directory fails immediately, without modifying it or writing
  a replacement status file. Phase run/consumed/log/score paths also reject reuse.
- A failing/incomplete validation phase blocks vNext. A completed validation
  phase is retained if vNext fails. No auto-retry, requeue, resume or cleanup.
- USR1 at90seconds before the Slurm limit, TERM and INT stop only this operator's
  own child process group, then record interruption and hash available private
  files. No other project/job/process group is touched. SIGKILL cannot guarantee
  a final receipt; a prior started state must not be treated as complete.
- The frozen runner writes `run.json` only after the whole phase finishes.
  Consumption receipts, logs, captured private responses and completed phases
  persist, but **all finished rows of an interrupted current phase are NOT
  guaranteed saved**. Partial/missing matrices fail scoring; do not invent rows
  or automatically rerun the consumed phase.
- Original-response attachments retain the frozen32-file/1MiB per-phase cap
  and32KiB per-response cap. Excess responses are marked `skipped_total_limit`
  in diagnostics. Do not claim every decoded string was retained. Complete
  phase run records contain returned answers/events; logs and captured originals
  remain private on Spartan.
- Two hours is a hard allocation budget, not a completion or VRAM guarantee.
  The90-second early signal intentionally stops work before that limit. See
  [the measured-pilot resource basis](BUDGET_AGENT_CONFIRMATION_20260929.md).

## Staged assets and precise pending command

The immutable operator archive/wrapper are staged in the existing project's
`envs/` directory, and the existing7,419-byte validation gold was copied there
with its unchanged SHA and mode0600. Gold remains absent from the public Git
repository and the inference bundle. The prior runtime archive exists; no new
dependency install,16GB expansion, real inference or Slurm execution occurred.

On an authorized future release, resolve a NEW release ID matching
`climate-full-72eaa90-[a-z0-9-]{6,64}`. The command below must be run only by
the coordinator after explicit full release and resolution of the shared GPU
slot/account ownership; the environment guard deliberately has no default ID.

```bash
# DO NOT execute merely because preparation/test-only passed.
CLIMATE_FULL_ROOT=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
: "${CLIMATE_FULL_RELEASE_ID:?coordinator must explicitly release this new run}"
test ! -e "${CLIMATE_FULL_ROOT}/runs/${CLIMATE_FULL_RELEASE_ID}"
env -u SBATCH_GRES -u SBATCH_GPUS -u SBATCH_GPUS_PER_NODE \
    -u SBATCH_PARTITION -u SBATCH_MEM_PER_CPU \
    CLIMATE_FULL_RELEASE_ID="${CLIMATE_FULL_RELEASE_ID}" \
    CLIMATE_OPERATOR_TAR="${CLIMATE_FULL_ROOT}/envs/budget-agent-full-operator-0fa11b5.tar" \
    CLIMATE_OPERATOR_SHA256=0965c471b2f13197ae0eaffa30ada17745734fc5cf77e4ff06f853830da73cef \
    CLIMATE_OPERATOR_GIT=0fa11b580b6ac7d7af1d1861da179acda1f7d9fc \
    sbatch --parsable --chdir="${CLIMATE_FULL_ROOT}" \
      --output="${CLIMATE_FULL_ROOT}/runs/${CLIMATE_FULL_RELEASE_ID}-%j.log" \
      "${CLIMATE_FULL_ROOT}/envs/budget-agent-full-0fa11b5.sbatch"
```

Resources in the frozen wrapper:single full A100,8CPU,32GiB RAM,30GiB scratch,
2h, no-requeue. `sbatch --test-only` on2026-09-29 returned a hypothetical
2026-10-05T16:09:49 cluster-local start; that is neither a reservation nor an
actual submission. Recheck with `--test-only` at the future release, not by
canceling/resubmitting existing jobs. Other users' jobs remain untouched.

## Targeted checks and unresolved items

The18 synthetic tests cover one extraction reused across phase plans, readonly
input hashes, explicit gold exclusion from inference/vNext, separate output
paths, serial phase order, inference/scoring/matrix/signal failures, retained
completed first phase, traversal/link/duplicate rejection, insufficient scratch,
no overwrite, both direct `verify_phase` success fixtures, wrong phase/scorer/
run hash/gold or missing slots. Fixtures contain no actual validation answers.
Tests passed again from clean operator0fa11b5; no unchanged local full suite
was rerun. Normal PR CI may run its full checks.

Unresolved:coordinator full release and shared GPU-slot ownership/availability;
actual full matrix results; long-input/adaptive real-model coverage and VRAM;
runtime disk capacity with the actual frozen archives. There is no independent
test promotion, semantic-quality result, deployment or automatic continuation.
Stop after this preparation and wait for the coordinator's explicit release.
