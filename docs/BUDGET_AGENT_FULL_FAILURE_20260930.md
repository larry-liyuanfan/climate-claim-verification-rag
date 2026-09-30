# Full r1 preparation failure: CPU diagnosis and minimal repair

Original repair-package status: **repair only; no replacement allocation authorized or submitted**.
The original release `climate-full-72eaa90-20260930-r1` is consumed and must not
be reused. Its logs, status, submission guard and original immutable archives
remain unchanged. [Compact diagnosis](verified-runs/budget-agent-full-failure-31542525.json).

**Subsequent separate release:** after acceptance, the coordinator authorized
`climate-full-72eaa90-20260930-r2`. The sole submitter used the resource-pinned
[one-shot helper](../hpc/submit_climate_full_20260930_r2.sh), cleared inherited
`SBATCH_*` overrides only in that helper, supplied an explicit project chdir,
passed test-only and submitted exactly one job31543304. See the
[actual submission receipt](verified-runs/budget-agent-full-r2-submission-20260930.json).
Initial state at02:22:46+10:00 was PENDING/Resources with StartTimeUnknown.
The preparation-only commands below are historical handoff, **not permission to
execute again**: the r2 submission lock is now consumed. Later, r2 completed;
the [full negative result](BUDGET_AGENT_FULL_CLOSEOUT_20260930.md) records120
executed slots and0 accepted answers. The r1 failure evidence below is retained.

## Observed failure and cause

Job31542525 ran on `gpu-a100-short` from02:02:42 to02:03:21+10:00 on2026-09-30:
FAILED1:0,39s wall time,34.931s TotalCPU,15,854,940KiB batch MaxRSS. Its operator
recorded34.432s, input_extractions1, no output receipts, and both validation and
authored-vNext `not_started`. This is not a failed model prediction and produces
no quality metric.

The runtime archive hash was verified, but runtime extraction was never recorded.
The old operator filtered tar members using the literal `lib/python3.10/site-packages/`
prefix. The real tar uses `./lib/...`: an actual local metadata scan found **0**
literal matches versus **39,292** PurePosixPath-normalized matches. Thus the
nonempty-member guard raised `ValueError(empty_archive)` before dependency
installation or inference. The outer handler discarded the code and only printed
`ValueError`, making the original log unhelpful.

Only the348,977,995-byte runtime archive was downloaded to ignored local artifacts
for diagnosis; SHA2423a755… matched the frozen manifest. It was not extracted.
The selected files total1,088,034,827bytes, with no links/special members, parent
traversals, absolute paths or normalized duplicate paths. The16GB input/model
bundle was neither rescanned nor extracted.

## Script hash discrepancy, separately explained

The original0fa11b5 archive member has314 CRLF line endings and hash09e438a6…,
exactly the runtime receipt. Replacing only CRLF with LF produces the previously
published working-file hashf442ec5c… and byte-for-byte equality. This was Git
archive line-ending conversion, not code contamination or the failure cause.
The new operator source is pinned `text eol=lf`; its next immutable archive must
match the committed source bytes. Historical hashes are retained and relabeled.

## Minimal patch and verification

- Normalize member names before the exact runtime allowlist selection; retain
  resolved-target containment, file/directory-only, duplicate and capacity checks.
- Explicitly reject parent-traversal components. Missing site-packages now yields
  `runtime_site_packages_absent`.
- Persist a fixed preparation/phase `stage` before each operation and an
  allowlisted `error_code`; never publish raw exception text, model strings,
  environment values or tar member names as diagnostics.
- 30 targeted synthetic CPU tests pass, including both prefix styles, malicious
  archive cases, exception-payload redaction and an actual `main()` failure-status
  fixture. The fixture does not execute a model, install dependencies or allocate
  Slurm resources. Independent read-only patch review found no substantive issue.

The inference72eaa90 and scorer208ff93 source archives, input/model/protocol hashes,
gold/selection, prompts/schema, budgets, questions and labels remain frozen.
No old unchanged full local suite is rerun; normal PR CI remains the full gate.

## Proposed r2 (not released)

Retain one A100,8CPUs,32GiB RAM,30GiB scratch,2h maximum, no requeue, and
90-second early signal. Candidate partitions `gpu-a100,gpu-a100-short` match the
coordinator's final r1 eligibility. The137-second nine-slot confirmation supports
the original bounded allocation, not a promise that120 slots finish within2h;
this extraction failure provides no evidence to increase the resources.

After coordinator acceptance, a new `climate-full-72eaa90-20260930-r2` release
would use a newly pinned operator archive/wrapper. It must have its own atomic
submission lock and output directory, check for duplicate Climate full jobs,
run `sbatch --test-only`, then submit **once**. R1 cannot be resumed/overwritten.
Exact new hashes and the proposed command are recorded in the repair handoff
after the code commit; until separate release, do not execute an actual `sbatch`.

## Pinned repair handoff

Operator source: `947645b3feed914e30d02dc5dbd05859c4a19667`.

| Prepared asset | SHA-256 |
|---|---|
| `envs/budget-agent-full-operator-947645b.tar` (1,525,760bytes) | `f9aa535d33cd8068c9a747aa8aba431a036b56e55cf0cb7a959daa36132e4d42` |
| Extracted Python script,18,986bytes/zero CRLF | `2ec2dd9e9664305c4b870def9df3a6d8ba0fb17b0d8984149e0ac79cac170b75` |
| `envs/budget-agent-full-947645b.sbatch` (unchanged wrapper) | `c266835b8dc614f221a66269e6f89224f7d7630bc550bbaa3c6f3d774bb56bf9` |

The script bytes in the Git archive equal the committed working source exactly.
All three hashes were verified after staging these small immutable assets on
Spartan. Remote `bash -n` passed. A clean extraction of the operator Git archive
passed the same30 targeted tests and Ruff. No new `sbatch --test-only` or actual
job was invoked in this repair package; schedulability must be rechecked at release.
The compact [r2 preparation receipt](verified-runs/budget-agent-full-r2-preparation-20260930.json)
records this distinction.

Proposed command sequence below is **not executed**, and requires the coordinator
to set the approval variable to the exact new release ID. It atomically reserves
a distinct submission lock after checking for an existing Climate full job, then
performs test-only before the single real submission. A failed test/submission
leaves its lock for explicit inspection; no automatic repeat is allowed.

```bash
set -euo pipefail
umask 077
CLIMATE_FULL_ROOT=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
CLIMATE_FULL_RELEASE_ID=climate-full-72eaa90-20260930-r2
test "${CLIMATE_R2_RELEASE_APPROVED:-}" = "${CLIMATE_FULL_RELEASE_ID}"
CLIMATE_QUEUED_NAMES=$(squeue -u "$(id -un)" -h -o '%j')
if printf '%s\n' "${CLIMATE_QUEUED_NAMES}" | grep -Fxq climate-agent-full-frozen; then
  exit 90
fi
test ! -e "${CLIMATE_FULL_ROOT}/runs/${CLIMATE_FULL_RELEASE_ID}"
mkdir "${CLIMATE_FULL_ROOT}/envs/${CLIMATE_FULL_RELEASE_ID}.submission-lock"
run_full_sbatch() {
  env CLIMATE_FULL_RELEASE_ID="${CLIMATE_FULL_RELEASE_ID}" \
    CLIMATE_OPERATOR_TAR="${CLIMATE_FULL_ROOT}/envs/budget-agent-full-operator-947645b.tar" \
    CLIMATE_OPERATOR_SHA256=f9aa535d33cd8068c9a747aa8aba431a036b56e55cf0cb7a959daa36132e4d42 \
    CLIMATE_OPERATOR_GIT=947645b3feed914e30d02dc5dbd05859c4a19667 \
    sbatch "$@" --partition=gpu-a100,gpu-a100-short --export=ALL \
      --output="${CLIMATE_FULL_ROOT}/runs/${CLIMATE_FULL_RELEASE_ID}-%j.log" \
      "${CLIMATE_FULL_ROOT}/envs/budget-agent-full-947645b.sbatch"
}
run_full_sbatch --test-only
CLIMATE_NEW_JOB=$(run_full_sbatch --parsable)
printf '%s\n' "${CLIMATE_NEW_JOB}" | tee \
  "${CLIMATE_FULL_ROOT}/envs/${CLIMATE_FULL_RELEASE_ID}.submission-lock/job-id"
```

This is an infrastructure rerun of the frozen comparison, not additional tuning,
new evaluation data or permission to retry a partial model run. If r2 fails, retain
its private state and return to diagnosis rather than reusing its directory.
