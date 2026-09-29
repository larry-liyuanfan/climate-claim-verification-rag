# Full r1 preparation failure: CPU diagnosis and minimal repair

Status: **repair only; no replacement allocation authorized or submitted**.
The original release `climate-full-72eaa90-20260930-r1` is consumed and must not
be reused. Its logs, status, submission guard and original immutable archives
remain unchanged. [Compact diagnosis](verified-runs/budget-agent-full-failure-31542525.json).

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
