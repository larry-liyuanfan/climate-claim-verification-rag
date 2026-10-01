# Fixed NEI47: CPU preparation implementation, not a completed experiment

## Subsequent authorized CPU execution: 2026-10-01

The implementation-only scope described below was followed by one separately
authorized submission, **31843230**; test-only reference **31843090** was not an
actual job. Exact executed source remained `782892d57365be62095fe3e31a7ee6f2a2f40b94`.
The [redacted closeout](verified-runs/scifact-nei47-cpu-closeout-31843230.json)
records `COMPLETED / 0:0`, 47/47 ready, zero gap/failed/unknown, and a matching
atomic complete marker. All 100 private-file hashes and all 47 capture/assistant
mask bindings were verified remotely; no gold or full frame was downloaded.

| Measurement | Observed value and boundary |
|---|---|
| Actual submission/start/end, Sydney time | 16:22:04 / 16:22:29 / 16:25:03 AEST |
| Requested allocation | 1 CPU, 4 GiB, 15 min, sapphire, no GPU, no requeue |
| Slurm elapsed / TotalCPU / batch MaxRSS | 154 s / 141.899 s / 289,528 K |
| Preparation wall / process CPU / process MaxRSS | 142.525773 s / 140.560524 s / 298,224 KiB |
| Scripted / real model responses | 47 scripted, 0 real model calls; unknown scripted cost 0 |
| Captured prompt / target tokens | 185,613 / 705 total; scripted tokenization, not billable LLM usage |
| Training / roster / weights | No optimizer steps, no training authorization, no roster or weights |

Process `getrusage` and Slurm sampling measure different scopes; their peaks are
not interchangeable. Slurm `billing=5` is a scheduler resource weight, not a
currency cost. The scheduler's test-only estimate did not predict the actual
25-second wait and is not reported as an SLA.

The first offline-preflight command completed its substantive checks but had a
Windows-to-SSH heredoc tail `NameError` afterwards. No job was submitted then.
The record was preserved, input CR transport corrected, and a separate successful
confirmation preceded the scheduling test and exclusive submission reservation.
No environment was reinstalled and no preparation job was retried.

Closeout compact SHA: `bbee6b6e0bf7d92ab6199911ddcc22896c933cdffcd57f035d0adeb75be62799`.
Complete marker SHA: `2cf645d23c0dc0db2f557e37d2cbde4f2b3f1f3818759f1524bb940dc6fcc121`.
This closes **CPU preparation only**. Original48/supplemental49 were not rerun;
there is no new training, model comparison, independent evaluation or Agent gain.

## Original implementation-package boundary

This bounded package follows semantic kernel `3f640175`. It adds code and
synthetic verification, not a new training roster, weights, model call,
training/evaluation run, or quality claim. The coordinator must separately
review the exact source/release hashes before any CPU submission. Utility8
source `e22143c`, its model/cache, and original48/supplemental49 are unchanged.

## Input boundary and chronology

The only annotation input is `complete-selected-fit-gold.json` in frozen
`posthoc/scifact-supplemental-fit-09f3d9e72216`. The five parent file hashes are
locked in `scifact_nei_preparation.INPUT_SHA`, including the already exposed
claim reports. A read-only remote existence/hash check matched all five on
2026-10-01; **no actual gold contents were read in this implementation package**.

1. Check parent compact/source SHA, selection, reservations and report hashes.
2. Require identical original96 ordered IDs/components and report order.
3. In that order, retain exactly 47 `not_applicable` + `NOT_ENOUGH_INFO` reports
   with reason `official_nei_not_supported_by_frozen_teacher` and zero old records.
4. Durably freeze the subset and reservation before reopening complete96 gold.
5. Require all original96 full rows/order and exactly the frozen47 empty-evidence
   rows. Any difference fails without replacing/reselecting a claim.

This is **versioned reprocessing of known NEI labels**, not prospective selection
before first label exposure, independent test data, or previously unseen sources.
No dev300, sealed test, unused validation12 or other project inputs are opened.

## Program capture versus physical observations

`capture_initial` accepts only `{id, claim}` and the production `SciFactBM25`.
The entire indexed ID→source-hash mapping must match supplied corpus, not just
top20 hits. The real job uses the same frozen public5183 corpus and tokenizer.
It calls `run_bounded_slot(adaptive, format_repaired)` with `CaptureTeacher`
and unchanged `CommonPacking`. A callable no-rerank stub preserves the normal
offered schema but must never execute. The sole scripted response is abstention.

The observer retains the full initial five-field capture, completed retrieve
event, stable aliases/current order, query identity, exact prompt bytes/token
identity and schema. Strict initial checks include empty before-context, no
scripted intervention origin, first-five selected context, remaining calls=5,
tools=4 and feedback=null. A successful capture reports `real_model_calls=0`,
`model_calls=0`, `scripted_responses=1`; original controller `model_calls=1` is
retained separately as a **fixture-provider attempt**, not relabelled as an LLM.

`program_candidates` accepts only a complete original row with empty evidence,
bound to claim, corpus, capture, frozen selection and parent gold hashes. It reuses
the existing terminal semantic kernel/tokenization, retaining full target, EOS,
assistant-only mask and explicit representation gaps. It cannot produce an
`observed_model`, model proposal, `gXX` ID or physical journal. The original
physical observation binder keeps its strict synthetic physical-identity checks;
the old `capture()` default and original48/supplemental49 preparation are unchanged.

## Failure and cost accounting

Output is exclusive `posthoc/scifact-nei47-program-<exact-source-prefix>`.
Run reservation/started markers precede input reads; selection files are fsynced
before gold. Every selected claim stays in the denominator (47), including gaps,
failed and unknown claims. No dropping, replacement, renormalization or retry.
Per-claim artifacts stay private and are read back, compared with expected object
identity, and revalidated for capture/target/token identity. Checkpoints survive
failure. All-ready results publish `complete.json` by atomic no-clobber hardlink;
otherwise `failed.json` is retained. Reservation/started without complete (e.g.
SIGKILL/disk failure) means incomplete/unknown, never success or permission to retry.

Compact output exposes counts/hashes/costs, not claim text, gold, frames or targets.
Known scripted responses and claims with unknown scripted cost are separate.
It records wall elapsed, Linux process CPU seconds and process MaxRSS KiB;
Windows synthetic runs explicitly report unavailable/null POSIX resource fields.
These are **process measurements**, not allocated billing. A future completed
job must additionally be joined with `sacct` elapsed/CPU/MaxRSS evidence.

## Entry, source freeze and future release

The CLI has no dataset/cohort-size/replacement arguments. It verifies exact source
archive/tree, wrapper and release hashes, fixed input hashes and local tokenizer.
The wrapper requests 1 CPU, 4 GiB, 15 minutes, no GPU and `--no-requeue`, uses the
existing runtime, sets `USE_TORCH=0`, preserves module PYTHONPATH and is offline.
No weights are loaded, Torch must remain absent from `sys.modules`.

After committing a clean tree, local packaging (no submission):

```powershell
$env:PYTHONPATH = 'src;scripts'
.\.venv-validation\Scripts\python.exe scripts/package_scifact_nei47.py --repo . --commit <EXACT_SHA> --output artifacts/nei47-release-<SHA_PREFIX> --bash E:/SoftWare/Git/bin/bash.exe
```

The generic source freezer still validates its legacy wrapper; the thin packager
**separately extracts/tests the actual NEI47 wrapper** with correct, wrong-revision,
wrong-wrapper and wrong-archive cases. `nei47-release.json` pins actual source,
wrapper, fixed inputs, tokenizer, config and resource shape. No old guard result
is presented as NEI47 evidence. Exact release SHA is in the handoff, outside its
own payload (no self-referential hash).

After coordinator approval, stage only those frozen files in a new
`$ROOT/envs/nei47-source-<SHA_PREFIX>`; unpack the verified archive into `source/`.
Set `CLIMATE_SOURCE_STAGE`, `CLIMATE_SOURCE_TAR`, `CLIMATE_SOURCE_GIT`,
`CLIMATE_SOURCE_SHA`, `CLIMATE_CPU_WRAPPER_SHA`, `CLIMATE_NEI_RELEASE` and
`CLIMATE_NEI_RELEASE_SHA` from the handoff. Future preflight and single submit:

```bash
bash -n "$CLIMATE_SOURCE_STAGE/nei47.sbatch"
sbatch --test-only --export=ALL --output="$CLIMATE_SOURCE_STAGE/slurm-%j.out" "$CLIMATE_SOURCE_STAGE/nei47.sbatch"
# Only after successful preflight and authorized release:
sbatch --parsable --export=ALL --output="$CLIMATE_SOURCE_STAGE/slurm-%j.out" "$CLIMATE_SOURCE_STAGE/nei47.sbatch"
```

These are prospective commands, **not actions performed in this package**.
Raw annotations and large artifacts stay Spartan; only compact evidence returns.

## Local verification scope

Six synthetic groups cover actual initial retrieval/capture; NEI-only binding;
BM25/order/prompt tampering; physical/program and legacy capture compatibility;
frozen subset/gold chronology; exclusivity/failure/all-denominator/token masks/hashes.
The six existing terminal-supervision groups regress the shared semantic kernel.

```powershell
$env:PYTHONPATH = 'src;scripts'
.\.venv-validation\Scripts\python.exe -m pytest -q tests/test_scifact_nei_preparation.py tests/test_scifact_terminal_supervision.py
.\.venv\Scripts\python.exe -m ruff check src/climate_rag/scifact_terminal_supervision.py src/climate_rag/scifact_program_capture.py src/climate_rag/scifact_nei_preparation.py scripts/prepare_scifact_nei47.py scripts/package_scifact_nei47.py tests/test_scifact_nei_preparation.py
.\.venv\Scripts\python.exe -m mypy --platform linux --follow-imports=silent src/climate_rag/scifact_terminal_supervision.py src/climate_rag/scifact_program_capture.py src/climate_rag/scifact_nei_preparation.py scripts/prepare_scifact_nei47.py scripts/package_scifact_nei47.py
```

Linux is the type-check platform for the POSIX-only allocated CLI; this does not
fake runtime platform checks. Existing Windows Torch/WSL validation environments
are reused, not installed again. Local fixture results do not establish 47/47
real completion, actual tokenizer costs or model improvement.

Verified locally on 2026-10-01: **12 targeted tests passed (10.30 s)**,
including the six old terminal semantic groups; Ruff and strict Linux-targeted
mypy passed for the scoped files, as did `bash -n` and patch whitespace checks.
The actual source/archive/NEI47 guard results are recorded by the subsequent
exact-commit packager in its release receipt, not inferred from these tests.
