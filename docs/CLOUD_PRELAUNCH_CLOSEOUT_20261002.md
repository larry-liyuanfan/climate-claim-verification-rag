# Cloud prelaunch closeout: CPU delta, no model result

Accepted parent: `7aec073d027777aab66539505f1be39d245c4cbc` (Linux CI36996077694).
This closes only newly reviewed launch gaps. It does not rerun or replace the
accepted feedback fixtures, consumed validation32, corpus5240, five routes/160
slots, model revisions, prompt, labels or historical negative results. No
Spartan connection, cloud login/payment/rental/upload, inference/training,
career-state write, merge or deployment is part of this package.

## Changes and their limits

- Effective capacity resolves `/proc/self/cgroup` against mountinfo and takes
  the minimum memory/CPU limit across the current group and visible ancestors,
  physical host and affinity. CPU quota stays an exact fraction:7.99 is not8.
  Requirements remain64GiB/8CPU. **v1, hybrid, non-root mounts, missing files and
  namespaced/hidden ancestors are refused**, not treated as unlimited. Some
  container offers will therefore fail even if their advertised resources look
  sufficient; use a verified complete hierarchy, not a bypass or guessed quota.
  These are allocation ceilings, not guaranteed exclusive free RAM/CPU and not
  an OS attestation against a provider fabricating its filesystem view.
- All formal entries use absolute venv Python with `-IB`. Isolated mode prevents
  script-directory/user-PYTHONPATH imports before the bootstrap. `-B` alone does
  not prevent reading old bytecode. Before project imports the child rejects
  project symlinks, `.pyc/.pyo` and native-extension candidates; source bytes are
  then checked against the exact archive. Fresh extracted source is required;
  do not reuse an editable tree or delete someone else's caches to force a pass.
- A stdlib-only Linux parent starts the independent subreaper **before project
  imports and full release validation**. The child performs source/runtime/input
  verification, preparation, worker/scorer and all final ledger/fsync work.
  The7200s ceiling includes a20s closeout reserve, TERM grace, KILL, bounded
  nonblocking reaping and receipt publication. Separate-session descendants
  remain owned; no unrelated processes or Slurm defaults are changed. Parent
  does not reread/recalculate ledgers. Available raw costs/traces remain intact.
- The final receipt is written exclusively as `.pending`, file-fsynced, then
  atomically linked without overwrite. A pending file is never a result.
  **Acceptance requires observed outer CLI exit0 AND a completed final deadline
  receipt AND inner exit/cost/complete-matrix quality evidence.** Publication and
  process exit cannot be one filesystem transaction; a receipt or `compact.json`
  alone cannot promote a timed-out/interrupted run. TERM during receipt writing
  also forces outer failure. A stuck kernel-uninterruptible process cannot be
  guaranteed reaped: it is reported as failure, not disguised as success; owner
  must inspect/terminate the paid instance if needed. This is not a sandbox or
  a hard-real-time guarantee for arbitrary blocked kernel I/O.
- A new destination CPU preflight checks exact source, transferred transport
  and every required inference member, independent gold checksum (not labels),
  interpreter presence, reused paths and effective compute capacity. Disk space
  is grouped by actual device; multiple views use minimum free space. Existing
  uploaded transport/source/environment already consume used bytes and are not
  subtracted twice. Additional reserves are extraction+4GiB work/cache,8GiB
  output,2GiB each source/environment,1GiB receipt/safety. A second filesystem
  cannot borrow free bytes from the input disk. Receipt SHA and stable plan hash
  bind authorization. Launch rechecks gold/disk/source/runtime; preparation
  rehashes transport and extracted bytes before starting any model process.
  Local inventory is **not** proof of destination bytes or GPU readiness.
- `jsonschema==4.23.0` is included in observed dependency/import identity (schema
  validation). Pinned key dependencies plus observed runtime bytes are **not a
  complete transitive wheel-hash lock**, an offline dependency bundle, or a
  reproducible driver image. Runtime identity is independently observed on the
  actual instance; old Spartan receipts are invalid.

## Evidence scope and transfer

Original local transport and31 necessary inference members were fully hashed
in the accepted parent. The16,122,255,360-byte tar also contains authored and
historical sidecars that are ignored by extraction. This CPU delta does **not**
rehash/repack that16GB file. Synthetic tiny archives exercise the new checks.
For a later authorized private transfer, prefer an audited transport containing
only the31 necessary members. Preserve every member's frozen logical SHA;
record the new transport SHA separately from the original tar's historical SHA,
and generate a new draft bound to that transport. Do not merely rename/relabel
the old hash. Keep gold separate, never inside worker input/projection; keep all
these private assets out of GitHub. Private upload is not public publication.

## Current exact command recipe (not executed here)

The coordinator must replace `EXACT_COMMIT`, `DRAFT_SHA`, `APPROVED_SHA` and
`REAL_OWNER_INSTANCE_ID` with verified handoff values. No placeholder authorizes
a run. Root is `/workspace/climate-replay`; source is a fresh verified extraction
at `/workspace/climate-replay/source/EXACT_COMMIT`. Draft and later separately
approved JSON belong under `/workspace/climate-replay/packages/`.

Owner-authorized setup only; independent environment, no editable install:

```sh
cd /workspace/climate-replay
python3.11 -m venv /workspace/climate-replay/venv
/workspace/climate-replay/venv/bin/python -I -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
/workspace/climate-replay/venv/bin/python -I -m pip install -r /workspace/climate-replay/source/EXACT_COMMIT/requirements/cloud-replay-20261002.txt
/workspace/climate-replay/venv/bin/python -I -m pip check
```

Create the dedicated `packages`, `assets`, `scoring`, `runtime`, `work`, `runs`
parent directories, **not** the per-run work/output directories. Place the
verified source archive at `packages/source.tar`, exact input at
`assets/input.tar`, and independent gold at `scoring/validation-gold.json`.
No source/ancestor symlinks; the normal `venv/bin/python` executable symlink is
allowed and its actual binary is inventoried. Source must contain no project
bytecode/native artifacts.

```sh
/workspace/climate-replay/venv/bin/python -IB /workspace/climate-replay/source/EXACT_COMMIT/scripts/run_cloud_replay.py asset-preflight --release /workspace/climate-replay/packages/release.unauthorized.json --release-sha DRAFT_SHA
/workspace/climate-replay/venv/bin/python -IB /workspace/climate-replay/source/EXACT_COMMIT/scripts/run_cloud_replay.py runtime-preflight --release /workspace/climate-replay/packages/release.unauthorized.json --release-sha DRAFT_SHA --instance-id REAL_OWNER_INSTANCE_ID
```

The first command is CPU-only; the second observes the actual CUDA runtime/GPU
but loads no weights. Coordinator reviews both receipts, exact source/archive/
entry, input logical and transport identities, gold checksum, quotas and disk.
Only the coordinator can issue the **separately authorized** JSON binding both
`asset_receipt_sha256` and `runtime_receipt_sha256`; this package emits neither
approval nor an authorized release. Preserve draft, receipts and their hashes.

```sh
/workspace/climate-replay/venv/bin/python -IB /workspace/climate-replay/source/EXACT_COMMIT/scripts/run_cloud_replay.py run --release /workspace/climate-replay/packages/release.authorized.json --release-sha APPROVED_SHA
```

Capture the command exit code, final `runtime/RUN_ID-deadline.json`, inner
`operator-status.json`, physical cost/exit evidence, full matrix and compact
hashes together. Nonzero exit, missing/stale receipt, timeout, incomplete matrix
or unreaped child means **unscored failure**, not successful abstention; no
automatic retry. Keep failed outputs. Reused run paths or receipts are rejected.

The7200s compute bound is **not** a billing cap. Setup, upload, waiting,
preflights, cleanup and storage all count against the coordinator's separate
USD20 total. No Auto-Pay, paid subscription or automatic retry. This worker does
not handle credentials/payment, rent, upload or terminate any resource. CPU
delivery does not authorize paid idle time or claim a rented Pod is ready.

## Validation and handoff

Affected synthetic tests cover ancestor/fraction/namespace limits, isolated
bootstrap with stale bytecode/native/symlink candidates, wrong/missing assets,
gold checksum without parsing, reused paths, source/receipt/plan drift, per-device
disk reserves, actual shared-operator slow final ledger, separate-session child,
TERM-ignoring finalization, late success after timeout, slow receipt/fsync,
receipt-stage interruption and blocked project imports under the real CLI.
Windows skips Linux process/path tests; exact-source Linux CI must pass them.
Ruff and scoped strict mypy include the three new helpers. Final exact-source
CI, clean-source reproduction and immutable source/archive/entry/draft hashes
are recorded in the single package HANDOFF.json and PR19 closeout, not guessed
before execution. The earlier accepted CPU evidence and all model negatives
remain unchanged; no new quality numbers or resume bullets result from this fix.
