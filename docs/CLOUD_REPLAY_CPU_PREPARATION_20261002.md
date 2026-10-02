# Standalone replay preparation — CPU delivery, not a model result

Baseline: `349c8c9e42142af8fb88e530faa580d989ffb248`. The feedback-conditioned
regression was committed first as `35057c8`. The exact final source, archive,
entry and unauthorized-release hashes are emitted by `package_cloud_replay.py`;
they are not copied from the historical Spartan release.

## Result and limits

The new backend is explicitly `standalone-linux-v1`. Slurm remains the old entry
and default. No SSH, Slurm submission, cloud account login/purchase/storage
creation, external data upload, weight loading, inference or training occurred.
No resume or career shared-state file is changed by this package.

The synthetic feedback policy branches on actual `tool_feedback` and displayed
sentences, not a call ordinal: same initial claim/candidates/first query; relevant
return → stop → separately cited verdict; empty/timeout → different legal query
→ actual returned evidence → stop/verdict. Original claim, earlier evidence,
five combined model calls/five tools, failed-call costs and physical ledger audit
remain checked. This proves a controller contract, **not real-model autonomy or
quality gain**. Historical job 32030221's negative result is unchanged.

The experiment remains the **same consumed validation32 / corpus5240 / five
routes / 160 slots**. Retrieval, binary-label and cost denominators remain
24/23/32. No new labels, dev300, frozen test or training. New gate/verdict latency
and behavior are unmeasured. Costs must be reported; any quality bought by more
compute is a tradeoff, not an efficiency claim.

## Boundaries actually implemented

- Separate draft schema: `authorization=none_draft`,
  `model_execution_authorized=false`, runtime receipt unobserved. Drafts and
  cross-backend/stale releases fail before output creation or model access.
- Canonical dedicated root and unique run ID bind source archive, extracted
  source, interpreter, input transport, work, output and scorer-only gold path.
  Existing run directories and symlink escapes are rejected. Failed runs are
  preserved; no retry or overwrite.
- Source files are compared with the bound archive and exact Git marker; extra
  importable source files fail. The entry imports from that exact source tree.
- Input extraction uses pinned corpus/protocol/manifest/model hashes. Historical
  unused sidecars are not extracted; unknown names, links and traversal fail.
  No gold is inside the inference transport. An explicitly rebound future
  transport can have a new tar SHA while preserving all logical identities;
  the old tar SHA remains historical provenance in the frozen policy.
- Worker projection contains no gold path/labels/scoring policy. Its exact
  fields, hash and parent-release hash are bound to supervisor reservation.
  Runtime receipt must be fresh Linux/cloud evidence, not a Spartan receipt or
  synthetic instance. Actual runtime distribution bytes and import origins are
  checked as well as versions/Python/driver/GPU/machine identity.
- Existing bounded worker supervisor, generation provider, controller and scorer
  are reused. Successful child reaping, physical wire audit and complete cost
  agreement precede the scorer's first gold read. Failure costs remain persisted.
  Prelaunch is bounded to240s; operator6900s with worker6000s and scorer300s;
  no scheduler, hidden warmup, unbounded retry or fake `SLURM_JOB_ID`.

This is application-level separation on an owner-controlled dedicated instance,
**not an OS security sandbox against a malicious same-UID process**. External
transfer and actual model execution still require their own authorization.

## Local asset evidence

CPU full-byte inventory: `artifacts/cloud-assets-20261002-verified.json` (private
local receipt; safe summary below). No asset content is published.

| Asset | Local relative location | Verified status |
|---|---|---|
| Original transport | `artifacts/budget-agent-inputs-20260929.tar` | 16,122,255,360 bytes; full SHA matches `563738f0be1f7bf7b99b8de20bcdeab552f7e893166dec260b8f7f1e7951c3c1` |
| Inference bytes | 31 required members in that archive | Every member including weight shards hashed against pinned identities; not loaded |
| Independent scoring gold | `artifacts/agent-validation-20260929/validation-gold.json` | 7,419 bytes; `d2dd28422bffacf87ded2153b3bfac4ca9e1edc903e1d4e7a88d3a40f5d2fd8b` |
| Frozen selection | `docs/verified-runs/budget-agent-validation-selection-20260929.json` | 1,242 bytes; `988b6682034a70966c8bbad5ff3c42202933fe85791b97a5f39e4a1b68071bc6` |
| Cloud Python/GPU/driver/instance | None observed | Not ready for execution; fresh runtime receipt required |

Corpus SHA: `c14315aee9feecbbbbc3b0c7101d978b52b47cee9a613336304bcaa736460c71`.
Protocol SHA: `abfcb61e9e6a54641f025101a4011fa88e07e3697a36f0acd308fa8e86052674`.
There is no currently missing logical input in the inspected local assets.
The original tar is available; a re-download is unnecessary. Old Spartan module
archives/receipts are not a portable runtime and are not used by this backend.

If public weights ever need recovery, the following is a **non-executed recipe**:

```sh
python scripts/prepare_agent_assets.py --lock configs/budget_agent_assets_20260929.json --output-dir NEW_PRIVATE_MODEL_DIRECTORY
```

It pins generator `Qwen/Qwen3-4B@350135a4de9a3407be836fa238cccc1d61503a85`
and reranker `Qwen/Qwen3-Reranker-4B@22e683669bc0f0bd69640a1354a6d0aebcfeede5`.
Loss of the exact protocol/selection/gold cannot be repaired from predictions or
synthetic labels; inventory must remain `needs_assets` until originals are recovered.

## Reproduction and future operator commands

CPU only, in the current isolated source checkout:

```sh
python scripts/preflight_cloud_assets.py --archive artifacts/budget-agent-inputs-20260929.tar --gold artifacts/agent-validation-20260929/validation-gold.json --full-hash --output NEW_ASSET_RECEIPT.json
python -m pytest -q tests/test_cloud_replay.py tests/test_stop_acquire_feedback.py tests/test_stop_acquire.py tests/test_targeted_replay.py tests/test_scifact_source_package.py
python -m ruff check src scripts tests
python -m mypy --platform linux --follow-imports=silent scripts/cloud_replay_contract.py scripts/run_cloud_replay.py scripts/package_cloud_replay.py scripts/preflight_cloud_assets.py scripts/run_targeted_replay.py scripts/run_targeted_replay_operator.py scripts/package_scifact_source.py
python scripts/scan_tracked_secrets.py --repository .
python scripts/package_cloud_replay.py --source-git EXACT_COMMIT --output NEW_PACKAGE_DIRECTORY --root /workspace/climate-replay --run-id stop-acquire-UNIQUE_ID
```

The package contains exact `source.tar`, source receipt and **unauthorized** JSON.
Linux CI runs all tests including native path/projection/supervisor cases; local
Windows skips those POSIX-specific cases. Local clean-source affected reproduction
must use imports from the extracted archive, never the editable checkout.

The following is a **future proposal, not permission or a command run here**:
one dedicated A10040/80 (80preferred), >=64GiB effective host RAM, >=8 effective
vCPUs, >=100GB disk, <=7200compute seconds, no automatic retry. Provider example:
Runpod on-demand or equivalent owner-controlled Linux host. Rental, transfer of
private inputs and inference require separate approval; do not resume Spartan.

After that approval, provision source/assets privately at the release's exact
paths. Create `root/{packages,assets,scoring,runtime,work,runs,source/EXACT_COMMIT}`;
extract the verified source tar into `root/source/EXACT_COMMIT`. Keep gold only
in `root/scoring/validation-gold.json` and the input tar in `root/assets/input.tar`.
Proposed fresh setup (not installed here; do not reuse Trip or Spartan environments):

```sh
python3.11 -m venv /workspace/climate-replay/venv
/workspace/climate-replay/venv/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
/workspace/climate-replay/venv/bin/python -m pip install -r /workspace/climate-replay/source/EXACT_COMMIT/requirements/cloud-replay-20261002.txt
/workspace/climate-replay/venv/bin/python -m pip check
```

Using that interpreter and exact extracted entry:

```sh
python scripts/run_cloud_replay.py runtime-preflight --release release.unauthorized.json --release-sha DRAFT_SHA --instance-id REAL_OWNER_INSTANCE_ID
# STOP: coordinator reviews observed runtime receipt, assets and exact package.
# A separately authorized release binds that receipt SHA; this package cannot issue it.
python scripts/run_cloud_replay.py run --release release.authorized.json --release-sha APPROVED_SHA
```

Never put a fake instance ID or old receipt into the approval. No cloud Linux,
CUDA or paid-resource readiness is claimed by this CPU package. A failure remains
unscored with cost and exit evidence; it does not authorize another run.
