# Runpod private POSIX output — bounded CPU repair

## Observed failure, not a model result

The coordinator's private failure receipt SHA is
`da094f6a3fd33bc87c516655eec1ac5f35697c9a9848193f7c7186691841a901`.
Source `4c256156a9702a3d0f8ecb4e2fd915e2b9203850` failed before model
loading: the MFS/FUSE `/workspace` directories still reported **0777** after
`mkdir/chmod 0700`. The existing `validate_private_directory` correctly refused
the loader directory. The observed outer exit was 1; the worker was reaped;
**0/160 slots, 0 generator calls, 0 reranker requests**. This is neither a
classification zero nor a new model quality result. Old inputs and failure
artifacts are preserved; this CPU writer has not accessed the Pod or Spartan.

## Minimal separation

The default `single-root-v1` mapping remains available. The new explicit
`runpod-private-posix-v1` storage contract requires the existing named Runpod
capacity contract plus a disjoint `private_root`, never `/workspace`.

| Role | Frozen path root |
| --- | --- |
| Source, source archive, unchanged input archive, venv | `/workspace/climate-replay` |
| Extracted inference-only corpus/model inputs | persistent `work/<run_id>/input` |
| Entire run output, including loader, every slot, generation/rerank ledgers, observations/responses, logs, cost and compact | private `runs/<run_id>` |
| Gold and scoring parent | private `scoring/validation-gold.json` |
| Storage, asset, runtime and deadline receipts, including deadline `.pending` | private `runtime` |
| Worker projection, temporary files, HF/XDG/Torch/CUDA/Triton caches | private `scratch/<run_id>` |
| Minimal provider allocation projection (no raw/billing/SSH fields) | persistent `runtime/<run_id>-provider-allocation.json` |

`private_root`, `private_scratch`, `storage_contract`, `storage_receipt` and its
SHA are frozen through release and worker projection. Gold path/hash remain
absent from the worker projection; the provider still sees no scoring metadata.
Only the post-exit scorer parses gold. Directory separation is a trusted-code
contract, not a sandbox against the same UID, root or the provider administrator.

The new gate checks actual UID, **0700 directories / 0600 regular files**, no
symlink/traversal, trusted ancestors, a separate device from MFS, and an observed
container-root POSIX filesystem (overlay/ext4/xfs/btrfs). A tiny non-sensitive
probe verifies real mkdir/chmod, file write/fsync and hardlink publication.
Successful chmod alone is never accepted. Existing privacy validation is
unchanged. Both asset preflight and the worker recheck private storage before
weight loading; the worker checks the already-created output/allocation tree.
The 0077 umask protects future slots and ledgers. Explicit private caches also
reset Python's previously cached default temporary directory.

### Separate storage accounting

Existing per-device free-space checks remain. Persistent reservations are
charged only to the provider's workspace budget. The container separately
reserves 8 GiB output, 4 GiB scratch/cache, 2 GiB headroom and 1 GiB receipts.
Gold is already present when the full destination preflight runs. No 16 GB
archive/model copy is placed on the container disk by this change.

The private-tree usage plus reservations must fit the provider's **30 GB
decimal** container allocation and the observed filesystem free space. This
cannot borrow from the **120 GB decimal** workspace or its huge shared `df`.
It is a private-project budget, **not independently verified whole-container
service quota/usage**: base image and other container consumption must still be
checked by the coordinator. Neither provider quota nor free disk is invented.

## Coordinator runbook delta — no automatic retry

The new package is unauthorized. Use the exact run ID, source SHA, draft SHA,
provider projection and paths in its final `HANDOFF.json`. Do not edit hashes to
make a receipt fit. Existing 32 consumed validation tasks, 5 routes / 160 slots,
24/23/32 denominators, models, prompts, decoding, labels, fixed controls,
scoring, budgets and 7200-second total deadline are unchanged.

For the packaged example `private_root=/root/climate-private`, the executing
UID must own the directories. If the real container uses a different UID/home,
regenerate a correctly bound draft for that actual private location; do not
pretend ownership or use privileged/host-namespace access.

```sh
umask 077
install -d -m 700 /root/climate-private
install -d -m 700 /root/climate-private/runs /root/climate-private/scoring \
  /root/climate-private/runtime /root/climate-private/scratch
```

Persistent source/packages/assets/venv/runtime and `work` parent must also
exist as before. Do **not** precreate the new `output`, `work/<run_id>` or
`scratch/<run_id>`: exclusive run reservation remains fail-closed.
Keep the original large input archive unchanged at the draft's existing asset
path. Place the minimal provider projection at its exact bound persistent path.

Before transferring any private labels, use the actual pinned venv and new
exact source for the new small permission/storage preflight:

```sh
/workspace/climate-replay/venv/bin/python -IB /workspace/climate-replay/source/EXACT_COMMIT/scripts/run_cloud_replay.py storage-preflight --release /workspace/climate-replay/packages/release.unauthorized.json --release-sha DRAFT_SHA
```

This reads only source/metadata and writes non-sensitive probes plus a private
receipt. It neither unpacks the large input nor loads weights. If permissions,
mount identity or private capacity fail, stop here. Privately transfer the
already-approved gold to the exact `gold_path`, with file mode 0600; validate
the existing fixed hash without parsing labels. Then run the existing
`asset-preflight` and `runtime-preflight --instance-id VERIFIED_POD_ID` using
the same exact draft. Runtime/asset receipts from the previous source/layout
are invalid. The coordinator binds **all three** fresh receipt SHAs in the
single exact-instance authorized release; no draft authorizes execution.

Run the unchanged isolated entry with outer stdout/stderr redirected to a
new **private** log under `/root/climate-private/runtime`, not `/workspace`.
The entry reserves its own scratch/output. No repeated run on failure, no
warmup and no second model job is authorized by this package.

### Recover before stopping

Treat the container private disk as disposable at stop/recreation. After the
one run terminates and children are reaped, the coordinator must privately
retrieve and hash-verify the entire run output, exact authorized release,
worker projection and runtime/deadline receipts **before stopping the Pod**.
Use an authorized private local/cloud destination, never the public repository
or career workspace; never automatically copy raw ledgers back to MFS. A local
redacted compact can be published only after independent review. Preserve old
failure evidence. Recovery/setup/storage all remain inside the existing total
round budget, not just the 7200-second compute ceiling.

## Verification scope

CPU synthetic tests cover immutable mappings and receipt/projection bindings,
actual POSIX UID/mode/no-symlink failures, simulated post-chmod 0777, private
loader/slot/ledger output, early refusal before large inventory/runtime/model,
cache/temp routing (including inherited HF/PyTorch overrides) and separate disk budgets. Windows skips native POSIX
checks; exact-commit Linux CI supplies those checks. Counts and frozen package
hashes are in final HANDOFF/PR19. No new model run or improvement is claimed.
