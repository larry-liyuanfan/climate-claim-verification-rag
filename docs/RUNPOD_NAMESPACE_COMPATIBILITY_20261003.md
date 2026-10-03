# Runpod namespace compatibility — bounded CPU delta

Accepted baseline: `46615fe974c4a8a8447a53d6a36d2d864a36d415`, Linux CI37011569988.
New coordinator-observed evidence (2026-10-03 Sydney), SHA
`f73d17225e009d799d452f1458cfb4ff7b9ad46bb87abc90840b558316ef6208`,
showed `0::/`, cgroup2 mount root `/`, finite memory249,999,998,976 bytes,
CPU1360000/100000=`68/5` (13.6), affinity128. Old complete-host check correctly
rejected the namespaced root; this was not proof of insufficient resources.
The coordinator stopped the Pod. This project writer made no cloud connection,
payment, restart, upload, model/weight load or Spartan access.

## Narrow change

- Default `complete-host-unified-v2` retains its complete-visible-hierarchy
  checks and still rejects this namespace. New `runpod-namespaced-visible-v1`
  is selected explicitly in draft/release/worker projection/receipts, never by
  automatic fallback. It requires canonical unified-v2 paths, finite legal root
  memory/CPU quotas and all visible ancestor limits. Exact fractional CPU and
  host/affinity/provider upper bounds must still meet64GiB/8CPU thresholds.
- Coordinator UI allocation metadata is projected to provider, Pod ID,
 16vCPU/250GB,120GB workspace/30GB container, image and original evidence SHA.
  GB is decimal, not GiB. Both projection SHA and original fact SHA bind the
  draft. Billing, SSH and dynamic kernel fields are omitted. These are reported
  allocation facts, not proof that hidden ancestors are unlimited. The runtime
  instance argument, coordinator receipt and actual RUNPOD_POD_ID environment
  must agree. Missing/empty provider environment fails at the CPU capacity gate;
  it is never filled from the CLI or historical receipt. This remains an
  owner-controlled identity binding, not an anti-spoofing OS attestation.
- New result explicitly states `hidden_ancestors_verified=false`,
  `exclusive_capacity_guaranteed=false` and
  `capacity_basis=visible_limits_and_provider_allocation`. Passing means the
  named contract accepts the observed upper limits; it guarantees neither64GiB
  currently free nor sustained8CPU. No stress audit or small work probe is used
  to manufacture such guarantees.
- Runtime v2 separates stable configuration/identity from observations.
  Source/Python/dependency bytes, GPU UUID/driver, Pod, allocation SHA, cgroup
  limits and host/affinity bounds still compare exactly on re-observation.
  Sampling time and `memory.current` are recorded, validated and excluded from
  identity equality. They are not a claim about available memory. Changed quota,
  GPU, allocation or identity still fails; mutable usage alone does not.
- Existing per-filesystem free-space reserves remain. The namespace branch also
  checks metadata-only usage across the entire visible `/workspace` volume
  (including files outside the project), plus additional reservations, against
  the provider-declared120,000,000,000-byte boundary. No16GB content rehash is
  involved in this usage scan. Symlinks are not followed; inodes are deduplicated;
  unreadable trees or unexplained nested mounts fail. `df` from shared `mfs` is
  not independent per-Pod quota verification. Server overhead and container
  usage remain unverified;30GB container space is not pooled with workspace.
  The check is an observed storage budget, not a service-side quota guarantee.
  The coordinator still owns setup/container-disk checks and paid lifecycle.

No claim/prompt/model/logical SHA/label/scoring/decoding/budget/7200s supervisor
change.32 consumed validation tasks, corpus5240,5routes/160slots and all negative
model results remain frozen. No new test access, resume metrics or career writes.

## Minimal runbook delta

Retain46615fe source package as historical evidence. The new package contains
`source.tar`, source receipt, `provider-allocation.json` and **unauthorized**
release, plus final HANDOFF with same-source CPU CI and hashes. It does not
create a runtime receipt or grant model execution. The provider projection is
generated from the coordinator's exact original JSON/SHA, not from guessed UI
values. No raw original account/billing/SSH JSON belongs in the public repository.

Coordinator alone resumes/rechecks the same Pod and privately replaces source/
draft with this newly frozen package. Copy `provider-allocation.json` to the
**exact `provider_allocation_receipt` path in the draft**. Use a fresh run ID;
retain previous failures. Reconfirm the UI allocation; if facts changed, obtain
new authentic evidence/projection and regenerate the bound draft, not an edited
SHA or fake receipt. Restore independent venv and exact source paths as in the
[absolute command recipe](CLOUD_PRELAUNCH_CLOSEOUT_20261002.md).
Check that the SSH-launched process receives the provider's actual RUNPOD_POD_ID
before transferring large assets. If absent, coordinator must resolve the real
instance/environment binding; do not set a guessed historical ID to force a pass.

```sh
/workspace/climate-replay/venv/bin/python -IB /workspace/climate-replay/source/EXACT_COMMIT/scripts/run_cloud_replay.py asset-preflight --release /workspace/climate-replay/packages/release.unauthorized.json --release-sha DRAFT_SHA
/workspace/climate-replay/venv/bin/python -IB /workspace/climate-replay/source/EXACT_COMMIT/scripts/run_cloud_replay.py runtime-preflight --release /workspace/climate-replay/packages/release.unauthorized.json --release-sha DRAFT_SHA --instance-id VERIFIED_POD_ID
```

Both must produce real destination receipts before the coordinator's separately
authorized exact-hash single160-slot release. Same absolute `run` command and
outer-exit0 + final-deadline-receipt + complete-inner-evidence acceptance rule
apply. No privileged mode, host namespace access or automatic retry. All setup,
upload, waiting, cleanup and storage remain within the coordinator's existing
USD20 entire-round budget, not merely7200 compute seconds. This CPU writer stops
at delivery; real restart/preflight/model execution is not claimed here.

## CPU evidence

Synthetic tests reproduce the observed68/5 configuration, old-path refusal,
missing/unlimited/below-threshold limits, GB/GiB distinction, current/ancestor
limits, allocation/original-fact hashes, Pod mismatch, dynamic memory samples
versus stable quota/GPU identity, and shared-df versus declared volume budget.
Only affected local checks are run; same-commit Linux CI verifies native cases
and integration. Final counts and exact hashes are in the package HANDOFF and
PR19, avoiding a circular source/CI hash claim in this file. No16GB rehash/repack.
