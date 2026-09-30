# Semantic-policy pair: CPU implementation and preparation only

This package tests a specific engineering hypothesis: the existing gap wire asks
for self-reports but lacks explicit scientific evidence-relation instructions.
One versioned prompt adds distinctions between topical relevance, SUPPORTS and
REFUTES, including subject/population, relation direction, quantities and
conditions. Lack of mention is not refutation. A document without a complete
rationale must not be output merely because it was retrieved. The support
self-report may justify either direction. Read/rewrite remain optional, tied
to a concrete information gap; immediate answer/abstain remain legal. No minimum
tool count, extra critic, training, or claimed Self-RAG/CRAG reproduction exists.

## Frozen comparison, no result-driven prompt changes

`scifact_semantic_policy.py` defines explicit original-G and semantic-G versions.
Original-G delegates directly to the existing renderer; its default system
prompt bytes, schema and mapper remain unchanged. Inherited counting/generation
both call the same version-selected renderer. Model manifests, greedy/nonthinking
decoding, bounded grammar, four routes, actions and budgets are unchanged.

New common packing takes the maximum of both empty-history prompts, both prompt
versions applied to the active real history, and the active actual prompt. It
does not mutate history, pad billed usage, or guarantee identical later paths.
Both arms must have equal ordered initial citable/preview IDs and hashes. A new
pair must rerun **both** arms: historical G results cannot stand in as its baseline.
Outer and durable nested provenance must match. The shared finalizer preserves
raw cost before auxiliary processing and retains proposal → strict decision →
actual event → next feedback links, including failures and export errors.

## Exposure ledger and selection

Before any new packing, SHA-verify the original preparation/assignment, legacy
diagnostic manifest and full 531-row sampling audit, then frozen r2/F/G run
receipts. Source marker alone is insufficient: verify every extracted source file
against its hash-pinned clean Git archive before private data access.
Decode the exact verified input bytes, not unchecked rereads. Whitelist only
train corpus/claims/gold, assignment, legacy audit/selection/protocol/manifests and
four tokenizer files; no dev/test member, weight file or raw model wire is copied.

Any real model attempt consumes a claim, including failed generation, abstention,
unknown usage or export error. Missing trace/planned-only exposure is separately
uncertain and conservatively excluded. Unknown or ineligible IDs, missing
components and incompatible source hashes fail closed. Deduplicate across
routes/arms but retain every exposure receipt. Exclude the entire component
**before** building matching options, including its members in other strata.

Use the existing fixed salt and maximum component/stratum matching, at most three
per legacy stratum and twelve total. All 531 IDs were already seen in gold-aware
preparation; remaining train is not an independent external test. Persist the
selected IDs/components, legacy strata and source hashes before new gold probes.
Shortfalls do not relax exclusions, quotas, strata or permit backfill.

## Selected-only CPU preparation

`prepare_scifact_semantic_pair.py` parses only selected train gold and probes only
those IDs. Old V1/no-common-packing strata are sampling labels, never asserted as
current opportunities. Both prompt versions use the new controller and same
common packing. A dummy reranker preserves the real allowed-action schema but
raises if invoked: no model or reranker is loaded. Read witnesses require an
executed legal read and declare their exact uncertain/unknown/empty-gap fixture
history, not arbitrary future model histories. Changed opportunities are recorded
without reselection. Full IDs, gold, witnesses and component ledger remain private;
only reviewed counts/hashes/booleans leave Spartan.

The allocation cap is one CPU, 4 GiB, ten minutes, with frozen runtime/overlay;
no polling, automatic retry or GPU submission. The draft future pair contains
up to 12 × 4 × 2 = 96 slots, with unchanged per-query limits (five calls,
8192 input/512 output tokens per call, 120 seconds) and separate preflight costs.
The proposed per-arm ceiling is 48 × 120 + 1440 = 7200 seconds on 1 A100/8 CPU/
32 GiB, supported by the previous <18 GiB peak RSS; this is **not a release**.
A future model runner/operator/source freeze, preflight and coordinator release
are still required. A negative later result does not authorize repeatedly tuning
this batch of questions for positive numbers. No current resume is changed.

Synthetic tests prove contracts only, not that a model understands the policy.
CPU preparation results, if successfully generated, are recorded separately.

## Completed CPU preparation: 31698106

The sole allocation completed at source
`779e49883570371ce6223cb2b4a8619df5924d3b`, exit `0:0`, in **123 seconds**,
with **105.408 CPU seconds** and **1,929,304 KiB batch MaxRSS**. It requested
one CPU / 4 GiB / ten minutes; `Requeue=0`, `Restarts=0`. No GPU, model inference,
retry or monitor was used. See the [compact](verified-runs/scifact-semantic-preparation-31698106.json)
and [resource/package receipt](verified-runs/scifact-semantic-preparation-resources-31698106.json).

The SHA-pinned source archive contains 364 unchanged Git blobs plus one
41-byte export-substituted `SOURCE_REVISION` marker (365 regular files total).
Both its exact extracted tree and the thirteen-file private input allowlist
passed verification. Source/input/wrapper SHA matched locally and on Spartan;
the six private output hashes were separately rechecked. Only the 3,953-byte
content-free compact was retrieved and published, with SHA
`4d73e0196fb64167798b89746020b15aca192ebdb94cf34a36d6bd274d7adadd`.

| CPU preparation observation | Interpretation |
|---|---|
| 12 distinct model-consumed IDs, zero uncertain IDs | r2/F/G repeated exposures are deduplicated; failures still consume IDs |
| 12 excluded components, 23 excluded eligible IDs | Whole-component exclusion precedes matching, not merely removal of old question IDs |
| 12 selected queries; three in each of four legacy strata | Fixed-hash maximum matching filled the declared quotas; no backfill or relaxed exclusion |
| 12 selected gold rows parsed; 30 packing probes; zero model calls | Only the selected train questions were re-probed; the 531-query packing pass was not rerun |
| Equal initial contexts; all legacy/current stratum transitions unchanged | Comparison is mechanically admissible under the declared fixture, not proven effective under a real model history |
| No official dev, retired test or unlabelled test access | All 531 eligible claims already had legacy gold-aware preparation exposure; this is not independent test evidence |

The fixture only covers the declared uncertain/unknown/empty-gap read/abstain
history. Its read witnesses cannot establish that a real model will ask to read,
rewrite or correctly interpret evidence. The prior F/G negative result remains
unchanged. The private draft holds 48 slots per arm (96 total), requires a fresh
original-G baseline and the common-packing identity gate, and is **not GPU release**.
No model-quality score or resume bullet follows from this preparation.

Verification at the frozen source: 64 targeted tests passed, including 25 new
contract cases; the same 64 passed from a clean extracted Git archive in 2.67 s.
Targeted Ruff, strict mypy on six source files, shell syntax and tracked-file
secret/PII checks passed. This package did not rerun the unchanged full suite;
the earlier 588-pass Linux/Torch result remains explicitly tied to `b248fe7`.
