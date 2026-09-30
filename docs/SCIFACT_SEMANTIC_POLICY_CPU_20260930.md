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
