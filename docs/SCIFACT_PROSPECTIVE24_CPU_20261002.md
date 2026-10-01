# Prospective conditional TRAIN24: whole-chain CPU input handoff

This package changes input selection and input plumbing only. It is **not a new
model result, independent test, GPU authorization or resume improvement**.
The decision policy remains `2ab219bb13564790b004797354dcc89554da2bf6`:
fixed-top1 / fixed-all / adaptive, caps 1/4/5, 72 slots, at most 240 physical calls.
Verifier, EvidenceVerdictRef, assembler, model, prompts and scoring mathematics
are unchanged. The runtime diff only reports the new shared preparation cost.

## Whole-chain review before submission

Review the actual consumer chain, not isolated scripts:

`metadata -> reservation -> gold-free claims -> BM25/CommonPacking -> frames
-> operator -> worker -> physical ledger -> post-exit scorer -> compact result`

The review identified and repaired old24 coupling in operator/worker/scorer/
preflight and historical-replay cost descriptions. All new consumers use one
input adapter. New input identity failures preserve completed generation costs,
return `no_quality`, mark unverifiable preparation cost unknown (not zero), and
never open gold or rerun inference. Old input defaults remain reproducible.

Before a cluster submission: targeted consumer-chain regressions, Ruff/type
checks, Bash syntax, clean exact-archive reproduction, actual environment import/
CLI checks, and `sbatch --test-only`. A long job is not a substitute for these
checks. Reuse accepted unchanged core checks; do not create serial review-only
jobs or retest consumed evaluation data for readiness. Static review cannot
promise absence of runtime failures; preserve failure receipts without blind
retries. No new GPU job belongs to this package.

## Data contract

- Remaining conditional pool: **64 components / 97 eligible IDs**.
- Component set SHA: `b421a1d70dd9e4f7c40b2b79731d392a4382e3d5c149a05f345a98270009a5eb`.
- ID set SHA: `56376abd5a8473a3738b37779221c6381c379864b5571e7057ee351db590068e`.
- Exact pinned metadata files bind the ID-to-component mapping as well as the
  marginal sets. Later natural/document/evidence-commit/utility8/NEI/mixed and
  old12/evidence-note reservation projections were reviewed: overlap zero.
- Salt `scifact-evidence-commit-prospective24-v1-20261002`, domain-separated
  SHA256 ordering, one claim per component, first 24 components. No labels,
  opportunity quotas, query text or outcomes select the roster; no replacement.
- Exclusive selection and component reservations precede selected query reads.
  A failure retains the same reservation, never resalts or replaces claims.
- All 531 eligible TRAIN rows had early gold-preparation exposure. Shared corpus
  also prevents a source-family-unseen claim. This is a conditional diagnostic,
  **not** globally unseen, independent validation/test or contamination-free.
- No supplemental supervision preparation, validation12, dev300 or frozen test.
  Selected IDs/query text, original corpus and private frames remain on Spartan.

## Actual no-generation preparation

Reuse all 5,183 original abstracts, unchanged SciFactBM25 scoring/order and
CommonPacking. Stop at the existing initial-frame observer before a physical
generation attempt exists. No fake generation prefix or scripted model answer.
The same frame serves all three arms. Aliases, sentence identities, candidate
order, selected ID order, tokenizer and file SHAs are checked by every consumer.

Record corpus/BM25 construction, 24 retrieval/packing timings and total shared
CPU preparation separately. This cost is paid once for a shared offline batch,
not multiplied by three, omitted, or claimed as end-to-end online latency.
Prompt probes reuse the existing fixed synthetic-feedback/first-two-ref cases;
their observed maximum is not an all-reachable-state bound.

Future scoring retains: child reaped -> physical cost -> complete 72 slots ->
physical response/ref audit -> selected TRAIN gold whitelist. This CPU package
performs **zero model calls and zero new real-gold deserializations**. Synthetic
tests exercise the successful scoring path and damaged-input failure path.

## Entrypoints and resource shape

- CPU preparation: `scripts/prepare_scifact_evidence_prospective.py` through
  `hpc/scifact_evidence_prospective_cpu.sbatch`: 2 CPU / 8 GiB / 20 min / no GPU.
- Shared adapter: `scripts/scifact_evidence_input.py`.
- Preflight: `scripts/preflight_scifact_evidence_commit.py --prepared ...
  --release ... --tokenizer ... --output ...` (old inventory mode retained).
- Future draft: `scripts/package_scifact_evidence_commit.py --input-compact ...`.
  It must reuse the **exact CPU preparation commit/archive**, not a later
  docs-only HEAD. Draft authorization remains non-executable.
- Future model proposal, not authorization: unchanged single A100 / 8 CPU /
  32 GiB / 30 GiB scratch / 40 min, supported by the prior 312 s run and bounded
  worker cap. No submission, priority change, retry or model tuning is implied.

Local affected-chain validation: **53 passed**; source/CLI type checking and
environment/archive checks are reported with the final preparation receipt.
No current resume or other project is modified.
