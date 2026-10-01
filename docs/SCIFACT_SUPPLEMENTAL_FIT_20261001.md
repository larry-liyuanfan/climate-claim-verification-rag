# Supplemental FIT action-data protocol (2026-10-01)

Status: prospective CPU preparation protocol; no training or heldout evaluation.
The original 48 and their shared-5,183-corpus outputs remain an unchanged control.
This package is not an independent test, learned-policy result or resume gain.

## Frozen selection, before reading new content

- Use only the prior hash-bound component assignment, family/partition metadata,
  aggregate consumption ledger and listed physical reservation receipts.
- Recompute exclusions at **component** level: original 48, tune/validation,
  model-consumed, uncertain, failed/unknown attempts and subsequent training,
  tune, bare-regression and read-conditional reservations. Remaining identity
  must equal 160 components (`c4d37c0d…ada85`) / 256 eligible TRAIN IDs
  (`e44bd95c…aede5`). The latter are claim IDs, not the old 256-document pool.
- Salt is exactly `scifact-supplemental-fit-v1-20261001`. Sort components by
  SHA-256 of canonical JSON `[salt,"component",component]`, then component ID;
  sort IDs within each component by `[salt,"claim",component,id]`, then numeric
  ID. Take the first ID in each of the first 96 components, at most one each.
- No label quota, text/rank/length inspection, read-opportunity selection,
  replacement, alternate salt or result-dependent additions. Eligibility is
  conditional on those receipts, not a global unexposed/decontamination claim.
  All 531 eligible TRAIN annotations were seen by earlier gold preparation.
- Exclusively persist private selection and component reservation files before
  opening the TRAIN member. They bind source/archive/wrapper, inputs, ordered
  candidates, selected IDs/components, salt/config, original control and corpus.
  A failed preparation does not release reservations or permit new selection.

## Selected-only loading and unchanged state construction

`load_reserved_claims` checks both frozen physical files before calling its
stream opener. Reuse the ID-first byte scanner; stream other TRAIN bytes for
member SHA/top-level ID checks, but deserialize/persist only selected gold.
No tune/validation/dev/test question or label is deserialized. Complete selected
official annotations remain on Spartan. Any malformed input fails closed with
reservations retained; it is not silently omitted or replaced.

Public candidate pool is the unchanged full 5,183 documents, BM25 Top-20.
Tokenizer, controller, CommonPacking, action schema, budgets, teacher, full
rationales, four-record cap and claim/alternative/state weights are unchanged.
No retrieval from gold; gold is used only by this explicitly scripted teacher.
There is no claim that actions are model-chosen or teacher targets unique.

- Official NEI has no evidence and the existing builder rejects it. Keep it
  selected, mark `not_applicable`, emit zero records and retain denominator.
- MIXED labels are legal **per-document** targets; preserve them where the
  unchanged schema permits. Synthetic mixed-label coverage verifies this.
- Retrieved but unusable positives, token/contract failures and other gaps remain
  gaps. Context-insufficient abstention is not an official NEI label.
- Record schema stays historical v2. New outer provenance, selection/reservation
  SHA and report annotation scope bind this supplemental cohort. The inherited
  `epoch_normalizer=48` is a legacy schema field, **not this cohort's size or an
  executable optimizer denominator**. No optimizer runs in this package.

## Reporting and exposure

Report original 48 and supplemental cohort separately; do not call a cross-cohort
difference a paired effect. Count selected claims, teacher-supported claims,
ready claims, unsupported/gap reasons, direct answers, context abstentions,
actual read executions, feedback/next states and new citable sentences. One
read repeated across alternatives is one executed claim, not several samples.
Retain the full selected denominator when reporting each action's effective
claim-normalized rational weight and missing mass; never renormalize away gaps.

Private frames retain observations/schema, feedback, call/tool decrements and
trajectory/state joins. These are not a separately persisted full controller
event log. Audit indexed documents, retrieved candidates, actual prepared
citable/preview text and supervised targets separately. Full public corpus may
expose documents owned by other known partitions or unowned documents; abandon
source-family-unseen claims. "Prepared" does not mean a model trained on it.

## Execution and validation contract

Entrypoint: `scripts/prepare_scifact_supplemental_fit.py`; wrapper:
`hpc/scifact_supplemental_fit_cpu.sbatch`. No cohort/seed/path tuning CLI.
Use Iris via existing `spartan-trip` authentication and the isolated Climate
root `portfolio_20260903/climate-public-retrieval-v2`; no other project writes.

One CPU, 4 GiB, 15-minute hard cap, no GPU/requeue/replacement. Original same-corpus
48 preparation took 115 s / 305,392 KiB Slurm MaxRSS: 96 claims extrapolate to
roughly 230 s before variable action/token costs; the cap is conservative, not
a promised runtime. Reuse existing environment; `USE_TORCH=0` is intentional.
After review, freeze exact Git archive and test **this new wrapper's** guard
including rejection of wrong wrapper/archive/revision identities. Legacy
packager guard is not evidence for this wrapper. Run `sbatch --test-only`, check
duplicate scoped job/output, then submit once. Do not compute on login node.

Local synthetic coverage includes stable ordering, component exclusions,
reservation-before-open, selected-only parsing after member hash, no replacement
for NEI/gaps, legal MIXED labels, unchanged teacher frames/tokens, new provenance
and exact rational action mass. Targeted Ruff/mypy and Bash syntax are required;
this does not assert a rerun of every historical test/module/dependency hash.

CPU completion requires raw artifact hashes, physical selection invariance,
Slurm accounting and compact provenance verification. Commit only compact
redacted aggregates and this report. Training/evaluation requires a subsequent
frozen actual-dataset/optimizer contract and coordinator review.
