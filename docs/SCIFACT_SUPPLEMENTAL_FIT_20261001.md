# Supplemental FIT action-data protocol (2026-10-01)

Status: the single frozen CPU preparation completed; no training or heldout evaluation.
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

## Measured closeout — job 31832344

Execution source `09f3d9e72216dd6d61adff27c10179176a41656c`; archive
`639b242b6f6c9a7a7cff9af3b08814821b5ed3530fade761d918ae4719e7b931`;
actual supplemental wrapper
`58b85e9cb26ed156f132dca5dfba9360f4d755ddb3c8d30dc611d7ba651d1ff8`.
Test-only proposed ID 31832330 was **not** submitted. Its future start estimate
was not a clock discrepancy or scheduling guarantee. Actual job 31832344
completed exit `0:0`, elapsed **108 s**, total CPU **97.933 s**, Slurm batch
MaxRSS **302,224 KiB**. Process `ru_maxrss=310,172 KiB` is a different measurement.
Allocation was one CPU / 4 GiB / no GPU. There was one submission and no retry.

[Byte-preserved compact](verified-runs/scifact-supplemental-cpu-31832344.json)
SHA `4ef3d782ecde3efd36ca9dfef1482b9117f3f0ed695cca020bacdec8c047284e`
(24,587 bytes); [submission](verified-runs/scifact-supplemental-cpu-submission-31832344.json)
and [closeout](verified-runs/scifact-supplemental-cpu-closeout-31832344.json) receipts
record checks and resources. All seven private artifacts were rehashed on
Spartan; the last stdout JSON equals the compact. No raw annotations, IDs,
frames or targets were downloaded or published.

| Measure | Original 48 shared-corpus control, unchanged | Supplemental fixed 96 |
|---|---:|---:|
| Selected / ready claims | 48 / 48 | 96 / 49 |
| Official NEI unsupported (zero records, retained) | 0 | 47 |
| Initial complete / post-read complete / context abstain | 45 / 1 / 2 | 46 / 0 / 3 |
| Natural executed read claims / new citable sentences | 1 / 19 | 0 / 0 |
| Records: answer / read / context abstain | 90 / 1 / 2 | 80 / 0 / 3 |
| Effective claim mass: answer / read / abstain | 45.5 / 0.5 / 2 | 46 / 0 / 3 |
| Missing claim mass, no renormalization | 0 | 47 |

These are different cohorts, **not a paired quality comparison**. Supplemental
labels were 35 SUPPORTS, 14 REFUTES and 47 NOT_ENOUGH_INFO; they were observed
only after selection and were not quotas. All 49 supported claims have total
retained mass one; no additional supported-claim preparation gap occurred.
Forty-five supplemental claims' targets cover all annotated documents; a legal
visible witness need not cover every gold document. There are 83 decision
records, 49 unique prepared prompts, 335,736 sequence tokens and maximum joint
length 6,070. Unsupported NEI records are not mislabeled as context abstention.

The fixed selection/reservation files remained unchanged. Selection SHA is
`72d0012fd12bf76396fb88c5f0d9581ef8d154996608e5fd167ea8f23e0a1355`;
effective data-protocol config SHA is
`d9ef15cce530bfda222e7df34b743ce0d2f539d218e3da6fa7d9bdad83d4352d`.

### Exposure and interpretation

The 5,183-document index is unchanged. Supplemental retrieved/prepared-text
union is 886 documents: 88 known FIT, 7 tune, 7 validation, 784 unowned. Actual
citable and preview sets contain 238 and 679 documents respectively and overlap;
they must not be summed as disjoint sets. All 47 supervised answer documents are
known FIT. Public cross-partition document exposure was explicitly permitted;
protected questions/labels were not deserialized and source-family-unseen is
not claimed. There was no learned-model invocation, training or heldout test.

### Decision / smallest next step

**Random supplemental TRAIN selection did not improve action coverage.** Preserve
the fixed 96 and the original control; do not search new salts, replace NEI, or
continue sampling until read examples appear. Merging the 49 ready claims with
the original 48 would yield 97 ready claims / 176 records but still only one
read-bearing claim, with read mass `0.5/97 = 0.515%` of the training claim mean
(versus `0.5/48 = 1.042%` in the original control). This would dilute rather than
demonstrate richer read supervision; no such training was launched.

Coordinator review deferred the merged SFT. The next bounded proposal is to
measure actual model errors and controlled tool counterfactuals on already
authorized TRAIN: when does read/rerank improve grounded correctness enough to
justify its cost? It requires a separate frozen model/trajectory comparison,
not another gold-visibility sample search. NEI calibration is a separate
capability and must not be advertised as tool-use improvement. This is a
methodological recommendation, not an executed experiment.

Local verification: **16** supplemental/shared synthetic tests passed (2.28 s),
targeted Ruff, four-explicit-file mypy (`--follow-imports=silent --platform linux`)
and Bash syntax passed. Independent static reviews found no blocker. Exact Git
blob/mode/source-marker package verification passed; the **new** wrapper guard
accepted correct identity and separately rejected wrong archive/revision/wrapper
identities. This is not a claim that all historical code or a real-model trainer
was revalidated.
