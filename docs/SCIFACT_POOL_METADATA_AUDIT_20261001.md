# SciFact candidate-pool metadata audit — 2026-10-01

Decision: no new data, training or protocol change. The FIT-state preparation
is valid for its frozen 256-document pool, but is not a controlled measurement
of model improvement over the older 5,183-document retrieval environment.

## Reproducible scope

The standalone [stdlib script](../scripts/audit_scifact_candidate_pool_metadata.py)
has SHA `37279d90f65fd9163235300c8788846315baca718a91ee30aeb20d49ff3683e4`.
The [byte-preserved compact result](verified-runs/scifact-pool-metadata-37279d90f65f.json)
has SHA `fc746da89cb374a4cc411507298b733561ed954c34949b8ac50fe33eed11e25a`.
It records every input hash, including downstream reservation files bound by
their terminal receipts. Source review used parent `73b9f2647bf2d6cfb77132f0e9009aa2bc8635cf`.

Run on Spartan with Python 3.9+:

```sh
python3 scripts/audit_scifact_candidate_pool_metadata.py
```

The hard-coded root is the existing Climate project workspace. The script
verifies its input allowlist and writes an exclusive, script-hash-named directory
under `posthoc/scifact-pool-metadata-37279d90f65f/`. It will not overwrite a prior
output. Only this small script was transferred; no full source archive or Slurm
job was created. The bounded invocation used a 30-second CPU limit and completed.

This audit read saved group/family/consumption metadata, previously exposed FIT
frames and targets, and saved old witness diagnoses. It did **not** run retrieval,
tokenization, inference, training, gold scoring, new claim selection or sampling.
Only two metadata members were extracted in memory from the hash-checked input
archive: preparation manifest and group assignment. No new claim/label texts,
unused validation, official dev/test texts or corpus text were deserialized.
Private frames, target annotations and reservation IDs stay on Spartan.

## One matched case: pool, ranking and evidence visibility

Exactly one of the three previously saved scripted-read cases overlaps the
original 48 FIT claims. Comparing only that case, without another retrieval:

| Saved state | Old full-corpus retrieval | New FIT-family retrieval |
|---|---|---|
| Candidate pool | 5,183 documents | 256 documents |
| Retrieved width | 20 | 20 |
| Same witness-document rank | 7 | 1 |
| Initial witness visibility | Preview, not citable | Sentences 0–18 citable |
| After the original scripted read | Sentences 0–18 citable | No read necessary |
| Stored complete-witness relation | Not eligible initially; eligible after read | Existing target fits initial context |

All six old candidates preceding the witness are unowned by the **eligible-TRAIN
family mapping**; none belongs to the FIT pool. Mechanically filtering the old
Top-20 by the FIT allowlist also places the witness first. Old and new Top-20
share three documents, with unchanged relative order and source-text hashes.
The old post-read and new initial maps of original sentence index to text hash
are identical (`cfa70bc1c3f689f0b7becf5100c0ef7a310a8872623f019ee7eed666762a5b92`).
The one already-exposed target rationale set fits both maps. This is a join of
saved metadata, not a new gold-quality evaluation.

The observed candidate removal is consistent with the rank/visibility change,
but does not identify its sole cause. `SciFactBM25` rebuilds document frequencies,
IDF and average document length on its pool (`scifact_retrieval.py`, `bm25.py`).
The unchanged production CommonPacking implementation then receives different
candidates and order. Pool membership, BM25 statistics and resulting packing
are coupled; no controlled IDF/packing ablation was executed. Do not attribute
this result to better model reasoning, or count scripted read as Agent success.

## Protected-family proof boundary

Saved ownership has FIT/tune/validation component counts **229/48/48** and document
counts **256/58/58**. Another **4,811** corpus documents have no owner in this
eligible-TRAIN-only mapping. They are not thereby unlabelled or safe background.

The original grouping implementation (`scifact_grounding.audit_groups`) used
train and dev claim groups and source families to quarantine overlap. Its saved
assignment retains claim-to-component mappings and eligible/quarantined/dev IDs,
but not the complete family-to-claim graph. Candidate preparation later rebuilt
family ownership from eligible TRAIN only. Dev-only and quarantined-only source
families can therefore appear unowned. Official unlabelled test was not
inventoried; full held-out-family coverage is not established by these artifacts.

The metadata does establish that eligible TRAIN components and dev components
are disjoint under the frozen lexical grouping, and that the selected 48/12/12
components follow their distinct saved partitions. It does not establish semantic
or pretraining decontamination. The **provable additional safe set outside the
existing 256-document FIT pool is empty under the available proof**, not evidence
that no safe documents exist. No background expansion is released.

## Conditional TRAIN eligibility — counts only, not a selected cohort

Starting from 229 FIT-owned eligible TRAIN components, subtracting the original
48 FIT components leaves 181; prior model-consumed components remove another
21 unique components, leaving **160 components / 256 eligible claim IDs**.
Three of the 24 consumed FIT components already overlap the original 48.
Reserved tune/validation are outside this FIT partition. Failed/unknown physical
attempts are already excluded by the component ledger; they are not restored
because a run failed. Listed later train/tune/regression/conditional reservations
add zero further removals after deduplication. Receipt and remaining-set hashes
are in the compact artifact.

These are eligibility counts conditional on the listed hash-bound ledger and
receipts, not a global attestation of no exposure. All **531** eligible TRAIN
claims were already gold-preparation-seen. No label quotas, usable rationale
sets, natural read opportunities or packing suitability were checked. These
256 claim IDs are distinct from the 256-document pool despite equal counts.
Neither IDs nor new examples were exported or selected.

Protocol choice remains with the coordinator/user. Retaining the 48 FIT claims
and declaring a shared-corpus/open-retrieval protocol would require separately
tracking documents indexed, documents present in SFT prompts and supervised
targets, and abandoning any source-family-unseen claim. Retaining the restricted
pool and adding TRAIN claims would instead require a separately frozen selection,
component-level reservations and consumption receipts before touching their
texts/labels. Neither option was executed or recommended solely from these counts.

## Validation

Five synthetic stdlib tests passed, covering hash rejection, path escape,
overlapping exclusions, alias-independent sentence/rank comparison and changed
sentence hashes. Targeted Ruff and strict mypy passed for script and tests.
The independent narrow review corrected digest serialization to the original
contract and made exposure coverage explicitly conditional. The unrelated
92-test suite, runtime manifests and source archive were not rerun/rebuilt.
No current resume or shared career file was changed; this is retained diagnostic
evidence, not a new model-quality or economic result.
