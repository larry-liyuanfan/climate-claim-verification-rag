# Cross-dataset identity audit — frozen CPU protocol

This is an identity check, not an evaluation, a new split, or permission to run
SciFact dev. Keep the 300 SciFact dev rows unchanged. No training, predictions,
retrieval scores or model outputs determine thresholds or retained rows.

## Inputs and provenance tiers

- SciFact: original prepared 300 dev claims and 5,183 paper abstracts; verify the
  preparation manifest and every input hash. Use `cited_doc_ids` only for claim
  association. The gold-containing dev file is parsed, but its `evidence` keys,
  labels and rationale indices are not used. A mutation regression protects this
  distinction. Cited papers do not identify the original claim-producing paper.
- Public Climate: already-consumed 1,075 train and 230 validation claims, with
  5,240 retrieval passages. Open only the two allowlisted train/validation JSON
  members in the existing SHA-locked preparation archive, never the merged claim
  file or old test member. Association uses **annotation-associated candidate
  IDs** and their article titles, not independently verified source provenance.
- Restricted Climate: already-consumed 154 fixed-dev claim texts. Claim-only:
  gold evidence keys are ignored; document text and independent source mapping
  are unavailable to this audit and remain unknown.
- Authored: 11 already-consumed development prompts, without source mapping.

Claim/document IDs have dataset-qualified namespaces. Equal bare numeric IDs
do not imply shared sources. Raw claim text and affected IDs stay in ignored
local audit output; public reporting contains counts and hashes only.

## Frozen matching and graph

Use NFKC, casefold and collapsed whitespace for exact text; Unicode word token
sets for Jaccard. Thresholds are fixed before inspection: **0.8 for claims and
0.9 for document text**. Document text is article-title plus passage or paper-title
plus complete abstract; differing granularities limit detection. No embeddings,
threshold tuning or semantic-equivalence claim. Negation can cause false-positive
lexical matches, while paraphrases can be missed.

Build connected components from claim matches, namespace-qualified shared-source
associations and document-variant families. Report both direct cross-track pairs
and transitive connections to consumed material. Record source-status counts and
missing mappings. A zero count cannot certify independence or exclude foundation
model pretraining exposure. Public test IDs are checked only as split metadata;
old test claim text/labels are not opened or evaluated.

## Reproduction

`scripts/audit_cross_dataset_identity.py` requires the original SHA-locked inputs
through `--public-archive`, `--split-manifest`, `--public-evidence`,
`--restricted-dev`, `--authored-protocol`, `--scifact-prepared` and a new `--output`
directory. It fails on changed inputs, denominators, missing documents or an
existing output directory. Run on CPU, not an HPC login node. No raw restricted
data is committed. `tests/test_cross_identity.py` covers namespace collision,
graph transitivity, fixed thresholds, deterministic ordering, gold-field
invariance and the archive-member allowlist.

Results will be appended from the once-executed frozen code and compact artifact;
this protocol does not predeclare a positive or negative overlap result.
