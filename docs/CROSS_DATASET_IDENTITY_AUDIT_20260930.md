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

## Once-executed CPU result

Frozen code `544a3f759e67af49b4a9f71c1718802015e90f11`, clean source checkout;
[compact result](verified-runs/cross-dataset-identity-20260930.json) SHA-256
`01e461344b3d044d1c313c681f5745ed123ad34ee9f78b3ceea9965e8805dd1b`.
All five denominators matched: 300 / 1,075 / 230 / 154 / 11 claims, and
5,183 / 5,240 documents. Found **zero direct target claim matches, zero cross-
namespace document matches, and zero SciFact dev claims connected to consumed
tracks** under this specific identity rule. No dev row was removed or reselected.

The full graph contains 206 claim-text edges, four document-variant edges and
334 components; its largest component has 1,374 claims. These are all-track
identity-graph counts, not retrieval scores. Broad public article associations
are deliberately conservative and must not be confused with positive rationale
labels or the narrower historical split grouping. Restricted source mapping and
authored source mapping remain unknown; this is **not independence certification**.

No model was run, no old test text/labels were opened, and this result does not
release SciFact dev for evaluation. Seven focused fixtures passed both locally
and from a clean Git archive. CI run `36668742278` passed 399 tests (one optional
Torch skip),
Ruff, 44-module mypy and tracked secret/PII scan. Private affected-ID and source-map
files remain ignored locally; only their hashes are included in the compact.
