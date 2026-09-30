# Original SciFact grounded evaluation — CPU preparation, not released

This is a separately versioned gold-aware preparation/scoring proposal, not a
new score for the old CLIMATE-FEVER validation/test, restricted154-query LoRA
development track, or the authored Agent pilots. None of those sets is fresh.
Historical public-v2 records state that SciFact promotion was not authorized and
its BEIR qrels were never opened. This package reads **original train/dev only**
for data/identity audit; it runs no model and selects no candidate on dev.

## Original data and leakage boundary

Official [download script](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/script/download-data.sh)
points to the release archive pinned here by SHA
`11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be`.
The archive is3,115,079bytes. Only3 allowlisted regular members are read:
corpus, claims_train and claims_dev. No test member is opened/extracted and no
BEIR qrels are read. Original files, gold labels and per-claim assignments stay
under ignored `data/`; only [content-free manifest](verified-runs/scifact-original-preparation-20260930.json)
is public. The archive's mutable `latest` URL is not trusted without the SHA.

The importer preserves5,183 abstracts with original sentence order,809 train
claims and300 dev claims. Gold alternatives and per-document SUPPORT/CONTRADICT
labels are kept, not flattened. `cited_doc_ids` denotes source-citation metadata,
**not evidence**; one dev row has a repeated cited ID, preserved as metadata.
Empty evidence is NEI. Across-document mixed labels remain explicit MIXED for
diagnostics; no forced binary claim label. Within-document mixed labels or
overlapping alternative rationales are unsupported by the official scorer and
fail closed rather than being silently merged.

Conservative connected components join claim token-Jaccard≥0.8, document
token-Jaccard≥0.9, shared evidence documents and shared cited-source documents.
NFKC/casefold word tokens keep numbers and negations. No model, label-dependent
sampling, gold-based context selection or future dev tuning is used. This audit
found571 claim components (largest9), with278 train claims connected to dev.
Those are quarantined; **531 train claims remain eligible**, and original300
dev claims stay unchanged. There were no multi-document variant components at
the frozen0.9 threshold. These are lexical/source-family protections, **not** a
proof against every semantic paraphrase or foundation-model pretraining exposure.

Inference files contain only claim ID/text and corpus ID/title/sentence text;
no evidence labels, cited-source hints, component assignment or gold sentence
indices. Only the independent scorer receives the gold files. No unfiltered
original-train inference file is exported: training must use the eligible file.
Dev's official role remains **labelled dev**; if released later, describe it as
an external held-out dev evaluation, not the official unlabelled test. A BEIR
test mapping of this same dev cannot become a second independent experiment.

## Exact scorer semantics and fixtures

`scifact_scoring.py` follows the pinned official
[metrics implementation](https://github.com/allenai/scifact/blob/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/verisci/evaluate/lib/metrics.py):
aggregate counts across claims before computing micro P/R/F1. Four metrics are
separate: document+label, document+label+complete rationale, sentence selection,
and sentence+label. Document rationalization considers the first3 predicted
sentence entries; sentence scoring considers all predictions and credits only
sentences in a fully covered gold rationale. Alternatives are OR, not AND.
Predicted order/duplicates are preserved; no silent deduplication boosts scores.
Empty evidence predictions receive no true-negative F1 reward; NEI false-evidence
rate is a separate diagnostic with an explicit NEI denominator. No claim verdict
accuracy or free-text entailment number is inferred from these metrics.

The wrapper rejects missing/duplicate/extra claim IDs before scoring, avoiding
the official iterator's missing-row denominator pitfall. Original prediction
labels are SUPPORT/CONTRADICT; SUPPORTS/REFUTES is an explicit project mapping,
not accepted original-file syntax. Corpus/indices are validated at parsing.

The official-shaped `[1,11,13]` versus `[[0,1],[11]]` fixture credits sentence11
but not sentence1. Including a missed second gold abstract and a false positive
gives abstract F1=.5, sentence F1=2/9. This and128 generated **synthetic fixture
matrices** matched all4 P/R/F1 outputs from exact official code; no real dev
predictions were used. Reference code hashes are checked before import.

## Future four-route comparison — design only

| Route | Work policy | Purpose |
|---|---|---|
| Fixed retrieval | Same initial retrieval, bounded grounded generation | Lower-work reference |
| Fixed rerank | Same retrieval, fixed4B rerank, bounded generation | Primary comparator |
| Deterministic extra-work | Fixed schedule of permitted expansion/read/rerank | Distinguish more work/context from adaptive decisions |
| Adaptive | Model chooses allowed tools from observations, bounded stopping | Measure actual decision/control value |

All routes will share model/revisions, corpus, eligibility, input/output/context
limits and failure accounting. Equal caps do not imply equal actual work; report
tokens, tool calls, latency and Pareto. Exact v3 action set, deterministic schedule,
prompts, maximum work and release hashes remain **unfrozen until CPU/pilot review**.
No official-dev model evaluation is runnable/released by this preparation script.

`assemble_context` accepts ranked abstracts and a tokenizer callback only (no
gold/query labels). It takes complete sentences in document-rank/original-index
order, stopping at a token bound without silent sentence clipping, and records
omitted sentences/documents. Production must use the pinned real tokenizer for
generator/reranker separately, count wrapper text, and reserve system/output
budget. Prefix truncation is a disclosed limitation, not relevance optimization.

Before a future one-shot dev release: freeze model/prompt/policy/tokenizer/data
hashes, selected train configuration and all4 routes; require a complete matrix,
pre-register a primary joint document/label/rationale metric and5,000 paired
**component-level** bootstrap replicates with all failures retained. Primary
comparison is adaptive versus fixed rerank, with deterministic extra-work as a
required attribution control. Claim-task success, if added, must have its own
name and NEI/mixed-label rules. No prewritten success numbers or repeated dev
tuning; no release while the development Agent still only abstains.

## Reproduce this CPU package

```powershell
python scripts/prepare_scifact_grounding.py --archive data/scifact-original-20260930/data.tar.gz --output data/scifact-original-20260930/prepared-r1
python -m pytest tests/test_scifact_grounding.py -q
# Optional reference crosscheck; download only the two pinned official files.
# Original reference scorer depends on pandas2.2.3, not a runtime core dependency.
python scripts/crosscheck_scifact_reference.py --reference-dir data/scifact-original-20260930/reference
```

Preparation refuses an existing output directory. Preserve its manifest/hashes;
do not overwrite it to hide changed split eligibility. This is CPU engineering
preparation, not a new model-quality result or current-resume bullet.
