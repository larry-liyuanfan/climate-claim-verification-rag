# Annotation-supported terminal candidates: synthetic-only

This CPU package prepares a **semantic interface**, not the 47 real NEI records,
a training cohort or a model-quality result. No actual NEI47/remote output was
read, no model was called or trained, and no job was submitted. Frozen utility8
`e22143c` and job 31834687 are unchanged. Old `build_claim`, its ≤4-record contract,
trainer, teacher, weights and production controller remain unchanged.

## Small API and binding

`scifact_terminal_supervision.terminal_candidates` accepts an existing accepted
observation report, a `{physical_attempt_id, slot, attempt_index}` reference,
complete supplied annotation-row bytes, sealed provenance, original corpus,
tokenizer and explicit `candidate_cap` (1–64). It calls `parse_gold`,
`captured_state`, `parse_action` and `tokenize_target`, passing the **explicit
stable alias registry and current candidate order** throughout. It never calls
legacy `visible_choices/answer_targets`, rebuilds a gold prompt or changes
`current_citable`, schema, retrieval order or a source sentence.

The provenance binds full original row-byte SHA-256, accepted-report identity,
reference and full supplied corpus identity. Claim ID and normalized text must
match the accepted slot and immutable captured claim. Prompt UTF-8 identity,
utility8 JSON-string identity, actual token IDs and original model raw/proposal
are rechecked. Response, proposal, capture and per-branch event origin remain
separate from the program-derived annotation candidates. In particular,
scripted B/C tools remain `model_selected=false`.

`annotation_source=externally_supplied_synthetic_complete_original_row` and
`complete_original_row_declared=true` are **explicit synthetic declarations**.
A four-field row and a matching hash do not prove authentic or complete official
annotation. A future real release must obtain full rows through authorized,
restricted original-archive reading and independently reviewed provenance.
This package provides no such reader, CLI or real-data scope switch.

## Target rules and missing-context distinction

- One answer document must exactly match one complete original OR rationale,
  including its label and original sentence order. No union of alternatives,
  partial rationale, sentence rewriting or model self-label is allowed.
- Generated targets contain **all currently complete-visible evidence
  documents**. Only per-document OR alternatives form the Cartesian product;
  the generator does not actively omit a complete-visible document. Different
  documents can carry different labels.
- Missing or incomplete evidence in another gold document does not invalidate
  a supported local answer. `whole_gold_document_coverage` is separate from
  `complete_current_visible_document_coverage`. A four-sentence full rationale
  is also distinct from the diagnostic of whether its first three sentences
  cover that selected rationale; no official quality metric is claimed.
- Only original `evidence={}` produces `abstain/insufficient_evidence`, with
  basis `official_annotation_NEI`. Nonempty or repeated `cited_doc_ids` are
  metadata, not positive evidence.
- Positive annotation with no complete-visible rationale yields
  `context_insufficient_for_annotation_view`, **no target**. Candidate/selected
  document membership, missing original sentence indices and unchanged runtime
  budgets are recorded. A selected long document's missing rationale remains a
  gap even if another read might expose it; no future read/rewrite/rerank policy
  is selected here.

`validate_terminal_candidate` uses the same binding and semantic checks. As a
**supervision-only completeness rule**, it rejects a target that omits another
complete-visible document. This does not change production `parse_action` or
declare every correct local production answer invalid.

## Caps and representation are explicit

The full product count is computed before enumeration. If it exceeds the cap,
the whole state returns an overflow gap and **zero** candidates, not a silently
selected prefix. Within the cap, every combination gets a record. Schema or
token failures retain the full target with an explicit unrepresentable status;
no sentence, document, JSON output or EOS is truncated. Candidate count is not
reduced, no weights are assigned, and no renormalization or training-ready
cohort is created. Representable candidates still require a separate reviewed
training selection/release.

## Synthetic verification

Six test groups in `tests/test_scifact_terminal_supervision.py` cover NEI versus
missing context; complete OR/labels/coverage; actual controller rewrite/rerank
and scripted origins; unread/truncated positives and budgets; provenance
tampering; and exact prompt/mask, cap, schema and token overflow. Captures come
from the existing `run_matrix` → `run_bounded_slot` → `JournalProvider` →
`audit_observations` path, not handcrafted gold-enriched observations.

The token-overflow fixture uses an explicitly synthetic tokenizer with wider
assistant-character encoding and unchanged prompt IDs; it is a budget-contract
test, not a measured Qwen token cost. Successful tokenization checks assistant-only
loss masking and EOS even when EOS equals PAD.

```powershell
$env:PYTHONPATH = 'src;scripts'
& .\.venv-validation\Scripts\python.exe -m pytest -q tests/test_scifact_terminal_supervision.py
& .\.venv\Scripts\python.exe -m ruff check src/climate_rag/scifact_terminal_supervision.py tests/test_scifact_terminal_supervision.py
& .\.venv\Scripts\python.exe -m mypy --follow-imports=silent src/climate_rag/scifact_terminal_supervision.py
```

Verified locally on 2026-10-01: **6 new groups passed** (9.72 s); the combined
new/receipt/state-supervision/captured-state-driver scope passed **55 tests**
(34.06 s). Ruff and strict mypy passed for the new code, and the whitelisted
patch passed whitespace and common-secret-pattern checks. These are synthetic
contract/regression results, not training completion or scientific accuracy.
