# Physical receipts → accepted observations (synthetic-only seam)

**Accepted observation ≠ correct supervision target ≠ Agent gain.** This library
audits lineage, not scientific relevance or the utility of a model decision. No
target is selected, no `program_teacher` label is assigned, and no training or
evaluation release is created. The source contract is frozen utility8 `e22143c`;
this package does not modify its submitted source or inspect its real outputs.

## Input and output boundary

`scifact_observation_receipts.audit_observations(artifacts, roster, tokenizer, corpus)`
takes explicitly supplied `(relative_name, JSON_bytes)` pairs, an independently
sealed slot roster, a tokenizer and original sources. It has **no filesystem
reader, CLI, model loader or gold loader**. The current roster scope must be
`synthetic_fixture`; this is not authorization to import real TRAIN trajectories.
`contract_revision` identifies the layout being audited, not proof that input
bytes were actually produced by that Git revision.

The result keeps every external slot, including not-run/incomplete slots, their
reported attempt counts and reserved physical IDs where inventory is readable.
Its physical receipt inventory retains missing/failed completion status rather
than hiding those calls among accepted observations. Invalid global inventories
fail closed for all slots. An invalid A lineage also invalidates its dependent
B/C references. It never substitutes a different trajectory or normalizes a
smaller successful denominator. File-byte SHA-256 receipts preserve the supplied
inventory. There is no adapter-to-training shortcut: targets and a separate
supervision-selection review remain required.

## Linkage and identity

| Logical observation | Actual input | Required state event |
|---|---|---|
| A attempt i | `A/frame-i.json`, own physical ID | Initial retrieve, then independently checked model execution |
| B/C attempt 0 | `initial-frame.json`, shared A physical ID | Inherited retrieve; no new generation |
| B/C attempt 1 | Own `frame-0.json`, own physical ID | Completed scripted read/rerank, never model-selected |

Physical IDs join reservation, finished receipt, complete raw response and the
attempt diagnostics. File-name IDs, slot ownership, usage, actual ordered
observation/schema, rendered prompt and token identity must agree. The existing
`captured_state` contract checks original source text, stable aliases, current
candidate order, visible sentences, selected context and previews. The adapter
reuses production `parse_action`, `action_schema`, `validate_prefix` and
`read_intervention`; it does not implement another controller or trainer.

The runtime `decision_execution_audit` is checked, not trusted. Proposals are
reparsed from the complete raw response. Read parameters must match the full
ordered executed selection; rewrite hashes use `normalise_claim(query)` UTF-8;
rerank preserves the candidate set and source identities while allowing a new
order. Execution consumes events one-to-one in order. B/C's new
`proposed_not_executed` tool action cannot bind to the earlier scripted event.
Repair inherits the preceding completed event and retains the exact controller
feedback suffix. Prefix status can legitimately precede an A controller-loop
rejection; both prefix and final status are retained as separate logical refs.
A terminal decision cannot be followed by another A attempt.

Hash domains are deliberately separate:

- `journal_prompt_identity`: `identity(prompt)`, i.e. JSON-string encoding.
- `prompt_utf8_sha256`: raw rendered prompt UTF-8, as used by supervision tokenization.
- `token_ids_identity`: utility8 insertion-order JSON identity of actual token IDs.
- `artifact_sha256`: the supplied complete file bytes, including original whitespace.

Response diagnostics and the private-attachment receipt must match the **original
raw bytes** (not reserialized JSON), character/byte counts and output-token usage.
Responses, parsed model proposals, physical events, `origin`, `model_selected`
and per-branch attempts are retained without correction. An invalid schema
proposal can be an accepted *observation of failure*, not a successful action.

## Six synthetic validation groups

`tests/test_scifact_observation_receipts.py` produces artifacts using the actual
`run_matrix` → `run_bounded_slot` → `JournalProvider` path and production
`response_diagnostics`, with a synthetic backend and original fixture sentences.

1. Two reads, nonidentity rerank, repair, terminal; 56 unique physical observations
   versus 72 logical references over 24 declared slots. Shared A0 is stored once.
2. Prefix valid parse versus A final read-loop rejection; unavailable B stays in
   the denominator, and C keeps its inherited status. A separate actual-runtime
   invalid-JSON → legal-answer trajectory checks the exact repair suffix.
3. Missing/failed/cross-slot/duplicate receipts, unused-ID finished-only receipt,
   orphan frames, and two legal same-action answers with altered raw bytes.
4. Wrong/duplicate/unknown candidate order, schema order, observation key order
   and confusing the two prompt hash domains.
5. Same-tool/different-valid-parameter event mismatch, wrong audit index, prefix
   proposal drift, terminal continuation, raw diagnostics, and attempted binding
   of an unexecuted B/C proposal to its scripted event.
6. Normalized rewrite raw-UTF8 hashing, external synthetic scope and duplicate
   JSON-key rejection.

Run from the Climate worktree in PowerShell:

```powershell
$env:PYTHONPATH = 'src;scripts'
& .\.venv-validation\Scripts\python.exe -m pytest -q tests/test_scifact_observation_receipts.py
& .\.venv\Scripts\python.exe -m ruff check src/climate_rag/scifact_observation_receipts.py tests/test_scifact_observation_receipts.py
& .\.venv\Scripts\python.exe -m mypy --follow-imports=silent src/climate_rag/scifact_observation_receipts.py
```

Verified locally on 2026-10-01: the latest six groups passed (20.50 s), Ruff
passed, and strict mypy passed for the adapter source. The preceding affected
scope run passed 89 tests (28.11 s), including utility8, captured-state driver,
state supervision and claim-group-mean regressions. The final narrow additions
(invalid-JSON fixture, tail-audit and malformed-slot rejection) were rechecked
with the six groups; no unchanged full-repository suite was rerun. The four-file
patch was checked for whitespace and common secret patterns; none were found.

These are CPU fixture-contract checks, not real-model results, independent test
quality, financial value or evidence of autonomous Agent capability. Real NEI47
conversion, physical reranker-call-to-event binding and real training selection
remain outside this package.
