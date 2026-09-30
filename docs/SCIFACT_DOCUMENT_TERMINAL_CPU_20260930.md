# SciFact document-terminal v1 — CPU contract, not a model result

The frozen Climate v3 contract has one claim-level label and at most three
sentence references. Original SciFact requires document-specific labels and
sentence sets; different documents can have opposite labels. This package fixes
that representational mismatch without changing frozen v3, its GPU job, the
official-compatible scorer, or the original 300 dev rows.

## Contract and evidence identity

```json
{"action":"answer","documents":[
  {"source_id":"c0","label":"SUPPORTS","sentence_ids":["c0:3","c0:1","c0:4","c0:0"]},
  {"source_id":"c1","label":"REFUTES","sentence_ids":["c1:2","c1:0"]}
]}
```

`source_from_abstract` preserves the original abstract sentence tuple. Index 0
means the first original sentence, **not an entire passage collapsed to index0**.
The controller issues local aliases, renders complete sentences under the input
budget, and only those currently visible aliases/indices enter the dynamic
grammar. Each document branch binds its source alias to its own sentence enum.
After a read replaces context, previous references are invalid. Previews remain
non-citable. Strict parsing rejects duplicate documents/sentences, unknown,
cross-document and no-longer-visible references. Grammar controls syntax; server
validation is still required for uniqueness and the cumulative sentence cap.

Terminal bounds were chosen as engineering limits, without using dev gold to
select them: **up to 5 documents, 8 sentences per document, 20 total**. The shared
512-output-token budget still applies and can truncate otherwise allowed output;
these maxima are not guaranteed simultaneously reachable. Shared caps remain
20 retrieved candidates, 5 context documents, 8,192 input tokens, 5 model calls,
2 repairs, 5 tools and a 120s deadline. Equal limits do not mean equal actual cost.

The terminal path renders exact server text; `to_original_prediction` maps original document IDs
and zero-based indices; SUPPORTS becomes SUPPORT, REFUTES becomes CONTRADICT.
It neither sorts nor deduplicates nor truncates to three sentences. The converter
validates citation hashes/text against the corpus and calls `parse_prediction`
before returning the original-format prediction. The frozen scorer's
`abstract_rationalized` metric uses the first three predicted sentences **per
document**; sentence metrics use all predictions. Different-label documents do
not automatically become NEI or trigger a majority-vote claim verdict.

Valid abstentions and controller failures both yield an empty evidence matrix,
but retain different termination reasons. They remain in the evaluation
denominator; failures are not renamed successful abstentions. There is no new
claim accuracy or unmeasured entailment score. Mechanical validity is not proof
that selected sentences support the model's label.

## Version isolation and entry points

- `src/climate_rag/scifact_terminal.py`: separate prompt/renderer, typed terminal
  answer, dynamic schema, parser and original-format converter.
- `src/climate_rag/scifact_agent_v1.py`: explicit controller fork of v3/e6d1ee1.
  Same fixed-retrieval, fixed-rerank, deterministic-extra and adaptive policies;
  only terminal schema/parsing/rendering and protocol identity differ. A source-
  equivalence regression prevents silent route/budget drift. Pure Source/budget/
  query helpers are reused; no monkeypatches or global prompt replacement.
- A provider must declare this terminal protocol and count the **separate full
  prompt including schema**. The old local v3 provider has a hard-coded global-
  label prompt and is deliberately rejected. This CPU package releases **no real
  model adapter or dev inference runner**; those require a separately frozen
  runtime review. No production or concurrent-service claim.
  Subsequent [CPU provider package](SCIFACT_LOCAL_PROVIDER_CPU_20260930.md) supplies
  the implementation and synthetic-only smoke entry; it still does not authorize
  real model execution or dev evaluation.
- CPU reproduction: `python -m pytest tests/test_scifact_terminal.py -q`.
  All data in these fixtures is synthetic. No model, SciFact dev inference/gold,
  restricted data, or old public test file is read by these tests.

Fixtures cover two documents × two sentences, opposite labels, a four-sentence
ordered answer whose first-three metric changes while sentence F1 does not,
alternative rationales, noncontiguous indices, empty retrieval, explicit failure,
invalid references, context replacement, aggregate limits, all four route budgets,
separate prompt rendering and the LMFE nested grammar. These are contract and
scorer-interoperability results, not an Agent performance improvement.

Case interpretation by this chat/coordinator is **model-assisted evidence review**,
not human/blind annotation. Any real external-dev experiment remains unreleased
pending its own adapter/tokenizer smoke, fixed artifacts and explicit release.
