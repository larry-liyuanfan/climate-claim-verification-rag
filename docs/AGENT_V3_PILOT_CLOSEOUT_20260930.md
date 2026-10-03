# Sentence-ID v3: completed development pilot, not Agent-quality evidence

Job **31601753** completed `0:0` on 2026-09-30 14:54:39–14:56:42 (+10),
Elapsed123 s, TotalCPU106.957 s, batch MaxRSS18,389,028 K. Allocation: one A100,
8 CPUs,32 GiB; allocation time is not measured GPU-active time. Peak GPU memory
was not recorded (`null`). Do not infer a GPU-memory or online SLA result.

## Frozen identity and completeness

- Source `e6d1ee159516bf7a24e86e9a06b26745abf4a42d`; protocol
  `9e0ccfaf32979cb9e93bbda059bf9d094f1c2a85f190041548472a60f1492d07`.
- Source archive `bd718a1650f63648267d6da9ba247423363adf7636ada6e63c8c6db33df60681`.
- Private `run.json`: `eeb459fb4cbd410f7630fac32f28b4665ac5413071e8f9a1b3d2c7da5dc70697`.
- [Public compact](verified-runs/agent-v3-pilot-31601753.json):
  `07823634cf5af699ec36bd281962f7d013264230a08c18722106d9dedeaca79f`.
- Private runtime-preflight receipt:
  `61db0c5cc5f3b85272b500fea80c8c103c4c094b297b5eac50a93ae809f3de83`.

There are exactly24 unique task/route slots: six consumed authored prompts × four
routes over the same5,240 public passages (corpus/model/reranker hashes in compact).
BM25 is active, **dense is not**. This run does not use SciFact, original benchmark
gold, a new independent test, or the later SciFact document-terminal provider.
No frozen source/result was edited and no retry or new job was submitted.

All four forced synthetic HF/LMFE checks passed: A/B/A fresh-state legal output,
EOS, count parity and deliberate one-token truncation. Preflight separately cost
1,124 input/37 output tokens; these forced actions are not autonomous tool use.

## Actual decisions and cost, including failures

| Route | Accepted answers /6 | Abstain | Repair exhausted | Model calls | Input / output tokens | Tools / rerank pairs |
|---|---:|---:|---:|---:|---:|---:|
| Fixed retrieval |4|1|1|8|8,607 /227|6 /0|
| Fixed rerank |5|1|0|6|6,304 /171|12 /100|
| Deterministic extra |6|0|0|6|7,143 /184|18 /120|
| Adaptive |4|1|1|8|10,104 /227|6 /0|

The28 evaluation generations total32,158 input/809 output tokens; all usage is
known. All28 ended with EOS, none reached the output cap; there were no JSON
failures or failed tool events. Six generations failed duplicate-reference
validation. All tool events were scripted initial/fixed-chain operations:
**zero model-selected tools and zero subsequent decisions after a model-selected
tool**. Private raw-structure inspection confirms rejected adaptive outputs were
answers, not uncounted malformed tool requests. Route elapsed sums are6.880,
8.025,8.869,7.130 s respectively: sequential warm development measurements, not
production latency, equal actual compute, or a quality–latency frontier.

## Five concrete trace cases

Raw claims, responses and source text remain on Spartan. The source IDs and
sentence hashes below are the execution's provenance fields, not human entailment
judgments. Here each original public passage is one sentence at index0.

1. **Multi-evidence / no-progress repair.** Fixed retrieval and adaptive both
   emitted `SUPPORTS` with IDs `[c1:0,c3:0,c1:0]` on all three attempts. The raw
   output hash was identical each time:
   `759a4dc3c595f93dcef4bbfbb51f248c4e44cd0ca7a0f517eb4d94c532687b20`.
   First error was `duplicate_sentence_reference`; the next two observations used
   `duplicate_sentence_reference; choose a new legal action; read known preview candidates before citation`.
   The same five visible IDs remained; no tool executed. Adaptive failed-cost
   subtotal4,215/90 tokens is retained (retrieval3,594/90). Reranked routes instead
   selected `[c1:0,c11:0,c3:0]`, including `Fossil fuel:15`, source hash
   `8811794e473b489df5fb4417f8ad201e7075071e750842d7680853cf39852445`, sentence hash
   `5ec5975b86c837f03db4ceb38c8f5d6054b4c3d6cbac98eeb25129af9326e061`.
   This is a valid distinct-reference change, not proof all conjuncts are supported.
2. **Empty-query abstention versus irrelevant retrieval.** The authored input is
   a deliberately nonexistent token, not a natural scientific proposition.
   Adaptive could choose `abstain` or `rewrite` and selected the former; zero
   current evidence makes this a reasonable diagnostic outcome, not a missed
   proven opportunity. Deterministic extra appended a generic retrieval query and
   returned `REFUTES`, citing e.g. `Climate change denial:221` (source hash
   `71ab10c46427a17b6048c9cea012fd6b43c942335fd6fe16a5502043158a605b`, sentence hash
   `95e396f7051c83d28b7f52458e076b4a1199caa359bbecdf34161bf9d08d798b`).
   Those known IDs do not make a meaningful refutation of a nonsense token.
   Therefore6/6 mechanical acceptance is **not** superior task success.
3. **Sea-level / identical result without tools.** Retrieval and adaptive have
   the same `SUPPORTS` label and ordered original citations: `GRACE and GRACE-FO:22`,
   `Sea level rise:4`, `Sea level rise:3`. First source hash
   `cbc9185ce408b37b0ea6283bf02f8eff254f3b102c45ae3384096b3b70407824`, sentence hash
   `40532daa9d27d77da51695c0cbe3eb4a3d7d9acd4972a1bcf6f4f063238d1279`.
   Adaptive adds prompt cost but no demonstrated retrieval benefit on this case.
4. **Number/year / same label, different provenance.** Both say `REFUTES` and
   share `Greenhouse gas:5` plus `Scientific consensus on climate change:17`.
   Adaptive substitutes `Eocene:45` for retrieval's `Ecology:433`; replacement
   source hash `aad947b1d89c424b9cadfee0dd6310f62f0ef2c7d3489273916cbb1edd686aa2`,
   sentence hash `8fc60e38e9ebd92c074097d24b0ed7b7e4565d1d5a7b20ec186123ddf5df6e46`.
   Neither identical answer counts nor identical labels establish equivalent
   evidence quality, particularly for a specific numeric/year claim.
5. **Infrared / source substitution without a tool.** Both say `SUPPORTS` and
   share `Greenhouse gas:13` and `Runaway greenhouse effect:24`. Adaptive replaces
   retrieval's `Cloud:326` with `Nuclear winter:155`, source hash
   `25ecb0accc5859dbff13c46a3a3f97797ba54387f5e9a498b3905ec70b91fdae`, sentence hash
   `70a3339a104a4ce0e92f9d45d1d8e6a89d3708e0c8e71f4e22a18c99e3061e28`.
   This changes citation selection, not retrieval state; quality is unmeasured.

The fourth accepted adaptive case, ice, shares retrieval's citation **set** but
changes order. Thus four matching labels and4/6 accepted answers must not be
reported as four identical answers.

## Mechanism, uncertainty and next decision

The frozen grammar constrains each reference to an enum and array length, but
not cross-item uniqueness; the strict post-generation controller rejects repeats.
Repair gives an error code/generic instruction, not the prior rejected decision.
It does not fingerprint no-progress failures. Greedy decoding with unchanged
evidence can therefore repeat an invalid answer (the prompts themselves are not
identical: budget/feedback vary). A `uniqueItems` declaration alone is not proof
that the installed generation backend enforces uniqueness.

Possible later engineering work: structured field-specific feedback, bounded
no-progress detection keyed by context/error/normalized invalid decision, and
synthetic repair/fee tests. Do not silently deduplicate a model decision, force
tool use, alter this release, or optimize the six consumed questions to make an
Agent count positive. "Abstain when evidence insufficient" may conflate temporary
missing context with irrecoverability, but this is a hypothesis, not measured
causation; a nonsense-query fixture cannot resolve it.

The next authorized step is **CPU-only preparation**, not GPU release, of at most
12 fixed-hash, component-distinct eligible SciFact train cases spanning real
initial-rationale coverage, replenishable candidates, missing gold and NEI. Keep
train gold separate from inference, measure actual packed visibility, preserve
the four unchanged routes and all costs. This biased diagnostic is not a dev/test
benchmark. No SciFact300-dev inference, training, critic addition, small-sample
bootstrap or current-resume quality claim is authorized by this closeout.
