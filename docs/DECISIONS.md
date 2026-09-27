# Decision log

## D9 — Audit ranking depth and checkpoint values before model selection

On 2026-09-27 a source audit found two distinct integrity defects. First,
PEFT 0.15.2 did not implement the loader's assumed key-mapping argument;
parameter counts did not prove checkpoint restoration. The repaired loader
compares every restored tensor and enabled/disabled/re-enabled embeddings.
CPU job `31364439` passed on the real selected public checkpoint, without
opening test or measuring a new retrieval gain.

Second, a small served-evidence cutoff had also limited metric rankings:

| Historical path | Exact source evidence | Correct interpretation |
|---|---|---|
| Restricted LTR `29504398` | `b47e437:configs/five_stage.candidate_supported_ltr.yaml:9`; `benchmark.py:231,283–294` | `final_k=5` reached the scorer; Recall@5/F1@5 valid, MRR equals MRR@5, nDCG@10 censored |
| Restricted 4B `29453918/29455049` | `c04e39c:configs/five_stage.qwen_reranker.4b.native.yaml:9`; `benchmark.py:219–224,256–267` | same five-rank cutoff; do not assert four complete ranking metrics improved |
| Public-v2 `30007546` | pre-repair `public_v2_runtime.score_index/predictions_from_rows` and downstream final cutoff | valid base-only Recall@5/F1@5; no complete Top-10 estimate |
| Restricted LoRA `29465819` | `c815070:scripts/evaluate_embedding_adapter_full_gate.py` enforces `top_k>=50` and preserves all ranks | Recall@5/MRR@10/nDCG@10 and F1@50 retain their original interpretation |

MRR over five available ranks equals MRR@5. The old nDCG ideal denominator
used up to ten positives, so relabelling it nDCG@5 would be wrong. Raw historical
JSON/archives remain unchanged; README, evidence and interview summaries carry
this correction. Zero values from an unconfigured verdict are not classifier
performance. The 23.21 ms and 4,835.41 ms sums of component P95 values are only
historical cost proxies, never percentiles of a measured request distribution.

The public profiler retains 100 ranks with Evidence F1@5. The legacy five-stage
entry point now independently retains up to `max(50, final_k)` ranks in
`rankings_<stage>.json`, and preserves `predictions_<stage>.json` as the smaller
served set. Pure-reranker candidate limits remain real limits; no unretrieved
gold or artificial tail is injected. Regression fixtures place gold at ranks
6, 10 and 50 and require consistent CLI reproduction:

```sh
climate-rag evaluate --claims CLAIMS.json --predictions rankings_rrf.json \
  --evidence-k 5 --retrieval-only --output-dir NEW_EVALUATION
```

New profiling job `31364586` remains pinned to `b797f66`; this later legacy-path
repair does not modify the queued checkout or recompute old results. The full
search tradeoff delivery still requires its verified output and accounting.

## D0 — Retire the consumed public test and enforce representation contracts

The CLIMATE-FEVER test was consumed by the 2026-08-25 BM25 lexical baseline.
This closeout does not run a tuned representation, fusion model or reranker on
that partition. A post-hoc audit found one train/test near-document variant at
0.913 token Jaccard. Both annotations are `NOT_ENOUGH_INFO`, so the decisive
retrieval subset remains clean, but the historical split fails the stricter
all-document rule. The baseline is retained with that warning; new candidate
test execution now fails closed. At the early 2026-09-03 closeout no
public-validation candidate was available. The subsequent public-v2 pilot and
diagnostic cycle ended without a valid full adapter result or promotion; see D9.

Public v2 preparation uses a grouped split: normalised/near-duplicate
claims, shared evidence IDs and normalised/near-duplicate evidence text all
join the same claim component. Token-posting candidate blocking avoids an
unbounded all-document pair scan. The verified full-data v2 build preserves
1,075/230/230 claims and reports zero cross-partition variants, but it is not
used to manufacture a second “independent” test result.

Paired representation evaluation requires identical query, corpus,
candidate-universe, width, cutoff and data hashes, plus at least 5,000 bootstrap
samples. LTR training rows now come from the exact serving-width RRF pool.
Unreachable positives are counted and excluded instead of receiving artificial
zero retrieval features. Existing job `29504398` remains historical dev
evidence; the correction itself is not reported as a new quality improvement
until a fresh non-test artifact passes the contract.

## D1 — Preserve the lexical baseline

The tokenizer lowercases text, normalizes `CO2`/`carbon dioxide`, and retains fact-changing negations. This keeps the BM25 baseline comparable in intent to the Group 045 notebook.

## D2 — Separate smoke and learned encoders

The hash encoder makes tests deterministic and is labeled `hash-baseline-*`. Qwen3/BGE claims require the learned adapter and recorded model name.

## D3 — One mapping for FlatIP, HNSW, and IVF-PQ

FlatIP is the exact normalized-inner-product reference. HNSW trades graph memory for approximate speed; IVF-PQ adds training and compression. A reusable embedding cache avoids repeated encoding, while its ID hash prevents row misalignment.

## D4 — RRF before learned fusion

BM25 and dense scores are not naturally calibrated. RRF is the parameter-light baseline. Learned fusion adds score/rank plus token, number, and year-consistency features.

## D5 — Never rename a fallback as LambdaMART

When LightGBM is present, `auto` trains `LGBMRanker(objective="lambdarank")`. Otherwise it trains and persists a deterministic linear pairwise logistic ranker named `linear_pairwise_ranknet_fallback`.

## D6 — Keep classification decoupled

The historical LoRA classifier/checkpoint is not distributed. Evaluation accepts predictions and calibration accepts logits. Retrieval serving returns no label until a real classifier is configured.

## D7 — Require paired evidence

Improvements are compared per claim on the same split. Paired bootstrap reports difference and interval. Target gains remain aspirations until a restricted-data run creates manifests and outputs.

## D8 — Deploy the measured winner, not the most complex stage

The fixed-dev run first selected BM25+dense RRF over BM25. A separate dense-encoder size gate then compared 0.6B and 4B at the same 1,024-dimensional output on an evidence-preserving sample: 4B reduced Recall@5, tied F1/MRR, ran about 7.1× slower and used about 5.5× the peak Torch GPU memory, so the full 4B index rebuild was stopped. A pure Qwen3-Reranker-0.6B replacement also regressed because it discarded the strong RRF order. Weighted rank fusion corrected that architectural error. The reranker size gate compared 4B on the identical split and resource-tested it with BF16 on a 20 GB MIG slice. A balanced RRF/4B fusion improved Recall@5 and Evidence F1@5 with paired intervals above zero. Its historical Top-10 fields came from Top-5 predictions, so they are not complete Top-10 evidence (see D9). An 8B eight-claim pilot then tied 4B F1/Recall, slightly reduced MRR and increased P95 by 60.8%, so the full 8B run was stopped at the gate. The 4B reranker remains the offline quality profile, while the 0.6B encoder with HNSW+RRF remains the latency default. Model size and fusion weights were selected on the same dev split, so the result is not described as independent test generalisation. The legacy LTR result was invalidated after finding injected, unretrieved positives with zero retrieval features. Candidate-supported LambdaMART removed that defect, but job `29484697` still collapsed on dev despite `0.9529` train pairwise accuracy; it is a valid rejected model. Job `29504398` restored the serving RRF prior, aligned the 100-candidate train/serve width and evaluated a fixed 4:1 fusion. Top-5 reciprocal rank improved over RRF with a positive paired interval, while Recall/F1 intervals crossed zero and absolute quality stayed below the 4B fusion, so the result is retained only as a CPU rank-position profile. IVF-PQ remains rejected because Recall@5 versus Flat was only `0.3688`; HNSW remains the ANN default.

Dynamic routing was tested after the model-size gate, not assumed from paper
results. A five-fold cross-fit ridge router over pre-rerank list agreement
avoided 56.49% of 4B calls and remained significantly better than RRF, but
preserved only 36% of the strong path's Evidence-F1 gain. Hashed claim-text
features reduced the call rate further but generalised no better than RRF.
Neither met the predeclared 80% gain-preservation threshold. They remain
auditable negative/partial results; always-4B and no-4B are still the two
selected quality/latency profiles.

The next dense-encoder gate was task adaptation, not another parameter-scale
guess. Mined train-claim negatives were converted to Qwen3-Embedding InfoNCE
rows with a claim-grouped validation split and false-negative controls. A
20-step LoRA run and runtime injection preflight completed. The 126-claim
evidence-preserving sampled screen improved Recall@5, MRR and nDCG with positive
paired intervals, but Evidence F1 crossed zero, so no sampled-only promotion
was made. Full-corpus job `29465819` then evaluated all 154 official-dev claims
and 1,208,827 documents. Recall@5 improved `0.2793→0.2970` with a paired 95%
interval `0.0014–0.0350`; MRR, nDCG and Evidence F1 also improved with positive
intervals. Here Evidence F1 uses Top-50, unlike downstream F1@5. The adapter therefore passes the pre-registered offline dev gate and
becomes the preferred dense retrieval candidate. The base 0.6B encoder remains
the fallback because this is dev-set model selection, not an independent test
or online A/B result. The failed first full run also changed artifact policy:
rebuildable multi-gigabyte adapted indexes are opt-in, while metrics, manifests
and hashes are mandatory.

