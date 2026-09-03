# Search representation training case

## Problem

The useful hiring story is not “several retrieval components exist”. It is a
controlled model-selection case: improve semantic recall over 1,208,827 climate
evidence passages without losing exact entity/year matches, then decide where a
cheap ANN/LTR path and an expensive cross-encoder belong.

Two evidence tracks are intentionally separate:

| Track | Purpose | Allowed claim |
|---|---|---|
| Restricted COMP90042 corpus | full-scale indexing, hard-negative task adaptation and fixed 154-claim offline development | model selection on offline dev only |
| Public CLIMATE-FEVER | reproducible data adapter and the historical BM25 external baseline | no new candidate test result; the frozen test was consumed on 2026-08-25 |
| Fixtures | executable schema and failure-path tests | no retrieval-quality claim |

## Method and reproducible chain

```text
claim-group split
  -> BM25 + base Qwen3 dense candidate generation
  -> false-negative-filtered hard-negative mining
  -> 20-step Qwen3-Embedding LoRA/InfoNCE task adaptation
  -> FlatIP reference / HNSW serving candidate
  -> RRF or RRF-prior LambdaMART
  -> Qwen3-Reranker-4B with rank-preserving fusion
  -> paired metrics, taxonomy slices and Pareto decision
```

The training data builder keeps an entire claim in one partition, removes gold
and duplicate-text false negatives and hashes its inputs. The 20-step LoRA run
is a bounded task-adaptation experiment, not pretraining. The
`evaluate-representation` command refuses a base/adapted comparison unless
query IDs, corpus hash, candidate-universe hash, candidate width, final cutoff
and data hash are identical. It reports Recall@5, MRR@10, nDCG@10 and Evidence
F1 with at least 5,000 paired bootstrap samples.

`mine-negatives` now constructs LTR rows from the exact Top-K RRF candidate set
used at inference. A positive outside that set is counted as unreachable and is
not injected with all-zero retrieval features. Five-stage runs persist feature
order, candidate width, positive reachability and BM25-only/dense-only/both
candidate counts. `audit-stage-contract` blocks promotion on width, feature,
source-support or source-distribution mismatch. This correction has tests but
has not been relabelled as a new quality win; historical LTR metrics remain
historical evidence.

The query taxonomy contains entity, numeric/year, geographic, lexical-mismatch,
semantic-inference, multi-evidence and unanswerable slices. These labels are
deterministic diagnostics derived from query/gold text, not human semantic
annotations.

## Results

On the restricted 154-claim offline-dev track, the task-adapted 0.6B encoder was
paired against the base encoder on the same 1,208,827-document corpus. Recall@5
rose `0.2793 -> 0.2970`, MRR@10 `0.3633 -> 0.3869`, nDCG@10
`0.2994 -> 0.3203`, and Evidence F1 `0.07253 -> 0.07544`; all four
5,000-sample paired intervals were above zero. The adapter therefore passed the
offline-dev promotion gate. This is not an independent-test or online A/B
result.

The downstream ranking case exposed two operating points. HNSW+RRF reached
Recall@5/Evidence F1 `0.2709/0.1785`. RRF-prior LambdaMART improved
MRR@10/nDCG@10 `0.3446/0.2495 -> 0.3648/0.2605` at a derived stage-cost P95
of `23.21 ms/query`, but Recall@5/F1 intervals crossed zero. Balanced
RRF/Qwen3-Reranker-4B reached the best dev Recall@5/Evidence F1
`0.3153/0.2131`, while its measured reranker P95 was `4.82 s/query`. Thus LTR
is retained as a low-latency rank-position profile and 4B as the offline quality
profile. These timings exclude BM25 and are not an end-to-end SLA.

## Leakage and frozen-test decision

The CLIMATE-FEVER frozen test was consumed by the BM25 baseline on 2026-08-25:
129 decisive test claims, Recall@5 `0.4571`, MRR@10 `0.4567`, nDCG@10
`0.4221`, Evidence F1@50 `0.06467`. No verdict model or cross-encoder was in
that run.

A 2026-09-03 post-hoc audit of the exact frozen split found zero shared evidence
IDs, zero normalised claim duplicates, zero near-duplicate claim pairs and zero
decisive-evidence variants across partitions. It also found one 0.913-Jaccard
near-document pair between train and test; both annotations are
`NOT_ENOUGH_INFO` and neither entered the 129-query decisive evaluation. This
does not retroactively turn the baseline into a candidate-selection result, but
it means the historical partition fails the stricter all-document audit.
Therefore `configs/public_evaluation_policy.json` marks the test consumed and
forbids every new candidate evaluation. The validation promotion gate was not
run in this closeout because no new public-validation representation artifact
exists. No new frozen-test score was produced.

## Problem-method-result-trade-off summary

I treated climate evidence retrieval as a representation-and-ranking decision,
not a model-size contest. I mined claim-grouped hard negatives, filtered false
negatives and applied a 20-step LoRA/InfoNCE adaptation to the efficient Qwen3
dense retriever; on the same full corpus and 154-query offline-dev set it
improved all four paired retrieval metrics. I then preserved lexical and dense
signals through RRF, corrected an unreachable-positive/train-serve mismatch in
LambdaMART, and compared a millisecond CPU ranker with a 4B cross-encoder. The
4B path achieved the best quality but cost seconds per query, while LTR improved
rank-position metrics at far lower stage cost. I kept both operating points and
retired the already-consumed public test rather than tune against it.

## Resume bullet candidates

- Adapted Qwen3 dense retrieval with claim-grouped hard negatives and a 20-step
  InfoNCE/LoRA task adapter over 1.21M evidence passages; on the same 154-query
  offline-dev set, raised Recall@5 `0.279 -> 0.297` and MRR@10
  `0.363 -> 0.387`, with positive 5,000-sample paired intervals across four
  retrieval metrics.
- Built a leakage-audited BM25+dense/HNSW -> RRF/LambdaMART -> Qwen3-4B
  ranking pipeline; selected a low-latency LTR profile and an offline quality
  profile (`0.315` Recall@5, `0.213` Evidence F1), while enforcing exact
  train/serve candidate contracts and blocking reuse of a consumed public test.

Both bullets require the qualifiers “offline dev” and “fixed in-memory/stage
benchmark” when expanded in an interview. They do not claim independent-test
generalisation, an online SLA, classification quality or an A/B result.
