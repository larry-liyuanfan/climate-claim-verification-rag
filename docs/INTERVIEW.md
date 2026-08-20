# Interview defence and code evidence map

## 90-second story

The original course system used BM25 candidate recall, BGE reranking and a
claim classifier under Colab constraints. My portfolio extension turns that
baseline into a reproducible multistage search lab: BM25 and dense recall share
one evidence schema; FlatIP, HNSW and IVF-PQ expose ANN trade-offs; RRF provides
a training-free baseline; LambdaMART-compatible features support learned fusion;
and a cross-encoder reranker plus paired bootstrap close the evaluation loop.
On Spartan I built BM25 and Qwen3-Embedding-0.6B artifacts for all 1,208,827
evidence passages. HNSW retained 0.9961 Recall@5 versus Flat while reaching
3,060.64 batch QPS in the fixed 154-query benchmark. On the fixed 154-claim dev
split, RRF improved Recall@5 from 0.1721 to 0.2709. A first pure-0.6B rerank
regressed because it erased the strong first-stage order. I corrected the
architecture by fusing cross-encoder rank back with RRF, then gated a BF16 4B
model on the identical 154-claim/7,700-pair split. Balanced RRF/4B fusion reached
Recall@5 0.3153, MRR@10 0.3961, nDCG@10 0.2849 and Evidence F1 0.2131; all four
5,000-sample paired intervals versus RRF were above zero. The full 4B job used a
20 GB A100 MIG slice and recorded P95 4.82 seconds/query, so I expose it as an
offline quality profile and keep HNSW+RRF as the latency default. Because model
size and weights were selected on dev, I do not call this independent test
generalisation. No public rank is claimed without an official source.

## Deep-dive questions

1. Why separate first-stage recall from cross-encoder reranking?
2. When should BM25 beat a dense encoder on climate claims?
3. Why use FlatIP as the sampled ANN reference rather than another ANN index?
4. Which HNSW parameters trade build time, memory, latency and recall?
5. What failure modes make IVF-PQ attractive or dangerous at this corpus size?
6. How do you ensure every ANN backend uses identical embeddings and row IDs?
7. Why use RRF instead of directly adding BM25 and cosine scores?
8. What labels are used to train the fusion ranker?
9. Why must hard negatives exclude gold evidence?
10. Which LambdaMART features could leak labels or dataset artifacts?
11. Why rerank Top-50 rather than every recalled document?
12. How does the yes/no-logit reranker score a claim-evidence pair?
13. How do retrieval misses differ from evidence-selection errors?
14. Why report Recall@K, MRR and nDCG together?
15. What does paired bootstrap test, and what does it not prove?
16. Why temperature-scale the classifier before selective abstention?
17. How is H-mean affected when evidence F1 or claim accuracy collapses?
18. Which artifact fields are required to reproduce an experiment?
19. Why are fixture-perfect metrics not resume-quality results?
20. What evidence is required before claiming the new pipeline beats Recall@5=0.223?
21. Why did pure 4B aggregate metrics rise while its paired intervals still cross zero?
22. Why is balanced 4B fusion a dev-selection result rather than an independent test claim?

## Code evidence map

| Question area | Code or artifact to open |
|---|---|
| Evidence/claim schemas and ID stability | `src/climate_rag/models.py`, `src/climate_rag/io.py` |
| BM25 tokenization and recall | `src/climate_rag/tokenize.py`, `src/climate_rag/bm25.py` |
| Dense encoding and FlatIP/HNSW/IVF-PQ adapters | `src/climate_rag/dense.py` |
| RRF and learned-fusion feature path | `src/climate_rag/fusion.py` |
| Hard-negative mining | `src/climate_rag/negatives.py` |
| Cross-encoder/Model Studio rerank adapters | `src/climate_rag/rerank.py` |
| Calibration and abstention | `src/climate_rag/calibration.py` |
| Recall/MRR/nDCG/F1/H-mean/bootstrap | `src/climate_rag/metrics.py` |
| Five-stage orchestration | `src/climate_rag/benchmark.py`, `src/climate_rag/pipeline.py` |
| Manifests and evidence-status fields | `src/climate_rag/artifacts.py`, `docs/EVIDENCE.md` |
| Unified CLI and HTTP service | `src/climate_rag/cli.py`, `src/climate_rag/service.py` |
| Spartan/Apptainer execution | `hpc/`, `hpc/README.md` |
| Executable regression evidence | `tests/`, `fixtures/` |
