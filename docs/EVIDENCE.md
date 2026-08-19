# Evidence and claim boundaries

## Current repository

| Evidence | Status | Verification |
|---|---|---|
| Package and five required CLI commands | `verified` | editable install and CLI integration tests |
| BM25, hash dense exact search, RRF, hard negatives | `verified` | deterministic unit tests |
| Pairwise LTR fallback and LightGBM persistence | `verified` | 24-test clean environment; commit `636e915` fixes persisted LightGBM feature metadata |
| Qwen3 encoder and FAISS FlatIP adapter | `verified-build` | full 1,208,827-vector Spartan build plus fixed-dev effectiveness run completed |
| FAISS HNSW/IVF-PQ | `verified` | jobs `29418470`/`29418595`; fixed 154-query FlatIP-grounded quality-speed comparison |
| Retrieval/end-to-end metrics | `verified` | synthetic fixtures with exact expected values |
| Bootstrap and calibration utilities | `verified` | seeded unit tests |
| Five-stage orchestration | `verified-smoke` | synthetic fixture with deterministic reranker |
| FastAPI retrieval | `verified-smoke` | API test confirms explicit no-classifier state |
| Full 1,208,827-document BM25 index | `verified` | Spartan job `29360715`; commit `a7b110e`; 40.333 s total, 126,334,728-byte artifact, Slurm MaxRSS 2,630,496 K |
| Full Qwen3 embeddings + FAISS FlatIP reference index | `verified-build` | Spartan job `29382416`; commit `2cd75e3`; 1,208,827 × 1,024-d vectors; 1,696.767 s total; 5,133,839,551-byte dense artifact; MaxRSS 22,583,288 K |
| HNSW quality-speed result | `verified` | Recall@5 vs Flat `0.9961`; batch QPS `3,060.64`; P50/P95 `12.88/15.41 ms`; 5,280,336,294-byte index |
| IVF-PQ quality-speed result | `verified-negative` | 66,211,820-byte index and batch QPS `8,436.04`, but Recall@5 vs Flat only `0.3688`; rejected by quality gate |
| Fixed-dev RRF retrieval result | `verified` | job `29435589`, 154 claims, `final_k=5`: Recall@5 `0.2709` vs BM25 `0.1721`; Evidence F1 `0.1785` vs `0.1168`; 5,000-sample paired intervals exclude zero |
| LightGBM LambdaMART fixed-dev result | `verified-negative` | Recall@5 `0.0029`; severe regression, not selected for deployment |
| Deterministic reranker fixed-dev result | `verified-negative` | Recall@5 `0.0127`; explicitly not Qwen3 and not selected |

## Historical Group 045 records

| Record | Status | Boundary |
|---|---|---|
| 1,208,827 evidence passages | `verified` | counted by full BM25 index artifact; data not redistributed |
| 1,228 train claims | `project-record-only` | final notebook output; not re-counted by the BM25 build |
| BGE/BM25-top-1000 Recall@5 0.223 | `project-record-only` | separate experiment, not reproduced here |
| Notebook dev F 0.1763 / Accuracy 0.6234 / H-mean 0.2749 | `project-record-only` | exact local notebook output |
| Rounded F 0.19 / Accuracy 0.61 / H-mean about 0.29 | `project-record-only` | training document; different/rounded run |
| Public rank 5 and snapshot metrics | `unsupported` | no stable official artifact located; removed |

## Resume rule

Before reporting an improvement, retain the data/split hash, metric definition, model/index parameters, hardware/runtime/cost, Git commit, paired comparison, error cases, and team-versus-individual boundary. Fixture scores only prove scorer behavior.

The `29382416` artifact manifest correctly records the job, Git SHA, environment, and restricted input hash, but its start and finish timestamps are identical because the old writer created both at artifact-write time. Runtime claims therefore use `metrics.json` and Slurm accounting. The follow-up code records command start time explicitly.

The fixed-dev run manifest records job `29435589`, commit `636e9159a14a59248ce8c4e93c396370c4af508e`, dev-claim hash `ea9976e8...`, config hash `8ae39fb2...`, `final_k=5`, 5,000 bootstrap samples, and `deterministic-feature-fallback`. It is a retrieval-only experiment: the classifier is unconfigured, and Recall@10/50 are not separately interpretable because only five final documents were emitted.

