# Evidence and claim boundaries

## Current repository

| Evidence | Status | Verification |
|---|---|---|
| Package and five required CLI commands | `verified` | editable install and CLI integration tests |
| BM25, hash dense exact search, RRF, hard negatives | `verified` | deterministic unit tests |
| Pairwise LTR fallback and persistence | `verified` | unit tests |
| FAISS/LightGBM/Qwen optional adapters | `implementation-only` | guards tested; full dependency/model run pending |
| Retrieval/end-to-end metrics | `verified` | synthetic fixtures with exact expected values |
| Bootstrap and calibration utilities | `verified` | seeded unit tests |
| Five-stage orchestration | `verified-smoke` | synthetic fixture with deterministic reranker |
| FastAPI retrieval | `verified-smoke` | API test confirms explicit no-classifier state |
| Full 1,208,827-document BM25 index | `verified` | Spartan job `29360715`; commit `a7b110e`; 40.333 s total, 126,334,728-byte artifact, Slurm MaxRSS 2,630,496 K |
| Full Qwen3/FAISS/LTR/reranker results | `future` | dense job `29360716` failed on home-cache quota before model loading; cache fix `2cd75e3` passed 21 tests; replacement `29382416` is queued; no effectiveness or ANN-performance claim yet |

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

