# Climate Claim Verification RAG

A reproducible search-and-ranking extension of the **2026 COMP90042 Group 045 team project**. The course system used BM25 candidate retrieval, BGE bi-encoder reranking, and a LoRA-tuned claim classifier. This repository turns its retrieval work into a testable package and explicit five-stage experiment:

```text
Claim
  ├─ BM25 lexical recall
  └─ learned dense recall (optional Qwen3-Embedding)
        ↓
      RRF fusion
        ↓
      LambdaMART or declared linear pairwise fallback
        ↓
      cross-encoder/API reranker
        ↓
      evidence set → external trained classifier
```

The repository does **not** claim an official leaderboard rank. Restricted course data, raw predictions, and private checkpoints are not redistributed.

## What is implemented

| Layer | Implementation | Truth boundary |
|---|---|---|
| Lexical retrieval | Deterministic inverted-index BM25 with the course tokenizer and trusted-artifact persistence | Full 1,208,827-document Spartan build verified; retrieval quality evaluation remains separate |
| Dense retrieval | Deterministic hash smoke encoder; Sentence Transformers adapter with reusable, ID-hashed embeddings | Full 1,208,827-document Qwen3 build verified; hash mode is not a semantic model |
| ANN | NumPy exact IP plus FAISS FlatIP, HNSW, and IVF-PQ adapters | Full-corpus fixed-query comparison verified; HNSW retained as the quality-speed default, IVF-PQ rejected by the quality gate |
| Fusion/LTR | RRF; LightGBM LambdaMART when installed; deterministic linear pairwise fallback | Fixed-dev RRF improved over BM25; the trained LambdaMART regressed sharply and is not a deployment candidate |
| Reranking | Optional local Qwen3 model, Alibaba Model Studio adapter, deterministic feature fallback | Pure Qwen replacement regressed; a base-heavy weighted-rank fusion preserved RRF and improved three ranking metrics, so it is retained only as a measured offline quality profile |
| Evaluation | Recall@K, hit rate, MRR@10, nDCG@10, evidence P/R/F1, claim accuracy, H-mean, paired bootstrap | Macro-averaged over claims |
| Confidence | Temperature scaling and coverage-risk/selective-abstention utilities | Requires real classifier logits |
| Serving | FastAPI evidence retrieval endpoint | Returns `classification.status=not_configured`; never fabricates a label |
| Provenance | Input hashes, Git SHA, environment, metrics, predictions, error cases, report | Generated for every CLI run |

## Quick start

```bash
python -m pip install -e ".[test]"
pytest

climate-rag index \
  --evidence fixtures/evidence.json \
  --backend both \
  --output-dir artifacts/smoke-index

climate-rag evaluate \
  --claims fixtures/claims.json \
  --predictions fixtures/predictions_candidate.json \
  --baseline-predictions fixtures/predictions_baseline.json \
  --output-dir runs/smoke-eval
```

The candidate fixture is intentionally perfect and tests only the scorer: Recall@5, Evidence F1, Accuracy, and H-mean are `1.0`. These are **not** climate fact-checking quality metrics. The deliberately flawed fixture baseline has Recall@5 `0.50`, Evidence F1 `0.50`, Accuracy `0.75`, and H-mean `0.60`.

## Main commands

Build BM25:

```bash
climate-rag index --evidence /data/evidence.json --backend bm25 --output-dir /artifacts/bm25
```

Build Qwen3 embeddings and FAISS FlatIP, retaining a validated reusable vector cache:

```bash
climate-rag index \
  --evidence /data/evidence.json \
  --backend dense \
  --encoder sentence-transformer \
  --model Qwen/Qwen3-Embedding-0.6B \
  --ann flat \
  --embeddings /artifacts/qwen3-0.6b.npy \
  --output-dir /artifacts/dense-flat
```

The cache sidecar stores encoder, dimension, document count, and ordered document-ID hash. A mismatch fails closed. HNSW and IVF-PQ reuse the same embeddings through the example configs.

Mine hard negatives and train fusion:

```bash
climate-rag mine-negatives \
  --claims /data/train-claims.json \
  --rankings /artifacts/train_rankings.jsonl \
  --limit 20 \
  --output-dir /artifacts/hard-negatives

climate-rag train-fusion \
  --features /artifacts/ltr_features.jsonl \
  --algorithm auto \
  --output-dir /artifacts/ltr
```

`auto` uses LightGBM LambdaMART when present. Otherwise it persists `linear_pairwise_ranknet_fallback`; it is never renamed LambdaMART. Candidate rows must be split by claim before held-out evaluation.

Run the fixed five-stage comparison:

```bash
climate-rag evaluate \
  --claims /data/dev-claims.json \
  --experiment-config configs/five_stage.example.yaml \
  --output-dir /artifacts/five-stage
```

It evaluates `bm25`, `dense`, `rrf`, `ltr`, and `ltr_reranker` with one claim split and one `final_k`, then bootstraps each stage against BM25. The configured reranker name is recorded. Deterministic fallback results cannot be described as Qwen3 results.

Serve retrieval:

```bash
climate-rag serve --bm25-index /artifacts/bm25/bm25.pkl.gz --host 0.0.0.0 --port 8000
```

`POST /retrieve` accepts `{"claim_text": "...", "top_k": 5}` and returns evidence plus an explicit unconfigured-classifier state.

## Data and artifacts

Claims use the course-compatible keyed JSON schema with `claim_text`, optional `claim_label`, and `evidences`. Evidence may be a keyed JSON object, list records, or JSONL. Installing `ijson` streams a keyed object; otherwise JSON is loaded into memory.

Every command writes `run_manifest.json`, `metrics.json`, `predictions.jsonl`, `error_cases.jsonl`, and `report.md`. Non-secret arguments, Git SHA, input hashes, environment, and Slurm job ID are recorded.

The restricted full corpus has been located on Spartan at:

```text
/data/gpfs/projects/punim2936/nlp/COMP90042_2026-main/data
```

It contains the 167 MB `evidence.json` and train/dev/test claim files. It is referenced only by Slurm/Apptainer configuration and must not be copied into GitHub. See [`hpc/README.md`](hpc/README.md).

### Verified Spartan BM25 build

The native-module run on 2026-08-18 produced a full-corpus lexical index. These are engineering measurements, not retrieval-quality scores:

| Field | Verified value |
|---|---:|
| Slurm job | `29360715` (`COMPLETED`, exit `0:0`) |
| Git commit | `a7b110e8d647e9e6f51272d02c2436d4a346a27c` |
| Evidence documents | `1,208,827` |
| Vocabulary | `531,996` |
| BM25 build time | `36.886 s` |
| Total command time | `40.333 s` |
| Slurm elapsed time | `51 s` |
| Serialized index | `126,334,728 bytes` |
| Slurm MaxRSS | `2,630,496 K` |

The restricted artifact remains under project storage at `climate-artifacts/bm25`; only its non-sensitive metrics and provenance are published.

### Verified Spartan Qwen3 dense and FlatIP build

The replacement job completed on 2026-08-19 after commit `2cd75e3` moved Hugging Face caches from the quota-limited home directory to project storage. These are full-corpus **index-build measurements**, not retrieval-effectiveness scores:

| Field | Verified value |
|---|---:|
| Slurm job | `29382416` (`COMPLETED`, exit `0:0`) |
| Git commit | `2cd75e3b2cc2af059a9880e9482d8e814c94cdc5` |
| Encoder | `Qwen/Qwen3-Embedding-0.6B` |
| Evidence vectors | `1,208,827` |
| Vector dimension | `1,024` |
| Dense/FlatIP build time | `1,694.743 s` |
| Total command time | `1,696.767 s` |
| Slurm elapsed time | `28 min 25 s` |
| Embedding cache | `4,951,355,520 bytes` |
| FAISS FlatIP index | `4,951,355,437 bytes` |
| Dense artifact total | `5,133,839,551 bytes` |
| Slurm MaxRSS | `22,583,288 K` (`21.54 GB` via `seff`) |
| Allocation | `1× H100`, `8 CPU`, `96 GB`; `0.474 H100-hours` wall allocation |

The cache sidecar records the model, dimension, document count, and ordered document-ID hash. The run manifest records job `29382416`, Git SHA, environment, and the restricted input hash. Its start/finish timestamps were identical because the original writer generated both at artifact-write time; the follow-up code fixes that provenance defect, so the verified duration above comes from `metrics.json` and Slurm accounting. The 4.95 GB embeddings, 4.95 GB index, and restricted corpus remain on Spartan.

The dense/FlatIP build itself produced no effectiveness result. The separately completed CPU-only ANN benchmark and fixed-dev run below provide the quality-speed and retrieval comparisons.

### Verified Spartan ANN quality-speed comparison

Jobs `29418470` and `29418595` reused the same 1,208,827-row, 1,024-dimensional Qwen3 embedding cache. The benchmark used 154 fixed dev queries, 32 FAISS CPU threads, three batch-search repeats, and FlatIP as the ANN ground truth.

| Index | Recall@5 vs Flat | Batch QPS | Single-query P50 / P95 | FAISS index bytes | Decision |
|---|---:|---:|---:|---:|---|
| FlatIP | `1.0000` | `5.15` | `411.84 / 432.68 ms` | `4,951,355,437` | exact reference |
| HNSW (`M=32`, `efSearch=64`) | `0.9961` | `3,060.64` | `12.88 / 15.41 ms` | `5,280,336,294` | retained quality-speed default |
| IVF-PQ (`nlist=4096`, `nprobe=32`, `m=32`) | `0.3688` | `8,436.04` | `1.52 / 1.69 ms` | `66,211,820` | rejected: excessive recall loss |

The QPS values are batched, in-memory measurements on this fixed 154-query/32-thread run; they are not an online-service SLA. HNSW build time was `374.750 s` with batch MaxRSS `19,080 M`; IVF-PQ build time was `385.315 s` with batch MaxRSS `17,073,512 K`. Restricted indexes remain on Spartan.

### Verified fixed-dev retrieval comparison

Job `29435589` evaluated 154 restricted dev claims at `final_k=5` using commit `636e915`, the same BM25/HNSW candidate stores, and 5,000 paired bootstrap samples. The run was CPU-only (`32 CPU`, `32 GB` request), completed in `1 min 42 s`, and reached batch MaxRSS `11,596,892 K`.

| Stage | Recall@5 | MRR@10 | nDCG@10 | Evidence F1 |
|---|---:|---:|---:|---:|
| BM25 | `0.1721` | `0.2513` | `0.1644` | `0.1168` |
| Qwen3 dense/HNSW | `0.2696` | `0.3308` | `0.2487` | `0.1768` |
| BM25+dense RRF | `0.2709` | `0.3446` | `0.2495` | `0.1785` |
| Pure Qwen3-Reranker-0.6B replacement (first run) | `0.2438` | `0.3053` | `0.2180` | `0.1573` |
| RRF + Qwen3-0.6B weighted rank fusion (4:1) | **`0.2890`** | **`0.3801`** | **`0.2739`** | **`0.1905`** |
| LambdaMART | `0.0029` | `0.0065` | `0.0030` | `0.0027` |
| LambdaMART + deterministic reranker | `0.0127` | `0.0359` | `0.0149` | `0.0111` |

Against BM25, RRF improved Recall@5 by `0.0988` (paired-bootstrap 95% interval `0.0543–0.1452`) and Evidence F1 by `0.0616` (`0.0360–0.0893`). The first pure Qwen replacement run (`29448904`) regressed, which exposed an architectural error: it discarded a strong first-stage ordering. Job `29452723` therefore rescored the same `7,700` RRF Top-50 claim-document pairs with a climate-specific instruction and fused Qwen rank back with RRF rank. The selected 4:1 base-heavy profile improved Recall@5 by `0.0181` versus RRF (95% interval `0.0005–0.0389`, two-sided bootstrap `p=0.044`), MRR@10 by `0.0355` (`0.0069–0.0648`), and nDCG@10 by `0.0244` (`0.0085–0.0414`). Evidence F1 rose by `0.0120`, but its interval `-0.0004–0.0248` crosses zero and is not claimed as stable. The job completed in `15 min 50 s`, used batch MaxRSS `12,709,584 K`, and recorded reranker P50/P95 `5.57/6.52 s` per query. HNSW+RRF remains the latency-oriented default; the 4:1 fusion is an offline quality profile. LambdaMART and the deterministic reranker remain rejected. Because `final_k=5`, recorded Recall@10/50 equals Recall@5 and is not presented as a wider-cutoff result. Claim classification was not configured, so zero claim accuracy/H-mean values are not classifier findings.

## Historical result boundary

Local Group 045 records contain non-equivalent evaluations:

- final notebook development output: Evidence F `0.1763`, Accuracy `0.6234`, H-mean `0.2749`;
- separate BGE/BM25-top-1000 experiment record: Recall@5 `0.223`;
- rounded training document: approximately Evidence F `0.19`, Accuracy `0.61`, H-mean `0.29`.

These are **historical project records**, not reproduced package results, and must not be mixed. An earlier README claimed a public rank and snapshot metrics without a stable official artifact; those claims have been removed.

## Attribution and publication boundary

- The 2026 COMP90042 submission was a **Group 045 team project**. Do not present the course pipeline, data, or team output as one person's independent work.
- This repository is a post-course portfolio engineering extension. Git history and the evidence ledger identify later additions.
- Course data, teammate identifiers, raw private predictions, checkpoints, and credentials are excluded.
- No open-source license is granted at present. Public visibility does not itself grant reuse rights.

## Technical references

- [Qwen3 Embedding technical report](https://arxiv.org/abs/2506.05176) and [official implementation](https://github.com/QwenLM/Qwen3-Embedding)
- [FAISS index families](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [LightGBM learning-to-rank parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html#learning-to-rank-parameters)
- [Spartan job submission](https://dashboard.hpc.unimelb.edu.au/job_submission/) and [container guidance](https://dashboard.hpc.unimelb.edu.au/software/containers/)

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`docs/DECISIONS.md`](docs/DECISIONS.md), [`docs/EVIDENCE.md`](docs/EVIDENCE.md), and the [interview defence/code map](docs/INTERVIEW.md).
