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
| Dense retrieval | Deterministic hash smoke encoder; optional Sentence Transformers adapter | Hash mode is not a semantic model |
| ANN | NumPy exact IP plus optional FAISS FlatIP, HNSW, and IVF-PQ adapters | Full 1.2M-corpus benchmark not yet run by this package |
| Fusion/LTR | RRF; LightGBM LambdaMART when installed; deterministic linear pairwise fallback | Artifact names the actual algorithm |
| Reranking | Optional local model, Alibaba Model Studio adapter, deterministic feature fallback | Fallback is never presented as Qwen3 |
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

The restricted artifact remains under project storage at `climate-artifacts/bm25`; only its non-sensitive metrics and provenance are published. Qwen3 embeddings, FAISS ANN comparisons, fusion, and reranking remain unverified until their own artifacts complete.

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
