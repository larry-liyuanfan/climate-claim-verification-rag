# Climate Evidence Retrieval and Grounded Verification

A reproducible search-and-ranking extension of the **2026 COMP90042 Group 045 team project**. The course system used BM25 candidate retrieval, BGE bi-encoder reranking, and a LoRA-tuned claim classifier. This repository now separates two evidence tracks: a restricted 1.21M-document scale/selection track on Spartan and a public CLIMATE-FEVER external benchmark that can be reproduced without course data.

```text
claim normalisation + entity/year constraints
  ├─ BM25 lexical recall
  └─ Qwen3 dense/HNSW recall
        ↓
      RRF → Qwen3-4B cross-encoder
        ↓
      evidence sufficiency / one bounded re-retrieval
        ↓
      structured verdict provider
        ↓
      validated citations or explicit abstention
```

The repository does **not** claim an official leaderboard rank. Restricted course data, raw predictions, and private checkpoints are not redistributed.

## What is implemented

| Layer | Implementation | Truth boundary |
|---|---|---|
| Lexical retrieval | Deterministic inverted-index BM25 with the course tokenizer and trusted-artifact persistence | Full 1,208,827-document Spartan build verified; retrieval quality evaluation remains separate |
| Dense retrieval | Deterministic hash smoke encoder; Sentence Transformers adapter with reusable, ID-hashed embeddings | Full 1,208,827-document Qwen3 build verified; a 20-step hard-negative LoRA adapter passed the full-corpus offline official-dev promotion gate; hash mode is not a semantic model |
| ANN | NumPy exact IP plus FAISS FlatIP, HNSW, and IVF-PQ adapters | Full-corpus fixed-query comparison verified; HNSW retained as the quality-speed default, IVF-PQ rejected by the quality gate |
| Fusion/LTR | RRF; LightGBM LambdaMART when installed; deterministic linear pairwise fallback | Fixed-dev RRF improved over BM25; the trained LambdaMART regressed sharply and is not a deployment candidate |
| Reranking | Configurable 0.6B/4B/8B local Qwen3 model, Alibaba Model Studio adapter, deterministic feature fallback | 0.6B exposed first-stage replacement failure; 4B plus balanced rank fusion improved all four fixed-dev ranking metrics; an 8B pilot failed the latency/quality Pareto gate, so 4B remains the offline quality profile |
| Public benchmark | CLIMATE-FEVER adapter, evidence-aware near-duplicate split and frozen-test BM25 baseline | 1,535 claims/7,675 annotations; final test is not used for model selection |
| Evaluation | Recall@K, hit rate, MRR@10, nDCG@10, evidence P/R/F1, verdict Macro-F1/Accuracy, citation quality, ECE/Brier and paired bootstrap | Retrieval and verification results remain separately labelled |
| Confidence | Temperature scaling, coverage-risk and selective abstention utilities | Requires provider or classifier confidence |
| Serving | FastAPI search/verify/trace/metrics endpoints with bounded re-retrieval | Verifier failure, invalid citation IDs or quote mismatch fail closed to `NOT_ENOUGH_INFO` |
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

climate-rag prepare-public \
  --output-dir data/climate-fever-public \
  --seed 20260825

climate-rag audit-public-split \
  --prepared-dir data/climate-fever-public \
  --output-dir runs/climate-fever-split-audit
```

## Public retrieval v2

The next public cycle is fully pre-registered in
[`configs/public_retrieval_v2.json`](configs/public_retrieval_v2.json). It uses
only the 1,075-claim train partition for hard-negative LoRA/InfoNCE training and
the 230-claim validation partition for pilot/full selection. The old consumed
test is permanently sealed. Six fixed Qwen3-Embedding-0.6B adapters cover
100/300 steps, rank 8/16, 4/8 hard negatives and temperatures 0.03/0.05; no more
than two can reach full validation.

After configuration freeze, exactly one official MTEB/BEIR SciFact transfer
event compares the base representation with the frozen selected adapter. The
complete method, promotion rule, Top-100 LambdaMART contract, 4B reranker fusion
and truth boundaries are in
[`docs/PUBLIC_RETRIEVAL_V2.md`](docs/PUBLIC_RETRIEVAL_V2.md).

The candidate fixture is intentionally perfect and tests only the scorer: Recall@5, Evidence F1, Accuracy, and H-mean are `1.0`. These are **not** climate fact-checking quality metrics. The deliberately flawed fixture baseline has Recall@5 `0.50`, Evidence F1 `0.50`, Accuracy `0.75`, and H-mean `0.60`. The full representation-training case and two evidence-grounded resume bullets are in [`docs/REPRESENTATION_TRAINING_CASE.md`](docs/REPRESENTATION_TRAINING_CASE.md).

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
  --limit 100 \
  --ltr-candidate-width 100 \
  --output-dir /artifacts/hard-negatives

climate-rag train-fusion \
  --features /artifacts/ltr_features.jsonl \
  --algorithm auto \
  --output-dir /artifacts/ltr
```

The retained 0.6B dense encoder is improved through task adaptation rather than
blind parameter scaling. `scripts/prepare_embedding_training.py` converts mined
training negatives into the current ms-swift Qwen3-Embedding InfoNCE format,
keeps every claim wholly in train or validation, removes gold/duplicate-text
false negatives, and emits hashed run artifacts. Spartan job `29462754`
completed a bounded 20-step LoRA/InfoNCE run, and preflight `29463845` verified
that 5,046,272 adapter parameters were injected into the serving encoder.

The claim-grouped sampled gate (`29463846`) retained all 368 labelled positives
for 126 held-out claims in a 5,000/1,208,827-document corpus. Recall@5 changed
from `0.6090` to `0.6303` (paired 95% interval `0.0048–0.0429`), while MRR and
nDCG intervals were also positive. Evidence F1 changed from `0.1087` to `0.1095`
with an interval crossing zero (`-0.0003–0.0026`), so that run remained only a
sampled screen.

The complete official-dev replacement gate (`29465819`) then evaluated all 154
claims and all 1,208,827 evidence passages. Recall@5 changed from `0.2793` to
`0.2970` (paired 95% interval `0.0014–0.0350`), MRR@10 from `0.3633` to
`0.3869`, nDCG@10 from `0.2994` to `0.3203`, and Evidence F1 from `0.07253` to
`0.07544`; all four 5,000-sample paired intervals were above zero. The adapter
therefore passes the pre-registered offline promotion gate. It is an
official-dev result, not independent test generalisation or an online A/B test.
Compact sampled and full-corpus records are published in
[`docs/verified-runs/qwen3-embedding-lora-sampled-gate-20260821.json`](docs/verified-runs/qwen3-embedding-lora-sampled-gate-20260821.json)
and
[`docs/verified-runs/qwen3-embedding-lora-full-gate-20260821.json`](docs/verified-runs/qwen3-embedding-lora-full-gate-20260821.json).

`auto` uses LightGBM LambdaMART when present. Otherwise it persists `linear_pairwise_ranknet_fallback`; it is never renamed LambdaMART. Candidate rows must be split by claim before held-out evaluation. LTR rows are generated from the exact RRF Top-K used by serving; positives outside that set are counted as unreachable instead of being injected with zero retrieval features.

Paired base/adapted comparison and train/serve consistency checks:

```bash
climate-rag evaluate-representation \
  --claims /data/dev-claims.json \
  --evidence /data/evidence.jsonl \
  --baseline-predictions /artifacts/base.json \
  --candidate-predictions /artifacts/adapted.json \
  --baseline-contract /artifacts/base-contract.json \
  --candidate-contract /artifacts/adapted-contract.json \
  --bootstrap-samples 5000 \
  --output-dir /artifacts/representation-pair

climate-rag audit-stage-contract \
  --training-contract /artifacts/training-contract.json \
  --serving-contract /artifacts/serving-contract.json \
  --output-dir /artifacts/stage-contract-audit

climate-rag build-pareto \
  --profiles configs/search_profiles.verified.json \
  --output-dir /artifacts/search-pareto
```

The `configs/stage_contract.*.example.json` files contain labelled fixture
values only. Promotion requires contracts generated from measured artifacts.

Run the fixed five-stage comparison:

```bash
climate-rag evaluate \
  --claims /data/dev-claims.json \
  --experiment-config configs/five_stage.example.yaml \
  --output-dir /artifacts/five-stage
```

It evaluates `bm25`, `dense`, `rrf`, `ltr`, and `ltr_reranker` with one claim split and one `final_k`, then bootstraps each stage against BM25. The configured reranker name is recorded. Deterministic fallback results cannot be described as Qwen3 results.

Serve multi-stage retrieval and grounded verification:

```bash
climate-rag serve \
  --bm25-index /artifacts/bm25/bm25.pkl.gz \
  --dense-index /artifacts/dense-hnsw \
  --reranker qwen-local \
  --reranker-model Qwen/Qwen3-Reranker-4B \
  --verifier model-studio \
  --verifier-model qwen3.7-plus \
  --max-queries 2 \
  --host 0.0.0.0 --port 8000
```

The service exposes `POST /api/search`, `POST /api/verify`, `GET /api/traces/{trace_id}`, `GET /metrics`, and `GET /health`; legacy `POST /retrieve` remains for compatibility. Model Studio credentials are read only from environment variables. Without a configured verifier the service returns an explicit abstention rather than fabricating a verdict.

## Public external baseline

The public adapter pins the upstream CLIMATE-FEVER source hash, deduplicates 5,240 evidence passages, and keeps claims connected by shared evidence, normalised claim duplicates, high-similarity claim text, and exact/near-duplicate evidence text in the same partition. The v2 `70/15/15` split (seed `20260825`) contains 1,075/230/230 claims and passes the strict cross-partition claim/document-variant audit. It is a new split protocol, not a way to re-open the historical frozen test.

The first frozen-test run is deliberately only a lexical baseline. Among the 129 test claims with SUPPORTS/REFUTES evidence, BM25 reached Recall@5/10/50 `0.4571/0.5490/0.7182`, MRR@10 `0.4567`, and nDCG@10 `0.4221`. It does **not** establish verdict quality or cross-encoder gains. Exact hashes and boundaries are in [`docs/verified-runs/climate-fever-public-bm25-test-20260825.json`](docs/verified-runs/climate-fever-public-bm25-test-20260825.json).

That historical frozen test has already been consumed. A post-hoc strict audit
found one cross-partition near-document pair at 0.90 Jaccard; both annotations
are `NOT_ENOUGH_INFO`, and the decisive-evidence audit remains clean. The result
is retained as a historical lexical baseline, but every new candidate test is
blocked by `configs/public_evaluation_policy.json`. Model selection is
validation-only; this project cycle no longer has an unused public test for a
new independent claim.

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

### Verified dense encoder size gate

Job `29458425` compared `Qwen3-Embedding-0.6B` with `Qwen3-Embedding-4B` at the same 1,024-dimensional output on an evidence-preserving 5,000-document sample. All 27 gold-evidence rows for the same eight claims were forced into the sample; this is a resource screen, not full-corpus retrieval evidence. The 4B candidate did not pass: Recall@5 changed from `0.950` to `0.925` (paired 95% interval `-0.075–0.000`), while MRR@10 and Evidence F1 tied. Document encoding fell from `50.95` to `7.21 docs/s`, and peak Torch GPU allocation rose from `3.17 GB` to `17.42 GB`. The retained full-corpus/latency-profile 0.6B index is therefore kept, and a full 4B rebuild was deliberately not submitted. This sampled gate supports that resource decision; it does not prove that 4B would be worse on the complete corpus. The compact record is in [`docs/verified-runs/qwen3-embedding-4b-pilot-20260820.json`](docs/verified-runs/qwen3-embedding-4b-pilot-20260820.json).

### Verified full-corpus embedding-adapter promotion gate

The retained 0.6B encoder was adapted instead of replaced. Job `29465819`
evaluated the 20-step hard-negative InfoNCE/LoRA checkpoint on all 154 untouched
official-dev claims, all 463 required evidence rows and the complete 1,208,827-
document corpus at 1,024 dimensions. It used commit `c815070`, one L40S, eight
CPUs and a 64 GB request, completed in `49 min 37 s` with exit `0:0`, reached
Slurm MaxRSS `22,890,736 K`, and consumed `0.827 L40S-hours` of wall allocation.

| Metric | Base 0.6B | Adapted 0.6B | Mean delta | Paired 95% interval |
|---|---:|---:|---:|---:|
| Recall@5 | `0.2793` | **`0.2970`** | `+0.0176` | `0.0014–0.0350` |
| MRR@10 | `0.3633` | **`0.3869`** | `+0.0236` | `0.0060–0.0432` |
| nDCG@10 | `0.2994` | **`0.3203`** | `+0.0210` | `0.0090–0.0341` |
| Evidence F1 | `0.07253` | **`0.07544`** | `+0.00291` | `0.00120–0.00482` |

The adapted corpus encoded in `2,888.48 s` (`418.50 docs/s`), built the in-memory
FlatIP reference in `1.86 s`, and recorded peak Torch GPU allocation
`25,243,138,560 bytes`. The first attempt had completed encoding/search but then
failed while persisting a rebuildable 4.95 GB adapted index under project quota.
The replacement saved `0` index bytes and retained only the small manifest,
metrics, report and restricted prediction artifact on Spartan. This operational
fix did not change the evaluation or its pre-registered gate.

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
| Pure Qwen3-Reranker-4B | `0.3054` | `0.3763` | `0.2738` | `0.1997` |
| RRF + Qwen3-4B weighted rank fusion (1:1) | **`0.3153`** | **`0.3961`** | **`0.2849`** | **`0.2131`** |
| Legacy LambdaMART (invalidated training set) | `0.0029` | `0.0065` | `0.0030` | `0.0027` |
| Legacy LambdaMART + deterministic reranker (invalidated upstream) | `0.0127` | `0.0359` | `0.0149` | `0.0111` |

Against BM25, RRF improved Recall@5 by `0.0988` (paired-bootstrap 95% interval `0.0543–0.1452`) and Evidence F1 by `0.0616` (`0.0360–0.0893`). The first pure Qwen replacement run (`29448904`) regressed, exposing an architectural error: it discarded a strong first-stage order. Job `29452723` preserved that order with 0.6B weighted-rank fusion; its selected 4:1 profile improved Recall@5, MRR and nDCG, but not Evidence F1 with a stable interval.

The aggregate five-stage metrics and exact input/artifact hashes are published in [`docs/verified-runs/five-stage-fixed-dev-20260819.json`](docs/verified-runs/five-stage-fixed-dev-20260819.json). A later audit found that the legacy LTR training builder injected unretrieved gold evidence with zero retrieval features, so the two LTR rows above are retained as failure-forensics evidence, not model-quality evidence. The corrected candidate-supported job `29484697` removed that defect but remained a negative gate: 1,169 train groups/26,626 rows reached `0.9529` train pairwise accuracy, while fixed-dev Recall@5/F1 collapsed to `0.0075/0.0059` versus RRF `0.2709/0.1785`. The compact record is in [`docs/verified-runs/candidate-supported-ltr-gate-20260822.json`](docs/verified-runs/candidate-supported-ltr-gate-20260822.json). Commit `b47e437` then recorded the RRF prior, matched the 100-candidate training and serving widths and predeclared a 4:1 rank-preserving fusion. CPU job `29504398` completed 1,169 groups/120,146 rows and raised fixed-dev MRR@10/nDCG@10 from RRF `0.3446/0.2495` to `0.3648/0.2605`; 5,000-sample paired intervals were `[+0.0032,+0.0390]` and `[+0.0012,+0.0228]`. Recall@5/F1 rose to `0.2801/0.1824`, but their intervals versus RRF crossed zero. CPU feature scoring took P95 `7.80 ms/query`. This is retained as a low-latency rank-position profile, not the main quality profile; see the [compact record](docs/verified-runs/rrf-prior-ltr-fusion-gate-20260822.json). Restricted predictions and candidate lists remain on Spartan.

The model-size gate then ran Qwen3-Reranker-4B in BF16 on the identical 154-claim/7,700-pair split. Full job `29453918` completed in `12 min 48 s` on one A100 `1g.20gb` MIG slice (`8 CPU`, `32 GB` request; batch MaxRSS `20,372,008 K`). It recorded P50/P95 `4.20/4.82 s` per query. Pure 4B aggregate metrics rose but paired intervals versus RRF crossed zero. The selected 1:1 RRF/4B rank fusion reached Recall@5 `0.3153`, MRR@10 `0.3961`, nDCG@10 `0.2849`, and Evidence F1 `0.2131`. Comparison job `29455049` measured deltas versus RRF of `+0.0444` Recall@5 (95% interval `0.0163–0.0733`, `p=0.0024`), `+0.0515` MRR (`0.0149–0.0883`), `+0.0354` nDCG (`0.0123–0.0576`), and `+0.0347` Evidence F1 (`0.0165–0.0535`). All values come from 5,000 paired bootstrap samples. Because the fusion weights and model size were selected on this same fixed dev split, these are dev-set model-selection results, not an independent test claim.

The optional 8B gate was also executed rather than left as a configuration claim. The first attempt (`29456746`) failed before inference because the shared project filesystem lacked room for the weight shards; four incomplete files totalling about 3.2 GB were removed, and commit `53a3782` moved the one-off cache to node-local ephemeral storage. Replacement pilot `29456898` completed in `2 min 19 s` on the same A100 `1g.20gb` MIG shape (batch MaxRSS `29,380,852 K`). On the same eight claims/400 pairs, the best 8B fusion tied 4B on Evidence F1 (`0.3016`) and Recall@5 (`0.4688`), was slightly lower on MRR@10 (`0.5042` vs `0.5104`), and raised P95 latency from `5.13 s` to `8.25 s` (`+60.8%`). The full 8B run was therefore deliberately not submitted. This is a resource-selection gate, not a full-dev 8B quality result; the derived record is in [`docs/verified-runs/qwen3-reranker-8b-pilot-20260820.json`](docs/verified-runs/qwen3-reranker-8b-pilot-20260820.json).

Two RouteLLM-inspired cost-aware gates then tested whether the 4B path could be
called selectively. Both used deterministic five-fold hash cross-fitting, so a
claim's labels never trained its own route. The candidate-list-agreement router
called 4B on `43.51%` of queries and reduced the analytical mean latency estimate
to `1.865 s/query`; Recall@5/F1 were `0.2878/0.1909`, significantly above RRF,
but this preserved only `38.05%/36.00%` of the always-4B gain and was
significantly below the strong path. Adding inference-safe hashed claim-text
features avoided `85.71%` of calls but collapsed to RRF-level quality. Both fail
the predeclared 80% gain-preservation target and are not selected. The compact
record is in
[`docs/verified-runs/rerank-router-gates-20260821.json`](docs/verified-runs/rerank-router-gates-20260821.json).

HNSW+RRF remains the latency-oriented default; balanced 4B fusion is the measured offline quality profile. The RRF-prior LambdaMART fusion is an optional CPU rank-position profile because it improved MRR/nDCG but not Recall/F1 conclusively and remains below the 4B fusion's absolute quality. IVF-PQ and both cost routers remain rejected. The legacy LTR rows are invalidated; candidate-supported job `29484697` is a separate valid negative gate, while `29504398` records the bounded RRF-prior recovery. Because `final_k=5`, recorded Recall@10/50 equals Recall@5 and is not presented as a wider-cutoff result. Claim classification was not configured, so zero claim accuracy/H-mean values are not classifier findings.

The [paper-to-hiring map](docs/PAPER_TO_HIRING.md) connects Qwen3 model sizing
and adaptation, rank-preserving reranking, cost-aware routing and candidate-
supported learning-to-rank to exact code, jobs, evidence and stop conditions.

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
