# Paper-to-hiring map

This map separates implemented ideas, measured results and future candidates.
Paper citations are not counted as project features unless a code path and an
artifact or test are named.

## 1. Qwen3: choose the operating point, then adapt it

The Qwen3 Embedding technical report provides 0.6B, 4B and 8B embedding and
reranking families for different efficiency/effectiveness regimes. The project
does not assume that parameter count transfers monotonically to climate claims.

- Paper: Zhang et al., *Qwen3 Embedding: Advancing Text Embedding and Reranking
  Through Foundation Models*, <https://arxiv.org/abs/2506.05176>.
- Size gate: job `29458425` compared 0.6B and 4B encoders at the same 1,024
  dimensions. Sampled Recall@5 was `0.950` versus `0.925`; 4B encoded about
  7.1 times slower and used about 5.5 times the peak Torch GPU memory.
- Task adaptation: claim-grouped mined hard negatives, false-negative controls
  and a 20-step InfoNCE/LoRA adapter improved full-corpus official-dev Recall@5
  `0.2793→0.2970`, MRR `0.3633→0.3869`, nDCG `0.2994→0.3203` and F1
  `0.07253→0.07544`, with all paired intervals positive.
- Hiring signal: model selection by measured Pareto frontier, then domain
  adaptation and full-corpus promotion—not a blind upgrade to a larger model.
- Boundary: offline official-dev selection, not independent test or online A/B.

## 2. Rank-preserving reranking is a multi-stage systems problem

Listwise reranking research such as RankZephyr highlights sensitivity to the
initial order and reranked candidate set. The first pure Qwen3 replacement in
this project discarded a strong RRF order and regressed; rank-level fusion fixed
that architectural mistake.

- Paper: Pradeep, Sharifymoghaddam and Lin, *RankZephyr: Effective and Robust
  Zero-Shot Listwise Reranking is a Breeze!*,
  <https://arxiv.org/abs/2312.02724>.
- Code: `src/climate_rag/reranker.py`, `src/climate_rag/fusion.py` and the
  fixed-dev evaluation scripts.
- Evidence: balanced RRF/Qwen3-Reranker-4B fusion reached Recall@5/F1
  `0.3153/0.2131` versus RRF `0.2709/0.1785`; four paired intervals were
  positive. The 8B pilot tied Recall/F1 and raised P95 by 60.8%, so it stopped.
- Boundary: the project uses Qwen3 pointwise relevance scores plus rank fusion,
  not RankZephyr itself, and the selected weights are same-dev model selection.

## 3. Cost-aware routing must preserve measured quality

RouteLLM frames routing as a quality/cost trade-off between weak and strong
models. The project translated that idea to the 0.6B/RRF versus 4B-reranker
decision and predeclared an 80% strong-path gain-preservation gate.

- Paper: Ong et al., *RouteLLM: Learning to Route LLMs with Preference Data*,
  <https://arxiv.org/abs/2406.18665>.
- Code: `scripts/evaluate_rerank_router.py` and
  `scripts/evaluate_text_rerank_router.py`.
- Evidence: candidate agreement called 4B for 43.51% of claims but preserved
  only 38.05%/36.00% of the Recall/F1 gain; the text router called 4B for
  14.29% and collapsed to RRF-level quality. Both were rejected.
- Hiring signal: inference-cost optimisation with cross-fit leakage controls and
  a quality constraint, including a negative result.
- Boundary: these are deterministic project-specific routers, not RouteLLM's
  preference-trained implementation or a general routing claim.

## 4. Learning-to-rank labels must be reachable by the serving candidates

LambdaMART can only learn a deployable fusion if training examples reflect the
candidate distribution seen at inference. The legacy builder inserted gold
evidence that neither retriever returned and assigned it zero BM25/dense
features, teaching an inverse retrieval signal. That run is invalidated rather
than reported as “LambdaMART failed”.

- Code correction: `src/climate_rag/training.py` and commit `78f95c0` retain
  only candidate-supported positives and skip unsupported claim groups. The
  first evaluation job `29479905` failed before model evaluation when a GPFS
  shared clone could not read one Git tree; that is an infrastructure failure,
  not a LambdaMART result.
- Evaluation: commit `023ed9b` copies Git objects into the node-local checkout
  and passes 47 local tests. Replacement job `29484697` is the sole active
  exact-SHA corrected gate; it remains queued and has no result at the time of
  this document.
- Hiring signal: training-serving consistency, label reachability audits and
  failure forensics in a learning-to-rank pipeline.

## 5. Sparse neural retrieval is a future candidate, not a hidden feature

SPLADE-v3 shows that learned sparse retrieval can improve over BM25 while
remaining compatible with inverted-index serving. It is a plausible next
candidate for fact-heavy claims with exact entity/number matching.

- Paper: Lassance et al., *SPLADE-v3: New baselines for SPLADE*,
  <https://arxiv.org/abs/2403.06789>.
- Current status: not implemented and not claimed.
- Promotion requirement: frozen train/dev split, index-size/build-time/QPS
  accounting, paired Recall/MRR/nDCG/F1 versus BM25+dense RRF, and a resource
  gate before it can enter the portfolio narrative.

## Interview summary

The technical story is:

1. build exact lexical and semantic recall over 1,208,827 passages;
2. choose ANN and model sizes with explicit quality, memory and latency gates;
3. adapt the efficient encoder using claim-grouped hard negatives;
4. preserve first-stage information while adding an expensive reranker;
5. test whether a router can save cost without losing the strong path;
6. audit training-serving consistency before trusting learned fusion;
7. publish positive, negative and invalidated results with different labels.
