# Evidence and claim boundaries

## 2026-09-27 correction: ranking depth, integrity and timing

This notice supersedes historical field labels, not the immutable run artifacts.
Public-v2 and restricted five-stage downstream saved Top-5 predictions while
calling their ranking fields MRR@10/nDCG@10 (and sometimes Recall@10/50).
Their Recall@5 and F1@5 remain usable for valid base/ranking runs; MRR can be
interpreted as MRR@5, but nDCG cannot be renamed because IDCG still used ten.
These old fields/intervals are not complete Top-10 evidence. Exact source
commits and corrected commands are recorded in [D9](DECISIONS.md#d9--audit-ranking-depth-and-checkpoint-values-before-model-selection).
The separate restricted 20-step LoRA full gate retained Top-50, so its MRR@10,
nDCG@10 and F1@50 remain correctly scoped. Never compare its F1@50 directly
with downstream F1@5. Fixture classifier scores are not live verdict quality.

- Real public checkpoint integrity passed in CPU job `31364439`: 392 tensors
  restored exactly and fixed query/document embeddings changed repeatably;
  [compact record](verified-runs/adapter-integrity-20260927.json). This is not
  an adapter quality improvement or proof of every historical pilot's cause.
- Clean code reproduction at `b797f66`: 101 passed, one Torch test skipped
  locally; source mypy/Ruff/CI passed. See [runtime record](verified-runs/search-tradeoffs-reproduction-20260927.json).
- New same-query, full-ranking P0 profiling is job `31364586`, still queued
  when this correction was written. No new quality/latency table is claimed.
  Old sums of stage P95 values are cost proxies, not measured request P95.


## 2026-09-03 representation-evaluation closeout

| Record | Status | Boundary |
|---|---|---|
| Public frozen-test policy | `verified-consumed-test` | CLIMATE-FEVER test was consumed on 2026-08-25 by `bm25-lexical-baseline-v1`; this closeout produced no new candidate test score and the validation gate was not run |
| Historical split post-hoc audit | `strict-warning / decisive-pass` | exact source SHA `8a4b9032...`; zero claim/shared-ID/exact-text leaks, one 0.913-Jaccard train/test document variant; both annotations NEI, so decisive-evidence variant count is zero |
| New v2 grouped split | `verified-public-data-preparation` | 1,535 claims/5,240 unique documents, 1,075/230/230 split; shared IDs, normalised/near claims and normalised/near evidence variants all have zero cross-partition leaks; no test model was run |
| Base/adapted comparison contract | `verified-code` | exact query/corpus/candidate-universe/width/cutoff/data hashes plus >=5,000 paired bootstrap; fixture tests are not quality evidence |
| LTR Top-K reachability correction | `verified-code; quality-pending` | feature rows now equal the serving-width RRF pool; unreachable positives are counted and excluded. Historical job `29504398` is not relabelled as a result of this code change |
| Query taxonomy | `verified-diagnostic-code` | entity, numeric/year, geographic, lexical mismatch, semantic inference, multi-evidence and unanswerable; deterministic heuristic slices, not human labels |
| Historical search Pareto record | `retired-component-cost-proxy` | Original values remain for audit, but sums of component P95 values are not request P95; the CLI now refuses this config for new Pareto decisions. The pending same-query public comparison must supply actual E2E timings |
| Iris isolated CPU preflight | `verified` | job `29926197`, exact SHA `09d2524`, 15 targeted tests passed in 7 s, exit `0:0`, batch MaxRSS `53,244 KiB`; public code/fixtures only, no GPU or frozen-test evaluation |

Compact closeout: `docs/verified-runs/representation-evaluation-closeout-20260903.json`.

## Current repository

| Evidence | Status | Verification |
|---|---|---|
| Package and CLI workflows | `verified-software` | Clean clone/new environment at `b797f66`: 101 passed, 1 Torch-dependent skip, Ruff and strict mypy (31 source modules); historical test counts below identify old runs, not the current total |
| BM25, hash dense exact search, RRF, hard negatives | `verified` | deterministic unit tests |
| Claim-grouped Qwen3-Embedding-0.6B InfoNCE/LoRA adaptation | `verified-offline-dev-promotion` | data job `29460405`, 20-step training `29462754`, injection preflight `29463845`, sampled screen `29463846`, full-corpus gate `29465819`; all 154 official-dev claims/1,208,827 documents; Recall@5 `0.2793→0.2970`, MRR `0.3633→0.3869`, nDCG@10 `0.2994→0.3203`, F1@50 `0.07253→0.07544`; all 5,000-sample paired intervals positive; offline dev, not independent test/online A/B |
| Pairwise LTR fallback and LightGBM persistence | `verified-code` | 48-test project environment; commit `636e915` fixes persisted LightGBM feature metadata; effectiveness requires a valid candidate-supported run |
| Qwen3 encoder and FAISS FlatIP adapter | `verified-build` | full 1,208,827-vector Spartan build plus fixed-dev effectiveness run completed |
| FAISS HNSW/IVF-PQ | `verified` | jobs `29418470`/`29418595`; fixed 154-query FlatIP-grounded quality-speed comparison |
| Retrieval/end-to-end metrics | `verified` | synthetic fixtures with exact expected values |
| Bootstrap and calibration utilities | `verified` | seeded unit tests |
| Five-stage orchestration | `verified-smoke` | synthetic fixture with deterministic reranker |
| Public CLIMATE-FEVER adapter and fixed split | `verified-public-data-contract` | upstream SHA `8a4b9032...e0b`; 1,535 claims/7,675 annotations/5,240 unique evidence; seed `20260825`; train/validation/test 1,075/230/230; zero shared-evidence and normalised-claim cross-split leakage |
| Public frozen-test BM25 baseline | `verified-public-external-retrieval-baseline` | 129 test claims with decisive evidence: Recall@5/10/50 `0.4571/0.5490/0.7182`, MRR@10 `0.4567`, nDCG@10 `0.4221`; compact record `docs/verified-runs/climate-fever-public-bm25-test-20260825.json` | Retrieval only; no cross-encoder or verdict model, so do not report claim accuracy or end-to-end RAG quality |
| Public retrieval v2 adapter matrix | `historical-inconclusive-adapter/no-promotion` | Array `30005221` observed six 64-query ties; diagnostic `30007124` failed integrity, so no valid full adapter result exists. Base-only `30007546` measured 126 decisive validation claims: RRF/4B Recall@5/F1@5 `0.6275/0.3969`. Historical Top-10 fields are censored. The original compact JSON is retained unchanged; fixed-probe restoration passed on 2026-09-27 without a quality rerun. Frozen test and SciFact remain closed |
| Grounded FastAPI service | `verified-code-and-fixture` | `/api/search`, `/api/verify`, trace, metrics and health tests; invalid citation IDs/quotes, provider failure and missing verifier all abstain | Model Studio credentials are not configured in the local environment; live Qwen3.7-Plus external-test verdict metrics are not yet verified |
| Full 1,208,827-document BM25 index | `verified` | Spartan job `29360715`; commit `a7b110e`; 40.333 s total, 126,334,728-byte artifact, Slurm MaxRSS 2,630,496 K |
| Full Qwen3 embeddings + FAISS FlatIP reference index | `verified-build` | Spartan job `29382416`; commit `2cd75e3`; 1,208,827 × 1,024-d vectors; 1,696.767 s total; 5,133,839,551-byte dense artifact; MaxRSS 22,583,288 K |
| Qwen3-Embedding-4B sampled size gate | `verified-negative-resource-gate` | job `29458425`, commit `ced5e32`: evidence-preserving 5,000-document/eight-claim screen at 1,024 dimensions; Recall@5 `0.925` vs 0.6B `0.950`, F1/MRR tied; `7.21` vs `50.95 docs/s`; peak Torch GPU bytes `17.42 GB` vs `3.17 GB`; no full rebuild |
| HNSW quality-speed result | `verified` | Recall@5 vs Flat `0.9961`; batch QPS `3,060.64`; P50/P95 `12.88/15.41 ms`; 5,280,336,294-byte index |
| IVF-PQ quality-speed result | `verified-negative` | 66,211,820-byte index and batch QPS `8,436.04`, but Recall@5 vs Flat only `0.3688`; rejected by quality gate |
| Fixed-dev RRF retrieval result | `verified` | job `29435589`, 154 claims, `final_k=5`: Recall@5 `0.2709` vs BM25 `0.1721`; Evidence F1 `0.1785` vs `0.1168`; 5,000-sample paired intervals exclude zero; compact hashes/metrics in `docs/verified-runs/five-stage-fixed-dev-20260819.json` |
| Qwen3-Reranker-0.6B pure-replacement result | `verified-negative` | job `29448904`, commit `01a571f`: 7,700 RRF Top-50 pairs; Recall@5/F1 `0.2438/0.1573` vs RRF `0.2709/0.1785`; paired intervals vs RRF cross zero; P50/P95 `4.88/5.88 s` per query |
| RRF + Qwen3-0.6B weighted-rank fusion | `verified-dev-selection/cutoff-corrected` | job `29452723`, commit `18faf6a`: selected 4:1 rank fusion; Recall@5 `0.2890` vs `0.2709` (delta interval `0.0005–0.0389`); F1@5 interval crosses zero; old Top-10 fields are not full-depth evidence; measured stage P50/P95 `5.57/6.52 s` |
| Qwen3-Reranker-4B pure replacement | `verified-inconclusive` | job `29453918`, commit `c04e39c`: 154 claims/7,700 pairs; Recall@5/F1 `0.3054/0.1997`, but paired intervals versus RRF cross zero; P50/P95 `4.20/4.82 s` |
| RRF + Qwen3-4B balanced weighted-rank fusion | `verified-dev-selection/cutoff-corrected` | jobs `29453918`/`29455049`: Recall@5/F1@5 `0.3153/0.2131`; both 5,000-sample paired intervals versus RRF above zero. Legacy MRR@10/nDCG@10 used Top-5 lists; not complete Top-10 results. Weights and size selected on the same dev, not independent test |
| Qwen3-Reranker-8B pilot Pareto gate | `verified-negative-pareto-gate` | job `29456898`, commit `53a3782`: same 8 claims/400 pairs as 4B pilot; tied F1/Recall@5 `0.3016/0.4688`, MRR `0.5042` vs 4B `0.5104`, P95 `8.25 s` vs `5.13 s`; full 8B intentionally not run |
| Candidate-agreement cost-aware 4B router | `verified-partial-dev-gate/not-selected` | job `29479185`, commit `95f72a5`: five-fold cross-fit, 43.51% strong calls, estimated mean `1.865 s/query`, Recall@5/F1 `0.2878/0.1909`; paired improvements vs RRF but only 38.05%/36.00% of always-4B gain preserved, below the 80% gate |
| Text-aware cost router | `verified-negative-dev-gate` | job `29479261`, commit `2ff8cf4`: five-fold cross-fit, 14.29% strong calls, Recall@5/F1 `0.2718/0.1788`; paired intervals vs RRF cross zero; rejected |
| Legacy LightGBM LambdaMART fixed-dev result | `invalidated-training-set` | Recall@5 `0.0029`, but the training builder injected unretrieved gold evidence with zero retrieval features; retain only as failure forensics, not a model-quality result |
| Legacy LTR + deterministic reranker result | `invalidated-upstream` | Recall@5 `0.0127`; downstream of the invalid legacy LTR candidates and explicitly not Qwen3 |
| Candidate-supported LambdaMART correction | `verified-negative-dev-gate` | Job `29484697`, commit `023ed9b`: 1,169 train groups/26,626 rows and 3,246 reachable positives; training pairwise accuracy `0.9529`, but fixed-dev Recall@5/F1 collapsed to `0.0075/0.0059` versus RRF `0.2709/0.1785`; both paired intervals versus BM25 were below zero. This is a valid negative result, not the earlier invalidated training set |
| RRF-prior, serving-width LTR correction | `verified-top5-rank-position-dev/cutoff-corrected` | Job `29504398`, commit `b47e437`: 1,169 groups/120,146 rows; RRF→4:1 RRF/LambdaMART MRR@5 `0.3446→0.3648`, paired interval `[+0.0032,+0.0390]`. Legacy field named MRR@10 used only five ranks; censored nDCG@10 is not a full-depth result. Recall@5/F1@5 `0.2709/0.1785→0.2801/0.1824` but intervals cross zero. CPU scoring P95 `7.80 ms` is one component, not E2E. Original compact record remains in `verified-runs/rrf-prior-ltr-fusion-gate-20260822.json` |

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

The dense encoder size gate is recorded in `docs/verified-runs/qwen3-embedding-4b-pilot-20260820.json`. Job `29458425` completed in `14 min 16 s` with exit `0:0` and Slurm MaxRSS `27,017,012 K`. It retains all gold evidence in the sampled corpus but evaluates only eight claims, so it supports the decision not to spend resources on a full 4B rebuild; it does not prove full-corpus 4B effectiveness.

The retained-0.6B adaptation chain is recorded in `docs/verified-runs/qwen3-embedding-lora-sampled-gate-20260821.json` and `docs/verified-runs/qwen3-embedding-lora-full-gate-20260821.json`. Data job `29460405` resolved 13,354/13,354 evidence IDs and produced a claim-grouped split; training job `29462754` completed 20 LoRA/InfoNCE steps; preflight `29463845` verified 5,046,272 injected adapter parameters. Sampled gate `29463846` used 126 held-out claims, 5,000 of 1,208,827 documents and all 368 labelled positives; Recall@5, MRR and nDCG intervals were positive, while Evidence F1 crossed zero, so no sampled-only promotion was made. The first full-corpus job `29464119` finished encoding/search but failed while persisting a rebuildable 4.95 GB adapted FlatIP index and produced no usable metrics. Commit `c815070` made index persistence opt-in. Replacement `29465819` completed at commit `c81507084f3b5355c29b0ed0351dc85672a0c619` in `49 min 37 s` with exit `0:0`, Slurm MaxRSS `22,890,736 K`, and `0.827 L40S-hours`. It evaluated all 154 official-dev claims, 463 required evidence rows and 1,208,827 documents with 5,000 paired samples. Recall@5 changed `0.2793→0.2970` (95% interval `0.0014–0.0350`), MRR `0.3633→0.3869` (`0.0060–0.0432`), nDCG `0.2994→0.3203` (`0.0090–0.0341`), and F1@50 `0.07253→0.07544` (`0.00120–0.00482`). The adapter passes the offline official-dev promotion gate. The successful run saved zero adapted-index bytes; restricted predictions, checkpoint, corpus and large base artifacts remain on Spartan. This is not independent test generalisation or an online A/B result.

The fixed-dev run manifest records job `29435589`, commit `636e9159a14a59248ce8c4e93c396370c4af508e`, dev-claim hash `ea9976e8...`, config hash `8ae39fb2...`, `final_k=5`, 5,000 bootstrap samples, and `deterministic-feature-fallback`. It is a retrieval-only experiment: the classifier is unconfigured, and Recall@10/50 are not separately interpretable because only five final documents were emitted.

The local Qwen3 reranker run manifest records job `29448904`, commit `01a571fbee549e4ecff7fec7eaf7fa6c076bd623`, the same dev-claim hash, config hash `4d2fbab...`, RRF as the reranker base, 154 queries, 7,700 pairs, and `final_k=5`. The paired RRF-vs-Qwen comparison is job `29449093` at commit `2de8293`. Restricted predictions and model cache remain on Spartan.

The corrected fusion run manifest records job `29452723`, commit `18faf6aa59b593ae7e362c6decdc031d6ef96575`, 154 queries and 7,700 pairs. Comparison job `29453474` bootstrapped every Qwen/fusion prediction file against the same RRF predictions. The selected `base4` profile is a rank-level fusion, not a learned score calibration: its 4:1 weights were evaluated alongside balanced and 2:1 profiles on the fixed dev split, so they are not an independent test-set hyperparameter claim.

The 4B run manifest records job `29453918`, commit `c04e39c9fb4983f010623ba14d0c8cf9ac371edd`, dev-claim hash `ea9976e8...`, config hash `64aa8c1d...`, BF16 inference, 154 queries and 7,700 pairs. Comparison job `29455049` evaluated pure, balanced, 2:1 and 4:1 outputs against the same RRF predictions with 5,000 paired samples. Balanced fusion was best on this dev split and therefore carries a `verified-dev-selection` boundary rather than an independent test label. Restricted predictions and the 4B model cache remain on Spartan.

The 8B resource gate is recorded in `docs/verified-runs/qwen3-reranker-8b-pilot-20260820.json`. Job `29456898` used node-local model caching after a storage-only failed attempt, finished with exit `0:0`, and did not beat the 4B pilot on quality/latency. No full-dev 8B result exists or is claimed.

