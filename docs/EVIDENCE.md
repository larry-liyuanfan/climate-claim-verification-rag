# Evidence and claim boundaries

## Current repository

| Evidence | Status | Verification |
|---|---|---|
| Package and five required CLI commands | `verified` | editable install and CLI integration tests |
| BM25, hash dense exact search, RRF, hard negatives | `verified` | deterministic unit tests |
| Claim-grouped Qwen3-Embedding-0.6B InfoNCE/LoRA adaptation | `verified-offline-dev-promotion` | data job `29460405`, 20-step training `29462754`, injection preflight `29463845`, sampled screen `29463846`, full-corpus gate `29465819`; all 154 official-dev claims/1,208,827 documents; Recall@5 `0.2793→0.2970`, MRR `0.3633→0.3869`, nDCG `0.2994→0.3203`, F1 `0.07253→0.07544`; all 5,000-sample paired intervals positive; offline dev, not independent test/online A/B |
| Pairwise LTR fallback and LightGBM persistence | `verified` | 24-test clean environment; commit `636e915` fixes persisted LightGBM feature metadata |
| Qwen3 encoder and FAISS FlatIP adapter | `verified-build` | full 1,208,827-vector Spartan build plus fixed-dev effectiveness run completed |
| FAISS HNSW/IVF-PQ | `verified` | jobs `29418470`/`29418595`; fixed 154-query FlatIP-grounded quality-speed comparison |
| Retrieval/end-to-end metrics | `verified` | synthetic fixtures with exact expected values |
| Bootstrap and calibration utilities | `verified` | seeded unit tests |
| Five-stage orchestration | `verified-smoke` | synthetic fixture with deterministic reranker |
| FastAPI retrieval | `verified-smoke` | API test confirms explicit no-classifier state |
| Full 1,208,827-document BM25 index | `verified` | Spartan job `29360715`; commit `a7b110e`; 40.333 s total, 126,334,728-byte artifact, Slurm MaxRSS 2,630,496 K |
| Full Qwen3 embeddings + FAISS FlatIP reference index | `verified-build` | Spartan job `29382416`; commit `2cd75e3`; 1,208,827 × 1,024-d vectors; 1,696.767 s total; 5,133,839,551-byte dense artifact; MaxRSS 22,583,288 K |
| Qwen3-Embedding-4B sampled size gate | `verified-negative-resource-gate` | job `29458425`, commit `ced5e32`: evidence-preserving 5,000-document/eight-claim screen at 1,024 dimensions; Recall@5 `0.925` vs 0.6B `0.950`, F1/MRR tied; `7.21` vs `50.95 docs/s`; peak Torch GPU bytes `17.42 GB` vs `3.17 GB`; no full rebuild |
| HNSW quality-speed result | `verified` | Recall@5 vs Flat `0.9961`; batch QPS `3,060.64`; P50/P95 `12.88/15.41 ms`; 5,280,336,294-byte index |
| IVF-PQ quality-speed result | `verified-negative` | 66,211,820-byte index and batch QPS `8,436.04`, but Recall@5 vs Flat only `0.3688`; rejected by quality gate |
| Fixed-dev RRF retrieval result | `verified` | job `29435589`, 154 claims, `final_k=5`: Recall@5 `0.2709` vs BM25 `0.1721`; Evidence F1 `0.1785` vs `0.1168`; 5,000-sample paired intervals exclude zero |
| Qwen3-Reranker-0.6B pure-replacement result | `verified-negative` | job `29448904`, commit `01a571f`: 7,700 RRF Top-50 pairs; Recall@5/F1 `0.2438/0.1573` vs RRF `0.2709/0.1785`; paired intervals vs RRF cross zero; P50/P95 `4.88/5.88 s` per query |
| RRF + Qwen3-0.6B weighted-rank fusion | `verified` | job `29452723`, commit `18faf6a`: selected 4:1 RRF/Qwen rank profile; Recall@5 `0.2890` vs `0.2709` (delta interval `0.0005–0.0389`); MRR/nDCG intervals positive; F1 delta interval crosses zero; P50/P95 `5.57/6.52 s` |
| Qwen3-Reranker-4B pure replacement | `verified-inconclusive` | job `29453918`, commit `c04e39c`: 154 claims/7,700 pairs; Recall@5/F1 `0.3054/0.1997`, but paired intervals versus RRF cross zero; P50/P95 `4.20/4.82 s` |
| RRF + Qwen3-4B balanced weighted-rank fusion | `verified-dev-selection` | jobs `29453918`/`29455049`: Recall@5/MRR/nDCG/F1 `0.3153/0.3961/0.2849/0.2131`; all four 5,000-sample paired intervals versus RRF above zero; 4B/weights selected on the same dev split, not an independent test |
| Qwen3-Reranker-8B pilot Pareto gate | `verified-negative-pareto-gate` | job `29456898`, commit `53a3782`: same 8 claims/400 pairs as 4B pilot; tied F1/Recall@5 `0.3016/0.4688`, MRR `0.5042` vs 4B `0.5104`, P95 `8.25 s` vs `5.13 s`; full 8B intentionally not run |
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

The dense encoder size gate is recorded in `docs/verified-runs/qwen3-embedding-4b-pilot-20260820.json`. Job `29458425` completed in `14 min 16 s` with exit `0:0` and Slurm MaxRSS `27,017,012 K`. It retains all gold evidence in the sampled corpus but evaluates only eight claims, so it supports the decision not to spend resources on a full 4B rebuild; it does not prove full-corpus 4B effectiveness.

The retained-0.6B adaptation chain is recorded in `docs/verified-runs/qwen3-embedding-lora-sampled-gate-20260821.json` and `docs/verified-runs/qwen3-embedding-lora-full-gate-20260821.json`. Data job `29460405` resolved 13,354/13,354 evidence IDs and produced a claim-grouped split; training job `29462754` completed 20 LoRA/InfoNCE steps; preflight `29463845` verified 5,046,272 injected adapter parameters. Sampled gate `29463846` used 126 held-out claims, 5,000 of 1,208,827 documents and all 368 labelled positives; Recall@5, MRR and nDCG intervals were positive, while Evidence F1 crossed zero, so no sampled-only promotion was made. The first full-corpus job `29464119` finished encoding/search but failed while persisting a rebuildable 4.95 GB adapted FlatIP index and produced no usable metrics. Commit `c815070` made index persistence opt-in. Replacement `29465819` completed at commit `c81507084f3b5355c29b0ed0351dc85672a0c619` in `49 min 37 s` with exit `0:0`, Slurm MaxRSS `22,890,736 K`, and `0.827 L40S-hours`. It evaluated all 154 official-dev claims, 463 required evidence rows and 1,208,827 documents with 5,000 paired samples. Recall@5 changed `0.2793→0.2970` (95% interval `0.0014–0.0350`), MRR `0.3633→0.3869` (`0.0060–0.0432`), nDCG `0.2994→0.3203` (`0.0090–0.0341`), and F1 `0.07253→0.07544` (`0.00120–0.00482`). The adapter passes the offline official-dev promotion gate. The successful run saved zero adapted-index bytes; restricted predictions, checkpoint, corpus and large base artifacts remain on Spartan. This is not independent test generalisation or an online A/B result.

The fixed-dev run manifest records job `29435589`, commit `636e9159a14a59248ce8c4e93c396370c4af508e`, dev-claim hash `ea9976e8...`, config hash `8ae39fb2...`, `final_k=5`, 5,000 bootstrap samples, and `deterministic-feature-fallback`. It is a retrieval-only experiment: the classifier is unconfigured, and Recall@10/50 are not separately interpretable because only five final documents were emitted.

The local Qwen3 reranker run manifest records job `29448904`, commit `01a571fbee549e4ecff7fec7eaf7fa6c076bd623`, the same dev-claim hash, config hash `4d2fbab...`, RRF as the reranker base, 154 queries, 7,700 pairs, and `final_k=5`. The paired RRF-vs-Qwen comparison is job `29449093` at commit `2de8293`. Restricted predictions and model cache remain on Spartan.

The corrected fusion run manifest records job `29452723`, commit `18faf6aa59b593ae7e362c6decdc031d6ef96575`, 154 queries and 7,700 pairs. Comparison job `29453474` bootstrapped every Qwen/fusion prediction file against the same RRF predictions. The selected `base4` profile is a rank-level fusion, not a learned score calibration: its 4:1 weights were evaluated alongside balanced and 2:1 profiles on the fixed dev split, so they are not an independent test-set hyperparameter claim.

The 4B run manifest records job `29453918`, commit `c04e39c9fb4983f010623ba14d0c8cf9ac371edd`, dev-claim hash `ea9976e8...`, config hash `64aa8c1d...`, BF16 inference, 154 queries and 7,700 pairs. Comparison job `29455049` evaluated pure, balanced, 2:1 and 4:1 outputs against the same RRF predictions with 5,000 paired samples. Balanced fusion was best on this dev split and therefore carries a `verified-dev-selection` boundary rather than an independent test label. Restricted predictions and the 4B model cache remain on Spartan.

The 8B resource gate is recorded in `docs/verified-runs/qwen3-reranker-8b-pilot-20260820.json`. Job `29456898` used node-local model caching after a storage-only failed attempt, finished with exit `0:0`, and did not beat the 4B pilot on quality/latency. No full-dev 8B result exists or is claimed.

