# Search tradeoffs and representation integrity — 2026-09-27

Status: P0 same-candidate profiling and P1 representation repair completed.
GPU job `31364586` completed in 22 min 14 s, exit 0:0. All three profiles,
archive hashes, query/candidate identity and payload manifests passed validation.
Recommend LambdaMART as the **offline low-latency candidate**, with Top-100 4B
optional when latency is secondary; no production default was changed.
No new adapter-quality or independent-test improvement is claimed.
Base commit: `272ab93256202e517a8aebf36992fbdf93dc922e`.

## Findings and repair

The frozen public-v2 environment uses PEFT 0.15.2, whose loader does not
implement the supplied `key_mapping`. Missing checkpoint keys can yield a
warning while initialized LoRA parameters still exist. Counting parameters
did not prove restoration. Sources: [PEFT loader](https://github.com/huggingface/peft/blob/v0.15.2/src/peft/peft_model.py),
[restoration code](https://github.com/huggingface/peft/blob/v0.15.2/src/peft/utils/save_and_load.py).

`adapter_integrity.py` accepts exact checkpoint keys or only the known
`base_model.model.model.* -> base_model.model.*` conversion. Missing keys,
collisions, unknown keys, shapes, non-finite values and unequal restored values
fail before corpus encoding. Fixed label-free query/document probes compare
enabled / disabled / re-enabled adapters after the actual embedding pipeline,
including normalization and dimension truncation. A reproducible difference
over 1e-7 is an integrity check, not a quality improvement.

Public-v2 predictions also retained only five IDs while reporting Top-10
metrics. New ranking lists retain Top-100; ranking metrics use their own
cutoffs and Evidence precision/recall/F1 explicitly use Top-5. `evidence_f1`
remains the pre-registered gate alias alongside `evidence_f1@5`. Other callers
retain legacy behavior unless opting into `evidence_k`. ANN@5-vs-Flat explicitly
compares the first five IDs. Retrieval-only metrics do not report unconfigured
classifier zeroes as accuracy. No verdict model was added.

## Frozen integrity preflight (no quality retry)

- Reuse only the original diagnostic `s100-r16-n4-t005` checkpoint from the
  2026-09-03 run; do not train or reselect among six pilots.
- Reuse the SHA-pinned public-v2 environment with PEFT 0.15.2.
- Run a real tiny-model checkpoint/toggle integration test, then two fixed
  queries and two fixed documents with the original encoder revision.
- CPU-only Slurm preflight: 4 CPU / 16 GiB / 15 min / 20 GiB temporary storage.
  This initial bound reuses the prior GPU preflight cold-load allowance, not
  an assertion that CPU runtime was measured. Only four short probes run;
  no corpus inference. No competing GPU allocation is requested.
- Use a new run destination; old input archives remain read-only. Do not load
  qrels, CLIMATE-FEVER frozen test, SciFact labels or a full-corpus index.
- Keep failure/no-effect results. No adapter promotion or quality retry follows
  automatically; further validation must be separately frozen and justified.

Entry point:

```sh
python scripts/probe_embedding_adapter.py --adapter-dir CHECKPOINT \
  --output NEW_REPORT.json --device cpu
```

Slurm wrapper: `hpc/adapter_integrity_preflight.sbatch`. Test scheduling with
`sbatch --test-only` first. Model caches stay on node-local temporary storage.
Future submissions must export `CLIMATE_EXPECTED_USER` with the authorised
account outside Git. The wrapper requires this value and checks the connected
identity; no SSH username or credential belongs in the public script.

## Historical evidence boundaries

The 126-query public validation Recall@5 and Evidence F1@5 remain valid for the
base-only downstream run. Its old MRR@10/nDCG@10 and paired differences/CIs are
censored, not complete Top-10 metrics. Renaming nDCG@10 to nDCG@5 is incorrect
because its ideal denominator used ten. Historical timing retains its original
depth and stage scope; new numbers must not replace the old immutable artifacts.

Follow-up source audit also found the same five-rank storage in restricted
five-stage LTR/4B paths. Their Recall@5/F1@5 remain valid; MRR is interpretable
only as MRR@5 and nDCG@10 is censored. In contrast, the separate restricted
20-step LoRA gate saved Top-50 and its MRR@10/nDCG@10/F1@50 retain their scope.
See [D9 source map and legacy-entry-point repair](DECISIONS.md#d9--audit-ranking-depth-and-checkpoint-values-before-model-selection).
The historical component-sum Pareto config is retired, not used to populate
the P0 comparison. None of these follow-up changes alters job 31364586.

Restricted 1,208,827-document / 154-query / 20-step LoRA remains offline-dev
evidence, not independent test or online A/B. This public loader defect alone
does not prove or disprove the integrity of that separate run. Pilot ties alone
cannot identify their exact cause. The consumed frozen test remains closed;
SciFact remains unopened.

## Completed package gates

1. Integrity passed: see the verified result below. This satisfies the plan's
   representation-repair option; independent transfer is not required or run.
2. Same-query serial-request profiling of LambdaMART and both bounded 4B widths
   completed; stage/E2E timings, process/GPU memory and time proxies are below.
3. Full-ranking metrics and 5,000 paired bootstrap comparisons completed on
   validation only. Direct Top-20/Top-100 comparison was computed from the saved
   metric vectors, with no further inference or frozen-test access.
4. Clean-checkout reproduction passed at `b797f6606ced013b1a0adee915c996b593f180a3`:
   fresh GitHub clone + new virtual environment, 101 tests passed,
   one real-Torch test skipped locally (the prior version passed on Spartan). Ruff passed and
   strict mypy passed all 31 source modules after annotation/Protocol repairs.
   This is source type checking, not static analysis of every experimental script.
   Exact runtime and CI evidence: [reproduction record](verified-runs/search-tradeoffs-reproduction-20260927.json).
   After legacy-cutoff and direct-comparison additions, the clean clone was
   fast-forwarded to `97b2dc8`, non-editable installation refreshed, and all
   106 tests passed with the same one local Torch-dependent skip; Ruff and
   31-module mypy passed. Seven real PEFT tests also passed inside job 31364586.

## Verified CPU result

Job `31364439`, exact compute SHA `1636ab253295b4ec0a33fcfc4471a706989073e5`,
completed with exit 0 in 110 s, MaxRSS 5,830,904 KiB and TotalCPU 49.965 s.
All 392 checkpoint tensors match after dtype conversion, covering 10,092,544
LoRA parameters in this **public 100-step** checkpoint (not the separate
restricted 20-step checkpoint). Query/document maximum embedding deltas are
0.0630181 / 0.0423742; enabled-output repeat delta is zero. Seven real PEFT
integration tests passed. No quality evaluation, training or test access ran.
See [compact evidence](verified-runs/adapter-integrity-20260927.json), including
model, adapter, probe and archive hashes. This proves the loader repair, not
better retrieval or failure of all historical pilots.

## Frozen P0 profiling comparison

`hpc/search_tradeoffs.sbatch` runs three fresh processes, sequentially:

- Existing Top-100 LambdaMART (11 unchanged features, no retraining).
- Existing Qwen3-Reranker-4B on Top-20 RRF candidates, fused with all Top-100;
  unscored tail remains reachable.
- Existing Qwen3-Reranker-4B on all Top-100 candidates, same fusion weights.

All use the same archived BM25, HNSW, corpus order, encoder revision and 126
decisive validation queries. Each query is encoded at batch size one. One
fixed unlabelled warmup is excluded. CUDA synchronization brackets stages and
the separately measured complete request. No sum of stage percentiles is
reported as E2E. Model loading, HTTP and concurrent traffic are excluded.
Per-process peak RSS and Torch allocated/reserved memory are retained; no
API requests occur. Serial runtime per 1,000 requests is a compute-time proxy,
not billed cost or monetary savings.

The comparison script rejects different inputs, query IDs, candidate pools,
commit, runtime, device, or model revisions, verifies prediction/trace hashes
and exact query coverage. All processes use `PYTHONHASHSEED=0`. It retains full rankings and 5,000 paired
bootstrap comparisons versus LambdaMART. These are development/validation
tradeoffs, not independent generalization. No adapter is used in this P0 run.

Resource proposal: one MIG 20 GiB, 8 CPU, 32 GiB RAM, 40 GiB local scratch,
45 minutes. Prior Top-100 rerank P95 was 9.28 s/query; `126 * 9.28 * 1.2` is
about 23.4 minutes for the two rerank widths, with load/serialization/safety
allowance. This is a walltime bound derived from prior measurements, not a
new measured duration. Do not submit while another authorized GPU job runs.

The prior Trip allocation had COMPLETED before submission. Job `31364586`
uses exact compute SHA `b797f6606ced013b1a0adee915c996b593f180a3`, a new
`checkouts/search-tradeoffs-b797f66` checkout and
`runs/search-tradeoffs-20260927-b797f66` beneath the existing Climate public-v2
root. `bash -n` and `sbatch --test-only` passed. It is the sole submitted
profiling job; do not cancel/requeue to guess at queue priority. Logs are
`slurm-31364586.out`. Completion requires the three profile archives,
comparison archive, manifests/hashes and final Slurm accounting; submission
alone is not a result. No new polling automation was created.

## Verified quality and cost

Same 5,240 public evidence documents, 126 decisive queries out of the fixed
230-claim validation partition, same ordered RRF Top-100 pool for every query.
No adapter, training or verdict model is used in this comparison.

| Route | R@5 | R@10 | R@50 | MRR@10 | nDCG@10 | Evidence F1@5 | E2E P50 / P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LambdaMART Top-100 | .6054 | .7038 | .9197 | .6340 | .5926 | .3828 | 63.6 / 77.8 ms |
| RRF + 4B Top-20 | .5948 | .7226 | .9022 | .6217 | .5936 | .3829 | 1,621.2 / 1,909.1 ms |
| RRF + 4B Top-100 | .6275 | .7370 | .9332 | .6260 | .5997 | .3969 | 7,976.8 / 9,298.4 ms |

| Stage P50 / P95 (ms) | LTR | 4B Top-20 | 4B Top-100 |
|---|---:|---:|---:|
| Query encoding | 43.2 / 54.4 | 43.5 / 54.2 | 43.3 / 54.8 |
| BM25 recall | 3.0 / 4.2 | 3.0 / 4.4 | 3.0 / 4.5 |
| HNSW recall | 2.8 / 3.3 | 3.4 / 10.8 | 4.2 / 11.6 |
| RRF fusion | 1.6 / 1.9 | 1.7 / 2.0 | 1.7 / 2.0 |
| Final ranking / rank fusion | 13.8 / 17.5 | 1,566.8 / 1,856.0 | 7,920.9 / 9,248.1 |

Stage percentiles are not additive. E2E is timed independently; model loading,
HTTP, serialization, concurrency and allocation idle are excluded. Each query
is measured once after one common warmup; these are not repeated-load SLA tests.

| Resource / time proxy | LTR | 4B Top-20 | 4B Top-100 |
|---|---:|---:|---:|
| Peak Torch allocated (GiB) | 2.25 | 10.54 | 11.07 |
| Peak Torch reserved (GiB) | 2.27 | 12.50 | 14.13 |
| Peak process RSS (GiB) | 3.97 | 6.44 | 6.27 |
| Serial compute seconds / 1,000 requests (extrapolated) | 62.8 | 1,639.4 | 8,083.9 |
| Model/index load (s; separately measured) | 20.79 | 32.40 | 11.48 |

All routes ran on the same A100 MIG 1g.20gb, Torch 2.1.2/CUDA 12.2,
eight CPU threads. The **actual encoder dtype was FP32**, despite the archival
protocol's intended BF16; both 4B rerankers were BF16. Same actual dtype is
checked across profiles. Memory is the Torch allocator peak, not total device
occupation; RSS includes loading. Reduced candidate width does not remove the
resident 4B model. Load times are cache/order dependent, not a cold-start comparison.
There were zero API calls. No provider price or monetary saving was measured.
The allocation used 0.3706 MIG-slice hours (not full-A100 GPU hours), TotalCPU
1,344.786 s and Slurm batch MaxRSS 13,536,104 KiB.

## Paired uncertainty and decision

5,000 paired query-bootstrap draws, seed 20260927; 95% percentile intervals.
All differences below are candidate minus baseline. These are exploratory
validation intervals, not independent-test or multiple-comparison-adjusted claims.

| Candidate versus baseline | ΔR@5 [95% CI] | ΔF1@5 [95% CI] |
|---|---:|---:|
| Top-20 versus LTR | −.01058 [−.05820, .03558] | .00009 [−.02355, .02290] |
| Top-100 versus LTR | .02209 [−.01878, .06270] | .01409 [−.00957, .03728] |
| Top-100 versus Top-20 | .03267 [−.00265, .07169] | .01400 [−.00060, .02962] |

The full record includes all six metric intervals. Top-100 R@50 is higher than
LTR and Top-20 with positive unadjusted intervals, but that does not establish
a Top-5 benefit. Crossing zero is neither proof of equivalence nor a reason to
retrain/retest until a positive result appears.

- **Default candidate: LTR.** It gives .6054 R@5 and .3828 F1@5 at 77.8 ms P95
  and 2.25 GiB Torch allocated. Its ranking is CPU-based, but query encoding
  still uses GPU: the complete route is not CPU-only.
- **Optional slower candidate: Top-100 4B.** It has the highest R@5/F1 point
  estimates, at 9.30 s P95 and 11.07 GiB allocated. Retain for latency-tolerant
  offline use, without claiming significant Top-5 superiority over LTR.
- **Do not select Top-20 as the default compromise.** F1 differs from LTR by
  only .00009 with a wide crossing-zero interval, while P95 is 1.91 s and
  allocated memory 10.54 GiB. It did not demonstrate a useful benefit here.

The mechanical F1/P95/Torch-memory Pareto test places all three point estimates
on its frontier, because even a tiny positive F1 difference counts. This is not
an uncertainty-aware dominance result; the recommendation also considers effect
size, uncertainty and deployment purpose.

Top-20 scores only the first 20 of the same initial RRF pool. The unscored tail
is retained at depth 100 but **cannot enter Top-5 under the frozen equal-weight
rank fusion**: the minimum scored sum is 2/80 while the maximum unscored score
is 1/81. This is a real coverage/cost tradeoff, not equivalent Top-100 reranking.
No query-dependent router was trained from these observations.

## Error slices and limits

| Deterministic diagnostic slice | n | LTR R@5 | Top-20 R@5 | Top-100 R@5 |
|---|---:|---:|---:|---:|
| Entity heuristic | 29 | .6707 | .6086 | .6431 |
| Number / year | 37 | .5797 | .5752 | .5977 |
| Geographic heuristic | 62 | .6151 | .6118 | .6608 |
| Lexical mismatch | 62 | .4960 | .4503 | .5000 |
| Semantic-paraphrase heuristic | 113 | .5832 | .5677 | .6012 |
| Multi-evidence | 89 | .6212 | .6287 | .6412 |
| Unanswerable | 0 | N/A | N/A | N/A |

These overlapping query/gold-derived tags are not manually verified intent
labels; geographic substring and lexical-overlap rules are coarse diagnostics.
The entity slice shows that a bigger reranker is not uniformly better; the
lexical-mismatch slice shows the potential cost of narrowing coverage. Neither
is a validated conditional-routing policy. The 104 non-decisive validation
claims were excluded, so this run cannot evaluate abstention or verdicts.

## Reproduction, evidence and handoff

The compact [result record](verified-runs/search-tradeoffs-20260927.json) contains
source/index/LTR/model revisions, per-profile prediction/trace hashes, all
metrics/intervals, taxonomy and final accounting. The 490,143-byte source bundle
SHA is `d3c9f3e4071eb1c1448718073f22e57fef32e0a8a68a92220ba1bc3eb3304ed1`;
its 16-file payload tree SHA is
`c9973581d005969a70172dd46382dd93d1b9d5f72c641016f5067e601686ea07`.
Only aggregates enter Git; raw predictions/traces and corpus archives do not.

```sh
# Check an existing extracted bundle; no model or corpus access.
python scripts/public_v2_stage_manifest.py verify BUNDLE
python scripts/compare_profile_metrics.py --run-dir BUNDLE \
  --baseline rerank20 --candidate rerank100 --output NEW_EXTERNAL_RESULT.json
# Reproduce inference only through the documented Slurm wrapper and pinned inputs.
# No new GPU run is needed to reproduce the supplemental paired statistics.
```

The direct statistics use script SHA `97b2dc8`, while inference remains pinned
to `b797f66`; later documentation/privacy fixes are not relabelled compute.
P0/P1 of the Climate work package are complete. An independent transfer score,
repaired-public-adapter quality gain, HTTP load test and production rollout are
not completed or claimed. [Career handoff](CAREER_HANDOFF_20260927.md) provides
two candidate bullets without changing current resume/shared career files.
