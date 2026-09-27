# Search tradeoffs and representation integrity — 2026-09-27

Status: real CPU integrity preflight and clean-checkout validation passed;
GPU profiling job `31364586` is PENDING Resources as of 2026-09-27 10:32 AEST.
Scheduler estimate: 13:48:44 AEST, not a guaranteed start.
No new model-quality improvement is claimed.
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

## Historical evidence boundaries

The 126-query public validation Recall@5 and Evidence F1@5 remain valid for the
base-only downstream run. Its old MRR@10/nDCG@10 and paired differences/CIs are
censored, not complete Top-10 metrics. Renaming nDCG@10 to nDCG@5 is incorrect
because its ideal denominator used ten. Historical timing retains its original
depth and stage scope; new numbers must not replace the old immutable artifacts.

Restricted 1,208,827-document / 154-query / 20-step LoRA remains offline-dev
evidence, not independent test or online A/B. This public loader defect alone
does not prove or disprove the integrity of that separate run. Pilot ties alone
cannot identify their exact cause. The consumed frozen test remains closed;
SciFact remains unopened.

## Remaining release gates

1. Integrity passed: see the verified result below. This satisfies the plan's
   representation-repair option; independent transfer is not required or run.
2. Same-query serial-request profiling of low-latency LambdaMART and bounded 4B
   widths must measure encoding, recall, ranking, end-to-end P50/P95, peak memory
   and compute cost. These will be offline benchmarks, not online SLA.
3. Full-ranking metrics and 5,000 paired bootstrap comparisons require new
   validation-only artifacts, without reopening frozen test.
4. Clean-checkout reproduction passed at `b797f6606ced013b1a0adee915c996b593f180a3`:
   fresh GitHub clone + new virtual environment, 101 tests passed,
   one real-Torch test skipped locally (the prior version passed on Spartan). Ruff passed and
   strict mypy passed all 31 source modules after annotation/Protocol repairs.
   This is source type checking, not static analysis of every experimental script.
   Exact runtime and CI evidence: [reproduction record](verified-runs/search-tradeoffs-reproduction-20260927.json).

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

No current resume or shared career material was modified.
