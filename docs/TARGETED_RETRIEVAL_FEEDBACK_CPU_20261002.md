# Targeted retrieval feedback — CPU implementation and frozen replay

Status: CPU implementation, synthetic whole-chain regression and input metadata
identity only. No weights loaded, no model inference/training, no Slurm submission,
and no real gold contents read in this package. Historical result identities remain
unchanged. No resume edit or quality-improvement claim.

## Hypothesis and changed mechanism

The prior v3 query guard required the original numbers/entities/qualifiers and
at least 50% lexical overlap. That can reject a useful subquestion or a search
for counter-evidence. The new opt-in protocol is
`sentence-targeted-feedback-v1-20261002`; old v3 defaults and historical runs stay
available. Its final claim never changes, but search text need not reproduce the
claim. Empty/duplicate/overlong queries, unsupported fields, excess calls and
unknown citations are still rejected. Purpose is intent, not semantic relevance.

An adaptive model action actually searches the corpus. The next request contains
the returned candidate IDs, empty-result status, full selected evidence, previews,
and the previous query history. RRF is recomputed over original per-query rankings
(k=60, Top20, stable source-ID ties), not recursively over fused rankings.
Previously displayed full sentences remain available. If old plus new selected
full context cannot fit, the controller records an explicit capacity failure;
it does not silently hide the new evidence or pretend a read succeeded.

Research inspiration: [IRCoT](https://aclanthology.org/2023.acl-long.557/) motivates
letting retrieved content influence subsequent retrieval. This is not its benchmark
or a reproduction, and none of its reported gains are attributed to this project.

## Controls, common permissions and limits

| Route | Search policy | Final decision |
|---|---|---|
| fixed_retrieval | Original claim BM25 Top20 | Same base model, exact-read sentence citations |
| fixed_rerank | Original BM25 → Qwen3-4B rerank | Same final decision contract |
| deterministic_extra | Up to two deterministic subquestion/counter-evidence queries → RRF → rerank | Same contract |
| fixed_multiquery | Model plans 0–2 queries **before seeing retrieval**; original+planned searches execute unchanged → RRF → rerank | Same model judges original claim |
| adaptive | Original BM25; model may search/read/rerank based on actual feedback | Same model judges original claim |

All routes share 5 generation calls, 5 tool calls, 2 extra queries, 20 candidates,
5 newly selected sources, 8,192 input tokens, 512 output tokens and 120 seconds per
slot. Greedy generation, fresh grammar state, immutable base weights and seed are
bound by the existing `GenerationBinding`; there are no warmup generations.
Generator and reranker use the accepted serial GPU-residency implementation.

**Fairness boundary:** the query **item** schema (`purpose`, `query`) is identical,
but an upfront array plan and adaptive action grammar are different control
protocols. Actual model/tool/token costs are not equal. The comparison assesses
whole policies under common ceilings; it is not a pure causal isolation of
feedback. Fixed multiquery is required alongside the stronger fixed-rerank control;
a gain over deterministic extra alone is insufficient.

Dense/ANN is not enabled in this bounded query-policy replay. Existing large-scale
retrieval, LoRA development and ANN results remain a separate evidence track.

## Frozen original input identities

The cohort is the **same 32 already-consumed CLIMATE validation claims**, in the
original order: 24 decisive/evidence-bearing plus 8 nondecisive. No favorable
reselection, new split, new labels or independent-test interpretation. No SciFact
dev300 or historical/v2 frozen test is accessed.

| Artifact | SHA-256 |
|---|---|
| Original input archive | `563738f0be1f7bf7b99b8de20bcdeab552f7e893166dec260b8f7f1e7951c3c1` |
| `validation-protocol.json` | `abfcb61e9e6a54641f025101a4011fa88e07e3697a36f0acd308fa8e86052674` |
| `evidence.jsonl` (5,240 records) | `c14315aee9feecbbbbc3b0c7101d978b52b47cee9a613336304bcaa736460c71` |
| Committed selection manifest | `988b6682034a70966c8bbad5ff3c42202933fe85791b97a5f39e4a1b68071bc6` |
| Ordered 32 IDs (compact JSON encoding) | `6b197778061b1e3cbe9e250a962bc34883c0d5224ce980ccb2313fd134195f8c` |
| External original gold file (hash only checked here) | `d2dd28422bffacf87ded2153b3bfac4ca9e1edc903e1d4e7a88d3a40f5d2fd8b` |
| Generator model identity | `d1dd9783afdf4e0fbd21eee824834d71b86982f5a5d5f6f371fe07f2f76f3cf6` |
| Reranker model identity | `de1d4ac39101816774439e68881e2308c5e5f1bd94d0b0dc4c492a56c2681052` |

Model-manifest **file bytes** and canonical model-identity hashes differ; both are
pinned in `targeted_replay.py`. Metadata inspection verified the archive members
and existing environment receipts without rehashing the whole 16 GB archive,
extracting weights or parsing gold. Future authorized preparation checks the
archive and model files on allocated scratch. Large assets stay on Spartan.

## Scoring and decision frozen before model execution

The complete matrix is 32×5=160 slots. Recall@5, MRR@10, nDCG@10 and Evidence F1
reuse the existing CLIMATE gold definition. Evidence F1 means **Top5 ranked
retrieval F1**, not final citation F1. Retrieval and official decisive-label
accuracy use the original 24 eligible tasks. All 32 remain in each route's
failure, latency and cost denominator. Nondecisive false answers are reported;
they are not automatically treated as correct NEI predictions. An unconfigured
verdict provider is rejected, not assigned a classification score of zero.
`validation_repair_exhausted` and `generation_budget_exhausted` are execution
failures, including on the eight nondecisive claims, not intentional abstentions.
Their consumed work remains charged and can prevent the efficiency gate passing
even when all 24 decisive quality scores improve.

Citation ID precision/recall/F1 are additional diagnostics. The physical raw
decision, immutable source text, actually displayed sentence and final citation
are cross-checked. Provenance/ID matches do not prove semantic entailment.

For **each** of fixed rerank and fixed multiquery:

- Run 5,000 paired bootstrap resamples, seed `20260929`, over matched task IDs.
- Quality gate: adaptive's lower 95% difference interval exceeds zero for both
  Evidence F1 and official decisive-label accuracy; mean Recall/MRR/nDCG must
  not regress.
- Efficiency gate additionally requires complete generator **and reranker**
  physical accounting, no more operational failures/pairs/reranker tokens, no
  higher mean generator tokens, and no higher mean slot elapsed time.
- Quality-only success is an extra-compute tradeoff, not an efficiency win.
  Both gates are development decisions, not independent generalization evidence.

Latency includes actual tool/model work and GPU swaps; model loading/index building
is recorded separately. All failure/unknown token totals are retained. Missing
slots or invalid physical provenance cause an unscored failed run, not a smaller
successful denominator. API currency cost is unavailable, never invented from
GPU time. No resume promotion is automatic.

## Whole-chain review before a queue submission

Changed production path:

`exact source/draft → supervisor → frozen original inputs → five-route runner →
real retrieval/read feedback → physical journals → reaped child → cost receipt →
provenance/config audit → original gold loader → paired compact score`.

One root writer implemented changes; independent read-only review found and
resolved silent context overflow, an overflow-test false positive, missing
physical-decision binding, and incomplete reranker cost gating. Synthetic tests
cover full success plus preparation/worker/scorer failures, irrelevant new IDs,
unchanged claims, fixed upfront planning, actual feedback, no preview citation,
retained evidence, query duplicates, ranking order, tampered label/physical IDs,
and reranker-accounting drift. The actual LMFE parser consumes the new plan-array
schema on CPU. Unchanged supervisor, source packing and runtime helpers are reused;
there is no new generalized audit framework.

This cannot prove model quality or eliminate all GPU-runtime risk. It deliberately
catches affected-chain problems before spending queue time.

## Reproduction and exact-source freeze

From this checkout (pinned validation environment; no model files needed):

```powershell
.\.venv-validation\Scripts\python.exe -m pytest -q tests/test_targeted_retrieval.py tests/test_targeted_replay.py
.\.venv-validation\Scripts\python.exe -m ruff check src tests scripts
.\.venv-validation\Scripts\python.exe -m mypy --platform linux src/climate_rag
.\.venv-validation\Scripts\python.exe -m mypy --platform linux --follow-imports=silent scripts/run_targeted_replay.py scripts/run_targeted_replay_operator.py scripts/package_targeted_replay.py
.\.venv-validation\Scripts\python.exe -m pytest -q
.\.venv-validation\Scripts\python.exe scripts/scan_tracked_secrets.py
# After review/tests, commit on own codex branch and freeze the exact clean SHA:
.\.venv-validation\Scripts\python.exe scripts/package_targeted_replay.py --source-git <40-hex-commit> --output <new-private-output-directory> --bash <bash-executable>
```

The package produces `source.tar`, `wrapper.sbatch`, `source-receipt.json`, and
`release.unauthorized.json`. The source archive is validated against exact Git
blobs and the real shell source guard is exercised. The draft's
`model_execution_authorized=false` is checked before any input preparation/model
access. Previous run authorizations cannot authorize this source.

Only **after separate exact-source authorization** and current allocation checks,
the coordinator can bind an authorized release and use the following commands;
none have been run for this package:

```bash
# All paths must be under the project's envs directory; values come from the
# exact frozen package and separately authorized release, never placeholders.
export CLIMATE_SOURCE_TAR=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/envs/<source.tar>
export CLIMATE_SOURCE_SHA256=<source-receipt-source_archive_sha256>
export CLIMATE_SOURCE_GIT=<exact-40-hex-commit>
export CLIMATE_TARGETED_RELEASE_FILE=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2/envs/<authorized-release.json>
export CLIMATE_TARGETED_RELEASE_SHA=<authorized-release-file-sha256>
sbatch --test-only --export=ALL <exact-packaged-wrapper.sbatch>
# Only if the exact package is authorized, nonduplicated and schedulable:
sbatch --parsable --export=ALL <exact-packaged-wrapper.sbatch>
```

The wrapper invokes `run_targeted_replay_operator.py --release … --release-sha …`;
it calls `run_targeted_replay.py worker …`, reaps that child, writes physical
costs, then calls `run_targeted_replay.py score …` in a separate CPU process.
No retry/resume is implicit. Runtime failures preserve costs and fail closed.

The candidate requests one A100, 8 CPU, 32 GiB RAM and 30 GiB scratch, reusing
accepted serial model-residency shape. The **two-hour allocation** is calibrated
from [job31543304's public compact](verified-runs/budget-agent-full-31543304.json),
SHA `0094b372d98144d90fe9c09c4998bca29dda9d189cc482753aa50f363a86082f`:
the old validation96 slots took 425.403935 seconds of summed episode time,
including 406.105126 seconds for 96 generator calls and 18.078303 seconds for
32 reranker requests. Its full two-phase operator took 696.434932 seconds
(receipt SHA `4f1e7a0e2e3d2fcf2dc64ddf5d4ac128e65dda59e41af2b880a16a65f9e12f34`).
Episode sums are **not** independent phase wall time.

At the new maxima of 800 calls and 128 reranks, 1.5× those per-call measurements
is approximately 5,185 seconds. The worker cap is 6,000 seconds including model
loading/index work; preparation600 + score300 + shell/receipts300 gives 7,200.
This is a bounded resource proposal, not a duration prediction, guarantee that
all160 slots complete, or online SLA. Slow execution is stopped and costs retained;
it is never silently retried. New prompt length and query generation can change
runtime. Exact authorization/schedulability checks remain required; no pending
job is cancelled or replaced here.
