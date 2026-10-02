# Climate: recovered evidence and next implementation decision

The latest Runpod experiment completed **160/160 slots** and is already fully
backed up locally. Reuse it. Autonomous acquisition did not improve quality:
all 32 acquisition gates stopped. This package adds a gold-free recorded-trace
replay and a CPU comparison contract, not another model run or a deployment.

## Recovery register

These statuses concern the registered local copies, not an exhaustive disk search.
Spartan is inaccessible; no connection or recovery prerequisite is introduced.

| Run / attempt | Executed source | Existing local evidence | Missing / recovery status | Does this package depend on missing originals? |
|---|---|---|---|---|
| Restricted representation 29465819 | c815070 | `verified-runs/qwen3-embedding-lora-full-gate-20260821.json` | Public aggregate exists; raw restricted records/models not recovered here. Remote inaccessible, not presumed lost. | No: reuse published dev results only |
| Public search 31364586 / 31364587 | b797f660 | `verified-runs/search-tradeoffs-20260927.json` | Aggregate/profile hashes exist; original per-query traces not recovered in this package | No: no fresh scoring |
| Budget full 31543304 / feedback 31587302 | 72eaa90 + operator 947645b / d7d5bf3 | Corresponding `verified-runs/budget-agent-*.json` | Compact exists; registered closeout says private original rows were not exported | No |
| Document bounded 31930176 | 49f168a25170 | `E:/Project/_climate_transfer/document-bounded-49f168a25170/run` | Raw candidate copy exists; historical 686-file receipt says `no_files_removed`. Completeness not re-audited now. | No |
| Targeted feedback 32030221 | 5004fa01a4d4 | `verified-runs/targeted-feedback-32030221.json`; `_climate_transfer/targeted-feedback-5004fa01a4d4/closeout` | Three local aggregates; original raw remains remote inaccessible, not recovered | No: local aggregate answers activation question |
| Runpod hc6rqkvsww16x2 prelaunch | 46615fe | `_climate_transfer/climate-runpod-namespace-4c25615` receipts | Namespace preflight failed before models; no model results generated | No; never infer quality zero |
| Runpod tih3av7a5sdxvd migration attempt | 4c25615 | `_climate_transfer/climate-runpod-migration-4c25615`; private-storage repair handoff | Pre-model POSIX permission refusal: 0/160 slots, zero generator/reranker. Some failure originals may remain in the volume. | No model-result recovery needed |
| stop-acquire-private-20261003, final Pod udm7fa2t6smwgl | **54031fa09ab0c3fc3ef610babd838b90784940fc** | **`E:/Project/_climate_transfer/private-recovery-54031fa-20261003/`** | Complete raw/compact/cost/outer/deadline backup, already accepted. Original tar 21,555,200 bytes. | No missing dependency; do not redownload or rerun |

The three cloud Pod IDs are infrastructure attempts/migrations, not three
independent model experiments. Their current console status is compute **Not
running**, with three retained 120 GB local `/workspace` volumes. Volume contents
are not freshly inspected while stopped; persistent assets must not be confused
with private `/root` container output. The latest private output was recovered
before stopping. No retained volume or Pod was deleted.

[Official Zero GPU recovery](https://docs.runpod.io/pods/troubleshooting/zero-gpus)
applies when the original machine's GPU is unavailable; it reopens that Pod's
storage, not an arbitrary new CPU Pod's access to old local disks. No explicit
zero-GPU option was visible in the inspected stopped-Pod menu. No A100 was started
for logs. Financial observations and the budget reconciliation stay in the private
handoff, not the public repository.

## Latest immutable identities and inherited acceptance

Executed source is **54031fa**, distinct from this later reporting/CPU commit.
Run ID is `stop-acquire-private-20261003`; final instance is `udm7fa2t6smwgl`.

| Artifact | SHA-256 |
|---|---|
| Original authorized release | `73074acb41fae4e29b39827bd154a6d1d45bfb127b0f1074ba6efd9da8846728` |
| Original private `inference/run.json` | `79e9fad3580731a7ffba2363f4ca83585c0007772bbce28863c7a7e27a2c1aa0` |
| [Original compact, byte-identical safe copy](verified-runs/stop-acquire-runpod-20261003.json) | `0b6a935c3c53de99c78847418d0fedc1a4ebeb8875bed852a1f962ac0680746d` |
| Original cost | `c64195a3e192f6117b532aa274f9f6a57cbe2b517bc9af20cf1d17b0ccd5dfe0` |
| New gold-free transition census (canonical LF bytes) | `1202ef26a6f7d42ea02b2faded516566a3ad418b83e59ebbec07769c159a34ae` |

The already accepted local `replay-audit/public-summary.json` records unchanged
score/cost reproduction. This package **does not repeat that CPU scoring**.
Outer exit 0, inner completed, deadline completed, worker reaped/exit 0, and
unknown usage zero are inherited actual acceptance, not inferred from test success.

## Full original comparison

All five routes have 32 tasks over the same 5,240-document corpus. Retrieval
denominator **24**, binary labels **23**, cost/failure **32** per route. The one
DISPUTED task remains in retrieval/cost, not relabelled for binary accuracy.
This is repeatedly consumed validation, not independent test or online A/B.

| Route | Recall@5 | MRR@10 | nDCG@10 | Evidence F1@5 | Binary correct /23 | Generator calls | Mean generator tokens | P50/P95 episode seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fixed_retrieval | .4319 | .4042 | .3876 | .2578 | 8 | 32 | 1641.56 | .745 / 1.331 |
| fixed_rerank | .4403 | .4726 | .4463 | .2563 | 11 | 32 | 1515.56 | 9.117 / 10.615 |
| deterministic_extra | .4403 | .4726 | .4465 | .2563 | 11 | 32 | 1559.94 | 9.089 / 11.194 |
| fixed_multiquery | .4299 | .4761 | .4365 | .2470 | **14** | 64 | 2088.53 | 15.173 / 17.023 |
| adaptive stop/acquire | .4319 | .4042 | .3876 | .2578 | 8 | **68** | **3520.63** | 1.644 / 2.365 |

The original 5,000 paired bootstrap for adaptive minus fixed_multiquery binary
accuracy is **−.2609**, 95% interval **[−.4783, −.0435]**, p=.0256 (23 pairs).
Neither original quality nor efficiency gate passed. Do not merge this with
32030221: that older protocol scored adaptive 12/23 and fixed_multiquery 15/23.

Physical totals: **228 generator calls**, **323,069 input + 7,370 output tokens**;
**96 reranker requests / 1,920 pairs / 252,801 nonpadding tokens**. Stage ledger
time is 360.088 s generator and 783.365 s reranker including swaps. Operator wall
time is **1293.492 s**. No unknown usage, no operational failures. The fixed
rerank/deterministic/multiquery routes each use 640 pairs; adaptive uses zero.
These are serial batch measurements, not HTTP SLA. API currency costs are null,
not zero; provider rental/migration/storage expense is a separate private ledger.

## What the real behavior establishes

[Recorded census](verified-runs/recovered-acquisition-census-20261003.json) checks
the actual saved physical observation/schema against each recorded attempt. It
does not open gold, change predictions, rescore quality, or call a model.

- All 32 first gates have five generation calls, four tools and two queries
  remaining; five full evidence passages and 15 unread previews; read/query
  branches open. All 32 actual proposals are validated `stop`.
- No autonomous query/read execution, new evidence delivery, or feedback-driven
  continuation occurs. The 68 calls are 32 gates plus 36 verdict attempts.
- Four duplicate-reference errors across three tasks are repaired by subsequent
  real verdict attempts. This demonstrates final-output validation recovery,
  not recovery from a failed acquisition tool.
- Fixed multiquery really executes 64 queries. Its 718 new candidate occurrences
  yield 25 additions to final citable contexts. Candidate novelty is not gold
  relevance or semantic support.
- Across five routes, 101 answers contain 303 citations to displayed sentences.
  Visibility/ID legality is not evidence that the sentences support the judgment.

Latest case inventory: stop→verdict **32**, valid autonomous acquire→delivery→next
decision **0**, actual autonomous empty/failed tool recovery **0**. The first
saved stop→answer example has three visible citations and an incorrect binary
label in the accepted public audit. It cannot be marketed as a reasonable-stop
success. The existing empty/timeout branching fixture demonstrates executor
reachability only; it is never substituted for a real model case.

## One different, falsifiable next hypothesis

**Equal ranking access and a common terminal interface may change the controller
comparison.** The current acquisition gate exposes read/query but **no rerank**;
the strongest fixed control uses the 4B reranker. Controls also retain the old
terminal interface while adaptive has a separate gate/verdict interface. This is
a documented whole-policy comparison, not a code fault or a causal feedback
estimate. A new comparison should remove these two confounds before attributing
failure to feedback or adding more semantic prompt fields.

Do not simply retry the prior relation/minimal-rationale v3. Its already consumed
SciFact TRAIN24 paired comparison lost rationalized F1 (.5882→.3500 for top1)
and cost more: [negative confirmation](SCIFACT_PAIRED_VERIFIER_RESULT_20261002.md).
That evidence does not support promoting the same intervention under a new name.

Prepared CPU contract: `climate_rag.acquisition_comparison.validate_comparison`.
It rejects mismatched initial frames, terminal contract, tools, cohort, models,
corpus or complete budgets. All three arms must expose **read/query/rerank**, with
optional autonomous acquisition and equal ceilings, not fabricated equal costs.
Hash identities and all budget fields must be present. Synthetic negative checks
cover the current missing-rerank condition and verifier/budget mismatches.

Prepared design: [next comparison](protocols/equal-capability-next-20261003.json).
Strong fixed multiquery freezes its upfront plan; deterministic workflow applies
fixed query/RRF/rerank rules; autonomous chooses optional tools after real feedback.
All use one terminal verifier, source/citation contract and failure denominators.
Measure original quality plus cited semantic support in an isolated evaluator;
never give evaluation annotations to the policy. Count every gate, query planner,
verdict, repair, rerank pair, model swap and rental/storage cost.

Falsification: if autonomous still does not acquire, or fails the preregistered
joint-quality and full-cost comparison, reject promotion. More calls alone do not
pass. Common-verifier change makes old 14/23 historical evidence, not a fresh
unmodified control; all controls must be measured together in the authorized run.

**Preparation status:** replay and CPU fairness check implemented; next three-arm
runtime integration, eligible gold-isolated local cohort and exact asset/config
binding remain pending. No new trial, training or public frozen-test consumption.
No new run is authorized by the old release. Do not tune prompts/labels/budgets on
these 32 to seek a positive result. The next cohort must be frozen without outcome
selection and documented as development, with all prior exposure recorded.
The packet is not GPU-ready and therefore requests no paid start yet. Once ready,
bind source/input/models/config/output and obtain one specific release within the
unchanged whole-round USD20 ceiling.

## Lightweight CPU use and package acceptance

```powershell
& .\.venv-validation\Scripts\python.exe scripts/replay_recovered_acquisition.py `
  --run E:/Project/_climate_transfer/private-recovery-54031fa-20261003/runs/stop-acquire-private-20261003/inference/run.json `
  --run-sha256 79e9fad3580731a7ffba2363f4ca83585c0007772bbce28863c7a7e27a2c1aa0 `
  --output E:/Project/_climate_transfer/new-count-only-replay.json
```

Use a new output filename; original files are never overwritten. The existing
census can be inspected instead of rerunning. Source and evidence identities are
preserved. The affected suite has **15 passing tests**; Ruff and strict typing
validate the two new CPU modules. Unchanged cloud/runtime/scorer validation is
reused. These checks certify local replay/contract code only, not the unimplemented
next three-arm runtime or a real acquisition case.

Recovery/reporting/CPU preparation complete is separate from the overall Agent
goal. The latter still lacks real autonomous acquisition benefit and semantically
supported final answers. See the [application case and STAR](CLIMATE_APPLICATION_CASE_20261003.md).
