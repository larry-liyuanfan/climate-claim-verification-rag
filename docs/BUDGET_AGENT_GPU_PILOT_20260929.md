# Single authored GPU pilot — operator record

**Terminal update:**31510619 completed0:0, but all9 generated responses failed
structured parsing/validation. The full evaluation is blocked, not released.
The earlier pending record below is historical; see terminal acceptance below.

Release: `climate-authored-pilot-ca9fa53-20260929-user-resume`. The coordinator
relayed the user's explicit bounded exception at18% remaining allowance after
the other project packages ended. This permits **one pilot**, not automatic
validation/vNext, retraining, new models, deployment or a retry. No credits/reset
were purchased. Shared career files remain coordinator-owned.

## Immutable execution

The executable source remains `ca9fa53f2d4b70091fd08a8a1512e263f329fc61`;
all four archives/model revisions/protocols remain those in
[CPU readiness](AGENT_ASSET_PREPARATION_20260929.md). CPU job31486505 is retained,
not rerun. First stage is the5,240-document public BM25 corpus, no dense/ANN.
Pilot is3 authored tasks ×3 frozen routes, at most15 generation attempts and
7,680 new tokens (reranker forward work is separate). No official test or gold
is read, and these authored examples cannot establish generalization quality.

The external [operator wrapper](../hpc/budget_agent_pilot_observed.sbatch) invokes
the exact frozen `hpc/budget_agent_pilot.sbatch` unchanged. Resources remain
one full A100,8CPU,32GiB host RAM,30GiB scratch,15min ceiling. Slurm owns CUDA
visibility; the submission script clears inherited CPU-only settings before
submission. This proposal has not been measured by a GPU pilot yet.

Operator wrapper SHA-256:
`569d824e10645c0fa0ce4604e33810a95085969bdf730adc182e5483227780a8`.
It adds only external1s NVML sampling, job start/end/exit records, pre-timeout
and termination snapshots of this job's small result/consumed receipts. It does
not modify model/data/code, arguments, RNG, tokenizer or inference behavior.
The original successful result archive and Slurm log stay distinct from the
operator diagnostics; missing results are not treated as successful execution.

## Observability and limits

- Slurm provides terminal state, Elapsed/TotalCPU/MaxRSS and allocation TRES.
- Successful run JSON provides generation-provider attempts, known input/output
  token counts, per-route/stage elapsed time, retrieval/rerank calls and candidate
  pair attempts. Failed attempts are not silently counted as successful forwards.
-1s NVML observations provide **sampled observed device-memory maximum**, not
  a precise CUDA allocation peak. Scratch high-watermark is not captured.
- The frozen runtime does not isolate generator/reranker load latency, TTFT or
  successful neural forward counts. Those remain unavailable; elapsed time
  outside measured query loops includes extraction/hash/import/loading/setup,
  not a pure model-load measurement. No instrumentation hooks were injected.
- If timeout/failure occurs, retain the log/status/available receipts, report the
  failure and stop. Do not auto-submit a replacement or final evaluation.

## Pre-submission checks

Read-only account queue was empty. The project mount had~276GiB available and
600 free inodes; the home quota was full. Use existing archives and node scratch,
not a new persistent model/environment tree or home cache. Refresh guards in
the operator submission script immediately before `sbatch --test-only` and the
single submission: no queued jobs, at least30 free inodes, exact source/operator
hashes, complete input size and unused result/diagnostic paths.

New operator-only checks: bash syntax,3 static contract tests, targeted Ruff.
The frozen executable's previously passing full CI remains the test evidence;
the unchanged CPU suite is not rerun. Static operator checks do not prove Slurm
signal delivery or successful GPU runtime behavior.

## Submitted; execution pending

Actual job **31510619** was submitted after successful `sbatch --test-only`.
The test-only response used31510618; that is **not** an allocated pilot.
Submission operator SHA-256:
`b2ac83bb40e8056d17b839bab7dedeb4cfdab965baab035b240246ca0a402f15`.
The wrapper is committed in `ae5166b872c9e0f71be09df1dd557c1804c01bf7`;
executable source remains ca9fa53. The remote wrapper/operator hashes matched.

Initial state was PENDING/Priority, priority13157 (fairshare13146, jobsize11,
age0). Requested8CPU/32G/1A100, with no allocation yet. Slurm defaulted to
Requeue=1; it was explicitly changed to0 on this exact job without cancellation
or resubmission, and read back as0/Restarts0. This prevents automatic retry.
The next scheduler snapshot estimated `2026-09-29T16:49:40` (scheduler time,
timezone not captured); estimates can change and do not mean the job is running.
The earlier test-only Oct5 estimate was superseded by this live job estimate.

Output remains a new `runs/budget-agent-pilot-ca9fa53-release-20260929.tar.gz`;
diagnostics use the same stem plus `-diagnostics/`, and Slurm log is
`runs/slurm-agent-pilot-31510619.log`. The [submission receipt](verified-runs/budget-agent-pilot-submission-20260929.json)
records identity and state, not model execution. Preserve queue age, do not
submit another job or poll with a new automation. A subsequent authorized
status check should inspect31510619 only and retain its failure/timeout evidence.

No actual weight loading, generation, tokens, forwards or GPU/resource timing is
claimed while pending. A resource-based estimate for32validation×3routes and
8authored×3routes must wait for the pilot. Only frozen workload ceilings can
currently be stated: at most160/40 generation attempts and81,920/20,480 new
tokens respectively, excluding reranker compute. Neither workload is released.

## Terminal acceptance: infrastructure passed, structured decisions failed

The [compact audit and three original observed cases](verified-runs/budget-agent-gpu-pilot-31510619.json)
were reproduced from the3,105-byte archive (SHA
`808c7ae89af7c135b601295da87a2acb7727345c6429ccd4e220b658abd89a48`).
It contains the consumed receipt and27,422-byte run JSON, not just an empty
success marker. Nine unique task×route records are present; exact source,
protocol, public corpus, model file-manifests and canonical manifests match.
The preserved operator copy of run JSON is byte-identical to the archive member.

| Original authored case | Routes / generation attempts | Output tokens | Completed nonempty rerank | Outcome |
|---|---:|---:|---:|---|
| Arctic sea ice decline |3 /3 |774 |20 pairs | All3 GeneratedResponseError; no accepted action |
| CO2 infrared absorption + ocean acidity |3 /3 |953 |20 pairs | All3 GeneratedResponseError; no accepted action |
| Explicit nonexistent-term query |3 /3 |228 |0 pairs | All3 GeneratedResponseError, **not** model-chosen abstention |

Both checkpoint shard groups loaded; nine real generation attempts emitted
10,533 input /1,955 output tokens, with known usage retained even on errors.
There were9 retrieval calls and3 fixed-chain rerank calls, including one empty
no-op. The2 completed nonempty20-pair stages imply40 batch-size-1 forward calls
by the frozen implementation; no independent forward profiler was installed.
There were **zero validated decisions, zero accepted model-selected tools and
zero published answers**. These records cannot support semantic/citation quality
or Agent task-success claims. Citation precision/completeness are undefined,
not100% because the controller suppressed every answer.

`GeneratedResponseError` in ca9fa53 combined `json.loads` and Pydantic validation.
The controller saved only type/usage, not chained cause or decoded text. All
original decoded responses are lost; no reconstruction is presented. Maximum
output length450 was below512, so hitting the token cap is not evidenced.
EOS/stop causes were not recorded. Static inspection found that cross-field
validators are not fully represented by `model_json_schema`, but this is only
a protocol limitation, **not a demonstrated cause of these nine failures**.

Resource observations:

- Slurm:153s elapsed,124.678 CPU-seconds,8CPU/32GiB/one full A100;
  batch MaxRSS18,192,068K =17.349GiB. Operator elapsed150s.
- Recorded query-loop elapsed totals56.337s. The remaining~93.663s includes
  hashing/extraction/import/model loading/index setup/archive work; it is **not**
  an isolated weight-load measurement. No TTFT/model-only generation latency.
- The sampler CSV contains148 `No devices were found` lines and **zero valid
  NVML records**; stderr is empty. Sampled VRAM and CUDA peak remain unknown.
  Successful CUDA model execution does not repair missing memory telemetry.

The pilot stopped every route at its first invalid response. Therefore153s
cannot justify extrapolating a successful32validation×3 +8authored×3 workload.
The existing one-A100/8CPU/32GiB envelope ran this failure path; GPU headroom and
successful multi-step latency are unverified. Keep the known ceiling of200
generation attempts /102,400 new tokens for those two future workloads, but
do **not** assign a submission walltime or GPU-hours before a passing canary.
The120s/query guard is a soft deadline, not a hard upper bound for all tool work.

Reproduce this audit locally without generation, labels or restricted data:

```powershell
.\.venv\Scripts\python.exe scripts/audit_budget_pilot_31510619.py --artifacts artifacts/pilot-31510619 --models artifacts/budget-agent-models-20260929 --evidence data/PUBLIC_EVIDENCE.jsonl --output artifacts/pilot-31510619/reproduced-audit.json
```

`PUBLIC_EVIDENCE.jsonl` must match the frozen5,240-document corpus SHA; it is
not a placeholder dataset. The audit validates identities, row completeness,
candidate IDs and token totals and retains the original three case trajectories.
No raw individual predictions are copied to the career workspace.

## Authorized CPU-only diagnostic repair; canary not released

The coordinator subsequently authorized a minimal diagnosis patch, not a model,
prompt, label, budget, schema or tolerant-parser change. It distinguishes
`json_decode` (line/column of the unchanged stripped JSON parse) from
`schema_validation` (bounded, redacted loc/type); decode failures remain separate.
Output UTF-8 fingerprint/character and byte counts, token-cap/EOS observations
and generation-call duration are recorded before parsing. An observed EOS or
cap flag is not promoted to an authoritative stopping cause. Failure token usage
still propagates to controller totals. Public metadata contains no generated
text, Pydantic input/context/error message, prompt or private filesystem path.

The optional original decoded string is stored only in an explicitly configured
Spartan Climate `runs/.../private-responses` attachment: owner-only0700 directory,
exclusive0600 files,32KiB per response, at most32 files/1MiB. Oversized output is
skipped, never silently truncated into parseable JSON. Storage failure does not
convert a decision to success or relax validation. Those files are outside the
node-local `result/` archive and must never enter GitHub/career materials.

The [prepared diagnostic operator](../hpc/budget_agent_diagnostic_canary.sbatch)
requires a **new** release ID and exact new source identity, preserves the existing
pilot inputs/phase, disables requeue and saves bounded failure receipts. It is
not submitted or uploaded by this closeout. Its SHA-256 is
`c257d8b49a1e2c2112cb209fbca0e86e8ae6ca1989c7c18f4d2f860c15e30dd6`.
It explicitly marks GPU-memory telemetry unavailable rather than repeating the
failed NVML observation as a measured peak. A subsequent release must set the
source/archive hashes and fresh output paths and pass `sbatch --test-only`.

CPU fixed-output fixtures exercise legal JSON, malformed JSON/Markdown fences,
cross-field conflicts, redaction, token retention and private attachment limits.
The13 new diagnostics cases plus22 affected controller regression cases passed
(35 total); Ruff passed and mypy checked37 source files. AST comparison against
ca9fa53 confirmed the system prompt, generation/chat-template arguments, budget
and strict decision/statement schemas are identical. Canary bash syntax passed.
These synthetic diagnostics do **not** diagnose the lost outputs or establish
that the real-model root cause has been fixed. Prompt/generate arguments and
strict schema remain unchanged. No pilot/canary/full rerun, deployment, shared
career update or resume claim is authorized by this diagnostic commit.
