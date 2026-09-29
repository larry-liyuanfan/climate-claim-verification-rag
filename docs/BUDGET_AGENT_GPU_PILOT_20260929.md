# Single authored GPU pilot — operator record

**Latest update:**31510619 and diagnostic31519179 both completed0:0, but each
rejected all9 responses. Only31519179 retained private originals: exact frozen
Pydantic replay confirms illegal non-rewrite `query` in9/9. The model-visible
contract repair is CPU-tested, not model-confirmed. Confirmation31520350 was
subsequently released/submitted (initially PENDING/Priority); full evaluation
remains unreleased. Earlier submission/readiness sections are historical.

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

## Historical CPU-only diagnostic repair (before canary release)

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

Frozen diagnostic source: `6a764a1b4f9137cf8e8c29abfbc14572c3a1039d`;
1,310,720-byte source archive SHA
`ecdfc7c05334af64d11adb0f03ead5042f9168e9c182968a51936ee33413ba34`.
The [readiness receipt](verified-runs/budget-agent-diagnostic-readiness-20260929.json)
separates this new CPU-tested identity from the old actual pilot runtime/operator.
Thirteen new diagnostics cases also passed from the clean Git archive with imports
verified against its source; its audit script reproduced identical structured
pilot evidence. No new source archive has been uploaded to Spartan and no new
Slurm job has been submitted. Await an explicit diagnostic-canary release.

## Diagnostic canary31519179: confirmed protocol failure

A later explicit release, `climate-response-diagnostic-6a764a1-20260929`, allowed
one diagnostics-only repeat of the original3 authored tasks ×3 routes.
The [submission receipt](verified-runs/budget-agent-diagnostic-submission-20260929.json)
preserves its initial state and is superseded by this
[terminal audit](verified-runs/budget-agent-diagnostic-canary-31519179.json).
The source was6a764a1, with unchanged models, input bundle, budget and prompt.
No validation/frozen-test run, dense retrieval, training or deployment occurred.

- COMPLETED0:0;142s Slurm elapsed,125.008 CPU-seconds;
  batch MaxRSS18,102,124KiB (17.264GiB);8CPU/32GiB/one full A100.
-9 generation attempts;10,531 input /2,165 output tokens. All usage is known.
-9/9 schema-validation failures, not JSON-decode failures; all9 observed EOS,
  zero observed token-cap hits. No accepted action, decision or published answer.
-9 retrieval calls;3 fixed-rerank calls, including one empty no-op and two
  nonempty20-pair stages. GPU peak memory was **not measured**.

Private originals were inspected only inside the existing Spartan Climate run.
The complete unmodified strings parsed as JSON. Frozen6a764a1 class AST and
Pydantic2.13.5 then validated those original objects without deleting or repairing
fields. The [content-free exact-validation receipt](verified-runs/budget-agent-canary-31519179-frozen-validation.json)
confirms all9 raised only the existing `query is only allowed for rewrite`
cross-field rule. Six raw objects proposed `answer` and three proposed `abstain`;
all carried a non-null, nonempty `query`. These **rejected candidates are not
accepted decisions**, and their factual/citation correctness was not assessed.
Only four query fields exactly echoed the input claim; no blanket echo claim
is made. Private directories inherit GPFS setgid (mode2700, access0700), files
are0600. No raw text was exported to this repository or career materials.

The original31510619 strings were not retained: this diagnosis must not be
retroactively attributed to that batch. Different outputs/tokens across these
two executions are not a quality comparison; observations include a dynamic
remaining-seconds field even with identical frozen code and input bundles.

Reproduce the compact audit without reading private originals:

```powershell
.\.venv\Scripts\python.exe scripts/audit_budget_canary_31519179.py --artifacts artifacts/canary-31519179 --inspection docs/verified-runs/budget-agent-canary-31519179-private-inspection.json --frozen-validation docs/verified-runs/budget-agent-canary-31519179-frozen-validation.json --output artifacts/canary-31519179/reproduced-audit.json
```

## Minimal model-visible contract repair (CPU-only)

The subsequent repair exposes existing conditional rules in the actual system
message: only rewrite accepts a nonblank query; other actions require null or
omission; only answer can carry a label/statements; answer requires sufficient
assessment, SUPPORTS/REFUTES and1–3 cited statements. There are no task-specific
answers or examples. `AgentDecision`, statement/budget schemas, action meaning,
whole-string JSON parsing, strict validation, decoding/generation parameters and
downstream citation checks are unchanged. No field deletion, Markdown stripping,
tolerant parsing or acceptance-set relaxation is used.

The runner records static system-message/schema SHA-256 and Pydantic version.
This identity explicitly excludes dynamic observations and the tokenizer chat
template; the model manifest continues to cover tokenizer/template assets.
The31 added CPU contract cases and35 prior affected regressions passed
(66 total); Ruff passed, mypy checked37 source files, and independent read-only
review found no blocking issue. These are fixtures, **not real-model success**.
The first diagnostic batch remains negative and its originals stay private.

A new source/prompt freeze is required before a separately authorized small
confirmation pilot. No new job has been submitted, no full evaluation is
released, and no career/resume files were changed. A passing protocol pilot
would still not establish semantic entailment, citation completeness, independent
benchmark improvement or an online SLA.

Frozen CPU-tested repair: `72eaa90567e3603a3b940edffbb21b684bf1d1ba`;
1,361,920-byte source tar SHA
`ad533a3b14a81ab658049d6261cfc22848ebd792f3333e9f635d327d034d4cb1`.
Static system-prompt SHA (Pydantic2.13.5):
`6b0163a4ef9da624e35dffb51fff34913ccb8a90db395c8dfec68bbe409febaa`.
The [new readiness receipt](verified-runs/budget-agent-protocol-readiness-20260929.json)
records unchanged controller AST/decide path outside the system-message assignment,
schema/operator/input identities and negative-result boundaries. The66 tests
also passed from a clean Git archive with source-import origin checked; the
compact canary audit reproduced identical structured evidence. This archive
has not been uploaded to Spartan. Waiting for a separate confirmation release.

## Subsequent single confirmation release:31520350

**Terminal update:** this historical submission record is superseded by the
[confirmation closeout](BUDGET_AGENT_CONFIRMATION_20260929.md). Job31520350
completed in137seconds; frozen CPU scoring confirmed2 mechanically checked
answers and7 action/query-contract failures. No full comparison is released.

Release `climate-protocol-confirm-72eaa90-20260929` explicitly authorizes only
the same3 authored questions ×3 routes, with frozen72eaa90 source/system-prompt
identities, unchanged models/inputs/budgets/operator and fresh output paths.
The [submission manifest](verified-runs/budget-agent-protocol-confirm-submission-20260929.json)
records remote source/archive checks, the empty account queue before submission,
and successful bash syntax / `sbatch --test-only`. Actual job31520350 was
submitted once at2026-09-29T17:06:35+10:00; its initial state was PENDING/Priority,
with no scheduler start estimate. The test-only estimate is not a reservation.
Requested resources are one full A100,8CPU/32GiB/30GiB scratch/15minutes,
no-requeue. Both earlier results are preserved.

No retry, further prompt/model/input/budget adjustment,32-query validation,
8-task vNext run, test reuse or deployment is released. Original responses and
full answer text stay on Spartan; only compact counts/hashes/validation outcomes
may be exported. Completion must distinguish schema-valid proposals, permitted
executed actions, validated final answers, model abstention and controller
rejections. Protocol compliance alone cannot demonstrate semantic or retrieval
quality. No new recurring monitor or shared career/resume edit was created.

A pre-completion queue snapshot showed PENDING/Priority, then with scheduler estimate
2026-09-29T22:50:00+10:00 (not guaranteed). Priority13171 decomposed as age14,
fairshare13146 and jobsize11, QoS publicgpu. There is no actual generation,
answer or resource result at that snapshot. The linked terminal closeout now
records the result; preserve this queue history rather than treating it as current.

For eventual closeout, the existing32-validation +8-vNext workload has120
routes and controller ceilings of200 generation attempts /102,400 new tokens,
240 tool calls,160 retrievals and80 rerank calls /1,600 attempted candidate pairs.
The3-task pilot ceilings are15 /7,680 /18 /12 /6 /120 respectively. These bounds
were checked against the frozen controller, not measured on the pending job.
The120-second guard is soft: synchronous tool work is not hard-preempted.
No full-run walltime, GPU-memory headroom or eligibility recommendation can be
made before confirmation results. Even a passed pilot needs a separate release.
