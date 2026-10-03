# Fair acquisition comparison: real run completed, autonomous policy not promoted

## Decision

**The 96-slot execution/recovery package is complete; the overall autonomous
Agent goal is not.** On the frozen, retrieval-exposed development cohort, fixed
multiquery and deterministic workflow each obtain **14/32** correct official
task labels, versus autonomous **12/32**. Autonomous again selects **stop32/32**,
despite read, query and rerank being available at every gate. It delivers no
additional evidence. Equal tools/final-answer contracts remove those particular
old comparison confounds, but do not establish semantic grounding or feedback
benefit. Do not promote this policy or retry the cohort seeking a positive score.

The deterministic workflow is the provisional **development/demo baseline**:
its point task score equals fixed multiquery with fewer generator calls/tokens
and shorter measured slot time. This is not a statistical non-inferiority result,
production recommendation or proof of semantic citation correctness.

## Exact execution and qualification

- Actual execution source: `e826a4d2178983365c87184e259f7ad43ba7b17a`, unchanged
  after the accepted CPU review. Later closeout/replay code is not that model run.
- Source archive SHA: `84ad2420bea73079ad51f111ba5f70e5d56d8b0d2a74b2baab65859d635958f7`.
- Run: `fair-three-arm-20261003`, one model execution, one A100-SXM4-80GB,
  observed16 CPU/250GB RAM; eight-thread CPU limits, not an online service.
- Frozen [protocol](protocols/fair-three-arm-20261003.json) and
  [implemented comparison](FAIR_THREE_ARM_CPU_20261003.md): 32 tasks ×3 policies,
  shared initial claim/BM25 Top20 frame, 5,240 passages, shared read/query/RRF/
  rerank executor, generator, final answer/abstain contract and failure rules.
- Per-slot ceilings remain5 generations/5 tools including bootstrap/2 queries/
  2 validation repairs/8192 input/512 output tokens/120 seconds. Actual spend
  differs; budgets do not force equal calls or force autonomous acquisition.
- Generator: Qwen3-4B revision `350135a4de9a3407be836fa238cccc1d61503a85`;
  reranker: Qwen3-Reranker-4B revision `22e683669bc0f0bd69640a1354a6d0aebcfeede5`.
- Model/corpus/cohort/exposure hashes remain frozen in the accepted release.
  No training, sealed-test reading, new gold in prompts or answer masking.
- 230 validation IDs →187 connected components →exclude31 whole components
  touching consumed32 →156 eligible components →32 fixed SHA-ordered tasks.
  The parent validation population was already retrieval-exposed: **dev policy
  comparison, not independent validation/test**. Selection is label-blind, not a
  claim that the trusted metadata preparer never parsed annotated evidence.
- Original source54031fa/160-slot recovery stays unchanged. It uses a different
  roster/protocol; old8/23 vs14/23 is not a paired before/after for these32 tasks.
- Original stopped Pod could not restart due to GPU availability. User then
  authorized autonomous execution/migration and removed the per-package dollar
  cap. Fresh storage/assets/runtime receipts and observed provider allocation
  bind the new instance; migrated assets are copies, not independent experiments.
  There was no model run on the rejected old instance and no automatic retry.

## Complete quality comparison

Source: [redacted original-score projection](verified-runs/fair-three-arm-runpod-20261003.json).
All task, abstention and execution-failure denominators are32 per route.

| Metric | Fixed multiquery | Deterministic workflow | Autonomous |
|---|---:|---:|---:|
| Official task correct |14/32 (43.75%)|14/32 (43.75%)|12/32 (37.50%)|
| Task AND nonempty official-ID proxy |2/32|2/32|1/32|
| Official-ID concordance, all tasks |2/32|2/32|1/32|
| Answered / intentional abstention / execution failure |26/6/0|27/5/0|26/4/2|
| Recall@5, evidence-bearing N=13 |.52308|.50769|.37692|
| MRR@10, N=13 |.50000|.51538|.48291|
| nDCG@10, N=13 |.47697|.46111|.43088|
| Top5 Evidence F1, N=13 |.40006|.38468|.31557|

The citation proxy checks that every cited source ID belongs to official decisive
evidence. **It is not semantic support, citation completeness or entailment.**
Intentional NEI/conflict abstention may earn task correctness but zero nonempty
joint proxy; errors/repair exhaustion never earn correct NEI automatically.
There is no blind human evaluation, author entailment assessment or semantic
judge. True grounded joint correctness is unmeasured, not an invented zero.

Secondary retrieval metrics use final ranked Top20 candidates, including
preview-only/not-read sources, on13 official evidence-bearing tasks. They do not
measure full-text delivery, citation F1 or model use. No unsupported verdict-model
zero is presented as a classification result: this run did execute the verifier.

Frozen5,000 paired bootstrap resamples, seed20261003:

- Autonomous−fixed task difference−.0625,95% interval[−.15625,0]; joint-ID
  proxy−.03125,[−.09375,0]; Evidence F1−.08449,[−.21148,.02939].
- Autonomous−deterministic task difference−.0625,[−.1875,.0625]; joint-ID
  proxy−.03125,[−.09375,0]; Evidence F1−.06911,[−.16264,.02613].
- All secondary Recall/MRR/nDCG intervals and original scorer statistics remain
  in the linked JSON. No positive promotion or significant superiority claim;
  a zero-inclusive interval does not prove equivalence.

## Actual behavior and complete physical cost

[Gold-free CPU receipt census](verified-runs/fair-three-arm-behavior-20261003.json)
matches every saved model observation/schema to its reserved physical call and
checks committed sentence hashes in later actual observations. The original
cloud scorer additionally verified token/response/model/state identities before
reading scorer-only gold. CPU replay does not rerun or rescore the model.

| Physical measure, all32 per route | Fixed | Deterministic | Autonomous |
|---|---:|---:|---:|
| Generator calls, including plan/gate/repair/final |64|33|68|
| Input / output tokens |148442/4867|86479/930|132616/1308|
| Rerank requests / pairs |32/640|32/640|0/0|
| Rerank nonpadding tokens |84094|85062|0|
| Rerank time incl. swaps, seconds |257.581|241.850|0|
| All charged tools, including initial retrieval |160|160|32|
| Mean slot time, seconds |16.118|9.081|2.144|
| Nearest-rank slot P50 / P95, seconds |15.688/19.306|8.948/10.500|1.932/5.418|
| Unique extra-text deliveries received later |59|58|0|
| Added citable sentences across cohort |207|205|0|

Overall: **165 generator calls,367537 input/7105 output tokens;64 rerank requests,
1280 completed pairs,169156 rerank tokens;0 unknown generation usage/0 unknown
rerank requests**. Physical generation time341.567s and rerank time499.432s
include failures and swaps; no free gate, repair or rerank call. Mean/P50/P95 are
offline serial slot timings on this allocation, not HTTP SLA or load throughput.
Autonomous is faster because it stops without reranking; lower task/ID-proxy
quality forbids calling that a quality-preserving cost win.

All32 autonomous gates offered acquisition/query/rerank and unread sources.
There were32 validated stops,0 acquires,0 extra-text receipts and no post-stop
tools. Deterministic had one duplicate-citation validation error repaired by an
additional verdict call; autonomous had six such errors, two slots ended in
`validation_repair_exhausted`. All are retained in N=32 and charged. No actual
tool failures or empty-query events occurred; synthetic tests are not substituted
for missing real recovery cases. Prompt receipt is delivery opportunity, not
semantic utilization, and this whole-policy design does not isolate feedback
causality.

## Honest real cases and implementation decision

Cases are the first occurrence within each specified route/behavior in frozen
task order, **not selected by score or presented as the global first event**:

- Task position0/autonomous: actual gate stop →separate verifier →answered.
  This proves a stop path, not a reasonable-stop or grounded-answer success.
- Task position26/deterministic: duplicate sentence reference →paid repair →
  accepted terminal answer. Structural repair success is not semantic correctness.
- Task position19/autonomous: stop →verdict repairs exhausted; retained failure,
  no fallback answer, no uncharged retry, no selective removal.
- **Missing:** autonomous acquisition →new evidence →later model decision;
  evaluated reasonable stopping; real empty/failed-tool recovery. There is no
  synthetic substitute and no claim that more control-route text caused gains.

The implementation decision is to keep deterministic search/retrieval and
explicit answer/abstain validation as the current demo, and decline autonomous
promotion. The observed problem occurs at the acquisition-policy decision
boundary before additional evidence is executed/delivered, not a newly failed
Torch/cache/permission environment. These results cannot identify downstream
feedback utilization or relationship reasoning because no autonomous feedback
was acquired. A subsequent package needs a materially different decision
mechanism and newly frozen data/semantic evaluation qualification; repeating
this gate on the same tasks or forcing tools cannot validate autonomy.

## Recovery, stopping and money

- Worker exited0/reaped; operator completed after1009.526s; independent deadline
  observed operator returncode0/all children reaped after1034.052s. No partial
  compact was treated as success.
- Entire3557-file result/runtime journal archive was recovered privately before
  stopping;2072612bytes; remote/local SHA
  `b83f2d1f901a747d2fc5af107f8b0cf798c425cd7cb867cc8f6bd6309d271d5f`.
  Complete projection and authorized release are separately recovered. No model
  caches or gold are added to Git/OneDrive/career materials.
- Raw-run SHA `3c9eafaa159321bcc202d5f5f18ddcdd8548917dde7e894f3b19def44e1cb7c3`;
  original compact `9ab26752340eb3bb917ceff088d168a949b994ed33231356ee4d1798b6a3d6de`;
  cost `794f75702a8f25daf2db9cc5e4afe7db5c3a443b591fa51775cfd626e20c9281`.
- Stop verified before01:32:43 UTC: **compute/container Not running**. No Pod or
  volume permanently deleted. No Spartan, paid API, parallel model, recharge,
  Auto-Pay change, resume edit or further coordinator message.
- Full conservative paid interval starts00:57:39.569 UTC, not the model-worker
  start: about35.1min including official migration, receipt checks and recovery.
  The16.8min operator run is not the whole bill; the old15min ancillary estimate
  is not presented as met. Current quote compute1.59/h +container/volume.021/h
  +three other retained volumes.099/h gives approximately **US$1.00 upper** for
  this interval, not an invoice or per-route dollar measurement.
- Latest loaded billing: posted totals3.302 (10/2) +.058 (10/3) =**US$3.360**,
  still one-hour delayed, with this GPU charge unposted. Conservative prior3.60
  +this interval1.00 =approximately **US$4.60 cumulative upper at stop**; original
  US$20 is not reset, estimated headroom15.40 at that time is not an exact invoice
  or new authority. Subsequent idle fees reduce it. Account funds are separate.
- Four retained120GB volumes continue **US$.132/h ≈US$3.168/day**, including
  historical storage. Auto-Pay remains disabled. Permanent deletion requires
  confirmation and was not done.

## CPU reproduction, personal contribution and career boundary

```powershell
.venv-validation/Scripts/python.exe -m pytest -q tests/test_fair_saved_replay.py
.venv-validation/Scripts/python.exe -m ruff check scripts/replay_fair_acquisition_result.py tests/test_fair_saved_replay.py
.venv-validation/Scripts/python.exe -m mypy scripts/replay_fair_acquisition_result.py --follow-imports=silent
.venv-validation/Scripts/python.exe scripts/replay_fair_acquisition_result.py --run E:/Project/_climate_transfer/fair-three-arm-20261003/private-recovery/recovered/runs/fair-three-arm-20261003/inference/run.json --run-sha256 3c9eafaa159321bcc202d5f5f18ddcdd8548917dde7e894f3b19def44e1cb7c3 --output E:/Project/_climate_transfer/fair-three-arm-20261003/private-recovery/NEW_UNUSED_CENSUS_PATH.json
```

The new10 synthetic tests cover privacy, unique receipt counts, bad hash/schema/
delivery/future/post-stop/initial/roster rejection, route-specific case ordering
and error-detail redaction; they are implementation tests,
not new model effects. Accepted e826 runtime tests/clean checkout/Linux CI are
reused; only closeout/replay deltas are checked anew. The
[independent review and clean reproduction receipt](verified-runs/fair-three-arm-closeout-20261003.json)
binds reviewed HEAD `7e3e6bc42f8935b6455ed13a432128c4215c4a52`; no remaining
blocker. The route-specific first-case correction does not change model scores.

My contribution is grouped negative sampling/representation adaptation and
retrieval/latency selection, then shared typed acquisition, immutable evidence
contracts, physical feedback/cost journals, independent post-exit scoring and
recoverable experiment execution. The business problem is how to find sufficient
claim evidence and select a quality/cost profile, with the goal of reducing
unsupported judgments; this run has not measured that semantic outcome.
No production adoption, time saving, financial ROI, independent-test
Agent gain or causal feedback benefit is measured. Existing training/search
achievements remain separately scoped in the [application/STAR handoff](CLIMATE_APPLICATION_CASE_20261003.md).
