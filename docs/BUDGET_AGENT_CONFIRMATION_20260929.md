# Frozen protocol confirmation: negative result, auditable execution

Job31520350 completed, but the explicit action/query prompt did not solve model
contract compliance:2 of9 responses passed structural and mechanical citation /
number checks;7 failed schema validation. This is not semantic accuracy, a
retrieval improvement, a production Agent or an independent-test result.
The same three authored pilot questions were reused, with no evidence gold.

## Identity and privacy

The [compact receipt](verified-runs/budget-agent-protocol-confirm-31520350.json)
contains the complete same3-task ×3-route matrix, counters, safe diagnostic
enums and SHA identities, with no original responses, answers or source prose.
Its bytes match the remote compact SHA-256
`0d9ca572d1976e842fb317761bf6fcad3d52448bfb93ebcd05262a0666da0973`.

- Inference Git: `72eaa90567e3603a3b940edffbb21b684bf1d1ba` (unchanged).
- Frozen scorer Git: `208ff931badff270cbbc9593c8ab54f5c79aec9e` (separate identity).
- Result archive:4,823bytes; SHA
  `4934332558a2b7abddb7467279f819ca93a04e44738202dc51b64947134a842f`.
- Static system prompt SHA:
  `6b0163a4ef9da624e35dffb51fff34913ccb8a90db395c8dfec68bbe409febaa`.
- Verified whole input bundle SHA:
  `563738f0be1f7bf7b99b8de20bcdeab552f7e893166dec260b8f7f1e7951c3c1`.

The allocated CPU audit streamed the complete16,122,255,360-byte input archive
hash, checked the result/source/prompt/protocol/model identities, and read only
the public evidence/protocol members, not model weights. All5,240 corpus IDs and
the frozen9-slot matrix were checked. Accepted answers were independently replayed
through the unchanged strict validator and quote/number checker. Hash multisets
of the owner-only original decoded strings matched all9 diagnostic events.
The full result and decoded response files remain on Spartan. Static prompt
identity does not fingerprint each dynamic observation or the tokenizer chat template.

## Outcomes and executed work

| Route (3 tasks each) | Legal answer proposals / mechanically accepted | Model abstentions | Failed schema | Retrieval calls | Fixed rerank calls / pairs | Generation input / output tokens | P50 / P95 seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_retrieval |0 /0|0|3|3|0 /0|3,836 /753|7.088 /11.059|
| fixed_rerank |1 /1|0|2|3|3 /40|3,936 /837|8.574 /11.634|
| adaptive |1 /1|0|2|3|0 /0|3,847 /747|5.655 /10.888|

Both accepted answers belong to `pilot-ice`. `pilot-multi` and `pilot-empty`
failed all three routes. All9 responses were parseable JSON, reached EOS and
did not hit max-new-token truncation. Replaying the untouched private originals
found the same first cross-field violation in all7 failures:
`query_only_allowed_for_rewrite`. No JSON repair, coercion, field deletion,
validator relaxation, new prompt or inference rerun was used in this closeout.

The recorded `controller_failure`/`stage_failed` category means the controller
caught invalid model output here; it is **not evidence of an infrastructure
crash or scoring-runner defect**. The model did not follow the explicit contract.
Later cross-field violations may be masked by the first failing validator.
There were zero legal rewrite/rerank proposals and zero model-initiated tool
executions. The controller executed9 initial retrievals and3 fixed rerank calls
(one empty call, zero pairs). Thus the12 tool calls must not be advertised as
model-selected multi-step reasoning. The two legal answer actions executed and
passed mechanical checks, but entailment/semantic support remains unmeasured.

All9 generations have complete token accounting:11,619 input and2,337 output
tokens; unknown rows0 and partial recorded tokens0. These are generation-only
counts, excluding reranker work. No currency conversion, API/GPU cost saving,
official quality, classification zero, bootstrap gain or abstention accuracy
is inferred. Three latency samples per route are descriptive offline timings,
not an online tail-latency SLA.

## Resource receipt and CPU audit recovery

Inference31520350:COMPLETED0:0,137seconds,125.423CPU-seconds;
batch MaxRSS18,031,200KiB (=17.195892GiB). The allocation was one full A100,
8CPU/32GiB and30GiB scratch with a15-minute walltime. GPU peak VRAM was not
measured. Per-query walltimes sum to62.139seconds; generation diagnostics sum
to60.815seconds. The remaining74.861seconds of job walltime is unallocated
overhead in this accounting, not a precise model-loading measurement.

Audit helper Git74cf4e0 has SHA
`52e9e98d9b8eeec974c9f14a73b614b435eae8985b521a8e9c1d1abb6dce29f7`.
CPU job31537753 failed before result scoring (5seconds,41,868KiB) because the
wrapper replaced the cluster dependency search path. The minimum wrapper-only
fix e7618aa preserves module-provided PYTHONPATH after the pinned Pydantic/source
paths. It does not change the frozen scorer or inference. Two synthetic audit
tests and Ruff passed; bash syntax and `sbatch --test-only` passed before the
single CPU replacement31537871. Replacement:COMPLETED0:0,20seconds,
15.529CPU-seconds,71,568KiB MaxRSS;1CPU/2GiB/1GiB scratch/5minutes, no GPU.
The replacement wrapper SHA is
`fce8b8870f298dd49be3f77863c66b5bce04a64814c3887c7864db3308440828`.

Reproduction is intentionally remote-only: use the frozen scorer archive and
`hpc/budget_agent_confirmation_audit.sbatch` under an allocated CPU job, with
the receipt's exact source/input/result objects. The helper refuses to overwrite
its compact destination. Do not retrieve or publish the full run archive to
reproduce the score, and do not resubmit an already completed audit merely to
change its output path. Fresh reproduction needs a separate scoped destination.

## Decision for the existing32-validation +8-vNext comparison

**Technically suitable for a fixed-protocol, failure-inclusive offline
comparison; not a passed Agent-quality gate.** No result-integrity/runtime defect
was found in this audit. Negative compliance is a model result and is not a
reason to revise tasks/prompts until a positive number appears. If separately
released, keep72eaa90,208ff93, the existing inputs/budgets/model hashes and all
failed rows. Reused32-task validation has24 evidence-bearing tasks; report that
denominator separately from32 calls per route. The8 authored vNext tasks have
no official gold/quality metric. Neither is a newly independent frozen test.

The pilot exercised only one generation per route. Adaptive rewrite/retrieval,
adaptive reranking, long-input memory and three-generation trajectories remain
unexercised by a real model. Their unit tests are not real-model efficacy.
Do not interpret this as proof that those branches fail or succeed.

| Controller ceiling | 32 validation | 8 authored vNext | Combined |
|---|---:|---:|---:|
| Route records |96|24|120|
| Generation attempts |160|40|200|
| New generation tokens |81,920|20,480|102,400|
| Tool calls |192|48|240|
| Retrieval calls |128|32|160|
| Rerank calls / attempted pairs |64 /1,280|16 /320|80 /1,600|

For scheduling discussion only: the observed generation rate is about38.43
output tokens/s (includes prefill/generation timing, not a pure decoding
benchmark). Dividing the102,400-token ceiling by this rate gives44.4minutes,
**not a prediction or guaranteed upper bound**. A provisional2-hour job budget
allows roughly2× that generation time plus40× the measured1.2-second
rerank work (about48seconds) and startup/I/O allowance. Larger contexts,
new adaptive routes, cold I/O and different GPU state can exceed this estimate.
The120-second query guard is soft and cannot establish a strict walltime bound.
The Slurm limit is the hard stop; an incomplete matrix must fail scoring.

Retain the measured full-A100/8CPU/32GiB/30GiB request as a provisional resource
shape, not a measured full-workload minimum or a VRAM guarantee. The observed
CPU MaxRSS is about54% of32GiB, but long-input GPU peaks were not measured.
No MIG fit, memory saving or full completion promise follows from this pilot.
Any full run needs coordinator authorization, a free permitted GPU slot and
`sbatch --test-only`; none was submitted here. Other projects/jobs were untouched.
