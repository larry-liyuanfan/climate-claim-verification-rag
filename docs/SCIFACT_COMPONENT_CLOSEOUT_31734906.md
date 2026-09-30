# Terminal component diagnostic: valid synthetic abstention, gate stopped

Job 31734906 is Slurm COMPLETED `0:0`, not a completed quality experiment.
It ran 2026-10-01 01:45:16–01:46:52 +10:00, 96 seconds, with batch TotalCPU
85.444 seconds and MaxRSS 10,346,536 KiB. Application status is
`stopped_no_semantic_effect_claim`. Source remains `81917f2`, not the documentation
HEAD. [Original compact](verified-runs/scifact-component-compact-31734906.json)
and [physical audit](verified-runs/scifact-component-physical-closeout-31734906.json)
are hash-bound; all four preflight and 33 formal slot states were reconciled.

## What actually happened

The first **explicit synthetic** screening claim says compound X improves Y.
Document 1 repeats that statement; document 10 describes compound Z's colour.
Both titles say `Explicit synthetic fixture, not real scientific evidence`.
The frozen system asks for scientific supporting/contradicting evidence and
permits abstention when no document is sufficient.

| Item | Verified observation |
| --- | --- |
| Frozen expected | `{"decision": "select", "document_ids": [1]}` |
| Actual raw | `{"decision": "abstain", "document_ids": []}` |
| Technical response | canonical parse valid; physical receipts complete; EOS reached; no grammar log, truncation, timeout or unknown cost |
| Calls and tokens | 1 synthetic call; 277 input / 15 output tokens |
| Remaining preflights | 3 unattempted, not 3 incorrect model answers |
| Formal diagnostics | screening 12, relation 12, oracle rationale 9: all 33 unattempted |

The worker passed all frozen packing checks, loaded the real model and generated
once. Model load took 15,100.121 ms; generation 1,085.542 ms; peak allocated GPU
memory 8,173,072,384 bytes. These are one-job diagnostics, not online latency/SLA.
Inference exit proof was persisted before the scorer opened targets. This CPU
audit did not read scoring targets or export real claims, gold or raw predictions.

Original compact's `task_success_all_planned=0` reflects an incomplete execution,
**not observed model accuracy**; `accuracy_given_valid=null` for all formal strata.
Its valid-only screening detail has zero counts because no screening was run.
Frozen candidate coverage remains **6/9 gold documents** from Stage A, with three
candidate gaps; neither zero evaluated rows nor this synthetic abstention changes it.

## CPU decoder evidence and limits

The [real-tokenizer branch probe](verified-runs/scifact-component-preflight-branch-20261001.json)
matches the exact 277-token prompt hash. Expected active and observed abstention
outputs both reach EOS, with decision-first and document-IDs-first object orders:
four accepted paths. At `{"decision": "`, seven legal next tokens are
`a, s, se, ab, select, abs, sel`; both `select` (1742) and `ab` (370) are allowed.
The observed text re-tokenizes to 14 tokens, compatible with 15 reported tokens
including EOS; the actual runtime token IDs/logits were **not saved**, so this is
not an actual-token trace or a measured branch probability.

Provider source uses greedy `do_sample=False`, one beam and the LMFE prefix mask.
The matching transformers 4.51.3 processor adds zero to permitted logits and
negative infinity to disallowed logits; no custom branch-positive bias or output
rewrite was found. Generation config's sampling temperature/top-p/top-k warnings
do not enable sampling. Tested active-path reachability rules out those paths
being impossible; it does not prove the grammar has no probability/segmentation
effects or explain the model's preference.

The title/system tension is a plausible ambiguity, **not an experimentally proven
cause**. One valid abstention is insufficient to diagnose general screening,
relation or rationale ability. No weights were loaded by the CPU probe; no model
call, new dataset or GPU job occurred. A typing-only annotation was corrected;
the same four synthetic paths passed twice on CPU. Ruff and strict mypy on the
new probe pass. The unchanged 151-test execution suite was not rerun.

## Definite stop mechanism and bounded next proposal

Both `execute_matrix` and independent `component_audit.aggregate` require
`valid AND prediction == expected` for a preflight to pass. Thus a semantic
disagreement blocks a diagnostic whose purpose is to measure semantic failures.
The old protocol behaved as written; this is **not another infrastructure fault**.
Do not edit the old target, relabel r2 as passed or replay it under its old release.

A separately versioned, explicitly post-observation protocol can distinguish
`technical_ready` from `semantic_match`: a valid wrong answer continues while
remaining a negative semantic measurement. Keep identity/packing/decoder,
provider/grammar, synthetic schema failure, deadline/EOS/budget, unknown-cost,
incomplete-receipt and persistence stop conditions. Formal schema failures remain
paid failures, without retries. Executor and independent scorer must both enforce
the new version; old records keep the old semantics.

Preferred carry-over: authenticate **only r2 preflight 0**, including exact source,
archive, run/identity/exit, original wire/response/physical receipts, decoder,
packing and complete usage. Reference it read-only with `carried=true`, without
calling the provider or writing the old directory; semantic match stays false.
Then at most three remaining synthetic plus 33 formal calls: **36 new, 1 carried,
37 cumulative**. New/carried/total calls, tokens and latency must be separate.
If that source chain cannot be verified, fail closed; do not silently run a new
37 and call it the same budget. This document is not model-run authorization.

## Cost ledger retained, no resume claim

| Attempt | Allocated A100-job seconds | Batch CPU seconds | Generation calls | Tokens input/output |
| --- | ---: | ---: | ---: | ---: |
| v1 / 31729507, pre-model packing failure | 81 | 68.866 | 0 | 0 / 0 |
| r2 / 31734906, valid synthetic semantic mismatch | 96 | 85.444 | 1 | 277 / 15 |
| Cumulative | **177** | **154.310** | **1** | **277 / 15** |

Allocation time is not measured active GPU kernel time. Two job peaks must not
be summed as simultaneous memory. The [resource receipt](verified-runs/scifact-component-resources-31734906.json)
retains per-job values. Both original directories/locks and the negative result
remain intact. This is already-consumed TRAIN diagnostic evidence, not independent
test, training, online A/B, Agent benefit, or a new resume result.
