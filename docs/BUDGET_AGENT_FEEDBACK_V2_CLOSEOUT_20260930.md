# Feedback-v2 pilot closeout — negative development result

Job31587302 completed with exit0:0 on2026-09-30,13:06:38–13:11:55+10:00.
The requested single A100/8CPU/32GiB allocation took317 seconds (0.0881 allocated
GPU-hours). Host MaxRSS was18,192,244KiB; TotalCPU300.972s. Neither value measures
GPU memory or GPU utilization. Operator elapsed312.436s excludes wrapper time.
Runtime preflight passed without generation; the pilot then loaded models and
generated33 outputs for18 task/route slots. No retry, duplicate job, new monitor,
model change, training, frozen-test access, deployment or resume update occurred.

Exact source:`d7d5bf3b2fd79b90f1d47218bbe88adb991910cb`; protocol SHA
`fd9f25ae9e64bfe0c0c5bfb8c73a88a1884267da70c01d8e896abc81b4e112d3`.
The [submission receipt](verified-runs/budget-agent-feedback-v2-submission-20260930.json)
retains the initial pending state and provisional scheduler estimates, not a
claim that those estimates were actual starts. Final raw output256,095bytes and
33 private response files remain on Spartan. Only content-free aggregates ship.

## Observed result, not an independent-quality improvement

The6 authored questions were previously exposed development diagnostics, not
official gold. There is no retrieval/semantic gold score or bootstrap here.

| Route | Slots | Generations | Final abstain | Repair exhausted | Accepted answer | Model-selected tools |
|---|---:|---:|---:|---:|---:|---:|
| Fixed retrieval | 6 | 11 | 5 | 1 | 0 | 0 |
| Fixed4B rerank | 6 | 11 | 5 | 1 | 0 | 0 |
| Adaptive | 6 | 11 | 5 | 1 | 0 | 0 |

Across33 attempts,19 parsed decisions were schema-valid, but4 answer decisions
then failed citation checks. The remaining15 were model abstentions.5 attempts
failed schema (all exceeded3 statements), and9 were malformed JSON at the512
output-token ceiling with no EOS. No output contained a thinking tag;
`enable_thinking=False` was set. These observations do not establish that greedy
decoding caused the failures or that increasing the budget would fix them.

Every route's first responses were3 answers before validation,2 abstentions,
and1 malformed JSON. Adaptive had all4 actions available on5 rows; its empty
retrieval row could abstain or rewrite. Thus the controller did not lock out all
tool actions.15/18 rows had5 full-context documents;3 had no context. Visibility
and lexical coverage do not establish that the claim is semantically supported.

All4 rejected parsed answers cited a retrieved **preview-only** document. Their
12 total citations included8 full-context and4 preview-only references, with0
outside the candidate set. Preview text was explicitly noncitable; rejection is
correct, not grounds for relaxing the checker. A future bounded read/open tool
would have to load full evidence before citation and have an extra-work control.

13 of15 abstention reasons exactly reproduced the prompt's example reason;
9 rows abstained after feedback. This supports investigating template copying
and repair-to-abstention behavior, **not a causal claim** that the prompt alone
caused the failures. Do not count an abstain-shaped response as semantic success.

## Cost and provenance

105,632 input and7,745 output tokens were consumed. The14 schema/JSON-invalid
attempts alone used47,388 input and6,432 output tokens; all are retained in the
ledger. The4 citation-rejected attempts are additional failed work, not included
in that schema/JSON-only subtotal. No unknown usage or budget violations were
reported. Request-time P50/P95 were7.20/33.46s retrieval,11.81/34.75s fixed rerank,
and7.25/33.29s adaptive (6 observations each). These are serial development
request timings, not online SLA, stable production percentiles or an A/B test.

- [Frozen v2 compact](verified-runs/budget-agent-feedback-v2-31587302.json), SHA
  `f5b986ebf528fa11669e22d8ff852a2fd21cdcb6b9a5340d402998cdd201c8a9`.
- Private run SHA:`2e1445d88e64f69fa3a0472fdd678e6f1e490669fd531f8e502ce4f48d45df2b`.
- `scripts/diagnose_feedback_pilot.py` reproduces the content-free failure shapes
  from that private run and its hash-matched raw files; it never exports text.
- [Reproduced private-file diagnosis](verified-runs/budget-agent-feedback-v2-diagnostics-31587302.json)
  SHA:`48839e1c48448d124910d55366fbd3470fb362e0a854be0a552bf5da8ad4d0b2`.

Decision: **do not release a full external comparison**. A separately reviewed
v3 development contract may address output length, redundant examples and
evidence access. Keep models and sampling fixed while testing those changes.
The old v1 and this v2 result/scorers remain unchanged.
