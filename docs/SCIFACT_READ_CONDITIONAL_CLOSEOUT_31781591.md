# Three scripted-read continuations: limited grounding, not Agent gain

## Outcome

Job **31781591** completed with exit `0:0`. Exactly three previously frozen
post-read states received one generation each: **3/3 correct document labels,
1/3 complete rationales**, and **zero model tool proposals/executions**. Correct
rationale counts are **FIT-overlap 0/1; not-direct-FIT 1/2**. All cases are exposed
TRAIN diagnostics; the two not-direct-FIT cases are not independent test cases.

All three answers selected the document already chosen by the scripted/oracle
read. The labels therefore describe conditional relation classification, **not
document-retrieval capability**. The initial reads were executed by a scripted
teacher; a replay event's `model_selected` field does not make it model behavior.

| Same three previously consumed claims | Original adaptive episode | Post-scripted-read conditional generation |
| --- | ---: | ---: |
| Document label correct | 0/3 | 3/3 |
| Document with complete rationale | 0/3 | 1/3 |
| FIT-overlap complete rationale | 0/1 | 0/1 |
| Not-direct-FIT complete rationale | 0/2 | 1/2 |
| Model-selected tools executed | 0 | 0 |

This is an input-intervention diagnostic, not an online A/B, retrieval gain,
independent-test improvement or autonomous Agent improvement. The script gives
the conditional run a selected evidence document that the original model did
not choose to read. It is not a same-input estimate of a new policy's effect.

## Anonymous failure mechanisms

Only mechanism summaries are public; no claim IDs/text, document IDs, gold
sentence indices or private per-case records are exported.

| Anonymous case | FIT overlap | Complete rationale after read | Mechanism |
| --- | --- | --- | --- |
| A | No direct overlap | Yes | Selected the correct document label and one complete annotated alternative. |
| B | No direct overlap | No | Correct document/label, but the chosen sentence has **zero overlap** with any annotated rationale. |
| C | Yes | No | Correct document/label, but again **zero overlap** with any annotated rationale. |

Rechecking the unchanged packed states against only the already-consumed gold
confirmed complete first-three-compatible rationale reachability in **all three**
post-read states, versus none in the original initial states. Every case has
at least one complete **single-sentence** alternative. The two failures are
therefore **wrong annotated-sentence selection**, not partial rationale selection,
context truncation, wrong labels or the scorer's first-three truncation.

All outputs contain one sentence, but this does not show that missing multi-sentence
capacity caused these particular failures. Likewise there is no basis to call
it a universal first-sentence lock: the coordinator's independent raw audit
observed the first visible sentence in only one of three outputs. The rule is
to match one complete official alternative, not the union of every alternative.

The complete-rationale document F1 is 1/3; sentence-selection/label F1 is 0.20.
Their denominators differ: the official sentence score counts annotated
alternative sentences. These are not claim-verdict accuracy or free-text
entailment scores. The separate conservative whole-answer diagnostic is 1/3.

## Physical integrity and resource accounting

The CPU closeout rehashed **12** indexed physical files and checked three new
raw-response/grammar receipts plus the original three adaptive raw receipts.
It revalidated the frozen tokenizer, actual observation/schema/prompt/token
bindings, raw→parse→render→prediction equality, one reservation per completed
call, unchanged old run/score hashes, and **144 checkpoint tensors / 72 active,
unmerged, frozen LoRA layers**. The actual original score was not overwritten.

Worker exit/reap precedes persisted costs and scoring. The auditor then used
only the same three consumed gold claims to verify metrics, subgroup results,
and evidence reachability. The audit made no model calls and did not consume
unused validation/dev/test. This is a scoped posthoc diagnostic, not blind review.

| Measured item | Result |
| --- | ---: |
| Actual generation calls / input tokens / output tokens | 3 / 6,044 / 90 |
| Generation time summed over calls | 3.904 s |
| Repair / warmup / reranker / model-tool execution | 0 / 0 / 0 / 0 |
| Unknown usage or unresolved calls | 0 |
| Slurm allocation elapsed | 76 s |
| Slurm TotalCPU | 64.313 CPU-s |
| Batch MaxRSS | 17,164,576 KiB |

Allocation elapsed includes staging/import/loading/scoring; it is not active
GPU-kernel time. These three sequential diagnostic calls are not a serving
latency benchmark or online SLA. There is no measured API bill, so token usage
is not presented as financial cost or savings.

## Reproduction and immutable receipts

- Execution source: `fc4ffd61a7615682911556838c66f47aa33d0293`.
- Exact release and single submission: [submission record](verified-runs/scifact-read-continuation-submission-31781591.json).
- Redacted physical audit: [compact result](verified-runs/scifact-read-conditional-closeout-31781591.json).
- CPU-only fixed-run auditor: `scripts/audit_scifact_read_continuation_closeout.py`.
  It requires the exact project-local inputs/runtime, makes no model call and
  refuses to overwrite an existing closeout directory. This is not a one-click
  public-data download or permission to rerun the GPU job.
- Remote compact SHA: `64fac97a50f75bc2d778265993ac8f3e489991fdae7ecb2e55ba89ab07ba07bd`.
  The pretty-printed public JSON has identical content, not identical bytes.
- Original score SHA: `1dea7d5dcec2bbd2d766d4b378f5ff3f887879202f5ca2d2c7470dfc403ecb1b`.
- Original run SHA: `b7ea2e1c50e3bdc8a965477284e205007f668c9039806403bc53df4e00e71281`.

## Next decision, not a new execution

There is **limited conditional capacity, with weak grounding**. The existing
[state-supervision proposal](SCIFACT_OBSERVATION_SUPERVISION_PROPOSAL_20261001.md)
remains a proposal: focus on selecting valid evidence sentences and justified
abstention in real packed contexts, then consider state-conditioned read targets
alongside already-sufficient/no-read controls. Do not substitute a target tool
count for answer quality, and do not automatically start new training or consume
reserved evaluation data. No current resume, shared career file or other project
was modified by this closeout.
