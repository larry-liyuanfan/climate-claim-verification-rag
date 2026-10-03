# SciFact train diagnostic r2: complete, no Agent-quality promotion

Job **31645005** completed successfully on 2026-09-30, but this is an
execution result, **not evidence of improved autonomous search**. All 12 claims
and four routes produced a scored slot. Most attempted answers failed strict
validation; all routes correctly labeled and rationalized only one of nine
gold documents. Adaptive never requested a tool, including in its unparsed
wire responses. No useful tool-feedback behavior was therefore observed.

This closes the separately authorized retry, not a new experiment. The failed
startup job 31620529, both submission locks, original logs and immutable r2
run/score remain preserved. No new GPU, prompt replay, dev/test opening,
resubmission, model change or resume change was made for this closeout.

## Scope, provenance and independent arithmetic

The [frozen preparation](SCIFACT_TRAIN_DIAGNOSTIC_CPU_20260930.md) deliberately
selected **12 distinct eligible-train components**, three per gold-aware
opportunity stratum. The original corpus has 5,183 documents. This is a biased
train mechanism diagnostic, not representative evaluation, held-out test,
online A/B or evidence of generalization. Foundation-model training exposure
is unknown; original 300-dev remains unreleased. The public Climate frozen test
remains retired. Neither that track nor the restricted 154-query development
track is mixed into these numbers.

Frozen executable Git: `80fcd07faed28890b70096386c5897f0343d3104`.
The [submission receipt](verified-runs/scifact-train-r2-submission-31645005.json)
binds source, wrapper, model bundles, protocol and query order. Its pending
snapshot is historical, not the current status. Later local Torch/typing changes
and the separate evidence-gap candidate were **not** in the executed archive.

The operator ran inference without gold, waited for child exit, then extracted
the selected train scoring bundle in a separate step. This is dataflow/scratch
isolation, **not an OS security boundary against the shared account**. Train-gold
selection means the sample is still not independent testing.

The [read-only audit](../scripts/audit_scifact_train_closeout.py) checks the exact
48-slot matrix, component uniqueness, frozen input/run/score/preflight hashes,
all actual-visible sentence and source hashes, per-attempt costs, route budget
equality and sampling-versus-actual first adaptive observation. Without importing
the scorer, it recomputes all four metric counts/F1 overall and per stratum, and
checks P50/P95 arithmetic. Counts match the saved scorer; no score was changed.
This is an independent implementation of arithmetic, not an external adjudicator
or a new human-label review. The existing frozen scorer also checked prediction
conversion and missing/unlinked trace failures before emitting its report.

Only the [content-free aggregate](verified-runs/scifact-train-r2-closeout-31645005.json)
is copied locally. Raw claims, selected gold, model responses, document IDs,
predictions and traces stay on Spartan. The final audit script was supplied
byte-for-byte over SSH stdin; the remote runner hashed those received bytes
before execution. `audit_script_sha256` matches the local reviewed file hash,
also recorded in the separate resource receipt. The audit was rerun after
review fixes; this is not an old aggregate attributed to newer code.

## Quality: actual gold scoring, not format acceptance

All values below are points on the same 12 selected claims. There are nine gold
documents and fourteen gold rationale sentences, plus three NEI claims. Original
SciFact document rationalization requires a complete alternative rationale in
the first three predicted sentences of that document, with its correct label.
Sentence metrics retain the original alternative-rationale denominator; partial
pieces of different alternatives cannot be combined into a correct rationale.

| Route | Abstract label / rationalized F1 | Sentence selection / labeled F1 | Correct/predicted/gold documents | Correct/predicted/gold sentences |
|---|---:|---:|---:|---:|
| Fixed retrieval | .1053 / .1053 | .0333 / .0333 | 1 / 10 / 9 | 1 / 46 / 14 |
| Fixed 4B rerank | .0833 / .0833 | .0290 / .0290 | 1 / 15 / 9 | 1 / 55 / 14 |
| Deterministic extra retrieval + rerank | .1053 / .1053 | .0370 / .0370 | 1 / 10 / 9 | 1 / 40 / 14 |
| Adaptive | .0952 / .0952 | .0294 / .0294 | 1 / 12 / 9 | 1 / 54 / 14 |

Every route produces false evidence for **1/3 NEI claims**. That is a separate
diagnostic, not claim-verdict accuracy. No unsupported zero classification
metric is substituted: claim-verdict accuracy and free-text entailment remain
unmeasured. No bootstrap, confidence interval, p-value or population-gain claim
is appropriate for this predeclared tiny, biased diagnostic.

| Route | Mechanically accepted answer | Genuine model abstention | Repair exhausted |
|---|---:|---:|---:|
| Fixed retrieval | 3 | 3 | 6 |
| Fixed rerank | 3 | 1 | 8 |
| Deterministic extra | 2 | 1 | 9 |
| Adaptive | 3 | 4 | 5 |

Normal failures stay in the denominator as empty predictions. Mechanical
acceptance proves citation identity and structural validity, not semantic truth.
Reranking the right document into context did not solve answer serialization
and evidence selection; advertising only a retrieval hit would hide this failure.

## What the model actually did

Across 107 train generation attempts, 98 wire responses chose `answer` and nine
chose `abstain`. Every response was fully stored, untruncated and recovered by its receipt SHA
on Spartan, including all 87 unsuccessful attempts. The wire top-level `action`
was first in **107/107** JSON objects. Consequently, this run does **not** support
the proposed `documents`-before-`action` branch-lock explanation.

The saved scorer's `actual_tools.proposed` uses the controller's action field,
which is assigned only after parsing. That field alone cannot rule out malformed
tool intent. The extra wire audit addresses that limitation: adaptive's 24 wire
actions were 20 answers and four abstentions, including 17 invalid answers;
there were no recoverable read/rewrite/rerank intents, no successfully parsed
tool decisions, no actual model-selected tool events and no post-tool attempts.

All adaptive attempts offered read/rewrite/rerank. Fixed routes performed their
prescribed retrieval/rerank operations, which must not be counted as autonomous
model-selected tools. Their rerank cost was 240 pairs each for fixed-rerank and
deterministic-extra. A tool being offered is not proof of a useful opportunity.

Strict-parser rejection counts:

| Route | Total sentence budget exceeded | Duplicate/excess documents |
|---|---:|---:|
| Fixed retrieval | 15 | 3 |
| Fixed rerank | 22 | 3 |
| Deterministic extra | 27 | 0 |
| Adaptive | 14 | 3 |

The grammar/schema bounds each document independently; the stricter parser
additionally enforces a cross-document total of twenty sentences and uniqueness.
These rejections are therefore compatible with valid JSON and per-document
grammar. The strict check prevented invalid answers from being released; it
was not a tool-disabling code path. Whether a revised representation or trained
policy would help remains a hypothesis, not an inference from this result.

There were 59 attempts after validation feedback: 57 repeated the preceding
error category, and 33 repeated its full response hash. Only two next attempts
became valid decisions. **Validation-repair feedback is not retrieval-tool
feedback**; this does not demonstrate learning from newly acquired evidence.

## Opportunity taxonomy: avoid a false causal story

The three claims in each stratum have one gold document each except NEI. Frozen
read witnesses are offline, gold-aware diagnostic probes; they were never given
to the real model. Alias/source hashes and actual first adaptive prompt counts
match the sampling state, so the same-budget witnesses are applicable here.

| Stratum | Actual adaptive initial state | Observed outcome and legitimate interpretation |
|---|---|---|
| Initial document opportunity, 3 | Complete original text of all three gold documents visible; each has a legal first-three rationale | One accepted answer, two repair exhaustions. These are not premature semantic abstentions or token-packing failures. |
| Top20 replenishable, 3 | Gold in candidate pool but outside requested context; each has a matching legal single-document read witness | One abstention, two repair exhaustions, no read. This proves an available acquisition path, not that the model perceived the correct preview or would answer correctly after reading. |
| Gold absent Top20, 3 | All gold documents absent from candidates and initial evidence | One accepted but non-gold answer, one abstention, one repair exhaustion. An unknown rewrite counterfactual remains; do not label this “no possible useful tool.” |
| NEI, 3 | No annotated supporting/refuting evidence | Two abstentions, one false-evidence answer. More tools are not inherently useful here. |

Fixed rerank and deterministic-extra brought all three replenishable gold
documents, **all their original sentences**, into actual first context; each
route still exhausted repairs on all three. This supports a distinction between
retrieval opportunity and valid evidence-bearing output, not a causal benefit
of autonomous retrieval. No gold document requested in the audited first
contexts was partially hidden by token packing. Initial adaptive preview payload
was nonempty, and prompts were below the 8,192-token cap; however, per-preview ID
lists were not saved in this trace. We cannot certify which individual gold
preview was shown, or claim the model attended to it. No prompt reconstruction
was run for this closeout.

The “useful feedback followed by a wrong action” category is **unobservable**, not
zero-error: there were no model-selected tool events. Zero calls does not prove
the model cannot use tool feedback. It only describes this frozen policy/model
on this biased sample. Zero-shot prompting also does not reproduce Search-R1's
answer-reward RL/retrieval-token loss mask or Adaptive-RAG's trained routing.

## Full charged cost and resource boundary

| Route | Calls | Input / output tokens, including failures | Whole-question P50 / P95 seconds |
|---|---:|---:|---:|
| Fixed retrieval | 24 | 87,906 / 4,670 | 10.349 / 23.366 |
| Fixed rerank | 29 | 110,515 / 6,953 | 23.159 / 27.104 |
| Deterministic extra | 30 | 106,293 / 7,410 | 21.784 / 27.476 |
| Adaptive | 24 | 92,738 / 4,448 | 10.669 / 22.359 |

Train subtotal: **397,452 input + 23,481 output tokens**, with zero unknown-usage
attempts. Separate real-model synthetic preflight: four calls, 2,720 input +238
output tokens, 7.301 seconds. Model load 21.478 seconds, reranker load 1.687
seconds and BM25 build .882 seconds are separate, not silently excluded and
represented as zero cost. There is no API billing or currency saving measured.
These tiny offline sequential latency points are not an online SLA or throughput.

Slurm: COMPLETED/0:0, 19:06:12–19:20:14 +10, Elapsed 14:02, TotalCPU 13:47.173,
batch MaxRSS 18,527,100K, allocated one A100 / eight CPUs /32G. The GPU was
allocated for 842 seconds (0.2339 allocated GPU-hours); active GPU utilization and
peak GPU memory were not measured. Requested walltime was a two-hour ceiling,
not billed runtime or an estimate of active computation.

Private diagnostics were counted once per phase/slot/attempt/kind, not deduplicated
by hash: 107 train response receipts (50,667 bytes) and four preflight receipts
(596 bytes); matching grammar receipts attempted zero bytes. All receipts report
zero missing, truncation, dropped bytes and I/O failure. Full train response
bytes were SHA-matched on the remote host; duplicate outputs still count as
separate charged attempts. The multiset of all111 physical response files and
51,263 bytes matches train plus preflight receipt multiplicities; identical
response hashes cannot hide a missing duplicate file. All four preflight
responses were also byte-verified, without imposing the train-only action schema
on its different synthetic probes. Controller `slot-raw.json` is not treated as a backup
for model response text. This result did not exhaust the shared diagnostics cap.

## Reproduce this audit, not the model run

From the isolated checkout in PowerShell, using the existing Iris SSH alias:

```powershell
Get-Content -Raw scripts/audit_scifact_train_closeout.py | ssh -o BatchMode=yes -o ConnectTimeout=15 spartan-trip "python3 - --root /data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2"
```

This reads only the selected frozen bundles and small completed artifacts and
prints an allowlisted aggregate. It performs no writes on Spartan, no dataset
transformation, tokenizer/model inference or scheduling. Audit failures suppress
private exception values. Sixteen synthetic audit tests cover raw-action ambiguity,
truncation/unknown recovery, repeated receipt accounting, independent first-three
scoring, unpredicted invalid gold, content-hash collisions, top-level arrays versus
objects, physical-file multiplicity and failed/unknown cost accounting. The short
stdin command above reproduces numeric checks but does not bind transport bytes;
the release audit additionally hashes exact stdin bytes before Python execution.

**Decision:** preserve the negative result; do not promote Agent/resume claims or
open official dev. Any future mechanism experiment needs a separately frozen,
gold-free common-visible-context control, charged schema/output overhead and
separate authorization. The independent evidence-gap candidate is not this run;
its changed visible-context budget prevents attributing package differences
solely to a reflection field. Stop after this audited handoff; no new monitor.
