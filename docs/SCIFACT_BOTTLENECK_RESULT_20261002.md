# Evidence bottleneck: completed execution, rejected quality gate

**Decision: retain the strongest archived v2 top1 baseline. Do not promote A,
B or C, retrain, rerun or change the scoring rule in this package.** Job
`32006925` completed successfully; the negative result is a model/mechanism
result, not another missing-dependency or process-exit failure.

This is the same already-consumed, metadata-selected TRAIN24 (15 positive
claims, nine NEI claims, 18 relevant documents), with early preparation gold
exposure. It is neither an independent test nor an online A/B experiment.
Raw claim text, IDs, model responses and gold remain on Spartan. This closeout
made zero model calls and decoded no additional gold rows.

## Execution and evidence identity

| Item | Verified value |
|---|---|
| Executed source | `7cd8c44e97e7e9351259ff75ae98a60dd2369b9a` |
| Source archive SHA-256 | `91e00c01456f9e6fee9476662392cc7ad7963b4a16d82784ef66ec40aeecf89a` |
| Authorized release SHA-256 | `2dd7dd7173fc79b5ac1057b931dbd75b965409ef459c4c4e330aabe671799837` |
| Model SHA-256 | `d1dd9783afdf4e0fbd21eee824834d71b86982f5a5d5f6f371fe07f2f76f3cf6` |
| Terminal | `COMPLETED`, exit `0:0`, 2026-10-02 06:22:58–06:25:53 +10:00 |
| Allocation / measured use | One A100, 8 CPU, 32 GiB requested; 175 s elapsed, 160.123 s TotalCPU, 9,281,620 K MaxRSS |
| Parent recorded duration | 161.5066 s; worker and scorer exited/reaped with code zero |
| Accounting | All 72 route slots / 96 physical calls; zero technical failures or unknown-usage attempts |
| Compact SHA-256 | `bd215b0d256d6bdecde955802276383b6924855130616b166ecee3bdd213de98` |
| Original quality SHA-256 | `8bea959f1e1fbfe151ae49abe12cb1fa6d018c898e936dab25216f2978a88b8b` |
| Completion SHA-256 | `e1a27671e91eafee3c2139e21ec89bf73805c82b6589e1ed53599c0e54ad12dc` |

The [original compact](verified-runs/scifact-bottleneck-32006925.json) is a
byte-identical copy. The [redacted diagnosis](verified-runs/scifact-bottleneck-32006925-diagnosis.json)
contains ordinal cases and counts, not source IDs/text. Original compact,
quality, completion and all 24 per-case file hashes were identical before and
after read-only analysis. Executed source is not this later documentation commit.
The diagnosis SHA-256 is
`e86446aa77435ddba0f27cd738d94e24979f929f3b5f9243e840b202d8e69ad1`.
Local closeout checks reconciled all 24 cases with unchanged official counts,
72 slots, cost sums, B/C deltas and an exact redaction-field allowlist. Documentation
links and tracked secret/PII scanning passed. A separate read-only review verified
the baseline comparison, five cases and cost/quality interpretation. Runtime
code did not change; its accepted CI was reused rather than rerun locally.

## Frozen comparison and quality

A uses a single evidence-first response (schema field order is not proof of
reasoning order). B/C share one immutable sentence selection, restored to
original sentence order. B's label stage sees the selected sentences **and the
full original top1 document**; C sees only those same selected sentences.
Positive B/C citations are locked to that selection. B/C do not select
different evidence sets. Both archived baselines are from the identical ordered
24 claims, with no new baseline calls.

| Route | Abstract label F1 | Joint abstract/rationale F1 | Sentence selection F1 | Sentence label F1 | Strict positives /15 | NEI false evidence /9 | Document-level unresolved /24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Strongest archived v2 top1 | .6471 | **.5882** | .5641 | .4872 | 5 | 2 | 8 |
| Archived v3 top1 | .5000 | .3500 | .4051 | .2532 | 4 | 7 | 2 |
| A | .5238 | .4286 | .3765 | .3294 | 6 | 9 | 0 |
| B | .6207 | .4138 | .3951 | .3951 | 0 | 2 | 13 |
| C | .5185 | .2963 | .3836 | .3014 | 0 | 1 | 15 |

The predeclared C-vs-B gate requires at least one strict improvement, no decline
in label F1/joint F1/NEI false-evidence reduction, and no extra technical failure.
C reduces NEI false evidence by one but loses **.1022 label F1 and .1175 joint
F1**. The gate is `not_supported`. A's extra strict positive does not justify
promotion: its NEI false evidence is 9/9 and joint F1 is below v2.

`Strict positives` is the existing conservative whole-answer diagnostic, not
official accuracy: exact document-set agreement, correct labels, no cited
sentences outside the union of gold rationales, and at least one complete
alternative rationale among the first three predicted sentences per document.
Official joint F1 allows extra sentences; therefore nonzero joint F1 and zero
strict positives are compatible. Sentences outside annotated gold are not
automatically scientifically false. `INSUFFICIENT` remains unresolved at the
document level, not a verified global NEI prediction.

## Why B/C have zero strict positives

Read-only analysis reuses `select_complete_fit` for **only the same 24 IDs**,
`parse_prediction`, and the unchanged `strict_whole_answer`. It compares stored
selections with original alternative rationale sets, without changing outputs
or scores. Correct label and first-three rationale counts were independently
reconciled to the stored official scorer totals (A: 11/9, B: 9/6, C: 7/4).

- **Retrieval reachability:** top1 is a gold document for 14/15 positive claims.
  Two positive claims require two or three gold documents; any top1-only route
  cannot meet exact whole-answer document coverage on them.
- **Selection:** 12/15 positive selections contain a complete top1 rationale
  somewhere; only 10/15 contain one in their first three original-order slots.
  Thirteen positive selections include at least one non-gold sentence. Reading
  context and final citations were coupled by the frozen contract.
- **B:** emits nine positive-claim answers, all with the correct document label,
  but **all nine include extra non-gold citations**; three additionally miss a
  complete first-three rationale and two have an incomplete document set.
  Six positive claims are unresolved; four already have a complete shared
  rationale. This is not merely a wrong-label problem.
- **C:** emits eight positive-claim answers; seven have the right label and all
  eight include extras. Seven positive claims are unresolved, including five
  with a complete shared rationale. One previously correct B label flips wrong.

Blocker counts overlap and must not be summed as disjoint failures. Removing
full text changes only three B/C outputs: one useful positive becomes
INSUFFICIENT (`case_10`), one correct refutation becomes support (`case_18`),
and one NEI false-positive answer becomes unresolved (`case_15`). These changes
explain the full 9→7 correct-label, 6→4 rationalized-document and 11→9 predicted-
document counts. No evidence-selection improvement occurred between B and C.

### Five illustrative redacted cases

Cases are selected after observing failures; they are explanations, not a new
evaluation sample. Their ordinal names correspond to the frozen original order.

| Case | Observed stored behavior | Diagnostic implication |
|---|---|---|
| 05 | B/C share five sentences: three annotated, two extra; both label correctly and cover a complete first-three rationale. Official joint credit passes, strict whole-answer fails. | Context copied into citations explains a strict failure without a label failure. |
| 08 | Both selected sentences exactly cover a gold rationale, yet B/C return INSUFFICIENT. A returns a strict-correct answer. | Evidence is available; label-stage over-abstention remains. |
| 09 | Eight selected sentences contain a complete rationale, but five are extras and the first three do not complete it. B/C labels are correct; A is strict-correct. | More selected context is not necessarily better final evidence; original-order/first-three scoring matters. |
| 10 | Identical four selected sentences include a complete rationale. B correctly refutes; C becomes INSUFFICIENT. | Masking full text loses a useful prediction despite identical selected evidence. |
| 18 | Identical four selected sentences include a complete rationale and one extra. B correctly refutes; C incorrectly supports. | C's regression is not only abstention; label direction also changes. |

## Cost: physical run versus independent deployment

All values below are measured call-ledger costs, **not online latency, SLA or
financial savings**. Currency cost is unknown (`null`), not zero. Historical
shared retrieval/preparation is unchanged and not recharged or represented as
newly measured end-to-end serving work.

| Physical stage | Calls | Input tokens | Output tokens | Ledger elapsed sum (s) |
|---|---:|---:|---:|---:|
| A | 24 | 55,086 | 1,810 | 53.446 |
| Shared selector | 24 | 16,487 | 717 | 20.173 |
| B label | 24 | 20,335 | 334 | 10.156 |
| C label | 24 | 10,416 | 336 | 9.714 |
| **Actual physical run** | **96** | **102,324** | **3,197** | **93.489** |

| Independent deployment | Calls | Total tokens | Ledger elapsed sum (s) |
|---|---:|---:|---:|
| Archived v2 top1 | 24 | 36,604 | 28.152 |
| Archived v3 top1 | 24 | 56,059 | 54.698 |
| A | 24 | 56,896 | 53.446 |
| Shared selector + B | 48 | 37,873 | 30.329 |
| Shared selector + C | 48 | 27,956 | 29.887 |

Do not sum the independent deployments to report physical consumption: that
would charge the same selector twice. C uses 26.18% fewer tokens than B, but
only 1.46% less summed ledger time and materially worse quality. Compared with
v2, C uses 23.63% fewer tokens but twice the calls, 6.17% more ledger time and
joint F1 .2963 versus .5882. This is not a quality/cost win. Model-load, parent,
scorer and scheduler timing are separate from summed generation costs.

## Submission review and single next decision

The exact execution source already passed [Linux CI](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36918087972):
1,449 tests, Ruff, strict source mypy, dependency and secret checks. Its synthetic
integration exercises production runner → exit/reap → accounting → scorer →
compact, including 72-slot failure denominators, 96-call success, POSIX timeout
and signal transitions. The real terminal run now also verifies this chain in
Spartan's actual environment. The older tokenizer-only preflight is not called
a whole-chain test. **Do not reopen accepted execution or submit another job
merely to repeat these checks.** Earlier failed jobs remain recorded unchanged.

For future changed executions, review data/visibility/scorer contracts together
before queuing; reuse accepted evidence for unchanged components. A change in
interpreter, pinned dependencies, CLI handoff or serialization needs a matching
environment/whole-chain fixture check, not just unit tests or `sbatch --test-only`
(which checks scheduler acceptance, not Python correctness). Scientific gains
cannot be guaranteed by code review.

**Only recommended next hypothesis:** abandon further full-text masking/field-
order tuning, retain v2 as the quality baseline, and separately test supervised
minimal-evidence selection that distinguishes readable context from final direct
citations. Keep the verdict baseline fixed so selection effects are identifiable;
require a preregistered quality/cost comparison before any promotion. This is a
new substantive hypothesis, not permission to train or infer in this package.
It may address excess citations, but does not by itself solve the demonstrated
label/abstention errors. This consumed TRAIN24 can diagnose it, not validate
independent generalisation. No resume change or Agent-gain claim follows here.
