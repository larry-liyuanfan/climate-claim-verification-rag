# Targeted retrieval feedback: completed negative result

## Decision

Job **32030221** completed, but the **adaptive route is not promoted**. Both frozen
quality comparisons and the overall efficiency gate failed. This is a replay of
32 already-consumed CLIMATE validation claims, not a fresh test, online experiment,
or pure causal estimate of feedback. The result identifies a controller-activation
problem; it does not establish that targeted retrieval cannot help.

[Safe compact + diagnostics](verified-runs/targeted-feedback-32030221.json) retains
all five routes, original frozen scores, physical costs and five redacted cases.
The [CPU preparation](TARGETED_RETRIEVAL_FEEDBACK_CPU_20261002.md) is historical;
its unauthorized draft was later separately authorized for this exact run.

## Run identity and denominators

- Runtime source: `5004fa01a4d47f4ba7e73be8e33ec436bd06edf4`.
- Authorized release SHA: `43ebd04a56a48a39d1b379064cb60516f46c766e19e2e03d66fd124fe796d5c7`.
- 5,240 evidence records; **32 tasks × 5 routes = 160 planned and finished slots**,
  original task order preserved. No new sample selection, training or inference
  was performed during closeout.
- The historical selection called 24 tasks “decisive”, but the actual frozen gold
  contains **17 SUPPORTS, 6 REFUTES, 1 DISPUTED, 8 NOT_ENOUGH_INFO**. Therefore:
  **retrieval denominator = 24**, **binary-label denominator = 23**, and
  **cost/failure denominator = 32 per route**. The DISPUTED task remains in the
  matrix and retrieval/cost results; it is not silently changed into a binary label.
  The frozen scorer already used this rule. Neither gold nor scoring was changed.
- `COMPLETED / 0:0`, 2026-10-02 **14:26:51–14:46:23 Australia/Sydney**;
  allocation elapsed **19m32s**, operator **1159.913s**. Batch MaxRSS
  **33,530,392 KiB (~31.98 GiB)**; recorded TotalCPU **19m30.751s**.
  One A100 was allocated (~0.326 allocated GPU-hours), not a GPU-utilization measure.
  Reserved RAM was 32 GiB, so memory headroom was small; this is not a throughput SLA.

## Five-route quality (unchanged frozen scoring)

| Route | Recall@5 | MRR@10 | nDCG@10 | Top-5 Evidence F1 | Binary labels correct / 23 |
|---|---:|---:|---:|---:|---:|
| fixed_retrieval | .4319 | .4042 | .3876 | .2578 | 10 / 23 |
| fixed_rerank | .4403 | .4719 | .4460 | .2563 | 12 / 23 |
| deterministic_extra | .4403 | .4719 | .4462 | .2563 | 10 / 23 |
| fixed_multiquery | .4299 | .4753 | .4423 | .2470 | 15 / 23 |
| adaptive | .4319 | .4042 | .3876 | .2578 | 12 / 23 |

The field `delivered_evidence_ids` in the historical run is the final ranked
candidate list, **not proof that all these sources were read by the verifier**.
Recall/F1 use its Top-5; MRR/nDCG use its Top-10. Full-text visibility below is
measured independently from the actual `current_citable` observations.

Citation-ID F1 means gold-source-ID overlap, not sentence entailment: respective
means are **.2158 / .2331 / .1915 / .2837 / .3116**. Adaptive's higher descriptive
ID-overlap mean is not an independently significant semantic-verification gain.
The frozen `nondecisive_answers` field actually counts answers on tasks with empty
gold evidence sets (2/4/4/5/4), not all non-binary labels or proven hallucinations.

### Paired quality uncertainty

The original **5,000 paired bootstrap** intervals (95%, seed 20260929) are
adaptive minus the comparator. Retrieval pairs = 24; binary-label pairs = 23.

| Metric | vs fixed_rerank: difference [95% CI] | vs fixed_multiquery: difference [95% CI] |
|---|---|---|
| Recall@5 | −.0083 [−.1403, .1215] | +.0021 [−.1285, .1292] |
| MRR@10 | −.0677 [−.1944, .0434] | −.0712 [−.2000, .0392] |
| nDCG@10 | −.0585 [−.1712, .0325] | −.0547 [−.1660, .0361] |
| Evidence F1 | +.0016 [−.0650, .0747] | +.0108 [−.0528, .0816] |
| Binary-label accuracy | .0000 [−.2174, .2174] | −.1304 [−.3478, .0870] |

All intervals cross zero: this establishes neither superiority nor equivalence.
The preregistered gate required positive lower bounds for **both F1 and accuracy**
against **both controls**, with no negative mean retrieval deltas. It fails;
MRR/nDCG means also decline. The cohort is small and repeatedly inspected.

## Full physical cost and failure accounting

| Route (32 slots each) | Generator calls | Generator tokens, total / mean | Reranker pairs / tokens | Failures | P50 / P95 seconds |
|---|---:|---:|---:|---:|---:|
| fixed_retrieval | 32 | 52,560 / 1642.50 | 0 / 0 | 0 | .765 / .981 |
| fixed_rerank | 32 | 48,441 / 1513.78 | 640 / 84,663 | 0 | 9.093 / 9.420 |
| deterministic_extra | 32 | 49,838 / 1557.44 | 640 / 84,543 | 0 | 8.905 / 9.429 |
| fixed_multiquery | 64 | 66,757 / 2086.16 | 640 / 83,523 | 0 | 13.944 / 15.140 |
| adaptive | 34 | 63,992 / 1999.75 | 0 / 0 | 1 | .980 / 1.033 |

Total **194 generator calls**, **274,366 input + 7,222 output tokens**;
**96 reranker requests / 1,920 completed pairs / 252,729 nonpadding tokens**.
Unknown usage attempts and unknown reranker requests are zero. Generator elapsed
was **276.610s**, reranker elapsed including serial model swaps **791.349s**.
These stage times differ from allocation and operator wall time. API currency cost
is unavailable, not zero. Local serial episode latency is not online service P95.

Adaptive returns 23 answers, 8 intentional abstentions, and **1 execution exhaustion**.
The exhausted task is included in the 32-slot cost/failure denominator even though
its label is NEI. No denominator is restricted to successful requests.

Post-run exploratory paired cost intervals (5,000 resamples, 32 pairs) do not
replace the frozen cost gate:

| Adaptive minus control | Generator tokens/request [95% CI] | Elapsed seconds/request [95% CI] |
|---|---|---|
| fixed_rerank | +485.97 [342.72, 758.56] | −8.160 [−8.387, −7.972] |
| fixed_multiquery | −86.41 [−237.13, 192.10] | −12.920 [−13.214, −12.626] |

Adaptive's reranker work is zero, explaining much of its shorter latency.
Against fixed_rerank it consumes more generator tokens and has more failures;
against fixed_multiquery it has more failures and its token reduction is uncertain.
**Both cost gates fail**. Shorter episodes without successful adaptive tool use
are not a quality-preserving efficiency win.

## What the actual traces did

All 32 adaptive first prompts permitted targeted `rewrite`; none executed it.
**23 answered immediately; 8 abstained immediately; 1 attempted the same already
selected `read` three times, each rejected as `read_loop`, then exhausted repairs.**
Thus 34 model calls are not 34 successful decisions. There were exactly 32 forced
initial retrieves, zero additional executed read/rewrite/rerank actions, and **zero
new-query → executed-tool → new-feedback → next-decision chains**. This is absence
of policy activation, not evidence that the retrieval executor failed.

| Route | Initial Top-20 has gold (tasks) | Gold text actually shown | Gold ID finally cited | Additional queries | New candidate occurrences / tasks with new gold |
|---|---:|---:|---:|---:|---:|
| fixed_retrieval | 16 | 13 | 9 | 0 | 0 / 0 |
| fixed_rerank | 16 | 13 | 10 | 0 | 0 / 0 |
| deterministic_extra | 16 | 13 | 8 | 64 | 346 / 0 |
| fixed_multiquery | 16 | 14 | 13 | 64 | 726 / 2 |
| adaptive | 16 | 13 | 12 | 0 | 0 / 0 |

Counts are tasks out of 32 (only 24 have gold evidence), except explicitly marked
candidate occurrences. New IDs are not new relevant evidence. The fixed-multiquery
plan successfully generated two queries per task and they executed before the final
decision, but this is an **upfront fixed plan**, not adaptive feedback use.
Within adaptive, three tasks had gold among candidates but not actual displayed
text; four answered with a wrong binary label despite displayed gold; one abstained
despite displayed gold. ID provenance alone does not establish correct inference.

## Five redacted cases

Examples are the first unused task in original order satisfying each declared
diagnostic category; category counts are reported, with no quality-based resorting.
They explain the full-cohort result, not a replacement evaluation. Exact task/slot
lookup is in Spartan's private `closeout-cpu-final-20261002/private-case-map.json`;
no claim, query, evidence text or original IDs are exported.

1. **Candidate present, not read (3 matching tasks).** Gold was in Top-20 but not
   displayed/cited. Adaptive immediately returned the correct binary label without
   a gold-ID citation. Fixed multiquery also did so despite two extra queries and
   14 new candidate occurrences. Correct label alone is not grounded success.
2. **Visible/cited gold, wrong verdict (4 matching tasks).** Adaptive saw and cited
   a gold source but returned the wrong label. Fixed multiquery also erred despite
   28 new candidates. This is a label-decision failure, not a missing-ID problem.
3. **Visible gold, unused (1 matching task).** Adaptive abstained immediately.
   Fixed multiquery answered correctly and cited gold, with no newly discovered
   gold source. Different prompt/control/call histories preclude attributing this
   contrast solely to additional retrieval or feedback.
4. **Repeated already-visible read (1 task).** On an NEI task, all three proposed
   `read` source lists equalled the current selection. No new tool ran, no new
   context arrived; all three attempts and their cost remain in the exhausted slot.
   Fixed multiquery answered despite no gold evidence, so it is not a clean rescue.
5. **Successful label and gold-ID overlap (7 matching tasks).** Adaptive answered
   correctly with gold-ID overlap after initial retrieval. Both controls also
   succeeded on this example. This shows baseline verification can work; it is not
   a demonstrated semantic-entailment or adaptive-search success.

## One next mechanism hypothesis (not implemented or authorized here)

**Separate the information-acquisition decision from the final verdict action.**
Hypothesis: competing terminal answer/abstain branches lead the free-action policy
to stop before acquiring useful evidence. Test a bounded two-stage `stop/acquire`
controller: acquisition selects one unseen preview source or one targeted query;
the existing verifier then judges the unchanged claim from actual returned text.
Already displayed read sets must be excluded from acquisition choices. Stopping
must remain legal—do not force every request to make a useless tool call.

A future separately authorized development comparison should retain the same
cohort/corpus/model/budgets/controls and test whether genuine acquisition chains
increase **and** gold-visible/cited joint correctness improves without failing
the original full-cost gate. If calls increase but relevance/joint correctness do
not, reject the hypothesis. No result here licenses a new frozen-test evaluation,
training run, model replacement, or independent/generalization claim. Additional
diagnostics alone are not a project-quality improvement.

## Reproduction and evidence custody

Run `scripts/summarize_targeted_32030221.py --run <original-run-directory>
--gold <original-frozen-gold> --output <new-private-closeout-directory>` on Spartan
with the existing numeric runtime. It verifies the four original artifact hashes,
terminal/reaped worker and 160-slot matrix before CPU-only analysis. It exports a
count-only aggregate; the case-to-task/slot map stays remote. Synthetic tests check
candidate-vs-visible-vs-cited separation, read-loop accounting, privacy and paired
cost statistics. No unchanged full suite was rerun manually.

Original artifact SHA-256 identities:

- operator: `d483a1426fdbeb445c2e676f77183a257dfffb07bde2a0634afaff620b448ed8`
- cost: `1bc867a1eea41bb7013423405dad5cbf7b0c6da3102c09d67b9834dd6795fa6d`
- frozen compact: `a07721e8f5a9cf477b2f3052dc65bfc64ebd7da8642f212af593c15ad8c3f19b`
- private run: `958244b0aeeae90d56b05686ea26bceda098067f843906d982aca5677646e7ae`
- exported diagnostics on Spartan: `60d6ee25d36c0aec54f55516760ee2feea033df0d075ffe0e414420794add8fa`

The original source/gold/output hashes are retained; the reporting commit is a
different identity from the executed source. No resume/shared career file was
edited, no model was rerun, and no merge or deployment was performed.
