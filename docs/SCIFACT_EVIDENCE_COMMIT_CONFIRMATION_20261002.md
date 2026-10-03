# Evidence-commit confirmation: exercised selection, bounded quality/cost tradeoff

## Decision

One authorized three-arm run completed on 2026-10-02 (Australia/Sydney, UTC+10).
Source-bound verifier references and unchanged-label assembly recovered complete
positive answers that the previous free terminal rewrite did not retain.
The adaptive route exercised document selection and feedback-conditioned
continuation, but it is **not generally better than the strong top1 baseline**:
one additional complete positive answer came with lower abstract-rationalized
F1 and substantially more tokens. Preserve the implementation and all negative
cases; do not promote a model, change the resume or automatically rerun.

The data are the same already-exposed TRAIN24: 15 evidence-bearing claims and
nine NEI. These results are not independent-test performance, a new training
result, a causal Agent study, or an online SLA. This is a new interface comparison,
not a same-prompt single-factor repair of the earlier direct-answer protocol.

## Execution and provenance

- Actual job **31944832**, COMPLETED / `0:0`; 00:43:50–00:49:02 UTC+10,
  **312 allocation seconds**, TotalCPU **294.600 s**, batch MaxRSS **9,310,224 KiB**.
- Source `2ab219bb13564790b004797354dcc89554da2bf6`; docs-only `daf61d6` was not
  repackaged. Source archive SHA `e05642eef2d7348b1439d2a86089c3982ab3c59fd7315b00f5ca7a17509833cf`;
  actual wrapper SHA `d30e1a2ebe2cc99630d776f2b9d3d365f1843d37a36a293dc6a0592c1895a815`.
- Authorized release SHA `0df679bb6ef9b58f875435b242d16de82541863e7bd13e52cf4804eecc14e556`.
  Test-only **31944831** was a simulation, not an extra executed GPU job.
- One A100, eight CPUs, 32 GiB RAM, 30 GiB scratch, 40-minute ceiling;
  normal QoS, Nice0, Requeue0. No automatic retry, model change or training.
- Actual dependency consumer, archive/shell guard and exclusive submission lock
  passed before submission. Initial-frame SHA and runtime SHA match the prior
  accepted run. CPU checks were reused, not rerun to create redundant evidence.
- **72/72 results**, **204/240 physical generations**, zero missing episodes,
  unknown token receipts, unknown backend timings or unassigned calls.
  Physical audit and process reap preceded existing official/strict scoring.
- [Redacted compact](verified-runs/scifact-evidence-commit-confirmation-31944832.json)
  SHA `b442f56873281804b3be2f252bf5ec4f44752520374969c56434bebeba81fd80`.
  Its reporting v2 adds paired counts and five illustrative cases without
  rerunning inference/scoring; the first compact remains preserved remotely.
  Raw gold, text, per-case responses and original predictions remain on Spartan.

## Complete three-arm result

| Outcome | Fixed top1 | Fixed all (first four) | Adaptive |
|---|---:|---:|---:|
| Strict complete positive answers | 3/15 | 3/15 | 4/15 |
| Correct explicit model NEI | 0/9 | 0/9 | 4/9 |
| Strict whole-answer total | 3/24 | 3/24 | 8/24 |
| Unresolved | 8 | 4 | 0 |
| Commit terminals | 16 | 20 | 19 |
| Model abstention terminals | 0 | 0 | 5 |
| NEI claims with false evidence | 3/9 | 5/9 | 5/9 |

Fixed empty-positive or over-limit sets are **unresolved**, not correct NEI;
this was frozen before the run. The four additional correct NEI decisions
therefore partly reflect the adaptive route's explicit abstention affordance,
not an equal-affordance proof of superior reasoning. Adaptive's five abstentions
include one positive miss. Zero unresolved does not mean zero wrong answers.

Paired positive outcomes against **each** baseline are: three jointly correct,
one adaptive-only correct, no baseline-only correct, eleven jointly incorrect.
Do not present the five-answer total difference as five extra grounded positive
answers or infer a general gain from this exposed, small diagnostic.

| Existing official scorer F1 | Fixed top1 | Fixed all | Adaptive |
|---|---:|---:|---:|
| Abstract label only | 0.6061 | 0.4615 | 0.5556 |
| Abstract rationalized | **0.4848** | 0.3846 | 0.4444 |
| Sentence label | **0.3659** | 0.2338 | 0.3409 |
| Sentence selection | 0.4634 | 0.3117 | **0.4773** |

Adaptive improves all four F1s over fixed-all, with higher precision but lower
recall. Against top1, three F1s are lower. These metric names retain their official
definitions; they are not a manufactured claim-level verdict accuracy or a
generic single "Evidence F1". Free-text entailment and claim-verdict accuracy
remain unmeasured (`null`), not zero-valued classification results.

## Full measured cost, not just calls

| Cost over 24 claims | Fixed top1 | Fixed all | Adaptive |
|---|---:|---:|---:|
| Physical generations | 24 | 96 | 84 |
| Input tokens | 41,137 | 156,761 | 219,774 |
| Output tokens | 892 | 3,244 | 3,171 |
| Total tokens | 42,029 | 160,005 | 222,945 |
| Model-backend generation seconds | 29.272 | 99.088 | 102.238 |

Adaptive uses **12.5% fewer generations than fixed-all**, but **39.3% more total
tokens and 3.2% more backend time**. Relative to top1 it uses **3.50× calls,
5.30× tokens and 3.49× backend time**. Fewer calls are not measured cost savings.
Across all arms, 417,672 input and 7,307 output tokens are accounted for.
Currency cost remains `null`; allocation time and summed backend time are not
per-request online latency. Historical retrieval was replayed, not retimed.

## What the actual trajectory shows

- **30 valid model verify proposals → 30 executed, valid verifier calls**.
  First selection differed from `c0` in **8/24** episodes. The verifier is
  fallible; a valid call is not a correct scientific judgment.
- **Six episodes** made another verify decision after INSUFFICIENT feedback.
  Four then abstained; two then committed following a positive verifier label.
  "Continue" is a derived action sequence, not a separate controller action.
  This observes feedback use, not the causal quality benefit of that use.
- **19 commits / five model abstentions**. Verification is required by the
  commit interface, so its occurrence does not prove spontaneous tool demand.
- **No episode issued multiple positive refs; at most one legal commit option
  appeared in an actual plan**. The subset-selection capability is implemented,
  but selection among competing positive subsets was not exercised here.
  The demonstrated variation is primarily which document to verify and whether
  to verify again, commit its sole positive ref, or abstain.

## Five paired, pseudonymous cases

The identifiers below are positions in the frozen 24-claim sequence, not raw
claims. Cases were selected after scoring to illustrate mechanisms and failures;
they are not a separately selected evaluation set. `c0`–`c4` are per-case source
aliases. No raw gold or scientific text is exported.

| Case | Observed paired behavior | Defensible interpretation |
|---|---|---|
| `case_16` | Top1 commits `c0`/REFUTES/four sentences; all commits `c0,c1,c2`; both fail the whole-answer test. Adaptive chooses `c1`, verifies REFUTES/two sentences and commits it unchanged; correct. | The sole additional positive answer is grounded in a real source choice. It is not selection among multiple issued positive refs or proof of general retrieval improvement. |
| `case_14` | All three retain `c0`/SUPPORTS/two sentences and are correct; adaptive needs plan→verify→commit versus one top1 call. | Immutable assembly preserves a useful verifier decision; the Agent adds cost without improving this case. |
| `case_07` | Top1 gets INSUFFICIENT and is unresolved; all commits `c1`/REFUTES and is wrong. Adaptive verifies `c0`, then `c4`, gets INSUFFICIENT twice, and explicitly abstains correctly. | Real continuation and abstention, with the fixed unresolved/NEI distinction preserved. |
| `case_02` | First four verifications are INSUFFICIENT, so both fixed routes are unresolved. Adaptive chooses `c4`, receives REFUTES/two sentences, and commits incorrectly. | Source binding prevents rewriting, not verifier hallucination; looking beyond top four can hurt. |
| `case_09` | Adaptive verifies `c0`, then `c1`, receives INSUFFICIENT twice, then abstains incorrectly. All commits `c2`/SUPPORTS/three sentences but also fails strict whole-answer scoring. | Repeated insufficient feedback and valid control flow are not proof that no usable evidence exists. No route resolves this case completely. |

## Reproduction and closeout boundary

The frozen worker/operator/scorer/package entries are listed in the
[protocol document](SCIFACT_EVIDENCE_COMMIT_20261002.md#execution-and-preflight).
The exact release, exclusive submission-lock receipts and authoritative results
are under the private Climate stage/run directories on Spartan. Reproducing a
new model run needs separate authorization; this confirmation is consumed.
The report's 6,573-token preflight number is only the observed maximum for the
24-question probes (first two references and predefined feedback), not a strict
bound across all reachable prompts. Frozen model inputs were not changed.

Only report/README/redacted evidence files change after this run; no new GPU,
training, protected split, default-branch integration, service deployment or
resume edit is included. The earlier 686-file temporary copy is still preserved
because deletion was tool-policy blocked; it was not deleted by another route.
The overall goal of robust, independently demonstrated Agent quality remains
open despite completion of this bounded confirmation.
