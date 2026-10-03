# Paired verifier confirmation: negative result, scores recovered without inference

Decision: **do not promote relation/minimal-rationale v3**. A more structured
response did not improve overall evidence verification and costs more. This is
the previously consumed conditional TRAIN24 diagnostic, not a fresh independent
test, online A/B, training gain or causal feedback experiment. No resume edit.

## Execution versus scoring recovery

| Item | Immutable identity / observation |
|---|---|
| Original GPU job | 31980221; FAILED/2:0; 762 s; MaxRSS 9,269,536K |
| Executed source | `5a12fd9f398afb1af772111253600e68410e885a` |
| Authorized release SHA | `a09e0488d2b2bcc1fee05ea242f3820b91eadc44c126121f48f4007c1b80d96e` |
| Model outcomes | Both workers exited 0/reaped; 144/144 slots; 392 calls; no unknown usage |
| Original scoring | Both `physical_or_feedback_identity_failed`; gold_read=false; files retained |
| Repair/replay source | `1f6f920682e97ea17c1ded6b86348a538a7020fe` |
| Replay archive SHA | `662557cd21ca039903dc8f0f7979bb5558aad19009712b141a65da425af12a06` |
| CPU replay | 31988504; COMPLETED/0:0; 49 s; MaxRSS 868,868K; zero model calls |
| Input preservation | Original before/after file manifests exactly equal; no record reconstructed |
| Source-tree evidence | 684 files matched archive after execution; receipt `704fb859e2692c23b8b9536bb6296c33cb5fffaa7600367e8495e4c068eaea6f` |

The source-tree receipt is explicitly a **post-execution read-only verification**,
not proof that the wrapper checked the complete tree before this run. The future
CPU entrypoint now calls the existing `verify_source_tree` before scoring. No
cancel/requeue or scoring repeat was used to add this evidence.

The precise failed assertion was `paired_effective_generation_binding`. The
released `expanded_model_defaults` represented `typical_p`, `epsilon_cutoff`,
`eta_cutoff`, `diversity_penalty`, `repetition_penalty`,
`encoder_repetition_penalty`, and `length_penalty` as integer JSON numbers instead
of equivalent floats. The released/recorded contract identity was `05c52688…1f68`;
the locally reconstructed object identity was `cabb4011…b384`. Values and key
order agreed. Actual loaded defaults, effective settings, schema, parser and
record-to-diagnostics identities passed.

The repair uses the already hash-bound release contract as the serialization
identity authority, still requiring semantic equality with the frozen contract,
saved-object identity equality, and unchanged physical configuration checks.
It does **not** normalize prompts, edit old receipts, erase the failure, change
gold, relax missing-record checks or change original scoring mathematics.

Regression coverage now includes both protocols' complete generate-fixture →
JSON-on-disk → audit paths with integer-valued JSON defaults, rejection of a
different self-reported contract, and separate-output success/failure tests that
hash every original file before/after. Exact repair
[CI 36907952703](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36907952703)
passed 1,400 tests, Ruff, strict typing and secret/PII scanning. Earlier tests had
covered in-memory defaults but missed the final authorized JSON representation;
passing a suite was not sufficient to rule out this boundary bug.

## Original six-cell quality and cost

All arms have 15 evidence-bearing and nine NEI claims. F1 below is the original
**abstract-rationalized micro-F1**, not a renamed general Evidence F1.

| Protocol / arm | Strict positive | NEI correct | Unresolved | Abstract-rationalized F1 | Physical calls | Input + output tokens |
|---|---:|---:|---:|---:|---:|---:|
| v2 fixed_top1 | 5/15 | 0/9 | 8 | 0.5882 | 24 | 35,702 + 902 |
| v2 fixed_all | 3/15 | 0/9 | 7 | 0.4906 | 96 | 141,752 + 3,120 |
| v2 adaptive | 5/15 | 6/9 | 0 | 0.5556 | 80 | 187,183 + 3,066 |
| v3 fixed_top1 | 4/15 | 0/9 | 2 | 0.3500 | 24 | 54,222 + 1,837 |
| v3 fixed_all | 2/15 | 0/9 | 1 | 0.2750 | 96 | 215,712 + 7,721 |
| v3 adaptive | 5/15 | 2/9 | 0 | 0.4000 | 72 | 196,026 + 4,091 |

The [compact artifact](verified-runs/scifact-paired-31980221-cpu-replay.json)
retains correct/predicted/relevant counts and P/R/F1 for all four original metric
families: abstract_label_only, abstract_rationalized, sentence_selection and
sentence_label. No averaging of per-case F1 or replacement denominator is used.

- Fixed-top1 v3−v2: two positive wins, three losses; fixed-all: zero wins, one loss.
- Within v2, adaptive−top1 has zero positive wins/losses and **5.20×** tokens.
  Adaptive−all gains two positives, but all was already a weaker comparator.
- Within v3, adaptive−top1 gains one positive (one win/no loss), with **3.57×**
  tokens; adaptive−all gains three. This does not surpass v2 top1's five positives
  or its higher original micro-F1 at much lower cost.
- Cross-version adaptive has three positive wins and three losses. Its F1 falls
  0.5556→0.4000 and NEI correct falls 6→2. This is the verifier-plus-feedback joint
  system delta, not an isolated feedback causal estimate.

Physical totals are **830,597 input + 20,737 output = 851,334 tokens**, 392 calls,
zero unknown token usage and zero unknown backend timings. Model API currency
cost remains `null` for this local GPU run: no fabricated monetary ROI.
Shared model extraction/preparation was 22.364 s, once. The two load/input checks
were 16.290/16.705 s and phase times 262.792/445.433 s; these overlap other timing
categories and must not be added together. Historical retrieval/packing's
31.534 s is listed once, separately, not counted twice as a new run cost.
Allocation used one A100 for 762 wall seconds; CPU replay used two CPU/8 GiB for
49 s. None of these aggregate timings is an online latency SLA.

## Observed tool feedback, not assumed autonomy

v2 adaptive proposed/executed 28 verifications; all 28 passed the response
contract. It continued after feedback five times, all following insufficient
evidence. v3 proposed/executed 24, with 22 valid verifications; one continuation
after feedback, none after a valid insufficient response. Fixed v3 had 22/24
valid top1 and 74/96 valid all verifications. A model response passing generation
syntax does not guarantee passing the stricter semantic tool contract.

The five-call adaptive budget still permits at most two document verifications.
The retained three-gold-document case remains structurally unreachable under the
custom whole-answer requirement; it is not dropped to improve the result.
Unannotated evidence is not thereby adjudicated scientifically false. Fixed empty
outputs remain unresolved, whereas adaptive may explicitly abstain; the NEI
counts therefore do not establish a fair classification improvement.

## Five redacted paired cases

These are deterministically chosen illustrations, not a new evaluation subset.
The compact contains every selected case's six cells, actual source aliases,
labels/sentence counts, proposed→executed→feedback→terminal trace, calls, issued
positive refs and legal commit-option counts. No original IDs, source text or raw
responses leave Spartan.

| Case | Observed comparison | Interpretation |
|---|---|---|
| case_10 | v2 top1 wrong; v3 top1 correct; same REFUTES/c0, citation count 3→1. v3 all remains wrong after extra commitments. | Minimal citation can help locally; more verified documents need not improve the complete answer. |
| case_08 | v2 top1/adaptive correct; v3 both wrong. c0 changes SUPPORTS/two sentences to REFUTES/one. | A shorter rationale cannot compensate for a wrong relation. |
| case_05 | All six cells fail; v3 all commits four source aliases with mixed labels. | Structured output is not grounding correctness. |
| case_17 | Only v3 adaptive is correct. Both adaptive versions select c1, verify, then commit; v2 SUPPORTS/three sentences changes to v3 REFUTES/one. | Verifier change explains an observed recovery; it is not evidence of better source-selection policy or positive-subset choice (one ref/one option). |
| case_07 | Both adaptive versions correctly abstain; v2 uses five calls, v3 three after a failed verifier response. Fixed arms remain unresolved. | Valid abstention outcome with interface asymmetry; do not label a failed tool response as verified scientific insufficiency. |

## Reproduction and next decision

The consumed original is under `runs/scifact-evidence-commit-paired-v2-v3-20261002`;
CPU outputs are under `posthoc/paired-score-1f6f920682e9`, both in the existing
Spartan Climate root. The CPU wrapper and scorer's `--report-directory` are
reproducible entrypoints. They require a new disjoint report directory; they do
not overwrite the original no-quality report. Do not rerun either job merely to
recreate this closeout; the existing exact-hash artifacts are the evidence.

Compact SHA: `d68fa9ace3ff7749c2f9ca5a86775d09ca7eb413214127425f109f3a0c4cb6e9`.
Raw gold, per-case responses and full manifests remain on Spartan.
Reject v3 promotion and preserve v2 top1 as the lower-cost diagnostic reference.
Any further model experiment needs a genuinely different hypothesis and separate
authorization; this package schedules no training, new cohort or GPU repeat.
