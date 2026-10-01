# Prospective conditional TRAIN24 confirmation: no top1 quality/cost advantage

## Decision

The once-frozen three-arm confirmation completed successfully. **Do not promote
the current adaptive policy or rerun it to seek a positive result.** It matched
fixed-top1 on complete evidence-bearing answers (5/15 each), was lower on all
four official evidence F1 measures, and used **5.16 times as many tokens**.
This supports retaining top1 as the cheaper evidence-quality baseline for this
diagnostic, not declaring the broader grounded-Agent objective complete.

Adaptive did make six correct explicit NEI decisions. The fixed interfaces
instead classify empty-positive outputs as unresolved, so this affordance is
not an equal-interface demonstration of better reasoning. Do not say top1
dominates every possible quality metric: it dominates the observed evidence-F1
and token-cost comparison, but not the asymmetrical whole-answer total.

## Scope and whole-chain readiness

These are **24 conditionally selected TRAIN components**, not an independent
validation/test or source-family-unseen set. All 531 eligible TRAIN rows were
seen by an earlier gold-preparation process. The roster was chosen once by
metadata/component hash before selected query reads, excluding the documented
later training/debug reservations. All three arms reuse the same frozen
retrieval/packing frames and unchanged policy/model/prompts/scoring. The older
TRAIN24 and this roster are different sets, not claim-level pairs.

The [CPU review and submission checklist](SCIFACT_PROSPECTIVE24_CPU_20261002.md#whole-chain-review-before-submission)
traced metadata → reservation → claims → frames → operator → worker → cost →
post-exit scoring. It repaired the new-input consumers' old-roster coupling
before submission. Evidence reused at the exact source identity: 53 affected
chain tests, 11 clean-archive tests, Linux strict types/Ruff, and
[CI with 1,329 tests](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36884594874).
Actual wrapper/import/argument checks, resource/cache checks and test-only
passed before the sole authorized GPU submission. A ready executable chain
does not imply a successful model-quality hypothesis.

## Execution and immutable identities

- Job **31956320**, COMPLETED / `0:0`, 2026-10-02 **01:44:48–01:49:48 UTC+10**.
  Allocation **300 s**, TotalCPU **291.408 s**, batch MaxRSS **9,297,560 K**.
- One A100 / 8 CPU / 32 GiB RAM / 30 GiB scratch / 40-minute ceiling. Test-only
  **31956238** is not an additional executed job. No cancellation or resubmission.
- Execution source `da243036871f61eef2e039a1618b8a3e1e1a00ac`; source archive
  SHA `d4c2ef55a9fc8675df2aba3e1bf537c22fcbd71fc8ee0c8f1f54df0a1421dbce`.
  Later documentation/report commits are not the model execution source.
- Authorized release SHA `db794da36ea061b5c0f537dd83829c7bd69e6763663410f407c6f5a5f72123ad`;
  wrapper SHA `d30e1a2ebe2cc99630d776f2b9d3d365f1843d37a36a293dc6a0592c1895a815`.
- Selection SHA `351523c9c44bb418e24ed17b40372e6bef7b87792f8ad0b78dcaeb2d22e22754`;
  shared frames SHA `e20a5188ba51bbaf940009338ee167d6d07bf4429652eaa06967265f37a4e29c`;
  runtime receipt SHA `6153e39d11dbf44b57b311359cf2b5846dcf6a45ef70cd9069f69e4c8c68d455`.
- **72/72 slots**, **200/240 physical generations**, zero missing episodes,
  unknown token/timing receipts or unassigned calls. Worker reaping, physical
  cost and result/ref audit preceded the selected TRAIN gold scoring.
- [Redacted compact](verified-runs/scifact-prospective24-confirmation-31956320.json),
  SHA `7315a387e66f60dde5aaa11840aca2a2c715168d9c7a043f575b2d65ebc46515`.
  Raw claims, gold, original predictions and physical responses stay on Spartan.

## Paired quality, with unchanged definitions

| Outcome | Fixed top1 | Fixed all (first four) | Adaptive |
|---|---:|---:|---:|
| Complete evidence-bearing answers | 5/15 | 3/15 | 5/15 |
| Correct explicit NEI | 0/9 | 0/9 | 6/9 |
| Strict whole-answer total | 5/24 | 3/24 | 11/24 |
| Unresolved | 9 | 7 | 0 |
| Commit terminals | 15 | 17 | 17 |
| Explicit model abstentions | 0 | 0 | 7 |
| NEI claims with false evidence | 2/9 | 2/9 | 3/9 |

For evidence-bearing claims, adaptive versus top1 has five jointly correct and
ten jointly incorrect cases, **no adaptive-only recovery**. Versus fixed-all,
three are jointly correct, two adaptive-only, none fixed-only and ten jointly
incorrect. Do not turn the six extra NEI decisions into six additional grounded
positive answers. One adaptive abstention misses an evidence-bearing claim;
zero unresolved is not zero errors.

| Official scorer micro F1 | Fixed top1 | Fixed all | Adaptive |
|---|---:|---:|---:|
| Abstract label only | **0.6667** | 0.5385 | 0.6286 |
| Abstract rationalized | **0.6061** | 0.5000 | 0.5714 |
| Sentence label | **0.5385** | 0.3939 | 0.4819 |
| Sentence selection | **0.6154** | 0.4394 | 0.5542 |

Adaptive improves these measures over fixed-all but not top1. These are the
existing official definitions, not a newly invented aggregate "Evidence F1".
Claim-verdict accuracy and free-text entailment remain unmeasured (`null`).
This bounded conditional diagnostic does not justify significance, generalization
or causal-benefit claims. No new bootstrap, changed scorer or protected-test run
was introduced after observing the results.

## Full cost and actual control behavior

| Cost across 24 claims | Fixed top1 | Fixed all | Adaptive |
|---|---:|---:|---:|
| Physical generations | 24 | 96 | 80 |
| Input tokens | 36,038 | 143,096 | 187,665 |
| Output tokens | 903 | 3,146 | 3,024 |
| Total tokens | 36,941 | 146,242 | 190,689 |
| Model backend seconds | 28.485 | 99.260 | 98.065 |

Adaptive uses 16.7% fewer calls than fixed-all, but **30.4% more tokens** and
only 1.2% less measured backend time. Against top1: 3.33 times the calls,
5.16 times the tokens and 3.44 times backend time. Fewer calls do not establish
financial savings. The total ledger accounts for 366,799 input and 7,073 output
tokens. Currency cost is unknown, not zero.

The shared CPU input preparation cost is **31.534 s once** across three arms,
including 1.159 s corpus/BM25 build and 24.187 s retrieval/packing (0.101 s
retrieval). These times are nested, not additive. The GPU run reuses frames;
neither generation timings nor the 300-second allocation are online SLA.

- Adaptive proposed, executed and obtained structurally valid results from
  **28 verifications**. It first selected a non-`c0` source in 6/24 episodes;
  one episode abstained immediately without verification.
- Five episodes chose another verification after INSUFFICIENT; all five then
  abstained after another INSUFFICIENT. This is real feedback-conditioned
  continuation, not observed positive-answer recovery from continuation.
- Seventeen commits and seven abstentions. All episodes produced a legal
  terminal, not necessarily a correct one. Verification is mandatory for commit,
  so a verify call alone does not prove spontaneous tool need.
- No episode issued multiple positive references; each actual commit offered
  only one legal selection. Do not claim demonstrated subset optimization.

## Three explanatory paired cases

Pseudonyms refer only to positions in this roster. The post-hoc illustrative
selector found three qualifying categories; absent win/loss categories are not
fabricated to fill a five-case quota. Aliases are local to each case.

| Case | Actual three-arm behavior | Interpretation |
|---|---|---|
| `case_14` | All three commit `c0`/SUPPORTS/three sentences correctly. Top1 uses one call; fixed-all four; adaptive plan→verify→commit uses three. | A valid Agent path can add overhead with no quality gain. |
| `case_04` | Both fixed routes return unresolved after INSUFFICIENT. Adaptive chooses `c4`, gets SUPPORTS/two sentences and commits incorrectly. | Source binding prevents answer rewriting, not verifier error; expanding source choice can hurt. |
| `case_23` | Top1 is unresolved. Fixed-all commits `c2`/SUPPORTS/one sentence but is incorrect. Adaptive verifies `c0`, then `c3`; two INSUFFICIENT responses lead to an incorrect abstention. | Tool feedback is fallible; valid continuation does not prove missing evidence or solve the claim. |

## Reproduction and closeout

The frozen [input adapter and preparation entrypoints](SCIFACT_PROSPECTIVE24_CPU_20261002.md#entrypoints-and-resource-shape)
and [operator/worker/scorer](SCIFACT_EVIDENCE_COMMIT_20261002.md#execution-and-preflight)
reproduce this contract. `scripts/export_scifact_prospective_commit.py` reads
terminal receipts and existing scores only; it does not regenerate or rescore.
Exporter SHA `6b7da95c6f213c2fac3f9eddf2077f3fdb88eed4bbd25cfc2242744f5de45df6`.
The compact includes shared preparation even on `no_quality`; missing or
unassigned cost is unknown, not zero. The reporting script and artifacts are
separate from the immutable inference source.

This package closes with a negative promotion decision. Preserve the useful
source-bound assembly and diagnostic, but do not relabel this policy a proven
quality/cost improvement, change the resume, retrain, resample or automatically
submit another job. Any next experiment requires a different falsifiable
hypothesis and a complete pre-submission review under its actual authorization.
Trip, Energy, FLARE, shared career files and default branches are unchanged.
