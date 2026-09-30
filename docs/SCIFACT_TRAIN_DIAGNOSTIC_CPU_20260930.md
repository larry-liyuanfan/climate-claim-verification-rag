# Eligible SciFact train diagnostic — CPU preparation, no GPU release

This is a deliberately **biased train diagnostic**, not a benchmark score,
independent test, or resume improvement. It follows the negative/limited
[v3 authored pilot](AGENT_V3_PILOT_CLOSEOUT_20260930.md). Its purpose is to
distinguish available evidence from a real opportunity to acquire evidence,
before redesigning the policy. No critic, prompt tuning, training, or changed
route/budget is included. Original300-dev and retired tests are not opened by
these commands. Foundation-model training exposure remains unknown.

## Frozen selection contract (before train-gold access)

Reuse the prepared5,183 original abstracts and531 eligible train IDs. Verify the
prior preparation manifest and only the four allowed corpus/train/assignment
files; never rerun preparation or read a dev member. The train gold JSONL contains
809 original rows: JSON decoding inspects IDs, but only531 eligible rows enter
`parse_gold`, selection or scoring. Quarantined rows cannot be substituted.

The shared anchor is initial BM25 Top20, with `k1=1.5,b=0.75`, title + original
abstract sentences as document text, original numeric doc IDs, and normalized
queries. Sampler and inference share `SciFactBM25`; `PYTHONHASHSEED=0` is required
before Python starts because the underlying implementation iterates query sets.

Initial visibility is the **adaptive first observation before any model action**:
the frozen SciFact renderer, cached Qwen tokenizer, actual controller `pack`,
full schema/previews/8192-token input budget,5-document requested context. A CPU
fixture captures it and emits abstain; this is not a model decision or benchmark.
Other routes' actual first observations are separately recorded during inference.
Top5 documents do not imply every sentence was visible.

Four exclusive strata, maximum3 cases each:

| Stratum | Predeclared rule |
|---|---|
| initial_doc_opportunity | At least one gold document has a complete alternative rationale of at most3 sentences wholly in actual initial visibility; it could be placed in predicted first3. |
| top20_doc_replenishable | No initial opportunity, but a gold document in initial Top20 can reveal such a rationale through an actually legal, same-budget single-document `read` probe. |
| gold_absent_top20 | Nonempty gold evidence, with no gold document in initial Top20. Rewrite could still find gold; this does not mean impossible. |
| nei | Original `evidence={}`; `cited_doc_ids` are not a label. |

Alternatives are OR; partial sentences from different alternatives do not form
one rationale. Opportunity is **per document**, not all-document or whole-claim
success. Record eligible-visible gold-document count / total gold-document count;
separately retain all-document coverage and union-of-rationale visibility.
Gold-in-Top20 without an actually packable <=3-sentence rationale is unclassified,
not forced into absent or replenishable. Read probes may select gold for offline
measurement only; their decisions, strata, components and gold never enter the
real model prompt, retriever, packer or inference bundle.

Selection uses SHA256 of fixed salt `scifact-eligible-train-diagnostic-20260930-v1`
and claim ID/component, with deterministic maximum component-to-stratum-slot
matching. One original component globally, never duplicated across strata.
Each component/stratum retains its lowest claim-hash representative. Missing
capacity is reported as a shortfall; no thresholds change, dev backfill, duplicated
components or synthetic labels. At most12 queries /48 slots, exact claim-major
four-route order frozen before inference.

## Files and reproducible entries

```text
PYTHONHASHSEED=0 python scripts/prepare_scifact_train_diagnostic.py --prepared <prior-prepared-r1> --tokenizer <frozen-four-file-cache> --output <new-private-dir>
```

Preparation requires a clean tracked **and nonignored-untracked** repository and
tracked entry/dependencies, to prevent an old HEAD being claimed for new code.
Outputs are write-once:

- `inference/`: exact `claims.jsonl` (`id,claim` only), `corpus.jsonl`, `protocol.json`.
- `private/`: selected gold, component/stratum/read-witness sidecar and sampling audit.
- `manifest.json`: source/input/output SHAs, counts, shortfalls,0 model generations.

Prepare separate tar allowlists for inference(3 files) and scoring(selected
gold/strata/manifest). Do not archive the whole prepared tree. The operator
extracts only inference first, waits for its child process to exit successfully,
then extracts scoring and starts a separate CPU scorer. The inference child has
no gold CLI argument or gold archive environment/path. This is enforced dataflow
and scratch separation, **not an OS security boundary against the shared account**.

`run_scifact_train_diagnostic.py` requires a separate release and allocated GPU;
no submission is authorized by this CPU package. It preserves the exact
SciFact provider/terminal, Qwen3-4B and4B-reranker manifests, BM25, greedy/nonthinking
decoding and all four frozen route policies/budgets. No automatic retry/resume.
The synthetic nested-document preflight runs before any train inference; its
cost and model/index load time are separate from the48(or fewer) whole-query slots.

Each slot durably saves raw controller output/usage **before** trace/export
processing. Rejected tool proposals get `event_index=null`; they are not executed
tools. Actual tool links retain failed/completed states and a subsequent-attempt
index. Alias→original-document/sentence/hash visibility is captured without
changing prompts. Export/trace audit errors retain raw costs and prevent a quality
report, rather than disappearing from denominators. Normal model/controller
failures remain explicit empty predictions with their termination reason/cost.

`score_scifact_train_diagnostic.py` joins separate selected gold only after
inference. Require the complete exact matrix; use `to_original_prediction`,
`parse_prediction`, then unchanged `score_original`. Report all four original
metric counts (`correct/predicted/relevant`) and point F1, by route/stratum;
NEI false evidence is separate, not claim-verdict accuracy. All failed/repair
tokens remain; unknown usage is a lower bound. Tool offers, proposals and actual
events are distinct. Whole-question P50/P95 are points on a tiny biased diagnostic,
not an SLA. **No bootstrap, confidence interval, significance or population gain.**

## Operator and provisional resource bound

New `run_scifact_train_operator.py` and `hpc/scifact_train_diagnostic.sbatch` do
not modify completed31601753 or frozen v3 sources. Source tar/revision/wrapper,
protocol, model and inference/scoring bundle hashes must all be bound at release;
exact allowlists reject extra/link/duplicate archive members. Output directories
are exclusive. Retain private responses and gold on Spartan; only reviewed compact
aggregates may be public.

Proposed ceiling:1 A100,8 CPUs,32 GiB RAM,30 GiB local scratch,**2 h maximum**.
The prior123-second short-passage run is not an adequate latency estimate for
multi-sentence abstracts. A conservative engineering bound is48×120 s query
deadline +4×45 s synthetic caps =99 minutes, leaving21 minutes for verified local
asset staging/load/scoring. Per-call timeout may only be checked after a backend
returns; the Slurm ceiling is the hard walltime. This is a ceiling proposal,
not a measured runtime prediction or an assurance it will schedule. Coordinator
must review actual selection/protocol, clean reproduction, tests and
`sbatch --test-only` before separately releasing one job. This package itself
performs no `sbatch`, inference or GPU allocation.

## CPU validation and results

Synthetic tests cover per-document versus whole-claim coverage, alternatives OR,
first3 bounds, unclassified pack failures, NEI, maximal deterministic component
matching, real-controller packing probes, failed/rejected tool proposals,
provenance mapping, raw-cost persistence, exact scoring matrices and untracked
source refusal, aggregate/attempt-cost reconciliation and unknown-usage lower
bounds. Clean archive of source `d0e2339cb9711d4b46211d88c561d2f80bd39b3c`
passed all22 focused tests. Full CI
[36673677403](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36673677403)
passed484 tests /1 optional skip, Ruff, mypy53 source files and tracked privacy
scan; companion36673682247 also passed. Local/remote `bash -n` passed.

### One frozen CPU sampling run

The source was clean and runtime import path checked before the only sampling
run. No model weights or real model generations were used; the real cached
tokenizer counted full prompts for531 eligible claims and legal read probes.

| Eligible-train category | Claims | Distinct eligible components | Selected |
|---|---:|---:|---:|
| Initial per-document opportunity |286|188|3|
| Top20 per-document read opportunity |13|12|3|
| No gold document in Top20 |17|14|3|
| Original NEI |214|135|3|
| Gold present but not packable within contract |1|not a selection stratum|0|

The final12 cases come from12 distinct original components; all four shortfalls
are0, yielding a frozen48-slot order. Component counts above are **within each
stratum** and must not be summed as globally disjoint. These are train-data
opportunity counts, not model accuracy, tool-use results, or benchmark coverage.
No parameter, threshold, label, query or corpus was changed to obtain the quotas.

Content-free receipts:

- [Preparation manifest](verified-runs/scifact-train-diagnostic-preparation-20260930.json),
  SHA `dceda481dba4966b32cfb0234449977cf224d5bceeaeb55f3eb5790aba364b8e`.
- [Bundle receipt](verified-runs/scifact-train-diagnostic-bundles-20260930.json),
  SHA `897fb4ec63fbf5cfe0d51911d803bdef386ba3494de8fa81902df02e7a6f615c`.
- Protocol `306e8a61eef6fb279c56ccfc2b496281df9cc8710e194d2272e6978a23dbe802`.
- Inference-only tar8,304,640 bytes:
  `92846f0904992e7b7137e550c46cd0c76b2f77b4f808a6d6be25482f90bf1cca`.
- Scoring-only tar40,960 bytes:
  `174ea64c476fc3abdd5575d53e9a1808c8fd5e402e07cf327e58295346d41444`.
- Exact Git source tar with `SOURCE_REVISION`:
  `a0979ba67d24e20a7b47a324b645e4fb4f3ea67cc4be5dcfd233fbf499e38c53`.
- Sbatch wrapper:
  `351ab5f7ef1f8e1762d99400c6db5cc47500ac30227e5a00c3551f0affc41ca4`.

Only content-free manifests are public; selection IDs, gold, witnesses, original
corpus and full sampling audit are excluded from Git. `sbatch --test-only`
accepted the proposed wrapper/resource shape. The displayed simulation ID31618509
and predicted2026-10-05 start are **not a submitted job or guaranteed start**.
At this closeout there is **no new GPU job**. Release of the frozen inference
bundle still requires the coordinator's separate review/authorization.
