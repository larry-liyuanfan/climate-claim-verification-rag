# Eligible SciFact train diagnostic — CPU preparation, no GPU release

Historical preparation scope below is preserved. The separately released job
31620529 subsequently failed at its source-archive guard, before model loading.
See the [startup repair](SCIFACT_STARTUP_REPAIR_20260930.md); its r2 attempt retains
the frozen r1 experiment/data identity. After separate coordinator release,
job31645005 completed; the [result closeout](SCIFACT_TRAIN_DIAGNOSTIC_CLOSEOUT_20260930.md)
preserves its negative train-only findings. The preparation description below
is historical, not permission for another submission or model run.

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
source refusal. Frozen source/actual train sampling receipts will be appended
after these checks pass; no selected-set or model metric is prewritten here.
