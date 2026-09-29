# Offline scoring contract closeout

Subsequent execution: frozen208ff93 was used in CPU audit31537871 to score
confirmation31520350; see the [terminal report](BUDGET_AGENT_CONFIRMATION_20260929.md).
The development/clean-archive verification statements below describe the earlier
scorer-only package, not a claim that the completed audit remains unexecuted.

This package changes only the post-hoc scorer, its synthetic tests and handoff.
Confirmation job31520350 still executes frozen source72eaa90567e3603a3b940edffbb21b684bf1d1ba.
Its prompt, input bundle, models, budgets and Slurm request are unchanged. No
actual pilot/validation/vNext predictions were used to design this audit; no
full evaluation, retired test, SciFact, training or GPU repeat was run.

## Correct denominators before paired scoring

The scorer requires the frozen protocol bytes/SHA and explicit phase. It first
validates phase/corpus/study identity and the exact unique `(task_id,strategy)`
Cartesian product, including tasks without evidence gold. Missing, duplicated,
extra or wrong-route records are rejected **before** aggregation/bootstrap;
normalized query hashes must match the protocol even on authored tasks.

Validation uses the existing checked-in
[selection manifest](verified-runs/budget-agent-validation-selection-20260929.json),
not a new split or parallel registry. The CLI always loads this manifest from
the scorer checkout, pins its whole-file SHA and explicitly requires32/24/8
task/evidence/non-evidence counts. It verifies protocol and gold byte hashes, selected IDs,
source archive/member provenance, claim hashes and evidence-bearing counts.

| Phase | Tasks/route | Complete slots | Retrieval denominator/route | Label denominator |
|---|---:|---:|---:|---|
| Reused validation |32|96|24|Actual SUPPORTS/REFUTES labels with a real model; not assumed24|
| Authored vNext |8|24|Not applicable|Not applicable|
| Authored pilot |3|9|Not applicable|Not applicable|

Every valid validation task participates in latency/work/failure summaries;
only the24 evidence-bearing rows enter retrieval averages and5,000 paired
bootstrap samples (seed20260929). A mechanically rejected/deadline answer does
not earn credit from the undelivered candidate list. Heuristic/no-verdict
execution has null label accuracy, not zero classification effectiveness.
Authored phases reject any injected gold and return null official quality and
bootstrap. `semantic_supportability` is always null; exact quotes and number
checks are not entailment or a hallucination-quality measurement.

## Cost and outcome accounting

Each route retains status and fixed reason-category counts and separately sums/averages tool,
generation/model, retrieval, rerank-call and rerank-candidate-pair counters.
Model-requested abstention requires a matching terminal abstain decision;
controller rejection and operational failure are separate. In particular,
`status=abstained` does not mean the model chose abstention. A model's free-text
reason that happens to equal a controller label is disambiguated by events.

Token summaries preserve all **recorded** token counts, split into complete
accounting rows and partial recorded counts from unknown-accounting rows.
Unknown rows are never silently removed. Recorded totals are lower bounds when
any usage is unknown; the corresponding baseline/candidate token comparison is
unavailable. Complete known usage and a common real provider are prerequisites
even for token-only comparison. Monetary cost comparison remains unavailable:
there is no validated currency/GPU/API pricing model, and generation tokens do
not account for reranker work. No savings are computed from partial totals.

Compact scores exclude model-written reason prose. Controller error enums remain
exact; event-confirmed provider abstentions use explicit model/heuristic categories.
Unknown error suffixes are rejected rather than exported as arbitrary text.
Original reasons, answers and full run archives stay on Spartan. GitHub/career
handoffs receive only reviewed compact counts, hashes and enums.

## Offline entry points (not executed on real results in this package)

Use the **new scoring checkout/archive**, not the scorer embedded in72eaa90.
The CLI records input-run SHA, protocol/gold/selection-manifest SHA, inference
source SHA, and the separate scorer Git/script SHA and dirty-state indicator.
It refuses to overwrite an existing score. These commands do not run inference.
Use an authorized allocated CPU context for real-result scoring on Spartan;
do not start model or data-compute work on a login node.

```bash
# Reused validation: frozen protocol/gold from the existing preparation, no new split.
python scripts/score_budget_agent.py \
  --run /path/to/private/validation-run.json \
  --protocol /path/to/frozen/validation-protocol.json \
  --expected-protocol-sha256 abfcb61e9e6a54641f025101a4011fa88e07e3697a36f0acd308fa8e86052674 \
  --phase validation --gold /path/to/frozen/validation-gold.json \
  --output /path/to/private/new-validation-audit.json

# Authored application tasks: no gold argument, no official quality metric.
python scripts/score_budget_agent.py \
  --run /path/to/private/vnext-run.json \
  --protocol configs/budget_agent_vnext_20260929.json \
  --expected-protocol-sha256 6ee90b8479192335fc543c3425c3fbbb89fea2d65ba9698fb54d86564bc3c3c2 \
  --phase vnext --output /path/to/private/new-vnext-audit.json
```

For confirmation31520350 substitute its private `run.json`, use `--phase pilot`
and the same authored protocol SHA. No gold is allowed. Run/answer text is not
copied to the local career workspace for scoring. Paths above are placeholders
to resolve against the verified artifacts, not permission to create new runs.

## CPU-only verification

Tests use wholly synthetic protocol/gold/claim/trace objects. They prove96/24
slot checking,32/24/actual-label denominators, shared missing-nongold rejection,
duplicate/extra/phase/identity failures, partial token preservation, honest
cost availability and event-grounded outcome categories. They do not estimate
real-model accuracy or change the interpretation of earlier negative runs.

```bash
python -m pytest tests/test_budget_agent_scoring.py -q
python -m ruff check scripts/score_budget_agent.py tests/test_budget_agent_scoring.py
python -m mypy scripts/score_budget_agent.py
```

The score API's injectable synthetic selection object exists for fixture tests;
the CLI uses the existing committed real selection manifest. Full actual
results and score files remain subject to their privacy boundary. Runtime
source identity and scorer identity must be reported separately at closeout.

Local verification:42 synthetic scorer tests passed; Ruff and explicit strict
mypy of `scripts/score_budget_agent.py` passed. No changed inference files and
no actual-result rescoring. The corrected scorer does not relabel earlier
negative pilot batches or silently overwrite their existing score artifacts.

Frozen scorer source: `208ff931badff270cbbc9593c8ab54f5c79aec9e`.
Clean source archive:1,402,880 bytes, SHA-256
`7498d367565eeb59bb7cf724c33f835608a0c9f01da3768d7308700a380c00a1`.
Archive scorer-script SHA-256:
`758bb04e9d99cdfb298c936de550dc66055a706fbe82bd19e1f5db200ccf3b86`.
The42 tests also passed from this clean archive; import origin, source revision
and pinned selection manifest were independently checked. Read-only review
found no remaining material blocker. No scorer archive was uploaded to Spartan
and no real outcome was rescored. This scorer identity is **not** the inference
identity of confirmation31520350, which remains72eaa90.
