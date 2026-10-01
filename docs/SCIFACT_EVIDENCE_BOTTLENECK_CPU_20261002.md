# Evidence bottleneck: CPU implementation, not model-result evidence

This is a new, falsifiable **input-visibility intervention**, prompted by the
[negative v2/v3 result](SCIFACT_PAIRED_VERIFIER_RESULT_20261002.md). It is not a
repeat of v3, a new training run, or authorization for any model generation.
Scope remains the same already-consumed conditional TRAIN24 and original top1
document; no new sample, real gold or protected split is read by this CPU package.

## Frozen routes

| Route | Physical calls per claim | Actual label input | Citation policy |
|---|---:|---|---|
| A | 1 | Original v3 visible top1 document, evidence-first prompt/schema order | Original v3 parser/projection |
| Shared selector | 1 | Claim + original top1 sentences | Select up to eight original IDs, including required context; no label or explanation |
| B | 1 after shared selector | Claim + locked selection + original full visible top1 text | Program binds all selected original sentences for a positive label |
| C | 1 after the same selector | Claim + identical locked selection, empty full-document field | Same as B |

The maximum is **96 physical generations and 72 route results**, not 120
physical calls. Independent deployment costs are A; selector+B; selector+C.
B and C share one exclusive on-disk selector artifact, read and identity-checked
before each branch. Their label schema, renderer, seed and 512-output ceiling
are identical; only `full_document` differs. Each stage retains the 8,192-input
limit. There is no retry, repair, replacement sample or automatic truncation.

A preserves v3 fields and decoder semantics. `force_json_field_order=False`
remains unchanged: changing prompt/schema order does **not** establish that the
model reasons about evidence first. No chain-of-thought claim is made.

Selector IDs are validated and mapped back to exact source text in original
sentence order. Empty selection, invalid/duplicate IDs, format failure and
overflow remain distinct failures. Both downstream routes then fail without a
label call. They are not scientific NEI. Label stages cannot add, delete or
replace citations. A valid document-level INSUFFICIENT is also not a whole-claim
NEI success; the original fixed-verifier unresolved convention is preserved.

## Pre-registered decision

B/C raw selected sentence sets and complete-rationale coverage are identical by
construction. Do not call a label projection change a selection-coverage gain.
The gate compares C against B on official abstract-label micro-F1, NEI false
evidence count (lower is better), and abstract-rationalized micro-F1: at least
one strictly improves, none deteriorates, and technical failures do not increase.
Otherwise record `not_supported`; no threshold changes to chase a positive.

Always retain the stronger historical **v2 top1** and original **v3 top1**, both
SHA-bound from the accepted compact artifact, with all failures and cost. Even a
supported mechanism is only conditional TRAIN evidence, not fresh-test
generalization or Agent benefit. Subsequent Agent-versus-strong-fixed comparison
would require a separate authorization; it is not part of this package.

## Reproduction and whole-chain verification

- `scripts/run_scifact_evidence_bottleneck.py::run_suite`: same production runner
  is injected with a synthetic provider in tests, writes authorization/planned
  slots/stage/private-wire/physical ledgers and immutable shared selection.
- `score_scifact_evidence_bottleneck.score_after_exit`: existing physical wire,
  effective generation, private-response and original SciFact scorer checks.
  Cost and fixed denominator survive failed exit, missing output or unknown cost;
  the gold callback is not reached until every identity audit passes.
- `export_compact`: whitelist aggregate outputs; no real claim, source text,
  original IDs, raw response or gold-row export.
- `tests/test_scifact_evidence_bottleneck.py`: full 24-claim synthetic matrix,
  actual provider/LMFE/config path with fake model output, numeric JSON roundtrip,
  shared-binding tampering, real visibility difference, citation lock, failure
  propagation, cost tampering and no-gold-before-audit regressions.

Example local check (no weights or real data):

```text
python -m pytest -q tests/test_scifact_evidence_bottleneck.py tests/test_scifact_relation_verifier.py tests/test_scifact_paired_comparison.py
python -m ruff check src scripts tests
python -m mypy src/climate_rag
```

Linux CI supplies the existing pinned CPU Torch/Transformers/LMFE stack.
Windows full-source mypy encounters the historical POSIX `resource.getrusage`
annotation difference; do not remove that Linux check or misreport it as an
application regression. New module type-checks locally; Linux full-source check
is the authoritative environment check.

## Single permitted cluster preflight

`hpc/scifact_evidence_bottleneck_cpu.sbatch` requests at most 2 CPU / 8 GiB /
10 minutes, no GPU, no requeue. It verifies archive, wrapper and all extracted
source bytes **before** executing. Existing pinned tokenizer files only; zero
weights/generation/training. `preflight_scifact_evidence_bottleneck.py` checks
the same 24 frozen frames, singleton/longest-eight/max-cardinality cases and an
all-visible-sentences B stress prompt. This is a bounded tokenizer envelope,
not proof covering every possible BPE concatenation or model response. Runtime
limits still reject any actual overflow.

The emitted `model-run-draft.json` binds exact source/model/tokenizer/input
identities and generation contract but has `model_execution_authorized=false`.
It does not extract weights, reserve another run or submit a GPU job. A new
exact-source approval and bounded parent supervision/exit receipt are required
before future real inference. This package stops after CPU preflight/CI receipt.

Actual receipt and source/CI identities are appended only after verification.
