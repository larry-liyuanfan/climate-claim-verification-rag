# Evidence-gap-carrying decision: isolated CPU candidate

**Status: engineering hypothesis, not a model-quality result.** No real train,
dev or test inputs, weights or new GPU jobs are used in this package. Frozen
diagnostic31620529 continues unchanged on source`d0e2339`; this candidate does
not auto-chain after it. The [fixed protocol](protocols/evidence-gap-candidate-v1.json)
defines the release conditions before any candidate real-model results exist.

## Motivation and what is *not* reproduced

[Self-RAG](https://arxiv.org/html/2310.11511v1) distinguishes retrieval need,
relevance and support using learned reflection tokens; its critic/generator
training is material to the method. [CRAG](https://arxiv.org/html/2401.15884v3)
uses a separately fine-tuned retrieval evaluator and corrective retrieval.
We borrow the distinction and feedback idea only. This untrained, zero-shot
JSON wrapper is **not a reproduction**, trained critic, calibrated confidence,
or demonstrated autonomous Agent. No paper performance is claimed for it.

## One model decision, not an extra critic call

```text
Same frozen controller/route/tools
  → pack full candidate prompt within existing caps
  → one wire: evidence_state → exact claim gap (or empty) → original decision
  → strict envelope/schema/span validation, explicit action projection
  → unchanged action parser/tool/terminal guards
  → next observation carries last model-reported gap + visible-ID delta
```

The compact state separately reports retrieval need, relevance and support.
The gap is at most160 characters, empty or an exact contiguous fragment of the
**controller-normalized immutable claim**. It is not a new query or explanation.
Evidence state is fallible self-report, not an evaluator score. Any state may
coexist with a legal immediate answer or abstention; no rule forces tools.
Nested actions preserve citation order and duplicates for the old validator;
invalid outputs are not silently corrected or replaced with abstention.

The next observation labels the gap as a previous valid-envelope proposal, not
an executed/accepted decision. A read-loop or rewrite constraint may still reject
the projected action. Added/removed IDs compare the preceding actual model
observation with the new **final packed current_citable**, not previews or packing
probes. The old controller's tool-completion feedback is a separate flag.
ID change—even after a completed tool—does not establish semantic progress.

## Isolation, mapping and cost

- `evidence_gap_candidate.py`: pure request preparation, strict envelope and
  exact-gap parsing, per-question adapter state; invokes the unchanged
  `SciFactDocumentAgentV1` for all four route policies.
- `evidence_gap_grammar.py`: a new single-sequence callback using LMFE0.11.3.
  Its constructor resets parser config; only **after construction**, with the
  same root parser and empty caches, the ordering flag is set in place. Effective
  tokenizer alphabet is preserved. The outer `required` order is state/gap/decision;
  nested schemas keep their order too. Old config and environment are untouched.
- `local_gap_provider.py`: a narrow versioned override with the frozen loader,
  manifest validation, greedy generation call, timeout, quota and accounting.
  Both count/generate use `render_gap_prompt`; no global renderer monkeypatch.
  Bare backend has a different terminal protocol and cannot bypass the adapter.
  Actual weight/GPU integration remains **unverified**, requiring future preflight.
- `smoke_evidence_gap_candidate.py`: only a four-file hash-verified cached
  tokenizer and synthetic inputs. It has no dataset, model or scheduler arguments.

The original wire is written by the backend to the quota-bound private attachment.
Ordinary mapping records contain its full SHA, attempted-byte receipt, status and
parsed action; receipt identity is reconciled without requiring a truncated stored
prefix to equal the full text. Quota truncation does not modify parsing or cost,
but means the full response cannot be claimed recoverable. Full raw text is not
duplicated in normal diagnostics. I/O failure is fail-closed.

Projection returns the unchanged nested action but charges **all wire tokens**.
Invalid JSON/state/span/action schema, repairs and previous-feedback overhead are
included. Unknown generation usage stays unknown/lower-bound, not zero-cost.
Calling `count_prompt` repeatedly during packing does not advance state. A fresh
adapter is created for every question **and route**, including repeated identical
claims. Final visible IDs/counts/hashes must be logged after the enlarged template
is packed; equal caps do not imply equal visible context or actual cost.

All four future candidate routes use this same template/model/caps. Candidate
adaptive versus old fixed routes is **not** a fair gain comparison. Paired old/new
synthetic packing measurements below are overhead diagnostics, not quality evidence.

## Dependencies and CPU reproduction

`jsonschema==4.23.0` is a runtime dependency of this candidate's parser, available
through the repository's test/dev extras. Its presence in CI does not verify the
frozen Spartan runtime. Any future real release must explicitly verify it there.
LMFE must be0.11.3; environment overrides are rejected without changing them.
The existing tokenizer cache must match the four frozen hashes at revision
`350135a4de9a3407be836fa238cccc1d61503a85`. No network download is attempted.

```bash
python -m pytest tests/test_evidence_gap_candidate.py -q
python -m ruff check src scripts tests
python -m mypy src/climate_rag
# From a clean source checkout; choose a new ignored output directory:
python scripts/smoke_evidence_gap_candidate.py \
  --tokenizer data/qwen3-4b-tokenizer-v3 \
  --output data/evidence-gap-candidate-cpu-20260930-r1
```

Tests cover strict envelope/duplicate keys/span, immediate termination for all
four routes, count purity, actual read versus rejected proposal, stale and
duplicate citations, cost retention, private truncation/hash mismatch, run-state
isolation and refusal of a bare candidate backend. Mock-HF tests exercise the
new provider's real Python count/generate methods with synthetic tensors; the
generation-call block is checked against the frozen method. A separate real
TokenEnforcer callback test rejects early decision and skipped gap while leaving
the old config unchanged. These are not weight-based generations.

The cached-tokenizer smoke will additionally check fresh A/B/A prefix state,
ordered mixed-label documents, wrong/stale IDs, and the regression control:
setting order only in the parser constructor wrongly accepts decision-first.
It measures actual final visibility under both templates for four **synthetic**
routes at the same2048-token fixture cap. Default future caps remain8192/512.
Successful CPU validation is not a claim about real-model usefulness.

The first cached-tokenizer smoke on source`48922af` stopped during synthetic
private-directory initialization: the harness supplied a nonexisting relative
path to the strict private-store contract. It had executed no prefix checks,
packing cases, real queries or model calls. The harness now creates an exclusive
absolute private directory, with a regression test; the failed output directory
is retained and a separate rerun directory is used. This is a CPU harness repair,
not a changed model policy or repeated data evaluation.

Linux CI36677108364 then caught a second harness portability issue: its new
private leaf directory used the default mode instead of owner-only permissions.
The strict private-store check correctly refused it (512 other tests passed).
Creation now requests0700 for the new output/private directory, with a POSIX
assertion in the existing regression. No policy, parser, provider or tokenizer
logic changes; the Windows tokenizer receipt below remains bound to its original
source rather than being relabelled as a new run.

### Verified CPU receipt and an explicit cost tradeoff

The repaired smoke passed on clean source
`ba387c3d7f6ba802012c45dc49235b405fd5ac3c`; its
[public synthetic-only receipt](verified-runs/evidence-gap-candidate-cpu-20260930.json)
has canonical LF SHA`c9538a9dbcb8a109d17dfa6f06df4a46ec0d377a973c86cc5fd50bcb50aad694`.
The original Windows-generated report has CRLF SHA
`b1ce2a2986a940b71d501851dc499257e58d6af4c9852ba35dc809389c17a5d7`;
the copies were compared and differ only in newlines, not values.
All three fresh A/B/A paths accepted the ordered mixed-label wire, and each
rejected decision-first, skipped gap, cross-document SID and stale alias. The
constructor-only regression control wrongly accepted early decision on all three,
confirming why the post-construction ordering flag is necessary. Each valid
fixture has1065 prompt tokens and106 scripted wire tokens; these are encoded
fixture lengths, not generations, cost savings or throughput.

The additional template demonstrably **displaces evidence** at a fixed2048-token
synthetic input cap; it is not free reasoning or an extra budget:

| Synthetic routes | Old visible sentences | Candidate visible sentences | Old / candidate packed input tokens |
|---|---:|---:|---:|
| Fixed retrieval, fixed rerank, deterministic extra |11|8|2012 /2017|
| Adaptive |10|7|2026 /2034|

The scripted abstention wire expands from14 to41 output tokens. These comparisons
measure prompt/serialization overhead only, not model behavior. The nearly equal
input token totals reflect budget saturation with different retained context,
**not** negligible overhead. No budget, template, fixture or threshold was changed
to conceal this tradeoff.

Validation:28 focused tests passed before the harness fix; the repaired source
passes29 focused tests, including the private-directory regression, both in the
working source and a fresh LF-exported Git archive. Source archive ZIP SHA:
`a4bb726aa998d891800bbba102b53c0cb7ca025d6ffe16236ce3cf2620cd70bf`;
the fresh process imported from its extracted`src`, not the editable working tree.
The pre-harness full local suite passed511 tests with2 environment skips
(Torch unavailable and POSIX-only ownership/symlink check); Ruff and mypy56-source
checks passed. The harness repair changes no runtime candidate/controller code.
No real queries were read, no model weights were loaded, and no GPU job was
submitted by either CPU smoke attempt. Real HF/weight usefulness remains untested.

## Stop/release rule

The frozen train diagnostic must first show at least one auditable, naturally
occurring gap: a complete alternative rationale absent from actual first context,
reachable through an existing legal same-budget tool, and not effectively handled
by the frozen adaptive route. Audit proposals, executed events, subsequent
visibility and original rationale metrics—not tool-use count or ID novelty.
If this mechanism is absent, inconclusive, or the run is invalid, **do not release
this candidate on real data**. Do not lower thresholds or resample to force a pass.

Even a passed mechanism gate requires separate coordinator approval, fixed all-four-
route protocol, runtime checks and a charged real-weight synthetic preflight.
No dev/test access, critic call, training, broader rewriting, web retrieval,
automatic rerun, or resume-quality claim is authorized by this package.
