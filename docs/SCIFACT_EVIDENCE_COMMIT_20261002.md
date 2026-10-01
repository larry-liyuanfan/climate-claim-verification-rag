# Evidence-commit: CPU candidate, not measured Agent improvement

## Decision and falsifiable question

The prior [bounded confirmation](SCIFACT_DOCUMENT_BOUNDED_CONFIRMATION_20261001.md)
removed all 25 structural failures but recovered zero strict positive whole
answers. Fixed terminal outputs sometimes retained a correct verifier judgment
then added unsupported-under-scoring predictions or changed its label. Keep
those results unchanged. This candidate asks whether **source selection and
selection of immutable verified subsets**, without a second label/rationale
rewrite, can preserve useful local decisions at an acceptable measured cost.

No model inference, training, new scientific scoring, protected-split access or
GPU submission is part of this CPU package. Synthetic traces are implementation
tests, not successful model trajectories. No current resume metric changes.

## New independent contract

Protocol: `scifact-evidence-commit-v1-20261002`; assembler:
`immutable-verdict-subset-v1`. Old direct-answer protocols remain available and
unchanged. This is an interface intervention, not a same-prompt decoder repair.

```text
Frozen original evidence
  → model chooses verify(document), or may abstain immediately
  → same single-document verifier + full physical-call receipt
  → immutable EvidenceVerdictRef (only valid SUPPORTS/REFUTES)
  → model chooses a published selection_id for a verified subset
  → deterministic assembler preserves labels and ordered original sentences
  → unchanged strict and official SciFact scoring, after exit/cost/wire audit
```

Each ref binds protocol/run-release/episode/claim/frame, source alias and original
document identity, source text hash, original sentence indices and hashes,
ordered selected sentence IDs, label, actual physical attempt and verifier
model/prompt/schema/raw identity. The frozen dataclass uses immutable tuples;
model requests cannot supply or overwrite ref content. Scoring reconstructs refs
from reserved/finished/private wire, rather than trusting saved result/feedback.
The release/episode/frame scope is recorded **before generation** in its physical
receipt, so old fixed-verifier calls cannot simply be assigned to a new run.

The model chooses `verify(source_id)`, `commit(selection_id)` or `abstain(reason)`.
Every legal nonempty subset of currently issued refs is exposed, in issuance
order. The finite selection enum avoids relying on LMFE `uniqueItems`; the
catalog is not filtered by gold, scientific correctness or a learned threshold.
Current-episode singleton selections remain legal after a second ref arrives.
Unknown/cross-run/cross-arm/cross-claim/stale-frame refs and invented selection
IDs are rejected. Labels and ordered sentence sets cannot be rewritten at commit.

The same original 5-document / 8-sentence-per-document / 20-total-sentence limits
apply. Assembly never truncates, deduplicates or changes a label to fit. Failed
verifications and INSUFFICIENT judgments issue no committable ref; neither is
claim-level NEI. Only a real model `abstain` can count as model abstention.

## Three-arm comparison fixed before inference

Use the same already-exposed TRAIN24, original model and original evidence;
15 evidence-bearing claims and nine NEI remain diagnostic, not independent test.
The same verifier prompt/schema/ordinary LMFE and scoring definitions are reused.

| Arm | Frozen policy | Physical call cap / claim | 24-claim maximum |
|---|---|---:|---:|
| `fixed_top1` | Verify first document in original retrieval order; commit its valid positive ref | 1 | 24 |
| `fixed_all` | Verify original first four documents; commit all valid positive refs in that order | 4 | 96 |
| `adaptive` | Model chooses next verification, commit subset or immediate abstention | 5 | 120 |
| Total | 72 episodes, no shared-call double counting or padded model calls | — | 240 |

The top-one baseline prevents a model that always selects the first document
from looking like useful dynamic selection merely by beating an all-positive
baseline. Both fixed arms use the same assembler as adaptive. Fixed empty or
over-limit positive sets are explicitly **unresolved**, not successful NEI; there
is no gold-assisted rejection, hidden truncation or extra final model call.

An adaptive verification costs a planning generation plus a verifier generation;
with a later commit/abstain it can make at most two verifications under five
calls. Fixed-all can inspect four. Shared ceilings are not equal actual compute
or equal inspection breadth. Every plan/verdict/failed call is charged. Initial
retrieval is a replayed historical cost, not newly benchmarked online retrieval.
Input/output limits remain 8192/512 tokens, tool cap five including initial
retrieval, and per-episode deadline 120 seconds.

**Verification is required by this interface to commit evidence.** A higher
verify rate alone therefore proves no spontaneous tool demand or Agent gain.
Evidence of a model contribution would require correct source/subset selection,
useful decisions to continue or stop, and quality/cost against **both** fixed
baselines. Positive grounding, NEI, unresolved states and every arm's recovery
are reported separately, including adaptive-minus-top1 and adaptive-minus-all.

## Execution and preflight

- Worker: `scripts/run_scifact_evidence_commit.py`.
- Runtime/state/ref/assembly: `src/climate_rag/scifact_evidence_commit*.py`.
- Scorer: `scripts/score_scifact_evidence_commit.py`, reusing the old scoring
  entry's unchanged defaults and gold-access order through explicit new hooks.
- Freeze: `scripts/package_scifact_evidence_commit.py`; actual wrapper:
  `hpc/scifact_evidence_commit.sbatch`; operator rejects draft authorization.
- Tokenizer-only probe: `scripts/preflight_scifact_evidence_commit.py`.

CPU tests exercise actual inherited `generate()` and LMFE callbacks using
prescribed synthetic tokens, then physical journal → verdict registration →
commit → original evidence rendering → original prediction conversion. A separate
72-slot synthetic test audits all calls before synthetic gold, and four failure
gates prove tokenizer/audit/gold cannot be accessed prematurely. Tests also cover
cross-release replay with fully rebuilt refs, modified raw/ref/selection/label,
negative feedback followed by verification or abstention, immediate abstention,
budget exhaustion, empty/over-limit refs and deadline/assembler agreement.
The actual watchdog consumes the actual new reservation shape; its unexpired
and timeout → child-group termination/reap paths are tested with synthetic
process transport, not claimed as a real GPU process demonstration.

The cached, hash-pinned Qwen tokenizer was run against all 24 original views,
with synthetic maximum-size verdict and failure envelopes: **maximum 6573 input
tokens, zero overflow, zero model calls and no gold read**. These are prompt-size
checks only, not model behavior or a quality result.

Whole-package validation and clean-source receipt are recorded with the source
candidate; earlier unchanged training/test results are not rerun as new evidence.
Local full regression: **1317 passed / one existing Windows-only POSIX permission
skip**, 240.74 seconds. Ruff and strict typing of all 105 library source files
passed; the six new/changed entry scripts passed strict checks with imported
legacy script diagnostics suppressed, not misreported as newly type-clean legacy
code. The source packager's existing strict check, actual Bash syntax check,
diff check and tracked secret/PII scan passed.

The frozen execution candidate is `2ab219bb13564790b004797354dcc89554da2bf6`.
Its exact-source [Linux CI](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36876570554)
completed with **1318 passed**, including the POSIX case skipped on Windows.
Fresh extraction of the hash-verified source archive imported the extracted
library (not the editable checkout) and passed **42 targeted tests in 46.44 s**:
the new core/runtime/entry tests plus the unchanged document scorer entry tests.
This uses the pinned validation environment; it is a clean-source reproduction,
not a newly provisioned environment or a scientific model run. Exact package
hashes and the scope of each check are in the
[CPU readiness receipt](verified-runs/scifact-evidence-commit-cpu-2ab219b.json).
Documentation-only closeout commits do not change or repackage this candidate.

## Proposed resources, still not authorized to submit

One A100, eight CPUs, 32 GiB host memory, 30 GiB scratch, **40 minutes**;
1980-second whole-worker cap, 120-second episode cap, normal QoS/Nice0/no-requeue.
The previous same-model job consumed 391 allocation seconds for 144 generations
and MaxRSS 9,271,396 KiB. Scaling 391 by `240/144` gives about 652 seconds; a 3×
timing allowance plus roughly seven minutes for preparation/scoring fits a
40-minute ceiling. This is a conservative estimate, not guaranteed runtime or
permission to submit. The global worker cap does not promise that 72 episodes
can each consume their full individual deadline. No automatic retries.

Only an exact-source/release authorization can enable a later real comparison.
Raw responses, per-case records and scientific data stay on Spartan; export only
redacted aggregates/case summaries. The earlier 686-file local temporary copy
was matched byte-for-byte to Spartan, but its removal was blocked by tool policy;
it remains preserved and is not falsely reported as deleted.
