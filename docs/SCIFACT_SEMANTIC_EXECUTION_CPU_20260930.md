# Semantic-policy paired execution: frozen CPU handoff, not GPU release

## Question and immutable inputs

Does one explicit scientific evidence-relation policy improve document-specific
judgment, and does any improvement actually involve useful model-selected tools?
The comparison is a **12-question, gold-stratified TRAIN diagnostic**, not an
independent test, an online service evaluation or a training intervention.

Reuse the exact [CPU preparation 31698106](SCIFACT_SEMANTIC_POLICY_CPU_20260930.md):
preparation source `779e49883570371ce6223cb2b4a8619df5924d3b`, compact SHA
`4d73e0196fb64167798b89746020b15aca192ebdb94cf34a36d6bd274d7adadd`.
New execution source is recorded separately in the final handoff receipt. No
reselection, altered gold, prompt tuning, new strata or renewed 531-row probe is
performed. Both policies use `gap=True`:

| Policy | Frozen prompt SHA-256 |
|---|---|
| `scifact-gap-original-v1` | `b4a6b1e5accd5a6b427dfe278245580c9ff16598c03d876c8920324cc2d30dfd` |
| `scifact-gap-semantic-policy-v1` | `2be482d567d0e57a546eebae440e9b04f86995a0724260b95c7d31278f834fab` |

## Entry points and isolation

- `scripts/package_scifact_semantic_pair.py`: stdlib-only mechanical array→JSONL
  conversion of the six frozen private artifacts. Checks compact/private/corpus
  hashes and order; does not sample, index, pack prompts, evaluate or load a model.
- `scripts/run_scifact_semantic_arm.py`: new explicit policy namespace, shared
  `LocalQwenSemanticGapProvider` and `run_semantic_slot`. Old F/G defaults remain.
- `scripts/run_scifact_semantic_operator.py`: exclusive staging/results, serial
  arm checks, then a separate scoring child after **both** inference processes exit.
- `scripts/score_scifact_semantic_pair.py`: official scoring kernels plus explicit
  terminal, wire, cost and evidence-chain audits; no old `--r2-run` prerequisite.
- `hpc/scifact_semantic_pair.sbatch`: allocation-only draft; **not submitted**.

The inference archive has exactly `claims.jsonl`, `corpus.jsonl`, `protocol.json`.
Claim rows allow only `id`/`claim`; the protocol also has an exact key allowlist.
Its ordered claims are independently bound to the CPU selected-inference hash,
so even a self-consistent replacement matrix is rejected. The protocol fixes
models/tokenizer/prompt/data hashes, all 12×4×2 slots, decoding and original budgets.
It contains no gold, strata, consumption ledger or oracle read witnesses.

The separate scoring archive contains selected gold, legacy strata, current
opportunities, consumption ledger, selection freeze, preparation draft and a
manifest. Scoring reconstructs the selected-gold JSON-array hash and checks the
original private SHA map; replacing labels/strata and updating an untrusted
manifest cannot create a new authorized dataset. Original files stay on Spartan.

Only generator/reranker model directories and manifests are extracted from the
existing shared model-asset archive. Its old evidence, validation/authored
protocols and argument files are **not** extracted into this run. Runtime and
grammar locks remain frozen. Child environment strips scoring paths and all
other Climate controls before injecting the four required execution identities.
This is dataflow isolation in allocated scratch, not an OS sandbox against a
shared SSH account.

## Failure and comparability contracts

Four synthetic real-provider preflight calls per arm remain separately charged,
with policy/prompt/source/protocol binding and started/completed case journals.
Journal failures stop before further generation. An interrupted started case is
an uncertain attempt, not silently zero-cost. Passing forced synthetic cases
would prove runtime compliance, not scientific quality or autonomous tool choice.

Every query slot reserves exclusive response/raw/result paths. Actual raw cost
is durable before auxiliary export; unknown usage or incomplete wire stops the
batch with its partial evidence retained. The completed-arm verifier requires
raw to equal the complete pre-finalizer projection, not a permissive subset.
Raw wire and grammar receipts are audited as multisets for **both gap arms**.

The second arm checks the newly run first arm's full matrix, source/archive,
protocol, models, prompt and preflight before starting; it rechecks its accepted
hash after inference and before extracting gold. Historical G is not a baseline.
Both arms share maximum-policy context packing and equal ordered initial visible
and preview identities. Later adaptive contexts and costs need not be identical.

## Planned measurements, not results

For each policy × route: official document/sentence correct/predicted/relevant
counts and F1, strict whole-final-answer custom diagnostic, legacy-stratum scores,
separately labeled current opportunity strata, NEI false evidence, genuine model
abstention versus failure-empty prediction, nonempty answer coverage, P50/P95,
known tokens including failures and unknown-cost flags. Preflight costs are
separate. Raw proposal → strict decision → actual event → next feedback → newly
visible evidence → final citation → gold-rationale closure is checked in place.

Initial preparation showed all three replenishable witness targets were in the
initial preview list in both policies (coordinator's private hash-based audit).
This rules out missing previews, **not** inattentive/insufficient previews or
successful action selection. If adaptive still fails, compare actual traces with
fixed-rerank/extra cases that had complete gold rationales visible. Do not infer
Agent benefit if fixed and adaptive improve together, or if only abstentions
increase. No bootstrap significance, causal reasoning benefit, SFT/RL success,
independent-test score, online SLA or current-resume update is supplied here.

## Release checklist and serial submission draft

No GPU/model execution is authorized by this file or by a successful package
build. The coordinator must separately approve the exact source archive, two
data bundle SHAs, bundle-receipt SHA and private protocol SHA after review.
Then verify no duplicate **this-package** allocation exists; never modify or
cancel unrelated account jobs. Run `sbatch --test-only` before actual submission.

The allocation cap per arm is one A100, eight CPU, 32 GiB, two hours, no requeue:
48 questions × 120 s + 1,440 s staging/load/preflight margin = 7,200 s. Prior
F/G peak RSS was below 18 GiB. Jobs must be serial via `afterok`; second-arm
predecessor checks independently reject incomplete or different first results.
Release variables must match the handoff receipt; do not substitute a directory
tree or copy unrelated private data. Do not execute this draft without release:

```bash
# After coordinator release and frozen environment exports only:
sbatch --test-only --export=ALL,CLIMATE_SEMANTIC_POLICY=scifact-gap-original-v1 "$FROZEN_WRAPPER"
first=$(sbatch --parsable --export=ALL,CLIMATE_SEMANTIC_POLICY=scifact-gap-original-v1 "$FROZEN_WRAPPER")
sbatch --test-only --dependency=afterok:"$first" --export=ALL,CLIMATE_SEMANTIC_POLICY=scifact-gap-semantic-policy-v1 "$FROZEN_WRAPPER"
sbatch --dependency=afterok:"$first" --export=ALL,CLIMATE_SEMANTIC_POLICY=scifact-gap-semantic-policy-v1 "$FROZEN_WRAPPER"
```

CPU validation covers only the new seams and frozen-helper regression tests;
the unchanged large/model suites are not rerun for this package. Final source,
package hashes and clean-source targeted validation are recorded in the handoff.

## Initial CPU handoff (superseded for release)

Execution source is **`44b0424abcfed65e4efdeeaf793f0d47ad92f734`**. The
[content-free handoff](verified-runs/scifact-semantic-execution-cpu-handoff-20260930.json)
records source/archive/wrapper identities; the
[bundle receipt](verified-runs/scifact-semantic-bundles-44b0424.json) pins the
formal private protocol and both exact member lists. Source archive SHA is
`3f755fc9e74ec098b3e07dd45f1372669aca8bcb05c5bc9a0a53c622b26ab1ea`;
formal protocol SHA is
`d6a2dafe62bff5f9b548027f63a8b34417181a2b9e2c04f9bced8ed2434ba644`.

All **80 targeted cases** passed, including 33 new seams; the same 80 passed
from the byte-verified clean source archive in 4.02 s. Targeted Ruff, strict
mypy over nine source files, tracked secret/PII scan and local/remote Bash syntax
passed. One tarfile future-default warning is retained; it is not a skipped test.
An initial Windows-CRLF archive failed exact-Git-byte validation and was neither
uploaded nor run; the published archive uses explicit LF and matches all 376
Git blobs plus its single expanded revision marker.

Mechanical conversion ran on the existing private preparation files on Spartan;
it loaded no tokenizer/model and performed no new query analysis. The inference
and scoring archives stay there. Only hashes/counts were downloaded. **This
execution-entry package submitted zero Slurm jobs and made zero model calls**;
the earlier CPU preparation job is separately recorded as 31698106. No GPU
release, new monitor, scientific success claim or resume change follows.

## Preflight completion-journal repair before release

A subsequent independent review found a cost-retention gap: failure to write a
completed-case journal already stopped further calls, but discarded the returned
usage before the wrapper could save it. The minimal repair returns a **failed
partial report** immediately, including the exact known usage. The wrapper tries
one summary write, then stops; it never retries the model. If that write also
fails, `PreflightPersistenceError.partial_report` preserves known in-memory costs
and explicitly records `summary_persisted=false`. Missing durable cost is unknown,
not zero. No-callback historical behavior and started-journal fail-fast remain.
The exception attribute is **in-memory only**: the CLI does not separately emit
it, so if both journal and summary writes fail, exact usage is not recoverable
from the child after exit. Treat that branch's durable total as unavailable/
unknown; do not claim the exact usage was persisted.

Five focused preflight tests passed, including completion-write failure with
both successful and failed summary persistence (synthetic 10 input / 17 output
tokens, exactly one model-stub call). Ruff and strict mypy on the two changed
source files passed. This supersedes the earlier execution freeze **for release**,
not its CPU evidence; preparation source, selected samples, both prompts, budgets
and original artifacts remain unchanged. The revised execution/package identities
are recorded in the repair handoff, with no new model call or allocation.

### Replacement freeze

The [repair handoff](verified-runs/scifact-semantic-execution-repair-99cd9ff.json)
and [bundle receipt](verified-runs/scifact-semantic-bundles-99cd9ff.json) pin
execution source **`99cd9ff707697ea395a9bc067f3ce91395071cdb`**. Its source archive
has 378 exact Git blobs plus the 41-byte expanded revision marker; all modes and
bytes match. The five focused tests also passed from this clean archive in
0.70 s (30 unrelated cases deselected). The earlier 80-case suite is historical
evidence at `44b0424`, not a newly rerun full-suite result at this revision.

Spartan mechanical conversion verified the unchanged private preparation hashes
and produced fresh exclusive inference/scoring bundles. Only the content-free
receipt was downloaded. Remote source/wrapper hashes match local copies and
Bash syntax passes. The `44b0424` files are retained as superseded/unreleased;
no model calls, allocations, new monitor or resume changes were made.

## Separately authorized serial GPU submission

After the CPU repair handoff was pushed, the coordinator separately released
the exact `99cd9ff` source/wrapper and both new bundle identities for one bounded
comparison. The [submission receipt](verified-runs/scifact-semantic-pair-submission-31706518-31706520.json)
records **31706518 → afterok → 31706520**, each with one A100, eight CPU,
32 GiB, two hours and no requeue. The first was running at the single submission
snapshot; the second was pending dependency. Test-only's October 6 estimate was
superseded by the actual September 30 start, not treated as a guaranteed schedule.

The submission script used a retained exclusive release lock, independently
checked scheduler command exit codes and never retried `sbatch`. An initial
unexecuted helper with a read-only query-status guard weakness was retained;
only the corrected `r2` helper ran. Source, model, prompts, selected queries,
budgets and evaluation contracts were unchanged. This is a submission handoff,
not final inference/preflight/quality evidence. The coordinator owns subsequent
status and result verification; no new polling task or resume update was created.
