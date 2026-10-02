# Equal-capability acquisition: implemented CPU package

## Decision and scope

This extends accepted CPU/recovery commit `101eb1c`; it does not repeat the
historical `54031fa` model run or its scoring. All 160 historical slots remain
recovered. The observed 32/32 autonomous stops are retained. Different historical
rerank capabilities and final-answer contracts limit interpretation, but do not
erase that behavior. No Spartan access, other project work or resume edits occur.

**Falsifiable next decision:** with equal initial evidence, optional reranking and
a common verifier, does autonomous acquisition actually acquire and consume
additional complete evidence, and does its whole-policy task/quality-cost tradeoff
justify preferring it over strong fixed or deterministic controls? More activity
without better measured task/proxy-quality is not a promotion. Continued stopping
is a valid negative result. No result threshold is adjusted after observing this
roster. Semantic citation correctness remains unmeasured: ID concordance cannot
justify promotion as a proven grounded-answer gain.

## Actual architecture and entry

```text
source/release/model/corpus/cohort identity + private storage/capacity preflight
  → one original-claim BM25 Top20 + common complete-text/preview frame
  → fixed plan | deterministic workflow | optional autonomous gate
  → shared typed read / RRF-query / full-candidate Qwen3 rerank
  → proposal → validation → charged execution → actual context commit
  → complete evidence + empty/error history in next physical model prompt
  → common answer/abstain verifier (no terminal tools)
  → supervisor reaps worker → physical cost → isolated scorer → private recovery
```

Production entry `scripts/run_cloud_replay.py` reuses the existing supervisor,
private-output contract, independent total deadline, destination disk check,
worker projection and post-exit scoring. It calls the opt-in implementation in
`scripts/run_targeted_replay.py` → `targeted_replay.run_matrix` → `SentenceAgentV3`.
This is not a disconnected consistency checker or placeholder runtime.

| Shared contract | All three routes |
|---|---|
| Initial information | Original immutable claim, BM25 Top20, first-five complete sources and capacity-limited previews; packed against both policy prompts with hash-token reserve; no answer masking |
| Tools | Same read, query/RRF k60 and rerank executor/format; empty result and recoverable error preserve state |
| Models | Qwen3-4B generator + Qwen3-Reranker-4B; pinned revisions/manifests and common 5,240-document corpus |
| Final decision | Same JSON/schema/prompt, SUPPORTS/REFUTES with at most three displayed unique sentences, or insufficient/conflicting-evidence abstention |
| Ceilings per slot | 5 physical generations, 5 tools including initial retrieval, 2 queries, 2 validation repairs, 8,192 input tokens, 512 output tokens, 120 seconds |
| Cost | Actual plan/gate/repair/verdict calls, every charged tool, rerank pairs/tokens/swaps, failures/unknown receipts and operator wall time; identical caps do not imply equal spend |

Fixed multiquery freezes a strong model plan using the common initial frame,
optionally reads up to five unread initial sources, executes up to two planned
queries, then reranks. Deterministic workflow uses the existing two-query
transformation, initial reads and rerank. Autonomous may read/query/rerank or
legally stop; stop is not an evidence-sufficiency verdict. There is no query after
stop, and a valid terminal decision ends generation. Query/rerank tools do not
have hidden free retries.

`fair_replay.audit_state` reconstructs phase, tool arguments, availability,
fixed execution order, source hashes, monotone context commits and exact newly
visible sentence increments. It verifies those complete sentences in a later
physical prompt. This proves delivery/continuation **opportunity**, not that a
real model used them correctly or that feedback caused an improvement. The
whole-policy comparison has no separate feedback-isolation ablation.

## Data qualification and frozen scoring

Only the registered public-v2 validation member and pinned inference corpus were
read. The sealed test and full prepared source were not opened. All validation
was already exposed to retrieval/model-selection work: this is explicitly a
**development policy comparison, not independent validation/test**.

The preparer projects claim text and **all** annotated candidate IDs before
selection. It reconstructs the original exact-normalized / .90 Jaccard claim and
evidence graph, excludes every component touching the canonical consumed32, and
selects one SHA-ordered claim per remaining component using a single fixed salt.
There are no label quotas, answer masking, favorable-opportunity filters,
replacement samples or alternate salts. The trusted preparer parses the official
annotated validation export; the correct claim is **label-blind selection**, not
that gold was never parsed. Scorer gold is exported only after roster freezing,
and never included in worker tasks, prompts, feedback or selection strategy.

- 230 validation IDs → 187 components; 31 consumed-overlapping components excluded.
- 156 eligible components → fixed 32 tasks, 96 route slots, at most 480 generations.
- The exact-normalized graph review repair changed neither graph nor roster;
  normalized-empty claim/evidence counts were both zero.
- Registered private cohort SHA: `f2b7c55b1e91f297a613df5762f44d46c2df7666cca8e234d77d5ca53488ec56`.
- Exposure-audit SHA: `999eda72085e34ab001fa1f8b8f4e4a5cd861adec699d91be9af76ccd60e6b8a`.
- Pre-model contract rebinding added the secondary retrieval metrics already
  specified in the preceding design and corrected their candidate-ranking scope
  during review; roster, tasks and exposure identity were unchanged. Exact byte
  hashes are in the final source/release receipts.

All N tasks per arm remain the denominator for task correctness, outcomes and
joint proxy. SUPPORTS/REFUTES use official exact labels; NEI requires an intentional
insufficient-evidence abstention, DISPUTED an intentional conflicting-evidence
abstention. Errors, deadlines and budget exhaustion are failures, never automatic
correct NEI. Incomplete matrices are not scored as complete experiments.

Citation schema/provenance legality is deterministic. The automatic citation
proxy requires every cited source to occur in official decisive evidence IDs;
**this is not entailment of the specific verdict**. Joint proxy requires correct
task AND nonempty ID-concordant citations; abstention gets zero joint proxy and
separate task correctness. There is no human blind annotation, author entailment
assessment or automatic semantic judge. True semantic citation support and true
joint correctness remain unmeasured, not fabricated zero-valued results.

Secondary Recall@5, MRR@10, nDCG@10 and Top5 Evidence F1 use final ranked Top20
candidates on official evidence-bearing tasks, including their empty/error
outcomes. The legacy field `delivered_evidence_ids` names these candidates and
includes preview-only/not-read sources: it is **not complete-text delivery**.
Actual full-text delivery uses the separately audited sentence-context receipts.
The retrieval denominator is separate from all-task correctness; retrieval F1
is neither citation nor semantic F1. Paired bootstrap uses 5,000
resamples, seed 20261003, one frozen component-selected query per unit, comparing
autonomous separately with each control. Cost vectors retain all physical calls
and reranker requests; unknown usage is not silently free.

## CPU verification and review

Synthetic fixtures substitute only the model boundaries; the test invokes the
actual production worker and three-arm matrix. They verify common initial/final
contracts, feedback-dependent subsequent acquisition/answer, real rerank delivery,
empty/error state preservation and charging, no post-stop tools, terminal-call
budgeting, source/run/model/cohort refusal, scorer isolation and fabricated-trace
rejection. They establish implementation behavior, **not real-model effect**.

```powershell
.venv-validation/Scripts/python.exe -m pytest -q tests/test_fair_acquisition.py
.venv-validation/Scripts/python.exe -m ruff check src scripts tests
.venv-validation/Scripts/python.exe -m mypy --platform linux --follow-imports=silent src/climate_rag/fair_acquisition.py src/climate_rag/fair_replay.py src/climate_rag/agent_v3.py src/climate_rag/targeted_replay.py src/climate_rag/targeted_score.py scripts/prepare_fair_cohort.py scripts/run_targeted_replay.py scripts/cloud_replay_contract.py scripts/cloud_destination.py scripts/cloud_private_storage.py scripts/run_targeted_replay_operator.py scripts/preflight_cloud_assets.py scripts/package_cloud_replay.py
```

Affected local suites passed 120 tests with 40 native-Linux skips before the final
audit delta; final delta checks and same-source Linux CI are recorded separately
in the delivery receipt, not replaced by an old CI. Full unchanged local suites
were not rerun. Exact source clean-archive reproduction is likewise recorded there.

Independent read-only review identified and repaired common-frame capacity,
Unicode identity, empty-pool rerank, false-abstention outcomes, missing canonical
consumption binding, state/argument reconstruction, newly-delivered sentence
forgery and terminal-verdict continuation. Data review repaired normalized-empty
graph parity. Exact reviewed HEAD and remaining limitations are recorded in the
final private review/handoff receipt; a prior review does not authorize later code.

## Executable draft, not execution authority

`package_cloud_replay.py --fair-binding <binding.json> --fair-gold-sha256 <hash>`
packages a clean exact commit and emits `release.unauthorized.json`. Missing
scorer identity is rejected. The package binds source archive/entry, cohort and
exposure, contracts, generator/reranker, corpus, single run ID/output, allocation,
limits and post-exit scorer. Worker projection contains no gold or scoring path.

Proposed single run: `fair-three-arm-20261003`, existing stopped Runpod Pod
`udm7fa2t6smwgl`, single A100 SXM; observed allocation 18 vCPU / 286 GB RAM,
30 GB container and 120 GB persistent volume. Existing namespace/private-POSIX
contracts are reused. Assets and pinned dependencies must pass destination checks
before model execution; no paid instance starts while local repairs are unfinished.

Public persistent source/models stay under `/workspace/climate-replay`; private
cohort/output/scorer stay under `/root/climate-private` with owner-only permissions.
Container output must be privately downloaded and SHA-verified **before stopping**
compute; it is not assumed to survive a restart or to be mountable by another Pod.
Run output is unique: `/root/climate-private/runs/fair-three-arm-20261003`.

Fresh storage/assets/runtime receipts and exact user release remain necessary.
The former release is not inherited. No GPU, actual CPU model, paid API or new
private upload was executed in this package. No Pod/volume was deleted.

## Business case and two-minute STAR

**Situation:** climate-claim checking needs sufficiently relevant evidence and
correctly bounded conclusions; a working tool call or valid citation ID alone
does not make an answer trustworthy. Historical autonomous acquisition stopped
on all 32 tasks and did not outperform the strong control.

**Task / personal contribution:** implement a fair, auditable strategy comparison
that can distinguish weak retrieval, evidence delivery, failed feedback use and
terminal judgment, while keeping prior negative results and private data intact.

**Action:** I connected three policies to one typed executor and verifier, froze
component-disjoint development tasks without outcome selection, separated gold
from inference, journaled physical calls/cost and verified source-to-prompt text
delivery. Independent review found trace-forgery and terminal-continuation gaps;
I fixed and regression-tested them before preparing an exact offline Runpod package.

**Result now:** an executable CPU-tested 32-task / 96-slot comparison package and
an explicit decision protocol, not a claimed Agent uplift. Real-model behavior,
semantic citation correctness and quality-cost promotion still require separately
authorized execution and result acceptance. The business value is evidence-based
quality/cost model selection and prevention of unsupported claim-checking claims;
time savings, production use and cost reductions are not yet measured.

Candidate wording, to be used only with these boundaries:

- Built a reproducible climate-evidence verification experiment with shared typed
  read/query/rerank tools and a grounded verifier; froze a leakage-controlled
  three-policy development comparison and audited physical evidence delivery,
  failures and complete inference costs.
- Diagnosed and preserved an all-stop autonomous-acquisition negative result,
  replacing mismatched controls with equal-capability baselines and independent
  review; prepared paired quality/cost scoring without claiming unmeasured semantic
  correctness or Agent gains.
