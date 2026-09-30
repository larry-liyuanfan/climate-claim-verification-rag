# Action protocol and corrective feedback: development v2

This is a newly authorized development package, not a reinterpretation of the
[negative frozen v1 comparison](BUDGET_AGENT_FULL_CLOSEOUT_20260930.md). Old
source archives,120 outputs, frozen scores and the consumed public test remain
unchanged. No deployment, RL or independent-test gain is claimed.

## Literature to implementation, not borrowed benchmark results

- [CRAG](https://arxiv.org/abs/2401.15884): retrieval-quality feedback can select
  corrective retrieval actions. Here we expose coverage and tool-result deltas;
  we do **not** reproduce CRAG's trained retrieval evaluator or web search.
- [Self-RAG](https://arxiv.org/abs/2310.11511): relevance and generation support
  need separate assessments. Its learned reflection tokens/training are not
  implemented here; a prompted action protocol is not Self-RAG training.
- [XGrammar](https://arxiv.org/abs/2411.15100): structured-decoding infrastructure
  addresses output syntax. V2 currently uses a discriminated JSON schema plus
  explicitly charged feedback/repair, **not grammar-constrained sampling**.
  Passing schema or exact-quote checks still does not demonstrate entailment.

## Explicit changes from legacy-v1

1. Four strict wire variants: rewrite carries query, rerank carries no query,
   answer requires sufficient assessment/label/citations, abstain carries an
   insufficiency/conflict/unknown assessment. Extra fields are rejected, even
   null placeholders. No field stripping, coerced action or automatic JSON repair.
2. A schema/quote/constraint rejection yields bounded untrusted prior-output
   context and safe error locations to a **new generation**. At most2 repairs,
   all sharing the5-generation overall cap, per-call token and120s row budgets.
   Known failed tokens count; unknown usage is never replaced by a zero-cost claim.
3. Each generation has its own attempt ledger; every tool has before/after
   candidate/context state and model-selected versus controller-fixed provenance.
   Rank calls are counted separately in pairs and time, not hidden in LLM tokens.
4. A lexical gap is no longer a hard semantic trigger. Adaptive can rewrite once
   even after rerank; each tool type is bounded, a final generation is reserved,
   disallowed repeats never execute. No-new-ID results feed back rather than
   immediately terminating. Same-set new ordering still participates in RRF.
5. Source IDs/text cannot change under the same identity. Candidate previews are
   untrusted, explicitly non-citable; only full current context may be cited.
   Rejected quote drafts identify statement positions, but the model must produce
   any corrected quote itself. Original rejected outputs stay in private records.
6. Per-task/route owner-only directories preserve the <=5 original responses
   without the old32-file study-wide cap. Incremental progress is private and
   labelled partial/not resumable. Final matrices still must be complete.

The original claim is immutable. For this first pilot the conservative query
guard still requires original numeric/entity/qualifier retention and50% lexical
overlap. This can obstruct subquestions or counter-evidence search; record it
as an architectural limitation, not justification to allocate more GPU. A later
targeted-search protocol must separate query purpose from final claim validity.

## Frozen development pilot, not a fresh holdout

`configs/budget_agent_feedback_pilot_20260930.json` contains6 **already exposed**
authored questions from the3-question pilot and8-question vNext set. They cover
an easy supported topic, multiple evidence, empty retrieval, combined sea-level
causes, an unsupported numeric claim and infrared/negation. No new human labels
were created. All230 CLIMATE-FEVER validation claims are already consumed; the
other32-selection complement must not be called fresh. Old test stays closed.

Each question runs fixed retrieval, fixed4B rerank and adaptive with the same
Qwen3-4B generator, public5,240-document corpus,20 candidates and5 context items.
Limits are identical (5 generations,2 repairs,3 tools,8,192/512 input/output
tokens per call,120s per row), not equal actual work. V1 and v2 differ in protocol,
feedback, loop rules and maximum generations, so this pilot is not an isolated
single-factor causal estimate. No sample is forced to use a tool for appearance.

Four separate acceptance layers:

- **Protocol:** legal attempts, eventual legal terminal actions, repair exhaustion,
  exact quote/ID validity, all failures/budgets/costs retained.
- **Actual agency:** parsed tool decision -> completed model-selected tool ->
  tool feedback -> next model decision. Fixed controller rerank does not qualify.
- **Retrieval:** initial/final IDs and context changes are descriptive here;
  no official gold and no fabricated Recall/bootstrap in this authored pilot.
- **Answer support:** quote occurrence is mechanical only. Any qualitative
  review is assistant-authored, not blind human or independent-model annotation.
  No independent semantic success metric is published from this pilot.

The new `score_budget_agent_feedback.py` understands repair exhaustion and checks
the per-attempt token sum against row totals. It does not modify or repurpose the
frozen v1 scorer. It exports aggregate counts/costs only; raw outputs and drafts
remain on Spartan. A negative real pilot is a negative result, not a reason to
silently change its protocol and call the next run a holdout.

## Resources and one-shot execution

CPU validation before release:328 tests passed,2 optional dependencies skipped
(Torch and LightGBM absent in this local environment); Ruff and strict mypy on
38 source modules passed. An additional nonrepairable-decode accounting test
was added and checked separately. Neither fixture success nor local dependency
skips replace the allocated real-runtime preflight.

One isolated `climate-feedback-v2-20260930-pilot-r1` run. The wrapper pins one
A100,8 CPU,32GiB RAM,30GiB scratch,45min, no requeue. Prior120 one-generation
slots took701s; this pilot has at most90 generations plus bounded tool work and
preflight.45min provides an explicit conservative margin, below the first
package's cumulative2 GPU-hour ceiling. Job scheduling time is not guaranteed.

Use `sbatch --test-only` before the sole actual submit. The new operator reuses
SHA-verified frozen model/runtime bundles, extracts once on the allocated node,
performs the tokenizer/dependency/model-hash preflight without generation, then
runs the new development source/protocol and separate v2 audit. No data/model
compute occurs on the login node; no other user's job is modified. Exact source,
protocol, model/input hashes and job ID must accompany the eventual result.

## Next evaluation contract — proposed, not executed

After pilot review, use official SciFact train for development and its labelled
dev split for one external held-out evaluation, **subject to a verified grouped
leakage audit and coordinator protocol approval**. Audit claim variants/shared
source documents/rationale families, preserve SUPPORT/CONTRADICT and sentence
indices/alternative rationale sets. `cited_doc_ids` are not gold evidence.
BEIR's test mapping may be the same labelled dev: ID/claim hashes must be checked,
and it must not be advertised as a second independent test. Official unlabelled
test is not required and is not accessed. Any contaminated grouped overlap must
be excluded or explicitly redesignated development before the freeze.

Pre-register label-independent sentence/context assembly and limits for generator
and reranker; record truncation/overflow, not gold-selected sentences. Freeze all
model/prompt/policy/budget/data hashes before evaluation. Compare fixed retrieval,
fixed rerank, deterministic extra-work control and adaptive; pair bootstrap by
claim/source family, primarily adaptive versus fixed rerank. Primary delivery
must jointly match the official verdict, document and sentence-rationale evidence;
also report coverage, answer-conditional accuracy, NEI false answers, initial/final
retrieval and all actual costs. Free-text entailment remains unmeasured without a
separate qualified assessment. Holdout may run once, not through iterative tuning.
