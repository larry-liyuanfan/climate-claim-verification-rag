# Frozen read opportunities: CPU replay, not Agent performance

## Outcome

The existing bare bounded route preserves all **12/12** initial candidate orders
and visible-sentence/preview identities relative to the frozen semantic-preparation
record. Its original three legal read witnesses still expose a complete official
alternative rationale of at most three sentences in **3/3** cases; all three
add a document-level opportunity absent from the initial citable context.
There was no reselection, substitute witness or search for new phrases.

| Frozen diagnostic stratum | Claims | Existing bare-route result |
|---|---:|---|
| Initial rationale opportunity | 3 | Still available initially |
| Original legal read can replenish | 3 | All three unchanged reads replenish a first3-eligible rationale |
| Gold absent from original Top-20 | 3 | Still absent; rewrite opportunity unknown |
| Official NEI | 3 | Still NEI, not a retrieval opportunity |

Initial first3-eligible gold-document opportunities are 3; after the scripted
reads there are 6. This is **potential evidence visibility**, not predictions,
an increase in benchmark accuracy, or successful autonomous tool use. The
historical semantic adaptive policies still have zero rationalized credit on
the three replenishable claims. Nothing in this replay overwrites that failure.

## Exact replay and limitations

- Execution source: `dcbe368c1c0208c4c7d5323d99b290873a897110`;
  source archive SHA `08a44fb8a5f94bc7599a76de1313f0a6d5498810819254ffa5a58335dd409584`.
- [Content-free compact](verified-runs/scifact-selected-read-opportunity-dcbe368.json)
  binds the original selected-input, gold, witness, corpus and tokenizer hashes.
  Per-claim states/diagnosis stay in Spartan; no original IDs/gold are published.
- The twelve old selected IDs/order are unchanged. Nine scripted abstentions
  and three original read-to-abstain trajectories produce exactly **15 fixture
  responses, zero model calls and zero reranker calls**.
- Reuses `run_bounded_slot(gap=False, arm="format_repaired")`, `PackingProbe`,
  the existing `SciFactBM25`, tokenizer, schemas and budgets. Actual generation
  format is bare, without a G adapter. Budget packing remains the existing
  `CommonPacking=max(bare, empty-G, active)`: it retains the G length bound but
  does not generate a G envelope. It is not pure-bare counting or semantic-pair packing.
- Candidate/source hashes, ordered visibility, full observation/schema/state
  hashes and actual bare-renderer prompt hashes are frozen before selected-gold
  parsing. The exact old alias-to-document targets are checked; no alternate
  target is searched if they differ. Preview text is not counted as citable evidence.
- The existing controller flags an accepted scripted read as `model_selected`;
  this CPU fixture explicitly does **not** turn that flag into real model behavior.
- This is previously gold-preparation-seen, deliberately stratified TRAIN data,
  not an independent test. The twelve all-gold-visible grounding-tune inputs
  are a different experiment and are not substituted here.

The short process used one CPU thread, a 60-CPU-second / 4-GiB process limit and
120-second outer timeout; measured elapsed was 13.17 seconds and MaxRSS 278,084 KiB.
It was a tokenizer/retrieval diagnostic, not a new Slurm job or inference run.
`USE_TORCH=0` intentionally disabled model loading; the tokenizer-only framework
notice is expected and does not mean the installed Torch environment disappeared.
Clean exact-source validation: 46 related tests, Ruff, strict Linux-platform
mypy and tracked secret scan passed. No queued/running evaluation asset changed.

## Minimal later four-route proposal — not authorized or executed

Keep this exact twelve-query selection, existing retriever/corpus, bare bounded
provider, common packing and original budget: candidate 20, context 5, at most
5 calls / 5 tools / 2 repairs, 8,192 input and 512 output tokens per call,
120 seconds per query-route. Reuse the existing routes without new prompts:

1. `fixed_retrieval`: evidence grounding without an extra tool decision.
2. `fixed_rerank`: existing reranker control; no reranker result is measured here.
3. `deterministic_extra`: existing deterministic query plus rerank control.
4. `adaptive`: real model choices through the unchanged read/rewrite/rerank schema.

That is 48 query-route slots for one frozen candidate, not an authorization to
run 48 slots or use the five-call ceiling. Exact execution/resource/call release
and a frozen provider choice are still required. Whether the candidate includes
the trained adapter depends on tune and subsequent separately authorized
validation; this CPU result alone does not select it.

Report real proposal → accepted tool → changed visible evidence → subsequent
decision → final rationalized credit, especially on the fixed three replenishable
claims. Keep absent-Top20 and NEI failures in the same twelve-query denominator;
report every failed/unknown attempt and full latency/cost. A read opportunity
does not prove the model uses it; public records do not establish useful rewrite
actions, so rewrite effectiveness remains unknown. No samples, prompts or search
terms may be added merely to make tool use succeed.
