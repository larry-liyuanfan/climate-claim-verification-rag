# Natural FIT24 and shared-prefix tool utility — CPU preparation

Status: metadata frozen and implementation prepared; no new model execution,
training, model download, protected-split access or GPU submission. A separate
coordinator exact-hash release is required. DRAFT is rejected by the operator.
The negative evidence-note run 31884581 remains unchanged and is not retried.

## Question, population and immutable selection

Can the original unadapted Qwen3-4B model use actual tool feedback to recover
whole-answer/rationale grounding, or does its natural policy fail to use tools?
This is **already exposed TRAIN/FIT**, not independent evaluation.

Use only the existing sealed FIT144 ID/component roster, old12 metadata and
utility8's pre-existing selection. The utility8 selection is authenticated by
its frozen cost → reaped exit → preparation → selection hash chain; its `role`
field is ignored. Never reopen its per-case quality reports to select samples.
The new selector accepts only ID/component projections and rejects extra fields.

- Pool: 144 claims / 144 components.
- Exclude whole old12/utility8 components and preserve prior protected components.
  Eight pool components are removed; 136 claims / 136 components remain.
- Fixed salt `scifact-natural-fit24-v1-20261001` was chosen before counting.
  Rank components by SHA-256 of compact UTF-8 JSON `[salt,"component",component]`;
  choose the minimum SHA of `[salt,"claim",component,id]` within each component.
  Deterministic lexical/ID tie-breaks; first at most 24 components; no quotas.
- Frozen result: 24 claims / 24 components / zero shortfall. No unavailable case
  is replaced. No query, gold label or model outcome enters this choice.
- Private selection SHA:
  `107c50a8ecc1fd4d1b352cdc7e2f86b2997d6abc027d5afeb176bab1d09f7acb`.
- Private freeze receipt SHA:
  `e1862dd449af5019e64f2834d2d50b23ad11b6949b88aa167497e90208b7c416`.
- Selector source SHA (local and remote identical):
  `eced2d4226c6b53aa4bbf56336c1255f1c1337c0426a72d57c36f0517930c44f`.
  CLI SHA at freeze:
  `9d4f4d08f5ebd21bd689e99295768d8e464092edfb37fed7b66d361ba0c58882`.

The original receipt is not overwritten to add fields. Exact source/archive
and the later release bind the selector and frozen selection. All raw IDs,
queries, predictions and private physical response records stay on Spartan.

## One natural trajectory, two controlled forks

Reuse `SciFactBoundedAgent`, original prompt/schema/renderer/CommonPacking and
default V3Budget: five model calls, five tools (including initial retrieval),
120 seconds, 8,192 input / 512 output tokens per call. No evidence-note field.
The base is loaded directly from the existing verified model archive; it is
never wrapped in PEFT. Before each physical generation, record actual class,
eval mode and absence of PEFT config/LoRA modules/parameters, not merely `gap=False`.

| Arm | Execution and origin |
| --- | --- |
| A | Full natural adaptive episode, including existing bounded repair behavior. The initial model proposal does not truncate A. |
| B | From the same genuine initial physical prefix, scripted read of the first preview-only candidate in original recall rank, then at most one new generation. It does not prefer references in the model's answer. |
| C | From that same prefix, scripted rerank of the unchanged candidate set, then at most one new generation. |

Both forks require a complete valid initial physical prefix: matching frame,
prompt/token IDs, source/alias hashes, known usage, legal decision, full stored
raw response, empty grammar receipt and observed EOS. JSON without EOS is not
usable; exactly 512 output tokens with EOS may be usable. No valid prefix means
unresolved branches, not replay, replacement or a later-step prefix.

Read/rerank feedback and actual citable ordering enter the next model frame.
Forks inherit time consumed before the initial decision. A further tool proposal,
illegal response, truncation, timeout or exception is unresolved; no branch
repair, extra generation or hidden terminal call. Scripted events remain tagged
`scripted_intervention`, never `model_selected`.

## Physical budget and resource candidate

Natural maximum: 24 × 5 = 120 generations. Fork maximum: 24 × 2 = 48.
Package ceiling: **168 generations; 48 rerank requests / 960 pairs**.
The serial reranker caps each request at 20 pairs and logs actual forward tokens,
latency including swaps, partial work and unknowns. No warmups or downloads.
Only one model occupies GPU memory at a time.

The initial physical generation belongs to A; B/C reference it logically.
Cost totals count its physical ID once. Initial retrieval counts as a tool;
shared-prefix copies do not count as new tool execution. Failures, prefill,
all tokens and unknown costs remain in the ledger. Earlier unknown cost cannot
be hidden by a later successful natural response.

Candidate allocation: A100 × 1, 8 CPUs, 32 GiB host RAM, 30 GiB scratch, **3 h**.
This is a conservative budget-derived ceiling, not a performance prediction:
72 episode/fork upper bounds × 120 s = 8,640 s, plus 600 s inference loading,
and 1,560 s outer preparation/audit/shutdown margin. Worker cap 9,240 s; inherited
slot watchdogs enforce tighter individual deadlines. Model identity/residency
and resource shape reuse the already-run base/utility transport, not a new model.
Exact release and `sbatch --test-only` are still required before submission.

## Scoring and claims

The inference subprocess must terminate and be reaped. Save unique generation,
reranker and tool costs **before tokenizer loading or gold access**. Interrupted
workers, partial JSON receipts and tokenizer failure still produce cost plus
`no_quality`, retaining all 72 planned slots. Unknown quantities remain null or
explicit lower bounds, never free zero.

After physical auditing, a separate scoring phase opens only the selected IDs
from the already authorized official TRAIN gold. Validation12, official dev300
and retired test are not opened. Nonterminal/invalid slots stay unresolved and
remain in denominators; empty failure is not correct NEI. Official micro evidence
F1 is not a measure of valid NEI abstention.

For an initial direct answer/abstention, a wrong → whole-answer/rationale-correct
scripted fork is a **recoverable program-teacher opportunity**. For an initial
tool choice, keep A's complete natural endpoint as comparator; do not attribute
a multistep strategy difference to one causal action. Candidate records reference
actual tool-change/feedback artifacts and require later review. They are not
human annotations, proof of autonomous Agent success, or training authorization.

## Reproduction and regression coverage

- `freeze_scifact_natural_metadata.py`: small pinned metadata only. The real
  freeze has already happened once; do not rerun `--freeze` on its reserved path.
- `prepare_scifact_natural.py`: reads frozen selection before selected query
  decoding, never reselects or reads gold; creates a separate inference bundle.
- `run_scifact_natural.py` / `run_scifact_natural_operator.py`: exact-release-only
  inference, using shared loader, physical ledger and original controller.
- `score_scifact_natural.py`: separate post-reap physical audit / TRAIN scoring.
- `package_scifact_natural.py`: exact Git archive and actual wrapper positive/
  negative guards. `hpc/scifact_natural.sbatch` never submits or retries itself.

Synthetic tests cover component exclusions/order/shortfall, rejecting selection
labels, full natural multistep execution, at-most-one fork generation, actual
feedback/citable changes, shared cost, no EOS, invalid initial prefix, no LoRA,
unknown prior usage, raw tampering, strict recovery vs strategy comparison,
failed-NEI denominators, timeout, lazy-tokenizer failure and partial result writes.
Legacy utility8 default limits and semantics remain separately regression-tested.

No outcome or resume improvement is claimed by this CPU package.
