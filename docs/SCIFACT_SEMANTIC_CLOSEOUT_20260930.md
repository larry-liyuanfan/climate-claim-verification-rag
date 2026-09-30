# Semantic-policy comparison: completed, no Agent benefit established

## Decision and scope

Stop this prompt comparison. Both adaptive routes produced **zero raw tool
proposals and zero model-selected tool events**. The new semantic instructions
did not increase correctly rationalized adaptive evidence documents. The next
useful question is grounding under sufficient context, not another zero-shot
prompt matrix or immediate action SFT. See the separate
[CPU-only feasibility proposal](SCIFACT_GROUNDING_FEASIBILITY_20260930.md).

This is twelve deliberately stratified eligible-TRAIN claims, four routes and
two prompt policies: 96 slots. It is not independent test, an estimate over the
benchmark population, online A/B, or evidence of causal Agent benefit. The
preparation already inspected gold for all 531 eligible TRAIN claims. The former
restricted 1,208,827-document/154-query retrieval/LoRA evidence, CLIMATE-FEVER
frozen-test history, synthetic fixtures and this SciFact diagnosis remain
separate; no earlier retrieval success is overwritten by this diagnostic.

## Identities and verification

- Execution: `99cd9ff707697ea395a9bc067f3ce91395071cdb`; [release/submission](verified-runs/scifact-semantic-pair-submission-31706518-31706520.json).
- New collector: `2182f61fae66a57ae47d7ec315edf93d061b7dc3`; script SHA `b19e7624c1733406e9e98ac7e5651d95742cf8e27ba7f667205c00d92324152a`.
- [Content-free compact](verified-runs/scifact-semantic-pair-closeout-31706518-31706520.json): **67,109 bytes**, SHA `f25a408ad8be651aca3007f0d379fe99b7ed268e5a37bd6a044c72a208af27ec`.
- Original/semantic run SHA: `557b7b8ddc7f4e40c1ac1abbe2abd68af37536427f3b106e9e0063e3f6ae746a` / `486f1c5866011e926b47fe2848189ce4c121213ab7fc86fc22a2c18ea68403f8`.
- Frozen score SHA: `5cee393c8d5c671def659885d12d3f663465cc633e4f6d0eab83c06fde07fa73`.

The completed frozen GPU scorer executed `physical()` and
`audit_slot(..., gap=True)` for both policies, including real synthetic
preflights, actual-visible source hashes, raw decisions, event links and final
evidence. The new stdlib collector verifies that exact score/run/source chain,
rechecks physical wire multiplicity, durable slot equality and known-cost sums,
then independently recomputes all **32 official metric groups** using the frozen
independent `rescore` function. Correct/predicted/relevant counts and P/R/F1 match.
It does not rerun models or pretend to reexecute the full Pydantic audit on CPU.

The coordinator separately recomputed the same official metrics and visibility
counts. Its stdin-only script-text hashes are
`72167b8fcb1eb988c9bdbcbd60cf1e45e76bc83d5dc5c40ee5ff64366270b49e`
and `a63f51c49b244844c1da4e61b82df4b07217c78ae42aa2889c39e92f4a993203`.
Those are **not disk artifact hashes or downloadable files**. The checked-in
collector and compact provide the reproducible project artifact.

## Quality: counts before percentages

Every row has nine relevant gold documents across twelve claims. Official
`abstract_rationalized` credit requires a correct document relation and one
complete alternative rationale inside the first three predicted sentence IDs.
It is not claim-verdict accuracy. The original scorer is unchanged.

| Route | Label-correct docs, old → new | Rationalized correct / predicted / relevant, old → new | Rationalized F1, old → new |
|---|---:|---|---:|
| Fixed retrieval | 2 → 2 | 1/27/9 → 1/35/9 | .055556 → .045455 |
| Fixed rerank | 4 → 3 | 2/34/9 → 3/34/9 | .093023 → .139535 |
| Deterministic extra evidence | 3 → 2 | 1/33/9 → 2/33/9 | .047619 → .095238 |
| Adaptive | 2 → 2 | 1/36/9 → 1/30/9 | .044444 → .051282 |

The adaptive F1 increase comes from fewer predicted documents, **not another
correct rationalized document**. Its custom strict whole-answer count changed
1/12 → 2/12 solely through valid NEI abstentions 1 → 2. These are not new evidence
discoveries or official claim accuracy. New fixed-rerank rationalized credit
increased despite fewer correct document labels; that is rationale placement,
not autonomous retrieval. No confidence interval/significance claim is offered
for this small, selected TRAIN diagnosis. Full sentence metrics and legacy
stratum counts are preserved in the compact, not hidden in a best-route summary.

## Where the pipeline failed

The denominator here is a gold document with at least one **complete, ≤3-sentence
alternative rationale actually visible**. It is a first3-credit-eligible
opportunity, not every terminal-legal rationale: the wire allows up to eight
sentences. Nor does one visible gold document imply all evidence for a multi-doc
claim is sufficient.

| Route / policy | First3-eligible gold docs visible | Correct label / wrong / omitted | Complete rationale anywhere / within first3, among correct-label docs |
|---|---:|---:|---:|
| Adaptive / original | 3 | 2 / 1 / 0 | 2 / 1 |
| Adaptive / semantic | 3 | 2 / 1 / 0 | 2 / 1 |
| Fixed rerank / original | 6 | 4 / 1 / 1 | 4 / 2 |
| Fixed rerank / semantic | 6 | 3 / 1 / 2 | 3 / 3 |

New adaptive emitted 30 document predictions: **27 did not match official gold**,
and 29 selected more than three sentences. New fixed-rerank emitted 34: 30 did
not match official gold, and 28 selected more than three sentences. Unannotated
does not prove factually incorrect; these are official-annotation mismatch and
over-selection diagnostics, not a human hallucination adjudication.

Code/interface review reported by the coordinator excluded a forced-answer
branch, disabled first-step tools, automatic sufficient state, automatic sentence
completion/sorting, thinking-template conflict and gold injection. Its small
grammar probe found all five actions could reach EOS; changing `anyOf` order did
not change the action-position allowed-token set. This is a targeted interface
check, not proof of model competence. Remaining **unproven hypotheses** include
greedy/nonthinking behavior, the 512-token full envelope and autoregressive
source→label→sentence ordering. They do not justify another sampling matrix yet.

## Cost and resources

| Policy | TRAIN calls / input / output tokens | Separate real synthetic preflight calls / input / output | Slurm elapsed | Batch MaxRSS |
|---|---|---|---:|---:|
| Original | 48 / 191,052 / 11,296 | 4 / 4,260 / 535 | 629 s | 18,541,472 KiB |
| Semantic | 48 / 200,604 / 11,302 | 4 / 5,056 / 535 | 627 s | 18,489,180 KiB |

Both jobs completed `0:0`: 31706518 at 23:00:49–23:11:18 and 31706520 at
23:11:22–23:21:49 on September 30, Sydney time. Each requested one A100, eight
CPU, 32 GiB and two hours. TotalCPU was 613.527 / 611.512 s; allocated CPUTimeRAW
was 5,032 / 5,016 s. GPU allocation elapsed is not measured utilization.
Unknown usage, validation repairs and failure/budget terminations were all zero.
This does not remove the implementation's unknown/unpersisted-cost failure path.

Adaptive P50/P95 were 10.289/12.288 s → 10.323/11.997 s. Fixed-rerank was
11.105/11.963 s → 11.118/12.258 s. These twelve-query offline timings are not
online SLA; model loading, tool time, question time, operator time and Slurm
time overlap and must not be added. No API bill, energy use or ROI is inferred.

## Exposure inventory and reproduction

Eligible TRAIN has **531 claims / 325 components**, not the earlier 571
train+dev joint components. Deduplicating the old and current model attempts
gives 24 consumed claim IDs / 24 excluded components; excluding entire components
removes 40 eligible claims. **491 claims / 301 components remain**, with zero
uncertain exposure. The old 508-claim remainder was a pre-semantic snapshot.

Remaining labels: SUPPORT 199, CONTRADICT 94, NEI 198; 328 annotated claim-document
pairs and 584 alternative rationale sets, of which 583 have ≤3 sentences.
The long alternative does not automatically disqualify its document if a shorter
alternative exists. Legacy strata are initial 276, replenishable 7, absent 9,
NEI 198 and unclassified 1. These historical packing labels are not new coverage
measurements. No new split, sample selection or packing was performed.

On Spartan, with the original immutable artifacts present, invoke
`python3 scripts/summarize_scifact_semantic_closeout.py --output <new-exclusive-path-under-ROOT/posthoc>`.
It writes aggregates only and refuses to overwrite. The original runs, score,
submission lock and source/data archives stay unchanged. Raw gold, text and
predictions are not part of the committed artifact.

Eleven synthetic export/ledger/error-layer checks, targeted Ruff and strict mypy
passed; the same eleven passed in 0.07 s from an explicit-LF clean archive of
the collector and tests, with the collector's bytes/hash identical. Tests reject
raw claim fields and per-item arrays and preserve an explicit
unclassified bucket. The former full model suite was not rerun. This closeout
submitted no Slurm job, invoked no model, opened no dev/test data and did not
change the current resume.
