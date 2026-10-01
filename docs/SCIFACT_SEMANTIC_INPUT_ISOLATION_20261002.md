# Verifier semantic-input isolation and consumed TRAIN24 diagnosis

## Decision

Keep the versioned verifier-input repair; **do not claim a model-quality gain**.
The same claim, visible document and schema now produce identical observations,
rendered prompts and token IDs across arm, prior-call and remaining-budget states.
CPU job **31963271 completed in 16 s**, with **1,440 arm/counter-state checks
over 120 document probes** (including reference states), 120 clean-history checks,
72 physical-record replays and zero new
model calls. The old protocol and its negative confirmation remain reproducible.

For the consumed cohort, all **18 claim-document annotations across 15 positive claims**
were retrieved and their complete rationales visible. The next implementation
priority is **verifier label and minimal-rationale selection**, not more retrieval.
Multi-document continuation is a secondary, observed controller gap. This CPU
diagnosis does not estimate how many errors a new model or policy would recover.

## Scope and input defect

Only the already consumed conditional TRAIN24 from [job 31956320](SCIFACT_PROSPECTIVE24_CONFIRMATION_20261002.md)
was inspected. Its 15 positives include ten common top1/adaptive errors. Earlier
gold preparation had seen all 531 eligible TRAIN rows: this is neither an
independent test nor a fresh validation set. No new roster, protected split,
training, inference, deployment or resume revision was performed.

The existing `case_09` top1/adaptive verifier calls shared the same document,
claim, visible sentences and schema, but exposed remaining physical generations
of 5 versus 4. Their rendered prompts and token IDs differed despite equal
2,827-token lengths. Historical prompt hashes were
`9822cf2f9fcb68b75c76f3a888c476c20bc469bdf96a1f71cdf658ac13d56768`
and `e93ad415075dae295dbf83c262a6e33399d6d357d3aed87024431aae5c81d1c3`.
This is a comparability defect, not proof that it caused the observed quality
gap. Known `do_sample=False`, `num_beams=1` and shared grammar do not reconstruct
the full inherited generation configuration or seed; those remain unknown.

## Versioned implementation and compatibility

The old default `scifact-evidence-commit-v1-20261002` is unchanged. The new explicit
`scifact-evidence-commit-semantic-v2-20261002` projects verifier inputs to an
allowlist: protocol, stage, immutable claim, currently citable sentences and
source. It omits control-budget and feedback fields, while retaining the same
output schema. Planner inputs still expose actual remaining budgets and feedback.
Physical call caps, token limits, timeout charging and shared accounting remain
enforced; no budget is reset or fabricated by this projection.

The release, worker, result, journal and audit bind the selected protocol.
Unknown protocols and cross-version replay are rejected. New execution uses a
separate output/attempt identity while reusing pinned old input frames. Preparation
source identity and execution source identity are separate: the old compact is
not rewritten to match new code. The release identifies the decision-policy
baseline plus `semantic_verifier_input_isolation_v2`, not a falsely unchanged
complete policy. A packaged draft is not GPU authorization.

## Whole-chain review before submission

Review covered renderer → operator/release → packager → worker → physical ledger
→ audit → diagnostic gold whitelist → report. Two independent static reviews
were completed before the sole CPU submission. They caught and closed:

- preparation/execution source coupling in the versioned packager;
- diagnosis reading a result without replaying its physical-response binding;
- an unauthenticated timing sidecar used as the Top20 candidate source.

The final diagnostic uses all 20 aliases in each authenticated initial frame,
and runs the existing `audit_episode` on all 72 episodes before gold access.
Only the original 24 selected IDs are decoded as gold through the existing
whitelist loader. It recomputes strict booleans solely to require equality with
the immutable original score, never to replace that score.

Validation at execution source `58975a914b3bb0f224db366b519fe267feb0fbbd`:

- 61 affected tests across the targeted checks, including the added packager test;
  old/new success paths, unknown/schema/deadline failures and charged budget caps;
- real frozen-tokenizer synthetic invariance probe; inherited generate/grammar
  integration tested for both old and new protocols with synthetic model inputs;
- 23 tests from a clean exact-source archive, CLI argument/import checks and Bash
  syntax; actual Linux entry-point import/help before Slurm submission;
- [Linux CI](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36890911422):
  **1,342 passed**, Ruff, strict source type checks and tracked secret/PII scan.

Software tests establish executable contracts, not scientific correctness.
No unchanged full suite was repeatedly run locally after documentation edits.

## CPU result and resource receipt

Job **31963271**, COMPLETED / `0:0`, 2026-10-02 **02:20:03–02:20:19 UTC+10**.
Elapsed **16 s**, TotalCPU **9.220 s**, MaxRSS **244,588 K**; 2 CPU / 8 GiB / ten-minute
ceiling, no GPU. One submission allowed `sapphire,cascade`, actually allocated
`sapphire`, following the cluster's capacity notice and `sbatch --test-only`.
No queued job was canceled or resubmitted. Tokenizer-only execution intentionally
sets `USE_TORCH=0`; the library's no-model-backend warning is not an install failure.

Each of 24 frames contributes five visible document probes; each document is
checked across three arms and four counter states. All **1,440 state checks** pass
for observation, schema, rendered prompt and token-ID equality; this count includes
each document's reference-state self-check. A separate clean-history/episode
comparison also passes for all 120 document probes. Maximum verifier input
length was **3,096 tokens** in these probes, not a bound on every planner state.
Legacy control sensitivity and the planner's actual remaining budget are preserved.

## Error decomposition: coverage is present, usable verdicts are not

All 15 positive claims / 18 claim-document annotations have zero initial Top20 misses,
zero retrieved-but-invisible documents, zero missing complete visible rationale,
and zero intrinsically unreachable first-three-sentence rationale. This is only
this consumed cohort; it does not establish perfect retrieval on new claims.

Below, cells are **affected claims / annotated claim-document pairs** unless stated otherwise;
document counts are not deduplicated across claims. All-positives denominator is
15 claims / 18 annotated pairs. The ten common errors have 13 annotated pairs.
Categories overlap and must not be summed as
mutually exclusive failure percentages. Extra sentences/documents mean annotation
mismatch, not independently established semantic falsehood.

| All 15 positives | Fixed top1 | Fixed all | Adaptive |
|---|---:|---:|---:|
| Annotated documents with valid verifier judgment | 14 / 14 | 15 / 18 | 15 / 15 |
| Label disagrees with annotation, including INSUFFICIENT | 3 / 3 | 4 / 4 | 4 / 4 |
| INSUFFICIENT on an annotated document | 1 / 1 | 1 / 1 | 1 / 1 |
| No complete rationale in first three selected sentences | 2 / 2 | 3 / 3 | 3 / 3 |
| No complete rationale anywhere in selected sentences | 1 / 1 | 2 / 2 | 2 / 2 |
| Selected sentences outside annotated rationale union | 7 / 7 | 8 / 9 | 8 / 8 |
| Visible, reachable gold document not attempted | 3 / 4 | 0 / 0 | 2 / 3 |
| Strict-correct document verdict omitted from final | 0 / 0 | 0 / 0 | 0 / 0 |
| Strict-correct document verdicts available | 5 / 5 | 7 / 7 | 5 / 5 |
| Complete strict-positive answers (claims only) | 5 / 15 | 3 / 15 | 5 / 15 |

| Ten common top1/adaptive errors | Fixed top1 | Fixed all | Adaptive |
|---|---:|---:|---:|
| Label disagreement | 3 / 3 | 4 / 4 | 4 / 4 |
| First-three rationale missing | 2 / 2 | 3 / 3 | 3 / 3 |
| Full selected rationale missing | 1 / 1 | 2 / 2 | 2 / 2 |
| Selected unannotated sentences | 7 / 7 | 8 / 9 | 8 / 8 |
| Reachable annotated document not attempted | 3 / 4 | 0 / 0 | 2 / 3 |
| Correct verdict omitted from final | 0 / 0 | 0 / 0 | 0 / 0 |

Fixed-all also outputs 11 unannotated documents across nine of the 15 positives
(nine across seven common-error claims). These are **unannotated**, not known
semantically false. In `case_08` and `case_09`, the correct gold verdict is retained
but an extra unannotated positive document makes the strict whole answer fail.
Do not relabel that as loss of the correct verdict. Among common errors, fixed-all
has two locally strict-correct verdicts, but no complete correct whole answer.

## Three interview/debugging cases

1. **`case_05`: over-selection, not missing recall.** All three arms visit the
   annotated document, choose the annotated label and include a complete rationale
   within their first three sentences. Each also selects an unannotated sentence,
   failing the unchanged strict contract. The same pattern holds in `case_06`,
   `case_10` and `case_21`. Better retrieval alone cannot repair this recorded output;
   semantic wrongness of the extra sentence has not been independently adjudicated.
2. **`case_23`: visible evidence but INSUFFICIENT.** All arms visit the annotated
   document yet return INSUFFICIENT and omit the rationale. The adaptive route
   continues but ultimately abstains on a positive claim. This supports checking
   relation judgment/abstention calibration, not silently expanding the context.
3. **`case_24`: two coupled failures.** Three gold documents are retrieved and
   visible. Top1/adaptive verify one and leave two unattempted. For the visited
   document, the label is correct and the full sentence selection contains a
   rationale, but its first three sentences do not, with extra unannotated sentences.
   Fixed-all visits all three and has one strict-correct document verdict, yet still
   fails the whole answer. Both evidence selection and multi-document coverage matter.

## Next implementation decision, not an automatic new experiment

Prioritize a verifier that separates relation judgment from **minimal sufficient
rationale selection**, under the unchanged scorer and isolated semantic inputs.
Use the already consumed examples for implementation diagnostics only. Keep
label/INSUFFICIENT calibration, first-three selection and extra-sentence penalties
separate; never trim output using gold at inference or weaken the strict scorer.
Only then evaluate a controller capable of completing genuinely multi-document
evidence. No observed correct verdict was dropped here, so a new commit-assembly
repair is not the first priority. Do not add retrieval loops based on these cases.

Any real model comparison needs its own exact-source authorization and an honest
evaluation scope. This package does not authorize new GPU calls, expose a new
test set or establish an accuracy/Agent benefit. Original official F1 and the
custom strict whole-answer measure remain separate and unchanged.

## Reproduction and provenance

Contract-only local reproduction at the execution revision:

```bash
python -m pytest -q tests/test_scifact_semantic_input.py tests/test_scifact_evidence_commit_entry.py
bash -n hpc/scifact_semantic_input_cpu.sbatch
python scripts/diagnose_scifact_semantic_input.py --help
```

The real diagnostic requires an authorized CPU allocation, the pinned prior
Spartan artifacts/tokenizer and the exact source archive; it refuses ordinary
login-node execution. Run the checked wrapper with `CLIMATE_SOURCE_STAGE`,
`CLIMATE_SOURCE_GIT`, `CLIMATE_SOURCE_SHA`, and `CLIMATE_CPU_WRAPPER_SHA` from the
identities below. It verifies the archive, extracted tree and actual wrapper
before invoking `--output <stage>/semantic-input-diagnosis.json`. Do not rerun
to seek a different answer or publish the underlying raw records.

| Identity | SHA-256 / revision |
|---|---|
| Execution source Git | `58975a914b3bb0f224db366b519fe267feb0fbbd` |
| Exact source archive | `bbedbf21c8ed4335cc084a911e85efc075c837c6826efe54206ca0666dec0b3f` |
| Actual CPU wrapper | `cca5c3a8f9d560e71a39fe39437f5caaf972c6920259dd8a7db30c4bc475d078` |
| Diagnostic script | `c07f80f7eb074c69d21bb8b26981f8287cbe8a9999b5bdbb0a61dfa34a67f7a6` |
| [Unmodified redacted compact](verified-runs/scifact-semantic-input-cpu-31963271.json) | `5864c9912ad8ada0e868392aa18c883409d3ea43d6910988339f747fe1c38cce` |
| Original quality artifact | `247fcdd764d5de4d39469786c1ef8b770e87a29fcf03033ab90fe47ed8d29eae` |
| Original selected roster | `351523c9c44bb418e24ed17b40372e6bef7b87792f8ad0b78dcaeb2d22e22754` |
| Original shared frames | `e20a5188ba51bbaf940009338ee167d6d07bf4429652eaa06967265f37a4e29c` |

The compact also records every tokenizer-file hash. Only pseudonymous positions,
counts and hashes are exported; raw claims, scientific sentences, gold, predictions
and physical responses remain on Spartan. Later report commits are not the CPU
execution source. No shared career material or other project was edited.
