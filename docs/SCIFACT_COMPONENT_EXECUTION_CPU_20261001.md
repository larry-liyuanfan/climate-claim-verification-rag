# Component provider/operator: CPU preparation, not model results

## What changed

This package connects the frozen Stage A inputs to a versioned component
provider, a single-attempt worker, a separate post-exit scorer and an opt-in
Spartan wrapper. It does **not** run a model, install a new dependency, submit
Slurm, read new claims/dev/test, train an adapter or update a resume.

Preparation source remains `426ff7343fb30e1ffcde4dfa4a43f1c00c210cfb`.
The execution source is a separate commit recorded in the validation receipt.
Frozen identities (raw inputs/targets stay on Spartan):

| Item | SHA-256 |
|---|---|
| Preparation protocol | `2ce563ccc34efbd5ee1a21cf12fa47fafd063c853c629996909cb37e7247ca1e` |
| 33 inference slots | `84524e2a837aacab5824e5562a61d827ef02b1832dd9ab9f383ae73caa309ac9` |
| Private scoring targets | `a6d38d3fe9d72e6beb0b16836e650e4519d2016e31b63ed1d79e05999b665d9a` |
| Qwen3-4B model manifest | `d1dd9783afdf4e0fbd21eee824834d71b86982f5a5d5f6f371fe07f2f76f3cf6` |

The twelve already-consumed TRAIN claims yield 12 screening, 12 relation and
9 oracle-conditioned rationale slots. Whole-input screening coverage remains
**6/9** annotated gold documents across 60 candidate occurrences. This frozen
coverage is independent of any future valid-output subset. No candidate
replacement, truncation, extra data, label-to-relation leakage or new selection.
Rationale inputs explicitly include the oracle relation; `scoring_targets_loaded=false`
means scoring alternatives/targets are withheld, not that every input is gold-free.

## Decoder compatibility and its limit

LMFE 0.11.3 routes numeric enum through unquoted string parsing. Its list-state
interaction rejects legal `[2,10,1]`, whitespace `[1 , 10]` and an eight-index
rationale, even without `anyOf`. `component-lmfe0113-number-state-v1` removes
integer `enum` **only from the decoder copy**, retaining the original prompt,
schema, counts and hashes. Both canonical/decoder hashes are traced.

This **relaxes decoder constraints**; it is not language equivalence. Unknown
IDs may be generated. LMFE also does not reliably enforce uniqueness or an
empty abstention array. The unchanged canonical `parse()` rejects these raw
outputs as paid schema failures, without sorting, deduplication, repair or retry.
Relation enums/const, min/max cardinality and greedy/nonthinking remain unchanged.

The frozen tokenizer probe covers 13 synthetic token paths, including different
orders, whitespace, eight indices, reversed object keys, three relations,
abstention and A/B/A switching. Complete EOS is reachable; incomplete prefixes
cannot end early; no grammar-error log or top-level ForceStop is accepted.
This establishes **tested synthetic token-path reachability**, not HF generation
or real-model accuracy. LMFE optionally imports the already installed CPU Torch;
no weights, tensor inference or model loader is used by this probe.

## Execution and audit contract

- Unique fixed release root: `runs/scifact-component-single-attempt-20261001-v1`.
  Do not delete/rename it to replay. Four **different** synthetic preflight
  inputs precede 33 planned diagnostics: maximum 37 actual attempts, no warmup.
- Each fixed phase/slot path has exclusive creation, full actual prompt/input,
  canonical and decoder schema, hashes and an fsynced started reservation
  **before** generation. Budgets: 8192 input / 512 output / 120 seconds each.
- A schema-failed diagnostic is paid and may be followed by another independent
  slot. Failed preflight, provider/deadline/unknown-cost or persistence failure
  stops; remaining slots are `not_attempted_after_stop`, not failed model output
  or missing preparation. All 33 planned slots and NEI strata remain visible.
- Raw responses and grammar logs are owner-only, bounded private artifacts.
  Completion-write failure never permits retry. The audit can recover physically
  verified response/exception cost from a missing/partial completion record,
  while retaining `provider_failed` and withholding quality credit.
- The operator ignores repeated managed termination signals while killing and
  waiting for its own child group. Only an actual exit proof opens scoring
  targets to a **new** process. Same-user dataflow separation is not an OS sandbox.
- The scorer checks original wire/schema/input/response hashes, physical raw
  receipts, token metadata and canonical parsing, not only cached predictions.
  Compact export includes all statuses/costs, screening pool gaps/omissions/
  unannotated extras, relation correctness, alternative/first-three/exact rationale
  coverage, excess sentences/redundancy and paired error localization.
- Empty selection with missing candidates is not full retrieval success;
  unannotated extras are not proven factually wrong. An interrupted matrix
  supports no semantic-effect conclusion. Oracle success does not prove Agent
  benefit or authorize action-SFT.

## Reproduce software validation

[Frozen validation receipt](verified-runs/scifact-component-execution-validation-20261001.json):
execution source `79f069d`, exact archive/blob/mode and component shell-guard
checks passed. Both targeted and clean-source suites: **144 passed, 1 Windows
POSIX skip**; the Linux microcheck is reported separately. Ruff, strict type
checks (10 files) and tracked secret/PII scan passed. The
[tokenizer receipt](verified-runs/scifact-component-tokenizer-20261001.json)
records all 13 synthetic paths. These are software checks, not model results.

Use the existing project validation environment, `PYTHONPATH=src;scripts` on
Windows (`src:scripts` on POSIX). No dependency installation is needed here.

```text
python -m pytest tests/test_component_execution.py tests/test_local_scifact_provider.py tests/test_scifact_components.py tests/test_bounded_scifact_runtime.py tests/test_private_diagnostics_v3.py tests/test_scifact_terminal.py -q
python -m mypy --follow-imports=silent src/climate_rag/component_decoder.py src/climate_rag/component_preflight.py src/climate_rag/component_execution.py src/climate_rag/component_audit.py src/climate_rag/local_component_provider.py src/climate_rag/scifact_component_scoring.py scripts/probe_component_tokenizer.py scripts/run_scifact_components.py scripts/score_scifact_components.py scripts/run_scifact_component_operator.py
bash -n hpc/scifact_components.sbatch
```

Tokenizer-only probe (set `USE_TORCH=0 USE_TF=0 HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1` for transformers):

```text
python scripts/probe_component_tokenizer.py --tokenizer-dir data/qwen3-4b-tokenizer-v3 --output artifacts/component-probe-unique.json
```

Local Windows owner/symlink test is reported as skipped, not passed. A separate
tiny check using the existing Spartan Stage A source and Python 3.10.4 verified
owner-only acceptance, public-mode rejection, symlink rejection and directory
fsync. Default login `python3` is too old for the package's dataclass slots;
loading the existing module fixes interpreter selection, without installation.
POSIX/fork is provided by Linux, not pip. Local Torch 2.7.1+cpu is already usable.

## Future execution: exact-hash review is still required

The wrapper proposes **one A100, 8 CPU, 32 GiB, 30 GiB scratch, 100 minutes**,
serial/no-requeue. 37 * 120 seconds + 1560 seconds staging/loading margin; a
ceiling, not an ETA. Only generator assets are extracted, not the reranker.
Existing pinned offline runtime/overlay/grammar wheels are reused in private
scratch; future code may install those existing wheels, but this CPU package
has not performed that installation.

After a separate coordinator release of the execution Git/archive/wrapper hashes,
the operator requires `CLIMATE_SOURCE_TAR`, `CLIMATE_SOURCE_SHA256`,
`CLIMATE_SOURCE_GIT` and
`CLIMATE_COMPONENT_RELEASE=scifact-component-single-attempt-20261001-v1`.
The exact wrapper is `hpc/scifact_components.sbatch`; first validate with
`sbatch --test-only`, then at most one submission. Neither command was executed
by this CPU package. The shell checks one 41-byte expanded `SOURCE_REVISION`,
the whole source archive hash and the actual wrapper bytes. The Python worker
and scorer enforce the frozen input/target hashes above.

Stop here before model execution. No new generalization, classification,
retrieval improvement, deployment, online SLA or resume result is claimed.
