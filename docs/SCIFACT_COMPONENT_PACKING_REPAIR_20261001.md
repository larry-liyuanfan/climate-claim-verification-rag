# Component packing failure: CPU-only recovery

## Failure and cost

The historical submission receipt is superseded by terminal job **31729507**:
FAILED `1:0`, 2026-10-01 01:13:50–01:15:11 cluster local time, 81 seconds.
One A100 was allocated; batch TotalCPU was **68.866 seconds**, MaxRSS
**9,675,684 KiB**. Allocated GPU time is not active GPU kernel time or zero cost.

The exact source `79f069d2eb4c8db3b82b152fa8a698b5ae3230ad` stops at
`run_scifact_components.py:37`, `frozen_input_packing`. This precedes worker
identity persistence, provider-private directory creation, provider construction
and generation. The preserved v1 directory contains only inference/operator
logs and two package-install logs; no identity, provider directory, inference
slot or started reservation exists. Operator status says `targets_read=false`.
Together, source control flow, traceback and physical outputs establish **zero
model construction/generation calls**, not merely missing successful results.
Earlier manifest validation did read model files for hashing; do not describe
the original job as having never read weights. No semantic metric is available.

## Root cause, not a missing dependency

Stage A constructed object keys as `component, claim, documents`, appending
`oracle_relation` for rationale. Documents used `document_id, title, sentences`.
Packing was computed then. `write_once → encoded(sort_keys=True)` subsequently
saved the objects in alphabetical key order. Reloading preserved that new order,
but the old renderer serialized the object unchanged. Canonical input identity
was the same while actual prompt bytes differed.

The [single bounded real-tokenizer CPU check](verified-runs/scifact-component-packing-repair-20261001.json)
used only the existing 33 frozen slots and existing tokenizer runtime:

| Check | Observed |
| --- | --- |
| Legacy prompt hash mismatches | 33/33 |
| Legacy token-count mismatches | 20/33, increases of 1–4 tokens |
| Legacy schema/input hashes and status | unchanged for all 33 |
| Stage A four assets vs runtime eight assets | identical packing on both paths |
| Restored full packing vs original expected | 33/33 exact on both paths |
| Real-tokenizer synthetic persistence roundtrip | 4/4 exact |

Both chat-template hashes are identical. Full-runtime tokenizer assets include
added/special tokens and config files, authenticated against the frozen model
manifest. The probe reads only eight small allowlisted files, not weight payloads
or the entire multi-GB archive. It reads no scoring targets, chooses no new data
and makes no model calls. Execution took **4.618 seconds**; this is a diagnostic,
not inference performance. `USE_TORCH=0 USE_TF=0` intentionally disables model
backends; the tokenizer-only warning does not mean Torch needs reinstalling.
Local Torch 2.7.1+cpu and the existing Spartan Linux/POSIX runtime remain usable.

## Minimal repair and unchanged scientific contract

`authored_input()` validates first, then reconstructs the original Stage A object
field order at all three levels. It never sorts arrays, drops fields, changes
values, truncates context or mutates inputs. The original prompts, schemas,
tokenizer assets, model identity, slots/protocol/targets, decoder and budgets
are unchanged. Strict expected-packing checks remain in place. The contract
**implementation** file changes; observed frozen prompt bytes do not.

The tests cover all three component types, recursive key permutations, actual
save/load, unknown-field rejection and late-slot mismatch before provider
construction. All 33 slots must pass before the provider can be created.

Both local and exact-source clean extraction pass **151 tests, 1 Windows POSIX
skip** (clean run 5.25 seconds). Ruff, strict mypy on six changed source files,
tracked secret/PII scan and shell syntax pass. This reuses the existing environment,
not a fresh dependency installation. The existing Linux owner/symlink/fsync
microcheck remains separate evidence, not a Windows pass. One pre-existing
Python 3.14 tar-extraction deprecation warning remains.

## Separate execution identity, not automatic retry

Prepared source: `81917f2c9a1b2060bd84a0688bd65a40b1c4e134`.
Its [validation receipt](verified-runs/scifact-component-repair-validation-20261001.json)
binds the exact archive/wrapper and CPU results. The packager checks all 415 Git
blobs/modes, 13 directories, one expanded revision marker and actual shell guard.
An initial ordinary Windows `git archive` was rejected for byte/mode mismatch;
only the packager's verified LF archive was transferred or used.

Prepared physical release: `scifact-component-single-attempt-20261001-r2`.
Metadata binds logical v1, `infra_retry=1`, original job/source/archive/log/status
hashes, zero prior calls and nonzero allocation cost. The operator requires those
exact old failure files and absence of model-stage markers before reserving r2.
v1 remains untouched. This is one specific infrastructure recovery, not a retry
framework or a renewed 37-call allowance after paid inference.

**No new GPU job, `sbatch --test-only`, or model call was executed by this CPU
package. Stop for separate exact-hash release.** Scientific scope remains the same
already-consumed twelve TRAIN claims / 33 component slots, with 4 distinct
synthetic preflights and at most 37 calls, 8192/512 tokens and 120 seconds per call.
No fresh dev/test, training, resume modification or Agent benefit is claimed.

Reproduce within the existing supported tokenizer-only CPU environment:

```text
USE_TORCH=0 USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:scripts python scripts/probe_component_packing.py --slots <frozen-slots> --stage-tokenizer <four-file-stage> --runtime-archive <existing-uncompressed-generator-archive> --work <new-private-directory> --output <new-compact-file>
```

Paths are private operator inputs. Do not run this again on newly selected data
or treat the reproduction command as permission for another model evaluation.
