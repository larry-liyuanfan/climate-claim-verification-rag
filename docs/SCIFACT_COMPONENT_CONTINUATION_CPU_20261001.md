# Technical preflight / semantic measurement continuation

Status: CPU implementation; **no new GPU release or model invocation**. This is
an explicit post-observation protocol revision, not an infrastructure retry,
generic resume, completion of the old protocol, or independent-test evidence.

## Why a distinct protocol

The immutable r2 run stopped after a technically valid but semantically wrong
synthetic abstention. Its negative exact-match result and all unattempted slots
remain intact. [The terminal record](SCIFACT_COMPONENT_CLOSEOUT_31734906.md)
retains the evidence and v1/r2 resource costs. The new protocol separates
technical readiness from semantic measurement; wrong answers are still wrong,
but valid answers do not prevent measurement of subsequent failure modes.

- Frozen: four synthetic payloads **and expected answers**, 33 diagnostic inputs
  (12 screening / 12 relation / 9 oracle rationale), model, prompt, schema,
  decoder, 8,192 input / 512 output tokens and 120 seconds per invocation.
- Technical preflight requires validated output and complete verified usage.
  Provider/schema/identity/persistence/budget/unknown-cost errors stop. A valid
  semantic mismatch is recorded as false, not repaired or promoted to success.
- Formal diagnostic schema failures remain paid failures and do not stop other
  independent slots. Provider failures stop; missing calls retain all planned
  denominators. No retries, replacement claims or hidden tool invocations.

## Carry exactly one old invocation

Only r2 job 31734906 preflight 0 can be referenced. The loader binds source Git,
source archive, worker identity, terminal proof, run, wire, response, reservation
and completion hashes, then independently reconciles private physical receipts,
strict parsing, decoder, packing and cost. Extra predecessor attempt directories,
hidden reservations, symlinks and any mismatched hash fail closed. The worker
performs this and all frozen packing checks **before provider construction**.

The new fixed release is
`scifact-component-technical-continuation-20261001-v1`. It writes only a carry
reference; it must not create `inference/preflight-00` or modify old files. The
remaining synthetic indices are 1, 2, 3. An independent scorer reopens the old
physical records and reconciles new records only after the child process exits.
The old scorer's default exact-match policy is unchanged.

Limits: **one carried + at most 36 new = at most 37 cumulative invocations**.
Reports split carried/new/cumulative calls, tokens, unknown costs and slot
latency. The old 277 input / 15 output tokens remain paid and semantically
negative. Historical **81 + 96 = 177 allocated GPU-job seconds** and 154.310
batch CPU seconds are retained with source/job identifiers; these are not active
GPU kernel time. A carried response does not prove that the new provider can
generate; the remaining live preflights must still exercise that path.

## Reproducible CPU validation

Use the existing validation environment; no installation is required. On Windows
set `PYTHONPATH=src;scripts`; on POSIX use `src:scripts`.

```text
python -m pytest tests/test_component_continuation.py tests/test_component_execution.py tests/test_scifact_components.py -q
python -m ruff check src/climate_rag/component_continuation.py src/climate_rag/component_audit.py scripts/run_scifact_component_continuation.py scripts/run_scifact_component_continuation_operator.py scripts/score_scifact_component_continuation.py scripts/probe_component_continuation.py tests/test_component_continuation.py
python -m mypy --follow-imports=silent src/climate_rag/component_continuation.py src/climate_rag/component_audit.py scripts/run_scifact_component_continuation.py scripts/run_scifact_component_continuation_operator.py scripts/score_scifact_component_continuation.py scripts/probe_component_continuation.py
bash -n hpc/scifact_component_continuation.sbatch
```

New tests build hash-bound **synthetic physical predecessor files**, not real
training/test evidence. They exercise valid-wrong continuation, technical stop,
paid formal-schema failures, all predecessor-file hashes, hidden calls, private
raw corruption, duplicate carry, persistence failure, independent scorer
agreement and tampering, ceilings and pre-constructor rejection.

The separate `probe_component_continuation.py` uses the real tokenizer and old
r2 physical artifacts read-only, without reading scoring targets or weights.
It also checks Linux owner permissions, symlink refusal, directory fsync and
actual child-process normal/interrupted reaping using tiny synthetic processes.

```text
USE_TORCH=0 USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:scripts python scripts/probe_component_continuation.py --tokenizer <existing-runtime-tokenizer> --work <new-project-cpu-probe-dir> --output <new-compact-report>
```

Exact Git archive/blob/mode verification, LF wrapper guard, clean extracted
source reproduction and the real-artifact probe must be attached to the release
receipt before separate coordinator review. The wrapper cannot be submitted on
the strength of this document alone. No new semantic result or resume claim is
created by CPU validation.
