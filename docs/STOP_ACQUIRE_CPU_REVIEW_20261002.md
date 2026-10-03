# Stop/acquire: whole-chain review before submission

## Decision and scope

This is an **opt-in CPU implementation and contract-validation package**, not a
model-quality result. No weights were loaded; no inference, training, Slurm job,
new gold/test/dev300 access, merge, deployment or resume change was performed.
The prior [32030221 negative result](TARGETED_RETRIEVAL_FEEDBACK_RESULT_20261002.md)
remains intact. An unauthorized exact-source draft is a handoff artifact, never
permission to run. Resource availability or queue order is not promised.

The new hypothesis is falsifiable: separating acquisition from final judgment
may activate useful evidence acquisition without forcing queries. More calls or
new source IDs alone do not count as improvement. Stop remains legal; previously
shown original sentences remain visible. No gold-derived opportunity signal,
real-claim example or hidden-context intervention is added.

## Consolidated review, fixes and release gate

Review the **whole path before submitting**, not only the last modified module:
package → exact release → worker/provider → controller → tool feedback → physical
cost ledger → post-exit audit/scorer → compact result. Findings were consolidated
and repaired locally before code publication; no queued job was used as a debugger.

| Finding caught before submission | Repair / regression |
|---|---|
| New gate code existed but replay/worker/package still selected the old protocol | Explicit opt-in protocol travels from release through provider and route mapping; production worker+supervisor+scorer functions are exercised with 160 synthetic slots |
| Old scorer could not interpret stop/acquire | New adaptive-only state replay plus existing physical-wire, source, budget and cost audit; release/run protocol must match before gold access |
| Schema accepted three reads with `context_k=2`, executor rejected after opening a running event | Same read limit in schema/parser/execution; reject before recording a tool execution; regression rejects both grammar input and manual wire |
| A tool could finish and then cross the deadline before execution status was recorded | Final execution status survives the error; deadline and source-mutation failures are charged and auditable, not missing slots |
| An execution failure could be relabeled as intentional abstention | Successful/abstaining outcomes require the actual final valid verifier decision; outcome-only tampering fails |
| Prior 31.977 GiB peak nearly filled a 32 GiB allocation | New wrapper/draft propose 48 GiB; legacy wrapper stays 32 GiB for reproducibility |

Release checks are **one bounded batch**: affected regressions, end-to-end
synthetic failure cases, real tokenizer and CPU Torch/provider binding, scoped
strict typing, Ruff, secret/PII scan, clean-source affected reproduction, exact
archive/wrapper/JSON checks. Unchanged historical suites/supervisors are reused;
only affected behavior is rerun locally. A later authorized Slurm submission
still needs its exact release approval and `sbatch --test-only`; neither happened
in this CPU package. CPU checks cannot establish CUDA numerical behavior, cluster
availability, real-model action choices or answer quality.

## Mechanism and preserved comparison

```
initial BM25 retrieval
  → gate: stop OR acquire(unseen read / targeted query)
  → actual tool, including empty/recoverable error feedback
  → next gate
  → stop → separate answer/abstain verifier
```

- Protocol: `sentence-stop-acquire-v1-20261002`, enabled only for `adaptive`.
- Four controls still use `sentence-targeted-feedback-v1-20261002`; historical
  adaptive is still the default. No sixth route or changed baseline prompt.
- Same proposed original **32 consumed validation claims / 5,240 documents**;
  retrieval denominator **24**, binary label **23**, cost/failure **32 per route**.
  This is repeated development evidence, not a fresh independent test or causal
  isolation of feedback. Frozen task/data/model hashes remain in `policy()`.
- Combined caps: **5 generations, 5 tools, 120 s per slot**, 8,192 input and 512
  output tokens **per physical call**. Gate, verdict and repairs share these
  limits. Acquisition requires capacity for the next gate and final verifier.
  Budget exhaustion is a charged execution failure, not fabricated abstention.
- Read masks derive from actually displayed complete sentences. Query text never
  replaces the immutable claim. Failed queries are counted and cannot be silently
  retried for free. New/empty/error results drive the next actual observation.
- Existing local model loader, LMFE decoder, generation binding, journal,
  retained-context packing and process supervisor are reused. No new general
  audit framework or trained routing model is introduced.

Conceptual references: [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/)
and [Sufficient Context](https://arxiv.org/abs/2411.06037). This implementation is
not a reproduction of their learned selectors or reported gains. Evidence
availability is not proof of correct reasoning or semantic entailment.

## Cost feasibility BEFORE any real run

Even immediate stop needs two generation calls. The same evidence is passed to
both phases; its prefill tokens cannot be excluded or called free cached work.
The [real-tokenizer synthetic check](verified-runs/stop-acquire-tokenizer-cpu-20261002.json)
uses 20 synthetic sources, three document lengths and scripted valid outputs,
including EOS. These are token-contract counts, **not model predictions, latency
measurements or a paired comparison on the real 32 claims**.

| Synthetic source length (repeated words/doc) | Stop: 2 calls | Read then stop: 3 calls | Query then stop: 3 calls |
|---|---:|---:|---:|
| 8 | 2,919 tokens | 4,207 | 4,369 |
| 64 | 3,549 tokens | 5,258 | 5,310 |
| 180 | 4,709 tokens | 7,230 | 7,050 |

Historical consumed32 mean generator-token baselines were **1,513.78** for fixed
rerank and **2,086.16** for fixed multiquery. Different synthetic text means these
numbers cannot estimate a paired real-task delta; nevertheless, repeated gate
prefill is a clear cost risk. This candidate is at most a **quality-only
hypothesis** pending real validation, not a claimed efficiency improvement.
The original conjunctive quality/cost gates remain unchanged; failure against
either control remains failure. Repairs, invalid outputs, all 32 slots, model
swaps and reranker tokens/pairs remain in the accounting.

Resource proposal uses 32030221: MaxRSS 33,530,392 KiB ≈31.977 GiB; 48 GiB gives
approximately 1.5× headroom. Historical 276.610 s/194 generator calls and
791.349 s/96 reranker requests (including swaps), conservatively scaled to
800/128 calls at 1.5×, give about 3,294 s. Retain the bounded 6,000 s worker and
2 h allocation proposal; this is a cap/basis, not a prediction for longer new
prompts or an online SLA. Stop/acquire itself never calls a reranker, but the four
fixed controls still do; the original conservative 128-request ceiling remains.

## Validation and reproducible entry points

Final local acceptance: **133 affected tests passed** from a clean staged-source
archive with imports verified to resolve into that archive, not the editable
checkout. The nine real-tokenizer cases reproduced identically from clean source;
compact token-contract SHA-256 is
`223ecd2cebe061c36beb2ae05ac2dec13ae61fa62b25204e5a10d00d863df87a`.
Strict scoped mypy passed for ten changed Python modules/scripts; repository-wide
Ruff, tracked secret/PII scan, diff whitespace and Bash syntax checks passed.
Three test warnings concern inactive sampling defaults in the inherited greedy
generation configuration; effective `do_sample=false` remains contract-checked.
Native Ubuntu WSL `setsid`/`killpg(SIGTERM)`/`wait` smoke reaped its synthetic child
with exit −15. This OS smoke is not a Linux full-project or Spartan GPU test.

The affected batch covers stop, unseen read, query→real feedback→next decision,
empty/timeout, malformed/duplicate read, invalid citations, context overflow,
combined-call exhaustion, physical costs, forged outcomes, old controls and a
160-slot synthetic end-to-end worker/score path. No synthetic accuracy is promoted.
CPU Torch checks use prescribed fake-model output through actual tensors,
provider and generation binding; model-file bytes are mocked in that test.
Real Qwen tokenizer/LMFE checks separately validate all four tokenizer-file hashes,
legal token paths and EOS, rejecting already-seen read and unknown citation paths.

```powershell
python -m pytest -q tests/test_stop_acquire.py tests/test_targeted_retrieval.py tests/test_targeted_replay.py tests/test_agent_v3.py tests/test_scifact_terminal.py tests/test_targeted_closeout.py
python scripts/smoke_stop_acquire_tokenizer.py --tokenizer data/qwen3-4b-tokenizer-v3 --output artifacts/stop-acquire-tokenizer-cpu.json
python -m ruff check src scripts tests
python scripts/scan_tracked_secrets.py --repository .
python scripts/package_targeted_replay.py --stop-acquire --source-git <exact-40-hex-commit> --output <new-empty-output-directory> --bash <bash-executable>
```

The final command freezes source and creates `release.unauthorized.json` with
`authorization=none_draft` and `model_execution_authorized=false`; it does not
submit a job. Run/archive/wrapper/release SHA values belong in the generated
package and coordinator handoff, not guessed or retroactively substituted here.

Windows validation uses Python 3.12.14 / Torch 2.7.1+cpu / Transformers 4.51.3.
Torch is installed; missing POSIX APIs on Windows are not a pip dependency.
Ubuntu WSL provides native Linux/POSIX; this does not make Windows validation a
proof of the different Spartan Torch 2.1.2/CUDA runtime. Reuse the accepted
unchanged supervisor evidence and Linux CI; do not reinstall or modify Trip's
environment. Changes here are limited to this isolated Climate branch.
