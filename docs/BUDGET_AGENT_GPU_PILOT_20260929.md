# Single authored GPU pilot — operator record

Release: `climate-authored-pilot-ca9fa53-20260929-user-resume`. The coordinator
relayed the user's explicit bounded exception at18% remaining allowance after
the other project packages ended. This permits **one pilot**, not automatic
validation/vNext, retraining, new models, deployment or a retry. No credits/reset
were purchased. Shared career files remain coordinator-owned.

## Immutable execution

The executable source remains `ca9fa53f2d4b70091fd08a8a1512e263f329fc61`;
all four archives/model revisions/protocols remain those in
[CPU readiness](AGENT_ASSET_PREPARATION_20260929.md). CPU job31486505 is retained,
not rerun. First stage is the5,240-document public BM25 corpus, no dense/ANN.
Pilot is3 authored tasks ×3 frozen routes, at most15 generation attempts and
7,680 new tokens (reranker forward work is separate). No official test or gold
is read, and these authored examples cannot establish generalization quality.

The external [operator wrapper](../hpc/budget_agent_pilot_observed.sbatch) invokes
the exact frozen `hpc/budget_agent_pilot.sbatch` unchanged. Resources remain
one full A100,8CPU,32GiB host RAM,30GiB scratch,15min ceiling. Slurm owns CUDA
visibility; the submission script clears inherited CPU-only settings before
submission. This proposal has not been measured by a GPU pilot yet.

Operator wrapper SHA-256:
`569d824e10645c0fa0ce4604e33810a95085969bdf730adc182e5483227780a8`.
It adds only external1s NVML sampling, job start/end/exit records, pre-timeout
and termination snapshots of this job's small result/consumed receipts. It does
not modify model/data/code, arguments, RNG, tokenizer or inference behavior.
The original successful result archive and Slurm log stay distinct from the
operator diagnostics; missing results are not treated as successful execution.

## Observability and limits

- Slurm provides terminal state, Elapsed/TotalCPU/MaxRSS and allocation TRES.
- Successful run JSON provides generation-provider attempts, known input/output
  token counts, per-route/stage elapsed time, retrieval/rerank calls and candidate
  pair attempts. Failed attempts are not silently counted as successful forwards.
-1s NVML observations provide **sampled observed device-memory maximum**, not
  a precise CUDA allocation peak. Scratch high-watermark is not captured.
- The frozen runtime does not isolate generator/reranker load latency, TTFT or
  successful neural forward counts. Those remain unavailable; elapsed time
  outside measured query loops includes extraction/hash/import/loading/setup,
  not a pure model-load measurement. No instrumentation hooks were injected.
- If timeout/failure occurs, retain the log/status/available receipts, report the
  failure and stop. Do not auto-submit a replacement or final evaluation.

## Pre-submission checks

Read-only account queue was empty. The project mount had~276GiB available and
600 free inodes; the home quota was full. Use existing archives and node scratch,
not a new persistent model/environment tree or home cache. Refresh guards in
the operator submission script immediately before `sbatch --test-only` and the
single submission: no queued jobs, at least30 free inodes, exact source/operator
hashes, complete input size and unused result/diagnostic paths.

New operator-only checks: bash syntax,3 static contract tests, targeted Ruff.
The frozen executable's previously passing full CI remains the test evidence;
the unchanged CPU suite is not rerun. Static operator checks do not prove Slurm
signal delivery or successful GPU runtime behavior.

## Result

Not submitted at this commit. Append the actual job ID, terminal evidence and
resource-based follow-on estimate here after the single pilot; do not infer
completion from the existence of an allocation or an empty result directory.
