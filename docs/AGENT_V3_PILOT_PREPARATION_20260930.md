# Sentence-ID v3 development pilot — preparation, not execution

CPU core commit: `35f192047bcead5d23bd7d4ae2e5d3d83608e51b`, CI
[36666084118](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36666084118)
passed383 tests with1 optional Torch skip. Clean git-archive targeted reproduction
passed31 with1 Windows POSIX skip; Linux CI exercised the POSIX test. Independent
read-only review closed cumulative quota, stderr fallback and known-usage P2s.

This package prepares **one** separately authorized development allocation.
No model run, GPU submission, SciFact dev evaluation or improvement is evidenced
by these files. Old v1/v2 records, original frozen tests and all other projects
remain unchanged. New operator and run directories are versioned independently.

## Frozen work design

- Same6 previously exposed authored claims as v2, same5,240 evidence-only corpus,
  same offline Qwen3-4B and Qwen3-4B reranker files. BM25 recall only; no dense
  invocation, LoRA training, model-size choice, sampling search or seed selection.
-4 routes/24 slots in fixed task-then-route order. Primary comparison is adaptive
  versus fixed rerank; deterministic extra retrieval/RRF/rerank is the stronger
  extra-work control. All caps follow the v3 CPU document; actual cost differs.
- Each existing evidence row is one intact passage mapped to original index0.
  It is not a new language-dependent sentence splitter or SciFact sentence gold.
- Private preflight first performs4 synthetic generation attempts: forced read
  A→B→A for fresh grammar state and EOS, then exactly1 output token to test
  incomplete-JSON/truncation accounting. This artificially constrained sequence
  proves runtime behavior only, **not model-selected tool use**. Its actual cost
  is stored separately from24 task slots. Failure stops before the pilot.
- Pilot records all failures, unknown output usage, known token subtotal,
  model/tool calls, rerank pairs, elapsed time, context and preview payload costs.
  Partial slot files are evidence, not a resumable or silently complete matrix.
- Compact output requires every unique task/route pair. Raw outputs, quotes,
  task traces, logs and original manifests stay in the private Spartan run.
  Mechanical answers and exact provenance are not semantic correctness.

## Entry points and isolation

`configs/agent_sentence_v3_pilot_20260930.json` freezes the protocol.
`scripts/run_sentence_agent_v3.py` is the no-gold allocated runtime.
`scripts/run_sentence_agent_v3_operator.py` extracts the verified existing input,
runtime and overlay archives once into node-local scratch. No legacy operator is
executed. Only its reviewed stdlib hashing/extraction/process helpers and
environment builder are imported. Grammar wheels install with
`--no-index --no-deps --require-hashes`; the two new wheel digests are pinned in
`hpc/agent_v3_grammar.lock`. Root raw quota is shared across preflight and all
pilot calls, not reset for each route or task.

`hpc/agent_sentence_v3_pilot.sbatch` requires exact source archive, source commit,
protocol and grammar bundle SHA values plus the separately released ID
`climate-sentence-v3-20260930-pilot-r1`. It verifies its own bytes against the
source archive and exports only the declared runtime identities. The result
directory and runner receipt use exclusive creation; there is no automatic
retry/resume or scheduler submission inside these entry points.

Proposed resources: one A100,8 CPU,32GiB RAM,30GiB scratch,60min, no requeue.
V2 elapsed317s for18 slots; v3 has24 slots, stronger extra work and4 runtime
checks.60min is a conservative upper envelope for24×120s plus extraction/model
load/preflight (not a latency estimate). Before any real submission, check exact
release duplicate jobs and actual quota, run `bash -n` and `sbatch --test-only`,
and obtain coordinator release for the reviewed SHA. Other tasks are untouched.

## Local preparation receipts

Grammar bundle (ignored local binary): `data/agent-v3-grammar-20260930.tar`.
SHA-256: `36ed51e0a9a71ded058cc0e7290a78a7f5f638b91bff517d49ad6ea83c70e410`.
Contains only the two pinned wheels under `wheels/`, downloaded by
`pip download --no-deps --only-binary=:all:` from the configured PyPI index.
The bundle is transferred to Climate's own `envs/` only; binaries are not added
to Git. Newly generated source archive and exact protocol SHA will be recorded
after the preparation commit, before scheduling review.

The child environment removes **all** `LMFE_*` settings; the provider additionally
constructs explicit LMFE0.11.3 parser settings (alphabet from the pinned library,
12 consecutive whitespaces, unrestricted field order, default max array20).
Every response/preflight records the effective settings and alphabet hash.
CPU fixtures poison known and future environment keys to prove isolation.
Preflight preserves generated usage/diagnostics even if JSON/action parsing fails.

`hpc/submit_climate_sentence_v3.sh` defaults to `--test-only`. A real `--submit`
additionally requires the coordinator-released ID plus exact source SHA marker,
an absent result directory and absent submission lock, matching all file hashes,
and no duplicate Climate release jobs. It clears inherited `SBATCH_*`, applies
identical explicit resources to test/real submission, then atomically reserves
the release with `mkdir`, records hash inputs, repeats test-only and submits once.
The lock stays even after failure; no automatic cancellation, unlock or retry.
Preparation/test-only prints a scheduler estimate, not an actual submitted job.

CPU tests cover same exposed task list, default-budget drift, A/B/A and EOS,
exact1-token truncation and budget overrun, complete matrix, private-text
exclusion, exclusive receipts and static operator invariants. These tests do
not replace HF/Torch preflight on an allocated node. SciFact dev remains unopened
for model evaluation; cross-dataset identity audit is still a separate gate.
