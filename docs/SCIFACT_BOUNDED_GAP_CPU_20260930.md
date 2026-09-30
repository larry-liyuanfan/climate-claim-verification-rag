# Bounded generation repair and paired evidence-gap preparation

Status: **CPU preparation only; no new model inference or GPU submission**.
The [48-slot r2 negative result](SCIFACT_TRAIN_DIAGNOSTIC_CLOSEOUT_20260930.md),
its source80 archive, raw responses, scores and exclusive run locks are unchanged.
This branch starts at `f8f31706bf3222f9faf9078d97533ad4e0975eff`.

## Problem, implementation and boundaries

The prior run had 78 total-sentence violations and nine duplicate/excess-document
violations. Retrieval opportunity was not enough for an accepted grounded answer.
The new `bounded_scifact_grammar.py` intersects pinned LMFE 0.11.3 syntax with an
immutable document/SID/read tracker **before generation**, retaining five docs,
eight sentences per doc, twenty total, mixed labels and supplied citation order.
It prunes an impossible next array item; it does not truncate/deduplicate a
completed model response, select evidence for the model, or force a tool/abstention.
Answer/read/rewrite/rerank/abstain remain legal when offered by the frozen route.

The outer parser survives inner UnionParser transitions; actual HF callbacks
reject LMFE's silent ForceStop/EOS fallbacks and missing predecessor state.
Raw control characters in JSON strings are rejected at entry, avoiding an
unclosable prefix; JSON escapes and structural whitespace remain legal.

### Performance without broad token shortcuts

Initial no-shortcut probes were interrupted after more than 143 and 217 CPU
seconds in free text. These are **interrupted probes, not successful smoke runs**.
Reversing trie intersection iteration alone completed the synthetic rewrite in
27.491 seconds of callbacks. The final implementation additionally partitions
plain string tokens from every quote, escape, control and boundary-crossing
token. Only plain tokens use a bounded length/whitespace cache; exceptional
tokens still traverse every outer-parser character. The partition is tied to
the exact tokenizer-tree object, not merely its alphabet. The selected token
always goes through the pinned LMFE state-application path. The installed LMFE
dependency is not patched, and its broad `json_freetext` cache is not forwarded.

The [final pinned-tokenizer CPU report](verified-runs/scifact-bounded-prefix-cpu-20260930.json)
passes 17 synthetic cases, including 8+8+4, 8+7+5, five documents, uniqueness,
Unicode/escapes, F and G envelopes, A/B/A, and rejection before a 21st sentence.
Two real-vocabulary differential probes agree on every step's allowed-token
set; their final sets contain 147,143 and 4,477 tokens, with EOS absent in both
incomplete states. These probes cost 14.393/15.806 seconds because they also run
the slow reference. Final optimized rewrite callbacks total 0.527 seconds;
the slowest single callback across all seventeen cases is 0.524 seconds.
These are synthetic CPU observations, **not online SLA or model-quality gains**.

## F versus F+G

`local_bounded_scifact_provider.py` preserves the verified local loader, model
manifest checks, greedy decoding, non-thinking template and physical token cost.
It uses a fresh constrained parser for every generation. F retains the original
prompt; G uses the existing zero-shot `evidence_gap_candidate` envelope and a
fresh adapter per question. Its state/gap are fallible model reports, not evidence
or trained reflection. Exact nonempty claim-span validation remains posthoc;
failure is charged and receives concrete bounded feedback, not a silent repair.

`scifact_bounded_agent.py` is a versioned copy of the frozen controller. Its
only policy-adjacent changes are a common fitting hook, a concrete invalid-gap
message and a version receipt. Route/tool/budget logic and strict action parser
are unchanged. F/G fit using max(F prompt, empty-history G prompt, active actual
prompt), while accounting bills each arm's **actual** tokens. This deliberately
changes packing relative to r2; a difference versus r2 cannot be attributed to
format alone. Actual initial ordered visible IDs/hashes and preview IDs/hashes
must agree between F/G or the pair is marked incomparable with costs retained.
Later history/tool-induced context differences are part of the G package, not
claimed equal-capacity observations. R2's exact preview IDs were not recorded;
only its ordered full-sentence views and prompt counts can be compared.

The runtime separates raw wire action, strict status, actual event, and next
feedback/decision. Actual previews are hashed after packing, not inferred from
the Top-K pool. No-tool answer and abstention are possible in both arms.

## Storage, protocol and release

Each arm has one exclusive preflight directory and 48 exclusive slot directories.
Each directory allows at most ten files for five model calls, with a byte cap of
`5 * (wire_bound + 16384)`, where wire_bound is at least 32768 and covers 512 times
the largest tokenizer token's UTF-8 size plus a margin. Truncated, missing or
unwritten raw responses are not complete receipts. The runner persists incurred
costs, then stops an unauditable batch. The post-inference scorer checks physical
response multisets, retaining repeated identical outputs as separate files.

The [frozen paired protocol](protocols/scifact-bounded-gap-pair-20260930.json)
has SHA `4cc5033928a0833ee99ad64ecd5a7b10c7a1fde7ac4f219d6334d2349d9daaf6`.
It keeps the same twelve biased eligible-train components, four routes, models,
data and V3Budget. There are **96 slots split into two serial 48-slot batches**.
Each proposed batch requests 1 A100, 8 CPU, 32 GiB, two hours:
48 × 120 seconds = 96 minutes, leaving 24 minutes for staging/load/preflight.
One two-hour job cannot cover 96 × 120 seconds; query caps were not shrunk.
Prior allocated MaxRSS was 18,527,100 K with elapsed 842 seconds. Both arms must
use identical frozen source/protocol hashes before the first call; no interim
policy tuning is permitted. G requires a completed F receipt with matching
source/archive SHA. Gold is extracted only after **both** inference children exit.
External scoring/inference/grammar hashes are checked against the preregistered
values before any extraction. No SciFact dev300 or retired Climate test is read.

`hpc/scifact_bounded_pair.sbatch` passes local and Spartan `bash -n`.
Its SHA is `8275b87c94c5862f410e6b2318f4c3f5e129b3bbb829a863d614544234ec313a`.
`sbatch --test-only` returned zero and a scheduler estimate of
2026-10-06 07:12:07 for the A100 shape. This is **not a job submission,
reservation, or promised start time**. No actual `sbatch` was issued.

## Verification and reproducibility

- Isolated Windows validation: Python 3.12.14, Torch 2.7.1+cpu,
  Transformers 4.51.3, PEFT 0.15.2, LMFE 0.11.3. Torch tensor smoke passed.
- Full suite: **587 passed, one Windows POSIX-only skip**, 35.10 seconds.
- Ruff `src scripts tests` and mypy all **61 source modules** passed.
- Synthetic/mock tests are not model results. The POSIX check requires Linux,
  not a fictitious `pip install POSIX`; Linux CI remains separately reported.

```powershell
.venv-validation\Scripts\python.exe -m pytest -q
.venv-validation\Scripts\python.exe -m ruff check src scripts tests
.venv-validation\Scripts\python.exe -m mypy src/climate_rag
.venv-validation\Scripts\python.exe scripts/smoke_bounded_scifact_prefix.py --tokenizer data/qwen3-4b-tokenizer-v3 --output artifacts/fresh-bounded-smoke.json --differential
```

The scripts `run_scifact_bounded_arm.py`, `run_scifact_bounded_operator.py`, and
`score_scifact_bounded_pair.py` are prepared inference/orchestration/scoring
entry points. They are **not evidence of a completed paired experiment**.
The fixed clean-source preparation is complete; actual GPU release remains with
the coordinator. Training, dev/test evaluation, default-branch integration,
deployment and current-resume edits are outside this package.

### Final CPU closeout

The [compact receipt](verified-runs/scifact-bounded-cpu-closeout-20260930.json)
freezes execution source `b248fe7f43b175b15b90ec6143538d5dab8e2f35` and tar SHA
`dbc3ef071b4144934f568fa728a51ac3f07e770c0a9acf3a83e17861279a7bbb`.
The 2,478,080-byte archive has 348 regular files, 13 directories and exactly
one 41-byte `SOURCE_REVISION`; all content matches the Git-blob allowlist.
Its only change from `fd33425` installs CPU Torch/Transformers/PEFT validation
dependencies in CI. The inference code, frozen protocol and budgets are identical.

- Fresh `fd33425` archive extraction and verified import path: **587 passed,
  one Windows POSIX skip**, 36.59 seconds, using the existing pinned isolated venv.
- Final [Linux CI 36705532161](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36705532161):
  **588 passed, zero skips**, 12.02 seconds, including the POSIX invariant and
  Torch-dependent checks. Dependency consistency, Ruff, mypy and secret scan pass.
- Earlier Linux `fd33425` CI was **579 passed / nine missing-Torch skips**;
  it is a historical result, not retroactively claimed as full coverage.
- Final source was uploaded to the Climate-only `envs` target in the receipt.
  Remote source/asset SHA checks, paired-wrapper archive guard and `bash -n` pass.
  At 2026-09-30 11:02:36 UTC, `sbatch --test-only` returned zero and estimated
  2026-10-06 06:27:36; its printed **31685002 is a simulation ID, not a submitted
  job**. The exact-name active queue was empty and both paired run directories
  were absent. No compute was submitted by this CPU closeout.

The receipt is documentation about the frozen execution commit, not part of that
commit's self-referential archive. These results establish execution readiness,
not a completed F/F+G experiment or an improvement in model quality.

The legacy packager also exports `artifacts/bounded-source-b248fe7/wrapper.sbatch`
and its `f6f386...` hash: that is the old `scifact_train_diagnostic.sbatch`, **not
the submission entry point**. Use only the separately guarded paired wrapper
`hpc/scifact_bounded_pair.sbatch` / remote `scifact-bounded-pair-8275b87c.sbatch`,
SHA `8275b87c94c5862f410e6b2318f4c3f5e129b3bbb829a863d614544234ec313a`.
