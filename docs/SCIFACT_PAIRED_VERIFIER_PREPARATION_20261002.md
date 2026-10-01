# Paired v2/v3 three-arm preparation — no model result

This bounded CPU package prepares **24 identical consumed TRAIN frames × two
verifier protocols × three arms = 144 slots**, with **480 total call reservations**
at most. It supersedes the earlier 48-slot top1-only proposal; there is no extra
48-slot run. No model weights, new sample or protected split is needed during
preparation. No new Slurm job is authorized by this report.

## Frozen comparison and interpretation

- Isolated semantic-input v2 and relation/minimal-rationale v3 retain their
  existing verifier, planner, original scoring projection and five-call policy.
- Each protocol gets its own release, output directory and 72-slot / 240-call
  ledger. `fixed_top1` / `fixed_all` / `adaptive` retain call caps 1 / 4 / 5.
- The same ordered claim IDs, frames, source visibility, model archive, tokenizer,
  source commit, decoder configuration and original scorer are release-bound.
- Report v3−v2 separately for each **fixed** policy. Within each version report
  adaptive−top1 and adaptive−all. Across-version adaptive changes the verifier
  **and** the feedback observed by the planner; do not attribute its entire change
  to feedback alone. Do not treat unannotated sentences as adjudicated falsehood.
- Report paired positive wins/losses, strict-positive and NEI counts separately,
  unresolved episodes, all four original official micro metrics, physical calls,
  token lower bounds/unknowns and elapsed/load costs. Micro metrics are compared
  from the original scorer, not reconstructed by averaging per-case F1.
- Both full protocol audits must pass before a paired quality report exists.
  Failed/unstarted slots stay in the **144** denominator. `no_quality`, even with
  worker exit code zero, is not a successful paired run. The original two-document
  adaptive limit remains; the three-gold-document case is not removed.

## Complete decoder binding

The accepted generator manifest binds `generation_config.json` SHA
`2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2`.
Its file version is 4.51.0; the accepted execution runtime is Transformers 4.51.3.
The draft release stores the **complete expanded defaults**, including compile
configuration, and the worker compares its actual loaded defaults before any
call. A silent model-config fallback therefore fails closed.

Each physical call records the default configuration, actual overrides, complete
effective configuration and hashes before `generate`. Greedy decoding explicitly
sets `do_sample=False`, one beam, 512 new tokens, `max_length=input+512`, and
`use_model_defaults=False`. This last argument prevents model-default refill.
Inherited temperature 0.6 / top-k 20 / top-p 0.95 remain recorded but their sampling
warpers are inactive under greedy decoding; expected library warnings are not
evidence that sampling was enabled.

Python/NumPy/Torch RNG seed **20261002** resets per physical generation. The receipt
records CUDA/TF32/cuDNN settings; it does not claim cross-device bitwise determinism.
BOS/PAD are 151643, model EOS is [151645,151643], and tokenizer/LMFE EOS is 151645.
Stop rules remain EOS, derived length and `max_time`, plus the hard episode/worker
watchdogs. Dynamic `max_time` is computed after prompt preparation but **before**
seed/config-receipt persistence; it is not a fresh timer read at the final CUDA
call. The original 120-second episode watchdog remains authoritative.

Every generation uses a fresh LMFE parser/callback. The **post-callback** parser
alphabet is checked against tokenizer data, not just the constructor's alphabet;
actual 12-whitespace / non-forced field order / 20-array settings are recorded and
audited against the bound tokenizer. Returned diagnostics must agree. Binding is
released on every exit path, including decoder/tokenizer/deadline failure before
generation; the next episode can proceed without deleting the failed reservation.
Unknown usage stays unknown, never a fabricated zero.

## Bounded execution and failure costs

The new paired wrapper caps one A100, eight CPUs, 32 GiB RAM, 30 GiB scratch and
90 minutes total. Wrapper setup time is deducted from the supervisor budget;
each launched protocol retains its own 40-minute ceiling and 1,980-second worker
cap. Shared extraction is done once (bounded to five minutes), not once per
protocol. Synchronous preparation has explicit POSIX timeouts. Two GPU workers
are launched **serially and directly** by the supervisor, never as nested process
groups inside two old operators. A child is killed/reaped before proceeding.

No second protocol starts after a worker/infrastructure failure or unknown usage,
or when the complete registered phase budget is no longer available. This rule
uses execution evidence, not first-phase quality. Every phase reservation exists
before model preparation. Supervisor termination during preparation preserves
the first phase's ledger and the second phase's unstarted slots. Full cost is
written before scoring; both GPU workers must have exited before any gold access.
Each original scorer has a separate 180-second subprocess limit.

`g00` in v2 and `g00` in v3 are different physical calls. The pair never deduplicates
by bare call ID. Shared preparation is counted once; two actual model loads remain
separate. Hard SIGKILL/storage failure can still prevent a final aggregate file:
durable individual reservations are the recovery evidence and must not be retried
or declared free.

## Whole-chain review before queueing

Read-only cross-review and synthetic integration tests found and repaired:

1. Reusing the old operator twice would collide on exclusive model extraction and
   inherit the wrong outer timeout. One serial supervisor shares only read-only inputs.
2. Default refill could change explicitly configured greedy decoding. Actual kwargs,
   defaults and resolved settings are bound together.
3. A failed pre-generation attempt could leave a stale binding and poison the next
   slot. Journal `finally` now releases it while preserving failure cost.
4. Naively subtracting real official-score objects would hit strings/`None` only
   after GPU completion. Tests now use real `score_original` structure and compare
   only numeric metrics after checking scorer identity.
5. Preparation/phase-transition termination could omit aggregate cost. Supervisor
   signal/deadline handling retains the complete 144-slot roster and ledgers.

Reuse accepted real-frame CPU job **31969505** and compact SHA
`7e21b603c225374723122763f700f73dacc04f4005cde7784e23e837814cb388` unchanged.
The prompt/semantic-input preflight is not rerun; this package changes execution
binding and paired comparison, not prompt semantics or the dataset.

## Reproduction and authorization boundary

Exact execution source is **`5a12fd9f398afb1af772111253600e68410e885a`**.
[Preparation receipt](verified-runs/scifact-paired-verifier-preparation-5a12fd9.json)
binds the 5,591,040-byte / 682-file source archive, actual paired wrapper,
parent/two child drafts and Spartan runtime receipt. Archive SHA is
`978ebecddea29d939852f85444e988d9d6f6ca3cbd894050b810180645e6b85e`;
parent draft SHA is
`cd4c2767766604e8b95e40ba0c25f21a44a9baa000eeed95e1595b987ad0e620`.
Later documentation commits are not this execution source.

The clean archive passed **60 tests**, with one Windows-only skip for the real
POSIX timer test. [Exact-source Linux CI 36900289223](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36900289223)
then passed **all 1,393 tests** (105.43 s, 14 expected warnings), Ruff, typing of
108 source files plus the source packager, and tracked secret/PII scanning.
The actual archive shell guard and Bash syntax passed.

Spartan metadata/import-only preflight matched the accepted Python 3.10.4 /
Torch 2.1.2 / Transformers 4.51.3 / LMFE 0.11.3 runtime, verified all **70 expanded
generation-default fields**, rejected the draft for execution, and validated the
candidate contract only in memory. No authorized release was written. Runtime
receipt SHA is `46fbc47025f2b81a641fcc4457308dd96120ff15732f9f1269196c1b4a7e85da`.
The first ad-hoc SSH probe omitted the module's existing PYTHONPATH and failed to
import Torch; preserving the module path, as the checked-in wrapper already does,
resolved it without installing anything or changing project source. No model or
data was loaded by either probe. This was caught **before any queue submission**.

```bash
python -m pytest -q tests/test_scifact_paired_comparison.py tests/test_scifact_evidence_commit_entry.py tests/test_scifact_document_decoder.py
python -m ruff check src scripts tests
python -m mypy src/climate_rag
bash -n hpc/scifact_evidence_commit_paired.sbatch
python scripts/package_scifact_paired_comparison.py --help
```

The packager takes exact commit, accepted runtime receipt/observation and existing
prospective input compact. It emits one source archive, parent draft and two child
drafts, each with SHA identities. All drafts fail executable release validation.
Only a later coordinator exact-hash release can authorize the GPU job. This CPU
delivery adds **zero model calls, zero training and zero new Slurm submissions**;
it supplies no new quality gain and does not update the resume.
