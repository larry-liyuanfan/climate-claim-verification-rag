# Prospective conditional TRAIN24: whole-chain CPU input handoff

This package changes input selection and input plumbing only. It is **not a new
model result, independent test, GPU authorization or resume improvement**.
The decision policy remains `2ab219bb13564790b004797354dcc89554da2bf6`:
fixed-top1 / fixed-all / adaptive, caps 1/4/5, 72 slots, at most 240 physical calls.
Verifier, EvidenceVerdictRef, assembler, model, prompts and scoring mathematics
are unchanged. The runtime diff only reports the new shared preparation cost.

## Whole-chain review before submission

Review the actual consumer chain, not isolated scripts:

`metadata -> reservation -> gold-free claims -> BM25/CommonPacking -> frames
-> operator -> worker -> physical ledger -> post-exit scorer -> compact result`

The review identified and repaired old24 coupling in operator/worker/scorer/
preflight and historical-replay cost descriptions. All new consumers use one
input adapter. New input identity failures preserve completed generation costs,
return `no_quality`, mark unverifiable preparation cost unknown (not zero), and
never open gold or rerun inference. Old input defaults remain reproducible.

Before a cluster submission: targeted consumer-chain regressions, Ruff/type
checks, Bash syntax, clean exact-archive reproduction, actual environment import/
CLI checks, and `sbatch --test-only`. A long job is not a substitute for these
checks. Reuse accepted unchanged core checks; do not create serial review-only
jobs or retest consumed evaluation data for readiness. Static review cannot
promise absence of runtime failures; preserve failure receipts without blind
retries. No new GPU job belongs to this package.

## Data contract

- Remaining conditional pool: **64 components / 97 eligible IDs**.
- Component set SHA: `b421a1d70dd9e4f7c40b2b79731d392a4382e3d5c149a05f345a98270009a5eb`.
- ID set SHA: `56376abd5a8473a3738b37779221c6381c379864b5571e7057ee351db590068e`.
- Exact pinned metadata files bind the ID-to-component mapping as well as the
  marginal sets. Later natural/document/evidence-commit/utility8/NEI/mixed and
  old12/evidence-note reservation projections were reviewed: overlap zero.
- Salt `scifact-evidence-commit-prospective24-v1-20261002`, domain-separated
  SHA256 ordering, one claim per component, first 24 components. No labels,
  opportunity quotas, query text or outcomes select the roster; no replacement.
- Exclusive selection and component reservations precede selected query reads.
  A failure retains the same reservation, never resalts or replaces claims.
- All 531 eligible TRAIN rows had early gold-preparation exposure. Shared corpus
  also prevents a source-family-unseen claim. This is a conditional diagnostic,
  **not** globally unseen, independent validation/test or contamination-free.
- No supplemental supervision preparation, validation12, dev300 or frozen test.
  Selected IDs/query text, original corpus and private frames remain on Spartan.

## Actual no-generation preparation

Reuse all 5,183 original abstracts, unchanged SciFactBM25 scoring/order and
CommonPacking. Stop at the existing initial-frame observer before a physical
generation attempt exists. No fake generation prefix or scripted model answer.
The same frame serves all three arms. Aliases, sentence identities, candidate
order, selected ID order, tokenizer and file SHAs are checked by every consumer.

Record corpus/BM25 construction, 24 retrieval/packing timings and total shared
CPU preparation separately. This cost is paid once for a shared offline batch,
not multiplied by three, omitted, or claimed as end-to-end online latency.
Prompt probes reuse the existing fixed synthetic-feedback/first-two-ref cases;
their observed maximum is not an all-reachable-state bound.

Future scoring retains: child reaped -> physical cost -> complete 72 slots ->
physical response/ref audit -> selected TRAIN gold whitelist. This CPU package
performs **zero model calls and zero new real-gold deserializations**. Synthetic
tests exercise the successful scoring path and damaged-input failure path.

## Entrypoints and resource shape

- CPU preparation: `scripts/prepare_scifact_evidence_prospective.py` through
  `hpc/scifact_evidence_prospective_cpu.sbatch`: 2 CPU / 8 GiB / 20 min / no GPU.
- Shared adapter: `scripts/scifact_evidence_input.py`.
- Preflight: `scripts/preflight_scifact_evidence_commit.py --prepared ...
  --release ... --tokenizer ... --output ...` (old inventory mode retained).
- Future draft: `scripts/package_scifact_evidence_commit.py --input-compact ...`.
  It must reuse the **exact CPU preparation commit/archive**, not a later
  docs-only HEAD. Draft authorization remains non-executable.
- Future model proposal, not authorization: unchanged single A100 / 8 CPU /
  32 GiB / 30 GiB scratch / 40 min, supported by the prior 312 s run and bounded
  worker cap. No submission, priority change, retry or model tuning is implied.

Local affected-chain validation: **53 passed**; source/CLI type checking and
environment/archive checks are reported with the final preparation receipt.
No current resume or other project is modified.

## Completed CPU receipt

[Redacted compact](verified-runs/scifact-prospective24-cpu-31953981.json), SHA
`0a1e231bdfbf0d905eef130b331da5027520e21f80e7db0c741c49febad2dee2`.

Job **31953981**: COMPLETED / `0:0`, **38 s**, TotalCPU 30.393 s,
Slurm batch MaxRSS **301,504 K**, 2 CPU / 8 GiB / 20 min / no GPU. It ran on
the public `cascade` CPU partition after a successful partition-specific
test-only. Initial scheduling estimates were conservative; actual backfill
started promptly. No cancellation, priority modification or duplicate job.

- Candidate/source: `da243036871f61eef2e039a1618b8a3e1e1a00ac`.
- Source archive SHA: `d4c2ef55a9fc8675df2aba3e1bf537c22fcbd71fc8ee0c8f1f54df0a1421dbce`.
- CPU wrapper SHA: `a8113495686f5bee6f55de8f8d2df8831dae5f38e193f32e5887a8b8aa2eeb24`.
- Selection SHA: `351523c9c44bb418e24ed17b40372e6bef7b87792f8ad0b78dcaeb2d22e22754`.
- Component reservation SHA: `275bd306c9a154b583987ae63df148e3264c2989b671b92ac7b453fc9610a20f`.
- Preparation SHA: `23e1835a0e4f1330c2d11d0d70a2dbfa5c4655c2192169b05240433f5e5aae72`.
- Frame SHA: `e20a5188ba51bbaf940009338ee167d6d07bf4429652eaa06967265f37a4e29c`.

All **24 distinct components / 24 claims** were frozen before query reading;
all frames authenticated. Corpus load/BM25 build 1.159 s, 24 retrievals 0.101 s,
retrieval plus packing 24.187 s, preparation total 31.534 s. These are nested
measurements, **not additive totals**. Process-reported MaxRSS 307,656 KiB is
separate from Slurm's sampled batch MaxRSS. Probe maximum 6,764 tokens / zero
observed overflows, limited to the documented synthetic feedback cases.

Validation before actual submission:

- **53 affected-chain tests**; **11 clean-archive tests**, imported from the
  extracted candidate, not the editable worktree.
- Ruff, secret/PII scan, Linux strict mypy on 106 source files; narrow CLI typing
  on eight changed entries with imported legacy scripts excluded.
- [Exact-candidate Linux CI](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36884594874):
  **1,329 passed**. Native Windows whole-source mypy reports two pre-existing
  POSIX `resource` attribute errors; Linux target and actual Linux CI pass.
- Exact Git archive: 660 regular files, unique 41-byte revision marker;
  clean shell guard and actual CPU wrapper/import/argument checks passed.
- CPU imports explicitly used `USE_TORCH=0` and loaded no Torch/model; the
  tokenizer-only warning is intentional, not a missing GPU environment repair.

Future draft SHA `234f0e67c46b81a4ffc1eaf48309c24a3822eb011fc63001a94ae07e8e3f768c`
binds this preparation and candidate. Actual GPU wrapper SHA remains
`d30e1a2ebe2cc99630d776f2b9d3d365f1843d37a36a293dc6a0592c1895a815`.
Its authorization is **DRAFT_CPU_READY_NOT_AUTHORIZED**. No new GPU job, model
result, calibration, split consumption or resume claim is included.
