# SciFact source-package startup repair — 2026-09-30

## Failure and scope

Job **31620529** failed `1:0`, with Start=End **2026-09-30T17:15:42+10:00**,
Elapsed **00:00:00**, batch MaxRSS **18564K**, and an empty stdout log. Its result
directory was not created. This is an infrastructure failure, not a negative
model prediction or a completed 48-slot evaluation.

The actual Slurm-spooled wrapper and archived wrapper had the same SHA256:
`351ab5f7ef1f8e1762d99400c6db5cc47500ac30227e5a00c3551f0affc41ca4`.
All four initial archive checksums and their canonical project paths passed.
The frozen source commit was `d0e2339cb9711d4b46211d88c561d2f80bd39b3c`;
the failed LF archive SHA256 was
`0903d8d2533198a7c3921d2ca4ef887aa5cf331aa695fa99083262ccff8f6794`.
Retain that archive, original submission lock and empty log without replacement.

**Root cause:** `SOURCE_REVISION` was already tracked as `$Format:%H$` and had
`export-subst`. Git archive expanded it; a manual `--add-virtual-file` added a
second member. The 41-byte and 40-byte payloads were concatenated by real
`tar -xOf`. After `tr -d '\r\n'`, the value contained the SHA twice and the
unchanged equality check exited 1 under `set -e`. This occurs before `module load`,
scratch/operator creation, gold extraction and model loading: **0 model calls**.
The prior member audit exempted revision markers without enforcing their count;
per-file hashes alone missed this packaging error.

## Minimal fix

- `scripts/package_scifact_source.py` runs from a clean exact-HEAD checkout using
  `git -c core.autocrlf=false -c core.eol=lf -c tar.umask=0022 archive`. It never
  appends a marker or reads prepared claims/gold/model assets.
- The full recursive Git tree defines the allowlist: every regular file, every
  parent directory, exact file mode and every Git blob byte must match. Only
  `SOURCE_REVISION` has the declared export-subst transform, exactly one regular
  member containing the source SHA plus LF (41 bytes).
- Reject missing/extra files or directories, mutated blobs, normalized-name
  collisions, noncanonical paths, links and special members. The existing
  extraction guard remains; the wrapper checks marker count/size and strengthens
  the SHA comparison to exact SHA+LF (no stripping embedded CR/LF), retaining the
  exact wrapper hash check.
- Packaging extracts the guard **from the frozen wrapper** and runs real Bash,
  tar and checksum tools before writing a success receipt. Git Bash drive paths
  are converted to `/e/...` so GNU tar cannot interpret `E:` as a remote host.
- Startup failures print only line number and exit code, never commands or env.
- `create_attempt_result` separates the frozen experiment from the exclusive
  output directory; arbitrary attempt IDs, overwrites and changed release IDs fail.

## Experiment unchanged; retry not yet submitted

| Identity | Value |
|---|---|
| Frozen experiment / protocol release | `climate-scifact-train-diagnostic-20260930-r1` |
| Proposed infrastructure attempt | `climate-scifact-train-diagnostic-20260930-r2` |
| Infrastructure retry number | `1` (not yet submitted; maximum authorized retries: 2) |
| Inference archive SHA256 | `92846f0904992e7b7137e550c46cd0c76b2f77b4f808a6d6be25482f90bf1cca` |
| Scoring archive SHA256 | `174ea64c476fc3abdd5575d53e9a1808c8fd5e402e07cf327e58295346d41444` |
| Frozen protocol SHA256 | `306e8a61eef6fb279c56ccfc2b496281df9cc8710e194d2272e6978a23dbe802` |
| Grammar archive SHA256 | `36ed51e0a9a71ded058cc0e7290a78a7f5f638b91bff517d49ad6ea83c70e410` |

The initial 12-claim / four-route matrix, query order, 5,183-document corpus,
models, greedy decoding, token budgets, scorer and inference-then-gold dataflow
are unchanged. No resampling, dev/test consumption, candidate-policy change or
comparison result is part of the repair. The evidence-gap CPU candidate remains
on its own branch/PR18, not in this baseline.

Future operator metadata contains `release_id=r1`, `attempt_id=r2`, `infra_retry=1`.
Only the attempt ID names `runs/<attempt_id>` and the independent submission lock;
the inference child still receives the original r1 release and protocol hash.
No r1 output or lock is reused, removed or overwritten.

## Reproduction and release gates

From the clean repair commit, with an ignored, **new** output directory:

```sh
python scripts/package_scifact_source.py --commit <full-repair-SHA> --output data/scifact-source-r2 --bash /bin/bash
```

On Windows use the existing Git Bash executable via `--bash` (no installation).
This produces `source.tar`, the exact `wrapper.sbatch` and `source-receipt.json`.
The utility contains no Slurm submission command. Synthetic regressions include
the original Git `--add-virtual-file` failure using real `tar -xOf`, malformed
archives, wrong wrapper bytes, clean-source provenance, repeatable packages,
exclusive output paths and r1/r2 identity separation. No fixture score is a
model-quality or retrieval-quality result.

After complete tests, lint/types, clean-source reproduction and an independent
review of the new hashes, use `sbatch --test-only` before any submission. Only
the coordinator may release the new attempt. Preserve the original resource
shape and experiment configuration; do not add a competing or automatic retry.

## Verified repair package (17:39 +10)

Frozen source `80fcd07faed28890b70096386c5897f0343d3104` passed the 32 new
synthetic regressions and a clean LF-checkout full run: **515 passed, 2 local
skips** (optional Torch; POSIX-specific ownership on Windows), Ruff, mypy54,
shell syntax and tracked secret/PII scan. The clean checkout reused the existing
validation venv with an explicit checkout `src` path; it was not a fresh dependency
installation. Its independently repackaged tar was byte-identical. GitHub
[CI36684713686](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36684713686)
also passed for that exact source.

New archive SHA256 is `3c3c80d8dbcd3cdfce4bd2a711e0c9a5e23b74fc9c967daf960554f114ee6646`
(2,201,600 bytes; 321 regular files / 12 directories), wrapper SHA256
`f6f3868262dadc81d0d6ececa068dbafddf29eb687af27fa6d90f25edf454f2f`.
Spartan independently passed all archive/path/protocol hashes, marker and wrapper
guards, and shell syntax. `sbatch --test-only` succeeded at17:39:04+10;
simulation ID31644047 and estimated start2026-10-06 04:15+10 are **not a submitted
job or promised schedule**. Neither the r2 lock nor result directory was created.
The content-free [receipt](verified-runs/scifact-startup-r2-preflight-20260930.json)
records the exact 13 explicitly set CLIMATE variables; no environment dump,
dependency installation, sampling or model call was performed. Await coordinator
review before the first infrastructure retry.
