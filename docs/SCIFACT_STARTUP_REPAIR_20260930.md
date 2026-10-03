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

## Frozen experiment and preflight identity (before submission)

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

## Separately authorized retry: job31645005

After independent coordinator review and explicit authorization, **one** real
job31645005 was submitted at **2026-09-30 17:44:27+10**. New test-only simulation
31645003 was not another job. All13 CLIMATE values and all frozen source/data/model
identities above were reused without re-exporting source or resampling. The
exclusive r2 lock was created; original assets and the r1 lock remain untouched.

At17:45:34+10, `sacct` and `squeue` confirmed **PENDING / Priority**, priority13338
(FairShare13337 + JobSize1, Age0). Estimated start2026-10-01 17:50+10 is not a
guarantee. Slurm's actual spooled wrapper SHA matched `f6f38682...454f2f`; resources
remain1 A100/8 CPUs/32 GiB/30 GiB scratch/2h, normal QOS, Nice0 and no requeue.
See the exact [submission receipt](verified-runs/scifact-train-r2-submission-31645005.json).
There are no model results or quality gains yet. No automatic further retry or
dev/test evaluation is authorized.

## User-requested local Torch coverage

The user requested installation of missing local validation dependencies. Only
this checkout's ignored `.venv-validation` was extended with CPU Torch2.7.1,
PEFT0.15.2, Transformers4.51.3, Safetensors0.5.3 and Accelerate1.6.0, retaining
NumPy1.26.4. `pip check` passed. With Hugging Face/Transformers offline, the
complete suite now gives **516 passed / 1 Windows-only POSIX skip** in41.52s.
The real tiny Torch/PEFT checkpoint/toggle test executes and passes; it does not
download model weights or run benchmark inference. See the
[local validation receipt](verified-runs/local-torch-validation-20260930.json).

POSIX is an operating-system contract, not a missing pip package. Its ownership/
symlink test passes in the existing Linux CI run above. Windows-specific skipping
is retained instead of pretending NTFS is POSIX or changing the user's OS/WSL.
The frozen Spartan runtime, source80fcd07 and submitted job31645005 are unchanged.

Installing these previously optional libraries also exposed8 strict-mypy issues
at third-party type boundaries. Follow-up source `d7aad104deb64f5b65c203d7c63df350e4569de1`
adds a local PEFT callable annotation, type-check-only imports for Transformers'
real definitions (runtime lazy exports unchanged), and an equivalent explicit
boolean preflight assertion. No global ignores or model changes were added.
Strict mypy54, Ruff and24 affected-entry/real-dependency regressions pass, with
independent read-only review; a final offline full run after this correction
passed516 tests with the same one Windows/POSIX skip (44.19s).
This follow-up **is not** the source tar of job31645005;
the already-submitted package remains80fcd07 and was not rebuilt.

Local environment reproduction (use the isolated venv interpreter):

```sh
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install peft==0.15.2 transformers==4.51.3 safetensors==0.5.3 accelerate==1.6.0 numpy==1.26.4
python -m pip check
```

For the eventual unchanged job closeout, separate absent legal tool opportunity,
opportunity hidden by actual prompt packing/action constraints, visible opportunity
with premature stopping, and wrong decisions after useful feedback. Zero calls
alone do not measure feedback use. Any future evidence-gap package comparison
must acknowledge its changed visible-context cost; field-specific mechanism
attribution would need a separately frozen common-context control, not another
unapproved run. No new training, GPU job or holdout evaluation follows from this
interpretation guidance.
