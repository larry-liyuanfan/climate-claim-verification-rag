# Agent Linux/runtime preparation — 2026-09-29

Historical CPU-readiness handoff. The subsequently released GPU pilot and its
negative integration result are recorded in [the pilot closeout](BUDGET_AGENT_GPU_PILOT_20260929.md).
Authorization/status statements below describe the earlier preparation package.

This follows the [CPU control handoff](BUDGET_AGENT_CPU_HANDOFF_20260929.md).
No GPU job, paid API, production deployment, test-set reuse or new model matrix
is authorized by this preparation. Model execution still awaits the coordinator's
Trip → Energy → Climate resource release.

## Verified inventory and decisions

- The known Climate model cache did not contain the required generator or 4B
  reranker. Historical reranker run manifests are not proof weights remain cached.
- Generator is now explicitly pinned to `Qwen/Qwen3-4B` revision
  `350135a4de9a3407be836fa238cccc1d61503a85`; reranker to
  `Qwen/Qwen3-Reranker-4B` revision `22e683669bc0f0bd69640a1354a6d0aebcfeede5`.
  These are the existing planned model families, not a new outcome-selected search.
- The 333 MiB existing runtime archive depends on the Spartan module
  `GCC/11.3.0 OpenMPI/4.1.4 PyTorch/2.1.2-CUDA-12.2.0`, not a self-contained
  Python/PyTorch binary distribution. Module availability was read-only verified.
  Its own packages include Transformers 4.51.3 but not LangChain core.
- Copying the Python3.12 CPU lock failed correctly: `websockets==17.1` requires
  Python≥3.11. A separate CPython3.10/Linux wheelhouse uses websockets16.0 and
  explicitly includes exceptiongroup1.3.1; LangChain core1.6.5/Pydantic2.13.5 remain.
- 34 wheels, SHA-locked dependencies and a Python3.10/Linux metadata closure check
  were packed into one 12,475,491-byte archive. SHA-256:
  `e7223ac2f4c08a736e763d485a47dcb088ce62c28fb9f1da556661a3f71da0ce`.
  SCP completed and the remote digest matched. Linux joint imports subsequently
  passed in CPU preflight31486505; GPU inference remains unvalidated.
- No full restricted dense index is copied into the public study. The available
  old HNSW index belongs to a different corpus. This minimal released comparison
  uses the same 5,240-document public BM25 first stage for all three routes,
  plus real Qwen reranking/generation when released. It is not a new HNSW result.

Official immutable model documentation:
[generator](https://huggingface.co/Qwen/Qwen3-4B/blob/350135a4de9a3407be836fa238cccc1d61503a85/README.md),
[reranker](https://huggingface.co/Qwen/Qwen3-Reranker-4B/blob/22e683669bc0f0bd69640a1354a6d0aebcfeede5/README.md).
The generator card requires Transformers≥4.51; old Torch uses the repository's
existing pytree compatibility bridge. Its actual joint imports passed the CPU
preflight below; loading parameters and running forward/generation did not run.

## Immutable artifacts and paths

Only the existing isolated Climate root is used remotely:
`/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2`.

| Input | Location relative to that root | Verification |
|---|---|---|
| Existing module-dependent runtime | `envs/runtime-py310-2423a755979c426ecae5d17deead154393de712efd10e84884d8c87ff00d29f8.tar.gz` | 348,977,995 bytes; full hash gate passed on the allocated CPU node |
| New Linux overlay | `envs/budget-agent-linux-overlay-20260929-e7223ac2f4c08a736e763d485a47dcb088ce62c28fb9f1da556661a3f71da0ce.tar.gz` | Remote SHA rechecked; still packed, one persistent inode |
| Model/input bundle | `envs/budget-agent-inputs-20260929-563738f0be1f7bf7b99b8de20bcdeab552f7e893166dec260b8f7f1e7951c3c1.tar` | 16,122,255,360 bytes; transfer completed and full hash gate passed on the allocated CPU node |
| Final executable source | `envs/budget-agent-source-ca9fa53-74e64f5732ff82a59abf59c630ea39502b4a72f6a9a7878c939cbc14c7622fdc.tar` | Transferred and remote SHA rechecked; `SOURCE_REVISION` expands to ca9fa53f2d4b70091fd08a8a1512e263f329fc61 |

Read-only inode inventory found 1,207 free of 500,000 at the later check, after
an earlier628; no cleanup or quota mutation was performed. All persistent
preparation uses archives, not a new venv or Hugging Face cache tree.

## Reproduction / boundaries

```powershell
.\.venv\Scripts\python.exe scripts/prepare_agent_assets.py --lock configs/budget_agent_assets_20260929.json --output-dir artifacts/budget-agent-models-20260929
.\.venv\Scripts\python.exe -m pip download --dest artifacts/budget-agent-linux-wheels-20260929 --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 --implementation cp --python-version 310 --abi cp310 --abi abi3 --abi none --only-binary=:all: --no-cache-dir langchain-core==1.6.5 pydantic==2.13.5 langsmith==0.14.1 websockets==16.0 'exceptiongroup>=1.0,<2'
.\.venv\Scripts\python.exe scripts/package_agent_overlay.py --wheelhouse artifacts/budget-agent-linux-wheels-20260929 --output artifacts/budget-agent-linux-overlay-NEW.tar.gz
.\.venv\Scripts\python.exe scripts/package_agent_inputs.py --models artifacts/budget-agent-models-20260929 --evidence data/PUBLIC_EVIDENCE.jsonl --authored-protocol configs/budget_agent_vnext_20260929.json --validation-protocol artifacts/agent-validation-20260929/validation-protocol.json --output artifacts/budget-agent-inputs-NEW.tar
```

Use the supplied SHA-lock inside the downloaded overlay archive for exact
reinstallation; the pip command only recreates an initial resolver request.
Unpinned transitive choices in a future resolver run are not assumed identical.
`PUBLIC_EVIDENCE.jsonl` denotes the existing export, checked against frozen SHA
`c14315aee9feecbbbbc3b0c7101d978b52b47cee9a613336304bcaa736460c71`.

The model downloader requires full revision IDs, uses public HTTPS without
credential configuration (`curl -q`), downloads no remote Python, and checks
file size plus LFS SHA256 or Git blob identity. Final model SHA manifests cover
all loadable files. Bounded resumable ranges prevent restarting multi-GB weights
after one interrupted response. A partial file is never counted as verified.

The input packer accepts only the two previously frozen protocol SHAs and public
corpus SHA; it includes no benchmark gold, retired test, SciFact or restricted
evidence. It emits separate pilot/validation/vNext arguments and execution
manifests. 32-query validation replay remains exposed old validation, while the
8 authored examples remain an application demonstration without official gold.

`hpc/budget_agent_pilot.sbatch` now unconditionally activates the module-dependent
runtime plus downloaded overlay under unique node TMPDIR. Four persistent
archive inputs must resolve within the isolated Climate root and match supplied
digests. Only regular files/directories can be extracted; nonportable venv
executables are not restored. Pip is offline, hash-required and node-local.
Argument parsing rejects external paths, duplicate/options outside the allowlist,
output overrides and unmatched reranker manifests. `CLIMATE_AGENT_PHASE` selects
the appropriate pre-frozen `args-pilot/validation/vnext.json`.

Set `CLIMATE_RUNTIME_TAR`, `CLIMATE_RUNTIME_SHA256`, `CLIMATE_OVERLAY_TAR`,
`CLIMATE_OVERLAY_SHA256`, `CLIMATE_SOURCE_TAR`, `CLIMATE_SOURCE_SHA256`,
`CLIMATE_RUN_BUNDLE`, `CLIMATE_BUNDLE_SHA256` and a new `CLIMATE_RESULT_TAR` under
`runs/`. `CLIMATE_PREFLIGHT_ONLY=1` checks all large-file hashes, imports the real
Linux packages/Qwen implementation, reads model config/tokenizers, and saves a
readiness report; it loads no real model parameters and generates no tokens.
The report is not an allocation receipt: job/resources require separate sacct evidence.
This is not an inexpensive login-node check. The coordinator subsequently authorized
one CPU-only readiness allocation after complete inputs. Use the separate
`hpc/budget_agent_preflight.sbatch`: sapphire,4CPU,8G RAM,24G node scratch,
20min ceiling, **no GPU directives**, CUDA visibility disabled. The IO-dominated
estimate covers about16GB of model input extraction and two hash passes plus
dependency/tokenizer startup; it is not a measured runtime. GPU pilot retains its
separate resource request and is not authorized by CPU preflight success.

Both Slurm wrappers read the common launcher from the exact SHA-verified source
tar, so they do not depend on a sibling file beside Slurm's spooled job script.

Before any submission: explicit resource release, exact committed source/archive
identities, completed input bundle, and `sbatch --test-only`. The single authorized
CPU preflight completed; no GPU pilot was submitted. Its resource proposal remains15min,
8CPU/32G/30G scratch; measured pilot usage must determine any final job.

## Preparation state at this commit

Code commit549cb46 passed full remote CI: **165 passed, 1 skipped**, Ruff,
36-source-file type check and tracked secret/PII scan
([run36446648805](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36446648805)).
Nineteen new preparation tests also passed from the clean source archive,
without reading weights or running CUDA. Bash syntax validation passed in WSL.
The added CPU-wrapper check raises the full suite to166 passed/1 skipped;
final executable source ca9fa53 has two successful CI runs
([36448440950](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36448440950),
[36448448746](https://github.com/larry-liyuanfan/climate-claim-verification-rag/actions/runs/36448448746)).

Both models are now fully downloaded and verified locally (29 files, including
27 loadable files). Exact model file-manifest hashes, revisions and sizes are in
[asset report](verified-runs/budget-agent-model-assets-20260929.json); the frozen
input archive identity is in [input manifest](verified-runs/budget-agent-inputs-manifest-20260929.json).
Generator manifest SHA `8bff1d6532f4c1e29eb87c3d2230da9b718a353b6d8039cdec93ca9a566bf917`;
reranker manifest SHA `e1f56457935dd69b67e3249cbd81fd7aabaf5e2f0b870aa173b6fcb573b564ed`.

All archives have transferred. The one-shot continuation completed its byte-count,
duplicate/result guards and `sbatch --test-only`, then submitted one CPU job.
**31486505 completed on sapphire, exit0:0, elapsed76s,4CPU/8G and no GPU TRES.**
Start/end were `2026-09-29T02:24:00`/`02:25:16` in scheduler-reported time
(timezone not recorded). Eligible partitions were `cascade,sapphire`;20min/24G
scratch were request ceilings, not usage measurements.

The batch step recorded TotalCPU66.238s and MaxRSS8,384,124K (~7.996GiB), almost
the entire8GiB allocation. This successful run is **not evidence of safe8GiB
headroom**; no CPU rerun is needed. Scratch high-watermark was not captured.
CPU setup time/RSS do not establish GPU load time, generation speed or VRAM.

The575-byte result archive contains only `result/run.json` (813bytes). Its local
and remote SHA-256 matched
`036fe036b04dd5b4c00ceb1fb8304737cbb18cb2917abb82f9b734b4ac3de6bf`.
The [compact readiness evidence](verified-runs/budget-agent-cpu-preflight-20260929.json)
preserves the actual report, scheduler facts, input identities and provenance limits.
LangChain core1.6.5, Pydantic2.13.5, Torch2.1.2, Transformers4.51.3 and
websockets16.0 imported together; two local Qwen3 configs/tokenizers passed.
`cuda_available=false`, `real_weights_loaded=false`, `model_generation_calls=0`.

The report directly contains protocol/model-manifest hashes, but not job ID or
archive-input hashes. The latter are linked through the matching operator script,
source archive/SOURCE_REVISION and successful fail-closed launcher hash gates;
they are not invented fields of `run.json`. Model-file verification and the
public corpus hash gate precede the preflight report. Original log and compact
archive remain under Climate `runs/`; no large artifact is added to GitHub.

## Next pilot handoff — prepared, not released

Use the same ca9fa53 source and four archive hashes above; no source, model,
protocol or candidate tuning is introduced by this closeout. Only after an
explicit GPU resource release, run `sbatch --test-only` for the frozen
`hpc/budget_agent_pilot.sbatch` and one new result path under Climate `runs/`.
The existing proposal is1A100,8CPU,32G host RAM,30G scratch,15min ceiling.
It is an unvalidated GPU pilot envelope, not a full-run estimate derived from76s.
Do not carry CPU `CUDA_VISIBLE_DEVICES=""` or `CLIMATE_PREFLIGHT_ONLY=1` into it;
CUDA visibility must come from its Slurm allocation. `CLIMATE_AGENT_PHASE=pilot`.

- Scope:3 authored questions × fixed retrieval/fixed rerank/adaptive routes,
  same5,240-document public BM25 first stage; no dense/ANN or benchmark gold.
- Frozen generator/reranker4B revisions above; reranker batch1/max length2048.
- Per-query ceilings:model calls3, tool calls3, input8192/output512 tokens per
  model call, candidates20/context5,120s/query. At most15 generation attempts
  and7,680 generated tokens across the3×3 pilot; reranker forward passes are
  separate and are not included in the generation-attempt count.
- Measure actual loading/forward/generation, per-route stage latency, failure
  behavior, host/GPU peaks and emitted traces before proposing any larger job.
  No automatic validation/vNext follow-on is authorized. Authored questions are
  a demonstration, not independent quality/generalization evidence.

This package ends at verified CPU readiness. Passing full CI for executable
ca9fa53 is retained without rerunning the unchanged suite; this closeout changes
only documentation/evidence. No GPU job, paid API, career file or current resume
was changed. Existing CPU negative results, restricted LoRA dev findings and
independent-test restrictions remain unchanged; there is no new model-quality result.
