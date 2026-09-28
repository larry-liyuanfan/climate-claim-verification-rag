# Agent Linux/runtime preparation — 2026-09-29

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
  SCP completed and the remote digest matched. **Linux imports and GPU inference
  are not yet validated**; dependency metadata/download success is not that result.
- No full restricted dense index is copied into the public study. The available
  old HNSW index belongs to a different corpus. This minimal released comparison
  uses the same 5,240-document public BM25 first stage for all three routes,
  plus real Qwen reranking/generation when released. It is not a new HNSW result.

Official immutable model documentation:
[generator](https://huggingface.co/Qwen/Qwen3-4B/blob/350135a4de9a3407be836fa238cccc1d61503a85/README.md),
[reranker](https://huggingface.co/Qwen/Qwen3-Reranker-4B/blob/22e683669bc0f0bd69640a1354a6d0aebcfeede5/README.md).
The generator card requires Transformers≥4.51; old Torch uses the repository's
existing pytree compatibility bridge, whose actual joint imports remain preflight work.

## Immutable artifacts and paths

Only the existing isolated Climate root is used remotely:
`/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2`.

| Input | Location relative to that root | Verification |
|---|---|---|
| Existing module-dependent runtime | `envs/runtime-py310-2423a755979c426ecae5d17deead154393de712efd10e84884d8c87ff00d29f8.tar.gz` | 348,977,995 bytes, digest encoded in prior artifact name; full rehash enforced on node |
| New Linux overlay | `envs/budget-agent-linux-overlay-20260929-e7223ac2f4c08a736e763d485a47dcb088ce62c28fb9f1da556661a3f71da0ce.tar.gz` | Remote SHA rechecked; still packed, one persistent inode |
| Model/input bundle | `envs/budget-agent-inputs-20260929-563738f0be1f7bf7b99b8de20bcdeab552f7e893166dec260b8f7f1e7951c3c1.tar` | 16,122,255,360 bytes; local SHA verified, transfer in progress at handoff; full remote rehash must occur on CPU node |
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
Linux packages/Qwen implementation, reads model config/tokenizers, and saves an
allocation receipt; it loads no real model parameters and generates no tokens.
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
identities, completed input bundle, and `sbatch --test-only`. None has been
submitted by this preparation. Pilot walltime/resource proposal remains15min,
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

Source and Linux overlay archives have transferred. **The16.12GB input transfer
is still running at this handoff. No actual CPU job ID is claimed yet.** A one-shot
local continuation is waiting on this exact transfer process: check input byte
count/no existing result/no duplicate named job → repeat `sbatch --test-only` →
submit one `clim-preflight-ca9fa53` CPU job. It has no polling loop, GPU request,
automatic retry, Git/shared-career writes or authentication mutation.

Read-only scheduler trials accepted both general CPU partitions: sapphire had
an Oct8 estimate, cascade Sep29 05:57 at a Sep29 02:00 check. Estimates are not
guarantees. Final submission uses one job with eligible partitions
`cascade,sapphire`, the same4CPU/8G/20min/24G request, project working directory
and `runs/slurm-agent-cpu-preflight-%j.log`; it does not submit competing jobs.
The operator pins are saved remotely in `envs/submit-climate-agent-cpu-ca9fa53.sh`.
That script rejects incomplete input sizes, existing result paths and an active
job with its unique name before proceeding.

When submission succeeds, the local generated receipt is
`artifacts/budget-agent-cpu-submission-20260929.json`. The intended sole output
is `runs/budget-agent-cpu-preflight-ca9fa53-20260929.tar.gz`. A later check must
read that receipt/job state and verify archive/import/tokenizer results before
claiming Linux readiness. If pending, preserve age; if failed, inspect first.
GPU pilot remains separately gated. There are no new model-quality results.
Existing CPU negative results, LoRA dev findings and independent-test restrictions
remain unchanged. No current resume or shared career files were edited.
