# Climate evidence-driven Agent — CPU-ready package (2026-09-29)

## Scope and ownership

Worktree `E:/Project/_codex_worktrees/climate-representation-eval`, branch
`codex/climate-search-tradeoffs-20260927`, baseline
`a42c24b5ea9b314c2fdc01549decae2ad2bee636`. Personal extension, not an assertion
that the original team project was solely authored here. No current resume,
career shared file, default branch, authentication or other project's code changed.

## Implemented decision loop

`AgentRequest` rejects gold/extra fields → real LangChain retrieval tool → exact
final-context evidence ledger → bounded decision → one optional rewrite/RRF or
one optional rerank → observe again → cited answer or explicit abstention.

- `budget_agent.py`: strict decision/payload schema; entity/year/number coverage;
  at most 3 attempted tools / 3 decision calls, one rewrite and one rerank;
  final-turn reservation, repeated-query/no-new-evidence stops; monotonic deadline.
- The ledger is regex/lexical, **not semantic sufficiency**. The model's own
  `evidence_assessment` is not an independent correctness judge. Entity retention,
  number retention, polarity/quantifier preservation and lexical drift guards
  reject obvious rewrite drift; passing them does not prove semantic equivalence.
- Answers require model-declared sufficient evidence, exact quotes and IDs from
  the model-visible Top-5, and numbers contained in each corresponding quote.
  Operational failures return abstention, not a measured semantic NEI verdict.
- Attempted tool/model calls are counted even when they fail. Generated malformed
  JSON retains measured tokens; failures before accounting expose `usage_known=false`.
  Late results are rejected and excluded from delivered-evidence scoring. This is
  not hard preemption of arbitrary synchronous Python work.
- `local_agent_model.py`: real Transformers Qwen local-generation entry, complete
  weight/config/tokenizer manifest checks, local-only loading, bounded input and
  output, non-thinking JSON prompt. **Not executed against model weights yet.**
  API follows the [official model card](https://huggingface.co/Qwen/Qwen3-4B/blob/350135a4de9a3407be836fa238cccc1d61503a85/README.md).
- `run_budget_agent.py`: same corpus, generator and budget ceilings for fixed
  retrieval / fixed rerank / adaptive. Actual calls can differ and are reported.
  Existing dense/HNSW and real Qwen reranker can be supplied explicitly; default
  CPU pilot uses BM25 + deterministic feature reranking, **not Qwen4B**. LTR remains
  an existing separately measured alternative; it is not silently stacked here.
- `score_budget_agent.py`: predictions first, separate gold file afterward;
  Recall@5, MRR@10, nDCG@10, Evidence F1@5 and 5,000 paired bootstrap when gold exists.
  No gold means unavailable metrics, never zeros. Citation/numeric integrity is
  not semantic supportability; human assessment and model-judge metrics are absent.

## Frozen task boundary

`configs/budget_agent_vnext_20260929.json` contains 3 authored CPU/prompt pilots and
8 distinct authored vNext application queries. No retired test claim/label was
opened, no SciFact/training job was added. **vNext is a frozen application set,
not a new independent generalization benchmark.** It has no official evidence
gold; official Recall/F1 cannot be reported for it. Any future official-score
replay must use separately labeled allowed train/validation data and preserve
their exposure, rather than relabeling these questions or old dev as held-out.
New independently annotated supportability is still not available.

Task/budget protocol SHA: `6ee90b8479192335fc543c3425c3fbbb89fea2d65ba9698fb54d86564bc3c3c2`.
Real local-model execution additionally requires `--execution-manifest` and its
expected SHA, binding protocol SHA, model-manifest file SHA, optional reranker
manifest SHA and every dense-index file digest before inference. This is not
silently inferred from a mutable model name. Generator-call/token counters exclude
encoder and reranker work; separate retrieval attempts / rerank-pair attempts are
recorded (and neural-work totals identify which backends were actually enabled).

The vNext phase refuses the heuristic provider. Receipt creation prevents silent
reuse of a chosen output path, not intentional manual creation of another path;
the evaluation policy still forbids tuning on a consumed final set. Freeze exact
model manifests and retrieval config after a released pilot **before** running
vNext; retain failures, do not tune/retry for better scores.

### Separate public-qrels replay (not the 8 authored examples)

`prepare_agent_validation.py` verifies the archived public-v2 preflight SHA and
reads **only** its `selection-only/validation-claims.json` member, never the combined
claims or test seal. It sorts IDs by `SHA256(agent-validation-20260929:<id>)`, taking
24 of 126 evidence-bearing validation claims and 8 of 104 nondecisive claims.
Selection is fixed without model outcomes. Retrieval denominator is 24; calls,
latency and failure denominator is 32 per route. This validation was used before
for model selection; it is not a new held-out test. Queries and gold are separate
files; gold cannot be passed to the runner/controller. Classification accuracy is
only enabled for a real model on available decisive SUPPORTS/REFUTES labels;
no heuristic/no-provider output is counted as classification effectiveness.

- Validation protocol SHA: `abfcb61e9e6a54641f025101a4011fa88e07e3697a36f0acd308fa8e86052674`.
- Normalized-claim gold SHA: `d2dd28422bffacf87ded2153b3bfac4ca9e1edc903e1d4e7a88d3a40f5d2fd8b`.
- [Selection manifest](verified-runs/budget-agent-validation-selection-20260929.json).
- The first scorer attempt caught raw-vs-normalized whitespace hash mismatch;
  gold hash generation was corrected to the runner's documented whitespace
  normalization. No retrieval/generation was rerun to change results.

### Actually measured CPU results

Implementation/pilot `401f8b8`, validation runner `7167686` (both clean at execution).
5,240 real public documents; 9 authored-pilot runs and 96 validation route runs;
**zero generative model calls**, all 96 correctly retain answer unavailable.
This is a heuristic routing + BM25/deterministic feature-reranker development
control, not a Qwen answer-quality demonstration.

| CPU route | Recall@5 | MRR@10 | nDCG@10 | Evidence F1@5 | Tools/query | P50/P95 ms |
|---|---:|---:|---:|---:|---:|---:|
| Fixed BM25 | .43194 | .40417 | .38759 | .25784 | 1.000 | 3.00 / 4.64 |
| Fixed deterministic rerank | .36389 | .37655 | .36985 | .21260 | 2.000 | 4.77 / 6.74 |
| Heuristic adaptive control | .37778 | .35810 | .36294 | .22302 | 1.594 | 3.57 / 8.13 |

The added heuristics **did not improve** the fixed BM25 control. Adaptive R@5
delta −.05417 (5,000 paired-bootstrap 95% CI −.13194, +.00833); F1 delta −.03482
(−.08929, +.01250). No promotion/no retuning on these outcomes. These single-pass
in-process timings exclude index construction; 32 requests are not online SLA.
[Full scores and paired intervals](verified-runs/budget-agent-cpu-score-20260929.json).
Raw run SHA `96f0b946a7f0068f970f3179f123e703c9286711a12a564255c7f2141e04b7aa`;
raw Windows score SHA `721fdd8a3c982fd74f521e34d5e848b44c6f61f4542a53a7bf8c310b4ebd8d19`;
the public JSON copy is LF-normalized by Git, with identical values.

Targeted 40 tests passed (25 new +15 prior integration); new modules pass strict
mypy and Ruff. Clean `git archive` export at `401f8b8` independently ran the 25
new tests. Full heavy suite was not repeated locally; remote CI remains a
separate status, not inferred from these checks. Three CPU pilot cases cover
single-topic evidence, compound evidence, and empty/OOV; fixture cases cover
wrong citation, missing year, entity/polarity drift and late result rejection.

## Reproduction

Use the existing `.venv` / `requirements-agent-demo.lock`; no new environment is
needed for CPU. Read only the known 5,240-document public `evidence.jsonl`, SHA
`c14315aee9feecbbbbc3b0c7101d978b52b47cee9a613336304bcaa736460c71`.

```powershell
$protocolSha = (Get-FileHash configs/budget_agent_vnext_20260929.json -Algorithm SHA256).Hash.ToLower()
.\.venv\Scripts\python.exe scripts/run_budget_agent.py --evidence E:/Project/climate-claim-verification-rag/data/climate-fever-20260825/evidence.jsonl --protocol configs/budget_agent_vnext_20260929.json --expected-protocol-sha256 $protocolSha --phase pilot --provider heuristic --output artifacts/budget-agent-pilot-20260929.json
.\.venv\Scripts\python.exe scripts/score_budget_agent.py --run artifacts/budget-agent-pilot-20260929.json --output artifacts/budget-agent-pilot-score-20260929.json
.\.venv\Scripts\python.exe -m pytest tests/test_budget_agent.py tests/test_budget_agent_scoring.py tests/test_langchain_evidence.py tests/test_agent_evidence_packet.py -q
```

## Resource request — NOT a submission

Read-only SSH preflight on 2026-09-29 found project filesystem 467G total, 171G
used, 297G available; **499,372 / 500,000 inodes used, only 628 free**. Scoped
`squeue -u "$USER"` returned no active jobs in that account at that instant;
this is not permission to consume GPUs or proof other accounts are idle.
The coordinator controls Trip → Energy → Climate ordering. No GPU/API call,
remote environment creation, model download or cleanup performed.

Prepared executable launcher: `hpc/budget_agent_pilot.sbatch` (Bash syntax checked,
not submitted). One immutable `git archive` source tar plus one input bundle;
safe member validation and extraction only under unique node TMPDIR; existing
locked runtime via explicit `CLIMATE_PYTHON`; persist one result tar plus Slurm log,
not a new project venv/cache. `SOURCE_REVISION` is expanded by git archive.
Input bundle contains public evidence, chosen protocol, execution/model manifests,
and operator-authored `args.json` (a CLI argument array, `{INPUT}` is replaced by
node-local bundle path). Existing model directories remain read-only. Runtime
checks exact LangChain 1.6.5; no pip/network fallback.

- Preflight: use the same launcher with `CLIMATE_PREFLIGHT_ONLY=1`; imports and
  model/data/manifest checks only, no model generation. Run inside a released
  allocation, not a login-node model/hash workload.
- Pilot: args select `--phase pilot`; 3 tasks ×3 routes, at most 15 generation
  attempts /7,680 new tokens. Explicit release then `sbatch --test-only` first.
- Validation/full: args select `--phase validation` and frozen replay protocol;
  32 ×3 runs, at most160 generation attempts /81,920 new tokens. Optional authored
  vNext is separate 8 ×3 application runs, not added to the quality denominator.
  Override walltime only after measured pilot establishes a bounded estimate.

Proposed release sequence: verify existing generator/reranker archives and exact
model hashes without unpacking to project storage → node-local short pilot,
one full A100 (`gpu-a100`, `gpu:A100:1` observed by sinfo), 8 CPUs, 32G RAM,
30G node-local scratch, **15 minutes /
0.25 allocated GPU-hours ceiling** → use measured peak RAM/VRAM and elapsed to
derive one final job. Generator Qwen3-4B + optional Qwen3-Reranker-4B + 0.6B query
encoder nominal BF16 weights about 17.2GB before activations, so a 20GB allocation
is not assumed sufficient. The available MIG inventory showed 10/20GB slices,
so a nonexistent 40GB MIG type is not requested. Pilot runtime is an estimate, not a measured speed.
Tentative final ceiling 45 minutes / 0.75 GPU-hours, contingent on pilot; no request
has been submitted and no `sbatch --test-only` scheduling claim is made.

Release blockers: explicit GPU slot release; inode capacity/approved archive
reuse; confirmed local generator weights + full manifest; GPU runtime lock
including compatible LangChain/Pydantic (existing old Py3.10 archive is not
assumed compatible with the current Py3.12 local lock). No production deployment.

## Usable wording now

个人扩展中，将气候证据检索封装为带实体/年份/数值账本的受限工作流，实现一次补检索、
可选重排、引用与数值校验及失败弃答，并保留固定链对照和独立计分入口；CPU 合同与
真实公开语料调用已可复现，真实模型回答质量仍待资源释放后验证。

Do not replace the stronger existing LoRA / quality-latency resume results with
unmeasured Agent claims. The real-model E2E demo, answer correctness comparison,
semantic supportability and economic/API cost are **not delivered yet**.
