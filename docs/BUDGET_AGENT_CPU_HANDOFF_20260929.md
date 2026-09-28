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

Proposed release sequence: verify existing generator/reranker archives and exact
model hashes without unpacking to project storage → node-local short pilot,
one A100 40GB slice/card, 8 CPUs, 32G RAM, 30G node-local scratch, **15 minutes /
0.25 allocated GPU-hours ceiling** → use measured peak RAM/VRAM and elapsed to
derive one final job. Generator Qwen3-4B + optional Qwen3-Reranker-4B + 0.6B query
encoder nominal BF16 weights about 17.2GB before activations, so a 20GB allocation
is not assumed sufficient. Pilot runtime is an estimate, not a measured speed.
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
