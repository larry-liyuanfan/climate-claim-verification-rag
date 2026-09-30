# Frozen Agent comparison: execution complete, quality not established

**Decision: retain the negative result; do not promote or retry.** Job31543304
completed all120 planned task/route slots, but yielded0 mechanically accepted
answers. There were105 schema failures,14 model-requested abstentions and1
exact-quote rejection. No model-directed rewrite or rerank executed. Completing
the infrastructure does not demonstrate an effective autonomous Agent.

This CPU-only closeout reads the original frozen scores and small receipts; it
does not change prompts, budgets, models, protocols, predictions or the scorer.
It adds explicitly posthoc descriptive diagnostics, not a new selection cycle.
PR18 remains Draft. No deployment, GPU job, training, inference rerun or current
resume edit belongs to this closeout.

## Provenance and denominator contract

- [Content-free compact](verified-runs/budget-agent-full-31543304.json):51,090 bytes,
  SHA256 `0094b372d98144d90fe9c09c4998bca29dda9d189cc482753aa50f363a86082f`.
- [Resource/audit receipt](verified-runs/budget-agent-full-resources-20260930.json)
  records Slurm step accounting, the failed preparation attempt and audit cost.
- Inference `72eaa90567e3603a3b940edffbb21b684bf1d1ba`; operator
  `947645b3feed914e30d02dc5dbd05859c4a19667`; frozen scorer
  `208ff931badff270cbbc9593c8ab54f5c79aec9e`.
- Qwen3-4B generator revision `350135a4de9a3407be836fa238cccc1d61503a85`;
  Qwen3-Reranker-4B revision `22e683669bc0f0bd69640a1354a6d0aebcfeede5`.
  Full source/model/config/corpus/run/score hashes are in the compact. Existing
  bundle receipts are reused; the16.1GB model/input archive was not rehashed here.
- Corpus:5,240 public documents, **dense disabled**. Initial BM25 candidate20,
  context5; fixed-rerank adds the4B cross-encoder. This is not the restricted
  million-document ANN/LoRA experiment or a new public frozen-test result.
- Validation:32 frozen tasks ×3 routes =96 slots, a repeated validation replay.
  Evidence metrics have24 gold-bearing claims; label delivery has23 decisive
  labels. The gold distribution is17 SUPPORTS,6 REFUTES,1 DISPUTED,8 NEI.
  DISPUTED explains why the two denominators differ.
- Authored-vNext:8 tasks ×3 routes =24 slots. These have no official gold;
  official retrieval/label metrics remain **null**, not zero.
- Budget:at most3 generations,3 tools,8,192 input tokens and512 output tokens
  per generation,120 seconds per row. In practice every row generated once.

The scorer archive SHA is
`7498d367565eeb59bb7cf724c33f835608a0c9f01da3768d7308700a380c00a1`.
Its actual Python member is17,097 bytes with292 CRLFs, SHA
`758bb04e9d99cdfb298c936de550dc66055a706fbe82bd19e1f5db200ccf3b86`,
matching both frozen score identities. CRLF-to-LF normalization alone yields
16,805 bytes and SHA `5a3afc492dacb786e67051d9d6900afa671f742a0222149f9cd9a7677e37956b`,
the208ff93 Git blob. This is a verified line-ending difference, not a code
change; both byte identities remain recorded. The archived scorer's dirty flag
is null, not an assertion of a clean working tree.

## Three-route outcome and cost comparison

All routes returned final status `abstained`, but that status is not the same as
a model choosing abstention. The controller separately recorded the causes.

| Validation route,32 tasks each | Accepted answers | Schema failures | Model abstentions | Quote rejection | Input/output generation tokens | P50/P95 seconds |
|---|---:|---:|---:|---:|---:|---:|
| Fixed retrieval |0|30|2|0|51,485 /5,002|3.420 /8.821|
| Fixed rerank |0|28|4|0|50,679 /5,172|3.996 /10.529|
| Adaptive |0|28|3|1|51,690 /5,157|3.360 /10.688|

| Authored route,8 tasks each | Accepted answers | Schema failures | Model abstentions | Input/output generation tokens | P50/P95 seconds |
|---|---:|---:|---:|---:|---:|
| Fixed retrieval |0|7|1|12,593 /1,884|5.708 /10.739|
| Fixed rerank |0|7|1|12,945 /2,355|9.423 /13.030|
| Adaptive |0|5|3|12,634 /1,791|5.263 /10.096|

There were120 controller-initialized retrievals,40 fixed rerank calls covering
800 pairs and120 generation calls. **Zero** adaptive extra tools executed;
fixed reranking is controller work, not a model-selected action. Parsed model
actions were14 abstentions and1 answer, with the answer rejected. All120
generations reached EOS; none reached the512-token cap. No observed budget
limit was exceeded. This does not establish timeout recovery or three-turn
behavior, because those paths were not exercised.

## Frozen evidence scores and uncertainty

These are the original scorer's **failure-gated delivered-evidence** metrics,
not initial retrieval scores or answer entailment. On stage failure/deadline,
`delivered_evidence_ids` is empty. Otherwise it retains the20 final candidates,
even after a model abstention or quote rejection. Top-5/Top-10 cutoffs operate
on that list. The frozen scores are not recomputed or replaced by diagnostics.

| Validation route,24 evidence claims | Recall@5 | MRR@10 | nDCG@10 | Evidence F1 |
|---|---:|---:|---:|---:|
| Fixed retrieval |0|0|0|0|
| Fixed rerank |0.069444|0.055556|0.056332|0.034722|
| Adaptive |0.041667|0.041667|0.041667|0.031250|

Frozen paired bootstrap:5,000 samples, seed20260929,24 pairs. Against fixed
retrieval, fixed-rerank95% delta intervals are R@5 `[0,0.180556]`, MRR@10
`[0,0.152778]`, nDCG@10 `[0,0.148162]`, F1 `[0,0.090278]`; reported two-sided
p=.25. Adaptive intervals are `[0,0.125]` for R/MRR/nDCG and `[0,0.093750]`
for F1, p=.708. **Every interval includes zero: no stable improvement or winner.**
Label delivery is0/23 for each route because no answer was accepted. It is not
the accuracy of the earlier unconfigured classifier, nor intrinsic standalone
verdict capability. Semantic supportability remains unmeasured; citation
precision is not100% merely because no answers were emitted.

Posthoc diagnostic only: pre-delivery retained context5 gold recall was0.431944
for fixed retrieval/adaptive and0.440278 for fixed rerank. Each route had16/24
queries with gold in candidate20. Context contained gold but delivery was cleared
on13/11/12 queries respectively. These descriptive snapshots explain why frozen
zeros do not mean BM25 never retrieved evidence. They have no new bootstrap,
do not replace frozen scores and do not prove an adaptive retrieval benefit.

## Failure diagnosis and total resource cost

105 `GeneratedResponseError/schema_validation` events contain103 root
`value_error` and2 overlong reasons. Private decoded-output retention was capped
at32 files per phase,32KiB per file,1MiB total. Validation retained32 files;
authored-vNext retained24. Hash-matched retained bytes cover47/105 failure rows,
not necessarily47 unique files. Structural JSON inspection found45 non-rewrite
actions carrying a non-null query,2 reasons over500 characters and1 non-answer
with answer fields. Flags overlap. The remaining58 failure payloads are not
available; their precise cross-field causes cannot be inferred. This inspection
is **not** a replay through the frozen Pydantic validator.

All192,026 input and21,361 output generation tokens count; unknown usage rows0.
Schema failures alone consumed166,587 input and19,390 output tokens (90.77% of
all output tokens). Quote rejection consumed1,918/439; model abstentions
23,521/1,532. No failed generation is treated as free.

Serial request time sums to589.321s: recorded generation565.217s, rerank22.556s,
retrieval0.612s. Operator elapsed696.435s also includes extraction, installation,
hashing, loading, scoring and other overhead; the107.114s difference is not a
pure model-loading measurement. Routes used fixed order without uniform explicit
warmup. Request P50/P95 excludes loading, is descriptive, and is not online SLA
or batch QPS. Token totals exclude reranker compute; no monetary ROI is inferred.

Slurm r2:701s elapsed,678.950 actual batch CPU-seconds,18,176,444KiB host MaxRSS
(17.33GiB),1 A100/8CPU/32GiB allocation. GPU peak memory is unrecorded, not equal
to host MaxRSS. CPUTimeRAW5,608 is allocated CPU-time, not actual CPU use.
The preserved failed r1 preparation used39s and34.931 batch CPU-seconds with no
inference. Combined r1+r2 allocation is740 GPU-seconds (0.20556 GPU-hours),
not GPU-kernel utilization or an API-price estimate. Both attempts are retained.

## Five real, sanitized cases

Cases are chosen by fixed outcome predicates and deterministic private-ID order,
not answer quality. The compact exposes no claim/task IDs or their enumerable
hashes, evidence IDs, model text or source prose. Private replay resolves identity.

1. **Parsed answer rejected:** adaptive produced a schema-valid answer but an
   inexact quote, rejected after11.487s and1,918/439 tokens. The same task failed
   schema on both fixed routes. A parseable answer is not a validated citation.
2. **Reranking did not fix the contract:** adaptive and fixed retrieval genuinely
   abstained (4.281s/4.463s), but fixed rerank failed schema after13.551s and495
   output tokens. Abstention correctness and entailment remain unmeasured.
3. **Evidence existed before output failure:** all routes retained20 candidates
   and5 context items, then schema failure cleared delivery. Latencies were
   2.726/3.064/2.721s. A zero delivered score must not erase retrieval execution.
4. **Authored abstention across routes:** all three legally abstained, at
   3.089/3.610/2.644s. There is no official gold, so these are not verified correct
   decisions or evidence of multi-step planning.
5. **Authored failure costs remain payable:** all routes failed schema at
   8.154/10.696/7.611s and290/386/290 output tokens. Fixed rerank additionally
   processed20 pairs. Neither failed generation nor fixed tools are free.

## Personal extension, replay and interview translation

The underlying COMP90042 Group045 course project remains team provenance.
This portfolio package adds the bounded source/constraint controller and tool
contracts (`src/climate_rag/budget_agent.py`), local provider, frozen protocol/
comparison runner, separate scorer, HPC operator and content-free audit. It
does not claim authorship of Qwen, CLIMATE-FEVER, official annotations or every
team notebook. Module details and earlier handoff are linked from
[the CPU package](BUDGET_AGENT_CPU_HANDOFF_20260929.md).

Audit code `af97ab2f81204ec047fc020378857928b387f796`, script SHA
`0de5c3f06bccc66b68576036b48f73374ccc244fd9a62161e1e8d6c774f73def`.
The actual small JSON/source audit took0.22s and24,576KiB RSS on Python3.9.25;
no dataset/model computation or new Slurm job. Original decoded strings and row
predictions stay on Spartan. Operator revalidates phase matrices/hashes; original
`frozen_score` objects are preserved, with diagnostics under `supplemental`.

Authorized operators can replay without GPU or inference using a **fresh** output
filename inside the existing release (existing output is never overwritten):

```bash
CLIMATE_ROOT=/data/gpfs/projects/punim2936/portfolio_20260903/climate-public-retrieval-v2
python3 scripts/audit_budget_agent_full.py \
  --root "$CLIMATE_ROOT" \
  --operator-archive "$CLIMATE_ROOT/envs/budget-agent-full-operator-947645b.tar" \
  --output "$CLIMATE_ROOT/runs/climate-full-72eaa90-20260930-r2/full-compact-replay.json"
```

Clean `git archive` reproduction at `af97ab2` passed10 targeted tests in0.08s
and Ruff on the audit/test modules (Python3.12.14). The unchanged full local
suite was not repeated; normal publication CI checks all tests, source mypy,
Ruff and tracked-text privacy. These checks validate software, not model quality.

Public readers can test the audit contracts with
`python -m pytest -q tests/test_budget_agent_full_audit.py`; full private data is
deliberately not redistributed. The publication validation receipt is recorded
separately from runtime/model quality.

**Problem–method–result–tradeoff:** To find whether model-selected search actions
improved grounded delivery beyond a fixed chain, I implemented a bounded
source-bearing controller, three-route frozen comparison and failure-inclusive
accounting. The120-slot real-model replay completed but emitted no accepted
answers and no adaptive tool calls;105 failures were structural. I retained the
negative result and avoided deployment/quality claims rather than treating
successful scheduling or mechanical rejection as semantic success.

Resume candidates for the coordinator, not direct edits:

- Preferred search evidence (earlier separate experiment): “负责气候证据表征适配：
  构建候选支持的 hard negatives 与 LoRA/InfoNCE 训练链，在120.9万文档、154条
  offline dev 声明上将 Recall@5 从0.2793提升至0.2970，并用5,000次配对 bootstrap
  核验；明确区分开发集适配与独立测试。” See [EVIDENCE](EVIDENCE.md).
- Optional engineering/interview evidence only: “实现带来源约束、预算和引用校验的
  检索决策原型，完成固定检索/固定精排/自适应三路线120次真实模型回放，按失败类型
  追踪 token 与延迟；发现结构化输出不稳定且未观察到主动工具执行，保留负结论。”

Do not replace a strong search bullet with raw failure counts merely to mention
Agent. This package does not support “task success uplift”, “self-correcting
multi-step Agent”, “production cost saving”, “independent-test gain” or “100%
grounded correctness”. A further contract/model change would need a new explicit
bounded plan and separate evidence; it is not authorized by this closeout.
