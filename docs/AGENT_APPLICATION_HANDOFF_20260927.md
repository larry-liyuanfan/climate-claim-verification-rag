# Climate：Agent 证据供给接口交接（2026-09-27）

## 所有权、版本与本轮范围

- 独占工作区：`E:/Project/_codex_worktrees/climate-representation-eval`。
- 分支：`codex/climate-search-tradeoffs-20260927`；增量基线 `b81cfde2d3d148e5797e1b63ba14606cd311dabe`。
- 已有检索/训练证据保持不变；本轮仅增加证据包、真实 LangChain 适配和应用演示。
- 最终交付 SHA 由后续记录提交填写；不修改主简历、求职共享文件、其他项目。
- 这是个人作品集扩展，课程团队原始系统不是全部个人完成。

## 应用问题与调用链

下游研究/问答 Agent 需要先拿到可定位的气候证据，而不是把搜索分数当成事实结论。

```text
Pydantic EvidenceQuery（claim_text；拒绝额外参数）
→ LCEL RunnableLambda 参数规范化
→ BaseRetriever / HybridRetriever
→ LangChain Document（citation ID + 原始 evidence ID + source metadata）
→ corpus/text SHA 与原文一致性检查
→ source-bearing evidence packet / no_evidence
```

真实组件位于 `src/climate_rag/langchain_evidence.py`：
`ClimateEvidenceRetriever`、`create_evidence_chain`、`create_evidence_tool`。
工具名 `retrieve_climate_evidence` 可交给下游工具注册/选择层；本包通过实际
`tool.invoke` 和 `chain.invoke` 验证，没有声称接入已评测 LLM planner。
底层 HybridRetriever 接受 BM25/dense/reranker，但演示只启用 BM25；没有新验证
百万规模 HNSW/LTR/4B 在线链。LCEL 当前一次查询，不把旧 HTTP 服务的两次检索逻辑
冒称本适配器能力。参数规范化/实体年份抽取是 regex，不是训练的 query-understanding 模型。

每条候选包含原始 `evidence_id`、全文 codepoint span、文本 SHA、语料 SHA、article、
publisher、检索路线/分数。检索的 `source=bm25` 不是出版来源。当前公共语料没有逐条 URL，
因此不编造链接：以 `evidence.jsonl` 的 ID 和哈希定位原句、article 定位原始条目。
引用 ID 可供后续答案引用，但来源校验并不证明引用蕴含结论。

输出 `answer=null`、`verdict.status=not_evaluated`。空结果要求调用方弃答或补充证据。
本包另有旧 FastAPI `/api/search`→packet、`/api/verify`→未配置弃答的 fixture 演示；
`verifier_not_configured` 不能作为分类错误率、准确率或真实 RAG 效果。

## 真实语料演示（非测试集评测）

读取已存在的公开 evidence，不读取任何 test claim 或标签，也没有调参。

- 原始发布者数据入口：<https://github.com/tdiggelm/climate-fever-dataset>。
- 本地 evidence：`E:/Project/climate-claim-verification-rag/data/climate-fever-20260825/evidence.jsonl`。
- 5,240 docs，1,682,924 bytes；SHA `c14315aee9feecbbbbc3b0c7101d978b52b47cee9a613336304bcaa736460c71`。
- 源缓存 SHA `8a4b9032d861be482ffb49dddfd283ffa6089e654f1e968040011882c5eb6e0b`；
  本轮不打开混有 frozen-test claims 的源缓存。
- 三个手工撰写查询，不是官方 benchmark 任务或自然语言准确率样本：

| 查询 | 实际行为 | 能证明/不能证明 |
|---|---|---|
| Arctic sea ice decline | Top-1 `Arctic sea ice decline:271`，3 个带来源候选 | 真实语料检索与 ID 保真；不证明 verdict |
| carbon dioxide greenhouse effect | Top-1 `Greenhouse effect:54`，3 个带来源候选 | 真实 StructuredTool 调用；不证明这些候选全相关 |
| zxqv_nonexistent_term_20260927 | 0 条、`no_evidence`、无答案 | OOV 空结果路径；不等价语义不可回答评测 |

完整本地演示输出：`artifacts/langchain-public-demo-20260927.json`（ignored）。
公开仅保存紧凑摘要 `docs/verified-runs/langchain-application-20260927.json`，不再分发语料。

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-agent-demo.lock
.venv/Scripts/python -m pip install --no-deps -e .
.venv/Scripts/python scripts/demo_langchain_evidence.py --evidence E:/Project/climate-claim-verification-rag/data/climate-fever-20260825/evidence.jsonl --expected-sha256 c14315aee9feecbbbbc3b0c7101d978b52b47cee9a613336304bcaa736460c71 --output artifacts/langchain-public-demo-20260927.json
.venv/Scripts/python scripts/demo_agent_evidence.py --output artifacts/fixture-api-demo.json
.venv/Scripts/python -m pytest -q tests/test_agent_evidence_packet.py tests/test_langchain_evidence.py
```

若已有语料路径不同，只替换 `--evidence`。脚本严格核对 SHA；不自动访问 benchmark test。
独立 Python 3.12 venv，直接及传递依赖锁定；LangSmith tracing 在真实演示中关闭，
无密钥、LLM/API 请求、GPU、Spartan 重训、全量本地测试或生产部署。

## 框架盘点与验收

| 状态 | 内容 |
|---|---|
| 既有 | 自研 BM25/ANN/RRF/LTR/reranker；FastAPI/Pydantic search/verify/trace；提供 Model Studio adapter，不代表当前已配置 |
| 本轮新增并实际执行 | langchain-core 1.6.5 BaseRetriever/Document、LCEL、StructuredTool；版本化候选证据包；真实公共语料三例 |
| 尚未证实 | LLM 工具选择正确率、回答质量、独立新 test 增益、线上 SLA/部署与采纳 |

定向测试覆盖原文/ID 篡改、重复 ID、非法 URL/摘要、空结果、schema 额外字段、
空白/超长/非字符串输入、候选宽度、真实 LCEL/工具返回一致性和 fixture API trace。
本轮不重跑未修改训练/索引的完整套件。最终验证数字与代码 SHA 见末尾版本记录。

官方接口依据（已查阅，不把主分支示例当锁定版本）：
[LangChain retrievers](https://docs.langchain.com/oss/python/integrations/retrievers)、
[Tools](https://docs.langchain.com/oss/python/langchain/tools)、
[Core composition](https://reference.langchain.com/python/langchain-core/tools/convert)。

## 旧质量结果与本轮应用演示分开

`docs/SEARCH_TRADEOFFS_20260927.md` 保留公开 validation 同候选三路线结果：
LTR Recall@5 .6054 / Evidence F1@5 .3828 / P95 **77.8 ms**；Top-100 4B 为
.6275 / .3969 / **9.30 s**。这是预热串行请求计时，不是 HTTP SLA；5,000 次配对区间
跨零，不能称无损提速或显著质量优势。此次 BM25/LangChain 演示没有继承这些延迟。

另一个 restricted 1,208,827-doc/154-query、20-step InfoNCE-LoRA offline-dev 轨的
Recall@5 .279→.297 仍保留，不能与 public demo/test 混合，不能称大规模预训练或 A/B。
已消费公开 frozen test 继续封存。

## 两条候选简历 bullet（交给简历 chat，不直接写简历）

- **证据检索工具：**将多阶段检索接入 LangChain Retriever/LCEL/结构化工具，保留原始证据 ID、语料/文本哈希和引用 span；在 **5,240 条公开证据**上完成真实查询与无结果演示，向下游 Agent 返回可追溯候选而非无依据 verdict。
- **质量—延迟选型：**统一候选、特征与评测口径，公开 validation 的 LTR 达到 **Recall@5 .605、P95 77.8 ms**；相对 4B Top-100 的 9.30 s 延迟保留低延迟路径，并用配对区间说明质量差异尚不确定，不宣称线上无损提速。

## 90 秒应用故事

“研究问答要判断气候声明，第一步不是让模型直接下结论，而是给它可核对的证据。
我把原来的检索算法接成 LangChain 标准 Retriever 和结构化工具：输入是受约束的声明，
输出是原始 ID、文本 span、语料版本和候选分数，查询没有证据就明确返回空结果。
实际用现有公开语料验证了海冰、温室效应两类查询，并测试未知词、额外参数和文本篡改。
这个接口解决的是 Agent 如何安全消费搜索能力，没把规则抽取包装成 LLM，也没把引用
合法当成事实成立。算法侧，我另外在固定 validation 候选上量了完整请求：LTR 的 P95
77.8 毫秒，4B Top-100 约9.30秒，后者的 Top-5 点估计更高但配对区间跨零。因此保留
两档取舍，而不是认为模型越大越好。真实答案模型和线上服务效果还需要独立验收。”
