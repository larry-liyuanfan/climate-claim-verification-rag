# Climate / FLARE 项目证据交接（2026-09-27）

本文件是项目负责 chat 给求职材料单写者的内容交接，不是当前简历更新。
没有修改 Trip、Energy、课程 FLARE 或求职共享目录；不要求重装 SSH 或登记重复项目。

## Climate：本轮 P0 / P1 已完成

- 仓库：`larry-liyuanfan/climate-claim-verification-rag`。
- 独立分支：`codex/climate-search-tradeoffs-20260927`；base
  `272ab93256202e517a8aebf36992fbdf93dc922e`。草稿 [PR #18](https://github.com/larry-liyuanfan/climate-claim-verification-rag/pull/18)，未合并、未部署。
- 推理 SHA：`b797f6606ced013b1a0adee915c996b593f180a3`；补充直接配对脚本
  SHA：`97b2dc8eb5748cbf25ae1a4fccbec8c405b72342`。后续文档不冒充计算版本。
- 个人贡献：hard-negative / InfoNCE-LoRA 实验链、候选与特征合同、模型加载修复、
  完整排名指标、分阶段及完整请求计时、质量—延迟决策。原始课程系统是团队输入。
- P0：同一公开语料、查询与候选的 LTR / Top-20 4B / Top-100 4B 实测完成。
- P1：选择“表征修复”路径，实际 392 个 checkpoint tensors 完整恢复，开关 adapter
  后 query/document 向量变化可复现；未重训、未宣称新质量提升，未打开 SciFact 或旧 test。
- 验证：干净检出非 editable 安装后 106 passed、1 个本地 Torch 缺失 skip；Ruff、
  31 模块严格 mypy 通过；实际 Spartan 环境另外 7 个 PEFT 集成测试通过。

### 同口径结果

公开 CLIMATE-FEVER，5,240 docs / 126 条 decisive validation queries；其余 104 条
非决定性证据声明未进入本轮，不能推断不可回答或分类效果。所有路线使用 base encoder，
不与 restricted 20-step LoRA 的结果串接。

| 路线 | Recall@5 | Evidence F1@5 | 预热串行 E2E P95 | 峰值 Torch allocated |
|---|---:|---:|---:|---:|
| LTR | .6054 | .3828 | 77.8 ms | 2.25 GiB |
| 4B Top-20 | .5948 | .3829 | 1.91 s | 10.54 GiB |
| 4B Top-100 | .6275 | .3969 | 9.30 s | 11.07 GiB |

建议低延迟场景选 LTR，Top-100 保留为可选慢速质量档，Top-20 不作为默认折中方案。
三组 Recall@5 / F1@5 的 5,000 次 paired-bootstrap 区间均跨零；不能写显著胜出、
等效或无损提速。三者点估计都在 Pareto frontier，不等于三者均有实际采用价值。
LTR 只有排序在 CPU，query encoder 仍在 GPU。此表不是 HTTP SLA、线上 A/B 或成本节省。

作业 `31364586` 退出 0:0，22分14秒；0.3706 MIG-slice hours，不是全 A100 GPU-hours。
源归档 SHA `d3c9f3e4071eb1c1448718073f22e57fef32e0a8a68a92220ba1bc3eb3304ed1`。
数据、索引、模型、预测、trace、完整统计与资源证据见：

- [三路线决策报告](SEARCH_TRADEOFFS_20260927.md)
- [compact artifact](verified-runs/search-tradeoffs-20260927.json)
- [加载完整性证据](verified-runs/adapter-integrity-20260927.json)
- [软件复现](verified-runs/search-tradeoffs-reproduction-20260927.json)

### 必须更正的旧口径

1. “public adapter 无有效提升”改为“历史加载完整性不足，pilot ties 无法判断效果；
   当前已验证修复，但未新增质量评测”。
2. 历史 public-v2 / restricted five-stage 只保存前五：旧 MRR@10 实为 MRR@5，
   nDCG@10 受截断且不能改名 nDCG@5。R@5 / F1@5 不受影响；新的本轮 Top-10 指标完整。
3. 单独 restricted 20-step full gate 保存了 Top-50，其 MRR@10/nDCG@10 仍有效，
   F1 是 F1@50，不与下游 F1@5 比大小。
4. 旧阶段 P95 相加不是请求 P95。本轮表格为完整请求独立计时；不要拼接百万向量 QPS。

### 问题—方法—结果—取舍

面向气候证据的语义召回与精排，我先用 claim 分组和假负例过滤构造 hard negatives，
完成 20-step LoRA/InfoNCE 任务适配；在 120.9 万文档的同一离线开发集，Recall@5
由 .279 提升至 .297。随后在独立的公开 validation 证据轨比较 LTR 和 4B 候选宽度，
修复候选不可达、checkpoint 恢复和排名截断问题，并测量完整请求。LTR 以 77.8 ms P95
取得 .605 Recall@5；Top-100 4B 点估计 .628，但耗时 9.30 s，Top-5 增益不确定。
因此推荐按延迟需求选择模型，而非堆参数；保留负结果并封存已消费 test。

### 两条搜索岗位候选 bullet（仅供简历 chat 取用）

- **表征适配：**面向120.9万条气候证据，设计 claim 分组 hard-negative mining 与
  20-step LoRA/InfoNCE；同一离线开发集 Recall@5 从 **.279 提升至 .297**，四项检索
  指标的5,000次配对区间均为正，保留独立测试边界。
- **排序选型：**构建 BM25/HNSW—RRF—LTR/4B 精排链，统一训练/服务候选与完整请求评测；
  公开 validation 低延迟档达到 **Recall@5 .605、P95 77.8 ms**，基于配对区间与显存
  取舍保留4B为可选慢速档，不盲目扩大模型。

未完成也不应写入：修复后 public adapter 质量提升、独立迁移泛化、线上 SLA、真实
verdict 效果、生产默认配置变更。本轮补强计划明确允许“表征修复/独立迁移二选一”，
因此没有为填正数再开训练或测试。下一步优先完成 PR 审核与材料接收，不新增 GPU 实验。

## FLARE：作品集证据已收口，行业交付由课程 chat 负责

只读核验 `wildfire-burn-window-decision-support` 当前 clean main 为
`d7d5962a42f9cf3ddccf97d482368ffe2f756b43`，不是本轮新增51年计算。
43类规则 → typed AST → 时空合同 → area-weighted aggregation → 7 typed tools
→ provenance 的作品集证据完整：176 IDs × 51 年 = 8,976 ID-year records，351个
非零权重、230个网格、零 nearest fallback；2020 direct/CSR 最大差0。
前6工具有30-call fixture基准与6/6故障恢复；第7工具只有 compact artifact 查询验证，
不能把前6的性能表扩展到全部7个工具。

来源为该仓库 `docs/evidence-closeout.md`、`docs/burn-unit-climatology.md` 与
`artifacts/public/vicclim6_murray_goldfields_burn_id_climatology_20260903.json`。
角色保持 **Data Science Industry Project / Vocational Placement**，不称 Research
Assistant、自主 Agent、安全批准、真实 ROI。天气代理与现场 FMC/风测量、计划 polygon
与 burn outcome 分层；课程十年研究、团队采纳需课程 chat 的单独交付证据。

精简候选句：将43类计划燃烧规则编译为 typed AST，以面积加权稀疏聚合形成176个
burn ID的51年可审计气候记录，并通过7个确定性工具提供带版本与失败状态的查询。
搜索岗位中只承担“复杂数据、typed tools、可信工程”辅助作用。

## 全局完成审计边界

Climate P0/P1、FLARE作品集证据已完成；这不等于整个四项目计划全部完成。
Trip、Energy及课程FLARE由各自chat执行，不能凭本项目CI通过替其验收。
本轮已看到其他chat继续推进的新状态，但未把未核验的新结果写入简历或修改其代码。
