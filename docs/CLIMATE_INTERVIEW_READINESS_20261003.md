# Climate：已完成、未完成与本人练习清单

本轮起点：`74b1cdb`，同一 managed worktree / `codex/climate-bounded-gap-repair-20260930`。
只补秋招展示与讲解；不训练、不运行模型、不读取 gold/test、不修改主简历。
业务问题是：**帮助查证人员找到相关、充分、可追溯的证据**。
检索命中只是提供候选，当前演示不自动认定证据充分，也不生成事实判定。

## 1. 五点状态

| 要求 | 我已完成的部分 | 尚未完成或需你验收 |
|---|---|---|
| 明确问题 | 查证任务、个人模块、交付物与事实边界已写清 | 你用自己的话在 30 秒内解释，而非背模型名 |
| 真实训练案例 | 负样本、数据分组、损失、LoRA、指标与边界的源码讲解见下文 | 你独立写损失、回答追问；未做单因素因果消融 |
| 可重复真实路径 | `--claim` 输入任意声明，真实公开 BM25 → Top 候选 → 分数排序 → 证据/来源/哈希 → 当次耗时 | 不是实时 dense/LTR/4B/verifier；相关不等于支持判断 |
| 错误分析 | 固定 12 题 × 3 路保存输出，原诊断不改分；六类讲解与缺失类别见下文 | 不是人工盲标；没有确证极性翻转或获取后成功的案例 |
| 独立防守核心代码 | 白板提纲、源码入口、现场改代码任务与验收标准已提供 | 必须由你本人完成，不能由测试数或 AI 写好文档替代 |

上一轮“完成”指展示/案例包，不表示本人掌握、全链资产恢复或自主 Agent 收益已经验收。

## 2. 三分钟实际演示

在项目 worktree 运行（整条命令；不需要云）：

```powershell
.venv-validation/Scripts/python.exe -X utf8 scripts/demo_recruitment_case.py --evidence E:/Project/_climate_transfer/fair-three-arm-20261003/cpu-input/evidence.jsonl --claim "Atmospheric carbon dioxide absorbs infrared radiation." --candidate-k 10 --top-k 3
```

换掉 `--claim` 的内容就是新查询；允许最多 2,000 字符。
使用其他语言不保证能匹配：当前是英文公开语料与现有 tokenizer，不做自动翻译。
`1 <= top_k <= candidate_k <= 100`；未提供真实语料时，新声明/新排序设置直接拒绝，
不会拿历史回放伪装成新搜索。默认无参数仍保留脱敏汇总回放。

演示顺序：

1. 先指出顶端训练/选型表是历史结果，并说明数据分母不同。
2. 当前配置是 BM25 `k1=1.5, b=0.75`、候选 10、证据 3；dense/LTR/rerank/verdict 均未启用。
3. 查看候选 ID、rank、score；按 BM25 分数降序，同分按 ID。
   选取 Top3 只是截断同一排名，不是第二阶段精排。
4. 阅读返回证据、Wikipedia article/source 与文本 SHA，检查是否真的覆盖声明关系。
   来源 metadata 来自语料，未伪造抓取时间、历史快照 URL 或审定标签。
5. 区分初始化（文件 hash、载入、建索引等）与请求（搜索/排序、证据打包）的当次耗时。
   请求耗时不含输出序列化、CLI 开销；单次计时不是 P50/P95 或线上 SLA。
6. 换成 OOV 查询观察空候选/空证据、`answer=null`。这是手工接口演示，不是模型恢复案例。

本轮真实 CPU 演示回执（公开语料，仅手工查询；不做 benchmark 评分）：

- 语料：5,240 documents，SHA `c14315aee9feecbbbbc3b0c7101d978b52b47cee9a613336304bcaa736460c71`。
- 本地完整输出：`E:/Project/_climate_transfer/fair-three-arm-20261003/recruitment-closeout/readiness-live-claim.json`。
- 空结果输出：同目录 `readiness-live-empty.json`。
- 输出保留在本地；Git 不收集完整证据或用户自行输入的私人声明。
- 实际执行脚本 SHA：`e6aea1494446a2883e4e6389375e0d17296e001eab5d5d213fd728d2e7247240`。
- 有结果输出 SHA：`17d1deab816786d95fd963363a54f91f12f9ce01990f78a86be818cfca46a52f`；
  空结果输出 SHA：`9eba81f863c5ae8ed567b7d4b62f55929add8e748283220ed286806ad7c03ed3`。
- 此次有结果的初始化147.14ms，请求0.914ms（搜索/排序0.882ms、打包0.032ms）；
  空结果初始化143.71ms，请求0.031ms。只是两个本地单次回执，不与历史 LTR 时延相减认领收益。

## 3. 训练白板：要讲清什么

### 难负样本与防泄漏

源码：`src/climate_rag/negatives.py:mine_hard_negatives` 和
`src/climate_rag/embedding_training.py:build_swift_infonce_dataset`。

从 BM25/dense 的高排名、非已知 gold 候选挖负例，保留 route/rank；不是随机挑无关句。
数据构建按 claim ID 分桶，避免同一 claim 的多正例拆到 train/eval。
每条训练样本是一个 query、一个正例、默认四个显式负例；删除已知 gold ID、
重复 ID 与当前正例/已选负例的规范化重复文本。正例或负例不够就跳过，不补造。

**不能扩大宣称：**未标注真证据仍可能被当成负例；ID 分组不自动解决所有同义 claim、
共享证据和语义文档变体。公开轨连通分组另有合同，不与受限训练分组混为一谈。

本轮修复一个 rank 边界：非连续显式 rank 不应被列表位置压低，非正 rank 才回退位置。
只修当前函数/回归测试，没有重新挖样本、训练或评分；历史提升不能归因于本次修复。

### InfoNCE：公式与梯度直觉

白板写单正例示意：

\[
L=-\log\frac{\exp(s(q,p)/\tau)}{\exp(s(q,p)/\tau)+\sum_j\exp(s(q,n_j)/\tau)}.
\]

对 softmax logit，正例梯度是 `P_positive - 1`，负例是 `P_negative`：相似度高的负例
占较大竞争份额，因此比容易负例更有训练信号；false negative 也可能带来错误梯度。
温度改变分布集中程度与梯度尺度，不是越小越好。
这是[InfoNCE 的教学解释](https://arxiv.org/abs/1807.03748)，不是声称本项目自行实现了整个 trainer。

**手算练习（合成数字，不是实验）：**logits `[2,1,0]`，第一个为正例，softmax 正例概率约
0.665、loss 约 0.408；解释为什么把第二个负例分数提高会让 loss 增大。
实际训练委托 ms-swift `--loss_type infonce`；历史脚本没有显式设置温度。
未恢复的 resolved trainer arguments 不靠猜测补齐，也不把其他 pilot 的温度当作该运行实参。

### LoRA：改了什么

\[
W'=W+(\alpha/r)BA,\quad A\in\mathbb{R}^{r\times d_{in}},\ B\in\mathbb{R}^{d_{out}\times r}.
\]

冻结基座、训练低秩更新，而不是重新预训练基座；该机制见[LoRA 原论文](https://arxiv.org/abs/2106.09685)。
历史执行 SHA `c815070...` 中的 `hpc/train_embedding_lora_pilot.sbatch` 配方为 rank8、alpha32、
all-linear、BF16、默认20 steps；已保存结果确认20 steps 与5,046,272注入参数。
源码默认配方不等于恢复了全部 resolved runtime arguments。

**结果与解释边界：**1,208,827 documents / 154 offline-dev queries，Recall@5
0.279329 → 0.296970（+1.764个百分点）；5,000 次 paired bootstrap 的差值95%区间
`[0.001404, 0.034957]`。这说明该组合在此开发评测上改善，不证明收益单独来自 hard negatives、
LoRA、温度或任一超参：没有单因素消融，不能编造因果解释。
是20步任务适配，非大规模预训练、独立测试或线上 A/B。
原汇总见 `docs/verified-runs/qwen3-embedding-lora-full-gate-20260821.json`。

## 4. 搜索白板：四个取舍

- **Flat 与 HNSW：**Flat 遍历向量作精确参照；HNSW 用分层近邻图近似搜索，
  查询搜索宽度与建图参数影响质量、耗时和内存，须实际测量。
  ANN-vs-Flat 的向量邻居 recall 不等于语义 evidence Recall@5。
  来源：[HNSW 原论文](https://arxiv.org/abs/1603.09320)；代码 `src/climate_rag/dense.py:FaissANNIndex`。
- **RRF：**`sum_route 1/(60+rank)`，各路去重、缺路不加分、ID 破同分；不直接混合不同尺度的原始分数。
  手算 A 排名 `(1,10)` 得0.030679，B `(2,2)` 得0.032258，B 可胜过单路第一。
  来源：[RRF 原论文](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)；代码 `fusion.py:reciprocal_rank_fusion`。
- **LTR：**11个固定特征：三路 score/reciprocal rank，加 token/number/year overlap、query/document length。
  LambdaMART 按 query 分组学习排序；缺路 reciprocal rank 为0。排序不能找回候选池之外的 gold，
  训练/推理宽度、特征顺序与输入含义必须一致。代码 `fusion.py:build_candidate_features/LightGBMLambdaMART`。
- **Cross-encoder：**联合处理 query-document，对候选逐对精排，不是预先存好所有 query 的向量。
  同一公开 validation 的 LTR：R@5 60.54%、离线 P95 77.8ms；4B Top100：62.75%、9.30s，
  Top5质量差值区间跨零。因此选低时延 LTR 候选，不能说统计等效或生产默认。
  历史路线包含 GPU query encoder，不是本次 CPU BM25；原报告 `search-tradeoffs-20260927.json`。

## 5. 六类错误怎么讲，不造新频次

复用固定顺序12题×3路、36条作者/模型辅助诊断：不是人工盲标、新正确率或独立测试。
原27条完整引用支持不足、4条不确定、4条局部一致弃答、1条不确定弃答保持不变。
下面是对已存笔记的讲解映射，允许多标签；没有新六类互斥标注或错误率。

| 类别 | 已保存的匿名观察 | 不可宣称 |
|---|---|---|
| 实体/范围 | 一般法规不覆盖指定污染物；其他地域不能直接证明目标地域 | 不能把所有 relation_scope 都计为实体错误 |
| 年份 | 不同年份人数不支持同年比较；后来的纪录不必反证原发表时点 | 不确定相对日期不硬判错误 |
| 数字 | 缺同年同口径分母/另一方数值，无法形成比率 | 有一个数字不等于比较充分 |
| 极性 | **没有确证肯定/否定翻转案例** | 引用不足以支持 REFUTES 不等于极性翻转已证实 |
| 缺证据/关系 | 总体升温材料未证明两套序列的偏离；补主题文本未覆盖决定性关系 | 缺当前证据不代表世界中无答案 |
| 提前停止 | 有“比较关系缺失、停止后仍确定作答”的保存观察，作为获取门诊断线索 | 没有反事实证明继续获取一定修复；32 stop 不全是错误 |

还要保留关系方向、因果、量词、上下文不确定与合理弃答，不能强塞进六类错误。
没有自主有效补证据、工具空结果恢复成功或真实极性翻转故事，就明确说没有；
OOV/合成练习只能讲接口或实现，不能补成实测故事。

## 6. 需要你本人做的事（尚未验收）

建议累计3–5小时；以表现为准，不以读完文档或耗时达标为准：

1. **90分钟白板：**不看稿写 InfoNCE/LoRA/RRF，手算上面两个例子；解释负例污染、
   grouping、ANN质量与语义召回区别，以及为什么没有选择最深4B精排。
2. **45分钟源码走读：**从负样本记录到训练行，再从候选到11特征与排序；指出自己的模块
   和团队输入。任选函数逐行解释，不能只背 README。
3. **45分钟现场修改：**在你自己的练习分支，为 `reciprocal_rank_fusion` 增加可选 source weights。
   默认全1必须完全保持旧输出；显式定义未知route/负值/NaN/全0的拒绝规则；测试路内去重、
   缺路、同分、top_k 和两路手算。仅合成小样例，不跑模型/benchmark。
   这是练习任务，**root没有替你实现，尚不能写成本人已掌握**。
4. **30分钟口述：**录一次两分钟 STAR，再用五分钟回答“提升为什么不是因果证明”
   “数据是否见过”“演示为什么没跑LTR”。选两个真实错误，讲清如何发现和决策。
5. **剩余时间补弱项：**只有能脱稿讲、预测测试结果、独立修改并验证后，第5点才算完成。

练习时从当前工作区另建自己的 `codex/` 分支/worktree，不与本执行chat并发写同一 checkout；
可离线、不需要云。未完成练习前不声称本人独立防守已验收。

## 7. 不用等、但仍未完成的内容

- 实时恢复历史 adapted embedding/LTR/4B 全链：权重和部分原轨迹未找到本地副本，
  远端不可访问不等于丢失。本轮没有恢复，不拿 BM25/fixture 替代其指标。
- 自主获取—反馈—后续决策的真实成功与质量收益：现有32 stop / 0 acquire，仍未证明。
  不作为本次秋招投递前置条件，不承诺强制调用就能改善。
- 全量人工语义盲标、独立 test 新提升、生产 SLA/采用/ROI：没有，不写进简历。
- HTML 浏览器视觉验收：未做；本次可验收交付为实际 CLI，不为浏览器限制新建服务。

**停止线：**本包完成后不追加模型运行或工程规模。你完成脱稿演示、白板、两个真实错误故事
和一次核心模块修改，即进入投递/面试；不是要求所有缺失研究结果都补齐。
