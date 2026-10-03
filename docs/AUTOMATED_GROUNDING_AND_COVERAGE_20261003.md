# 去人工评测后的 Climate 补齐包

本轮只写 Climate managed worktree；不连接 Spartan、不修改其他项目、求职目录或主简历。
人工盲标不再是完成条件。自动模型诊断是**代理评估**，不冒充人工或官方真值。

## 已完成：全量保存输出的自动语义支持评分

业务问题：给查证人员返回的引用，是否支持声明中的具体关系，而非只是主题相关或 ID 合法。
复用原 `fair-three-arm-20261003` 的完整 32 题×3 路、96 槽真实输出；不重新生成答案。
实际原模型源码为 `e826a4d2178983365c87184e259f7ad43ba7b17a`，不改写历史评分。
原 run SHA：`3c9eafaa159321bcc202d5f5f18ddcdd8548917dde7e894f3b19def44e1cb7c3`。

新增本地 NLI sidecar：`cross-encoder/nli-deberta-v3-small`，固定 revision
`fa2804872c3b4bd748f38c0185cc85775361e735`，阈值0.80在测量前冻结。
原声明为 hypothesis、实际显示的完整引用句按原顺序连接为 premise：
SUPPORTS用 entailment 概率，REFUTES用 contradiction 概率，不生成否定声明替换原文。
逐句另算；超长 pair不截断，单独计 unscored。judge不接收 route、gold或获取轨迹。
所有引用须与终答前实际送达的 ID、句序、文本及哈希一致；复用已验收的原 physical/delivery audit。

| 路线 | 分母 | 有答案/弃答/失败 | 引用集通过代理门槛 | 单条引用通过/计分 |
|---|---:|---|---:|---:|
| 强固定多查询 | 32 | 26/6/0 | 3/32 | 7/78 |
| 确定性工作流 | 32 | 27/5/0 | 2/32 | 7/81 |
| 模型自主获取 | 32 | 26/4/2 | 5/32 | 8/78 |

弃答和失败保留全任务分母，代理通过指标记0；不据此判定弃答错误或正确 NEI。
5,000次 paired bootstrap，seed20261003：自主−固定的差值0.0625，95%区间
[-0.0625,0.1875]；自主−确定性0.09375，区间[-0.03125,0.25]。
**两组都不能宣称提升。** 该NLI模型来自通用SNLI/MNLI，气候数字、时间、多证据组合
可能误判，未经本批题校准；结果不是任务正确率，也不是任务＋语义引用联合正确率。
原官方任务指标仍为14/32、14/32、12/32，ID对照不等于语义支持。

实际本地316次NLI forward、23,522输入tokens；评分28.541s，模型加载另10.127s，
付费API0；原Agent成本复用，不重新计作一次模型实验。
私有完整输出：`E:/Project/_climate_transfer/fair-three-arm-20261003/nli-proxy-fa280487-v1/`。
`compact.json` SHA：`9541a6e16ece354d594a2f4f3e32bed38d340710287aeef4fc46b5c5d0c4e8e4`。
模型清单SHA：`acdf21ba8b03afc80194d3ef136fd49b46bc22445f3b16122f2ec14061ddb37e`。
实际评分脚本SHA：`fa32da3a59fb88ad2260133d2906190db635fba00b85a185d0d2c2a66ccfcdd1`。
逐槽概率和原句不进Git、OneDrive或求职材料。

## 已实现：一次有明确反证条件的 coverage 获取门

旧自主路线32次全部stop、0 acquire仍是事实；不是环境故障，也不因合同差异撤销。
新假设：把“关系/时间/数量/范围是否覆盖”显式绑定原claim片段与已显示句ID，
可能改变获取决策。只增加获取门的结构，不强迫工具、不减少两条基线能力、不改终答合同。

`fair-acquisition-coverage-v2-20261003`：gate返回1–3个原claim片段、coverage状态、
已显示句ID、一个既有 stop/read/query/rerank 动作和停止原因。
主题命中不能充当关系覆盖；缺口可以合法stop，self-coverage仍不是语义证明。
固定/确定性路线的规划、共同verifier指令保持v1原字节；三路共享工具、预算和初始信息。
初始packing按包含完整coverage schema的最大提示计算，不能偷减基线可见材料。
提议、参数校验、执行、句子送达、后续物理提示、决策和终答分别留痕。

生产loader固定**已经消费的同32题**：原cohort文件SHA
`f2b7c55b1e91f297a613df5762f44d46c2df7666cca8e234d77d5ca53488ec56`，
canonical tasks SHA `de088bac5fb93beafd2393a438cbc42ac0f99479ff173817a734c01d071e5b25`。
不同哈希定义不混称。无换题、遮答案、标签配额、结果筛选或读取sealed test。
这是单次 regression/机制诊断；**不能成为独立测试提升**。

CPU synthetic集成已经验证：真实worker接三路；不同反馈改变后续合法动作；rerank送达与可引用状态；
stop后无暗中获取；cost保留；仅篡改coverage、协议错配、cohort漂移均拒绝。
这些只证明实现，不证明真实模型会获取、利用反馈或变得更正确。
独立只读review的三个阻断已修复：audit/score协议绑定、云端source合同版本、生产cohort身份。
没有专门反馈隔离消融时，三路策略表现也不能解释成纯反馈因果收益。

另用固定 Qwen3-4B revision 的真实 tokenizer 校验1024/2048/4096/8192四档容量，
不加载生成权重，provider为合成stop；三路各档的初始frame一致。
1024统一安全拒绝；2048仍可能在后续工具反馈时容量不足；4096/8192的合成完整入口
均合法，最大实际提示2647tokens。容量不足未删出分母。这是tokenizer/实现回执，
不是新模型行为或评分。回执：
`E:/Project/_climate_transfer/fair-three-arm-20261003/coverage-token-budget-v2.json`，
SHA `5bad53e3aacfa7779f5c5ab59d44ee5460f3132df6f490cbf2364ca8ea66531a`。

反证：再次all-stop、无关获取、未送达下一物理提示、或无语义支持，均不提升为成功Agent。
不追加同批调参循环来追求正数。协议见 `protocols/coverage-three-arm-20261003.json`。

## 搜索演示与历史资产

历史受限LoRA、公开LTR原件仍未找到；远端不可访问不等于丢失。
历史训练及质量/时延报告继续保留，不用fixture或新模型代替历史产物。
新增公开Qwen基础embedding/HNSW重建已完成：5,240条公开文档、5,240个1024维向量，
4线程CPU编码2173.478s，建图0.354s，HNSW M32/efConstruction200/efSearch64，
索引22,886,274bytes。最长文档515tokens，拒绝静默截断；未读gold或sealed test。
这是新公开基础模型索引，**不是历史受限LoRA索引恢复**。

公开4B checkpoint已从固定官方revision完整下载并校验，共8,059,546,851bytes。
manifest SHA `e1f56457935dd69b67e3249cbd81fd7aabaf5e2f0b870aa173b6fcb573b564ed`
与历史身份一致；恢复的是基础模型，不是旧逐查询推理输出、LTR模型或完整历史链。

本轮手工输入 `Atmospheric carbon dioxide absorbs infrared radiation.`，实际运行
BM25 Top20＋Qwen3 base query encoding＋HNSW Top20＋RRF k60 Top20。
BM25首位为`Carbon dioxide in Earth's atmosphere:4`，dense与RRF首位为
`Carbon dioxide:188`；未用任务标签挑选结果，不生成事实判定。
同次CPU检索：BM25 0.691ms、query forward 422.121ms、HNSW 5.104ms、RRF 0.077ms。
query模型冷加载7186.279ms；encoder子进程总10195.892ms含加载，不能和forward再重复相加。
索引/稀疏初始化115.644ms另列；计时不含hash、文本包装或序列化，不是E2E/SLA/质量比较。

实际新产物（完整证据仅留本地）：

- `E:/Project/_climate_transfer/public-learned-demo-hnsw-20261003/reconstruction.json`
  SHA `cba4cb265f516d27943673069317622fdd41d8e5910ab4d1b96bebbca7588183`。
- 同目录`index.faiss` SHA `c921ceb0b95afa0e743cc9848e8a2c4211394e5c8dcb534066863668ea92d241`。
- 同目录`live-learned-co2.json` SHA `d0ddee2c4a87ce219f69362ceeb6dfe76ccc4391511fb2963a1afbe34a8aa3c6`。
- 编码执行源码快照SHA `45fa7f6942f33760c9f5b6244ed5fe4ba4b5bd028c8f018dd40a61b8fbb5caaf`；
  索引执行源码SHA `0f024f0a3eca5623ee14b5565fbebe40d7c15c252152bded9c31851312cf948a`。
  编码过程中仅修索引的已有文件拒绝及typing注释，没有重算向量或改编码逻辑。

Windows原生Torch与FAISS的OpenMP冲突通过独立进程隔离解决，没有开启duplicate-runtime绕过。
查询编码与检索索引相互核对模型文件、语料、向量维度、文档ID/文本映射，产物拒绝覆盖。

## 完成口径与求职转译

可交付：真实训练案例、错误诊断、自动代理测量、真实搜索路径和有明确失败条件的新获取实现。
未完成：新coverage真实模型结果、历史LTR/LoRA原件恢复、生产采用/SLA/ROI。
人工标注不再阻塞；个人白板、源码防守和现场改核心模块仍需本人练习。

建议项目定位：**气候证据检索与排序**，不是已经证明收益的自主事实核查Agent。
可用表述：

- 面向120.9万条气候证据，构建hard-negative→InfoNCE/LoRA→ANN→融合/排序训练链；
  在154条offline-dev上Recall@5由27.93%提升到29.70%，保留5,000次配对bootstrap与泄漏边界。
- 在独立于上述受限轨的公开validation上比较低延迟LTR与4B精排，报告召回—时延取舍；
  对真实三路获取轨迹增加引用语义代理和送达审计，发现自主路线提前停止，未宣称Agent质量收益。

不要把负结论伪装成收益，也不用声称节省时间、线上生产采用或人工盲标完成。

## 无需人工参与的验收与仍不可认领的成果

完成条件已改为：原官方gold任务评分、确定性送达/合同检查、固定自动NLI语义代理、
真实模型运行与成本各自分开报告，不要求用户或他人新增人工标注。
自动代理不能证明真实语义正确；减少无依据判断是业务目标，不是本轮已经测得的收益。
生产采用、用户节时、在线SLA需要实际部署和使用数据，不能用离线演示补造。

## Runpod（本轮未启动付费计算）

2026-10-03读取加载完成的账单：10月2日US$3.302，10月3日已入账US$1.991，
合计US$5.293；原整轮US$20不重置，减去已入账后为US$14.707，
尚须扣未入账及后续卷费，**不是精确剩余授权余额**。账单约延迟1小时，按UTC统计。
页面账户余额US$9.58另列，不当作授权预算。Auto-Pay Disabled。
四个既有Pod未启动，保留卷总页面费率US$0.133/h（约US$3.192/day），
本轮没有GPU/API费用，但停止计算不停止卷费；未永久删除任何卷。
