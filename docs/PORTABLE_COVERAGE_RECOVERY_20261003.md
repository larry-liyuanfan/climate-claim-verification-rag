# 可迁移实验包：不绑定旧服务器

本轮已经完成一次真实96槽比较。迁移指的是保存/恢复同一结果，不算独立实验。
没有新运行许可；下面CPU回放不执行生成模型、付费API或上传私人材料。
不连接Spartan，不等待账户恢复，不删云端卷。需要新推理时重新绑定实际目标机、输入、
源码、模型、预算、输出目录和用户授权，**旧release不能复制到另一台机器直接运行**。

## GitHub保存的完整安全部分

同一writer分支 `codex/climate-bounded-gap-repair-20260930` 保存：

| 内容 | 路径 / SHA256 |
|---|---|
| 全三路分母、指标、bootstrap、成本与耗时 | `docs/verified-runs/coverage-regression32-result-20261003.json` / `6c33c11d6390af815ed985dfa9d114d28d1055ea58eedb5ed9e9a036f452f05a` |
| 全量gold-free行为统计、失败原因与按固定顺序的案例位置 | `docs/verified-runs/coverage-regression32-behavior-20261003.json` / `8dca6c2171e19723cb56681cb6bbe138ea3e0467c2333084abb5247b8caf60ab` |
| 独立自动NLI proxy与模型/评分身份 | `docs/verified-runs/coverage-regression32-nli-proxy-20261003.json` / `9da813f577954fc1ea0518b7d80854df59e1d1a0a7c9404950a999bcd4f84df2` |
| 预注册、初始信息、任务消费/选择/评分合同 | `docs/protocols/coverage-three-arm-20261003.json` / `d7cc2c796e470a0bf3d8a7244d7f88fd9e8f07bb8d0d773113f7940a2e6d5f28`（运行时原文件hash） |
| 真实结果和业务决策 | `docs/COVERAGE_REGRESSION_RESULT_20261003.md` |

前三个生成JSON用窄范围Git `-text` 属性保留原字节，Windows/Linux checkout不自动转行尾。
源代码、测试和文档都由Git提交保存，不需要旧Pod才能取回。完整旧结果分别保留，
不把54031fa、e826和f792协议/样本的数字混成同一测试。

公共仓库**不保存**受限corpus、gold、逐题claim/引文/提示/概率、原始private manifest、
凭据或大模型缓存。没有“所有原文都已上GitHub”这一承诺。
安全汇总与hash可供求职、开发和核对；需要完整重放的授权开发者另取私密归档。
这避免旧服务器依赖，但目前私密备份仍是本地副本，不冒称已有第二份异地云备份。

## 完整私密备份的身份

归档已从实际运行实例回收，与远端hash对照一致，再停止计算。
物理位置和操作凭据仅在本地handoff，以下公开指纹不含原文或gold：

| 产物 | SHA256 / 定义 |
|---|---|
| 完整输入 `input.tar` | `563738f0be1f7bf7b99b8de20bcdeab552f7e893166dec260b8f7f1e7951c3c1`，16,122,255,360 bytes；含原模型/语料资产，不是历史LoRA/LTR原件 |
| 完整结果 `recovery.tar.gz` | `7baafd040cc90b8ac361b3954fd6baa98290e9fa72a7832b91479aad573e855b`，3,868成员、17,208,435 uncompressed bytes；整个run、输入绑定、隔离评分文件和监督回执 |
| 执行源码原包 `source.tar` | `f8bffda39fbb2b60d7ab90a42affa89fa21babbe023e1761bb4fe0225c1fccc8`，6,717,440 bytes；对应f792 |
| 原始run | `48d597d3abc82b8309cf2952895c787ff388c589371c37c1cecd6609cca778a6` |
| operator回执 | `113424f1a4bc904073eed3d83dc8d0d9f3682101246d39d946f6a0a40fb110e7` |
| 原官方compact / cost | `f2d5c9a1b483ed97f668299fbb7d6674ff72908058bd9bbe33a661c36ed02968` / `d41e4fc210a755580af00289a491f078b513d5f552e4ddb75b6f2dee38c42bcd` |
| deadline退出回执 | `27ecc2c46bf2dce68df2ff8649c0583d4d9549d06326f8cf33fc8f8a99a95d3d`，worker/operator退出0且reaped/completed |
| 实际授权release | `19d36757a8a749aa02cb557a63ae931e0b0f4fcc06b3769f790dd8c1ef1a8901`，仅当前已完成运行有效 |

完整archive先检查无绝对路径、父目录穿越和链接，再以安全提取方式恢复；
私密目录0700、文件0600，并检查实际权限而不是MFS声称的mode。
运行时MFS保存模型资产，owner-only POSIX保存输入绑定、评分与逐题输出；
容器私密路径停止后可能消失，故**回收必须先于停机**。新服务器也沿用此分离，
不假定新CPU Pod能读取旧Pod本地卷。

公开恢复清单：

| run / 执行源码 | 当前可复用内容 | 未恢复 / 是否阻塞本轮 |
|---|---|---|
| stop-acquire-private / 54031fa | 完整160槽历史私密备份，原汇总及行为回放 | 无需重复下载/评分；不阻塞 |
| fair-three-arm / e826a4d | 完整96槽及旧NLI，原all-stop结论 | 不用新协议撤销旧结果；不阻塞 |
| coverage-regression32 / f792b9e | 完整96槽、完整输入、监督与成本、当前NLI和诊断 | 完整恢复；本轮完成 |
| 历史restricted LoRA / public LTR | 本地报告、摘要、登记hash | 原权重和部分逐题轨迹未恢复，远端不可访问；不能演示那条完整历史推理链，但不阻塞本轮结果解释 |
| 曾经启动失败的云迁移尝试 | 预运行权限/资源回执 | 0模型槽，不存在可“恢复”的模型成果；不阻塞 |

## 执行身份、环境与接口

真实模型执行源码：`f792b9e0c09f95ddc46c7e613541b90eeded3edc`，
entry `scripts/run_cloud_replay.py` SHA `afa5aa39d5e19950f2376e0ffbec55776518dc77017a168db1dd36f1abc60ef0`。
主要入口沿用实际模块，不重写框架：

```text
冻结claim＋相同初始frame
  → 固定多查询 / 确定性工作流 / 自主gate
  → 共用read/query/rerank → 文本送达＋状态更新 → 下一物理模型提示
  → 共用终答和引用合同 → 隔离退出评分＋完整成本 → 私密回收＋安全汇总
```

模型：

- `Qwen/Qwen3-4B@350135a4de9a3407be836fa238cccc1d61503a85`；资产身份
  `d1dd9783afdf4e0fbd21eee824834d71b86982f5a5d5f6f371fe07f2f76f3cf6`。
- `Qwen/Qwen3-Reranker-4B@22e683669bc0f0bd69640a1354a6d0aebcfeede5`；资产身份
  `de1d4ac39101816774439e68881e2308c5e5f1bd94d0b0dc4c492a56c2681052`。
- 语料hash `c14315aee9feecbbbbc3b0c7101d978b52b47cee9a613336304bcaa736460c71`；
  cohort `f2b7c55b1e91f297a613df5762f44d46c2df7666cca8e234d77d5ca53488ec56`；
  原始有序tasks对象的canonical JSON hash `de088bac5fb93beafd2393a438cbc42ac0f99479ff173817a734c01d071e5b25`；
  exposure audit `999eda72085e34ab001fa1f8b8f4e4a5cd861adec699d91be9af76ccd60e6b8a`。
  原cohort文件hash、原始tasks的canonical JSON hash与后续claim文本规范化用途不同。
  canonical JSON保留原claim空白；不从已规范化输出冒算原输入hash。

实际Linux Python3.11.13、Torch2.7.1+cu126、Transformers4.51.3、NumPy1.26.4、
tokenizers0.21.4、accelerate1.6.0、Pydantic2.11.7/core2.33.2、
jsonschema4.23.0、lm-format-enforcer0.11.3、interegular0.3.3、safetensors0.5.3。
单A100 80GB；申请合同最低GPU39GiB、CPU8、RAM64GiB、disk100GiB；
实际节点CPU16/RAM250GB/container30GB/persistent120GB。
GPU瞬时约9GB与整个容器memory.peak约55.9GB都**不是worker峰值或未来最小资源证明**。

## CPU回放与后续云开发

无模型、无网络的查看（只读Git内安全汇总）：

```sh
python -c "import json; p='docs/verified-runs/coverage-regression32-result-20261003.json'; d=json.load(open(p)); print(d['study_kind']); print({k:(v['official_task_correct'],v['cost']['generation_calls']) for k,v in d['routes'].items()})"
python scripts/demo_recruitment_case.py
```

第二条保留训练/搜索与**旧v1**行为展示，不冒充当前coverage链实时运行。
有授权私密archive时，复用当前后验分析代码重建安全汇总：

```sh
python scripts/export_fair_run.py --run-dir "$PRIVATE_RUN_DIR" --output "$SAFE_AGGREGATE"
python scripts/replay_fair_acquisition_result.py --protocol fair-acquisition-coverage-v2-20261003 --run "$PRIVATE_RUN_DIR/inference/run.json" --run-sha256 48d597d3abc82b8309cf2952895c787ff388c589371c37c1cecd6609cca778a6 --output "$SAFE_BEHAVIOR"
```

不需要重跑官方评分或160/96历史模型。NLI已完成，完整compact在Git，不需要每次迁移重算。
未来需要全套资产时先核SHA、容量、权限、Python/Torch/解析器、模型revision，
再创建新机器的source/asset/runtime/private-storage回执和**默认禁止执行**的新draft。
先准备依赖与资产，不付费等待本地修复；监督入口使用isolated Python `-IB`、
完整deadline、唯一输出、gold隔离及退出回收。新真实运行仍需新身份放行，旧批准已消费。

迁移原结果只复制上述已验证private包，不上传到公开Git；欲改存新的私有云目的地，
需明确目的地/访问权限及受限材料许可，不能把“GitHub保存”推断成任意公开私密数据的授权。
