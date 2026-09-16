# A2D 最小评测闭环 v0.1

训练机产出 bundle → 拉取并验哈希 → 现有 gRPC 评测 → 按场景和模型分析 → 生成下一轮实验任务单 → 人工确认后回训练机。

当前实现是 **L1 诊断控制层**：登记模型、冻结协议、接入现有评测结果、生成报告与交接草稿。真实仿真执行由原有 `orchestrate.py/evaluate.py` 负责。本代码不重复启动仿真，不修改当前800次测试，不自动创建训练任务，发布代码不包含本机运行记录。

## 已可用的五个环节

| 环节 | 输入 | 输出 |
|---|---|---|
| 训练交接 | bundle 内 manifest：训练run、数据版本、step、环境 | manifest 中 models 的训练溯源 |
| 拉取与接收 | 本地bundle，或SSH地址＋明确远端文件＋预期SHA256 | 外部SHA与所有内部文件校验；不解压执行代码 |
| 评测后端接入 | 当前gRPC评测目录的plan/contract/episodes | 按模型×任务冻结合约、校验身份和覆盖 |
| 结果与决策 | 每轮终态、接触、持续帧数 | 成功/有效、invalid、失败分类、Wilson区间、decision |
| 下一轮交接 | 已完成评测快照和未决问题 | next_experiment.json草稿，填写一个假设和单变量改动后再训练 |

当前支持 `step_gt / H16 / 300 / policy seed42 / 5cm连续5动作观测`，其他协议显式拒绝，不能默默混合。数据、策略、token类型、权重、执行模式的变化作为比较角度记录；V0不做跨口径总排名。

## 本机验证记录（运行产物不随代码发布）

- `runs/history_three_models`：上一轮三模型、box/B19共600次尝试，已完成回放式报告生成（没有重跑机器人）。
- `runs/pro5000_four_models`：当前四模型、两场景共800次计划；只读观察，未完成输出 `wait`。
- 每个cycle的 `latest.json` 指向不可变快照；`state.json`是当前摘要，`observer.json`是观察器状态。

```bash
cd tools/a2d_eval_loop
python3 loop.py tick --cycle runs/pro5000_four_models
python3 -m unittest discover -s tests -v
```

注册新的评测周期（task-dir需要既有评测计划和evaluate.py）：

```bash
python3 loop.py init --cycle runs/<新周期> \
  --task-dir /path/to/box --task-dir /path/to/bottle \
  --hypothesis '本轮比较什么，只改变什么' \
  --parent runs/pro5000_four_models
```

`init`验证bundle，不代表模型加载/推理smoke通过；smoke由现有后端承担。历史模型的manifest是声明证据，不能替代训练loader核查。

按明确路径拉取新bundle（参数需按实际交接填写；本次未新增远端传输）：

```bash
python3 loop.py pull --host user@training-host --port 22 \
  --remote /absolute/path/latest-raw.tgz \
  --sha256 <训练端提供的64位哈希> --dest /local/new-bundle.tgz
```

仅使用已有SSH认证、BatchMode；失败不遗留可被当成完成品的文件。拉取只读远端，不安装环境、不触发训练。拉取传输入口已实现，但本次没有联网端到端测试它；实际闭环验证使用已在本机的bundle。

## 判定与分析

- 基础设施有效性、抓取结果、科学结论分开。`completed`必须实际执行300动作；`policy_rejected`在有效分母内，可能拒绝前已经成功。
- reset/通信错误单列，不算模型抓取失败。100是尝试预算，不是强制补齐100个有效样本。
- 分别报告每模型、每场景，不用一个平均数隐藏任务差异。区间是描述性Wilson区间；共用仿真、布局不完全配对时不宣称显著性。
- `wait`：样本未齐；`retry-eval`：证据缺失/不一致；`new-experiment`：诊断结果已齐，进入失败分析/下一假设草稿。
- V0不产生自动`promote`，也不会把草稿自动交给训练机。所有结果为diagnostic，未认证固定layout、expert replay、密封holdout及晋级阈值。
- 下一轮未必重训：先区分数据/标签、控制/预处理、指标/环境问题。只有训练假设和单变量改动确认后，才填写任务单的training_job。

## 边界和迭代顺序

1. V0（已做）：连接现有产物，保存完整证据快照，输出报告与任务草稿。
2. V1：把临时gRPC runner整理成稳定CLI，统一任务/模型注册、首次smoke、恢复合同。
3. V2：固定完整初态，专家replay标定，离线E1/E2与闭环共同验收，冻结可比阈值。
4. V3：接训练平台任务ID和导出/传输；经明确授权后才调度远端训练。不会由本次骨架自动升级自治权限。

观察器每60秒检查一次；只在产物改变时生成快照，三次连续错误或72小时上限退出。创建cycle目录的`PAUSE`文件只暂停此报告观察器，**不会暂停仿真评测**。控制层使用本机文件锁；不声称支持多机器并发控制。代码或评测plan发生变化需新cycle，旧证据保持不变。

代码位于训练仓库 tools/a2d_eval_loop；仿真执行后端仍由 fk-issac-logistics 管理。运行记录、模型和机器配置留在执行机器，不纳入 Git。

## 人工确认规则

测试阶段逐步确认。每项新增行动先说明具体输入、输出、影响和预计范围，收到用户明确确认后才执行。前一步确认不授权后续步骤。

1. 下载：说明开发机、模型版本、源路径、目标路径、大小及预期哈希。
2. 部署：说明依赖变更、环境及对运行任务的影响。
3. 测试：说明模型、场景、轮数、种子、动作协议、成功标准和耗时。
4. 归档或发布：说明批次、目标目录或远端分支和具体文件范围。
5. 下一轮：先提供证据和单变量实验草案，经人工确认后启动。

已经授权的测试与观察器可在原定范围内继续。next_experiment.json 只是草案，new-experiment 只是决策标签；均不代表启动授权。当前 CLI 不实现授权身份校验，人工门禁由操作者执行，不能作为无人值守调度入口。
