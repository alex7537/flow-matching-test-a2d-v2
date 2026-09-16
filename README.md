# A2D 模型评测闭环 · v0

本分支 `feat/a2d-eval-loop-v0` 将**训练得到的模型、仿真评测结果和下一轮实验建议**串起来。当前实现负责验收、记录和分析；训练与机器人动作执行仍由已有系统承担。

## 一眼看懂流程

```text
开发机训练并导出 bundle
        ↓ 人工确认模型和下载位置
pull：下载并校验文件哈希
        ↓ 人工确认部署和测试参数
已有 gRPC 测试器：加载模型 → 观测 → 预测动作 → 执行 → GT/接触判定
        ↓ 写入逐轮结果
本分支：init 冻结身份和协议 → tick/watch 校验、快照、汇总
        ↓
按模型 × 场景报告成功率、异常和失败类型
        ↓
生成下一轮实验草案 → 人工分析并确认 → 新一轮开发/训练或评测
```

**每项新增行动执行前，先说明输入、输出和影响，再由人工确认。**上一项确认不授权下一项；已授权的测试可按原计划继续。CLI 没有内置审批系统，此规则由操作者执行。

## 哪部分负责什么

| 部分 | 职责 | 是否由本分支启动 |
|---|---|---|
| 开发机训练 | 训练、选择权重、导出 bundle | 否 |
| `loop.py pull` | 按指定 SSH 路径下载，校验外部和包内哈希 | 人工确认后调用 |
| Isaac / gRPC 测试器 | 重置场景、获取图像和状态、执行动作、记录抓取结果 | 否，接入已有任务 |
| `loop.py init` | 登记 bundle、计划和评测代码哈希，建立独立 cycle | 人工确认后调用 |
| `loop.py tick` / `watch.py` | 读取结果、检查一致性、保存快照、更新报告 | 已授权范围内观察 |
| 下一轮任务 | 提出待验证假设和单变量改动 | 只生成草案，不自动启动 |

## 怎么测，怎么比较

- 当前只接入 **逐步 RPC＋GT、H16、300 动作上限、模型 seed 42** 的既有结果。每个场景单独比较，场景种子按计划变化。
- 任务标准是“曾抓起来”：多指接触（拇指＋至少另外两指）与抬升至少 **5 cm** 同时成立，连续 **5 次动作后的观测**满足；之后掉落不扣除成功。
- **成功率 = 成功次数 / 有效尝试次数**。有效尝试包括正常完成和策略保护拒绝；reset/基础设施异常另列。保护拒绝前已经成功的尝试仍算成功。
- 对失败区分：没有有效多指接触、接触与抬升未同时满足、持续次数不足。
- 控制层检查逐轮摘要的一致性，**不会从每个物理帧重新计算接触和成功**。相同 seed 也不保证完整物理初态一致，因此当前结果用于诊断，不直接证明训练改动的因果效果。

## 如何运行

需要 Linux/Python 3 标准库、可访问的 bundle 和已有评测目录。每个评测目录需符合当前 adapter 的格式，包含 `plan.json`、`evaluate.py`，运行后产生 `contract.json`、`episodes.jsonl`；不是任意日志目录都能接入。既有 gRPC runner 尚未作为本工具的可移植执行后端发布。

以下使用占位路径；在对应步骤确认后运行：

```bash
cd tools/a2d_eval_loop

# 接入已有计划，冻结身份并生成首份结果快照
python3 loop.py init --cycle runs/my_cycle \
  --task-dir /path/to/box --task-dir /path/to/bottle \
  --hypothesis '相同协议下比较候选模型在两个场景的抓取表现'

# 单次更新；也可以选择持续观察
python3 loop.py tick --cycle runs/my_cycle
python3 watch.py --cycle runs/my_cycle --interval 60 --max-hours 72
```

`watch.py` 在源文件变化时更新快照，连续三次错误或达到时限后退出。cycle 中创建 `PAUSE` 文件只暂停该观察器，**不会停止仿真测试**。协议或代码变更需新建 cycle，保留旧证据。

## 看哪些输出

| 输出 | 用途 |
|---|---|
| `manifest.json` | 冻结模型、评测计划和代码身份 |
| `latest.json` | 定位最新快照 |
| `snapshots/.../REPORT.md`、`aggregate.json` | 各场景/模型的计数、成功率、描述性置信区间和失败分类 |
| `decision.json` | `wait`：尚未测齐；`retry-eval`：缺少运行证据；`new-experiment`：进入下一轮分析 |
| `next_experiment.json` | 下一轮任务草案，假设和改动需人工补充确认 |

`new-experiment` **不代表自动开始实验**。硬性合约不一致会报错，需检查后处理。原始视频、模型、机器配置和运行目录不提交到仓库。

## 当前完成程度

已验证：9 项控制层单元测试；本机历史 600 次尝试、六组统计与原报告一致；已对当前任务做只读接入。

尚未完成：新 SSH 下载路径端到端联调、可移植仿真 runner、完整初态恢复和专家 replay 标定、远端训练任务接口。没有自动训练、自动模型晋级或完整无人值守闭环。

详细入口：[工具说明](tools/a2d_eval_loop/README.md) · [闭环说明](docs/A2D_EVAL_LOOP.md) · [开源参考](tools/a2d_eval_loop/docs/OPEN_SOURCE_PATTERNS.md) · [原始 loop 方案](docs/LOOP_FIRST_INSTANCE.md)

---

## 原训练仓库说明

以下为继承自基础分支的训练实现说明，其数据语义描述不替代外部候选 bundle 的独立溯源核查。


这是一个基于处理后 A2D HDF5 的 `RGB condition -> flow matching -> joint chunk` 训练骨架。

当前仓库只保留一条主线：

- 条件输入是可配置 RGB 视角子集；可用视角为 `rgb_head`、`rgb_left_hand`、`rgb_right_hand`，训练时通过 `data.image_keys` 选择一路、两路或三路
- 监督目标默认是绝对 joint target；需要 delta 时必须配套计算 delta stats
- 模型输出未来一段 joint action chunk

RGB encoder 现在支持两种后端：

- `cnn`：当前仓库自带的最小共享 CNN，便于本地 smoke test
- `timm`：更贴近 `fan_dev` 的视觉路线，可切到 `vit_small_r26_s32_224`

## 唯一数据主线

仓库不再支持“旧 HDF5 + 外部 JPEG 目录”。唯一输入链路是：

```text
原始新 HDF5（trajectory/cameras/rgb_* 内嵌 RGB）
  -> scripts/preprocess_a2d.py
处理后 HDF5（observations/rgb_* 为逐帧 JPEG，含 qpos/action/phase）
  -> flow_matching_test.a2d_dataset --compute-stats
index_cache.json + norm_stats.json
  -> flow_matching_test.train
```

原始文件中的 state/action 映射为：

- state：`arm2_pos(7) + hand2_pos(6)`
- action：`arm2_pos(7) + hand2_pos(6)`

action 固定为 13 维实际执行关节位置，归一化统计使用 train episodes 的逐维 min/max；稀疏的 `*_pos_target` 不参与训练标签。
默认 `action_offset_steps=1`，因此时刻 `t` 的观测对应
`action[t+1:t+1+action_horizon]`；输出 `chunk[0]` 是下一帧绝对关节位置，
不再重复当前 proprio。旧 checkpoint 未记录该字段时按历史语义 `offset=0` 解释，不能
与新窗口语义混用或直接 resume。

模型条件输入默认包含配置选中的 RGB spatial tokens 和归一化 13 维 proprio state token。

## Policy 对比实验

训练器与具体策略已解耦，配置通过 `policy.type` 选择策略；当前实现为：

```yaml
policy:
  type: flow_matching
```

策略实现放在 `flow_matching_test/policies/`。新增 policy 时各自实现
`ActionPolicy.compute_loss()` 与 `sample_actions()`，并在 factory 注册新的
`policy.type`；训练循环统一负责 `loss.backward()`、optimizer、checkpoint 和日志，
无需为每种 loss 复制一份 trainer。当前支持：

- `flow_matching`：连续时间速度场目标，Euler 采样
- `imle`：与 psi-policy 对齐的 RS-IMLE 候选匹配，默认每个条件 20 个候选
- `diffusion`：离散 cosine noise schedule 的 epsilon 预测，默认 15 步 DDIM 采样

三种 policy 的 loss 数值空间不同，不应直接横向比较；正式比较使用同一数据、网络
主体、训练步数与 seed，并以 rollout 成功率、sample action MSE 和推理延迟为准。

先执行预处理（示例选择两路相机）：

```bash
python3 scripts/preprocess_a2d.py \
  --src /path/to/new_raw_hdf5/success \
  --dst /path/to/a2d_processed \
  --image-keys rgb_head rgb_right_hand \
  --image-size 224 --jpeg-quality 92 --workers 4
```

再建立索引和归一化统计：

```bash
python3 -m flow_matching_test.a2d_dataset \
  --data-dir /path/to/a2d_processed \
  --image-keys rgb_head rgb_right_hand \
  --compute-stats --rebuild-index
```

如果你要更贴近 `fan_dev`，推荐用 `timm` 后端，并把 `timm_model_name` 设为
`vit_small_r26_s32_224`。

## `timm` 最小检查

如果你想先确认环境是否支持 `vit_small_r26_s32_224`，推荐按这 3 步做：

1. 安装 `timm`

```bash
pip install timm
```

2. 先只检查模型名是否存在

```bash
python3 -m flow_matching_test.check_timm_env \
  --model-name vit_small_r26_s32_224 \
  --list-only
```

3. 再检查能否真正实例化和前向

```bash
python3 -m flow_matching_test.check_timm_env \
  --model-name vit_small_r26_s32_224 \
  --pretrained
```

如果第 3 步能通过，基本说明：

- 当前环境里已经有 `timm`
- 这个模型名在当前 `timm` 版本中可用
- 预训练权重可以下载或从缓存中加载
- 该 backbone 至少能完成一次最小前向

## 模型主链路

```text
selected RGB views -> encoder token map -> ObsComposer -> obs tokens
noisy action chunk + time embedding -> action tokens
action self-attn + cross-attn to obs tokens
predict velocity
MSE(pred_velocity, target_velocity)
```

loss 目前只有一个：

- 标准 flow matching velocity MSE
- 训练时间参数采用 `t ~ Uniform(time_eps, 1.0)` 的 straight-line CFM

## 启动训练

最小 `cnn` 版本：

```bash
cd /home/psibot/Downloads/flow-matching-test
python3 -m flow_matching_test.train \
  --config configs/minimal_rgb_flow.yaml \
  data.data_dir=/path/to/a2d_processed
```

`timm` 版本：

```bash
python3 -m flow_matching_test.train \
  --config configs/minimal_rgb_flow_timm.yaml \
  data.data_dir=/path/to/a2d_processed
```

如果你想把训练过程同步写成 `rerun` 的 `.rrd`：

```bash
python3 -m flow_matching_test.train \
  --config configs/minimal_rgb_flow.yaml \
  data.data_dir=/path/to/a2d_processed \
  visualization.rerun.enabled=true
```

当前最小 `rerun` 链路会记录：

- 每个 epoch 的训练/验证标量
- 一条 sample 的所选 RGB 输入
- 这条 sample 的 GT / Pred action chunk
- 归一化动作空间与反归一化动作空间下的 action 曲线

默认输出到当前 run 目录下的 `training.rrd`。

如果你想把训练指标同步打到 `wandb`：

```bash
# A800:先在 W&B 撤销曾暴露的旧 key，再交互式写入容器本地密钥文件。
install -d -m 700 /root/.secrets
read -rsp "New W&B API key: " WANDB_KEY && echo
install -m 600 /dev/null /root/.secrets/wandb_api_key
printf '%s' "$WANDB_KEY" > /root/.secrets/wandb_api_key
unset WANDB_KEY

source ./activate_a800.sh  # 自动读取 /root/.secrets/wandb_api_key

WANDB_MODE=online python3 -m flow_matching_test.train \
  --config configs/minimal_rgb_flow.yaml \
  data.data_dir=/path/to/a2d_processed \
  logging.wandb.enabled=true
```

当前最小 `wandb` 链路会记录：

- 每个 epoch 的 train/val、static/continuous/keyframe loss 与 sample action MSE
- head/backbone 两组学习率，以及两组梯度范数的 epoch mean/max
- 三个不参与反向传播的视觉 encoder 监控指标：
  - `encoder_update_ratio`：一个 epoch 内 backbone 参数实际变化量 / epoch 初参数量
  - `encoder_grad_param_ratio_mean`：backbone 梯度范数 / backbone 参数范数的 batch 均值
  - `encoder_feature_std`：诊断 batch 上原始视觉 token 的逐通道标准差均值，用于监测特征坍缩
- git、dataset、stats、split、segmentation provenance
- 本次 run 的配置、`best_epoch / best_val_loss` 与最终 summary

这三个 encoder 指标只用于观察，不会加到 CFM、RS-IMLE 或 Diffusion 的训练 loss，
因此不会改变现有三 policy 的优化目标或公平比较协议。

默认使用 offline 模式，W&B 异常会自动降级为 no-op，不会中断训练；短任务需要实时同步时才临时设置 `WANDB_MODE=online`。

A800 的 `HOME` 被显式重定向到共享 CFS 的 `$WORK/home`，只用于非敏感缓存且目录权限固定为 `700`；任何 API key、SSH key、token 与 shell history 都不得写入该目录，W&B 凭据只允许由 TI-ONE Secret 注入或保存在容器本地 `/root/.secrets/wandb_api_key`（`600`，容器重建后重新注入）。

如果只是本地先试，不想真的上传远端，可以这样：

```bash
python3 -m flow_matching_test.train \
  --config configs/minimal_rgb_flow.yaml \
  data.data_dir=/path/to/a2d_processed \
  logging.wandb.enabled=true
```

常用 smoke test：

```bash
python3 -m flow_matching_test.train \
  --config configs/cpu_smoke.yaml \
  data.data_dir=/path/to/a2d_processed \
  training.output_dir=/tmp/a2d_cpu_smoke
```

`cpu_smoke.yaml` 使用 CNN、两路 64×64 RGB、batch size 2、2 epochs；每个 epoch
只运行 2 个 train batch 和 1 个 val batch。它只验证数据、前后向、指标和 checkpoint
链路，不能用于判断模型是否收敛。

如果你已经有了 `best.ckpt`，也可以像 `fan_dev` 那样单独导出 checkpoint eval 的 `.rrd`：

```bash
python3 -m flow_matching_test.export_rerun_eval \
  --ckpt /path/to/best.ckpt \
  --split val \
  --num-samples 8
```

这个脚本会：

- 重新加载 checkpoint 和配置
- 在 `train` 或 `val` split 上取若干条 sample
- 导出 RGB + GT/Pred action 的 `.rrd`
- 旁边再写一个同名 `.json` summary

## 协作与分支约定

`main` 是唯一长期分支和可部署事实源；一切改动从最新 `main` 创建短命分支，通过 PR 审查并使用 **Squash and merge** 合并，合并后删除该分支，禁止直接 push `main`（强制分支保护待账号支持后开启）。

本仓库发布不依赖 GitHub CLI `gh`，不得因其缺失阻塞发布；前置检查仅要求 `git remote -v` 指向正确的 `origin` 且 `ssh -T git@github.com` 认证通过，提交与推送使用原生 Git，PR 通过 GitHub 网页或已连接的 GitHub 接口创建。

分支名使用 `<类型>/<描述>`，例如 `fix/runbook-typo`、`report/level0`、`feat/prefix-mask`；A800、bundle manifest 和 runbook 中的 `git_sha` 始终指向已合并的 `main` commit，不使用未合并分支作为正式训练或部署基线。

训练、评估与部署产物本体不进入 Git；统一登记到 `artifacts_index.md`，记录存放位置、SHA-256 与对应 `git_sha`。训练曲线保存在 W&B，交付级模型保存在自包含 bundle、A800 或 COS。

标准流程：

```bash
git switch main
git pull --ff-only origin main
git switch -c <type>/<description>
# 修改、验证、commit
git push -u origin <type>/<description>
# 在 GitHub 创建 PR → review → Squash and merge → 删除远程与本地短命分支
```

## 文件说明

- `flow_matching_test/a2d_dataset.py`：处理后 HDF5 Dataset、切分、归一化与模型输入适配
- `scripts/preprocess_a2d.py`：原始内嵌 RGB HDF5 转换为训练格式
- `docs/DATA_PIPELINE.md`：完整数据管线与验证协议
- `artifacts_index.md`：重要外部产物的位置、SHA-256 与代码血统索引
- `flow_matching_test/policies/`：统一 policy 接口、factory 与独立的 flow-matching policy 实现
- `flow_matching_test/model.py`：旧导入路径的兼容别名，已有脚本和 checkpoint 无需迁移
- `flow_matching_test/observation.py`：最小 observation 模块，负责 encoder / concat / obs composer
- `flow_matching_test/check_timm_env.py`：检查 `timm` 环境、模型名和预训练权重是否可用
- `flow_matching_test/rerun_logger.py`：最小 `rerun` 训练/评估可视化封装
- `flow_matching_test/wandb_logger.py`：最小 `wandb` 实验日志封装
- `flow_matching_test/export_rerun_eval.py`：加载 `best.ckpt` 并导出 checkpoint eval `.rrd`
- `flow_matching_test/train.py`：训练循环、验证和 checkpoint
- `flow_matching_test/export_bundle.py`：将 checkpoint 导出为带哈希校验的 rollout bundle
- `rollout/`：Isaac Sim 4.5 固定网格 rollout、三阶段判据与结果汇总
- `configs/minimal_rgb_flow.yaml`：`cnn` 版配置
- `configs/minimal_rgb_flow_timm.yaml`：`timm` 版配置

## Rollout bundle 与 Isaac Sim

手工导出一个 bundle：

```bash
python3 -m flow_matching_test.export_bundle \
  --ckpt /path/to/best.ckpt \
  --out /path/to/eval_bundle_fm_step50k \
  --execute-horizon 8 \
  --data-version DATA_VERSION \
  --archive
```

也可以将训练配置中的 `deployment.eval_bundle.enabled` 改为 `true`，每次出现新的
best checkpoint 时会自动生成 `eval_bundles/eval_bundle_<run>_step<step>.tgz`。

在 4090 的 Isaac Sim 容器中运行：

```bash
/isaac-sim/python.sh rollout/run_rollout.py \
  --bundle /workspace/bundles/eval_bundle_fm_step50k \
  --grid rollout/eval_grid.yaml \
  --out /workspace/results/fm_step50k

/isaac-sim/python.sh rollout/report.py --in /workspace/results/fm_step50k
```

首次运行前必须在 `rollout/eval_grid.yaml:sim` 填入已校准场景 USD、机器人/物体/
末端/相机 prim path、13 个实际 articulation DOF 名称以及至少两个接触传感器路径。
当前数据的 action 顺序是右臂 7 个主动关节，加右手 6 个主动关节：
`1_1, 2_1, 3_1, 4_1, 5_1, 1_2`。本机已有
`/home/psibot/Downloads/InspiredHand_RuiYan/RuiYan_Hand_Right_Mimic.usd`；应使用该
Mimic 资产，让另外 5 个手部关节按资产内规则联动：拇指 `1_3 = 1.675 * 1_2`，
其余四指的远端关节与对应近端关节保持 `1.0` 倍。rollout 只下发上述 6 个主动
手部 DOF，不应再额外拟合或重复下发 11 个手部 DOF。
