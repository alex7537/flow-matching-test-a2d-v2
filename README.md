# Flow Matching Test

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
wandb login <YOUR_WANDB_API_KEY>

WANDB_MODE=online python3 -m flow_matching_test.train \
  --config configs/minimal_rgb_flow.yaml \
  data.data_dir=/path/to/a2d_processed \
  logging.wandb.enabled=true
```

当前最小 `wandb` 链路会记录：

- 每个 epoch 的 train/val、static/continuous/keyframe loss 与 sample action MSE
- head/backbone 两组学习率，以及两组梯度范数的 epoch mean/max
- git、dataset、stats、split、segmentation provenance
- 本次 run 的配置、`best_epoch / best_val_loss` 与最终 summary

默认使用 offline 模式，W&B 异常会自动降级为 no-op，不会中断训练；短任务需要实时同步时才临时设置 `WANDB_MODE=online`。

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
