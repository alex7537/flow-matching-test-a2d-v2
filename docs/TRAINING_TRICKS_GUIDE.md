# 模仿学习策略训练技巧与实验路线指南

本文档是本仓库 CFM、RS-IMLE 与 Diffusion 策略的训练实践指南，适用于 ViT
视觉编码器、13-DOF 动作空间（`arm2_pos(7) + hand2_pos(6)`）、A800 训练和
Isaac Sim rollout 评估。

训练预算的公式与换算方法见 [TRAINING_PLANNING_GUIDE.md](TRAINING_PLANNING_GUIDE.md)，
数据生成与验证契约见 [DATA_PIPELINE.md](DATA_PIPELINE.md)。

当前 1,090-episode Frozen/0.1× 五轮消融结果见
[2026-07-21 实验报告](../reports/cfm_a2d_450gb_vit_freeze_ablation_20260721.md)。

## 1. 核心原则

1. 先证明数据、时序和推理路径正确，再调超参数。
2. 每个技巧必须对应一个已确认的问题，每次实验只改变一个变量。
3. 不同数据规模优先比较 optimizer steps 和样本曝光量，不直接比较 epoch 数。
4. offline 指标是路标，固定协议下的 rollout 成功率才是最终裁判。
5. 建议不等于已实现；本指南明确标记 `已实现`、`待消融` 和 `待实现`。

## 2. 训练前必须成立的数据契约

### 2.1 动作语义与归一化

- state 和 action 均使用执行后的关节位置：`arm2_pos(7) + hand2_pos(6)`。
- action 必须是 13 维，不使用稀疏的 `*_pos_target` 作为监督标签。
- min/max 只从固定训练集计算，逐维映射到 `[-1, 1]`。
- train、validation、export 和 rollout 必须使用同一份 stats，并记录 stats digest。
- 数据变更后必须重新验证 stats 与 train episode membership 的绑定。
- CI 应覆盖 `denormalize(normalize(a)) ≈ a`，目标误差小于 `1e-5`。
- CFM validation 的 noise、time 与 action sampling seed 按样本索引固定；
  `val_sample_action_mse` 默认平均 3 个固定 seed，避免 checkpoint 排名受抽样运气影响。

### 2.2 观测与动作的时序对齐

本仓库的默认且经过测试的契约是：

```text
obs[t] -> action[t+1 : t+1+H]
action_offset_steps = 1
```

因此 `chunk[0]` 是下一帧的绝对关节位置，不重复当前 proprio。旧 checkpoint
若没有记录 offset，则按历史 `offset=0` 解释；两种 checkpoint 不得直接混用或
resume。任何修改必须同时更新数据测试、checkpoint provenance 和 rollout wrapper。

### 2.3 训练与推理一致性

发布前逐项确认：

- camera keys、图像尺寸、裁剪和归一化一致；
- rollout 调用 `model.eval()`，且关闭训练增强；
- action stats、action offset、action horizon 和 execute horizon 显式保存；
- CFM ODE 或 DDIM 推理步数来自 bundle，不在 rollout 端静默覆盖；
- 如果启用 EMA，验证、export 和 rollout 必须明确加载 EMA 权重。

### 2.4 固定评估协议

- 把物体位姿、朝向和随机种子作为版本化评估集合。
- 初筛每个 checkpoint 至少 20 次 rollout；候选接近时扩大到 50 次或更多。
- 报告成功次数、总次数和不确定性，不只报告百分比。
- 保存 failure stage：`approach`、`close`、`lift`、`timeout`、`simulator_error`。
- 评估配置中不得保留影响执行的 `pending` 路径或阈值。

## 3. 当前 A2D 450GB 基线预算

当前数据版本的已解析训练规模为：

```text
episodes                    1,090
train episodes                981
validation episodes           109
基础训练窗口              146,324
transition 窗口            16,520
transition factor                2
有效训练样本              162,844
validation 样本             16,323
batch size                       32
steps/epoch                    5,089
```

完整五轮基线：

```text
epochs                         5
total steps               25,445
warmup steps                1,272
sample presentations      814,220
base LR                      1e-4
schedule        linear warmup + cosine
```

可用设备级 Skill 复算：

```bash
python ~/.codex/skills/plan-training-run/scripts/plan_training.py \
  --samples 162844 \
  --batch-size 32 \
  --epochs 5 \
  --warmup-ratio 0.05
```

## 4. 技巧清单与当前状态

| 技巧 | 解决的问题 | 当前状态 | 当前基线 |
|---|---|---|---|
| per-dim min/max | arm/hand 数值范围不同 | 已实现 | `[-1,1]` |
| next-action offset | 避免输出重复当前状态 | 已实现 | offset 1 |
| fixed split/stats digest | 防止数据漂移和泄漏 | 已实现 | manifest + SHA |
| transition oversampling | 关键动作窗口稀少 | 已实现，待消融 | factor 2 |
| RGB augmentation | 光照和轻微视角变化 | 已实现，待消融 | 轻量颜色与裁剪 |
| frozen ViT | 降低算力和过拟合风险 | 已实现 | backbone LR 0 |
| ViT 0.1x LR | 允许视觉特征适应任务 | 已完成五轮 offline 对比，待 rollout | backbone LR `1e-5` |
| linear warmup | 抑制早期不稳定 | 已实现 | 1,272 steps |
| cosine decay | 后期细化参数 | 已实现 | 当前衰减到 0 |
| gradient clipping | 限制异常梯度 | 已实现 | global norm 1.0 |
| watchdog | 检测卡死和未捕获异常 | 已实现 | timeout 300 s |
| EMA | 平滑权重与动作输出 | 已实现 raw/EMA 双权重保存与验证 | decay 0.999 |
| 双 best checkpoint | loss 与 MSE 可能分叉 | 已实现 | 分别按 val loss / action MSE |
| per-dim action error | 定位 arm/hand 弱项 | 待实现 | 无 |
| 固定 rollout 闭环 | 验证实际任务效果 | 协议已有，待完整执行 | n=20 起 |

## 5. 学习率、EMA 与 batch size

### 5.1 学习率

当前推荐起点：

```text
head LR       = 1e-4
frozen ViT    = 0
0.1x ViT LR   = 1e-5
optimizer     = AdamW
betas         = (0.9, 0.95)
weight decay  = 1e-4
```

确定 baseline 后，再做 `{3e-5, 1e-4, 3e-4}` 三点扫描。三组必须使用相同
数据、seed、batch、总 steps、warmup 比例和评估点。早期 spike 先检查 clip 前后
梯度；下降近似直线且过慢才支持“LR 偏小”的判断。

### 5.2 EMA

EMA 作为不参与前向、反向或 optimizer 更新的影子权重，默认从训练开始按下式累积：

```text
ema = decay * ema + (1 - decay) * weight
```

当前默认 `decay=0.999`。checkpoint 同时保存 raw/EMA 权重，validation 同时记录
raw/EMA 指标，并分别保存 `best_action_mse.ckpt` 与
`best_ema_action_mse.ckpt`；bundle 导出必须显式选择 `raw` 或 `ema`，并在
manifest 记录实际变体。
EMA 本身不改变主权重训练轨迹，但 raw/EMA 的 rollout 结果仍需分别报告。

### 5.3 Batch size

不要只因显存有余量就直接把 batch 改为 256。改变 batch 会同时改变 optimizer
更新次数和梯度噪声。先用 100–200 steps 测量候选 `32/64/128` 的：

- samples/s 和 seconds/step；
- 峰值显存与持续 GPU 利用率；
- loss、梯度和稳定性。

改变 batch 后重新计算 steps、warmup 和样本曝光量。LR 随 batch 放大只是待验证假设，
不得当作固定线性规则。受控消融期间保持 batch 不变。

## 6. 调度、checkpoint 与停止规则

- Warmup 通常从总 steps 的 5% 起步；当前 1,272 steps 位于合理区间。
- 当前 trainer 的 cosine 终点为 0。完成后如需续训，应建立新的第二阶段 schedule，
  不得直接增加原 run 的 epochs。
- 将非零终点做成配置项后，可消融 `min_lr_ratio={0,0.01,0.1}`。
- 当前同时保存 `best_val_loss.ckpt`、`best_action_mse.ckpt` 和 `latest.ckpt`；
  `best.ckpt` 保持为 `best_val_loss.ckpt` 的兼容路径。
- 达到最小预算后，若验证指标连续两次改善低于 1%–2%，且 rollout 不再改善，
  可以停止；这是一条启发式规则，不是数学定律。
- train loss 不能单独决定停止，尤其是在 oversampling 后。

## 7. Transition oversampling

当前 factor 2 的行为是：把未来 16 步中含 arm keyframe 的窗口额外加入一次训练
索引，不复制原始 HDF5，也不增加真实轨迹时长。

```text
原始 transition 占比    16,520 / 146,324 = 11.29%
factor 2 后占比          33,040 / 162,844 = 20.29%
```

验证集保持 factor 1，以真实分布评估。正式消融顺序：

1. 随机抽查至少 20 个标记窗口，确认阈值识别的确是关键动作。
2. 固定其他变量，训练 factor `1/2/3`。
3. 同时看 `val_keyframe_loss`、`val_continuous_loss` 和 rollout。

| 现象 | 决策 |
|---|---|
| keyframe loss 降且 rollout 升 | oversampling 有效，可考虑更高 factor |
| keyframe loss 降但 rollout 持平 | 离线收益未转化，优先保留温和设置 |
| continuous loss 升且平稳段抖动 | oversampling 过强，回退 factor |

## 8. 图像增强

只增强部署时可能真实出现的变化：

- 保留轻量亮度、对比度、饱和度、色相和约 ±5% 的裁剪；
- 谨慎使用大幅裁剪和平移，它们会破坏图像坐标与动作的空间对应；
- 不使用水平翻转和旋转，避免改变左右臂与重力方向语义；
- 训练与 rollout 的 resize、相机顺序和像素归一化必须共用合同。

## 9. 方法特定消融

### CFM

- 保持 `t ~ Uniform(0,1)` 作为基线。
- 推理 ODE steps 消融 `{5,10,20}`；这是推理实验，通常不需要重新训练。
- 同时测 RTX 4090 上的延迟和 rollout 成功率。

### Diffusion

- DDIM steps 消融 `{8,16,32}`，同时记录控制延迟。
- 同时评估 raw 与 EMA 权重，并用固定 rollout 判断最终部署变体。

### RS-IMLE

- IMLE loss 只能在同方法内比较，不能与 CFM/Diffusion loss 绝对值横比。
- checkpoint 更依赖 `val_sample_action_mse` 和 rollout。

方法对比必须统一 backbone、数据、action horizon、训练预算、seed 和 rollout 协议。

## 10. 每个实验的监控与产物

最低记录要求：

- train/validation aggregate、continuous、keyframe loss；
- `val_sample_action_mse`；
- head/backbone LR；
- clip 前后 grad norm、被 clip 的 step 比例；
- 13 个动作维度各自的预测误差；
- 固定 validation 样本的预测/真值曲线；
- heartbeat、failure report 和非零失败退出码；
- Git commit、dirty diff、resolved config、数据/split/stats SHA；
- checkpoint SHA 和最终 rollout `successes/n`。

W&B online/offline 是传输策略，不得改变本地 `metrics.jsonl`、checkpoint 和 provenance
作为事实源的要求。仪表盘延迟或同步失败不能导致训练产物丢失。

## 11. 推荐实验顺序

严格按以下顺序推进，上一层没有结论时不堆叠下一层技巧：

1. 正确性审计：归一化、offset、相机、bundle 和固定 rollout 集。
2. 完成 Frozen 与 0.1x ViT 的同预算五轮对比。（2026-07-21 已完成）
3. 固化 checkpoint/config/code/data SHA，并完成每组至少 20 次 rollout。
4. 选择 backbone baseline，再做 head LR 三点扫描。
5. 在最佳 LR 上做无 EMA/有 EMA 消融。
6. 审计关键窗口，再做 oversampling factor `1/2/3`。
7. 对同一 checkpoint 做 ODE/DDIM 推理步数与延迟消融。
8. 最后才比较 CFM、Diffusion 和 RS-IMLE。

如果两个候选的 rollout 差距落在小样本噪声范围内，应增加试验次数，而不是立即增加
新的训练变量。

## 12. 故障速查

| 症状 | 优先排查 |
|---|---|
| train 降、validation 不降 | 过拟合、数据量、增强、backbone 解冻范围 |
| offline 都降、rollout 很差 | offset、归一化、相机预处理、执行合同 |
| 动作抖动 | EMA 消融、chunk 执行和平滑、关键窗口权重 |
| 靠近物体但不闭手 | keyframe 标注、hand 维误差、oversampling |
| 偶发完全乱动 | 坏演示、梯度 spike、输出越界 |
| 平稳段变差、关键段变好 | oversampling 过强 |
| epoch 边界锯齿 | sampler、shuffle、LR 和日志聚合方式 |
| dashboard 空白但进程存活 | 检查本地 metrics、日志频率和同步状态 |

## 13. 文档维护规则

- 每个技巧加入主线前必须有问题描述、单变量实验和 rollout 证据。
- 每次正式实验留一页记录：假设、唯一变量、预算、结果、结论、产物 SHA。
- 已实现状态必须以代码和测试为依据；计划项不得写成已完成。
- 每季度做一次减法消融，删除不能稳定改善 rollout 的技巧。
