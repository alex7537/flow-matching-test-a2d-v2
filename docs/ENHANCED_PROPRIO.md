# Enhanced Proprio V3

本实验在 V3 RGB+proprio baseline 上增加动态状态上下文，不修改 13 维 action 语义、RGB encoder、CFM Transformer 或 condition token 数量。

## Observation contract

时刻 `t` 的 observation：

```text
proprio            = normalize(qpos[t])                       # 13
joint_delta         = scale(qpos[t] - qpos[t-1])               # 13
previous_action     = normalize(action[t])                     # 13
hand_tracking_error = scale(action[t,7:] - qpos[t,7:])         # 6
```

监督保持：

```text
action label = action[t+1 : t+17]
```

所以增强上下文只使用当前和上一帧，不读取未来 action label。因为需要 `t-1`，每个 episode 的首帧不再构成训练样本。

V3 `previous_action` 语义仍是：

```text
arm actual qpos(7) + hand commanded target(6)
```

rollout 每个控制步调用 `record_execution_feedback(action, actual_qpos)`，确保 arm context 使用 actual qpos、hand context 使用下发 target，且 `joint_delta` 始终是一帧差而不是两次 inference 之间的多帧差。

## Normalization

- `proprio`、`previous_action` 使用原 train min/max；
- `joint_delta` 使用 `2 * delta / state_span`，以0为中心；
- hand error 使用 state/action hand span 的逐维较大值缩放；
- 所有增强状态裁剪到 `[-3,3]`；
- 不新增或修改 dataset stats/split manifest。

## Model compatibility

现有13维 proprio 仍经过原 `proprio_proj` 生成一个 token。三个新 projection 的输出相加到该 token：

```text
proprio_token
+ joint_delta_proj(joint_delta)
+ previous_action_proj(previous_action)
+ hand_tracking_error_proj(hand_tracking_error)
```

新 projection 的 weight/bias 全零初始化，因此加载旧 best checkpoint 后，初始 policy 行为与旧模型一致。旧 checkpoint 允许且只允许缺少这六个参数；其他 missing/unexpected key 都会中止训练。

## Training plan

```text
source checkpoint    V3 RGB+proprio best raw
source epoch/step    46 / 323,172（日志从0计数）
train episodes       981
val episodes         109
effective train      219,040
val samples          17,755
steps/epoch          6,845
epochs               20
total steps          136,900
warmup               6,845（5%）
head peak LR         1e-5
ViT peak LR          1e-6
schedule             new warmup + cosine
init_from            best raw
resume_from          null
```

配置：`configs/a2d_450gb_v3_enhanced_proprio_cfm_a800_20ep_init_best.yaml`。

## Verification

- 本地目标测试：34 passed；
- 开发机 Python3.11 全套测试：43 passed；
- 完整 V3 Dataset tensor/预算预检：通过；
- 真实 best checkpoint 只缺六个预期 projection 参数；
- CUDA batch forward/backward 和新 projection 非零梯度：通过；
- 1 train batch + 1 val batch trainer smoke：通过；
- bundle/rollout enhanced context round-trip：通过。

科学验收仍需与原 V3 baseline 使用相同 rollout trial×seed 协议比较；离线 loss 不能单独决定提升。
