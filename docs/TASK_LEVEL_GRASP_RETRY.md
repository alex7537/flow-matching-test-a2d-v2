# Task-level Grasp Retry

本功能位于 rollout 编排层，不修改 CFM、action head 或 bundle。原有 policy 仍负责 chunk-level closed loop：

```text
observation -> predict action chunk -> execute -> replan
```

外层新增 task-level 状态机：

```text
ATTEMPT
  -> APPROACH
  -> VERIFY_GRASP
       | stable contact -> LIFT
       | no contact / contact lost / lift timeout
       v
     RECOVER
       -> open hand
       -> retreat arm to safe pregrasp
       -> settle
       -> clear policy history
       -> change sampling seed
       -> next ATTEMPT
```

## Retry triggers

| 原因 | 判定 |
|---|---|
| `approach_timeout` | 在指定 action steps 内没有进入 approach 距离与旋转阈值 |
| `close_timeout` | 已 approach，但没有在预算内形成稳定多指接触 |
| `contact_lost_after_close` | 已确认抓住，随后连续若干步失去所需接触数 |
| `lift_timeout` | 已确认抓住并保持接触，但物体没有在预算内达到 lift 成功高度 |
| `attempt_budget_exhausted` | 单次 attempt 的 chunk 预算耗尽且尚未抓住 |
| `lift_attempt_budget_exhausted` | 单次 attempt 结束时已抓住但尚未完成 lift；有剩余 attempt 时恢复重试 |
| `task_step_budget_exhausted` | 所有 attempts 共用的 policy action 总预算耗尽；不再恢复或重试 |

稳定接触和 lift 成功仍由 `ThreePhaseChecker` 的 `close_contact_count`、`close_hold_steps`、`lift_height_m` 与 `lift_hold_steps` 决定。

## Recovery

恢复不会 reset 物体。目标默认使用当前 trial 的 `initial_joint_positions`，也可以显式配置13维 `recovery_joint_positions`。

恢复顺序固定为：

1. arm 保持当前实际位置，只把6维 hand 插值到恢复目标，先松开物体；
2. hand 保持打开，再把7维 arm 插值退回安全预抓取位置；
3. 在恢复目标保持若干步；
4. 清空视觉/proprio/action history；
5. sampling seed 按 attempt 增加 `seed_stride`，避免从相同噪声重复同一候选。

## Configuration

```yaml
execution:
  # 推荐先用较短 execute horizon 验证，例如 CLI --execute-horizon 4。
  max_chunks: 16
  task_grasp_retry:
    enabled: false
    max_attempts: 2
    approach_timeout_steps: 96
    close_timeout_steps: 64
    lift_timeout_steps: 96
    contact_loss_hold_steps: 5
    max_total_policy_steps: 512
    recovery_open_steps: 10
    recovery_retreat_steps: 30
    recovery_settle_steps: 5
    seed_stride: 100003
    recovery_joint_positions: null
```

这些 timeout 以已执行 action step 计数；若控制频率为30Hz，96/64/5步约为3.2/2.1/0.17秒。

## Result fields

`results.jsonl` 每个 trial 增加：

```text
attempt_count
retry_count
first_attempt_success
recovered_success
recovery_steps
total_control_steps
task_termination_reason
attempts[]
  - sampling_seed
  - retry_reason
  - retry_performed
  - approach/close/lift summary
```

汇总报告增加 first-attempt success、recovered success、retry rate、平均 attempts、平均 policy/recovery/total steps、retry reason 与 task termination reason 计数。

## Safety gate

启用 `enabled: true` 前必须完成：

- `initial_joint_positions` 或 `recovery_joint_positions` 是经过验证的张手安全预抓取位；
- 至少两个接触传感器路径有效，并验证接触计数方向正确；
- approach、close、lift timeout 使用单 trial 视频标定；
- 先只运行一个 `--trial-id`，确认张手发生在退臂之前；
- 检查 `max_total_policy_steps`，确保异常情况下任务必然终止；
- 再运行固定 seed 的 retry-off / retry-on 对照，分别报告首次成功率和恢复成功率。

当前代码和假环境路径已验证，但尚未在真实 Isaac 场景完成安全验收，因此默认保持关闭。
