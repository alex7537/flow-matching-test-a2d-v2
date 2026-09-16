# 开源参考与本项目取舍（2026-09-16）

| 官方参考 | 借鉴 | 本项目落点 |
|---|---|---|
| [LeRobot eval入口](https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_eval.py) 与 [自定义policy说明](https://github.com/huggingface/lerobot/blob/main/docs/source/bring_your_own_policies.mdx) | 统一策略接口，独立评测入口与按任务结果 | CFM/DP/IMLE/Wan/RDT后端适配与控制层分离；V0直接接已有gRPC结果 |
| [ManiSkill示范学习评测](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html) | success_once与success_at_end分开，保持重置及仿真后端一致 | 当前“曾抓起”是primary；最终保持不偷偷替代主指标；seed不替代完整初态 |
| 本地robot-benchmark-loop skill | 身份冻结、原子逐轮记录、覆盖资格、invalid分离 | manifest、snapshot、aggregate、decision和next_experiment |

这些设计不能直接证明我们任务的科学有效性。机器人、动作语义、相机和资产仍需本项目标定；不为使用benchmark而迁移到另一仿真器。

原训练远端loop分支：`agent/add-robot-ml-loop-instance@c73247ce50da4500ad7f5a8d7d3ca5c894ce0ce5`。原账本的pending不伪造补成passed；本V0用真实本机证据建立独立cycle，之后再衔接原lifecycle。
