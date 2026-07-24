# Artifact Index

重要训练、评估与部署产物的本体不进入 Git；本文件只登记其位置、SHA-256 与代码血统。文件经过移动或复制后必须重新执行 SHA-256 校验，位置失效时应及时更新本索引。

| 日期 | 产物 | 存放位置 | SHA-256 / 校验入口 | 对应代码 | 备注 |
|---|---|---|---|---|---|
| 2026-07-22 | 1,090-episode CFM ViT 0.1× Level 0 测试包 | A800: `/share_data/zhangyurui/flow-matching-test-a2d-v2/rollout_artifacts/level0_prep_1090ep_cfm_vit_finetune_01x_step25445/`；psibot: `~/rollout_handoff/level0_prep_1090ep_cfm_vit_finetune_01x_step25445/`；单文件 tar: `~/rollout_handoff/a2d_1090ep_cfm_vit_finetune_01x_level0_prep_20260722.tar` | 目录内 `sha256sum -c SHA256SUMS`；bundle `760478e61468f58058e729fe25ae0ea005455aa31a1beb650158c368d73373e8`；外层 tar `f0eeb5d75bf92d93dc54b46d4c40c5b00ec474d04b49bb51646b0bc510c4cd4c` | HEAD `fd2b7512` + `training_source.patch` `11d8024a` | best epoch 4 / step 25445；offset 1；双 RGB 640×480；16×13；CFM 5 steps；固定参考复现 max abs error 0；best ckpt `2e355246…e5ea9` |
| 2026-07-15 | 66-episode CFM best bundle | A800: `/share_data/zhangyurui/flow-matching-test-a2d-v2/runs/a800_66ep_cfm_1507_20260715_141129/eval_bundles/eval_bundle_a800_66ep_cfm_1507_best_step4674.tgz` | `ebe1d1eb52de01e25d8c581c357c5bd71c5f5ccb30717c7a065812579335ecd8` | `6e5a561d839d4cfaf87306890ea9c3c7a6f74ab7` | best epoch 18 / step 4674 |
| 2026-07-15 | Level 0 自包含交付目录 | A800: `/share_data/zhangyurui/flow-matching-test-a2d-v2/rollout_artifacts/level0_prep_66ep_step4674/`；psibot: `~/rollout_handoff/level0_prep_66ep_step4674/` | 目录内执行 `sha256sum -c SHA256SUMS` | 模型 `6e5a561`；打包流程 `b843b0f` | 包含 bundle、固定输入/输出、A800 lock 与 SHA 清单 |
| 2026-07-15 | Level 0 固定参考输入 | 上述交付目录：`reference_input.npz` | `8b8e5c37da29de9c692ad0ca2e679c3890698da4f9409247fcbd10b1fec20e10` | `b843b0f` | 与参考输出配套，不可跨 bundle 复用 |
| 2026-07-15 | Level 0 固定参考输出 | 上述交付目录：`reference_output.json` | `ba87845c1974139274e4b4554706de83faec21483302cb6d574d6a8cd2ab60bb` | `b843b0f` | 固定 episode、frame 与 noise seed |
| 2026-07-15 | A800 环境锁文件 | 上述交付目录：`requirements.lock.a800.txt` | `b837b556084016ca3616e9fa90112bf690fc381d007698f4c58c615fd06f5475` | `b843b0f` | Level 0 环境对齐依据 |
| 2026-07-15 | 6 条 val 关键点可视化 | psibot: `/home/psibot/Downloads/flow-matching-test/outputs/a800_66ep_cfm_1507_val_keypoints/` | 图 `c232393a3d28bb2ec3eb2a1e9c3487e4432ad2c69ae53b3f847c7e78e6547ddf`；JSON `f973abb010144a0a2e4b4ad35644e0a66ae1572559fd2d520fb6ac5e3fc564cd` | `333f3b5` | 3 条 arm-pregrasp 弱位姿留给 rollout 裁决 |
| 2026-07-15 | 66-episode CFM W&B run | `https://wandb.ai/z1135783608-psibot/a2d-flow-matching/runs/83vd821c` | W&B run id: `83vd821c` | `6e5a561` | 训练曲线与在线指标的权威查看入口 |
| 2026-07-16 | 66-episode RS-IMLE best Level 0 自包含交付目录 | A800: `/share_data/zhangyurui/flow-matching-test-a2d-v2/rollout_artifacts/level0_prep_66ep_imle_step6150/` | 目录内执行 `sha256sum -c SHA256SUMS`；bundle `992a26752e3f0c13de8360e1f486665e4f04a5c8a57782fb8a30eda2dd4d1e83` | bundle `9f8e60a`；训练 `bb1b0b6` | best epoch 24 / step 6150；固定 val[0] frame 0、seed `20260715`；W&B: `giif7fgx` |
| 2026-07-16 | 66-episode Diffusion best Level 0 自包含交付目录 | A800: `/share_data/zhangyurui/flow-matching-test-a2d-v2/rollout_artifacts/level0_prep_66ep_diffusion_step7380/` | 目录内执行 `sha256sum -c SHA256SUMS`；bundle `4fe7012e72092ff9cd402d5fe6176b03add94653810855db342fa561eb099b85` | bundle `9f8e60a`；训练 `bb1b0b6` | best epoch 29 / step 7380；DDIM clamp 已纳入 bundle；固定 val[0] frame 0、seed `20260715`；W&B: `9gwrd6a8` |

## 登记规则

- 文件类交付物必须登记 SHA-256；目录类交付物必须提供自身的校验清单或明确的校验入口。
- “对应代码”填写实际产生该产物的 `git_sha`，不得用当前 HEAD 替代历史血统。
- W&B 保存训练曲线；A800、COS 或交付目录保存产物字节；Git 仅保存本索引和生成方式。
- 新增重要产物时，在同一个 PR 中更新本文件和 `CHANGE.md`，不提交产物本体。
- CFM、RS-IMLE 与 Diffusion 的训练 loss 和 sample MSE 仅用于各自的数值健康检查，不得跨 policy 排名；最终比较以相同 rollout 协议下的成功率为准。
