# 部署与闭环推理概览

[English](DEPLOYMENT_OVERVIEW.md) | **简体中文**

本文串联 action-only 策略的训练产物、推理 bundle 与 Isaac rollout，依据仓库中的 exporter 和 wrapper 接口整理，不代表新完成了一次部署验收。

## 整体流程

```text
checkpoint + 训练配置 + normalizer
    → 导出 raw / EMA bundle
    → 校验文件哈希、数据血统、动作语义
    → 相机 RGB + 当前实际关节状态
    → 按训练约定预处理、构建历史
    → 生成归一化 action chunk
    → 反归一化、检查动作范围
    → 仿真执行指定前缀
    → 获取新观测并重新规划
```

## 导出与加载

[export_bundle.py](../flow_matching_test/export_bundle.py) 打包模型权重、推理配置、统计量、可选 split，以及 manifest。记录源 checkpoint SHA、选择准则、raw/EMA 变体、训练 step/epoch、数据版本、stats digest、环境和文件校验值。

[policy_wrapper.py](../rollout/policy_wrapper.py) 校验必需文件、schema、哈希、13维动作布局、stats digest 和动作语义，严格加载权重，调用 eval 模式，并构建约定的观测历史。

## 输入输出

默认输入是 head RGB、right-hand RGB，以及可选的13维实际关节状态。共享 Hybrid ViT 提取视觉 token，策略输出未来16步、每步13维的绝对关节位置序列。

V3标签为实际手臂7维加手部 commanded target 6维；观测仍为实际 arm/hand 状态。输出必须使用 bundle 自带统计量反归一化，不能按速度或增量解释。旧 executed-hand bundle 必须匹配自己的语义。

## 三种“步数”不要混淆

- prediction horizon：模型预测的动作长度，常见为16。
- execute horizon：本次实际执行前几步，再观测重规划；必须在1到chunk长度之间。
- sampling steps：如CFM的5步Euler，是生成一个chunk的数值积分次数，不是执行5个机器人动作。

缩短执行 horizon 会改变重规划与动作衔接。相机顺序、resize、normalization、关节顺序与映射、action offset、控制时序、权重变体及采样设置都必须匹配。

## 验收与证据

先检查bundle完整性及固定输入输出，再验证仿真接口，最后在明确初态和成功判据下统计成功/有效试验数、无效试验、失败阶段、延迟与视频。模型加载成功不等于抓取成功。

仿真集成见[部署仓库](https://github.com/alex7537/fk-issac-logistics)，访问可能需要权限。公开案例见[抓取展示分支](https://github.com/alex7537/flow-matching-test-a2d-v2/tree/docs/grasp-success-gallery)。

相关：[产物索引](../artifacts_index.zh-CN.md) · [公平对比协议](POLICY_COMPARISON_PROTOCOL.zh-CN.md)
