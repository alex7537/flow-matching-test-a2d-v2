# 任务 v2:在 ws-05 (RTX 4090) 完成 66-episode CFM 模型的 Level 0~3 rollout 验收

> 本版取代旧 ws05_rollout_task.md。变更:目标 bundle 换为 66 条模型;`6→11` 改为 USD mimic 验证;
> 考卷/位姿统一 robot-local 坐标;新增"3 条弱 val 位姿"重点测试项。

## 背景与通过标准

A800 已完成 66 条完整 lift episode 的 CFM 训练(30ep,val 健康收敛)并导出 bundle。
本任务:在 ws-05 上按 Level 0→3 验收「训练 → 部署 → Isaac 闭环」。
**Level 3 通过标准(定量):≥ 4/6 个 val 物体位姿达成「5 次采样中 ≥ 3 次三阶段成功」。**
三阶段判据:抓取位姿到位精度 / hand 闭合成功 / lift 后物体不掉(policy 自主 lift,路径 B,外部 lift primitive 停用)。

## 关键事实(执行时逐项校验,不可凭记忆)

| 项目 | 值 |
|---|---|
| 代码仓库 | `git@github.com:alex7537/flow-matching-test-a2d-v2.git`(Private,ws-05 用只读 Deploy Key) |
| 代码 commit | 推理基线优先 checkout bundle manifest 记录的 `6e5a561d839d4cfaf87306890ea9c3c7a6f74ab7`；若使用更新 main，必须记录实际 sha 并完成同 seed 对账 |
| 部署产物 | A800 `runs/a800_66ep_cfm_1507_20260715_141129/` 下 best bundle TGZ |
| TGZ SHA-256 | `ebe1d1eb52de01e25d8c581c357c5bd71c5f5ccb30717c7a065812579335ecd8` |
| best checkpoint | epoch 18 / step 4674(确认 bundle manifest 内 ckpt step=4674) |
| dataset / stats | `a2d_parallel_1507_rgb_v1` / stats digest `f34e7703…` |
| 训练环境 | torch 2.4.1+cu121 / timm==0.9.16 / fp32 |
| ws-05 | RTX 4090 24G, Driver 570.195.03(兼容 cu121) |
| 模型已知弱点 | 6 条 val 中 3 条 arm pregrasp 误差偏大(0.078~0.192 rad²);hand 预测稳定 |
| 位姿坐标系 | **一切物体位姿使用 robot-root local 坐标**;覆盖区 X 0.592~0.887m、Y -0.318~0.050m |

## 通用纪律

1. ws-05 为共享机(qingyangli 的采集任务在跑):一切安装限定项目 venv/容器,不写全局配置,不 conda install;推理时段暂停采集须与机主明确交接,不 kill 非本任务进程。
2. 采集管线代码与场景资产只读;需要改造(Level 3 的 plan→policy 替换)一律 fork 副本。
3. 每级产出 `LEVEL{N}_REPORT`(md/json)落 rollout 产物目录,含全部数值与判定。
4. 任一校验失败即停并汇报,禁止"重新生成一份继续"。
5. 资产/标定/物理参数缺失时明确报错停止,禁止手调假参数产出正式结论。

## 阶段 0:准备

### 0.1 A800 侧(先做)
- 用 **本次 bundle** 生成同 seed 参考输出:任选一条 val episode 的 frame 0,固定 seed,
  完整推理(编码→CFM 5 步 Euler→反归一化),保存 `reference_input.npz`
  (实际模型输入) + `reference_output.json`(16×13 chunk + episode/frame/seed/环境版本)。
  (旧 step460 参考输出对应 6 条模型,不可复用。)
- 打包上传:bundle TGZ + SHA、参考输入 NPZ/输出 JSON、`requirements.lock.a800.txt`。
  通道:COS(凭据到位)或任何双方可达通道;落地后 `sha256sum` 必须等于上表值。

### 0.2 ws-05 环境(推荐 Docker policy-server 架构)
- 先确认采集管线运行形态(原生/容器)与 Isaac 场景资产位置——**这决定 Level 1~3 的 Isaac 侧形态**。
- policy 容器:基础镜像 `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` + `timm==0.9.16, h5py, numpy<2, pyyaml, Pillow` + 仓库代码;bundle 用 volume 挂载不打进镜像;镜像 tag 带 git sha。
- Isaac 侧保持采集原生环境不动;rollout client 经 localhost 调用容器内 policy(obs 进/chunk 出,接口序列化参考 msgpack+jpeg bytes)。
- 若容器路线受阻,回退项目 venv 方案亦可,纪律同上。

## Level 0:部署强校验(不需 Isaac,可立即执行)

1. TGZ SHA-256 比对 → 解包 → policy_wrapper 加载:
   - ✅ 应通过:bundle 内文件哈希、stats_digest=f34e7703、segmentation 版本、policy_type=flow_matching
   - ⚠️ 记录:torch/CUDA/timm 版本差异(不可习惯性忽略,写入报告)
2. 同 seed 数值一致性:加载 `reference_input.npz` 作为模型输入，按 JSON 记录的 seed 推理，
   16×13 输出 vs A800 `reference_output.json`，**atol ≤ 1e-4**。
   超差先关 TF32(`torch.backends.cuda.matmul.allow_tf32=False`、cudnn 同)重试并记录两种设置下 atol。
3. 越界哨兵:构造超出 train min/max ±10% 的假输出,确认硬失败真实触发。
4. 产出 `LEVEL0_REPORT`。

## Level 0.5(新增):USD mimic 验证——hand rollout 的解锁条件

RuiYan URDF 声明的耦合(权威规则,勿用轨迹回归替代):
```
1_3 = 1.675 × 1_2;  2_2 = 1.0 × 2_1;  3_2 = 1.0 × 3_1;  4_2 = 1.0 × 4_1;  5_2 = 1.0 × 5_1
6 维控制关节顺序:1_1, 2_1, 3_1, 4_1, 5_1, 1_2(与 hand2_pos 逐维对应,已在 8349 帧上验证全等)
```
步骤:确认 USD 导入保留 mimic → 无接触状态下对 6 个主动关节逐一做单位输入 →
断言 11 个物理关节响应满足上述比例(逐条单测)。
- 全部通过 → hand rollout 解锁。
- 任一失败(Isaac 导入器丢 mimic 是已知坑)→ 回退方案:按 URDF 比例构造显式耦合层
  (真值来自 URDF 声明,不做数据回归),单测固化后再解锁。
产出 mimic 验证报告(比例实测值表)。

## Level 1:观测对齐

1. 机器人+物体摆到某条 val episode 初始状态(robot-local 坐标换算)。
2. 渲染 rgb_head/rgb_right_hand vs 该 episode frame 0:输出训练帧、渲染帧、叠图、差分、SSIM、关键点重投影误差。
3. 阈值不跨渲染器通用:本次对齐即建立标定基准,阈值随场景版本存档。
4. 排查项:相机内外参、分辨率、FOV、色彩空间(linear vs sRGB)。
5. 相机配置必须来自采数资产/标定文件,禁止手调后作为正式依据。

## Level 2:GT 回放与物理标定

1. 物理参数(质量/摩擦/solver/PD)从采集配置读取填入 `run_metadata.json` 的 null 字段,来源路径入报告。
2. 任选 5 条 episode 按采集端执行语义回放 executed 轨迹(arm setpoint 推进、hand 逐帧),
   验证:抓取成功 + lift 后不掉 + 三阶段判据在已知成功轨迹上判"成功"。
3. **留出验证**:第 6 条不参与任何参数调整,单独回放。
4. 回放失败先归因执行语义/物理参数(与模型无关),校准后参数写入 metadata 注明"经 GT 回放校准"。

## Level 3:val 位姿闭环(本任务通过标准)

- 复用采集 client 主循环,plan 阶段(sampler+cuRobo)替换为 policy 推理;execute 阶段不动;
  lift 由 policy chunk 驱动,server vertical lift primitive 停用。
- 固定 `execute_horizon=16`(整段开环;16/8/4 对照属 Level 4,不在本任务)。
- 测试位姿:**6 条 val episode 的物体位姿**(robot-local),每位姿采样 5 次。
- **重点观察项**:3 条 arm-pregrasp 弱 val 位姿的成败与失败段归因——
  它们是"覆盖缺口 vs 定位精度 vs 多模态"三种解释的裁决样本;
  若弱位姿失败但多次采样中存在成功支,倾向多模态解释(MSE 有偏,任务实际可完成)。
- 每次 rollout 落 `run_metadata.json`(版本/dt/chunk/horizon/物理参数快照)。
- 产出 `LEVEL3_REPORT`:6×5 矩阵 + 逐段失败归因 + 通过/失败判定。

## 失败归因指引

- L0 数值超差 → TF32/cudnn/torch 版本/图像预处理路径
- L0.5 比例不符 → Isaac 导入器丢 mimic → 显式耦合层回退
- L1 差异大 → 外参/色彩空间,先解决再往下(纯视觉模型极敏感)
- L2 回放失败 → 执行语义或物理参数,与模型无关
- L3 失败而 0~2 全绿 → 闭环特有:推理端图像预处理与训练 Dataset 逐步一致性、proprio 归一化、chunk 执行节奏
- 预期:val 密集区位姿应能成功;弱 3 条允许失败但必须归因。Level 4 的 182 网格
  (robot-local 坐标重新生成,19 占用格为主指标、6 空格入 sparse 组)通过后空档跑,非本任务阻塞项。

## 汇报要求

每级完成汇报报告路径 + 关键数值(L0: atol;L0.5: 比例表;L1: SSIM/重投影;L2: 回放矩阵+参数;L3: 6×5 矩阵)。任何一级失败,汇报现象与已排查项后暂停待决策。
