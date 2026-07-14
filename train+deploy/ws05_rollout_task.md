# 任务:在 ws-05 (RTX 4090) 上完成 A800 训练产物的 Level 0~3 rollout 验收

## 背景与目标

A800 上已用 6 条完整 lift episode 训出首个带完整 provenance 的 checkpoint(step 460)。
本任务目标:在 ws-05 上完成四级验收协议的 Level 0~3,证明「A800 训练 → bundle 部署 → Isaac Sim 闭环」链路正确,
最终标准:**模型在训练位姿上能完成抓取**。Level 3 通过后才启动 350GB 全量训练(不在本任务范围)。

验收协议全文见仓库 `train+deploy/ROLLOUT_DEPLOYMENT_PLAN.md`。

## 关键事实(执行时校验用,不可凭记忆改动)

| 项目 | 值 |
|---|---|
| 代码仓库 | `git@github.com:alex7537/flow-matching-test-a2d-v2.git`(Private) |
| 训练代码 commit | `c03a1de7912915a5cf54e9018b34a05a94ad1663` |
| 封版 tag | `v0.1-pipeline-frozen` |
| A800 训练目录 | `/share_data/zhangyurui/flow-matching-test-a2d-v2/runs/a800_6episode_full_20260713_205141` |
| 部署产物 | 上述目录 `eval_bundles/` 下 **step460 的 tgz**(不是裸 best.ckpt) |
| best.ckpt SHA-256 | `9a15e2db1109fe269908af8c777705e4b251fe55cb8f50b11e8432fc291f3eb1` |
| stats digest | `27f6d3c1d844...9959046aa`(完整值见 bundle manifest) |
| 训练环境 | torch 2.4.1+cu121 / timm==0.9.16 / fp32 |
| ws-05 | RTX 4090 24G, Driver 570.195.03, CUDA 12.8(兼容,无版本问题) |
| 训练指标参考 | train sample-action MSE 0.00812 / flow loss 0.103 / keyframe 0.191 / continuous 0.073 |

## 通用纪律(全程适用)

1. **ws-05 是共享机器**:所有安装限定在项目路径下的独立 venv(`--system-site-packages` 可选,不写任何全局配置,
   不 conda install,不改 `~/.bashrc`、`~/.gitconfig`,git 身份用仓库级 `git config`,不带 `--global`)。
2. **推理时段可暂停采集任务**(已获机主同意),但暂停/恢复要和 qingyangli 的采集进程明确交接,不 kill 不属于本任务的进程。
3. **只读原则**:采集管线的代码与场景资产只读引用或 fork 副本后修改,不在原目录改动。
4. **每级验收产出一份报告文件**(`LEVEL{N}_REPORT.md` 或 json),落在 rollout 产物目录,含全部数值与通过/失败判定。
5. **任一校验失败即停**:停下来记录现象并汇报,不要「重新生成一份继续」——尤其 stats/哈希类失败意味着传输或环境问题,绕过等于埋雷。

## 阶段 0:环境与产物准备

### 0.1 A800 侧(如尚未完成)
- 打包上传到 COS:step460 bundle tgz + 其 SHA-256 记录、`requirements.lock.a800.txt`、
  **同 seed 参考输出**(任选一条训练 episode 的一帧 obs,固定 noise seed 跑一次完整推理,把输出的 13 维 action 存为 json,连同所用 episode/帧号/seed 一起记录)。

### 0.2 ws-05 侧环境
```bash
# 项目目录自定,例如 ~/rollout_a2d 或数据盘路径
mkdir -p $ROLLOUT_ROOT && cd $ROLLOUT_ROOT
python -m venv venv && source venv/bin/activate
# torch:先检查系统/conda base 是否已有兼容版本(>=2.x, cuda 可用);
# 若无,pip 装 torch==2.4.1+cu121(与 A800 对齐)
pip install timm==0.9.16 h5py numpy'<2' pyyaml Pillow
pip check
python -c "import torch,timm; print(torch.__version__, torch.cuda.is_available(), timm.__version__)"
# 代码:git clone(只读 deploy key 或 HTTPS token),checkout c03a1de
```

### 0.3 产物落地校验
```bash
# COS 下载 bundle tgz → sha256sum 必须 == A800 记录值,不一致即停
```

## Level 0:部署校验(不需要 Isaac,可立即执行)

1. **policy_wrapper 加载 bundle**,预期:
   - ✅ 通过:bundle 内 5 文件 SHA-256、stats_digest、分段版本
   - ⚠️ 允许:torch/CUDA 版本差异提示(记录具体差异值)
2. **同 seed 数值一致性**(Level 0 核心):
   - 用 A800 参考输出的同一 episode/帧/seed,在 ws-05 跑同样的完整推理(编码→CFM 采样→反归一化)
   - 13 维输出与参考值逐维对比,**通过标准:atol ≤ 1e-4**
   - 若超差:先设 `torch.backends.cuda.matmul.allow_tf32=False` 与
     `torch.backends.cudnn.allow_tf32=False` 重试(4090 默认 TF32 行为可能与 A800 不同);记录两种设置下的 atol
3. **越界哨兵触发测试**:构造一个反归一化后超出 train min/max ±10% 外延的假输出,确认硬失败路径真实触发
4. 产出 `LEVEL0_REPORT`:校验结果 + atol 数值 + 哨兵触发记录 + 环境版本对照表

## Level 1:观测对齐(需要 Isaac Sim + 采集场景资产)

前提:确认采集管线的 Isaac 场景/USD/相机配置在 ws-05 上的位置与访问权限(它就是采集任务用的那套,优先原样复用)。

1. 将机器人与物体摆到某条训练 episode(6 条之一)的初始状态
2. 渲染两路相机(rgb_head / rgb_right_hand),与该 episode 第 0 帧训练图像对比
3. 产出:训练帧、渲染帧、叠图、差分图、SSIM、关键点重投影误差
4. 阈值不用跨渲染器通用值:本次对齐即建立标定基准,阈值连同场景版本一起存档
5. 重点排查项:相机内外参、分辨率、视野、色彩空间(线性 vs sRGB)
6. 产出 `LEVEL1_REPORT` + 对照图存档

## Level 2:GT 回放 + 物理参数校准

1. 物理参数(质量/摩擦/solver/PD)**直接从采集配置读取**填入 `run_metadata.json`(当前为 null 的字段),来源路径记入报告
2. 选 6 条中的 5 条,把 executed action 序列按采集端同款执行语义在 Isaac 里回放:
   - arm 按 setpoint 推进(与采集 client 的 execute 阶段一致)
   - hand 逐帧下发
   - 验证:能完成抓取 + lift 后物体不掉 + 三阶段判据函数在已知成功轨迹上判「成功」
3. **留出验证**:第 6 条 episode 不参与任何参数调整,单独回放验证,防止物理参数对特定轨迹过拟合
4. 若回放不成功:调物理参数直至 GT 可复现,最终参数写入 metadata 并注明「经 GT 回放校准」
5. 产出 `LEVEL2_REPORT`:每条回放的三阶段判定 + 最终物理参数快照 + 留出条验证结果

## Level 3:训练位姿闭环(本任务的通过标准)

改造方式:**复用采集 client 主循环,将 plan 阶段(sampler + cuRobo 候选规划)替换为 policy 推理**,
execute 阶段原封不动(与数据生成语义一致)。lift 段:当前模型为路径 B(自主 lift,lift 在 action 里),
故 server vertical lift primitive 停用,由 policy 输出的 chunk 驱动全程。

固定配置:
- `execute_horizon=16`(整段开环执行,先不做 receding horizon;16/8/4 对照属于 Level 4,不在本任务)
- 物体摆放:6 个训练位姿逐一测试
- 每个位姿采样 **5 次**(CFM 随机采样,单次失败可能只是抽到差的模式)

**通过标准(定量,写死):≥ 4/6 个训练位姿达成「5 次采样中 ≥ 3 次完成三阶段成功」。**
三阶段判据:抓取位姿到位精度 / hand 闭合成功 / lift 后物体不掉。

每次 rollout 落盘 `run_metadata.json`(Isaac/torch/timm 版本、dt、chunk、execute_horizon、物理参数快照——代码已支持)。
产出 `LEVEL3_REPORT`:6×5 结果矩阵 + 逐段失败归因(失败卡在哪一阶段)+ 通过/失败判定。

## 失败时的归因指引(供排查参考,不要跳级猜测)

- Level 0 数值超差 → TF32/cudnn 设置、torch 版本、图像预处理路径差异
- Level 1 差异大 → 相机外参/色彩空间,先解决再往下,纯视觉模型对此极敏感
- Level 2 GT 回放失败 → 执行语义(控制模式/增益)与采集端不一致,或物理参数错,**与模型无关**
- Level 3 失败但 0~2 全绿 → 闭环特有环节:推理端图像预处理与训练 Dataset 是否逐步一致(resize 插值、归一化顺序)、proprio 读取与归一化、chunk 执行节奏
- 预期管理:本模型仅 6 条数据,训练位姿上应能复现抓取(过拟合模型的本分);若 Level 3 通过,顺带在空档跑 Level 4 的 182 网格作为基线(非阻塞、非必须)

## 汇报要求

每完成一级,汇报:报告文件路径 + 关键数值(Level 0: atol;Level 1: SSIM/重投影;Level 2: 回放成功数+最终物理参数;Level 3: 6×5 矩阵)。
任何一级失败,汇报现象与已排查项后暂停,等待决策。
