# A2D Flow Matching 交付执行手册(封版 → Isaac Sim Rollout)

> 交付对象:负责 ws-05 (RTX 4090) rollout 验收的同事/agent
> 本手册是执行入口;逐级细节与判定标准以仓库内
> `train+deploy/ws05_rollout_task_v2.md` 为准,两者冲突时以任务书 v2 为准。

---

## 一、交付物清单(接收时逐项核对)

| # | 交付物 | 位置 | 校验 |
|---|---|---|---|
| 1 | 代码仓库 | `git@github.com:alex7537/flow-matching-test-a2d-v2.git`(Private) | 交接基线不得早于 `a65d3c9`;实际 HEAD 用 `git rev-parse HEAD` 记录,历史见 `CHANGE.md` |
| 2 | Level 0 完整交付目录 | psibot `~/rollout_handoff/level0_prep_66ep_step4674/`(已从 A800 拉取，含 bundle TGZ + 同 seed NPZ/JSON + lock + `SHA256SUMS`) | 整目录传输，每次落地只执行 `sha256sum -c SHA256SUMS` |
| 3 | 执行任务书 | 仓库 `train+deploy/ws05_rollout_task_v2.md` | 随代码 clone 获得 |

模型速览:CFM policy,66 条完整 lift episode 训练,best = epoch 18 / step 4674;
dataset `a2d_parallel_1507_rgb_v1`,stats digest `f34e7703…`;
已知弱点:6 条 val 中 3 条 arm-pregrasp 误差偏大(rollout 时重点观察,见任务书 Level 3)。

---

## 二、前置准备(执行前一次性完成)

### 2.1 访问权限
- [ ] GitHub 仓库只读 Deploy Key(在 ws-05 生成 `ssh-keygen -t ed25519`,公钥加到仓库 Settings → Deploy keys,私钥放 `/root/.ssh`,不落共享盘)
- [x] A800 → psibot 完整目录已传输并通过四项 SHA-256 校验
- [ ] psibot → ws-05 的文件通道(scp / 内网共享 / U 盘，只传上表单一目录)
- [ ] 与机主 qingyangli 约定推理时间窗(采集任务可暂停,交接要明确,不 kill 非本任务进程)

### 2.2 需要向采集侧确认的输入(缺任一项则对应 Level 明确报错停止,禁止手调假参数)
- [ ] Isaac 场景资产 / USD 位置与版本(采集管线同款)
- [ ] 相机内外参标定文件
- [ ] 物理参数来源(质量/摩擦/solver/PD 的采集配置)
- [ ] 采集管线运行形态(原生 / 容器)——决定环境搭建路线

---

## 三、执行序列(按序推进,每级出报告后再进下一级)

### Step 1 — 传输与落地校验
```bash
# psibot 已落地并验证的唯一发件源
cd ~/rollout_handoff/level0_prep_66ep_step4674
sha256sum -c SHA256SUMS       # TGZ / NPZ / JSON / lock 四项已全绿

# 将上述整个目录传入 ws-05；ws-05 落地后再执行一次
cd <ws05-path>/level0_prep_66ep_step4674
sha256sum -c SHA256SUMS       # 四项必须全部 OK，任一失败即停
```

### Step 2 — 环境(二选一,推荐容器)
```bash
# 路线 A:Docker policy-server(推荐,隔离最彻底)
#   基础镜像 pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
#   + pip install timm==0.9.16 h5py "numpy<2" pyyaml Pillow
#   + git clone 仓库(Deploy Key);bundle 以 volume 挂载,不打进镜像
# 路线 B:项目路径下独立 venv(共享机纪律:不写全局配置、不 conda install)
# 两条路线都以 requirements.lock.a800.txt 为版本基准,timm 必须 0.9.16
# 推理基线优先 checkout bundle manifest 的 git_sha:
git checkout 6e5a561d839d4cfaf87306890ea9c3c7a6f74ab7
python -c "import torch,timm; print(torch.__version__, torch.cuda.is_available(), timm.__version__)"
```

### Step 3 — Level 0:部署强校验(不需 Isaac,可立即执行)
```text
1) policy_wrapper 加载 bundle:
   通过项:文件哈希 / stats_digest=f34e7703 / segmentation 版本 / policy_type=flow_matching
   记录项:torch/CUDA/timm 版本差异(写入报告,不得忽略)
2) 同 seed 数值一致性:用参考包 NPZ 输入 + 记录的 seed 推理,
   16×13 输出 vs 参考 JSON,通过标准 atol ≤ 1e-4
   (超差:先关 TF32 与 cudnn TF32 重试,两种设置的 atol 都记录)
3) 越界哨兵:构造超 train min/max ±10% 的假输出,确认硬失败触发
→ 产出 LEVEL0_REPORT
```

### Step 4 — Level 0.5:USD mimic 验证(hand rollout 解锁条件)
```text
RuiYan URDF 权威耦合(勿用轨迹回归替代):
  1_3=1.675×1_2; 2_2=2_1; 3_2=3_1; 4_2=4_1; 5_2=5_1
  6 维控制顺序:1_1, 2_1, 3_1, 4_1, 5_1, 1_2
步骤:确认 USD 导入保留 mimic → 无接触状态逐关节单位输入 →
     断言 11 物理关节响应比例(逐条单测)
失败回退:按 URDF 比例构造显式耦合层(真值取 URDF 声明),单测固化后解锁
→ 产出比例实测表
```

### Step 5 — Level 1:观测对齐(需 Isaac + 采集场景资产)
```text
val episode 初始状态摆位(robot-root local 坐标)→ 渲染两路相机 vs 训练 frame 0
产出:训练帧/渲染帧/叠图/差分/SSIM/重投影误差;阈值随场景版本存档
排查项:内外参、分辨率、FOV、色彩空间(linear vs sRGB)
→ 产出 LEVEL1_REPORT + 对照图
```

### Step 6 — Level 2:GT 回放与物理标定
```text
物理参数从采集配置填入 run_metadata.json 的 null 字段(来源路径入报告)
5 条 episode 按采集执行语义回放(arm setpoint 推进 / hand 逐帧)
→ 抓取成功 + lift 不掉 + 三阶段判据判"成功"
第 6 条留出,不参与任何参数调整,单独回放验证
→ 产出 LEVEL2_REPORT(回放矩阵 + 最终物理参数快照)
```

### Step 7 — Level 3:val 位姿闭环(最终通过标准)
```text
改造:fork 采集 client,plan 阶段替换为 policy 推理;execute 阶段不动;
     lift 由 policy chunk 驱动(路径 B),server lift primitive 停用
配置:execute_horizon=16 固定;6 条 val 物体位姿(robot-local)× 每位姿采样 5 次
通过标准:≥ 4/6 位姿达成「5 采 ≥ 3 次三阶段成功」
重点观察:3 条 arm-pregrasp 弱位姿的成败与失败段归因
→ 产出 LEVEL3_REPORT(6×5 矩阵 + 逐段归因 + 判定)
```

### Step 8(可选,非阻塞)— Level 4:182 网格泛化地图
Level 3 通过后空档执行;网格按 robot-local 坐标重新生成,
19 个训练占用格为主指标、6 个空格入 sparse 组;execute_horizon 16/8/4 对照。

---

## 四、纪律红线(全程适用)

1. 任一校验失败即停并汇报,禁止"重新生成一份继续"。
2. 资产/标定/物理参数缺失 → 明确报错停止,禁止手调假参数产出结论。
3. 采集管线代码与资产只读;改造一律 fork 副本。
4. 共享机隔离:一切安装限定容器/项目 venv,不写全局配置。
5. 每级报告落 rollout 产物目录,与 run_metadata.json 同级,进入 provenance 链。

## 五、汇报格式

每级完成 → 报告路径 + 关键数值:
L0: atol | L0.5: 比例表 | L1: SSIM/重投影 | L2: 回放矩阵+参数 | L3: 6×5 矩阵。
失败 → 现象 + 已排查项,暂停待决策(归因指引见任务书 v2 末节)。
