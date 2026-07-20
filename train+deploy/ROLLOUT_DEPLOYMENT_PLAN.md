# Rollout 部署方案:A800 云端训练 + psibot(4090)本地 Isaac Sim 评测

> 适用项目:`flow-matching-test` / `psi-policy`
> 数据合同:schema_version=2,双 RGB + 本体感知,13-DOF executed joint-position action chunk；原始数据无 timestamp，30Hz 仅为待采集侧确认的部署假设
> 更新日期:2026-07-13

---

## 1. 架构总览

```
┌─────────────────────────┐          ┌──────────────────────────────┐
│  云端开发机 (A800)        │          │  本地 psibot (RTX 4090)       │
│                         │          │                              │
│  · 数据预处理             │   COS    │  · Isaac Sim 4.5/5.x 容器     │
│  · flow matching 训练    │ ───────▶ │  · rollout harness           │
│  · 产出 eval_bundle/     │  (coscmd)│  · 固定评测网格 + 三阶段判据    │
│                         │          │  · 结果/视频/指标落盘          │
└─────────────────────────┘          └──────────────────────────────┘
```

分工原则:训练不需要渲染,A800 的算力和显存用在训练上;rollout 需要 RTX 渲染
(双 RGB 相机),必须在有 RT Cores 的卡上跑,4090 是理想选择。两台机器之间
**只传文件,不做在线通信**。

---

## 2. eval_bundle 规范(A800 侧产出)

每次训练产出一个自包含目录,杜绝 ckpt 与 stats 版本错配:

```
eval_bundle_<exp_name>_<step>/
├── ckpt.pt                  # 模型权重(建议只存 state_dict,不存 optimizer)
├── norm_stats.json          # train min/max + range_eps=1e-4,含 train episode 摘要
├── config.yaml              # 推理必需配置,见下
├── manifest.json            # 数据/分段/训练环境 provenance + 文件 SHA-256
└── README.md                # 一句话说明这个 ckpt 是什么实验
```

`config.yaml` 至少包含:

```yaml
model:
  vit_backbone: <名称/patch size/输入分辨率>
  cfm:
    num_inference_steps: 10        # 直线 CFM 的 ODE 步数
action:
  dim: 13
  chunk_size: <H>                  # 每次推理产出的 action chunk 长度
  offset_steps: 1                  # chunk[0] 对应观测后的下一帧动作
  execute_horizon: <h>             # 实际执行几步后重新推理(h <= H)
  target: executed_joint_position # arm2_pos(7)+hand2_pos(6),与训练一致
  range_guard_margin_ratio: 0.1    # 反归一化输出超出训练范围外延即硬失败
obs:
  cameras:
    - name: cam_top                # 与采数时的相机命名对齐
      resolution: [W, H]
    - name: cam_wrist
      resolution: [W, H]
  proprio_dim: <D>
  freq_hz: 30                      # 部署假设,不是 timestamp 验证结论
  freq_hz_status: deployment_assumption_unverified_no_dataset_timestamps
joint_order: [j0, j1, ..., j12]    # 显式列出,防止仿真端顺序错位
```

**A800 侧打包脚本已接入训练 checkpoint 回调**,并额外保存:

- `stats_digest` 与 `segmentation_version/thresholds`；
- Python、torch、CUDA build、cuDNN、timm、numpy、训练 GPU；
- rollout 环境期望值，包括 Isaac Sim 版本、dt、solver iterations、物体质量与摩擦；
- 所有 bundle 文件的 SHA-256，4090 加载时逐文件校验。

未知物理参数保持 `null`，禁止填写猜测值；首次计分 rollout 前从实际 USD、PhysX
材质和 Isaac Sim 环境补齐。

---

## 3. 同步流程(COS 中转)

A800 侧上传:

```bash
tar czf eval_bundle_fm_step50k.tgz eval_bundle_fm_step50k/
coscmd upload eval_bundle_fm_step50k.tgz psi-policy/eval_bundles/
```

psibot 侧下载:

```bash
coscmd download psi-policy/eval_bundles/eval_bundle_fm_step50k.tgz ~/bundles/
tar xzf ~/bundles/eval_bundle_fm_step50k.tgz -C ~/bundles/
```

走 COS 而不是直连 `scp` 的原因:不依赖跳板机/内网打通,断点续传,团队成员
(xuzhuoran、liyichao)也能取同一份 bundle 复现评测。

---

## 4. psibot 环境准备(一次性)

### 4.1 驱动与 Docker

```bash
# 驱动:535+ Production Branch(.run 安装器),已装可跳过
nvidia-smi   # 确认 4090 可见

# NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 4.2 Isaac Sim 容器

```bash
docker login nvcr.io        # 用户名 $oauthtoken,密码为 NGC API key
docker pull nvcr.io/nvidia/isaac-sim:4.5.0
```

启动常驻容器(缓存目录挂载很重要,首次 shader 编译要 10 分钟以上,
缓存后秒级启动):

```bash
docker run -d --name isaac-rollout --gpus all \
  -e "ACCEPT_EULA=Y" -e "PRIVACY_CONSENT=Y" \
  --network host \
  -v ~/docker/isaac/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/psi-policy:/workspace/psi-policy:rw \
  -v ~/bundles:/workspace/bundles:ro \
  -v ~/rollout_results:/workspace/results:rw \
  nvcr.io/nvidia/isaac-sim:4.5.0 \
  sleep infinity
```

### 4.3 推理依赖

先走**单进程方案**:把推理依赖装进 Isaac Sim 自带 Python:

```bash
docker exec -it isaac-rollout bash
cd /isaac-sim
./python.sh -m pip install torch --index-url https://download.pytorch.org/whl/cu121
./python.sh -m pip install timm einops pyyaml h5py imageio[ffmpeg]
./python.sh -c "import torch; print(torch.cuda.is_available())"
```

若遇到依赖冲突(Kit 锁死 numpy/torch 版本等),降级到双进程方案(见 §7)。

### 4.4 仿真资产

以下与 ckpt 无关,但必须在 psibot 上就位,且与评测网格定义一致:

- 机器人 USD/URDF(13-DOF 构型,关节顺序与 `joint_order` 一致)
- 评测场景 USD(桌面、物体资产、光照)
- 相机内外参:两个仿真相机的位姿/FOV/分辨率必须复刻采数时的双 RGB 视角
  (建议从采数标定文件直接生成仿真相机配置,不要手调)

放置于 `~/psi-policy/assets/sim/`,纳入 git 或 COS 版本管理。

---

## 5. Rollout Harness 实现(核心)

### 5.1 文件结构

```
psi-policy/rollout/
├── run_rollout.py          # 入口,SimulationApp 初始化 + 主循环
├── sim_env.py              # 场景加载、相机、机器人、reset 到网格点位
├── policy_wrapper.py       # 加载 bundle:ckpt + stats + config,封装 infer()
├── success_checker.py      # 三阶段成功判据(接近/抓取/提起,按已定义标准)
├── eval_grid.yaml          # 固定评测网格(物体位姿 × 初始构型 × 座位数)
└── report.py               # 汇总 jsonl → 成功率表格
```

### 5.2 主循环骨架(`run_rollout.py`)

```python
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "enable_cameras": True})

# --- app 创建之后才能 import omni/isaac 模块 ---
import torch, yaml, json
from sim_env import GraspEnv
from policy_wrapper import Policy
from success_checker import ThreePhaseChecker

PHYSICS_DT = 1.0 / 120.0
RENDER_DT  = 1.0 / 30.0      # 每 4 个物理步渲染一帧 → 与数据合同 30Hz 对齐

def main(bundle_dir, grid_path, out_dir):
    cfg    = yaml.safe_load(open(f"{bundle_dir}/config.yaml"))
    policy = Policy(bundle_dir, device="cuda")          # 内含 train min/max + range_eps 归一化
    env    = GraspEnv(physics_dt=PHYSICS_DT, render_dt=RENDER_DT, cfg=cfg)
    grid   = yaml.safe_load(open(grid_path))
    checker = ThreePhaseChecker(cfg)

    results = []
    for trial in grid["trials"]:
        obs = env.reset(trial)                          # 物体位姿+初始构型
        checker.reset()
        for step in range(grid["max_steps"]):
            # 30Hz 观测:双 RGB + proprio(arm2_pos 当前值)
            chunk = policy.infer(obs)                   # [H, 13] 绝对位置
            for a in chunk[: cfg["action"]["execute_horizon"]]:
                obs = env.step(a)                       # 每步含 4 个物理子步
                checker.update(env.state())
            if checker.done():
                break
        results.append({"trial_id": trial["id"], **checker.summary()})
        env.save_video(f"{out_dir}/videos/{trial['id']}.mp4")   # 复盘用

    with open(f"{out_dir}/results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    app.close()
```

### 5.3 必须严格对齐的三处(rollout 静默出错的高发区)

**(a) 时序假设显式化。** 首次部署先使用 `rendering_dt=1/30`、
`physics_dt=1/120`，每个控制步为 4 个物理子步；由于原始数据没有 timestamp，
该频率是待确认的部署假设，必须写入每次 rollout 的 `run_metadata.json`，不能
描述成已经验证的数据事实。

**(b) 归一化闭环。** `policy_wrapper.Policy.infer()` 内部完成:
obs 用训练时的 min/max + `range_eps` 归一化 → 模型前向(CFM ODE 采样)→
action 反归一化 → 输出物理单位的 13 维 joint-position 目标。stats 只从
bundle 读,**绝不**在 rollout 端重算；每次推理还要逐维检查输出是否落在
train min/max 的 ±10% 外延内，越界立即停止并报告维度与数值。

**(c) 关节顺序与控制模式。** 仿真端 articulation 的关节索引按
`config.yaml: joint_order` 显式重排;控制器用位置目标模式
(policy 输出是绝对执行位置 `arm2_pos`,不是 delta)。PD 增益先取采数
真机/原仿真的同一组参数,不要用 Isaac Sim 默认值。

### 5.4 评测网格与判据

- `eval_grid.yaml` 固化全部 trial:物体类别 × 摆放位姿 × 机器人初始构型,
  每个 trial 有稳定 `id`,不同 ckpt 之间逐 trial 可比。
- 三阶段判据(接近 → 抓取闭合 → 提起保持)沿用已定稿的标准实现在
  `success_checker.py`,每阶段单独记录,报告里输出分阶段成功率——
  这样能区分"够不着"和"phantom grasp"两类失败。
- 随机种子固定(物理、CFM 采样各一个),同一 bundle 重复跑结果应可复现。

### 5.5 首次端到端验收必须分四级

1. **观测对齐**：用训练 episode 的初始 object/robot state 渲染两路 RGB，
   保存训练帧、渲染帧、叠图与差分图，并记录 SSIM 和可复现的物体/机械臂
   关键点重投影误差；不要求像素一致，也不预设跨渲染器通用 SSIM 阈值，
   通过阈值必须由同一套标定基准确定并随场景版本保存。
2. **GT 回放**：不加载模型，回放该 episode 的 executed action，验证位置控制、
   setpoint 执行语义及三阶段判据；若采集环境不是同版本 Isaac Sim，则用部分
   完整 lift episode 校准质量、摩擦、solver 与 PD 参数，再用未参与校准的 episode
   验证，禁止只把单条 GT 调到成功；GT 失败时先修环境，不调模型。
3. **训练位姿单点闭环**：加载 bundle 在见过的位姿运行，定位图像预处理、
   proprio 归一化和 action 执行节奏问题；本级固定 `execute_horizon=16`，不引入
   receding-horizon 对照变量。
4. **182 网格全量**：前三关通过后才运行；六条数据阶段主要产出泛化半径基线，
   不能把远离训练位姿的失败归因于部署链路。

第四级先以 `execute_horizon=chunk_size=16` 建立基线，再用命令行
`--execute-horizon 8` 和 `--execute-horizon 4` 做 receding-horizon 对照。
每次结果与 `run_metadata.json` 都必须记录 chunk size、execute horizon、dt、
库版本和物理参数快照，保证不同 checkpoint/策略可比。

---

## 6. 日常使用流程

```bash
# ① A800:训练到 checkpoint 节点,自动导出并上传 bundle
python export_bundle.py --step 50000 && coscmd upload ...

# ② psibot:拉取 bundle
coscmd download psi-policy/eval_bundles/eval_bundle_fm_step50k.tgz ~/bundles/ && tar xzf ...

# ③ psibot:跑 rollout
docker exec -it isaac-rollout /isaac-sim/python.sh \
  /workspace/psi-policy/rollout/run_rollout.py \
  --bundle /workspace/bundles/eval_bundle_fm_step50k \
  --grid   /workspace/psi-policy/rollout/eval_grid.yaml \
  --out    /workspace/results/fm_step50k

# ④ 汇总报告
docker exec -it isaac-rollout /isaac-sim/python.sh \
  /workspace/psi-policy/rollout/report.py --in /workspace/results/fm_step50k
```

调试期可加 `--livestream 2` 启动,用 Isaac Sim WebRTC Streaming Client
从工作机连 psibot 看实时画面(4090 有 NVENC,支持);批量评测时关掉以省性能。

---

## 7. 备选:双进程方案(仅当依赖冲突时启用)

若 Isaac Sim 内置 Python 与推理依赖打架:

```
[Isaac Sim 进程]  ── obs (ZeroMQ REQ/REP, 本机) ──▶  [conda 推理进程]
      ◀────────────── action chunk ──────────────
```

- 推理进程:你自己的 conda 环境,起 `zmq.REP` 服务,收 obs(JPEG bytes +
  proprio)、回 action chunk。
- 仿真进程:Isaac Sim Python 只需 `pyzmq`(轻量,几乎不会冲突)。
- 仍是**同一台 psibot 上的本地通信**,延迟 <1ms,不涉及 A800。

---

## 8. 验证清单(首次跑通前逐项打勾)

- [ ] 容器内 `torch.cuda.is_available()` 为 True,且渲染无 RTX 报错
- [ ] 空场景 headless + enable_cameras 能取到两路图像,分辨率与 config 一致
- [ ] 双 RGB 已保存训练帧/渲染帧/叠图/差分图，并记录 SSIM、关键点重投影误差与标定阈值来源
- [ ] 关节顺序:给单关节发正弦指令,确认仿真中动的是预期关节
- [ ] 归一化 round-trip:norm(denorm(x)) == x,且 bundle SHA-256 与 `stats_digest` 校验通过
- [ ] `manifest.json` 的 segmentation、torch/CUDA/timm provenance 已打印并核对
- [ ] `run_metadata.json` 已记录 Isaac Sim、dt、solver iterations、质量、摩擦与 execute horizon
- [ ] 物理参数由多条 GT 回放校准，并在未参与校准的完整 lift episode 上验证通过
- [ ] 用训练集中一条真实 episode 的 obs 离线喂给 policy,输出的 action 与
      数据集中记录的 `arm2_pos` 走势一致(open-loop sanity check)
- [ ] 单个 trial 全流程跑通并产出视频
- [ ] 固定种子重复跑同一 trial 两次,结果一致

## 9. 已知风险与对策

| 风险 | 对策 |
|------|------|
| 首次启动 shader 编译极慢 | 挂载 cache 目录(§4.2),只慢第一次 |
| sim-to-sim gap:采数环境与 Isaac Sim 物理差异 | 先跑 open-loop sanity check(清单第 6 项)隔离"模型问题"与"仿真差异" |
| 4090 上仿真+推理抢显存 | 推理用 fp16/bf16;CFM 步数从 10 起调;必要时降相机分辨率与训练输入一致即可,不要超采 |
| bundle 与代码版本漂移 | manifest.json 记 git_sha,rollout 启动时打印比对 |
| 反归一化 action 静默漂移 | 首次推理即执行逐维 train range ±10% 硬哨兵 |
| Isaac/材质变化导致成功率漂移 | 每次评测保存完整物理参数快照并与固定网格共同版本化 |
