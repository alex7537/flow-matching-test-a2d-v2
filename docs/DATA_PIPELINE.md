# A2D 数据处理管线部署方案

> 目标:将 `a2d_curobo_collect_run1/success` 下的原始 HDF5 抓取数据,转换为 flow matching policy 训练可用的高效数据管线。本文档面向执行 agent,包含全部背景、决策依据、部署步骤与验证清单。

---

## 1. 背景与核心问题

### 1.1 原始数据

- 单条 episode 一个 `.hdf5` 文件,约 546 MB,轨迹长度 ~133 步。
- 训练硬件:单张 RTX 4090 (24GB),本地训练,数据总量 100–200 GB(约 200–370 条 episode,3–5 万帧)。
- 训练目标:conditional flow matching policy,多相机观测 token 条件化,预测 action chunk(horizon=16)的关节指令。

### 1.2 为什么不能直接在原始文件上训练

原始 RGB 数据集的 HDF5 chunk 布局为 `chunks=(17, 60, 80, 1)` + LZF 压缩:

- 时间维一个 chunk 跨 17 帧。随机读取单帧需解压覆盖该帧的全部 192 个 chunk(约 16 MB),取出 0.9 MB 有效数据,**读放大约 17 倍**;三相机单样本约 48 MB 解压量。
- 训练分辨率 224×224,原始 480×640 中 87% 像素读入后即丢弃。
- 结论:**必须离线预处理**,将图像 resize + JPEG 编码,重写为随机读取友好的布局。处理后总数据量约 3–5 GB,首个 epoch 后全部进入 Linux page cache,IO 瓶颈消失。

### 1.3 原始 schema 关键字段(state/action 映射依据)

| 原始路径 | 形状 | 语义 | 用途 |
|---|---|---|---|
| `trajectory/arm2_pos` | (T, 7) | 手臂关节实测值 | → state + action |
| `trajectory/hand2_pos` | (T, 6) | 手部关节实测值 | → state + action |
| `trajectory/arm2_pos_target` | (T, 7) | 稀疏手臂指令记录 | 仅审计 |
| `trajectory/hand2_pos_target` | (T, 6) | 手部指令记录 | 仅审计 |
| `trajectory/waist_pos` | (T, 2) | 腰部关节 | 待定,见 §6 |
| `trajectory/cameras/rgb_{head,left_hand,right_hand}` | (T, 480, 640, 3) uint8 | 三路可用 RGB | 通过 `--image-keys` 选择观测子集 |
| `trajectory/cameras/depth_*` | (T, 480, 640) f32 | 深度 | 当前管线不用,见 §7 |
| `trajectory/phase` | (T,) str | 22 种阶段标签 | 保留,用于分析/加权 |
| `meta/success`, `grasp/quality_score` 等 | attrs | 质量元数据 | 保留在输出 attrs |

**决策依据**:`arm2_pos_target` 仅 8/133 帧有效，而 `*_pos` 133/133 帧完整，因此 policy 学习成功 episode 的实际执行关节轨迹。

---

## 2. 管线架构

> **W&B 分级纪律:**分钟级调试/小数据 run 可在 A800 使用 `online` 实时上传，但 `WandbLogger` 必须在异常时只告警一次并永久降级为 no-op；350GB 长训固定以 `wandb==0.28.0` offline 落盘，再搬运到 psibot 用 `requirements.lock.wandb-sync.txt` 固定的 `wandb==0.27.0` 同步，并以 `scripts/verify_wandb_sync.py` 的服务端 run 存在性及 history 行数校验作为成功判据；online 密钥不得落入共享 `/share_data`。

```
原始 success/*.hdf5  (100–200 GB, chunk 布局对随机读极不友好)
        │
        │  scripts/preprocess_a2d.py   ← 一次性,ProcessPool 并行
        ▼
处理后 a2d_processed/*.hdf5  (~3–5 GB, schema 归一化)
        │    observations/qpos      (T, 13)  f32   arm2_pos ⊕ hand2_pos
        │    observations/rgb_*     (T,)     vlen uint8 (jpeg, 224²)
        │    action                 (T, 13)  f32   arm2_pos ⊕ hand2_pos
        │    phase                  (T,)     vlen str
        │    segment_type           (T,)     u8    static/continuous/keyframe
        │    arm_keyframe           (T,)     u8    arm 关键帧掩码
        │    attrs: segmentation_version / thresholds / action semantics
        │
        │  python a2d_dataset.py --compute-stats   ← 一次性
        ▼
index_cache.json + norm_stats.json  (绑定 train episode 内容哈希的 min/max 统计)
        │
        │  src/data/a2d_dataset.py :: build_dataloaders(cfg)
        ▼
train_loader / val_loader
     · episode 级 train/val 切分(固定 seed,防相邻帧泄漏)
     · 每 worker 惰性打开 h5 句柄(fork 安全)
     · action chunk 尾部 padding + action_mask
     · 图像增强仅 train(pad-crop 抖动 + color jitter)
     · state/action 按 train min/max 归一化到 [-1, 1](clip ±3)
```

## 3. 文件清单与仓库落位

| 文件 | 建议落位 | 作用 |
|---|---|---|
| `preprocess_a2d.py` | `scripts/preprocess_a2d.py` | 原始 → 训练格式的一次性转换 |
| `a2d_dataset.py` | `flow_matching_test/a2d_dataset.py` | Dataset / DataLoader / 归一化 / 索引 / stats CLI |
| 本文档 | `docs/DATA_PIPELINE.md` | 方案说明 |

两个脚本均已存在(与本文档同批交付),无需重写;落位后仅需按 §6 核对待确认项。

## 4. 部署步骤

```bash
# ── 推荐:审计门禁 + quarantine + 预处理 + manifest + stats ──
python scripts/ingest_a2d.py \
    --src /path/to/raw_rgb_episodes \
    --output-root /path/to/datasets \
    --dataset-version a2d_rgb_complete_v1 \
    --admission complete-lift \
    --image-keys rgb_head rgb_right_hand --workers 4

# ── 0. 依赖 ──────────────────────────────────────────────
pip install h5py opencv-python numpy torch
pip install PyTurboJPEG   # 可选,jpeg 解码提速 2–4 倍,需系统 libturbojpeg

# ── 1. 离线预处理(一次性,4 进程约 1–2 小时)──────────────
python scripts/preprocess_a2d.py \
    --src /home/psibot/Downloads/flow-matching-test/a2d_curobo_collect_run1/success \
    --dst /home/psibot/data/a2d_processed \
    --image-keys rgb_head rgb_right_hand \
    --image-size 224 --jpeg-quality 92 --workers 4
# 若 §6 确认腰部会动,追加 --include-waist
# 改变 --image-keys 时请使用新的 --dst；脚本会拒绝复用视角配置不同的旧产物。

# ── 2. 建索引 + 归一化统计(一次性)────────────────────────
python -m flow_matching_test.a2d_dataset \
    --data-dir /home/psibot/data/a2d_processed \
    --image-keys rgb_head rgb_right_hand --compute-stats \
    --motion-threshold 1e-4 --keyframe-threshold 0.1

# ── 3. 训练代码接入 ──────────────────────────────────────
# from src.data.a2d_dataset import A2DConfig, build_dataloaders
# cfg = A2DConfig(data_dir="/home/psibot/data/a2d_processed", ...)
# train_loader, val_loader, norm_stats = build_dataloaders(cfg)
```

训练 yaml 相对现状需要的改动:

```yaml
data:
  data_dir: /home/psibot/data/a2d_processed   # 指向处理后目录
  image_keys: [rgb_head, rgb_right_hand]  # 从可用视角中选择任意非空子集
training:
  device: cuda        # 现为 cpu(冒烟配置)
  batch_size: 16      # 4090 + ViT-S 全参数微调起点；先实测 1-2 batch 再提高
  num_workers: 8      # 现为 0;盯 GPU 利用率不足再加到 12–16
  pin_memory: true
```

## 5. 关键设计决策(勿随意更改)

1. **episode 级 train/val 切分**,固定 seed。同轨迹相邻帧高度相似,按 sample 切分会造成 val 泄漏、val loss 假低。
2. **归一化统计仅在 train episodes 上计算**,绝对 state/action 使用逐维 min/max 映射到 [-1, 1],常量维度有 `range_eps=1e-4` 保护。
3. **`norm_stats.json` 保存排序后的 train episode 内容哈希列表及摘要**；Dataset 按实际 split 重算摘要并硬校验，checkpoint 同时内嵌 normalizer 与摘要。
4. **h5 句柄绝不在主进程打开后 fork**。Dataset 内每个 worker 首次 `__getitem__` 时惰性打开自己的句柄(`_handles` dict)。这是 hdf5 + `num_workers>0` 的经典崩溃/静默错误来源。
5. **action chunk 尾部 padding**:episode 末尾不足 horizon 时重复末帧,并返回 `action_mask`;训练 loss 建议乘以 mask。
6. **预处理输出先写 `.tmp` 再 rename**,中断可安全重跑(已存在的输出自动跳过,幂等)。
7. **train 对包含 arm 实际位置变化点的窗口做可配置过采样，val 始终保持原始窗口分布**。
8. **模型默认将归一化后的 13 维当前 joint state 编码为 proprio condition token**。
9. **分段在预处理阶段物化**：processed HDF5 保存 `segmentation_version`、阈值、`segment_type` 与 `arm_keyframe`，Dataset 只读取并校验版本，不在线重判。
10. **数据集版本不可原地改写 stats**：train 成员变化后创建新数据目录或使用新的 `--norm-stats` 文件名，旧 checkpoint 继续绑定旧摘要。

## 6. 待确认项(部署前必须核对)

- [ ] **waist_pos 是否进 state**:读若干 episode 检查 `trajectory/waist_pos` 的 std。接近 0 → 不加;明显变化 → 预处理加 `--include-waist`(它影响相机视角与手臂基座,不可忽略)。
- [x] **action 契约**:固定为实际执行的 `arm2_pos(7)+hand2_pos(6)=13` 维，Dataset 建索引时校验 shape、layout 与 semantics。
- [ ] **控制频率与 horizon**:确认采集频率,horizon=16 对应的物理时长是否覆盖一个有意义的动作片段。

## 7. 已知取舍与后续扩展

- **depth 未纳入**(占原始体积近一半,当前管线纯 RGB)。鉴于系统存在空间深度感知的失败模式,后续若加 depth 输入分支:在 `preprocess_a2d.py` 中补充 depth 读取(建议 resize + float16 + 每帧独立 chunk),重跑预处理即可,无需重新采集。
- **phase 标签已保留**:可用于 (a) 按阶段分解 val 误差(预期 pregrasp 易学、finger squeeze/guarded settle 等精细阶段难学);(b) 阶段加权采样。
- **数据量偏少的风险**:3–5 万帧、疑似单一物体。缓解手段优先级:冻结预训练 backbone > 加强图像增强 > 收集更多样化数据(检查 meta 中 `disturbance_shift_m` / `sample_angle_deg` 评估现有多样性)。
- **验证协议**(与数据管线配套):val loss 计算需固定 noise 与 t 的采样 seed 才具备 checkpoint 间可比性;每隔数千步跑一次完整 ODE 采样(10 步 Euler)的 action MSE 作为更接近推理行为的指标;checkpoint 选择最终以 rollout 成功率为准;使用 EMA 权重(decay≈0.999)评估与部署。

## 8. 验证清单(部署后逐项执行)

```bash
# 0. 时间/结构对齐审计。没有 timestamp 时只能确认索引长度一致，报告会明确标记 not_verifiable。
python scripts/check_data_alignment.py \
    --data-dir /path/to/raw_episodes \
    --image-keys rgb_head rgb_right_hand \
    --output alignment_report.json

# 1. 预处理产物完整性:episode 数量一致、总大小符合预期(~3–5 GB)
ls /home/psibot/data/a2d_processed/*.hdf5 | wc -l

# 2. 单文件 schema 抽查
python -c "
import h5py
f = h5py.File(sorted(__import__('glob').glob('/home/psibot/data/a2d_processed/*.hdf5'))[0])
f.visit(print); print(dict(f.attrs))"

# 3. Dataset 冒烟:取一个 batch,检查形状与数值范围
#    images (B, 3*history, 3, 224, 224) ∈ [0,1]
#    state  (B, 13*history)  大致 ∈ [-1,1]
#    action (B, 16, 13)      大致 ∈ [-1,1](若大量贴 ±3 边界 → stats 有误)
#    action_mask (B, 16)

# 4. 吞吐验证:纯 dataloader 迭代速度应 > 训练前向所需吞吐
# 5. 训练中:watch -n1 nvidia-smi,GPU 利用率 < 80% → 先查 jpeg 解码
#    (装 PyTurboJPEG)与 num_workers,而不是改模型
```
