# A2D视觉编码器：从双RGB到condition tokens

本文只描述基础 action-only CFM/DP/IMLE 共用的视觉 observation 路线，不包含 Joint WAM 的9帧 Wan视频latent。

## 当前配置

```yaml
data:
  image_keys: [rgb_head, rgb_right_hand]
  image_size: 224
  history_steps: 1

model:
  encoder_type: timm
  timm_model_name: vit_small_r26_s32_224
  timm_pretrained: true
  timm_token_mode: spatial
  timm_tokens_per_frame: 49
  d_model: 384
  use_proprio: true
```

`timm`是模型库和工厂；`vit_small_r26_s32_224`是具体的Hybrid ViT；ImageNet系列预训练权重由`pretrained: true`加载。

## 完整形状链

每个相机独立产生token，但两个相机共享同一套视觉权重：

```text
RGB                              [B,1,3,224,224]
去掉history维用于逐帧编码         [B,3,224,224]
ImageNet mean/std归一化           [B,3,224,224]
ResNetV2-26输出                   [B,2048,7,7]
HybridEmbed 1×1 Conv2d           [B,384,7,7]
flatten + transpose              [B,49,384]
prepend learned CLS              [B,50,384]
ViT Transformer输出              [B,50,384]
选择spatial或CLS                 [B,49,384]或[B,1,384]
```

开发机上`timm 0.9.16`对该模型的实测属性为：

```text
patch_embed_type  HybridEmbed
backbone_type     ResNetV2
input             224×224
grid_size         7×7
num_patches       49
ResNet channels   2048
token dimension   384
num_prefix_tokens 1
projection        Conv2d(2048,384,kernel=1,stride=1)
```

### 这里的“空间”是什么

`[B,2048,7,7]`不是机器人三维空间。对单个样本，它是一个二维图像网格，每个网格位置拥有一个2048维特征向量：

$$
x_{i,j}\in\mathbb R^{2048},\qquad i,j\in\{1,\ldots,7\}.
$$

卷积感受野会重叠，所以每个位置并非严格对应一块互不重叠的32×32原始像素；`s32`表示最终空间步长为32。

### 1×1 Conv2d在做什么

1×1卷积不改变7×7位置，只在每个位置应用同一个通道投影：

$$
\mathbb R^{2048}\rightarrow\mathbb R^{384}.
$$

因此这里是通道压缩，不是空间降采样，也不是一维卷积。展平后得到49个384维patch embeddings。

### ViT处理前后

在ViT Transformer之前，49个token主要来源于各自ResNet空间位置。加入CLS和位置编码后，所有token经过多层self-attention：

```text
[CLS, patch_1, ..., patch_49]
             ↓ self-attention
[global CLS, contextualized spatial_1, ..., spatial_49]
```

因此最终49个spatial tokens仍保留位置索引，但每个token已经可以包含整张图的上下文。它们不是分割mask，也不是49个网络层。

## Spatial与CLS两条出口

`TimmRGBEncoder`在完整ViT之后选择输出：

```python
if token_mode == "spatial":
    encoded = encoded[:, num_prefix_tokens:]
else:
    encoded = encoded[:, :1]
```

### 默认spatial路线

```text
head相机             [B,49,384]
right-hand相机       [B,49,384]
proprio投影          [B, 1,384]
拼接                 [B,99,384]
```

### CLS消融路线

```text
head CLS             [B,1,384]
right-hand CLS       [B,1,384]
proprio投影          [B,1,384]
拼接                 [B,3,384]
```

CLS是从预训练开始就加入序列的可学习向量，通过self-attention聚合全局信息。CLS路线只减少传给Action Transformer的condition KV数量；ViT内部仍然计算ResNet特征、49个patch和全部self-attention。

两个路线的`cond_pos`形状分别是`[1,99,384]`和`[1,3,384]`，所以不能把已训练的spatial checkpoint只改YAML后当成CLS模型使用。公平消融必须从相同预训练ViT重新训练。

## 两个相机如何区分

`rgb_head`和`rgb_right_hand`使用相同视觉backbone，然后按照配置中的固定顺序拼接。共享权重保证相同视觉模式使用同一特征空间；拼接位置与condition位置编码让Action Transformer区分两路token。

当前实现没有显式执行双目标定、极线匹配或三角测量。模型只能从同步双视角、固定相机关系和动作监督中隐式学习相对几何。

## 视觉encoder如何被训练

基础路线没有分类、分割或深度loss。视觉特征只通过策略目标接收梯度：

```text
action loss
→ Action Transformer cross-attention
→ observation tokens
→ ViT Transformer
→ HybridEmbed
→ ResNetV2
```

预训练ViT使用`backbone_lr_multiplier: 0.1`，即视觉backbone峰值学习率是action head的十分之一。这是为了适应机器人图像，同时降低破坏预训练视觉表征的风险。

## 替代模型的接口边界

任何新视觉模型最终都必须输出token map：

```text
encoder(obs) -> {camera_name: [B,L,D_native]}
TokenLevelConcat -> [B,total_tokens,384]
```

- DINO/DINOv2仍可使用ViT结构和CLS/patch tokens，但patch大小可能产生196或256个空间token。
- 当前实现的`tokens_per_frame`是截取前N个token，不是二维均匀池化；将196直接设置为49会丢弃其余空间位置。公平49-token对比应先把完整特征网格池化到7×7。
- U-Net通常输出`[B,C,H,W]`；可用1×1投影到384维并池化/展平为`[B,49,384]`。如果没有mask、depth或重建辅助loss，完整U-Net decoder不一定有用。
- Depth不是RGB encoder的替代配置，而是新模态；需要独立的数据、归一化、encoder、时间同步和部署合同。

## 代码入口

- `flow_matching_test/observation.py::TimmRGBEncoder`：调用timm、归一化、选择spatial/CLS；
- `flow_matching_test/observation.py::TokenLevelConcat`：通道对齐和固定顺序拼接；
- `flow_matching_test/policies/flow_matching.py::_encode_obs`：追加proprio token与condition位置编码；
- `configs/a2d_v3_multitask_box300_bottle300_cfm_a800_100ep_scratch.yaml`：99-token spatial基线；
- `configs/a2d_v3_multitask_box300_bottle300_cfm_cls_a800_100ep_scratch.yaml`：3-token CLS消融。

## 最小验收

在训练前至少断言：

```text
spatial: head=[B,49,384], hand=[B,49,384], condition=[B,99,384]
CLS:     head=[B, 1,384], hand=[B, 1,384], condition=[B, 3,384]
```

科学比较保持dataset/split、action contract、ViT权重、proprio、batch、optimizer steps和seed一致；最终比较离线action MSE以及同协议rollout的approach、contact、lift和retention。
