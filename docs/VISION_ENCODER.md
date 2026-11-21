# A2D Vision Encoder: Dual RGB to Condition Tokens

**English** | [简体中文](VISION_ENCODER.zh-CN.md)

This document covers the shared observation encoder for the action-only CFM, Diffusion Policy, and IMLE routes. It does not describe Joint WAM's nine-frame Wan video latents.

## Configuration

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

`timm` is the model library; `vit_small_r26_s32_224` is the particular Hybrid ViT. The pretrained flag loads its ImageNet-family initialization.

## Shape trace for each view

The cameras share visual weights but produce separate tokens. With one observation frame:

```text
RGB input                         [B,1,3,224,224]
remove history axis for encoding  [B,3,224,224]
ImageNet mean/std normalization    [B,3,224,224]
ResNetV2-26 feature map            [B,2048,7,7]
HybridEmbed 1x1 projection         [B,384,7,7]
flatten + transpose               [B,49,384]
prepend learned CLS               [B,50,384]
ViT self-attention                [B,50,384]
spatial / CLS selection           [B,49,384] / [B,1,384]
```

The Chinese record reports these model properties measured with `timm 0.9.16`: HybridEmbed, ResNetV2, 7 × 7 feature grid, 49 patches, 2,048 backbone channels, 384-dimensional tokens, and one prefix token.

The 7 × 7 grid is **image-feature space**, not robot XYZ coordinates. Each location carries a 2,048-dimensional feature vector before projection. Convolutional receptive fields overlap; the locations are not disjoint 32 × 32 image tiles. The `s32` suffix denotes the final spatial stride.

The 1 × 1 convolution applies the same channel projection, 2048→384, at every location. It does not change the 7 × 7 grid. After self-attention, each spatial token retains a position index but can contain context from the whole image. These tokens are neither segmentation masks nor network layers.

## Spatial tokens versus CLS

Selection happens **after the complete ViT**:

```python
if token_mode == "spatial":
    encoded = encoded[:, num_prefix_tokens:]
else:
    encoded = encoded[:, :1]
```

| Condition source | Spatial baseline | CLS ablation |
|---|---|---|
| Head camera | `[B,49,384]` | `[B,1,384]` |
| Right-hand camera | `[B,49,384]` | `[B,1,384]` |
| Optional proprioception | `[B,1,384]` | `[B,1,384]` |
| Total with proprioception | `[B,99,384]` | `[B,3,384]` |

CLS is a learned prefix token initialized as part of the pretrained model and contextualized by attention. Selecting CLS reduces condition tokens passed to the Action Transformer; it does not avoid computing the ResNet, all patches, or ViT attention.

The condition positional embeddings have different shapes: `[1,99,384]` versus `[1,3,384]`. A trained spatial checkpoint cannot become a CLS model by editing YAML. A controlled ablation starts from the same pretrained visual initialization and retrains.

## Multi-view conditioning

The two views are concatenated in the configured order. Shared weights put their visual patterns in a common feature space; token positions and condition positional embeddings distinguish the views.

This encoder does not explicitly perform stereo calibration, epipolar matching, or triangulation. Relative geometry must be learned implicitly from synchronized views, fixed camera relationships, and action supervision.

## How the encoder learns

There is no classification, segmentation, or depth objective in the baseline:

```text
policy action loss
 -> Action Transformer cross-attention
 -> observation tokens
 -> ViT Transformer
 -> HybridEmbed
 -> ResNetV2
```

With `backbone_lr_multiplier: 0.1`, the visual backbone peak learning rate is one tenth of the action head rate. This allows adaptation while limiting disruption of pretrained features.

## Replacing the backbone

A replacement must respect the observation-token interface:

```text
encoder(obs) -> {camera_name: [B,L,D_native]}
TokenLevelConcat -> [B,total_tokens,384]
```

- DINO/DINOv2 may produce 196 or 256 patch tokens. The current `tokens_per_frame` operation truncates the first N tokens; it is not uniform two-dimensional pooling. A fair 49-token comparison should pool the complete grid to 7 × 7.
- A U-Net map `[B,C,H,W]` can be projected to 384 channels, pooled, and flattened. A full decoder is not automatically useful without an appropriate auxiliary task.
- Depth requires a new modality contract, not just another RGB encoder name.

## Source and validation

- [TimmRGBEncoder / TokenLevelConcat](../flow_matching_test/observation.py)
- [CFM observation conditioning](../flow_matching_test/policies/flow_matching.py)
- [Spatial baseline configuration](../configs/a2d_v3_multitask_box300_bottle300_cfm_a800_100ep_scratch.yaml)
- [CLS ablation configuration](../configs/a2d_v3_multitask_box300_bottle300_cfm_cls_a800_100ep_scratch.yaml)

Before training, assert the condition shapes above. Compare spatial/CLS with the same data/split, action contract, pretrained ViT, proprioception setting, batch size, optimizer-step budget, and seed. Evaluate offline action MSE and matched rollouts, including approach, contact, lift, and retention.
