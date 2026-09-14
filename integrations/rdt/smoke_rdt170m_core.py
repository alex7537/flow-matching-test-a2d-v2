#!/usr/bin/env python3
"""Run one bounded RDT-170M loss/backward step on a real A2D sample.

Without optional encoder paths it isolates the action core with shape-correct
zero conditions. With ``--siglip-dir`` it additionally validates frozen visual
features from real A2D images. It never validates task-language semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from transformers import SiglipImageProcessor, SiglipVisionModel

from a2d_hdf5_vla_dataset import HDF5VLADataset
from models.rdt_runner import RDTRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-tail", action="store_true")
    parser.add_argument("--siglip-dir")
    parser.add_argument("--empty-lang-embed")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this bounded A800 smoke test")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["A2D_RDT_DATA_DIR"] = args.data_dir
    os.environ["A2D_RDT_SPLIT"] = args.split
    os.environ["A2D_RDT_SEED"] = str(args.seed)

    dataset = HDF5VLADataset()
    sample = dataset.get_item(args.index)
    if args.require_tail and sample["meta"]["valid_action_steps"] == 64:
        for offset in range(1, len(dataset)):
            sample = dataset.get_item(args.index + offset)
            if sample["meta"]["valid_action_steps"] < 64:
                break
        else:
            raise RuntimeError("could not sample a tail-padded action chunk")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    load_start = time.monotonic()
    runner = RDTRunner.from_pretrained(args.model_dir, map_location="cpu")
    runner = runner.to(device=device, dtype=dtype)
    runner.train()
    load_seconds = time.monotonic() - load_start

    state = torch.from_numpy(sample["state"]).unsqueeze(0).to(device, dtype)
    actions = torch.from_numpy(sample["actions"]).unsqueeze(0).to(device, dtype)
    action_mask = (
        torch.from_numpy(sample["state_indicator"]).reshape(1, 1, 128).to(device, dtype)
    )
    action_time_mask = (
        torch.from_numpy(sample["action_time_mask"]).reshape(1, 64).to(device, dtype)
    )
    scope = "action_core_only_zero_image_and_language_embeddings"
    if args.siglip_dir:
        processor = SiglipImageProcessor.from_pretrained(args.siglip_dir)
        vision = SiglipVisionModel.from_pretrained(args.siglip_dir).to(device, dtype=dtype)
        vision.eval()
        background_rgb = tuple(int(x * 255) for x in processor.image_mean)

        def prepare(image: np.ndarray | None) -> torch.Tensor:
            if image is None:
                pil = Image.new("RGB", (384, 384), background_rgb)
            else:
                pil = Image.fromarray(image)
                width, height = pil.size
                if width != height:
                    side = max(width, height)
                    squared = Image.new("RGB", (side, side), background_rgb)
                    squared.paste(pil, ((side - width) // 2, (side - height) // 2))
                    pil = squared
            return processor.preprocess(pil, return_tensors="pt")["pixel_values"][0]

        images = []
        for history_id in range(2):
            images.extend([
                prepare(sample["cam_high"][history_id]),
                prepare(sample["cam_right_wrist"][history_id]),
                prepare(None),
            ])
        pixels = torch.stack(images).to(device, dtype=dtype)
        with torch.no_grad():
            img_tokens = vision(pixel_values=pixels).last_hidden_state
        img_tokens = img_tokens.reshape(1, 4374, 1152).detach()
        scope = "frozen_siglip_real_a2d_images_empty_language_embedding"
        del vision, pixels
        torch.cuda.empty_cache()
    else:
        # RDT-170M's immutable config expects 4,374 SigLIP image tokens of
        # width 1,152. Zeros isolate the action core for the first gate.
        img_tokens = torch.zeros((1, 4374, 1152), device=device, dtype=dtype)

    if args.empty_lang_embed:
        lang_tokens = torch.load(args.empty_lang_embed, map_location="cpu")
        if lang_tokens.ndim == 2:
            lang_tokens = lang_tokens.unsqueeze(0)
        lang_tokens = lang_tokens.to(device, dtype=dtype)
    else:
        lang_tokens = torch.zeros((1, 1, 4096), device=device, dtype=dtype)
    lang_mask = torch.ones(lang_tokens.shape[:2], device=device, dtype=torch.bool)
    ctrl_freq = torch.tensor([30], device=device, dtype=torch.long)

    runner.zero_grad(set_to_none=True)
    step_start = time.monotonic()
    loss = runner.compute_loss(
        lang_tokens=lang_tokens,
        lang_attn_mask=lang_mask,
        img_tokens=img_tokens,
        state_tokens=state,
        action_gt=actions,
        action_mask=action_mask,
        ctrl_freqs=ctrl_freq,
        action_time_mask=action_time_mask,
    )
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss: {loss.item()}")
    loss.backward()
    step_seconds = time.monotonic() - step_start

    trainable = sum(p.numel() for p in runner.parameters() if p.requires_grad)
    grad_params = [p for p in runner.parameters() if p.grad is not None]
    finite_grads = all(torch.isfinite(p.grad).all().item() for p in grad_params)
    grad_l2 = torch.sqrt(
        sum(torch.sum(p.grad.detach().float() ** 2) for p in grad_params)
    ).item()
    final_grad = runner.model.final_layer.ffn_final.fc2.weight.grad.detach().float()
    selected = torch.as_tensor(
        np.flatnonzero(sample["state_indicator"]), device=final_grad.device
    )
    unselected = torch.as_tensor(
        np.flatnonzero(sample["state_indicator"] == 0), device=final_grad.device
    )

    print(json.dumps({
        "status": "RDT170M_A2D_CORE_BACKWARD_OK",
        "scope": scope,
        "checkpoint": args.model_dir,
        "dataset_split": args.split,
        "episode": sample["meta"]["source_file_name"],
        "task": sample["meta"]["task_id"],
        "step_id": sample["meta"]["step_id"],
        "valid_action_steps": sample["meta"]["valid_action_steps"],
        "state_shape": list(state.shape),
        "action_shape": list(actions.shape),
        "image_token_shape": list(img_tokens.shape),
        "language_token_shape": list(lang_tokens.shape),
        "active_action_dimensions": int(action_mask.sum().item()),
        "valid_action_steps_from_mask": int(action_time_mask.sum().item()),
        "supervised_scalar_elements": int(
            action_mask.sum().item() * action_time_mask.sum().item()
        ),
        "active_indices": np.flatnonzero(sample["state_indicator"]).tolist(),
        "loss": float(loss.detach().float().item()),
        "trainable_parameters": trainable,
        "gradient_parameter_tensors": len(grad_params),
        "gradient_l2": grad_l2,
        "all_gradients_finite": finite_grads,
        "final_head_active_rows_grad_mean_abs": float(final_grad[selected].abs().mean().item()),
        "final_head_inactive_rows_grad_mean_abs": float(final_grad[unselected].abs().mean().item()),
        "load_seconds": load_seconds,
        "forward_backward_seconds": step_seconds,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }, indent=2))


if __name__ == "__main__":
    main()
