from __future__ import annotations

import argparse
from typing import Any

import torch


def _shape_of(output: Any) -> str:
    if isinstance(output, torch.Tensor):
        return str(tuple(output.shape))
    if isinstance(output, (list, tuple)):
        return "[" + ", ".join(_shape_of(item) for item in output) + "]"
    if isinstance(output, dict):
        return "{" + ", ".join(f"{key}: {_shape_of(value)}" for key, value in output.items()) + "}"
    return type(output).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a timm model can be imported and instantiated.")
    parser.add_argument("--model-name", default="vit_small_r26_s32_224", help="timm model name to validate")
    parser.add_argument("--image-size", type=int, default=224, help="Dummy input spatial size")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy input batch size")
    parser.add_argument("--pretrained", action="store_true", help="Try loading pretrained weights")
    parser.add_argument("--num-classes", type=int, default=0, help="num_classes passed to timm.create_model")
    parser.add_argument("--list-only", action="store_true", help="Only check if the model name exists in timm")
    args = parser.parse_args()

    try:
        import timm  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("timm is not installed in the current environment") from exc

    model_names = set(timm.list_models())
    print(f"timm_version={getattr(timm, '__version__', 'unknown')}")
    print(f"model_name={args.model_name}")
    print(f"model_exists={args.model_name in model_names}")
    if args.model_name not in model_names:
        raise SystemExit(f"Unknown timm model: {args.model_name}")

    if args.list_only:
        return

    model = timm.create_model(
        args.model_name,
        pretrained=bool(args.pretrained),
        num_classes=int(args.num_classes),
    )
    model.eval()
    print(f"model_type={type(model).__name__}")

    dummy = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    with torch.no_grad():
        output = model(dummy)
    print(f"forward_output={_shape_of(output)}")


if __name__ == "__main__":
    main()
