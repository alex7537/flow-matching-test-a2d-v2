from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


class BaseObservationEncoder(nn.Module, ABC):
    """Minimal modality encoder interface for future multi-modal expansion."""

    @property
    @abstractmethod
    def input_keys(self) -> Sequence[str]:
        """Observation keys consumed by this encoder."""

    @property
    @abstractmethod
    def output_names(self) -> Sequence[str]:
        """Token-map names produced by this encoder."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Native token dimension emitted by this encoder."""

    @abstractmethod
    def forward(self, obs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode raw observations into a token map."""


class CNNRGBEncoder(BaseObservationEncoder):
    """Shared CNN encoder for the RGB views selected by ``image_keys``."""

    def __init__(self, *, image_keys: tuple[str, ...], d_model: int) -> None:
        super().__init__()
        self._image_keys = tuple(image_keys)
        self.tokens_per_frame = 1
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)
        self._embedding_dim = int(d_model)

    @property
    def input_keys(self) -> Sequence[str]:
        return list(self._image_keys)

    @property
    def output_names(self) -> Sequence[str]:
        return list(self._image_keys)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def encode_rgb(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError(f"RGB obs must have shape [B,T,3,H,W], got {tuple(value.shape)}")
        batch_size, steps = value.shape[:2]
        encoded = self.backbone(value.reshape(batch_size * steps, *value.shape[2:]))
        encoded = self.out_norm(encoded)
        return encoded.reshape(batch_size, steps, -1)

    def forward(self, obs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token_map: dict[str, torch.Tensor] = {}
        for key in self._image_keys:
            if key not in obs:
                raise KeyError(f"Missing RGB observation key: {key}")
            token_map[key] = self.encode_rgb(obs[key])
        return token_map


class TimmRGBEncoder(BaseObservationEncoder):
    """Shared timm encoder that mirrors the fan_dev RGB tokenization route."""

    def __init__(
        self,
        *,
        image_keys: tuple[str, ...],
        model_name: str,
        pretrained: bool,
        tokens_per_frame: int = 1,
        token_mode: str = "spatial",
    ) -> None:
        super().__init__()
        if tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be > 0")
        self._image_keys = tuple(image_keys)
        self.model_name = str(model_name)
        self.tokens_per_frame = int(tokens_per_frame)
        self.token_mode = str(token_mode)
        if self.token_mode not in {"spatial", "cls"}:
            raise ValueError("timm token_mode must be 'spatial' or 'cls'")
        try:
            import timm  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("TimmRGBEncoder requires timm to be installed") from exc

        self.backbone = timm.create_model(
            self.model_name,
            pretrained=bool(pretrained),
            global_pool="",
            num_classes=0,
        )
        pretrained_cfg = getattr(self.backbone, "pretrained_cfg", {}) or {}
        mean = pretrained_cfg.get("mean", (0.485, 0.456, 0.406))
        std = pretrained_cfg.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("input_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self._embedding_dim = self._infer_feature_dim()

    @property
    def input_keys(self) -> Sequence[str]:
        return list(self._image_keys)

    @property
    def output_names(self) -> Sequence[str]:
        return list(self._image_keys)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _forward_backbone_tokens(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(x)
        if output.ndim == 2:
            return output[:, None, :]
        if output.ndim == 3:
            return output
        if output.ndim == 4:
            return output.flatten(2).transpose(1, 2)
        raise ValueError(f"Unsupported timm output shape: {tuple(output.shape)}")

    def _infer_feature_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            tokens = self._forward_backbone_tokens(dummy)
            return int(tokens.shape[-1])

    def encode_rgb(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5:
            raise ValueError(f"RGB obs must have shape [B,T,3,H,W], got {tuple(value.shape)}")
        batch_size, steps = value.shape[:2]
        pixels = value.reshape(batch_size * steps, *value.shape[2:])
        pixels = ((pixels + 1.0) * 0.5 - self.input_mean) / self.input_std
        encoded = self._forward_backbone_tokens(pixels)
        prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        if self.token_mode == "spatial":
            encoded = encoded[:, prefix_tokens:]
        else:
            encoded = encoded[:, :1]
        if encoded.shape[1] < self.tokens_per_frame:
            raise ValueError(
                f"requested {self.tokens_per_frame} {self.token_mode} tokens, "
                f"but backbone produced {encoded.shape[1]}"
            )
        encoded = encoded[:, : self.tokens_per_frame]
        return encoded.reshape(batch_size, steps * self.tokens_per_frame, -1)

    def forward(self, obs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token_map: dict[str, torch.Tensor] = {}
        for key in self._image_keys:
            if key not in obs:
                raise KeyError(f"Missing RGB observation key: {key}")
            token_map[key] = self.encode_rgb(obs[key])
        return token_map


class TokenLevelConcat(nn.Module):
    """Concatenate token-map entries in a fixed order and align output dims."""

    def __init__(
        self,
        *,
        order: Sequence[str],
        input_dims: Mapping[str, int],
        output_dim: int,
    ) -> None:
        super().__init__()
        self.order = [str(name) for name in order]
        if not self.order:
            raise ValueError("Token concat order must not be empty")
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        self.adapters = nn.ModuleDict()
        for name in self.order:
            if name not in input_dims:
                raise ValueError(f"Concat order key missing from input_dims: {name}")
            dim = int(input_dims[name])
            if dim <= 0:
                raise ValueError(f"input_dims[{name}] must be > 0")
            self.adapters[name] = nn.Identity() if dim == self.output_dim else nn.Linear(dim, self.output_dim)

    def forward(self, token_map: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
        tokens: list[torch.Tensor] = []
        for name in self.order:
            if name not in token_map:
                raise KeyError(f"Token concat missing token: {name}")
            tokens.append(self.adapters[name](token_map[name]))
        part_lengths = [int(token.shape[1]) for token in tokens]
        return torch.cat(tokens, dim=1), part_lengths


class ObsComposer(nn.Module):
    """Minimal observation composer matching the future multi-modal structure."""

    def __init__(
        self,
        *,
        encoders: Sequence[BaseObservationEncoder],
        concat: TokenLevelConcat,
    ) -> None:
        super().__init__()
        if not encoders:
            raise ValueError("ObsComposer requires at least one encoder")
        self.encoders = nn.ModuleList(encoders)
        self.concat = concat
        self._output_dim = int(concat.output_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _encode_all(self, obs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        token_map: dict[str, torch.Tensor] = {}
        for encoder in self.encoders:
            outputs = encoder(obs)
            token_map.update(outputs)
        return token_map

    def forward(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        token_map = self._encode_all(obs)
        encoded, _ = self.concat(token_map)
        return encoded


def build_rgb_obs_composer(
    *,
    image_keys: tuple[str, ...],
    encoder_type: str,
    d_model: int,
    timm_model_name: str,
    timm_pretrained: bool,
    timm_tokens_per_frame: int,
    timm_token_mode: str,
) -> ObsComposer:
    encoder_type = str(encoder_type)
    if encoder_type == "cnn":
        encoder = CNNRGBEncoder(image_keys=image_keys, d_model=d_model)
    elif encoder_type == "timm":
        encoder = TimmRGBEncoder(
            image_keys=image_keys,
            model_name=timm_model_name,
            pretrained=bool(timm_pretrained),
            tokens_per_frame=int(timm_tokens_per_frame),
            token_mode=str(timm_token_mode),
        )
    else:
        raise ValueError(f"Unsupported encoder_type: {encoder_type}")

    input_dims = {name: int(encoder.embedding_dim) for name in encoder.output_names}
    concat = TokenLevelConcat(order=list(encoder.output_names), input_dims=input_dims, output_dim=int(d_model))
    return ObsComposer(encoders=[encoder], concat=concat)
