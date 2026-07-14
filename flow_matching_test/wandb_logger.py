from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

try:
    import wandb
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    wandb = None


logger = logging.getLogger(__name__)


def _sanitize_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_config(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class WandbLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: str,
        project: str | None,
        entity: str | None,
        run_name: str,
        mode: str = "offline",
        tags: list[str] | None = None,
        group: str | None = None,
        job_type: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.run = None
        self.mode = os.environ.get("WANDB_MODE", str(mode))

        if not self.enabled:
            return
        if wandb is None:
            logger.warning("wandb logging was enabled but wandb is not installed; disable wandb logging.")
            self.enabled = False
            return

        init_kwargs = {
            "project": project,
            "entity": entity,
            "name": run_name,
            "dir": str(Path(output_dir).expanduser().resolve()),
            "mode": self.mode,
            "tags": list(tags) if tags else None,
            "group": group,
            "job_type": job_type,
            "config": _sanitize_config(copy.deepcopy(dict(config or {}))),
        }
        init_kwargs = {key: value for key, value in init_kwargs.items() if value is not None}
        try:
            self.run = wandb.init(**init_kwargs)
        except Exception as exc:  # pragma: no cover - depends on external service
            self._disable("init", exc)

    def _disable(self, operation: str, exc: Exception) -> None:
        if self.enabled:
            logger.warning("wandb %s failed; disabling wandb for this run: %s", operation, exc)
        self.enabled = False
        self.run = None

    def log_metrics(self, metrics: Mapping[str, Any], *, step: int) -> None:
        if not self.enabled or self.run is None:
            return
        payload = {
            str(key): float(value)
            for key, value in metrics.items()
            if value is not None
        }
        try:
            self.run.log(payload, step=int(step))
        except Exception as exc:  # pragma: no cover - depends on external service
            self._disable("log", exc)

    def update_summary(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            sanitized = _sanitize_config(copy.deepcopy(dict(payload)))
            for key, value in sanitized.items():
                self.run.summary[str(key)] = value
        except Exception as exc:  # pragma: no cover - depends on external service
            self._disable("summary update", exc)

    def save_text(self, name: str, content: str) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            artifact_path = Path(self.run.dir) / str(name)
            artifact_path.write_text(content, encoding="utf-8")
            self.run.save(str(artifact_path), base_path=str(Path(self.run.dir)))
        except Exception as exc:  # pragma: no cover - depends on external service
            self._disable("artifact save", exc)

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.finish()
            self.run = None
        except Exception as exc:  # pragma: no cover - depends on external service
            self._disable("finish", exc)

    @property
    def run_url(self) -> str | None:
        if not self.enabled or self.run is None:
            return None
        try:
            return str(self.run.url)
        except Exception as exc:  # pragma: no cover - depends on external service
            self._disable("run URL read", exc)
            return None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "run_url": self.run_url,
        }
