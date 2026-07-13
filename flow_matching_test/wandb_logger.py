from __future__ import annotations

import copy
import json
import logging
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
        mode: str = "online",
        tags: list[str] | None = None,
        group: str | None = None,
        job_type: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.run = None
        self.mode = str(mode)

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
        self.run = wandb.init(**init_kwargs)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int) -> None:
        if not self.enabled or self.run is None:
            return
        payload = {str(key): float(value) for key, value in metrics.items()}
        wandb.log(payload, step=int(step))

    def update_summary(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled or self.run is None:
            return
        sanitized = _sanitize_config(copy.deepcopy(dict(payload)))
        for key, value in sanitized.items():
            self.run.summary[str(key)] = value

    def save_text(self, name: str, content: str) -> None:
        if not self.enabled or self.run is None:
            return
        artifact_path = Path(self.run.dir) / str(name)
        artifact_path.write_text(content, encoding="utf-8")
        wandb.save(str(artifact_path), base_path=str(Path(self.run.dir)))

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        wandb.finish()
        self.run = None

    @property
    def run_url(self) -> str | None:
        if not self.enabled or self.run is None:
            return None
        try:
            return str(self.run.url)
        except Exception:
            return None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": self.mode,
            "run_url": self.run_url,
        }
