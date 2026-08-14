from __future__ import annotations

import logging

from flow_matching_test import wandb_logger as wandb_module


def test_wandb_environment_override_and_failure_degrade(monkeypatch, caplog, tmp_path) -> None:
    class FailingRun:
        def __init__(self) -> None:
            self.calls = 0

        def log(self, payload, *, step) -> None:
            self.calls += 1
            assert payload == {"loss": 1.0}
            assert step == 3
            raise RuntimeError("simulated outage")

    class FakeWandb:
        def __init__(self) -> None:
            self.kwargs = None
            self.run = FailingRun()

        def init(self, **kwargs):
            self.kwargs = kwargs
            return self.run

    fake = FakeWandb()
    monkeypatch.setattr(wandb_module, "wandb", fake)
    monkeypatch.setenv("WANDB_MODE", "online")
    logger = wandb_module.WandbLogger(
        enabled=True,
        output_dir=str(tmp_path),
        project="test",
        entity=None,
        run_name="test-run",
        mode="offline",
    )
    assert logger.mode == "online"
    assert fake.kwargs["mode"] == "online"

    with caplog.at_level(logging.WARNING):
        logger.log_metrics({"loss": 1.0, "missing": None}, step=3)
        logger.log_metrics({"loss": 2.0}, step=4)

    assert fake.run.calls == 1
    assert not logger.enabled
    assert caplog.text.count("disabling wandb for this run") == 1


def test_wandb_alert_and_failed_finish(monkeypatch, tmp_path) -> None:
    class FakeRun:
        def __init__(self) -> None:
            self.alert_kwargs = None
            self.exit_code = None

        def alert(self, **kwargs) -> None:
            self.alert_kwargs = kwargs

        def finish(self, *, exit_code=None) -> None:
            self.exit_code = exit_code

    class FakeAlertLevel:
        ERROR = "error-level"

    class FakeWandb:
        AlertLevel = FakeAlertLevel

        def __init__(self) -> None:
            self.run = FakeRun()

        def init(self, **kwargs):
            return self.run

    fake = FakeWandb()
    monkeypatch.setattr(wandb_module, "wandb", fake)
    logger = wandb_module.WandbLogger(
        enabled=True,
        output_dir=str(tmp_path),
        project="test",
        entity=None,
        run_name="test-run",
    )

    logger.alert(title="stalled", text="no heartbeat", level="ERROR")
    logger.finish(exit_code=1)

    assert fake.run.alert_kwargs == {
        "title": "stalled",
        "text": "no heartbeat",
        "level": "error-level",
    }
    assert fake.run.exit_code == 1
