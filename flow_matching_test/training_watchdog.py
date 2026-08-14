from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class StallDetails:
    stage: str
    step: int
    idle_seconds: float
    timeout_seconds: float

    @property
    def message(self) -> str:
        return (
            f"training made no batch progress for {self.idle_seconds:.1f}s "
            f"during {self.stage} at step {self.step}; "
            f"watchdog timeout is {self.timeout_seconds:.1f}s"
        )


class TrainingWatchdog:
    """Terminate a training process that stops producing batch heartbeats."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        check_interval_seconds: float,
        on_stall: Callable[[StallDetails], None],
        terminate: Callable[[int], None] = os._exit,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be > 0")
        self.timeout_seconds = float(timeout_seconds)
        self.check_interval_seconds = float(check_interval_seconds)
        self.on_stall = on_stall
        self.terminate = terminate
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_heartbeat = time.monotonic()
        self._stage = "startup"
        self._step = 0

    def heartbeat(self, *, stage: str, step: int) -> None:
        with self._lock:
            self._last_heartbeat = time.monotonic()
            self._stage = str(stage)
            self._step = int(step)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("watchdog already started")
        self._thread = threading.Thread(
            target=self._run,
            name="training-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=self.check_interval_seconds + 1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.check_interval_seconds):
            with self._lock:
                idle_seconds = time.monotonic() - self._last_heartbeat
                stage = self._stage
                step = self._step
            if idle_seconds < self.timeout_seconds:
                continue
            details = StallDetails(
                stage=stage,
                step=step,
                idle_seconds=idle_seconds,
                timeout_seconds=self.timeout_seconds,
            )
            try:
                self.on_stall(details)
            finally:
                self.terminate(1)
            return
