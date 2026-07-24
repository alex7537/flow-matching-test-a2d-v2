from __future__ import annotations

import threading

from flow_matching_test.training_watchdog import TrainingWatchdog


def test_watchdog_reports_stall_and_terminates() -> None:
    reported = []
    terminated = threading.Event()
    exit_codes = []

    def terminate(exit_code: int) -> None:
        exit_codes.append(exit_code)
        terminated.set()

    watchdog = TrainingWatchdog(
        timeout_seconds=0.05,
        check_interval_seconds=0.01,
        on_stall=reported.append,
        terminate=terminate,
    )
    watchdog.heartbeat(stage="train_epoch_2", step=123)
    watchdog.start()

    assert terminated.wait(timeout=1.0)
    watchdog.stop()
    assert exit_codes == [1]
    assert len(reported) == 1
    assert reported[0].stage == "train_epoch_2"
    assert reported[0].step == 123
    assert "no batch progress" in reported[0].message
