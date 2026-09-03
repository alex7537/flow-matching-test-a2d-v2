#!/usr/bin/env python3
"""One-shot successor state machine: completed CFM -> cache -> smokes -> Joint WAM."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime
from pathlib import Path


WAN_COMMIT = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
WAN_ARCHIVE_URL = f"https://github.com/Wan-Video/Wan2.2/archive/{WAN_COMMIT}.tar.gz"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class Successor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.control_dir = Path(args.control_dir).expanduser().resolve()
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.control_dir / "state.json"
        self.events_path = self.control_dir / "events.jsonl"
        self.stop_path = self.control_dir / "STOP"
        self.lock_stream = (self.control_dir / "runner.lock").open("a+")
        try:
            fcntl.flock(self.lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another successor state machine already owns the lock") from exc
        self.root = self.control_dir / "workspace"
        self.code_dir = self.root / "code"
        self.venv_dir = self.root / "venv"
        self.wan_dir = Path(args.wan_runtime_dir).expanduser().resolve()
        self.cache_dir = Path(args.cache_dir).expanduser().resolve()
        self.run_root = Path(args.run_root).expanduser().resolve()
        self.ledger_cli = Path(args.ledger_cli).expanduser().resolve()
        self.ledger_path = Path(args.ledger).expanduser().resolve()
        self.active_phase = "infrastructure"

    def ledger(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, str(self.ledger_cli), *arguments, "--path", str(self.ledger_path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"lifecycle ledger command failed: {' '.join(arguments)}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        return result

    def ledger_record(
        self,
        phase: str,
        status: str,
        evidence: str,
        *artifacts: str,
    ) -> None:
        command = [
            "record",
            "--phase",
            phase,
            "--status",
            status,
            "--skill",
            "robot-ml-lifecycle",
            "--evidence",
            evidence,
        ]
        for artifact in artifacts:
            command.extend(["--artifact", artifact])
        self.ledger(*command)

    def ledger_attempt(
        self,
        *,
        phase: str,
        action: str,
        outcome: str,
        error: str = "",
        evidence: str = "",
    ) -> None:
        command = [
            "attempt",
            "--phase",
            phase,
            "--action",
            action,
            "--outcome",
            outcome,
        ]
        if error:
            command.extend(["--error", error])
        if evidence:
            command.extend(["--evidence", evidence])
        self.ledger(*command)

    def event(self, phase: str, status: str, **details: object) -> None:
        payload = {"time": now(), "phase": phase, "status": status, **details}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, self.state_path)
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    def check_stop(self) -> None:
        if self.stop_path.exists():
            self.event("control", "stopped", reason=f"kill switch present: {self.stop_path}")
            raise SystemExit(2)

    @staticmethod
    def trainer_commands() -> list[str]:
        commands = []
        for path in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                command = path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except (OSError, PermissionError):
                continue
            if "flow_matching_test.train" in command:
                commands.append(command)
        return commands

    @staticmethod
    def gpu_processes() -> list[str]:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def wait_for_predecessor(self) -> None:
        run = Path(self.args.predecessor_run).resolve()
        summary_path = run / "summary.json"
        failure_path = run / "failure.json"
        self.event(
            "wait_predecessor",
            "waiting",
            run=str(run),
            expected_run_name=self.args.predecessor_name,
        )
        while True:
            self.check_stop()
            if failure_path.exists():
                raise RuntimeError(f"predecessor failure marker exists: {failure_path}")
            if summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("run_name") != self.args.predecessor_name:
                    raise RuntimeError("predecessor summary run_name mismatch")
                metrics_path = run / "metrics.jsonl"
                rows = [
                    json.loads(line)
                    for line in metrics_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if len(rows) != 100 or int(rows[-1].get("epoch", -1)) != 99:
                    raise RuntimeError("predecessor did not complete exactly 100 epochs")
                break
            matching = [
                command
                for command in self.trainer_commands()
                if self.args.predecessor_name in command
            ]
            if not matching:
                raise RuntimeError("predecessor exited before writing a valid summary")
            time.sleep(self.args.poll_seconds)

        deadline = time.time() + 600
        while any(self.args.predecessor_name in cmd for cmd in self.trainer_commands()):
            if time.time() >= deadline:
                raise RuntimeError("predecessor summary exists but trainer did not exit")
            time.sleep(10)
        if self.trainer_commands():
            raise RuntimeError("another flow_matching_test.train process is active")
        deadline = time.time() + 600
        while self.gpu_processes() and time.time() < deadline:
            time.sleep(10)
        if self.gpu_processes():
            raise RuntimeError("GPU is occupied after predecessor completion")
        self.event("wait_predecessor", "passed", summary=str(summary_path))

    def run_command(
        self,
        *,
        phase: str,
        command: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.check_stop()
        log_path = self.control_dir / f"{phase}.log"
        self.event(phase, "running", log=str(log_path), command=command)
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"{phase} failed with exit code {result.returncode}; see {log_path}")
        self.event(phase, "passed", log=str(log_path))

    def prepare_source(self) -> None:
        archive = Path(self.args.source_archive).expanduser().resolve()
        if sha256_file(archive) != self.args.source_sha256:
            raise RuntimeError("source archive SHA256 mismatch")
        self.root.mkdir(parents=True, exist_ok=True)
        if self.code_dir.exists():
            marker = self.code_dir / ".source_sha256"
            if not marker.exists() or marker.read_text().strip() != self.args.source_sha256:
                raise RuntimeError("existing successor code directory has different provenance")
        else:
            temporary = self.root / "code.partial"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir()
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(temporary, filter="data")
            (temporary / ".source_sha256").write_text(self.args.source_sha256 + "\n")
            os.replace(temporary, self.code_dir)
        self.event("prepare_source", "passed", source_sha256=self.args.source_sha256)

    def prepare_wan(self) -> None:
        marker = self.wan_dir / ".source_commit"
        if self.wan_dir.exists():
            if not marker.exists() or marker.read_text().strip() != WAN_COMMIT:
                raise RuntimeError("existing Wan runtime directory has different provenance")
            return
        self.wan_dir.parent.mkdir(parents=True, exist_ok=True)
        archive = self.wan_dir.parent / f"Wan2.2-{WAN_COMMIT}.tar.gz"
        partial = archive.with_suffix(archive.suffix + ".partial")
        self.run_command(
            phase="download_public_wan",
            command=["wget", "-O", str(partial), WAN_ARCHIVE_URL],
        )
        os.replace(partial, archive)
        temporary = self.wan_dir.parent / "Wan2.2.partial"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(temporary, filter="data")
        children = [path for path in temporary.iterdir() if path.is_dir()]
        if len(children) != 1 or not (children[0] / "wan/modules/vae2_2.py").is_file():
            raise RuntimeError("downloaded Wan archive has unexpected layout")
        os.replace(children[0], self.wan_dir)
        shutil.rmtree(temporary)
        marker.write_text(WAN_COMMIT + "\n")
        self.event(
            "prepare_wan",
            "passed",
            commit=WAN_COMMIT,
            archive_sha256=sha256_file(archive),
        )

    def prepare_environment(self) -> None:
        if not self.venv_dir.exists():
            self.run_command(
                phase="create_venv",
                command=[
                    "/opt/conda/bin/python3.11",
                    "-m",
                    "venv",
                    "--system-site-packages",
                    str(self.venv_dir),
                ],
            )
        self.run_command(
            phase="install_environment",
            command=[
                str(self.venv_dir / "bin/pip"),
                "install",
                "-r",
                str(self.code_dir / "requirements.lock.a800.txt"),
            ],
        )
        self.run_command(
            phase="verify_environment",
            command=[
                str(self.venv_dir / "bin/python"),
                "-c",
                "import torch,h5py,cv2,timm,yaml,wandb,einops; "
                "assert torch.cuda.is_available(); print(torch.__version__)",
            ],
        )

    def build_cache(self) -> None:
        self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        self.run_command(
            phase="build_joint_latent_cache",
            cwd=self.code_dir,
            env=environment,
            command=[
                str(self.venv_dir / "bin/python"),
                "-u",
                "-m",
                "scripts.precompute_wan_joint_latents",
                "--data-dir",
                self.args.data_dir,
                "--cache-dir",
                str(self.cache_dir),
                "--vae-checkpoint",
                self.args.vae_checkpoint,
                "--wan-runtime-repo",
                str(self.wan_dir),
                "--batch-size",
                "4",
            ],
        )
        manifest = json.loads(
            (self.cache_dir / "joint_video_latent_manifest.json").read_text()
        )
        if int(manifest.get("episode_count", -1)) != 600:
            raise RuntimeError("joint cache episode count must be 600")
        if int(manifest.get("total_windows", -1)) != 93771:
            raise RuntimeError("joint cache total_windows must be 93771")
        self.event(
            "verify_joint_latent_cache",
            "passed",
            manifest_sha256=sha256_file(
                self.cache_dir / "joint_video_latent_manifest.json"
            ),
        )

    @staticmethod
    def verify_run(run_dir: Path, expected_steps: int | None = None) -> None:
        failure = run_dir / "failure.json"
        summary = run_dir / "summary.json"
        metrics = run_dir / "metrics.jsonl"
        if failure.exists() or not summary.exists() or not metrics.exists():
            raise RuntimeError(f"run gate failed for {run_dir}")
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        if not rows:
            raise RuntimeError(f"run has no metrics: {run_dir}")
        latest = rows[-1]
        if expected_steps is not None and int(latest.get("global_step", -1)) != expected_steps:
            raise RuntimeError(f"unexpected global_step for {run_dir}")
        required = (
            "train_loss",
            "train_action_flow_loss",
            "train_video_flow_loss",
            "val_loss",
            "val_action_flow_loss",
            "val_video_flow_loss",
            "grad_norm_head_mean",
            "grad_norm_backbone_mean",
        )
        for key in required:
            value = float(latest[key])
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite {key} in {run_dir}")
        if latest["train_action_flow_loss"] <= 0 or latest["train_video_flow_loss"] <= 0:
            raise RuntimeError(f"both flow losses must be positive in {run_dir}")
        if latest["grad_norm_head_mean"] <= 0 or latest["grad_norm_backbone_mean"] <= 0:
            raise RuntimeError(f"head/backbone gradients must both be nonzero in {run_dir}")

    def training_command(
        self,
        *,
        run_name: str,
        run_dir: Path,
        overrides: list[str],
    ) -> list[str]:
        return [
            str(self.venv_dir / "bin/python"),
            "-u",
            "-m",
            "flow_matching_test.train",
            "--config",
            "configs/a2d_v3_multitask_joint_wam_scratch_100ep.yaml",
            f"data.data_dir={self.args.data_dir}",
            f"data.joint_video_latent_cache_dir={self.cache_dir}",
            f"training.run_name={run_name}",
            f"training.output_dir={run_dir}",
            "visualization.rerun.enabled=false",
            *overrides,
        ]

    def run_smokes(self) -> None:
        specs = [
            (
                "smoke_1step",
                [
                    "training.num_epochs=1",
                    "training.max_train_steps=1",
                    "training.max_val_steps=1",
                    "training.warmup_steps=0",
                    "training.ema_enabled=false",
                    "logging.wandb.enabled=false",
                ],
                1,
            ),
            (
                "smoke_100step",
                [
                    "training.num_epochs=1",
                    "training.max_train_steps=100",
                    "training.max_val_steps=10",
                    "training.warmup_steps=10",
                    "training.ema_enabled=false",
                    "logging.wandb.enabled=false",
                ],
                100,
            ),
            (
                "smoke_1epoch",
                [
                    "training.num_epochs=1",
                    "training.max_train_steps=null",
                    "training.max_val_steps=null",
                    "training.warmup_steps=180",
                    "logging.wandb.enabled=false",
                ],
                3603,
            ),
        ]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        for label, overrides, expected_steps in specs:
            run_dir = self.run_root / f"joint_wam_{label}_{self.args.source_sha256[:8]}"
            if run_dir.exists():
                raise RuntimeError(f"smoke output already exists: {run_dir}")
            command = self.training_command(
                run_name=f"joint_wam_{label}_{self.args.source_sha256[:8]}",
                run_dir=run_dir,
                overrides=overrides,
            )
            self.run_command(
                phase=label,
                command=command,
                cwd=self.code_dir,
                env=environment,
            )
            self.verify_run(run_dir, expected_steps=expected_steps)
            self.event(f"verify_{label}", "passed", run_dir=str(run_dir))

    def launch_formal(self) -> None:
        if self.trainer_commands() or self.gpu_processes():
            raise RuntimeError("GPU/trainer became occupied before formal launch")
        run_name = f"joint_wam_multitask_box300_bottle300_v3_scratch_100ep_seed42_{self.args.source_sha256[:8]}"
        run_dir = self.run_root / run_name
        if run_dir.exists():
            raise RuntimeError(f"formal output already exists: {run_dir}")
        command = self.training_command(
            run_name=run_name,
            run_dir=run_dir,
            overrides=["logging.wandb.mode=online"],
        )
        log_path = self.control_dir / "formal_launcher.log"
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=str(self.code_dir),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        self.event(
            "formal_train",
            "starting",
            pid=process.pid,
            run_name=run_name,
            run_dir=str(run_dir),
            log=str(log_path),
            planned_epochs=100,
            planned_steps=360300,
            warmup_steps=18015,
        )
        try:
            deadline = time.time() + 300
            while time.time() < deadline:
                self.check_stop()
                if process.poll() is not None:
                    raise RuntimeError(
                        f"formal trainer exited during startup: {process.returncode}"
                    )
                if (run_dir / "failure.json").exists():
                    raise RuntimeError("formal trainer wrote failure.json during startup")
                if run_dir.exists() and self.gpu_processes():
                    self.event(
                        "formal_train",
                        "running",
                        pid=process.pid,
                        run_name=run_name,
                        run_dir=str(run_dir),
                        log=str(log_path),
                    )
                    return
                time.sleep(10)
            raise RuntimeError("formal trainer did not become GPU-active within 300 seconds")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            raise

    def run(self) -> None:
        try:
            self.ledger("check")
            self.wait_for_predecessor()
            self.prepare_source()
            self.prepare_wan()
            self.prepare_environment()
            self.build_cache()
            self.ledger_record(
                "infrastructure",
                "passed",
                "Predecessor completed; isolated source, Wan runtime, venv, and latent cache verified",
                f"source_sha256={self.args.source_sha256}",
                f"cache_manifest={self.cache_dir / 'joint_video_latent_manifest.json'}",
            )
            self.active_phase = "train"
            self.ledger_record(
                "train",
                "in_progress",
                "Beginning bounded 1-step, 100-step, and 1-epoch gates before formal launch",
            )
            self.run_smokes()
            self.ledger_attempt(
                phase="train",
                action="Joint WAM bounded smoke ladder",
                outcome="success",
                evidence="1-step, 100-step, and 1-epoch deterministic gates passed",
            )
            self.launch_formal()
            self.ledger_attempt(
                phase="train",
                action="Launch Joint WAM scratch 100-epoch formal run",
                outcome="success",
                evidence="Trainer remained alive and GPU-active during bounded startup check",
            )
        except SystemExit:
            raise
        except Exception as exc:
            try:
                self.ledger_attempt(
                    phase=self.active_phase,
                    action="Execute queued Joint WAM successor",
                    outcome="failure",
                    error=f"{type(exc).__name__}: {exc}",
                    evidence=str(self.state_path),
                )
            except Exception:
                pass
            self.event(
                "state_machine",
                "failed",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", required=True)
    parser.add_argument("--predecessor-run", required=True)
    parser.add_argument("--predecessor-name", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--wan-runtime-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--ledger-cli", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    Successor(parse_args()).run()
