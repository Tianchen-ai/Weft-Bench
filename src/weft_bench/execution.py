from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import uuid


class ExecutionError(RuntimeError):
    def __init__(self, stage, command, completed):
        self.stage = stage
        self.completed = completed
        super().__init__(
            f"{stage}: exit {completed.returncode}: {shlex.join(command)}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )


class Executor:
    """Transport only. Native programs own measurement and numerical checks."""

    def __init__(self, target, local_directory):
        self.target = target
        self.settings = target.settings
        self.local = Path(local_directory)
        self.local.mkdir(parents=True, exist_ok=True)
        deployment = self.settings.get("deployment", {})
        self.host = deployment.get("host")
        self.cpus = deployment.get("cpus", [])
        self.timeout = self.settings.get("timeout", 1800)
        self.env = {str(k): str(v) for k, v in self.settings.get("env", {}).items()}
        root = deployment.get("workspace", str(self.local.resolve()))
        self.workspace = (
            str(Path(root) / ("run-" + uuid.uuid4().hex[:12]))
            if self.host
            else str(self.local.resolve())
        )
        self.commands = []

    def command(
        self, argv, *, cwd=None, env=None, affinity=False, stage="run", check=True
    ):
        command = [str(x) for x in argv]
        variables = {**self.env, **(env or {})}
        if affinity and self.cpus:
            command = ["taskset", "-c", ",".join(map(str, self.cpus)), *command]
        if self.host:
            command = ["env", *[f"{k}={v}" for k, v in variables.items()], *command]
            shell = shlex.join(command)
            if cwd:
                shell = "cd " + shlex.quote(str(cwd)) + " && " + shell
            command = ["ssh", self.host, shell]
            process_env = None
            process_cwd = None
        else:
            process_env = {**os.environ, **variables}
            process_cwd = cwd
        self.commands.append(command)
        result = subprocess.run(
            command,
            cwd=process_cwd,
            env=process_env,
            text=True,
            capture_output=True,
            timeout=self.timeout,
        )
        if check and result.returncode:
            raise ExecutionError(stage, command, result)
        return result

    def upload(self, source, relative):
        source = Path(source)
        # Retain the exact worker/header inputs with the run, including SSH runs.
        import shutil

        snapshot = self.local / relative
        if source.resolve() != snapshot.resolve():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, snapshot)
        source = snapshot
        destination = str(Path(self.workspace) / relative)
        if not self.host:
            if source.resolve() != Path(destination).resolve():
                Path(destination).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            return destination
        self.command(["mkdir", "-p", str(Path(destination).parent)], stage="prepare")
        with source.open("rb") as stream:
            result = subprocess.run(
                ["ssh", self.host, "cat > " + shlex.quote(destination)],
                stdin=stream,
                capture_output=True,
                timeout=self.timeout,
            )
        if result.returncode:
            raise RuntimeError(result.stderr.decode(errors="replace"))
        return destination

    def write(self, relative, text):
        path = self.local / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return self.upload(path, relative)

    def download(self, source, destination):
        import shutil

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.host:
            with destination.open("wb") as stream:
                result = subprocess.run(
                    ["ssh", self.host, "cat " + shlex.quote(str(source))],
                    stdout=stream,
                    stderr=subprocess.PIPE,
                    timeout=self.timeout,
                )
            if result.returncode:
                raise RuntimeError(result.stderr.decode(errors="replace"))
        elif Path(source).resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return destination
