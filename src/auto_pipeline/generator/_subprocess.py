"""Subprocess helpers for kernel agents."""

from __future__ import annotations

import os
import select
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from auto_pipeline.core import (
    AGENT_FAILED,
    AGENT_NOT_FOUND,
    AGENT_SUCCESS,
    AGENT_TIMEOUT,
    Artifact,
    RunnerPrompt,
)
from auto_pipeline.core import cleanup_process_on_exit, terminate_process_family


def resolve_executable(executable: str) -> Optional[str]:
    if not executable:
        return None
    path = Path(executable).expanduser()
    if path.exists():
        return str(path.resolve())
    return shutil.which(executable)


def write_prompt_file(prompt: RunnerPrompt) -> Path:
    prompt.output_dir.mkdir(parents=True, exist_ok=True)
    path = prompt.output_dir / "PROMPT.md"
    path.write_text(prompt.text, encoding="utf-8")
    return path


def run_agent_subprocess(
    *,
    agent_type: str,
    executable: Optional[str],
    command: List[str],
    prompt: RunnerPrompt,
    display_command: Optional[List[str]] = None,
) -> Artifact:
    prompt.cwd.mkdir(parents=True, exist_ok=True)
    prompt.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = prompt.output_dir / f"{agent_type}.log"

    if executable is None:
        message = f"{agent_type} executable not found"
        _write_log(log_file, prompt.cwd, display_command or command, "", footer=message)
        return Artifact(
            status=AGENT_NOT_FOUND,
            workdir=prompt.output_dir,
            message=message,
            log_file=log_file,
        )

    env = os.environ.copy()
    env.update(prompt.env)
    env.update(_prompt_env(prompt))
    timeout = prompt.timeout_sec if prompt.timeout_sec and prompt.timeout_sec > 0 else None

    try:
        returncode, timed_out = _run_with_streaming_log(
            command=command,
            cwd=prompt.cwd,
            env=env,
            timeout=timeout,
            log_file=log_file,
            display_command=display_command or command,
        )
    except OSError as exc:
        message = f"{agent_type} failed to start: {exc}"
        _write_log(log_file, prompt.cwd, display_command or command, "", footer=message)
        return Artifact(
            status=AGENT_FAILED,
            workdir=prompt.output_dir,
            message=message,
            log_file=log_file,
            files=guess_output_files(prompt.output_dir, prompt.cwd),
        )

    if timed_out:
        message = f"{agent_type} timed out after {prompt.timeout_sec}s"
        _append_log_footer(log_file, message)
        return Artifact(
            status=AGENT_TIMEOUT,
            workdir=prompt.output_dir,
            message=message,
            log_file=log_file,
            files=guess_output_files(prompt.output_dir, prompt.cwd),
        )

    status = AGENT_SUCCESS if returncode == 0 else AGENT_FAILED
    message = f"{agent_type} exited with code {returncode}"
    _append_log_footer(log_file, message)
    return Artifact(
        status=status,
        workdir=prompt.output_dir,
        message=message,
        log_file=log_file,
        files=guess_output_files(prompt.output_dir, prompt.cwd),
        metadata={"returncode": returncode},
    )


def _run_with_streaming_log(
    *,
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    timeout: Optional[int],
    log_file: Path,
    display_command: Iterable[str],
) -> tuple[int, bool]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    header = _render_log_header(cwd, display_command)
    log_file.write_text(header, encoding="utf-8")

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
    )
    assert process.stdout is not None

    with cleanup_process_on_exit(
        process,
        match_environ={"BENCHMARK_OUTPUT_DIR": env["BENCHMARK_OUTPUT_DIR"]},
    ):
        deadline = time.monotonic() + timeout if timeout else None
        stdout_fd = process.stdout.fileno()
        timed_out = False

        with log_file.open("ab", buffering=0) as log:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    terminate_process_family(process)
                    break

                wait_sec = 0.2
                if deadline is not None:
                    wait_sec = max(0.0, min(wait_sec, deadline - time.monotonic()))

                readable, _, _ = select.select([stdout_fd], [], [], wait_sec)
                if readable:
                    chunk = os.read(stdout_fd, 8192)
                    if chunk:
                        log.write(chunk)
                    elif process.poll() is not None:
                        break

                if process.poll() is not None:
                    _drain_pipe(stdout_fd, log)
                    break

            if timed_out:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    terminate_process_family(process)
                _drain_pipe(stdout_fd, log)

    return process.returncode if process.returncode is not None else -9, timed_out


def _drain_pipe(fd: int, log) -> None:
    while True:
        readable, _, _ = select.select([fd], [], [], 0)
        if not readable:
            return
        chunk = os.read(fd, 8192)
        if not chunk:
            return
        log.write(chunk)


def _prompt_env(prompt: RunnerPrompt) -> Dict[str, str]:
    prompt_file = prompt.output_dir / "PROMPT.md"
    return {
        "BENCHMARK_PROMPT_FILE": str(prompt_file),
        "BENCHMARK_OUTPUT_DIR": str(prompt.output_dir),
        "BENCHMARK_WORKDIR": str(prompt.cwd),
    }


def guess_output_files(output_dir: Path, cwd: Path) -> Dict[str, Path]:
    files: Dict[str, Path] = {}
    for root in (output_dir, cwd):
        source_dir = root / "submission"
        if source_dir.is_dir():
            files.setdefault("source_dir", source_dir.resolve())
        for name in ("impl.py",):
            candidate = root / name
            if candidate.is_file():
                files.setdefault("impl", candidate.resolve())
    for candidate in sorted(output_dir.glob("*_impl.py")):
        if candidate.is_file():
            files.setdefault("impl", candidate.resolve())
            break
    return files


def _render_log_header(cwd: Path, command: Iterable[str]) -> str:
    rendered = shlex.join([str(part) for part in command])
    return "\n".join([f"$ cd {cwd}", f"$ {rendered}", ""])


def _append_log_footer(path: Path, footer: str) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(f"\n# {footer}\n")


def _write_log(
    path: Path,
    cwd: Path,
    command: Iterable[str],
    body: str,
    *,
    footer: str = "",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_render_log_header(cwd, command)]
    if body:
        lines.extend([body.rstrip(), ""])
    if footer:
        lines.append(f"# {footer}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
