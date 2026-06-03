"""OpenCode runner."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Mapping, Optional, TextIO

from auto_pipeline.generator._subprocess import guess_output_files, resolve_executable, write_prompt_file
from auto_pipeline.generator.opencode.exporter import append_export_result_to_log, export_session_from_log
from auto_pipeline.generator.opencode.live_bridge import OpencodeLiveBridge
from auto_pipeline.core import AGENT_FAILED, AGENT_NOT_FOUND, AGENT_SUCCESS, AGENT_TIMEOUT, Artifact, RunnerPrompt
from auto_pipeline.core import cleanup_process_on_exit, terminate_process_family


_POLL_INTERVAL_SEC = 5


@dataclass(frozen=True)
class OpenCodeRunResult:
    """Low-level result from one ``opencode run`` invocation."""

    status: str
    returncode: int
    timed_out: bool
    started: bool
    message: str
    log_file: Path
    prompt_file: Path
    live_bridge: Mapping[str, object]
    session_export: Mapping[str, object]

    @property
    def ok(self) -> bool:
        return self.status == AGENT_SUCCESS


class OpenCodeAgent:
    """Runs ``opencode run`` and leaves benchmark semantics to callers."""

    type = "opencode"

    def __init__(
        self,
        *,
        opencode_bin: str = "opencode",
        skill: str = "",
        model: str = "",
        output_format: str = "default",
        dangerously_skip_permissions: bool = True,
        extra_args: Optional[List[str]] = None,
    ) -> None:
        self.opencode_bin = opencode_bin
        self.skill = skill
        self.model = model
        self.output_format = output_format
        self.dangerously_skip_permissions = dangerously_skip_permissions
        self.extra_args = list(extra_args or [])

    def run(self, prompt: RunnerPrompt) -> Artifact:
        result = self.run_opencode(prompt)
        metadata = {"returncode": result.returncode, "timed_out": result.timed_out}
        if result.live_bridge:
            metadata["opencode_live_bridge"] = dict(result.live_bridge)
        if result.session_export:
            metadata["opencode_session"] = dict(result.session_export)
        return Artifact(
            status=result.status,
            workdir=prompt.output_dir,
            message=result.message,
            log_file=result.log_file,
            files=guess_output_files(prompt.output_dir, prompt.cwd),
            metadata=metadata,
        )

    def run_opencode(
        self,
        prompt: RunnerPrompt,
        *,
        cwd: Optional[Path] = None,
        prompt_text: Optional[str] = None,
        log_name: Optional[str] = None,
        session_title: str = "",
        extra_env: Optional[Mapping[str, str]] = None,
        tmpdir: Optional[Path] = None,
        live_bridge: bool = False,
        export_session: bool = False,
        deny_external_directory: bool = True,
    ) -> OpenCodeRunResult:
        """Run OpenCode with runner-owned env, logging, live bridge, and export."""

        effective_prompt = prompt
        if cwd is not None or prompt_text is not None:
            effective_prompt = replace(
                prompt,
                cwd=Path(cwd).expanduser().resolve() if cwd is not None else prompt.cwd,
                text=prompt.text if prompt_text is None else str(prompt_text),
            )
        effective_prompt.cwd.mkdir(parents=True, exist_ok=True)
        effective_prompt.output_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = write_prompt_file(effective_prompt)
        log_file = effective_prompt.output_dir / (log_name or f"{self.type}.log")
        title = session_title or effective_prompt.title

        executable = resolve_executable(self.opencode_bin)
        command_executable = executable or self.opencode_bin
        command = self._build_command(command_executable, effective_prompt, title=title)
        display = self._build_command(command_executable, effective_prompt, title=title, hide_prompt=True)

        env = self._build_env(
            effective_prompt,
            extra_env=extra_env,
            tmpdir=tmpdir,
            deny_external_directory=deny_external_directory,
        )
        bridge = OpencodeLiveBridge(output_dir=effective_prompt.output_dir) if live_bridge else None
        if bridge is not None:
            bridge.configure_env(env)

        if executable is None:
            message = f"opencode executable not found: {self.opencode_bin}"
            _write_command_log(log_file, effective_prompt.cwd, display, message)
            return OpenCodeRunResult(
                status=AGENT_NOT_FOUND,
                returncode=-1,
                timed_out=False,
                started=False,
                message=message,
                log_file=log_file,
                prompt_file=prompt_file,
                live_bridge=bridge.summary() if bridge is not None else {},
                session_export={},
            )

        timeout = effective_prompt.timeout_sec if effective_prompt.timeout_sec and effective_prompt.timeout_sec > 0 else None
        try:
            returncode, timed_out = _run_streaming_process(
                command=command,
                cwd=effective_prompt.cwd,
                env=env,
                timeout=timeout,
                log_file=log_file,
                display_command=display,
                live_bridge=bridge,
            )
        except OSError as exc:
            message = f"opencode failed to start: {exc}"
            live_summary = bridge.summary() if bridge is not None else {}
            if live_summary:
                _append_live_bridge_footer(log_file, live_summary)
            _append_log_footer(log_file, message)
            return OpenCodeRunResult(
                status=AGENT_FAILED,
                returncode=-1,
                timed_out=False,
                started=False,
                message=message,
                log_file=log_file,
                prompt_file=prompt_file,
                live_bridge=live_summary,
                session_export={},
            )

        live_summary = bridge.summary() if bridge is not None else {}
        if live_summary:
            _append_live_bridge_footer(log_file, live_summary)

        session_export = {}
        if export_session:
            session_export_result = export_session_from_log(
                log_file=log_file,
                output_file=effective_prompt.output_dir / "opencode-session.md",
                output_dir=effective_prompt.output_dir / "opencode-session",
                raw_json_file=effective_prompt.output_dir / "opencode-session.json",
                session_title=title,
                opencode_bin=executable,
                cwd=effective_prompt.cwd,
                timeout_sec=120,
            )
            append_export_result_to_log(log_file, session_export_result, label="opencode")
            session_export = session_export_result.to_dict()
            _append_session_export_footer(log_file, session_export)

        status = AGENT_TIMEOUT if timed_out else AGENT_SUCCESS if returncode == 0 else AGENT_FAILED
        message = (
            f"opencode timed out after {effective_prompt.timeout_sec}s"
            if timed_out
            else f"opencode exited with code {returncode}"
        )
        _append_log_footer(log_file, message)
        return OpenCodeRunResult(
            status=status,
            returncode=returncode,
            timed_out=timed_out,
            started=True,
            message=message,
            log_file=log_file,
            prompt_file=prompt_file,
            live_bridge=live_summary,
            session_export=session_export,
        )

    def _build_env(
        self,
        prompt: RunnerPrompt,
        *,
        extra_env: Optional[Mapping[str, str]],
        tmpdir: Optional[Path],
        deny_external_directory: bool,
    ) -> dict[str, str]:
        env = os.environ.copy()
        env.update(prompt.env)
        env.update({str(key): str(value) for key, value in dict(extra_env or {}).items()})
        env.update(
            {
                "BENCHMARK_PROMPT_FILE": str(prompt.output_dir / "PROMPT.md"),
                "BENCHMARK_OUTPUT_DIR": str(prompt.output_dir),
                "BENCHMARK_WORKDIR": str(prompt.cwd),
                "PWD": str(prompt.cwd),
            }
        )
        if tmpdir is not None:
            tmpdir = Path(tmpdir).expanduser().resolve()
            tmpdir.mkdir(parents=True, exist_ok=True)
            env["TMPDIR"] = str(tmpdir)
        if deny_external_directory:
            env["OPENCODE_PERMISSION"] = opencode_permission_without_external_asks(
                env.get("OPENCODE_PERMISSION")
            )
        return env

    def _build_command(
        self,
        executable: str,
        prompt: RunnerPrompt,
        *,
        title: str = "",
        hide_prompt: bool = False,
    ) -> List[str]:
        command = [executable, "run"]
        if self.dangerously_skip_permissions:
            command.append("--dangerously-skip-permissions")
        if self.skill:
            command.extend(["--agent", self.skill])
        if self.output_format:
            command.extend(["--format", self.output_format])
        title = title or prompt.title
        if title:
            command.extend(["--title", title])
        if self.model:
            command.extend(["-m", self.model])
        command.extend(self.extra_args)
        command.append("<prompt>" if hide_prompt else prompt.text)
        return command


def opencode_permission_without_external_asks(existing: Optional[str]) -> str:
    permission: dict[str, object] = {}
    if existing:
        try:
            parsed = json.loads(existing)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            permission.update(parsed)
    permission["external_directory"] = "deny"
    return json.dumps(permission, sort_keys=True)


def _run_streaming_process(
    *,
    command: list[str],
    cwd: Path,
    env: Mapping[str, str],
    timeout: Optional[int],
    log_file: Path,
    display_command: list[str],
    live_bridge: Optional[OpencodeLiveBridge] = None,
) -> tuple[int, bool]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"$ cd {cwd}\n")
        log.write(f"$ {shlex.join([str(part) for part in display_command])}\n")
        log.flush()

        if live_bridge is not None:
            live_bridge.start()
        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            with cleanup_process_on_exit(
                process,
                on_signal=lambda signum: _log_signal_cleanup(log, signum),
                match_environ={"BENCHMARK_OUTPUT_DIR": env["BENCHMARK_OUTPUT_DIR"]},
            ):
                reader = threading.Thread(target=_stream_stdout, args=(process, log), daemon=True)
                reader.start()
                deadline = time.monotonic() + timeout if timeout else None
                timed_out = False
                while process.poll() is None:
                    if deadline is not None and time.monotonic() >= deadline:
                        timed_out = True
                        terminate_process_family(process)
                        break
                    time.sleep(_POLL_INTERVAL_SEC)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    terminate_process_family(process)
                reader.join(timeout=5)
                log.write(f"\n[run finished, returncode={process.returncode}, timed_out={timed_out}]\n")
        finally:
            if live_bridge is not None:
                live_bridge.stop()
    return process.returncode if process.returncode is not None else -9, timed_out


def _stream_stdout(process: subprocess.Popen, log: TextIO) -> None:
    if process.stdout is None:
        return
    try:
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            log.write(line)
            log.flush()
    except (ValueError, OSError):
        return


def _log_signal_cleanup(log: TextIO, signum: int) -> None:
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = f"signal {signum}"
    log.write(f"\n[received {name}; terminating opencode process family]\n")
    log.flush()


def _write_command_log(path: Path, cwd: Path, command: list[str], message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"$ cd {cwd}",
                f"$ {shlex.join([str(part) for part in command])}",
                f"# {message}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _append_log_footer(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n# {message}\n")


def _append_session_export_footer(path: Path, export: Mapping[str, object]) -> None:
    status = export.get("status")
    session_id = export.get("session_id") or ""
    markdown_file = export.get("markdown_file") or ""
    json_file = export.get("json_file") or ""
    session_count = export.get("node_session_count") or 0
    message = export.get("message") or ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n# opencode session export: "
            f"status={status} session_id={session_id} "
            f"markdown_file={markdown_file} json_file={json_file} "
            f"session_count={session_count} message={message}\n"
        )


def _append_live_bridge_footer(path: Path, summary: Mapping[str, object]) -> None:
    status = summary.get("status")
    live_dir = summary.get("live_dir") or ""
    session_tree_file = summary.get("session_tree_file") or ""
    session_count = summary.get("node_session_count") or 0
    subagent_count = summary.get("subagent_session_count") or 0
    event_count = summary.get("event_count") or 0
    error = summary.get("error") or ""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n# opencode live bridge: "
            f"status={status} live_dir={live_dir} "
            f"session_tree_file={session_tree_file} "
            f"session_count={session_count} subagent_count={subagent_count} "
            f"event_count={event_count} error={error}\n"
        )
