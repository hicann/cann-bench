"""Resolve a task's evaluation adapter and retain the backend report unchanged."""

import os
from pathlib import Path
import subprocess
import sys

from .adapters.python_example import plan as example_plan
from .contracts import ContractError, RunResult


class Engine:
    def __init__(self, catalog, adapters=None):
        self.catalog = catalog
        self.adapters = {"python_example": example_plan}
        self.adapters.update(adapters or {})

    def plan(self, task_id, submission, reports_dir, *, suite_id=None, **options):
        task = self.catalog.task(task_id)
        suite = self.catalog.suite(suite_id) if suite_id else None
        if suite and not any(member["task"] == task.id for member in suite["members"]):
            raise ContractError(f"{task_id} is not a member of {suite_id}")
        if submission.kind not in task.data["evaluation"]["submission_kinds"]:
            raise ContractError(f"{task_id} does not accept {submission.kind} submissions")
        adapter = self.adapters.get(task.adapter)
        if adapter is None:
            raise ContractError(f"{task_id}: no executable adapter registered for {task.adapter!r}")
        command = adapter(self.catalog.root, task, submission, reports_dir, **options)
        return task, suite, command

    def run(self, task_id, submission, reports_dir, *, suite_id=None, **options):
        task, suite, command = self.plan(task_id, submission, reports_dir, suite_id=suite_id, **options)
        reports_dir = Path(reports_dir).resolve()
        # A per-run directory keeps old or concurrent reports out of this result.
        reports_dir.mkdir(parents=True, exist_ok=False)
        env = os.environ.copy()
        paths = [str(self.catalog.root / "src"), str(self.catalog.root)]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        # Keep the CLI's stdout available for its structured execution result.
        completed = subprocess.run(command, cwd=self.catalog.root, env=env, stdout=sys.stderr, check=False)
        reports = tuple(sorted(path.name for path in reports_dir.glob("*.json") if path.is_file()))
        return RunResult(task.id, task.version, suite_id, suite["version"] if suite else None,
                         task.adapter, completed.returncode, reports)
