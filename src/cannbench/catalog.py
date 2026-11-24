"""Resolve task and suite identities without importing task implementations."""

from pathlib import Path

from .contracts import (
    OBJECTIVES, SCOPES, SUBMISSIONS, TASK_CLASSES, ContractError, TaskSpec,
    identifier, read_document, within,
)


class Catalog:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def task(self, task_id, *, check_assets=True):
        path = within(self.root, f"tasks/{identifier(task_id)}/task.yaml")
        data = read_document(path)
        try:
            if data["id"] != task_id or not isinstance(data["version"], str) or not data["version"]:
                raise ContractError(f"{task_id}: mismatched identity")
            if data["status"] not in TASK_CLASSES:
                raise ContractError(f"{task_id}: unknown status")
            objective, scope = data["objective"], data["scope"]
            if objective is not None and objective not in OBJECTIVES:
                raise ContractError(f"{task_id}: unknown objective")
            if scope is not None and scope not in SCOPES:
                raise ContractError(f"{task_id}: unknown scope")
            assets = data["assets"]
            execution = data["evaluation"]
            runtime = data["resources"]
            if not isinstance(assets, dict) or not isinstance(runtime, dict):
                raise ContractError(f"{task_id}: invalid assets or resources")
            if any(kind not in SUBMISSIONS for kind in execution["submission_kinds"]):
                raise ContractError(f"{task_id}: unknown submission kind")
            scoring_rule = execution.get("scoring_rule")
            if scoring_rule is not None:
                rule = read_document(within(self.root, scoring_rule))
                identifier(rule["id"])
                if type(rule["version"]) is not int or rule["version"] < 1:
                    raise ContractError(f"{task_id}: scoring rule requires a positive version")
            legacy = data.get("legacy")
            if legacy:
                source = within(self.root, legacy["path"], required=check_assets)
                bench = within(self.root, legacy["bench_root"], required=check_assets)
                if not source.is_relative_to(bench) or source == bench:
                    raise ContractError(f"{task_id}: invalid legacy selector")
                if source.relative_to(bench).as_posix() != legacy["selector"]:
                    raise ContractError(f"{task_id}: legacy selector does not match its path")
                for filename in legacy["files"]:
                    asset = within(self.root, filename, required=check_assets)
                    if not asset.is_relative_to(source) or (check_assets and not asset.is_file()):
                        raise ContractError(f"{task_id}: invalid legacy file: {filename}")
                if execution["adapter"] == "cann_legacy":
                    if (objective, scope) != ("implement", "operator"):
                        raise ContractError(f"{task_id}: CANN legacy adapter requires implement/operator")
                    if assets.get("contract") != f'{legacy["path"]}/proto.yaml':
                        raise ContractError(f"{task_id}: legacy contract does not match selector")
            else:
                if objective is None or scope is None:
                    raise ContractError(f"{task_id}: native tasks need an objective and scope")
                required = {"problem", "contract", "workloads", "answer_example"}
                if objective == "optimize":
                    required.add("performance_baseline")
                if objective in {"implement", "optimize", "repair"}:
                    required.add("oracle")
                if objective in {"optimize", "diagnose", "repair"}:
                    required.add("starter")
                if not required.issubset(assets):
                    raise ContractError(f"{task_id}: missing roles {sorted(required - set(assets))}")
                if data["status"] == "published":
                    minimum = runtime["minimum"]
                    if not runtime["validated_platforms"] or not isinstance(minimum, dict):
                        raise ContractError(f"{task_id}: published tasks need validated resources")
                    fields = ("cards", "device_memory_bytes", "host_memory_bytes")
                    if any(type(minimum.get(key)) is not int or minimum[key] < 1 for key in fields):
                        raise ContractError(f"{task_id}: incomplete minimum resources")
                    if not isinstance(minimum.get("interconnect"), str) or not minimum["interconnect"].strip():
                        raise ContractError(f"{task_id}: specify minimum interconnect")
            for path_value in assets.values():
                within(self.root, path_value, required=check_assets)
            for profile in runtime["profiles"]:
                settings = read_document(within(self.root, profile))
                for path_value in settings.get("platform_metadata", {}).values():
                    within(self.root, path_value, required=check_assets)
        except (KeyError, TypeError) as exc:
            raise ContractError(f"{task_id}: malformed task contract: {exc}") from exc
        return TaskSpec(path, data)

    def suite(self, suite_id, *, check_assets=True):
        suite = read_document(within(self.root, f"suites/{identifier(suite_id)}.yaml"))
        try:
            if suite["id"] != suite_id or not suite["version"]:
                raise ContractError(f"{suite_id}: mismatched suite identity")
            if suite["status"] == "published" and not suite["source_revision"]:
                raise ContractError(f"{suite_id}: published suites require a fixed source revision")
            seen = set()
            for member in suite["members"]:
                task = self.task(member["task"], check_assets=check_assets)
                if task.id in seen or task.version != member["version"]:
                    raise ContractError(f"{suite_id}: duplicate or version mismatch for {task.id}")
                if suite["status"] == "published" and task.data["status"] != "published":
                    raise ContractError(f"{suite_id}: published suite references non-published task {task.id}")
                seen.add(task.id)
            for path in suite["assets"].values():
                within(self.root, path, required=check_assets)
        except (KeyError, TypeError) as exc:
            raise ContractError(f"{suite_id}: malformed suite contract: {exc}") from exc
        return suite

    def task_ids(self):
        return sorted(path.parent.name for path in (self.root / "tasks").glob("*/task.yaml"))

    def suite_ids(self):
        return sorted(path.stem for path in (self.root / "suites").glob("*.yaml"))
