"""Use the catalog while keeping the legacy CLI available."""

import argparse
import json
from pathlib import Path

from .catalog import Catalog
from .contracts import ContractError, Submission, within
from .engine import Engine


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cannbench")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("suites")
    commands.add_parser("tasks").add_argument("--suite")
    commands.add_parser("show").add_argument("task")
    validator = commands.add_parser("validate")
    validator.add_argument("--suite")
    validator.add_argument("--schema", action="store_true", help="Also validate against JSON Schema (requires jsonschema)")
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("task")
        command.add_argument("--suite")
        command.add_argument("--source-dir", type=Path, required=True)
        command.add_argument("--reports-dir", type=Path, required=True)
        command.add_argument("--device-id", type=int)
        command.add_argument("--case-id", type=int)
        command.add_argument("--no-perf", action="store_true")
    args = parser.parse_args(argv)
    catalog = Catalog(args.root)
    try:
        status = 0
        if args.command == "suites":
            result = catalog.suite_ids()
        elif args.command == "tasks":
            result = (
                [row["task"] for row in catalog.suite(args.suite, check_assets=False)["members"]]
                if args.suite else catalog.task_ids()
            )
        elif args.command == "show":
            result = catalog.task(args.task, check_assets=False).data
        elif args.command == "validate":
            suites = [catalog.suite(key) for key in ([args.suite] if args.suite else catalog.suite_ids())]
            ids = [row["task"] for row in suites[0]["members"]] if args.suite else catalog.task_ids()
            tasks = [catalog.task(key).data for key in ids]
            if args.schema:
                try:
                    from jsonschema import Draft202012Validator
                except ImportError as exc:
                    raise ContractError("Install jsonschema to use --schema") from exc
                for name, instances in (("task", tasks), ("suite", suites)):
                    with within(catalog.root, f"schemas/{name}.schema.json").open(encoding="utf-8") as stream:
                        schema = json.load(stream)
                    validator = Draft202012Validator(schema)
                    for item in instances:
                        error = next(validator.iter_errors(item), None)
                        if error:
                            raise ContractError(f"{item['id']}: {error.json_path}: {error.message}")
            result = {"tasks": len(ids), "suites": len(suites)}
        else:
            engine = Engine(catalog)
            submission = Submission("source", args.source_dir)
            params = dict(suite_id=args.suite, device_id=args.device_id, no_perf=args.no_perf, case_id=args.case_id)
            if args.command == "plan":
                _, _, command = engine.plan(args.task, submission, args.reports_dir, **params)
                result = {"argv": command}
            else:
                run_result = engine.run(args.task, submission, args.reports_dir, **params)
                result = run_result.as_dict()
                status = run_result.exit_code
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return status
    except (ContractError, OSError) as exc:
        parser.exit(2, f"cannbench: {exc}\n")
