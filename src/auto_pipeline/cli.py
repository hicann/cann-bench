"""CLI for cann-bench benchmark pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from auto_pipeline.core import run_from_config
from auto_pipeline.core import ParentProcessSignal


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m auto_pipeline.cli",
        description="Run the cann-bench benchmark generation pipeline.",
    )
    parser.add_argument("--config", required=True, help="pipeline YAML config")
    parser.add_argument("--workspace", required=True, help="runtime agent source workspace")
    parser.add_argument("--model", help="model for PyPTO generation and conversion")
    parser.add_argument("--output", help="runtime output root")
    parser.add_argument("--devices", help="device ids, e.g. 0,1 or 1-7")
    parser.add_argument("--parallel", type=int, help="maximum number of benchmark tasks to run in parallel")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    try:
        parser = create_parser()
        args = parser.parse_args(argv)
        runtime = {
            key: value
            for key, value in {
                "workspace": args.workspace,
                "model": args.model,
                "output": args.output,
                "devices": args.devices,
                "parallel": args.parallel,
            }.items()
            if value is not None
        }
        return run_from_config(Path(args.config), runtime=runtime)
    except ParentProcessSignal as exc:
        print(f"received {exc.signal_name}; child processes terminated", file=sys.stderr)
        return int(exc.code)
    except KeyboardInterrupt:
        print("interrupted; child processes terminated", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
