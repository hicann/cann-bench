#!/usr/bin/python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Build trusted tensor fixtures without loading a submission package.

python -m kernel_eval.cache_cli build --task level4/linear_cross_entropy --cache-dir /cache/lce
"""

import argparse
import sys

import torch

from .config import Config
from .data.data_generator import INPUT_DIST_CHOICES
from .data.tensor_cache import validate_config
from .eval.evaluator import Evaluator


def create_parser():
    parser = argparse.ArgumentParser(description="Build versioned input/reference safetensors fixtures")
    parser.add_argument("command", choices=("build",))
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--tasks-root", default="")
    parser.add_argument("--task", action="append", required=True,
                        help="Exact tasks-root-relative task path (repeatable)")
    parser.add_argument("--case-id", type=int, help="Optional case number within each selected task")
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--input-dist", choices=INPUT_DIST_CHOICES, default="uniform")
    parser.add_argument("--threads", type=int, help="CPU intra-op threads for trusted reference construction")
    return parser


def run(args):
    if args.eval_seed < 0:
        raise ValueError("Cache build requires a nonnegative deterministic eval seed")
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("--threads must be positive")
        torch.set_num_threads(args.threads)
    config = Config(tasks_root=args.tasks_root, device_type="cpu", enable_profiler=False,
                    eval_seed=args.eval_seed, input_dist=args.input_dist,
                    tensor_cache_dir=args.cache_dir, tensor_cache_mode="require",
                    tensor_cache_tasks=args.task)
    validate_config(config)
    evaluator = Evaluator(config)
    cases = evaluator.case_loader.scan_all()
    for task in dict.fromkeys(args.task):
        selected = [case for case in cases if case.rel_path == task
                    and (args.case_id is None or case.case_num == args.case_id)]
        if not selected:
            raise ValueError(f"No cases found for {task}, case-id={args.case_id}")
        for case in selected:
            location = evaluator.build_tensor_cache(case)
            print(f"[tensor-cache] {case.get_case_id_str()}: {location}", flush=True)
    return 0


def main():
    args = create_parser().parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"[ERROR] Tensor cache build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
