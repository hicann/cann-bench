#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Golden whl 打包脚本

将 tasks/ 下所有 golden.py 收集到一个纯 Python cann_bench whl 包中。
每个 golden 函数注册到 torch.ops.cann_bench 命名空间，
使 run_evaluation.sh 能将其作为"AI算子"加载评测，
验证 golden(NPU) 与 golden(CPU fp64) 的精度一致性。

用法:
    OP_NAME 请先设置为所选算子目录 proto.yaml 中的 operator.name。
    python scripts/utils/build_golden_wheel.py                   # 构建所有算子
    python scripts/utils/build_golden_wheel.py --operator "$OP_NAME" # 只打包指定算子
    python scripts/utils/build_golden_wheel.py --install         # 构建后自动安装
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import yaml
from pathlib import Path


def get_project_root() -> Path:
    """获取项目根目录"""
    # scripts/utils/build_golden_wheel.py -> 项目根
    return Path(__file__).resolve().parent.parent.parent


def _available_level_choices(tasks_root: Path) -> tuple[int, ...]:
    """Return the level directories available below one task root."""
    levels = []
    for path in tasks_root.glob("level*"):
        if not path.is_dir() or not path.name[5:].isdigit():
            continue
        levels.append(int(path.name[5:]))
    return tuple(sorted(levels))


def scan_golden_operators(
    tasks_root: Path, operator_filter: list = None, level_filter: int = None
):
    """扫描 tasks/ 目录，收集每个算子的信息

    Returns:
        list of dicts: [{rel_path, op_name, func_name, golden_path}]
    """
    operators = []
    for proto_path in sorted(tasks_root.rglob("proto.yaml")):
        rel_path = proto_path.parent.relative_to(tasks_root).as_posix()

        # 读取 proto.yaml
        with open(proto_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "operator" not in data:
            continue

        op_info = data["operator"]
        op_name = op_info.get("name", "")
        schema = op_info.get("schema", "")

        # 从 schema 提取函数名: "op_func(Tensor x) -> Tensor y" -> "op_func"
        func_name = op_name.lower()  # 默认
        if schema:
            match = re.match(r"^(\w+)\s*\(", schema.strip())
            if match:
                func_name = match.group(1)

        # Level 筛选
        if level_filter is not None:
            level_str = f"level{level_filter}"
            if rel_path.split("/", 1)[0] != level_str:
                continue

        # Operator 名称筛选
        if operator_filter:
            if op_name.lower() not in [o.lower() for o in operator_filter]:
                continue

        golden_path = proto_path.parent / "golden.py"
        if not golden_path.exists():
            continue

        operators.append(
            {
                "rel_path": rel_path,
                "op_name": op_name,
                "func_name": func_name,
                "golden_path": golden_path,
            }
        )

    return operators


def generate_init_py(operators: list) -> str:
    """生成 cann_bench/__init__.py

    将每个 golden 函数 import 并注册到 torch.ops.cann_bench 命名空间。
    """
    import_lines = []
    op_entries = []

    for op in operators:
        func_name = op["func_name"]
        # 子模块名用 func_name（snake_case）
        import_lines.append(f"from ._goldens.{func_name} import {func_name}")
        op_entries.append(f'    "{func_name}": {func_name},')

    # 用普通字符串模板避免 f-string 与 {} 冲突
    TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it
# under the terms and conditions of CANN Open Software License Agreement
# Version 2.0 (the "License"). You may not use this file except in compliance
# with the License. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT
# WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""CANN Bench - Golden reference operators (PyTorch implementations)

将 tasks/ 下的 golden.py 函数收集打包，注册到 torch.ops.cann_bench 命名空间。
用于验证 golden(NPU) 与 golden(CPU fp64) 的精度一致性。
"""
__version__ = "1.0.0"

import torch

IMPORTS_BLOCK

_OPS = {
OPS_DICT
}

# 注册到 torch.ops.cann_bench 命名空间（cann_matcher 通过 hasattr/getattr 查找）
for _name, _func in _OPS.items():
    setattr(torch.ops.cann_bench, _name, _func)
'''

    imports_block = "\n".join(import_lines)
    ops_dict_block = "\n".join(op_entries)

    content = TEMPLATE.replace("IMPORTS_BLOCK", imports_block).replace(
        "OPS_DICT", ops_dict_block
    )
    return content


def generate_goldens_init_py() -> str:
    """生成 cann_bench/_goldens/__init__.py（空包声明）"""
    return """#!/usr/bin/env python3
# Golden operator submodules
"""


def generate_setup_py() -> str:
    """生成 setup.py（纯 Python 包）"""
    return """#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="cann_bench",
    version="1.0.0",
    description="Golden reference operators for CANN benchmark verification",
    packages=find_packages(),
    package_data={"cann_bench._goldens": ["*.py"]},
    install_requires=["torch", "numpy"],
    python_requires=">=3.8",
    zip_safe=False,
)
"""


def build_wheel(
    build_dir: Path,
    output_dir: Path,
    operators: list,
    tasks_root: Path,
    verbose: bool = False,
):
    """构建纯 Python cann_bench whl 包

    Args:
        build_dir: 临时构建目录
        output_dir: whl 输出目录
        operators: 算子信息列表
        tasks_root: tasks 目录路径
        verbose: 详细输出
    """
    pkg_dir = build_dir / "cann_bench"
    goldens_dir = pkg_dir / "_goldens"

    # 清理并创建目录
    if build_dir.exists():
        shutil.rmtree(build_dir)
    pkg_dir.mkdir(parents=True)
    goldens_dir.mkdir()

    # 1. 拷贝每个 golden.py -> _goldens/<func_name>.py
    for op in operators:
        src = op["golden_path"]
        dst = goldens_dir / f"{op['func_name']}.py"
        shutil.copy2(str(src), str(dst))
        if verbose:
            print(
                f"  拷贝: {op['rel_path']}/golden.py -> _goldens/{op['func_name']}.py"
            )

    # 2. 生成 _goldens/__init__.py
    (goldens_dir / "__init__.py").write_text(generate_goldens_init_py())

    # 3. 生成 cann_bench/__init__.py
    init_content = generate_init_py(operators)
    (pkg_dir / "__init__.py").write_text(init_content)

    # 4. 生成 setup.py
    (build_dir / "setup.py").write_text(generate_setup_py())

    # 5. 构建 whl
    print("\n[INFO] 构建 whl 包...")
    dist_dir = output_dir / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(dist_dir),
    ]
    result = subprocess.run(cmd, cwd=str(build_dir), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] whl 构建失败:\n{result.stderr}")
        sys.exit(1)

    # 查找生成的 whl
    whl_files = list(dist_dir.glob("cann_bench*.whl"))
    if not whl_files:
        print("[ERROR] 构建后未找到 whl 文件")
        sys.exit(1)

    whl_path = whl_files[0]
    print(f"[OK] whl 包构建成功: {whl_path}")
    return whl_path


def install_wheel(whl_path: Path, verbose: bool = False):
    """安装 whl 包（先卸载旧版本）"""
    print("\n[INFO] 卸载旧版本 cann_bench...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "cann_bench", "-y"],
        capture_output=True,
        timeout=30,
    )

    print(f"[INFO] 安装: {whl_path}")
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps", str(whl_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"[ERROR] 安装失败:\n{result.stderr}")
        sys.exit(1)

    print("[OK] 安装成功")


def main():
    parser = argparse.ArgumentParser(description="Golden whl 打包脚本")
    parser.add_argument(
        "--task-dir",
        type=str,
        default=None,
        help="指定 tasks 目录（默认: 项目根目录下的 tasks）",
    )
    parser.add_argument(
        "--operator",
        type=str,
        nargs="*",
        default=None,
        help="只打包指定算子（proto.yaml 中的 operator.name），可指定多个",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=None,
        help="只打包指定级别（按 --task-dir 下的 level<N> 目录校验）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认: 项目根目录下 dist/golden_wheel）",
    )
    parser.add_argument("--install", action="store_true", help="构建后自动安装")
    parser.add_argument("--clean", action="store_true", help="清理构建临时目录")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    project_root = get_project_root()
    tasks_root = Path(args.task_dir) if args.task_dir else project_root / "tasks"

    if not tasks_root.exists():
        print(f"[ERROR] tasks 目录不存在: {tasks_root}")
        sys.exit(1)

    if args.level is not None:
        available_levels = _available_level_choices(tasks_root)
        if args.level not in available_levels:
            choices = ", ".join(str(level) for level in available_levels) or "无"
            parser.error(
                f"--level {args.level} 在 {tasks_root} 下没有对应目录；可选级别: {choices}"
            )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else project_root / "dist" / "golden_wheel"
    )
    build_dir = output_dir / "_build_tmp"

    # 扫描算子
    print(f"[INFO] 扫描 tasks 目录: {tasks_root}")
    operators = scan_golden_operators(
        tasks_root,
        operator_filter=args.operator,
        level_filter=args.level,
    )

    if not operators:
        print("[WARN] 未找到任何算子")
        sys.exit(0)

    print(f"[INFO] 找到 {len(operators)} 个算子:")
    for op in operators:
        print(f"  {op['op_name']} ({op['func_name']}) <- {op['rel_path']}")

    # 清理模式
    if args.clean:
        if build_dir.exists():
            shutil.rmtree(build_dir)
        print("[OK] 清理完成")
        return

    # 构建 whl
    whl_path = build_wheel(
        build_dir, output_dir, operators, tasks_root, verbose=args.verbose
    )

    # 安装
    if args.install:
        install_wheel(whl_path, verbose=args.verbose)

    # 验证
    print("\n[INFO] 验证注册...")
    verify_script = (
        "import torch; import cann_bench; "
        "ops = [n for n in dir(cann_bench) if not n.startswith('_')]; "
        "print(f'cann_bench 模块导出: {len(ops)} 个算子'); "
        "registered = [n for n in ops if hasattr(torch.ops.cann_bench, n)]; "
        "print(f'torch.ops.cann_bench 注册: {len(registered)} 个算子'); "
        "print(f'前10个: {registered[:10]}')"
    )
    verify_cmd = [sys.executable, "-c", verify_script]
    subprocess.run(verify_cmd, timeout=30)

    print(f"\n[OK] 完成! whl 包: {whl_path}")
    print(f"安装命令: pip install --no-deps {whl_path}")
    print(
        '评测命令: ./scripts/run_evaluation.sh --task-dir "tasks/level<N>/<op_dir>" --no-perf'
    )
    print("请将占位路径替换为本次已打包算子的实际目录；自定义题集请使用对应路径。")


if __name__ == "__main__":
    main()
