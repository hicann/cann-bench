#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
包管理模块

职责：
1. 扫描源码目录结构
2. 检查/编译whl包和run包（支持迭代隔离编译失败的算子）
3. 安装whl包和run包
4. 扫描cann_bench模块提供的算子接口
5. 匹配tasks 中的算子定义
"""

import os
import re
import shutil
import subprocess
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from ..config import get_config


# 在迭代编译中识别失败的算子：
# bisheng/g++ 直接报错的行形如 `.../csrc/ops/<op>/op_kernel/foo.cpp:LINE:COL: error:`
# make 的 Error 1 行形如 `.../csrc/ops/<op>/op_kernel/foo.cpp.o] Error 1`
_OP_ERROR_LINE_RE = re.compile(
    r"csrc/ops/([A-Za-z0-9_]+)/op_[a-z]+/[^\s:]+\.cpp"
    r"(?:\.o)?(?:[^\n]*\berror\b|[^\]]*\]\s+Error 1)"
)

# 最多迭代多少轮，避免一直循环
_MAX_COMPILE_ROUNDS = 6
# SIGTERM 宽限时间
_BUILD_TIMEOUT_SEC = 600

# CANN OPP 安装路径探测。
# install.sh 中 --install-path 接收 OPP 根目录（如 /home/.../cann-9.0.0/opp），
# 安装脚本会在其下创建 vendors/custom_ops/ 子目录。
_CANN_OPP_VENDOR_SUBDIR = "vendors/custom_ops"


def _detect_opp_path() -> str:
    """动态探测 CANN OPP 根目录路径（传给 install.sh --install-path）。

    优先级：
    1. ASCEND_OPP_PATH 环境变量（CANN 安装后 source set_env.sh 会设置）
    2. $ASCEND_HOME_PATH/opp（Toolkit 内嵌 OPP）
    3. /usr/local/Ascend/opp（OPP 独立安装默认路径）
    """
    for base in (
        os.environ.get("ASCEND_OPP_PATH", ""),
        os.path.join(os.environ.get("ASCEND_HOME_PATH", ""), "opp"),
        "/usr/local/Ascend/opp",
    ):
        if base and os.path.isdir(base):
            return base
    # 均未命中，返回优先级最高的路径（install.sh 会在安装时创建）
    opp = os.environ.get("ASCEND_OPP_PATH", "")
    if not opp:
        home = os.environ.get("ASCEND_HOME_PATH", "")
        opp = os.path.join(home, "opp") if home else "/usr/local/Ascend/opp"
    return opp


@dataclass
class PackageInfo:
    """包信息"""
    source_dir: str
    whl_path: Optional[str] = None
    run_path: Optional[str] = None
    has_build_sh: bool = False
    has_dist: bool = False
    # 编译失败的算子及其错误摘要 {op_name: error_excerpt}。
    # 在开启 iterative compile 时由 build_packages 填充，evaluator 会把它们
    # 作为"编译失败"记录合入最终报告。
    compile_errors: Dict[str, str] = field(default_factory=dict)


@dataclass
class InterfaceInfo:
    """接口信息"""
    name: str
    callable: Any
    signature: str = ""

    def __repr__(self):
        return f"InterfaceInfo(name={self.name}, signature={self.signature})"


class PackageManager:
    """包管理器"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self._interface_cache: Dict[str, InterfaceInfo] = {}

    def scan_source_dir(self, source_dir: str) -> PackageInfo:
        """扫描源码目录结构

        检查目录中是否存在：
        - build.sh 编译脚本
        - dist/ 目录（包含whl和run包）
        - cann_bench/ Python包目录
        """
        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"源码目录不存在: {source_dir}")

        package_info = PackageInfo(source_dir=str(source_path))

        # 检查 build.sh
        build_sh = source_path / "build.sh"
        package_info.has_build_sh = build_sh.exists()

        # 检查 dist 目录
        dist_dir = source_path / "dist"
        package_info.has_dist = dist_dir.exists()

        if package_info.has_dist:
            # 扫描whl包
            whl_files = list(dist_dir.glob("cann_bench*.whl"))
            if whl_files:
                package_info.whl_path = str(whl_files[0])

            # 扫描run包
            run_files = list(dist_dir.glob("cann_bench*.run"))
            if run_files:
                package_info.run_path = str(run_files[0])

        return package_info

    def check_dist_packages(self, source_dir: str) -> Tuple[Optional[str], Optional[str]]:
        """检查dist目录的whl包和run包

        返回: (whl_path, run_path)
        """
        dist_dir = Path(source_dir) / "dist"
        if not dist_dir.exists():
            return None, None

        # 查找whl包
        whl_path = None
        whl_files = list(dist_dir.glob("cann_bench*.whl"))
        if whl_files:
            whl_path = str(whl_files[0])

        # 查找run包
        run_path = None
        run_files = list(dist_dir.glob("cann_bench*.run"))
        if run_files:
            run_path = str(run_files[0])

        return whl_path, run_path

    def build_packages(self, source_dir: str, iterative: bool = True) -> PackageInfo:
        """执行build.sh编译生成包。

        默认使用**迭代隔离模式**：若 `build.sh` 失败，扫描编译输出识别出哪些
        `csrc/ops/<op>/` 下的 kernel/plugin 源码未编译通过，把它们挪到
        `<submission>/_quarantine/<op>/`，记录错误摘要，再重试。最多迭代
        `_MAX_COMPILE_ROUNDS` 轮。这样即便 50 个算子里有 30 个编译不过，
        剩下 20 个也能进入正常评测流程。

        Args:
            source_dir: 提交源码根目录（含 build.sh）
            iterative: 启用迭代编译（默认 True）。设 False 时回退到一次性编译，
                       失败即 raise —— 用于希望"整体要么全过要么全失败"的场景。

        Returns:
            PackageInfo — 若迭代成功，含 whl_path 和非空 compile_errors
            （映射为 {snake_case_op_name: 错误摘要}）。

        Raises:
            RuntimeError: 最后一轮仍然失败且没产出 whl；或者 iterative=False
                          下 build.sh 首次失败。
        """
        source_path = Path(source_dir)
        build_sh = source_path / "build.sh"

        if not build_sh.exists():
            raise FileNotFoundError(f"build.sh不存在: {build_sh}")

        if not iterative:
            return self._build_once(source_path)

        compile_errors: Dict[str, str] = {}
        log_dir = source_path / "_compile_logs"
        log_dir.mkdir(exist_ok=True)

        for round_n in range(1, _MAX_COMPILE_ROUNDS + 1):
            self._clean_build_artifacts(source_path)
            log_path = log_dir / f"compile_round_{round_n}.log"
            print(f"[INFO] 编译第 {round_n} 轮: bash build.sh → {log_path.name}")

            rc, log_text = self._run_build(source_path, log_path)
            if rc == 0 and self._wheel_exists(source_path):
                print(f"[INFO] 编译成功（第 {round_n} 轮）")
                package_info = self.scan_source_dir(source_dir)
                package_info.compile_errors = compile_errors
                return package_info

            # 解析出失败的算子
            errs = self._parse_failing_ops(log_text)
            if not errs:
                print(f"[ERROR] 编译失败但未识别到 csrc/ops/<op>/ 错误，放弃")
                print(f"[ERROR] 日志见 {log_path}")
                raise RuntimeError(
                    f"build.sh 失败（第 {round_n} 轮），无法定位失败算子；"
                    f"详见 {log_path}"
                )

            new_ops = {op: msg for op, msg in errs.items() if op not in compile_errors}
            if not new_ops:
                print(f"[ERROR] 同一批算子在第 {round_n} 轮仍然失败，放弃")
                raise RuntimeError(
                    f"build.sh 失败（第 {round_n} 轮），隔离后仍然是同一批算子在报错"
                )

            moved = self._quarantine_ops(source_path, set(new_ops.keys()))
            print(f"[INFO] 第 {round_n} 轮隔离 {len(moved)} 个算子: {sorted(moved)}")
            compile_errors.update(new_ops)

        # 超过 _MAX_COMPILE_ROUNDS 仍未成功
        raise RuntimeError(
            f"build.sh 经过 {_MAX_COMPILE_ROUNDS} 轮迭代仍未产出 whl 包，放弃"
        )

    def _build_once(self, source_path: Path) -> PackageInfo:
        """非迭代模式：老行为，失败即抛异常。"""
        print(f"[INFO] 执行编译: bash build.sh")
        log_path = source_path / "_compile.log"
        rc, log_text = self._run_build(source_path, log_path)
        if rc != 0:
            print(f"[ERROR] 编译失败（完整日志 {log_path}）:")
            print(log_text[-2000:])
            raise RuntimeError(f"build.sh 执行失败，rc={rc}")
        print(f"[INFO] 编译成功")
        package_info = self.scan_source_dir(str(source_path))
        if not package_info.whl_path:
            raise RuntimeError("编译后未找到whl包")
        return package_info

    def _run_build(self, source_path: Path, log_path: Path) -> Tuple[int, str]:
        """跑 bash build.sh，把 stdout+stderr 同步写日志文件。
        返回 (returncode, 日志文本)。超时视作编译失败。"""
        try:
            with log_path.open("w") as logf:
                proc = subprocess.run(
                    ["bash", "build.sh"],
                    cwd=str(source_path),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=_BUILD_TIMEOUT_SEC,
                )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = 124  # 约定的 timeout 退出码
        except Exception as e:
            log_path.write_text(f"[build_submission] 启动失败: {e}\n")
            rc = 1
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        return rc, log_text

    def _parse_failing_ops(self, log_text: str) -> Dict[str, str]:
        """从编译日志里识别出哪些 op 的源码导致了错误，并给每个 op 切一段错误摘要。"""
        # 第一遍：识别出失败的算子名
        ops = set()
        for m in _OP_ERROR_LINE_RE.finditer(log_text):
            ops.add(m.group(1))
        # 第二遍：给每个算子抓 1-3 段错误上下文
        errors: Dict[str, str] = {}
        for op in ops:
            pat = re.compile(
                r"(^[^\n]*csrc/ops/" + re.escape(op) + r"/[^\n]*\berror\b:[^\n]*\n"
                r"(?:[^\n]*\n){0,3})",
                re.MULTILINE,
            )
            hits = pat.findall(log_text)
            excerpt = "".join(hits[:3]).strip() or "(error detail not captured)"
            errors[op] = excerpt
        return errors

    def _quarantine_ops(self, source_path: Path, op_names: set) -> List[str]:
        """把 csrc/ops/<op>/ 目录挪到 _quarantine/<op>/ 下，阻止下一轮 CMake
        configure 再把它加回来。成功移动的算子列表作为返回值。"""
        quarantine = source_path / "_quarantine"
        quarantine.mkdir(exist_ok=True)
        moved = []
        for op in op_names:
            src = source_path / "csrc" / "ops" / op
            if not src.is_dir():
                continue
            dst = quarantine / op
            if dst.exists():
                shutil.rmtree(src, ignore_errors=True)
            else:
                shutil.move(str(src), str(dst))
            moved.append(op)
        return moved

    def _clean_build_artifacts(self, source_path: Path) -> None:
        for p in ("build", "dist"):
            shutil.rmtree(source_path / p, ignore_errors=True)
        for egg in source_path.glob("*.egg-info"):
            shutil.rmtree(egg, ignore_errors=True)

    def _wheel_exists(self, source_path: Path) -> bool:
        dist = source_path / "dist"
        return dist.is_dir() and any(dist.glob("*.whl"))

    def install_run_package(self, run_path: str) -> bool:
        """安装run包（NPU内核包）

        run包为 makeself 格式，使用 --quiet 静默安装，--install-path 指定
        CANN OPP 根目录以触发安装脚本生成 set_env.bash。
        安装失败时抛出 RuntimeError，不静默忽略。
        """
        run_file = Path(run_path)
        if not run_file.exists():
            raise FileNotFoundError(f"run包不存在: {run_path}")

        print(f"[INFO] 安装run包: {run_file.name}")

        # 动态探测 OPP 根目录，传给 --install-path
        opp_path = _detect_opp_path()
        print(f"[INFO] OPP 根目录: {opp_path}")

        try:
            # 设置可执行权限
            os.chmod(run_path, 0o755)

            # 执行安装：
            #   --quiet        静默安装
            #   --install-path OPP 根目录，触发生成 set_env.bash（含 LD_LIBRARY_PATH 等环境变量）
            abs_run_path = str(run_file.resolve())
            result = subprocess.run(
                [abs_run_path, "--quiet",
                 f"--install-path={opp_path}"],
                cwd=str(run_file.parent),
                capture_output=True,
                text=True,
                timeout=120  # 2分钟超时
            )

            if result.returncode != 0:
                output = result.stderr or result.stdout
                raise RuntimeError(
                    f"run包安装失败 (rc={result.returncode}): {output}"
                )

            print(f"[INFO] run包安装成功")

            # source set_env.bash 将 LD_LIBRARY_PATH / ASCEND_CUSTOM_OPP_PATH
            # 设到当前进程，确保 dlopen 能找到 libcust_opapi.so
            self._source_set_env_bash(opp_path)

            return True

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"run包安装超时 (120s): {run_file.name}")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"run包安装异常: {e}")

    @staticmethod
    def _source_set_env_bash(opp_path: str) -> None:
        """Source set_env.bash 并将导出变量应用到当前进程。

        install.sh 在传入 --install-path 时会在
        <opp_path>/vendors/custom_ops/bin/set_env.bash 中写入 export 语句。
        此方法解析该文件并将变量设到 os.environ，使后续的 dlopen 能正确找到自定义算子库。
        """
        vendor_path = Path(opp_path) / _CANN_OPP_VENDOR_SUBDIR
        set_env_path = vendor_path / "bin" / "set_env.bash"

        if not set_env_path.exists():
            print(f"[WARN] set_env.bash 不存在: {set_env_path}")
            # 兜底：直接将 op_api/lib 加入 LD_LIBRARY_PATH
            lib_dir = vendor_path / "op_api" / "lib"
            if lib_dir.is_dir():
                cur = os.environ.get("LD_LIBRARY_PATH", "")
                lib_str = str(lib_dir)
                if lib_str not in cur:
                    os.environ["LD_LIBRARY_PATH"] = f"{lib_str}:{cur}" if cur else lib_str
                    print(f"[INFO] 已将 {lib_str} 加入 LD_LIBRARY_PATH（兜底）")
            return

        content = set_env_path.read_text()
        for match in re.finditer(r'export\s+(\w+)=(.+)', content):
            var_name = match.group(1)
            var_value = match.group(2).strip()
            # 展开 ${VAR_NAME} 引用为当前环境变量值
            var_value = re.sub(
                r'\$\{(\w+)\}',
                lambda m: os.environ.get(m.group(1), ''),
                var_value,
            )
            os.environ[var_name] = var_value
            print(f"[INFO] source: {var_name}={var_value}")

    def install_whl_package(self, whl_path: str) -> bool:
        """安装whl包（Python包）

        安装策略：先卸载旧版本，再安装新版本（不使用force-reinstall，避免重装依赖）
        """
        whl_file = Path(whl_path)
        if not whl_file.exists():
            raise FileNotFoundError(f"whl包不存在: {whl_path}")

        print(f"[INFO] 安装whl包: {whl_file.name}")

        try:
            # 先卸载旧版本（如果存在）
            print(f"[INFO] 卸载旧版本 cann_bench")
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "cann_bench", "-y"],
                capture_output=True,
                timeout=30
            )

            # 安装新版本（--no-deps 避免重装 torch/torch_npu 等依赖）
            print(f"[INFO] 安装: pip install --no-deps {whl_path}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-deps", whl_path],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                print(f"[ERROR] whl包安装失败: {result.stderr}")
                raise RuntimeError(f"pip install 失败: {result.stderr}")

            print(f"[INFO] whl包安装成功")
            return True

        except subprocess.TimeoutExpired:
            raise RuntimeError("whl包安装超时")
        except Exception as e:
            raise RuntimeError(f"whl包安装异常: {e}")

    def install_packages(self, package_info: PackageInfo) -> bool:
        """安装所有包（run包 + whl包）

        安装顺序：先安装run包，再安装whl包。
        run包安装失败时抛出 RuntimeError，不继续安装whl包。
        """
        # 安装run包（如果存在）—— 失败即 raise，不继续
        if package_info.run_path:
            self.install_run_package(package_info.run_path)

        # 安装whl包（必须成功）
        if package_info.whl_path:
            return self.install_whl_package(package_info.whl_path)

        return False

    def scan_interfaces(self) -> List[InterfaceInfo]:
        """扫描cann_bench模块接口

        扫描已安装的cann_bench模块，获取所有算子接口
        """
        interfaces = []

        # 1. 尝试从 cann_bench 模块扫描
        cann_bench_imported = False
        try:
            # 重新导入模块（确保使用最新安装的版本）
            if 'cann_bench' in sys.modules:
                del sys.modules['cann_bench']

            import cann_bench
            cann_bench_imported = True

            # 扫描模块属性
            for name in dir(cann_bench):
                if name.startswith('_'):
                    continue

                attr = getattr(cann_bench, name)
                if callable(attr) and not isinstance(attr, type):
                    # 获取函数签名
                    try:
                        from inspect import signature
                        sig = str(signature(attr))
                    except Exception:
                        sig = ""

                    interfaces.append(InterfaceInfo(
                        name=name,
                        callable=attr,
                        signature=f"{name}{sig}"
                    ))
        except ImportError:
            pass

        # 2. 尝试从 torch.ops.cann_bench 扫描（由 cann_bench 模块注册）
        try:
            import torch
            if hasattr(torch.ops, 'cann_bench'):
                # torch.library 注册的 ops 需要从已扫描的接口或模块推断名称
                # 如果 cann_bench 已导入，从其模块属性推断
                for interface in interfaces:
                    op_name = interface.name.lower()
                    if hasattr(torch.ops.cann_bench, op_name):
                        op = getattr(torch.ops.cann_bench, op_name)
                        if callable(op) and op not in [i.callable for i in interfaces]:
                            interfaces.append(InterfaceInfo(
                                name=op_name,
                                callable=op,
                                signature=f"torch.ops.cann_bench.{op_name}"
                            ))
        except ImportError:
            pass

        # 3. 如果两者都没找到，抛出错误
        if not interfaces:
            if not cann_bench_imported:
                raise ImportError("无法导入 cann_bench 模块，且 torch.ops.cann_bench 中无算子")

        return interfaces

    def print_interfaces(self, interfaces: List[InterfaceInfo]) -> None:
        """打印接口信息

        格式化显示扫描到的接口列表
        """
        print("")
        print("=" * 60)
        print("扫描到的 cann_bench 接口:")
        print("=" * 60)

        if not interfaces:
            print("  未找到任何接口")
            print("=" * 60)
            return

        for i, iface in enumerate(interfaces, 1):
            print(f"  {i}. {iface.signature}")

        print("")
        print(f"共 {len(interfaces)} 个算子接口")
        print("=" * 60)
        print("")

    def match_operators(self, interfaces: List[InterfaceInfo]) -> List[str]:
        """匹配tasks 中的算子

        根据接口名称，匹配tasks 中的算子定义
        返回匹配到的算子名称列表
        """
        from ..registry.loader_registry import get_task_loader

        operator_loader = get_task_loader(tasks_root=self.config.tasks_root)
        all_operators = operator_loader.list_operators()  # 获取所有算子

        matched = []
        interface_names = [iface.name.lower() for iface in interfaces]

        for op_info in all_operators:
            # 尝试多种匹配方式
            op_name_lower = op_info.name.lower()
            op_func_name = op_info.get_function_name().lower()

            if op_name_lower in interface_names or op_func_name in interface_names:
                matched.append(op_info.name)

        return matched

    def prepare_from_source(
        self,
        source_dir: str,
        verbose: bool = False,
        iterative_compile: bool = True,
    ) -> Tuple[List[str], PackageInfo]:
        """从源码目录准备评测环境

        流程：
        1. 扫描源码目录
        2. 检查/编译包（iterative_compile=True 时把编译失败的算子隔离并记录，
           False 时任意算子编译失败即 raise）
        3. **安装前 snapshot Timing API**（安全防护）
        4. 安装包
        5. 扫描接口
        6. 返回算子列表

        返回: (算子列表, 包信息 — 其 compile_errors 字段在 iterative 模式下
              可能非空，供评测器合成 FAIL 记录)

        注意：APIGuard.verify() 由 Evaluator.evaluate_from_source() 在安装后调用。
        """
        from ..security.api_guard import APIGuard

        print(f"[INFO] 扫描源码目录: {source_dir}")

        # Step 1: 扫描源码目录
        package_info = self.scan_source_dir(source_dir)

        if verbose:
            print(f"[INFO] 目录结构:")
            print(f"       - build.sh: {'存在' if package_info.has_build_sh else '不存在'}")
            print(f"       - dist目录: {'存在' if package_info.has_dist else '不存在'}")
            if package_info.whl_path:
                print(f"       - whl包: {Path(package_info.whl_path).name}")
            if package_info.run_path:
                print(f"       - run包: {Path(package_info.run_path).name}")

        # Step 2: 检查/编译包
        if package_info.whl_path:
            print(f"[INFO] 使用现有whl包: {Path(package_info.whl_path).name}")
        else:
            # 需要编译
            if not package_info.has_build_sh:
                raise RuntimeError(f"源码目录无build.sh且无whl包，无法编译")

            print(f"[INFO] 无现有whl包，执行编译...")
            package_info = self.build_packages(source_dir, iterative=iterative_compile)

        # Step 3: 安装前 snapshot Timing API（安全防护）
        guard = APIGuard()
        guard.snapshot()
        print(f"[INFO] Timing API snapshot 完成")

        # Step 4: 安装包
        if not self.install_packages(package_info):
            raise RuntimeError("包安装失败")

        # Step 5: 扫描接口
        interfaces = self.scan_interfaces()
        self.print_interfaces(interfaces)

        if not interfaces:
            raise RuntimeError("未扫描到任何cann_bench接口")

        # Step 6: 匹配算子
        matched_operators = self.match_operators(interfaces)

        if not matched_operators:
            print(f"[WARN] 未匹配到tasks 中的算子")
            return [], package_info

        print(f"[INFO] 匹配到 {len(matched_operators)} 个tasks 算子:")
        for op_name in matched_operators:
            print(f"       - {op_name}")

        return matched_operators, package_info

    def prepare_skip_build(self) -> List[str]:
        """跳过编译安装，直接扫描已安装的cann_bench

        返回: 算子列表
        """
        print(f"[INFO] 跳过编译安装，扫描已安装的cann_bench")

        # 扫描接口
        interfaces = self.scan_interfaces()
        self.print_interfaces(interfaces)

        if not interfaces:
            raise RuntimeError("未扫描到任何cann_bench接口，请先安装cann_bench包")

        # 匹配算子
        matched_operators = self.match_operators(interfaces)

        if not matched_operators:
            print(f"[WARN] 未匹配到tasks 中的算子")
            return []

        print(f"[INFO] 匹配到 {len(matched_operators)} 个tasks 算子:")
        for op_name in matched_operators:
            print(f"       - {op_name}")

        return matched_operators