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
2. 检查/编译whl包和run包（编译失败即整批计编译失败，不隔离不补救）
3. 安装whl包和run包
4. 扫描cann_bench模块提供的算子接口
5. 匹配tasks 中的算子定义
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
import importlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

from ..config import get_config


# 编译失败时从日志识别失败算子（用于错误摘要）：
# bisheng/g++ 直接报错的行形如 `.../csrc/ops/<op>/op_kernel/foo.cpp:LINE:COL: error:`
# make 的 Error N 行形如 `.../csrc/ops/<op>/op_kernel/foo.cpp.o] Error 1`
#
# 要求 `error` 出现在编译器诊断的规范位置（紧跟文件路径的 `:line:col:` 之后），
# 而不是同一行任意位置出现 "error" 即算。否则像
#   csrc/ops/good_op/.../good.cpp:10:5: note: previous error from csrc/ops/bad_op/...
# 这种 note/交叉引用行会把 good_op 误判为失败算子（注：仅影响错误摘要归属精度，
# 不影响评分——build_packages 已把本次提交所有算子整批标记为编译失败）。
# 仅放宽 error 的位置不够（非贪婪 [^\n]*? 仍会命中 note 行）；故按诊断结构精确匹配。
_OP_ERROR_LINE_RE = re.compile(
    r"csrc/ops/([A-Za-z0-9_]+)/op_[a-z]+/[^\s:]+\.cpp"
    r"(?:"
    r"(?::\d+)*:\s*(?:fatal\s+)?error\b"  # 编译器诊断：file.cpp[:line[:col]]: (fatal )?error
    r"|"
    r"\.o[^\]]*\]\s+Error\s+\d+"  # make：file.cpp.o] Error N
    r")"
)

# submission 构建超时。run_evaluation.sh 会显式透传，直接调用 Python
# 入口时仍保留 600 秒默认值。
_DEFAULT_BUILD_TIMEOUT_SEC = 600


def resolve_build_timeout_sec(value=None) -> int:
    """Resolve positive seconds with explicit value > environment > default."""
    source = "--build-timeout" if value is not None else "CANN_BENCH_BUILD_TIMEOUT_SEC"
    raw = (
        value
        if value is not None
        else os.environ.get(
            "CANN_BENCH_BUILD_TIMEOUT_SEC", str(_DEFAULT_BUILD_TIMEOUT_SEC)
        )
    )
    try:
        if isinstance(raw, bool) or not isinstance(raw, (str, int)):
            raise ValueError("not an integer")
        timeout = int(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{source} must be an integer") from exc
    if timeout <= 0:
        raise ValueError(f"{source} must be positive")
    return timeout


# CANN OPP 安装路径探测。
# install.sh 中 --install-path 接收 OPP 根目录（如 /home/.../cann-9.0.0/opp）。
# 常规 submission 使用 custom_ops；大型算子仓可生成其他 vendor，例如
# ops-transformer 的 custom_transformer，因此安装后必须探测本次实际写入的目录。
_CANN_OPP_DEFAULT_VENDOR_SUBDIR = "vendors/custom_ops"


def _detect_opp_path() -> str:
    """动态探测 CANN OPP 根目录路径（传给 install.sh --install-path）。

    MC2_OPP_INSTALL_ROOT 仅改变本次生成 .run 包的安装目标，不改变
    ASCEND_OPP_PATH（编译期仍须使用系统 CANN 的头文件和基础算子）。
    未设置时保留原有的系统 OPP 探测行为。
    """
    private_root = os.environ.get("MC2_OPP_INSTALL_ROOT", "")
    if private_root:
        private_path = Path(private_root).expanduser()
        if not private_path.is_absolute():
            raise ValueError("MC2_OPP_INSTALL_ROOT must be an absolute path")
        private_path.mkdir(parents=True, exist_ok=True)
        return str(private_path.resolve())

    home = os.environ.get("ASCEND_HOME_PATH", "")
    for base in (
        os.environ.get("ASCEND_OPP_PATH", ""),
        os.path.join(home, "opp") if home else "",
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


def _get_arch_str() -> str:
    """返回当前平台架构字符串，用于拼装 tiling 库路径。"""
    import platform

    machine = platform.machine().lower()
    return "aarch64" if machine in ("aarch64", "arm64") else "x86_64"


def _snapshot_vendor_env_scripts(opp_path: str) -> Dict[Path, int]:
    """记录已安装 vendor 的环境脚本及修改时间。"""
    vendors_root = Path(opp_path) / "vendors"
    if not vendors_root.is_dir():
        return {}
    snapshot: Dict[Path, int] = {}
    for set_env_path in vendors_root.glob("*/bin/set_env.bash"):
        try:
            snapshot[set_env_path] = set_env_path.stat().st_mtime_ns
        except OSError:
            continue
    return snapshot


def _detect_installed_vendor_path(
    opp_path: str,
    before: Dict[Path, int],
) -> Path:
    """定位本次 run 安装实际创建或更新的 custom vendor 目录。"""
    vendors_root = Path(opp_path) / "vendors"
    after = _snapshot_vendor_env_scripts(opp_path)
    changed = [path for path, mtime in after.items() if before.get(path) != mtime]
    if changed:
        # 正常安装会重写 set_env.bash；修改时间最新的脚本对应本次 run 包。
        return max(changed, key=lambda path: (after[path], str(path))).parent.parent

    # 某些文件系统的时间戳粒度较粗，或安装器未重写已有脚本。此时优先采用
    # vendors/config.ini 的 load_priority，再回退到默认 vendor 和现有环境脚本。
    priority_paths: List[Path] = []
    config_path = vendors_root / "config.ini"
    if config_path.is_file():
        try:
            match = re.search(
                r"^\s*load_priority\s*=\s*([^#\r\n]+)",
                config_path.read_text(),
                flags=re.MULTILINE,
            )
            if match:
                priority_paths = [
                    vendors_root / name.strip()
                    for name in match.group(1).split(",")
                    if name.strip()
                ]
        except OSError:
            pass

    default_path = Path(opp_path) / _CANN_OPP_DEFAULT_VENDOR_SUBDIR
    candidates = priority_paths + [default_path]
    candidates.extend(path.parent.parent for path in after)
    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if (candidate / "bin" / "set_env.bash").is_file():
            return candidate
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return default_path


def _add_vendor_lib_dirs(vendor_path: Path) -> None:
    """将 vendor 目录下的 .so 搜索路径加入 LD_LIBRARY_PATH。

    除了 op_api/lib（set_env.bash 已包含的路径），还需加入 tiling 目录，
    因为 CANN framework 在部分场景下不自动加载 tiling 库时，
    LD_LIBRARY_PATH 中的 tiling 路径可以作为 dlopen 搜索的兜底。
    """
    arch = _get_arch_str()
    lib_dirs = [
        vendor_path / "op_api" / "lib",
        vendor_path / "op_impl" / "ai_core" / "tbe" / "op_tiling",
        vendor_path
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "op_tiling"
        / "lib"
        / "linux"
        / arch,
    ]
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    for lib_dir in lib_dirs:
        if lib_dir.is_dir():
            lib_str = str(lib_dir)
            if lib_str not in cur.split(":"):
                cur = f"{lib_str}:{cur}" if cur else lib_str
                print(f"[INFO] 已将 {lib_str} 加入 LD_LIBRARY_PATH")
    os.environ["LD_LIBRARY_PATH"] = cur


@dataclass
class PackageInfo:
    """包信息"""

    source_dir: str
    whl_path: Optional[str] = None
    run_path: Optional[str] = None
    has_build_sh: bool = False
    has_dist: bool = False
    # 编译失败的算子及其错误摘要 {op_name: error_excerpt}。
    # build.sh 编译失败时由 build_packages 填充，evaluator 会把它们
    # 作为"编译失败"记录（0 分）合入最终报告。
    compile_errors: Dict[str, str] = field(default_factory=dict)
    # build.sh 整体编译失败标志。为 True 时不安装、不评测，本次提交相关算子
    # 全部按编译失败计 0 分（不再隔离/补救失败算子，详见 build_packages）。
    build_failed: bool = False


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

    def check_dist_packages(
        self, source_dir: str
    ) -> Tuple[Optional[str], Optional[str]]:
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
        """执行build.sh编译生成包（整体编译，编译失败即整批计 0 分）。

        **不做隔离/补救**：`build.sh` 跑一次，成功则返回带 whl 的 PackageInfo；
        失败则不修改用户源码、不挪走/重编任何算子，而是汇总编译错误并把本次提交
        提供的所有算子（`csrc/ops/<op>/`）标记为编译失败（`build_failed=True`，
        `compile_errors` 覆盖全部算子）。上层据此让相关算子全部按编译失败计 0 分。

        这样做的原因是**评测并行化的需要**：原先的迭代隔离会修改共享源码树（移动
        `csrc/ops/<op>/` 目录）并串行多轮重编，无法在去中心化/并行评测下安全执行；
        且 bench 不应修改用户提交的源码。详见 README / docs/spec 的编译评分说明。

        Args:
            source_dir: 提交源码根目录（含 build.sh）
            iterative: 已废弃（保留以兼容调用方），无论取值均为整体编译。

        Returns:
            PackageInfo — 编译成功：whl_path 非空、compile_errors 为空、
            build_failed=False；编译失败：whl_path 为空、build_failed=True、
            compile_errors 覆盖本次提交的所有算子（{snake_op: 错误摘要}）。

        Raises:
            FileNotFoundError: build.sh 不存在。
        """
        source_path = Path(source_dir)
        build_sh = source_path / "build.sh"

        if not build_sh.exists():
            raise FileNotFoundError(f"build.sh不存在: {build_sh}")

        self._clean_build_artifacts(source_path)
        log_path = source_path / "_compile.log"
        build_args = list(getattr(self.config, "source_build_args", []) or [])
        build_cmd = shlex.join(["bash", "build.sh", *build_args])
        print(f"[INFO] 编译: {build_cmd} → {log_path.name}")
        rc, log_text = self._run_build(source_path, log_path)

        if rc == 0 and self._wheel_exists(source_path):
            print("[INFO] 编译成功")
            package_info = self.scan_source_dir(source_dir)
            package_info.compile_errors = {}
            package_info.build_failed = False
            return package_info

        # 编译失败：不隔离、不补救、不修改源码 —— 汇总错误，本次提交所有算子计编译失败
        print(
            f"[ERROR] 编译失败（rc={rc}），本次提交相关算子按编译失败计 0 分；详见 {log_path}"
        )
        per_op = self._parse_failing_ops(log_text)  # 能定位到的算子 → 各自错误片段
        tail = "\n".join(log_text.splitlines()[-40:]).strip() or "(无编译日志输出)"
        submission_ops = self._discover_submission_ops(source_path)

        compile_errors: Dict[str, str] = {}
        for op in submission_ops:
            compile_errors[op] = (
                per_op.get(op) or f"build.sh 编译失败，编译日志末尾摘要：\n{tail}"
            )
        # 异常布局下未发现 csrc/ops/* 时，至少保留日志能定位到的算子；再不行用整体摘要
        if not compile_errors:
            compile_errors = per_op or {"<build>": f"build.sh 编译失败：\n{tail}"}

        package_info = self.scan_source_dir(source_dir)  # 无 whl → whl_path=None
        package_info.compile_errors = compile_errors
        package_info.build_failed = True
        return package_info

    def _discover_submission_ops(self, source_path: Path) -> List[str]:
        """枚举提交里提供的算子（csrc/ops/<op>/ 目录名，snake_case）。"""
        ops_dir = source_path / "csrc" / "ops"
        if not ops_dir.is_dir():
            return []
        return sorted(
            p.name
            for p in ops_dir.iterdir()
            if p.is_dir() and not p.name.startswith((".", "_"))
        )

    def _run_build(self, source_path: Path, log_path: Path) -> Tuple[int, str]:
        """跑 bash build.sh，把 stdout+stderr 同步写日志文件。
        返回 (returncode, 日志文本)。超时视作编译失败。"""
        try:
            build_timeout_sec = resolve_build_timeout_sec(
                getattr(self.config, "build_timeout_sec", None)
            )
            build_args = list(getattr(self.config, "source_build_args", []) or [])
            print(f"[INFO] submission 编译超时: {build_timeout_sec}s")
            with log_path.open("w") as logf:
                proc = subprocess.run(
                    ["bash", "build.sh", *build_args],
                    cwd=str(source_path),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=build_timeout_sec,
                )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = 124  # 约定的 timeout 退出码
            with log_path.open("a") as logf:
                logf.write(f"\n[ERROR] build.sh timed out after {build_timeout_sec}s\n")
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
        vendor_env_before = _snapshot_vendor_env_scripts(opp_path)

        try:
            # 设置可执行权限
            os.chmod(run_path, 0o755)

            # 执行安装：
            #   --quiet        静默安装
            #   --install-path OPP 根目录，触发生成 set_env.bash（含 LD_LIBRARY_PATH 等环境变量）
            abs_run_path = str(run_file.resolve())
            cmd = [abs_run_path, "--quiet", f"--install-path={opp_path}"]
            result = subprocess.run(
                cmd,
                cwd=str(run_file.parent),
                capture_output=True,
                text=True,
                timeout=120,  # 2分钟超时
            )

            if result.returncode != 0:
                # 首次安装失败时，探测 makeself 安装器是否支持 --force
                # （跳过共享库验证）。当 libcust_opapi.so 的 ELF NEEDED 链
                # 缺少 libopapi.so 时，验证器单独加载 libcust_opapi.so 无法
                # 解析 l0op::Contiguous 等内置符号，--force 可兜底绕过。
                first_output = result.stderr or result.stdout
                if self._run_supports_force(abs_run_path):
                    print(
                        f"[WARN] run包首次安装失败，尝试 --force 重试: {first_output[:200]}"
                    )
                    result = subprocess.run(
                        [
                            abs_run_path,
                            "--quiet",
                            "--force",
                            f"--install-path={opp_path}",
                        ],
                        cwd=str(run_file.parent),
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )

            if result.returncode != 0:
                output = result.stderr or result.stdout
                raise RuntimeError(f"run包安装失败 (rc={result.returncode}): {output}")

            print("[INFO] run包安装成功")

            # source set_env.bash 将 LD_LIBRARY_PATH / ASCEND_CUSTOM_OPP_PATH
            # 设到当前进程，确保 dlopen 能找到 libcust_opapi.so
            vendor_path = _detect_installed_vendor_path(opp_path, vendor_env_before)
            print(f"[INFO] 本次安装的 OPP vendor 目录: {vendor_path}")
            self._source_set_env_bash(vendor_path)

            # 校验：ASCEND_CUSTOM_OPP_PATH 必须包含当前 vendor 路径。
            # CANN framework 依赖此变量定位 custom vendor 目录中的 tiling 库；
            # 若缺失，dlopen libcust_opapi.so (BIND_NOW) 会因 tiling 符号未解析而失败。
            vendor_str = str(vendor_path)
            cur_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
            if vendor_str not in cur_opp:
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
                    f"{vendor_str}:{cur_opp}" if cur_opp else vendor_str
                )
                print(f"[INFO] 兜底设置 ASCEND_CUSTOM_OPP_PATH={vendor_str}")

            return True

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"run包安装超时 (120s): {run_file.name}")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"run包安装异常: {e}")

    @staticmethod
    def _run_supports_force(run_path: str) -> bool:
        """探测 makeself 安装器是否支持 --force（跳过共享库验证）。

        不同 CANN 版本打包的 makeself 安装器参数可能不同，通过 --help
        输出探测而非硬编码，避免对不支持的版本传无效参数。
        """
        try:
            r = subprocess.run(
                [run_path, "--help"], capture_output=True, text=True, timeout=10
            )
            return "--force" in (r.stdout + r.stderr)
        except Exception:
            return False

    @staticmethod
    def _source_set_env_bash(vendor_path: Path) -> None:
        """Source set_env.bash 并将导出变量应用到当前进程。

        install.sh 在传入 --install-path 时会在实际 vendor 的 bin 目录中
        写入 set_env.bash；vendor 名称不要求固定为 custom_ops。
        此方法解析该文件并将变量设到 os.environ，使后续的 dlopen 能正确找到自定义算子库。
        """
        set_env_path = vendor_path / "bin" / "set_env.bash"

        if not set_env_path.exists():
            print(f"[WARN] set_env.bash 不存在: {set_env_path}")
            # 兜底：设置 ASCEND_CUSTOM_OPP_PATH + 补全 LD_LIBRARY_PATH
            # ASCEND_CUSTOM_OPP_PATH 是关键：CANN framework 用它定位 custom vendor 目录，
            # 内部构造 tiling 库绝对路径并 dlopen(RTLD_GLOBAL)，使 BIND_NOW 符号解析成功。
            # 若未设置，dlopen libcust_opapi.so 会因 tiling 符号缺失而失败。
            vendor_str = str(vendor_path)
            cur_opp = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
            if vendor_str not in cur_opp:
                os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
                    f"{vendor_str}:{cur_opp}" if cur_opp else vendor_str
                )
                print(f"[INFO] 已设置 ASCEND_CUSTOM_OPP_PATH={vendor_str}（兜底）")
            # 补全 LD_LIBRARY_PATH：op_api/lib + tiling 目录
            _add_vendor_lib_dirs(vendor_path)
            return

        content = set_env_path.read_text()
        for match in re.finditer(r"export\s+(\w+)=(.+)", content):
            var_name = match.group(1)
            var_value = match.group(2).strip()
            # 展开 ${VAR_NAME} 引用为当前环境变量值
            var_value = re.sub(
                r"\$\{(\w+)\}",
                lambda m: os.environ.get(m.group(1), ""),
                var_value,
            )
            os.environ[var_name] = var_value
            print(f"[INFO] source: {var_name}={var_value}")

        # set_env.bash 仅含 op_api/lib，不含 tiling 目录 —— 补全遗漏的路径
        _add_vendor_lib_dirs(vendor_path)

    def install_whl_package(self, whl_path: str) -> bool:
        """安装whl包（Python包）

        安装策略：强制重装当前提交，但不安装依赖。提交 wheel 通常都使用相同
        的 ``cann_bench==1.0.0`` 版本；普通 ``pip install`` 会把新提交误判为
        已安装。``--no-deps`` 保证 force-reinstall 不会重装 torch/torch_npu。
        """
        whl_file = Path(whl_path)
        if not whl_file.exists():
            raise FileNotFoundError(f"whl包不存在: {whl_path}")

        print(f"[INFO] 安装whl包: {whl_file.name}")

        try:
            print(f"[INFO] 安装: pip install --force-reinstall --no-deps {whl_path}")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    "--no-deps",
                    whl_path,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.stdout:
                for line in result.stdout.strip().splitlines():
                    print(f"[pip] {line}")
            if result.stderr:
                for line in result.stderr.strip().splitlines():
                    print(f"[pip-warn] {line}")

            if result.returncode != 0:
                raise RuntimeError(f"pip install 失败: {result.stderr}")

            # 验证安装位置（诊断步骤，失败不应影响已成功的安装）
            try:
                verify = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import importlib.util; spec = importlib.util.find_spec('cann_bench'); "
                        "print(spec.origin if spec else 'NOT FOUND')",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                verify_out = verify.stdout.strip()
                if verify_out == "NOT FOUND":
                    print(
                        "[WARN] pip 返回成功但 cann_bench 模块不可被 find_spec 发现；"
                        "后续 scan_interfaces 可能失败"
                    )
                elif verify_out:
                    print(f"[INFO] cann_bench 安装位置: {verify_out}")
            except subprocess.TimeoutExpired:
                print("[WARN] cann_bench 安装位置验证超时，已跳过（不影响安装结果）")

            print("[INFO] whl包安装成功")
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
        cann_bench_import_error = None
        try:
            # 重新导入模块（确保使用最新安装的版本）
            for module_name in list(sys.modules):
                if module_name == "cann_bench" or module_name.startswith("cann_bench."):
                    del sys.modules[module_name]
            importlib.invalidate_caches()

            import cann_bench

            cann_bench_imported = True

            # 扫描模块属性
            for name in dir(cann_bench):
                if name.startswith("_"):
                    continue

                attr = getattr(cann_bench, name)
                if callable(attr) and not isinstance(attr, type):
                    # 获取函数签名
                    try:
                        from inspect import signature

                        sig = str(signature(attr))
                    except Exception:
                        sig = ""

                    interfaces.append(
                        InterfaceInfo(
                            name=name, callable=attr, signature=f"{name}{sig}"
                        )
                    )
        except ImportError as e:
            cann_bench_import_error = e
            import traceback

            print(f"[WARN] cann_bench 导入失败: {e}")
            traceback.print_exc()

        # 2. 尝试从 torch.ops.cann_bench 扫描（由 cann_bench 模块注册）
        try:
            import torch

            if hasattr(torch.ops, "cann_bench"):
                # torch.library 注册的 ops 需要从已扫描的接口或模块推断名称
                # 如果 cann_bench 已导入，从其模块属性推断
                for interface in interfaces:
                    op_name = interface.name.lower()
                    if hasattr(torch.ops.cann_bench, op_name):
                        op = getattr(torch.ops.cann_bench, op_name)
                        if callable(op) and op not in [i.callable for i in interfaces]:
                            interfaces.append(
                                InterfaceInfo(
                                    name=op_name,
                                    callable=op,
                                    signature=f"torch.ops.cann_bench.{op_name}",
                                )
                            )
        except ImportError:
            pass

        # 3. 如果两者都没找到，抛出错误
        if not interfaces:
            if not cann_bench_imported:
                error_detail = (
                    f": {cann_bench_import_error}" if cann_bench_import_error else ""
                )
                raise ImportError(
                    f"无法导入 cann_bench 模块{error_detail}，且 torch.ops.cann_bench 中无算子"
                )

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
            print("[INFO] 目录结构:")
            print(
                f"       - build.sh: {'存在' if package_info.has_build_sh else '不存在'}"
            )
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
                raise RuntimeError("源码目录无build.sh且无whl包，无法编译")

            print("[INFO] 无现有whl包，执行编译...")
            package_info = self.build_packages(source_dir, iterative=iterative_compile)

        # 编译整体失败：不安装、不扫描接口，直接返回（上层据 compile_errors 计 0 分）
        if getattr(package_info, "build_failed", False):
            print(
                f"[ERROR] 编译失败，跳过安装与评测；{len(package_info.compile_errors)} "
                f"个算子将按编译失败计 0 分"
            )
            return [], package_info

        # Step 3: 安装前 snapshot Timing API（安全防护）
        guard = APIGuard()
        guard.snapshot()
        print("[INFO] Timing API snapshot 完成")

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
            print("[WARN] 未匹配到tasks 中的算子")
            return [], package_info

        print(f"[INFO] 匹配到 {len(matched_operators)} 个tasks 算子:")
        for op_name in matched_operators:
            print(f"       - {op_name}")

        return matched_operators, package_info

    def prepare_skip_build(self) -> List[str]:
        """跳过编译安装，直接扫描已安装的cann_bench

        返回: 算子列表
        """
        print("[INFO] 跳过编译安装，扫描已安装的cann_bench")

        # 扫描接口
        interfaces = self.scan_interfaces()
        self.print_interfaces(interfaces)

        if not interfaces:
            raise RuntimeError("未扫描到任何cann_bench接口，请先安装cann_bench包")

        # 匹配算子
        matched_operators = self.match_operators(interfaces)

        if not matched_operators:
            print("[WARN] 未匹配到tasks 中的算子")
            return []

        print(f"[INFO] 匹配到 {len(matched_operators)} 个tasks 算子:")
        for op_name in matched_operators:
            print(f"       - {op_name}")

        return matched_operators
