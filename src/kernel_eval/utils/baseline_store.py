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
集中式 baseline 性能数据存储

从评测集根目录下的 metadata/<hardware>.json 加载 baseline_perf_us 和 t_hw_us，
支持多硬件 dict 格式和按平台扩展。

文件位置（每个评测集根目录下都有 metadata/ 子目录）：
    tasks/metadata/910b2.json           — CANN 主评测集 (910b2 平台)
    tasks/metadata/910b1.json           — CANN 主评测集 (910b1 平台，未来扩展)
    bench_lab/cv_agent_bench/metadata/910b2.json — CV Agent Bench
    bench_lab/stanford_bench/metadata/910b2.json — StanfordBench
    ...

查找策略：从 bench_root 开始，逐级向上查找 metadata/<hardware>.json，
直到找到或到达 project_root。这确保无论 bench_root 是评测集根目录
还是子目录（如 tasks/level1/exp），都能正确定位 baseline 数据。

多平台扩展：
    新增平台只需在 metadata/ 下加一个 JSON 文件（如 910b1.json），
    BaselineStore 通过 hardware 参数自动选择对应文件。
    稀疏覆盖友好：新平台文件只需包含已测量的用例，
    未覆盖的用例自动 fallback 到默认平台（910b2）的数据。

JSON 格式（level → op_dir → case_id 三级嵌套）:
{
  "_metadata": { ... },
  "level1": {
    "exp": {
      "1": { "baseline_perf_us": 13.78, "t_hw_us": 1.09 },
      "2": { "baseline_perf_us": 29.06, "t_hw_us": 8.74 }
    }
  }
}

多硬件支持:
  值可以是 float（单硬件，隐式默认）或 {hardware: float} dict（多硬件），
  与 baseline_resolver.py 现有行为一致。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .baseline_resolver import DEFAULT_HARDWARE, resolve_hardware

logger = logging.getLogger(__name__)

# 默认硬件的 baseline 文件名（如 "910b2.json"）
DEFAULT_PLATFORM_FILE = f"{DEFAULT_HARDWARE}.json"


class BaselineStore:
    """集中式 baseline 性能数据存储。

    从评测集根目录下的 metadata/<hardware>.json 加载 baseline_perf_us 和 t_hw_us，
    支持多硬件按平台扩展和向上查找。

    使用方式:
        # 默认平台 (910b2)
        store = BaselineStore(bench_root=Path("/path/to/tasks"))
        store.load()

        # 查询 baseline
        perf = store.get_perf("level1/exp", 1)          # → 13.78
        t_hw = store.get_t_hw("level1/exp", 1)          # → 1.09
        has  = store.has_baseline("level1/exp", 1)      # → True

        # 其他平台 (910b1) — 自动查找 metadata/910b1.json
        store_b1 = BaselineStore(bench_root=Path("/path/to/tasks"), hardware="910b1")
        store_b1.load()  # 加载 metadata/910b1.json, 未覆盖的用例 fallback 到 910b2
    """

    def __init__(
        self,
        bench_root: Path,
        project_root: Path = None,
        hardware: str = DEFAULT_HARDWARE,
    ):
        self.bench_root = Path(bench_root)
        self.project_root = Path(project_root) if project_root else None
        # 将硬件名称（含产品型号别名）解析为 baseline 逻辑名
        self.hardware = resolve_hardware(hardware)
        self._data_hardware = self.hardware
        self._data: Dict[str, Any] = {}
        self._fallback_data: Dict[str, Any] = {}  # 默认平台 fallback
        self._baseline_dir: Optional[Path] = None  # 找到的 metadata/ 目录路径
        self._loaded = False

    def _find_metadata_dir(self) -> Optional[Path]:
        """从 bench_root 开始逐级向上查找 metadata/ 目录。

        查找逻辑：
        1. 先检查 bench_root 本身是否包含 metadata/ 子目录
        2. 逐级向上查找，直到找到或到达 project_root / 文件系统根
        3. 这确保子目录（如 tasks/level1/exp）也能找到 tasks/metadata/

        Returns:
            metadata/ 目录的完整路径，若未找到返回 None
        """
        current = self.bench_root.resolve()

        boundary = self.project_root.resolve() if self.project_root else None

        while True:
            candidate = current / "metadata"
            if candidate.is_dir():
                return candidate

            if boundary and current == boundary:
                break
            parent = current.parent
            if parent == current:
                break
            current = parent

        return None

    def load(self) -> None:
        """Load baseline data from metadata/<hardware>.json.

        查找策略：从 bench_root 向上查找 metadata/ 目录，然后加载对应硬件的 JSON。
        若非默认硬件且该平台文件不存在，fallback 加载默认平台的 JSON。
        """
        metadata_dir = self._find_metadata_dir()
        if metadata_dir is None:
            logger.debug(
                "BaselineStore: no metadata/ dir found from %s upward", self.bench_root
            )
            self._loaded = True
            return

        self._baseline_dir = metadata_dir

        # Versioned suites may forbid cross-platform fallback. In particular,
        # unpublished 950 metadata and missing A3 measurements must not silently
        # inherit 910B2 timings. Legacy suites keep their original behavior.
        manifest_path = metadata_dir.parent / "suite_manifest.json"
        if manifest_path.is_file():
            manifest = self._load_json(manifest_path)
            if manifest.get("metadata_fallback_policy") == "same_platform_only":
                platform_file = metadata_dir / f"{self.hardware}.json"
                self._data = (
                    self._load_json(platform_file) if platform_file.is_file() else {}
                )
                self._fallback_data = {}
                if not self._data:
                    logger.warning(
                        "BaselineStore: suite=%r has no published metadata for %r; "
                        "cross-platform fallback is disabled.",
                        manifest.get("suite"),
                        self.hardware,
                    )
                self._loaded = True
                return

        # 加载目标平台的 JSON
        platform_file = metadata_dir / f"{self.hardware}.json"
        if platform_file.is_file():
            self._data = self._load_json(platform_file)
            self._data_hardware = self.hardware
            logger.debug(
                "BaselineStore loaded %s (%d top-level keys)",
                platform_file,
                len(self._data),
            )
        elif self.hardware == DEFAULT_HARDWARE:
            logger.warning(
                "BaselineStore: 默认平台 %r 的文件不存在 (%s)，baseline 数据将为空。",
                DEFAULT_HARDWARE,
                platform_file,
            )

        # 非默认平台时，加载默认平台作为 fallback
        if self.hardware != DEFAULT_HARDWARE:
            default_file = metadata_dir / DEFAULT_PLATFORM_FILE
            if default_file.is_file():
                self._fallback_data = self._load_json(default_file)
                logger.debug("BaselineStore loaded fallback %s", default_file)
            # 如果目标平台文件也不存在，fallback 数据作为主数据
            if not self._data and self._fallback_data:
                self._data = self._fallback_data
                self._data_hardware = DEFAULT_HARDWARE
                self._fallback_data = {}

        # 默认平台文件缺失时，目标平台也没有 → 数据为空
        if not self._data:
            # 尝试加载默认平台文件作为唯一数据源
            default_file = metadata_dir / DEFAULT_PLATFORM_FILE
            if default_file.is_file():
                self._data = self._load_json(default_file)
                self._data_hardware = DEFAULT_HARDWARE

        self._loaded = True

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """加载 JSON 文件"""
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("BaselineStore: failed to load %s: %s", path, e)
            return {}

    def _resolve_entry(
        self, rel_path: str, case_id: int, data: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Resolve the JSON entry for a given (rel_path, case_id) from specified data.

        Path resolution:
        - "level1/exp", 1 → data["level1"]["exp"]["1"]
        - "flash_attention", 1 → data["flash_attention"]["1"]
        """
        if data is None:
            data = self._data
        if not data:
            return None

        current = data
        parts = rel_path.split("/")

        for part in parts:
            if part in current and isinstance(current[part], dict):
                current = current[part]
            else:
                return None

        case_key = str(case_id)
        if isinstance(current, dict) and case_key in current:
            entry = current[case_key]
            if isinstance(entry, dict):
                return entry

        return None

    def _resolve_perf_value(self, value: Any, hardware: str) -> float:
        """Resolve a performance value, handling multi-hardware dict format."""
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            v = value.get(hardware)
            if v is None:
                return 0.0
            return float(v)
        return 0.0

    def get_perf(self, rel_path: str, case_id: int, hardware: str = None) -> float:
        """Get baseline_perf_us for a specific case.

        查找顺序：目标平台文件 → 默认平台 fallback → 0.0

        Args:
            rel_path: relative path like "level1/exp" or "flash_attention"
            case_id: case number (integer)
            hardware: target hardware name (None 则使用 store 初始化时的 hardware)

        Returns:
            float: baseline_perf_us, 0.0 if not found
        """
        return self._get_value_and_source(
            rel_path, case_id, "baseline_perf_us", hardware
        )[0]

    def get_t_hw(self, rel_path: str, case_id: int, hardware: str = None) -> float:
        """Get t_hw_us for a specific case.

        查找顺序同 get_perf。

        Args:
            rel_path: relative path like "level1/exp" or "flash_attention"
            case_id: case number (integer)
            hardware: target hardware name (None 则使用 store 初始化时的 hardware)

        Returns:
            float: t_hw_us, 0.0 if not found
        """
        return self._get_value_and_source(rel_path, case_id, "t_hw_us", hardware)[0]

    def get_sources(
        self, rel_path: str, case_id: int, hardware: str = None
    ) -> Dict[str, Any]:
        """Return per-field origins without changing legacy value/fallback selection.

        A metadata origin identifies a file/platform, not proof of measurement.
        Measurement declarations remain in that file's provenance metadata.
        """
        sources = {}
        for field in ("baseline_perf_us", "t_hw_us"):
            _, source = self._get_value_and_source(rel_path, case_id, field, hardware)
            if source:
                sources[field] = source
        return sources

    def _get_value_and_source(self, rel_path, case_id, field, hardware):
        hw = hardware or self.hardware
        for data, file_hardware, lookup_hardware, is_fallback in (
            (self._data, self._data_hardware, hw, False),
            (self._fallback_data, DEFAULT_HARDWARE, DEFAULT_HARDWARE, True),
        ):
            entry = self._resolve_entry(rel_path, case_id, data)
            if entry is None or field not in entry:
                continue
            raw = entry[field]
            value = self._resolve_perf_value(raw, lookup_hardware)
            # Preserve the existing fallback semantics, including zero/missing.
            if not value > 0 and not is_fallback:
                continue
            if not value > 0:
                return value, {}
            source_hardware = (
                lookup_hardware if isinstance(raw, dict) else file_hardware
            )
            return value, {
                "requested_hardware": hw,
                "source_hardware": source_hardware,
                "fallback_used": source_hardware != hw,
                "source": f"metadata/{file_hardware}.json",
            }
        return 0.0, {}

    def has_baseline(self, rel_path: str, case_id: int, hardware: str = None) -> bool:
        """Check whether baseline exists for the given case+hardware.

        Args:
            rel_path: relative path
            case_id: case number
            hardware: target hardware name

        Returns:
            True if baseline data exists for this case+hardware
        """
        hw = hardware or self.hardware

        # 查目标平台
        entry = self._resolve_entry(rel_path, case_id, self._data)
        if entry is not None:
            bp = entry.get("baseline_perf_us")
            if bp is not None:
                if isinstance(bp, dict):
                    return hw in bp and bp[hw] is not None
                return True

        # fallback 到默认平台
        if self._fallback_data:
            entry = self._resolve_entry(rel_path, case_id, self._fallback_data)
            if entry is not None:
                bp = entry.get("baseline_perf_us")
                if bp is not None:
                    if isinstance(bp, dict):
                        return (
                            DEFAULT_HARDWARE in bp and bp[DEFAULT_HARDWARE] is not None
                        )
                    return True

        return False

    # === StanfordBench 特殊查询 ===

    def get_stanford_perf(self, level: str, py_stem: str) -> float:
        """Get baseline_perf_us for StanfordBench using (level, py_stem) lookup.

        StanfordBench uses file stem names (e.g., "23_Softmax_1") as keys,
        which is different from the CANN (rel_path, case_id) pattern.
        """
        # 先查目标平台
        level_data = self._data.get(level)
        if level_data and isinstance(level_data, dict):
            perf = level_data.get(py_stem)
            if perf is not None:
                return float(perf)

        # fallback 到默认平台
        if self._fallback_data:
            level_data = self._fallback_data.get(level)
            if level_data and isinstance(level_data, dict):
                perf = level_data.get(py_stem)
                if perf is not None:
                    return float(perf)

        return 0.0
