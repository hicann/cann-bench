# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Shared public A5 resource model; no device or private constants required."""

import copy
import json
from pathlib import Path

import pytest

from docs.hap_thw_model import hap_lib as H
from docs.hap_thw_model import compute_mc2_a5 as mc2


def complete_profiles(include_dt=False):
    """Synthetic rates for validation only, never published hardware values."""
    names = ("910b2", "950pr", "910c") + (("950dt",) if include_dt else ())
    profiles = {}
    for name in names:
        profile = {
            "hbm": 1000,
            "l2": 2000,
            "l2cap": 100,
            "cube": {"float16": 1000},
            "vec_basic": {"float32": 1000},
            "cast_rate": 1000,
            "sfu_factor": 0.5,
            "div_factor": 0.25,
        }
        if name in H.PUBLIC_PLAT:
            profile.update(copy.deepcopy(H.PUBLIC_PLAT[name]))
        profiles[name] = profile
    return profiles


@pytest.mark.parametrize(
    "plat,cube,hbm", [("950pr", 432e12, 1.6e12), ("950dt", 486e12, 4e12)]
)
def test_public_components_work_without_private_constants(monkeypatch, plat, cube, hbm):
    monkeypatch.setattr(H, "PLAT", H._MissingConstants())
    assert H.t_cube_us(cube, "float16", plat) == pytest.approx(1e6)
    assert H.t_read_us(hbm, plat) == pytest.approx(1e6)
    assert H.t_network_us(2e12, plat) == pytest.approx(1e6)
    with pytest.raises(RuntimeError, match="complete L2/vector"):
        H.t_write_us(1, plat)
    with pytest.raises(RuntimeError, match="complete L2/vector"):
        H.t_vec_us(1, "float32", plat)


def test_mc2_uses_shared_component_functions(monkeypatch):
    monkeypatch.setattr(H, "t_cube_us", lambda *args: 12.5)
    monkeypatch.setattr(H, "t_read_us", lambda *args: 15.75)
    monkeypatch.setattr(H, "t_network_us", lambda *args: 13.0)
    op, case = next(mc2.a5_cases())
    result = mc2.compute(op, case, "950dt")
    assert result["t_hw_us"] == 15.75
    assert result["components_us"] == {
        "cube_us": 12.5,
        "read_us": 15.75,
        "network_us": 13.0,
    }


@pytest.mark.parametrize(
    "key,value", [("hbm", 4.1e12), ("link", 1e12), ("cube", {"float16": 1e12})]
)
def test_conflicting_private_profile_is_rejected(key, value):
    profiles = complete_profiles(True)
    profiles["950dt"][key] = value
    with pytest.raises(RuntimeError, match="Hardware constant conflict: 950dt"):
        H._merge_public_constants(profiles)


def test_partial_public_fields_are_filled_without_mutating_input():
    profiles = complete_profiles(True)
    del profiles["950dt"]["hbm"]
    profiles["950dt"]["cube"] = {"tf32": 1000}
    combined = H._merge_public_constants(profiles)
    assert "hbm" not in profiles["950dt"]
    assert combined["950dt"]["hbm"] == H.PUBLIC_PLAT["950dt"]["hbm"]
    assert combined["950dt"]["cube"]["tf32"] == 1000
    H._validate_platform_constants(combined)


def test_legacy_complete_profiles_remain_supported(monkeypatch):
    profiles = H._merge_public_constants(complete_profiles())
    H._validate_platform_constants(profiles)
    monkeypatch.setattr(H, "PLAT", profiles)
    assert set(H.compute_both(input_bytes=1, output_bytes=1)) == {
        "910b2",
        "950pr",
        "910c",
    }
    with pytest.raises(RuntimeError, match="Complete constants for 950dt unavailable"):
        H.compute_both(platforms=("950dt",))


def test_complete_dt_uses_shared_full_model_and_confidence(monkeypatch):
    profiles = H._merge_public_constants(complete_profiles(True))
    H._validate_platform_constants(profiles)
    monkeypatch.setattr(H, "PLAT", profiles)
    result = H.compute_both(output_bytes=10)
    assert set(result) == {"910b2", "950pr", "910c", "950dt"}
    assert result["950dt"]["bottleneck"] == "write"
    assert result["950dt"]["confidence"] == "Unconfirmed"
    assert H.confidence_950dt("cube", 10) == "Confirmed"
    assert H.confidence_950dt("write", 101) == "Confirmed"


def test_incomplete_dt_is_not_silently_accepted():
    profiles = complete_profiles(True)
    profiles["950dt"]["l2"] = None
    with pytest.raises(
        RuntimeError, match=r"950dt.l2 must be a finite positive number"
    ):
        H._validate_platform_constants(profiles)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
@pytest.mark.parametrize("key", ["l2", "cast_rate", "vec_basic"])
def test_invalid_full_model_rates_are_rejected(key, value):
    profiles = complete_profiles(True)
    profiles["950dt"][key] = {"float32": value} if key == "vec_basic" else value
    with pytest.raises(RuntimeError, match=f"950dt.{key}"):
        H._validate_platform_constants(profiles)


def test_zero_l2_capacity_models_all_writes_to_hbm(monkeypatch):
    profiles = complete_profiles(True)
    profiles["950dt"]["l2cap"] = 0
    H._validate_platform_constants(profiles)
    monkeypatch.setattr(H, "PLAT", profiles)
    assert H.t_write_us(4e6, "950dt") == pytest.approx(1.0)


def test_public_runtime_constants_do_not_read_the_editable_template(
    monkeypatch, tmp_path
):
    source = Path(H.HERE) / "hap_platform_constants.public.json"
    (tmp_path / source.name).write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setattr(H, "HERE", str(tmp_path))
    assert H._load_public_constants() == H.PUBLIC_PLAT
    (tmp_path / "hap_platform_constants.template.json").write_text(
        "{}", encoding="utf-8"
    )
    assert H._load_public_constants() == H.PUBLIC_PLAT


def test_public_runtime_rates_reject_invalid_values(monkeypatch, tmp_path):
    data = copy.deepcopy(H.PUBLIC_PLAT)
    data["950dt"]["hbm"] = -1
    (tmp_path / "hap_platform_constants.public.json").write_text(
        json.dumps(data), encoding="utf-8"
    )
    monkeypatch.setattr(H, "HERE", str(tmp_path))
    with pytest.raises(RuntimeError, match="Invalid public hardware constants"):
        H._load_public_constants()
