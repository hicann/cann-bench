# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import json
from pathlib import Path

import pytest

from ascend_arena_e2e.ascend_arena.runners.cases import load_test_cases, select_cases


ROOT = Path(__file__).resolve().parents[2]
COUNTS = {
    "adaptive_avg_pool3d": (6, 49),
    "expand_into_jagged_permute": (5, 62),
    "group_quant": (5, 40),
    "group_norm_silu_quant": (5, 26),
    "gather_elements_v2": (5, 22),
    "chamfer_distance_grad": (5, 20),
    "apply_rotary_pos_emb": (6, 23),
}


def test_registered_operators_match_available_tasks_and_baselines():
    registry = json.loads((ROOT / "src/ascend_arena_e2e/ops.json").read_text())
    manifest = json.loads((ROOT / "src/ascend_arena_e2e/dependency/baselines.json").read_text())
    tasks = {p.name for p in (ROOT / "bench_lab/ascend_arena_bench").iterdir() if p.is_dir()}
    assert set(registry) == set(manifest["operators"]) == tasks == set(COUNTS)
    assert all(not p.startswith("loss/logit/") for p in manifest["repositories"]["ops-nn"]["paths"])


@pytest.mark.parametrize("operator", COUNTS)
def test_numbered_cases_preserve_stage_selection(operator):
    directory = ROOT / "bench_lab/ascend_arena_bench" / operator / "cases"
    for stage, expected in zip(("smoke", "complete", "profile"), (*COUNTS[operator], 10)):
        rows = load_test_cases(directory / f"{stage}_cases.json")
        assert [r["case_id"] for r in rows] == [f"case_{i:02d}" for i in range(1, len(rows) + 1)]
        assert len(select_cases(rows, stage)) == expected
        if stage == "profile":
            assert len(rows) == 10


def test_rotary_profiles_cover_all_supported_precisions_and_layouts():
    rows = load_test_cases(ROOT / "bench_lab/ascend_arena_bench/apply_rotary_pos_emb/cases/profile_cases.json")
    assert {r["inputs"]["query"]["dtype"] for r in rows} == {"float16", "bfloat16", "float32"}
    assert {r["attrs"]["layout"]["value"] for r in rows} == {1, 4}
    assert len({json.dumps({k: r[k] for k in ("inputs", "attrs")}, sort_keys=True) for r in rows}) == 10


@pytest.mark.parametrize('operator', COUNTS)
def test_scaled_profiles_retain_original_correctness_coverage_and_input_budget(operator):
    from math import prod
    directory = ROOT / 'bench_lab/ascend_arena_bench' / operator / 'cases'
    profiles = load_test_cases(directory / 'profile_cases.json')
    correctness = select_cases(load_test_cases(directory / 'complete_cases.json'), 'complete')
    correctness += select_cases(load_test_cases(directory / 'smoke_cases.json'), 'smoke')
    signature = lambda c: json.dumps({k: c[k] for k in ('inputs', 'attrs')}, sort_keys=True)
    covered = {signature(c) for c in correctness}
    for row in profiles:
        scale = row.get('meta', {}).get('measurement-scale')
        if scale is None:
            continue
        assert signature({'inputs': scale['original_inputs'], 'attrs': scale['original_attrs']}) in covered
        if 'previous_inputs' in scale:
            assert signature({'inputs': scale['previous_inputs'], 'attrs': scale['previous_attrs']}) in covered
            previous = scale['previous_inputs']['x']
            current = row['inputs']['x']
            assert current['shape'][0] > previous['shape'][0]
            assert current['shape'][1:] == previous['shape'][1:]
            assert {k: v for k, v in current.items() if k != 'shape'} == {k: v for k, v in previous.items() if k != 'shape'}
            assert row['attrs'] == scale['previous_attrs']
        assert any(prod(v['shape']) > prod(scale['original_inputs'][k]['shape'])
                   for k, v in row['inputs'].items())
        sizes = {'float16': 2, 'bfloat16': 2, 'float32': 4, 'int32': 4, 'int64': 8}
        assert sum(prod(v['shape']) * sizes[v['dtype']] for v in row['inputs'].values()) <= 64 * 1024**2


def test_expand_profile_output_sizes_match_declared_contract():
    import yaml
    directory = ROOT / 'bench_lab/ascend_arena_bench/expand_into_jagged_permute/cases'
    proto = yaml.safe_load((directory / 'proto.yaml').read_text())
    allowed = next(a for a in proto['operator']['attrs'] if a['name'] == 'output_size')['enum']
    for row in load_test_cases(directory / 'profile_cases.json'):
        output_size = row['attrs']['output_size']['value']
        assert output_size in allowed
        assert f'output_size={output_size}' in proto['coverage_axes']['attrs_signature']
        assert output_size >= row['inputs']['permute']['shape'][0]
