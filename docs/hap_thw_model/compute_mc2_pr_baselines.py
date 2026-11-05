#!/usr/bin/env python3

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Check the published MC2 950PR measured baselines without writing files.

The historical filename is retained for existing check commands. This script
checks the measurement declaration, case coverage and data fingerprints only.
It cannot reproduce or independently authenticate a hardware measurement and
never fits, generates, rounds or overwrites baseline_perf_us or t_hw_us.
"""

import argparse
import json
import math

try:
    from docs.hap_thw_model import compute_mc2_a5 as cases_model
except ModuleNotFoundError:
    import compute_mc2_a5 as cases_model


QUANT_DTYPES = {"int8", "float8_e4m3fn", "float8_e5m2"}
PROXY_FIELDS = frozenset(
    {
        "model",
        "generator",
        "source_file",
        "not_measured_on_target_hardware",
        "source_baseline_sha256",
        "source_provenance_sha256",
        "resource_model_sha256",
        "hardware_constants",
        "proxy_model_sha256",
        "resource_definition",
        "fit",
        "formula",
        "rounding",
        "fits",
    }
)


def verify(bench=cases_model.BENCH):
    """Validate the recorded PR measurements; do not read DT data or run a model."""
    path = bench / "metadata/950pr.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["_metadata"]
    source = meta["mc2_baseline_source"]
    if (
        meta["hardware"] != "950pr"
        or source["kind"] != "measured"
        or source["target_hardware"] != "950pr"
        or source["source_hardware"] != "950pr"
    ):
        raise ValueError("MC2 950PR baselines must declare measurements on 950PR")
    if PROXY_FIELDS.intersection(source):
        raise ValueError("MC2 950PR measured provenance must not contain proxy fields")
    evidence = source["provenance_evidence"]
    if evidence["kind"] != "measurement_declaration" or not evidence["statement"]:
        raise ValueError(
            "MC2 950PR measurement declaration requires a measurement declaration"
        )

    selected = sorted(
        (
            (op, case)
            for op, case in cases_model.a5_cases(bench)
            if "950pr" in cases_model.supported_platforms(case)
        ),
        key=lambda item: (item[0], item[1]["case_id"]),
    )
    if not selected:
        raise ValueError("No supported MC2 950PR cases found")
    case_ids, values = {}, []
    for op, case in selected:
        case_id = case["case_id"]
        subset = "quantized" if case["dtype"][0] in QUANT_DTYPES else "nonquantized"
        ids = case_ids.setdefault(f"{op}/{subset}", [])
        if case_id in ids:
            raise ValueError(f"Duplicate MC2 950PR case: {op}/{case_id}")
        ids.append(case_id)
        try:
            value = data["level4"][op][str(case_id)]["baseline_perf_us"]
        except KeyError as exc:
            raise ValueError(f"Missing MC2 950PR measurement: {op}/{case_id}") from exc
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"MC2 950PR measurement must be finite and positive: {op}/{case_id}"
            )
        values.append((op, case_id, value))

    expected = {
        "case_count": len(selected),
        "case_ids": case_ids,
        "case_sha256": cases_model.fingerprint(selected),
        "baseline_sha256": cases_model.fingerprint(values),
    }
    for field, value in expected.items():
        if source.get(field) != value:
            raise ValueError(
                f"Stale MC2 950PR {field}: {path}; check the measurement records, not a resource model"
            )
    print(
        f"950pr: {len(selected)} measured baselines; "
        "case coverage and fingerprints verified"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    verify()
