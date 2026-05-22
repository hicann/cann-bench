#!/usr/bin/env python3
"""Post-build smoke for cann-bench:cann9.0.0-* execution image.

Verifies in order:
  [1] python / torch / torch_npu importable, log versions
  [2] torch_npu can see at least one NPU device
  [3] npu-smi info works (driver/runtime loaded)
  [4] CANN compiler version.info readable (proves CANN install intact)

Exits 0 with "ALL CHECKS PASSED" only if all four pass.
"""

import os
import subprocess
import sys

failed = []

# [1] python/torch/torch_npu versions
try:
    import torch
    import torch_npu

    py = ".".join(str(v) for v in sys.version_info[:3])
    print(f"[OK]   [1] python {py}, torch {torch.__version__}, torch_npu {torch_npu.__version__}")
except Exception as e:
    print(f"[FAIL] [1] import/version: {e}")
    failed.append(1)

# [2] torch_npu device visible
try:
    import torch_npu

    count = torch_npu.npu.device_count()
    assert count > 0, f"device_count = {count}"
    print(f"[OK]   [2] torch_npu.npu.device_count() = {count}")
except Exception as e:
    print(f"[FAIL] [2] torch_npu device_count: {e}")
    failed.append(2)

# [3] npu-smi info exits 0
try:
    subprocess.check_call(
        ["npu-smi", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("[OK]   [3] npu-smi reachable")
except Exception as e:
    print(f"[FAIL] [3] npu-smi info: {e}")
    failed.append(3)

# [4] CANN version readable. ascendhub base lays version.info per
# component, no aggregate at toolkit root — compiler/ is the canonical core.
try:
    vfile = os.path.join(os.environ["ASCEND_HOME_PATH"], "compiler", "version.info")
    with open(vfile) as f:
        line = f.read().strip().splitlines()[0]
    print(f"[OK]   [4] CANN compiler {line}")
except Exception as e:
    print(f"[FAIL] [4] CANN compiler version.info: {e}")
    failed.append(4)

if failed:
    sys.exit(f"\nFAILED: {failed}")
print("\nALL CHECKS PASSED")
