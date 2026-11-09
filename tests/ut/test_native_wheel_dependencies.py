"""Native wheels must preserve the environment's ABI-matched torch pair."""

from email.parser import Parser
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("project", ["examples/aclnn_launch_example", "src/cann_bench_utils"])
def test_native_wheel_metadata_does_not_replace_torch(tmp_path, project):
    root = Path(__file__).resolve().parents[2]
    # egg_info generates the dependency metadata used by the wheel without
    # rebuilding the native operators or importing torch_npu.
    result = subprocess.run(
        [sys.executable, "setup.py", "egg_info", "--egg-base", str(tmp_path)],
        cwd=root / project,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metadata = next(tmp_path.glob("*.egg-info/PKG-INFO"))
    requirements = Parser().parsestr(metadata.read_text()).get_all("Requires-Dist", [])
    assert not any(req.lower().startswith(("torch", "torch_npu", "torch-npu"))
                   for req in requirements), requirements
