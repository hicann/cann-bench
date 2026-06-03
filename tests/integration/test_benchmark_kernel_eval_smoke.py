#!/usr/bin/python3
# coding=utf-8

from pathlib import Path

from auto_pipeline.core import CannBenchClient


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_kernel_eval_cpu_no_perf_smoke_with_source_dir(tmp_path):
    source_dir = tmp_path / "submission"
    package_dir = source_dir / "cann_bench"
    package_dir.mkdir(parents=True)
    package_dir.joinpath("__init__.py").write_text("from .gelu import gelu\n", encoding="utf-8")
    package_dir.joinpath("gelu.py").write_text(
        "import torch\n\n"
        "def gelu(x, approximate='none'):\n"
        "    return torch.nn.functional.gelu(x, approximate=approximate)\n",
        encoding="utf-8",
    )
    source_dir.joinpath("setup.py").write_text(
        "from setuptools import find_packages, setup\n\n"
        "setup(name='cann_bench', version='1.0.0', packages=find_packages())\n",
        encoding="utf-8",
    )
    build_sh = source_dir / "build.sh"
    build_sh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "python setup.py clean bdist_wheel\n",
        encoding="utf-8",
    )
    build_sh.chmod(0o755)

    client = CannBenchClient(
        REPO_ROOT,
        timeout_sec=90,
        env={"ALLOW_TIMING_TAMPERING": "1"},
    )

    result = client.eval_submission(
        bench_name="cann",
        source_dir=source_dir,
        task_selector="tasks/level1/gelu",
        reports_dir=tmp_path / "reports",
        extra_args=[
            "--device",
            "cpu",
            "--no-perf",
            "--case-id",
            "1",
            "--no-subprocess-isolation",
            "--warmup",
            "0",
            "--repeat",
            "1",
        ],
    )

    assert result.ok, result.stdout + result.stderr
    assert "--bench-name" in result.command
    assert result.report_files
