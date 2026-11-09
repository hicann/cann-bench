#!/usr/bin/python3
# coding=utf-8

from pathlib import Path

from auto_pipeline.core import CannBenchClient


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_kernel_eval_cpu_no_perf_smoke_with_source_dir(tmp_path):
    # Keep the pipeline smoke independent of the changing public task suite.
    bench_root = tmp_path / "tasks"
    task_dir = bench_root / "level1" / "gelu"
    task_dir.mkdir(parents=True)
    (bench_root / "metadata").mkdir()
    (bench_root / "metadata" / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    task_dir.joinpath("proto.yaml").write_text(
        "operator:\n"
        "  name: Gelu\n"
        "  category: Elementwise\n"
        "  difficulty: L1\n"
        "  inputs: [{name: x, dtype: [float32]}]\n"
        "  outputs: [{name: y, dtype: [float32]}]\n"
        "  schema: gelu(Tensor x) -> Tensor y\n",
        encoding="utf-8",
    )
    task_dir.joinpath("cases.yaml").write_text(
        "cases:\n"
        "- operator: Gelu\n"
        "  case_id: 1\n"
        "  input_shape: [[16, 16]]\n"
        "  dtype: [float32]\n"
        "  attrs: {}\n"
        "  value_range: [-1, 1]\n",
        encoding="utf-8",
    )
    task_dir.joinpath("golden.py").write_text(
        "import torch\n\n"
        "def gelu(x: torch.Tensor) -> torch.Tensor:\n"
        "    return torch.nn.functional.gelu(x)\n",
        encoding="utf-8",
    )

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
        task_selector=str(task_dir),
        reports_dir=tmp_path / "reports",
        extra_args=[
            "--device",
            "cpu",
            "--no-perf",
            "--torch-op-guard-mode",
            "warn",  # 冒烟测试的 submission 直接调用 torch.nn.functional.gelu，block 模式会阻断执行
            "--case-id",
            "1",
            "--warmup",
            "0",
            "--repeat",
            "1",
        ],
    )

    assert result.ok, result.stdout + result.stderr
    assert "--bench-name" in result.command
    assert result.report_files
