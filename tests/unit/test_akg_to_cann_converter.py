import importlib
import sys
import types
from pathlib import Path

import pytest

from auto_pipeline.prompt.registry import build_case_material
from auto_pipeline.core import CannBenchClient
from auto_pipeline.core import AGENT_SUCCESS, Artifact
from auto_pipeline.converter.registry import create_converter
from auto_pipeline.generator.akg.converter import load_operator_schema
from auto_pipeline.core import CannBenchCase


REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_submission(adapter, case: CannBenchCase, artifact: Artifact, *, output_dir: Path):
    return adapter.build_submission(case.bench_name, case, artifact, output_dir=output_dir)


def _stanford_relu_case(tmp_path: Path) -> CannBenchCase:
    task_path = tmp_path / "KernelBench" / "level1" / "19_ReLU.py"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "",
                "class Model(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "",
                "    def forward(self, x):",
                "        return torch.relu(x)",
                "",
                "",
                "def get_init_inputs():",
                "    return []",
                "",
                "",
                "def get_inputs():",
                "    return [torch.randn(4, 4)]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return CannBenchCase(
        bench_name="stanford",
        task_dir=task_path.parent,
        operator="ReLU",
        rel_path="level1/19_ReLU",
        files={"task": task_path},
        metadata={},
    )


def test_benchmark_material_preserves_exp_schema_for_generation():
    client = CannBenchClient(REPO_ROOT)
    case = client.load_case("cann", "tasks/level1/exp")

    material = build_case_material(case)

    assert material.bench_name == "cann"
    assert material.op_name == "exp"
    assert {task_file.key for task_file in material.task_files} >= {"proto", "cases", "golden"}
    assert case.metadata["schema"] in material.require_text


def test_load_operator_schema_preserves_schema_parameter_order(tmp_path):
    proto_path = tmp_path / "proto.yaml"
    proto_path.write_text(
        "\n".join(
            [
                "operator:",
                "  name: Scatter",
                "  schema: scatter(Tensor data, int dim, Tensor indices, Tensor updates, str? reduce=None) -> Tensor y",
                "  inputs:",
                "    - name: data",
                "    - name: indices",
                "    - name: updates",
                "  attrs:",
                "    - name: dim",
                "      default: 0",
                "    - name: reduce",
                "      default: null",
            ]
        ),
        encoding="utf-8",
    )
    case = CannBenchCase(
        bench_name="cann",
        task_dir=tmp_path,
        operator="Scatter",
        rel_path="tasks/level1/scatter",
        files={"proto": proto_path},
        metadata={},
    )

    schema = load_operator_schema(case)

    assert tuple(parameter.name for parameter in schema.parameters) == ("data", "dim", "indices", "updates", "reduce")
    assert schema.tensor_inputs == ("data", "indices", "updates")
    assert dict(schema.attrs) == {"dim": 0, "reduce": None}


def test_akg_to_cann_converter_packages_exp_modelnew_output(tmp_path):
    client = CannBenchClient(REPO_ROOT)
    case = client.load_case("cann", "tasks/level1/exp")
    adapter = create_converter("akg-agent", "cann", {})
    code = (
        "class ModelNew:\n"
        "    def to(self, device):\n"
        "        self.device = device\n"
        "        return self\n"
        "    def __call__(self, *, x, base=-1.0, scale=1.0, shift=0.0):\n"
        "        return x\n"
    )
    output = Artifact(status=AGENT_SUCCESS, workdir=tmp_path / "work", output_text=code)

    submission = _build_submission(adapter, case, output, output_dir=tmp_path / "submission")

    assert submission.kind == "cann"
    assert submission.operator == "Exp"
    assert submission.source_dir == (tmp_path / "submission").resolve()
    assert submission.source_dir.joinpath("build.sh").is_file()
    assert submission.source_dir.joinpath("setup.py").is_file()
    assert submission.source_dir.joinpath("cann_bench", "__init__.py").is_file()
    wrapper = submission.source_dir.joinpath("cann_bench", "exp.py").read_text(encoding="utf-8")
    impl = submission.source_dir.joinpath("cann_bench", "exp_triton_ascend_impl.py").read_text(encoding="utf-8")
    assert "def exp(x, base=-1.0, scale=1.0, shift=0.0):" in wrapper
    assert "return model(x=x, base=base, scale=scale, shift=shift)" in wrapper
    assert "from .exp_triton_ascend_impl import ModelNew" in wrapper
    assert "apply_triton_patches()" in impl
    assert "class ModelNew:" in impl
    assert submission.metadata["function_name"] == "exp"


def test_akg_to_stanford_converter_packages_relu_modelnew_output(tmp_path):
    case = _stanford_relu_case(tmp_path)
    adapter = create_converter("akg-agent", "stanford", {})
    code = (
        "import torch.nn as nn\n"
        "\n"
        "\n"
        "class ModelNew(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "\n"
        "    def forward(self, x):\n"
        "        return torch.relu(x)\n"
    )
    output = Artifact(status=AGENT_SUCCESS, workdir=tmp_path / "work", output_text=code)

    submission = _build_submission(adapter, case, output, output_dir=tmp_path / "submission")

    assert submission.kind == "stanford"
    assert submission.operator == "ReLU"
    assert submission.source_dir == (tmp_path / "submission").resolve()
    ai_op = submission.source_dir.joinpath("ai_op.py").read_text(encoding="utf-8")
    assert "_sys.path.insert(0, _op_dir)" in ai_op
    assert "apply_triton_patches()" in ai_op
    assert "class ModelNew(nn.Module):" in ai_op
    assert submission.metadata["converter"] == "akg-agent-to-stanford"
    assert submission.metadata["target_benchmark"] == "stanford"


def test_akg_to_cann_submission_imports_and_calls_packaged_exp(tmp_path, monkeypatch):
    client = CannBenchClient(REPO_ROOT)
    case = client.load_case("cann", "tasks/level1/exp")
    adapter = create_converter("akg-agent", "cann", {})
    code = (
        "class ModelNew:\n"
        "    def to(self, device):\n"
        "        return self\n"
        "    def __call__(self, *, x, base=-1.0, scale=1.0, shift=0.0):\n"
        "        return (x, base, scale, shift)\n"
    )
    output = Artifact(status=AGENT_SUCCESS, workdir=tmp_path / "work", output_text=code)

    submission = _build_submission(adapter, case, output, output_dir=tmp_path / "submission")

    triton_module = types.ModuleType("triton")
    triton_language_module = types.ModuleType("triton.language")
    triton_module.language = triton_language_module
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "triton", triton_module)
    monkeypatch.setitem(sys.modules, "triton.language", triton_language_module)
    monkeypatch.syspath_prepend(str(submission.source_dir))
    for module_name in list(sys.modules):
        if module_name == "cann_bench" or module_name.startswith("cann_bench."):
            sys.modules.pop(module_name, None)
    exp_module = importlib.import_module("cann_bench.exp")
    sentinel = object()

    assert exp_module.exp(sentinel, base=2.0, scale=3.0, shift=4.0) == (sentinel, 2.0, 3.0, 4.0)


def test_akg_to_cann_converter_requires_generated_code(tmp_path):
    client = CannBenchClient(REPO_ROOT)
    case = client.load_case("cann", "tasks/level1/exp")
    adapter = create_converter("akg-agent", "cann", {})
    output = Artifact(status=AGENT_SUCCESS, workdir=tmp_path / "work")

    with pytest.raises(ValueError, match="generated AKG Triton Ascend code is empty"):
        _build_submission(adapter, case, output, output_dir=tmp_path / "submission")


def test_wrapper_signature_uses_schema_parameter_order(tmp_path):
    proto_path = tmp_path / "proto.yaml"
    proto_path.write_text(
        "\n".join(
            [
                "operator:",
                "  name: Scatter",
                "  schema: scatter(Tensor data, int dim, Tensor indices, Tensor updates, str? reduce=None) -> Tensor y",
                "  inputs:",
                "    - name: data",
                "    - name: indices",
                "    - name: updates",
                "  attrs:",
                "    - name: dim",
                "      default: 0",
                "    - name: reduce",
                "      default: null",
            ]
        ),
        encoding="utf-8",
    )
    case = CannBenchCase(
        bench_name="cann",
        task_dir=tmp_path,
        operator="Scatter",
        rel_path="tasks/level1/scatter",
        files={"proto": proto_path},
        metadata={},
    )
    adapter = create_converter("akg-agent", "cann", {})
    output = Artifact(
        status=AGENT_SUCCESS,
        workdir=tmp_path / "work",
        output_text="class ModelNew:\n    pass\n",
    )

    _build_submission(adapter, case, output, output_dir=tmp_path / "submission")

    wrapper = (tmp_path / "submission" / "cann_bench" / "scatter.py").read_text(encoding="utf-8")
    assert "def scatter(data, dim=0, indices, updates, reduce=None):" not in wrapper
    assert "def scatter(data, dim=0, indices=None, updates=None, reduce=None):" not in wrapper
    assert "def scatter(data, dim=0, indices=indices" not in wrapper
    assert "def scatter(data, dim=0, indices" not in wrapper
    assert "def scatter(data, dim, indices, updates, reduce=None):" in wrapper
    assert "return model(data=data, dim=dim, indices=indices, updates=updates, reduce=reduce)" in wrapper


def test_wrapper_preserves_optional_tensor_schema_default(tmp_path):
    proto_path = tmp_path / "proto.yaml"
    proto_path.write_text(
        "\n".join(
            [
                "operator:",
                "  name: MoeGatingTopKSoftmax",
                "  schema: moe_gating_top_k_softmax(Tensor x, Tensor? finished=None, int k) -> Tensor y",
                "  inputs:",
                "    - name: x",
                "    - name: finished",
                "  attrs:",
                "    - name: k",
                "      default: 1",
            ]
        ),
        encoding="utf-8",
    )
    case = CannBenchCase(
        bench_name="cann",
        task_dir=tmp_path,
        operator="MoeGatingTopKSoftmax",
        rel_path="tasks/level3/moe_gating_top_k_softmax",
        files={"proto": proto_path},
        metadata={},
    )
    adapter = create_converter("akg-agent", "cann", {})
    output = Artifact(status=AGENT_SUCCESS, workdir=tmp_path / "work", output_text="class ModelNew:\n    pass\n")

    _build_submission(adapter, case, output, output_dir=tmp_path / "submission")

    wrapper = (tmp_path / "submission" / "cann_bench" / "moe_gating_top_k_softmax.py").read_text(
        encoding="utf-8"
    )
    assert "def moe_gating_top_k_softmax(x, finished=None, k=1):" in wrapper
    assert "k='1'" not in wrapper
    assert "return model(x=x, finished=finished, k=k)" in wrapper


def test_wrapper_parses_quoted_schema_defaults_as_python_literals(tmp_path):
    proto_path = tmp_path / "proto.yaml"
    proto_path.write_text(
        "\n".join(
            [
                "operator:",
                "  name: Gelu",
                '  schema: gelu(Tensor x, str approximate="none") -> Tensor y',
                "  inputs:",
                "    - name: x",
                "  attrs:",
                "    - name: approximate",
                "      default: none",
            ]
        ),
        encoding="utf-8",
    )
    case = CannBenchCase(
        bench_name="cann",
        task_dir=tmp_path,
        operator="Gelu",
        rel_path="tasks/level1/gelu",
        files={"proto": proto_path},
        metadata={},
    )
    adapter = create_converter("akg-agent", "cann", {})
    output = Artifact(
        status=AGENT_SUCCESS,
        workdir=tmp_path / "work",
        output_text="class ModelNew:\n    pass\n",
    )

    _build_submission(adapter, case, output, output_dir=tmp_path / "submission")

    wrapper = (tmp_path / "submission" / "cann_bench" / "gelu.py").read_text(encoding="utf-8")
    assert "def gelu(x, approximate='none'):" in wrapper
    assert "approximate='\"none\"'" not in wrapper


def test_wrapper_parses_lowercase_boolean_schema_defaults(tmp_path):
    proto_path = tmp_path / "proto.yaml"
    proto_path.write_text(
        "\n".join(
            [
                "operator:",
                "  name: ArgMax",
                "  schema: argmax(Tensor x, bool keepdim=false) -> Tensor y",
                "  inputs:",
                "    - name: x",
                "  attrs:",
                "    - name: keepdim",
                "      default: false",
            ]
        ),
        encoding="utf-8",
    )
    case = CannBenchCase(
        bench_name="cann",
        task_dir=tmp_path,
        operator="ArgMax",
        rel_path="tasks/level1/argmax",
        files={"proto": proto_path},
        metadata={},
    )
    adapter = create_converter("akg-agent", "cann", {})
    output = Artifact(
        status=AGENT_SUCCESS,
        workdir=tmp_path / "work",
        output_text="class ModelNew:\n    pass\n",
    )

    _build_submission(adapter, case, output, output_dir=tmp_path / "submission")

    wrapper = (tmp_path / "submission" / "cann_bench" / "argmax.py").read_text(encoding="utf-8")
    assert "def argmax(x, keepdim=False):" in wrapper
    assert "keepdim='false'" not in wrapper


def test_wrapper_keeps_required_attrs_required_when_proto_has_no_default(tmp_path):
    proto_path = tmp_path / "proto.yaml"
    proto_path.write_text(
        "\n".join(
            [
                "operator:",
                "  name: TopK",
                "  schema: top_k(Tensor x, int k, int dim, bool largest=True) -> Tensor values",
                "  inputs:",
                "    - name: x",
                "  attrs:",
                "    - name: k",
                "    - name: dim",
                "    - name: largest",
                "      default: true",
            ]
        ),
        encoding="utf-8",
    )
    case = CannBenchCase(
        bench_name="cann",
        task_dir=tmp_path,
        operator="TopK",
        rel_path="tasks/level1/top_k",
        files={"proto": proto_path},
        metadata={},
    )
    adapter = create_converter("akg-agent", "cann", {})
    output = Artifact(
        status=AGENT_SUCCESS,
        workdir=tmp_path / "work",
        output_text="class ModelNew:\n    pass\n",
    )

    _build_submission(adapter, case, output, output_dir=tmp_path / "submission")

    wrapper = (tmp_path / "submission" / "cann_bench" / "top_k.py").read_text(encoding="utf-8")
    assert "def top_k(x, k, dim, largest=True):" in wrapper
    assert "k=None" not in wrapper
    assert "dim=None" not in wrapper
    assert "return model(x=x, k=k, dim=dim, largest=largest)" in wrapper


def test_wrapper_keeps_required_attr_required_after_optional_tensor_default(tmp_path):
    proto_path = tmp_path / "proto.yaml"
    proto_path.write_text(
        "\n".join(
            [
                "operator:",
                "  name: Foo",
                "  schema: foo(Tensor x, Tensor? mask=None, int k) -> Tensor y",
                "  inputs:",
                "    - name: x",
                "    - name: mask",
                "  attrs:",
                "    - name: k",
            ]
        ),
        encoding="utf-8",
    )
    case = CannBenchCase(
        bench_name="cann",
        task_dir=tmp_path,
        operator="Foo",
        rel_path="tasks/level1/foo",
        files={"proto": proto_path},
        metadata={},
    )
    adapter = create_converter("akg-agent", "cann", {})
    output = Artifact(
        status=AGENT_SUCCESS,
        workdir=tmp_path / "work",
        output_text="class ModelNew:\n    pass\n",
    )

    _build_submission(adapter, case, output, output_dir=tmp_path / "submission")

    wrapper = (tmp_path / "submission" / "cann_bench" / "foo.py").read_text(encoding="utf-8")
    assert "def foo(x, mask=None, *, k):" in wrapper
    assert ", k=None" not in wrapper
    assert "return model(x=x, mask=mask, k=k)" in wrapper


def test_akg_to_cann_converter_reads_impl_file_output(tmp_path):
    client = CannBenchClient(REPO_ROOT)
    case = client.load_case("cann", "tasks/level1/exp")
    adapter = create_converter("akg-agent", "cann", {})
    workdir = tmp_path / "work"
    workdir.mkdir()
    impl_file = workdir / "impl.py"
    impl_file.write_text("class ModelNew:\n    pass\n", encoding="utf-8")
    output = Artifact(status=AGENT_SUCCESS, workdir=workdir, files={"impl": impl_file})

    submission = _build_submission(adapter, case, output, output_dir=tmp_path / "submission")

    impl = submission.source_dir.joinpath("cann_bench", "exp_triton_ascend_impl.py").read_text(encoding="utf-8")
    assert "class ModelNew:" in impl


def test_akg_to_cann_converter_rejects_workdir_as_submission_dir(tmp_path):
    client = CannBenchClient(REPO_ROOT)
    case = client.load_case("cann", "tasks/level1/exp")
    adapter = create_converter("akg-agent", "cann", {})
    workdir = tmp_path / "work"
    workdir.mkdir()
    output = Artifact(
        status=AGENT_SUCCESS,
        workdir=workdir,
        output_text="class ModelNew:\n    pass\n",
    )

    with pytest.raises(ValueError, match="refusing to overwrite agent workdir"):
        _build_submission(adapter, case, output, output_dir=workdir)


def test_akg_to_cann_converter_rejects_submission_dir_containing_workdir(tmp_path):
    client = CannBenchClient(REPO_ROOT)
    case = client.load_case("cann", "tasks/level1/exp")
    adapter = create_converter("akg-agent", "cann", {})
    workdir = tmp_path / "parent" / "work"
    workdir.mkdir(parents=True)
    output = Artifact(
        status=AGENT_SUCCESS,
        workdir=workdir,
        output_text="class ModelNew:\n    pass\n",
    )

    with pytest.raises(ValueError, match="refusing to overwrite agent workdir"):
        _build_submission(adapter, case, output, output_dir=workdir.parent)
