import importlib.util

import yaml

from auto_pipeline.generator.pypto.case_classifier import classify_cases, write_class_cases
from auto_pipeline.generator.pypto.dispatcher import write_dispatcher


def _write(path, cases):
    path.write_text(yaml.safe_dump({"cases": cases}), encoding="utf-8")
    return path


def test_classify_splits_by_dim_and_dtype(tmp_path):
    path = _write(tmp_path / "cases.yaml", [
        {"case_id": 1, "input_shape": [[2, 3, 4]], "dtype": ["float16"]},
        {"case_id": 2, "input_shape": [[2, 3]], "dtype": ["float16"]},
        {"case_id": 3, "input_shape": [[5, 6, 7]], "dtype": ["float32"]},
        {"case_id": 4, "input_shape": [[8, 9, 10]], "dtype": ["float16"]},
    ])
    classes = classify_cases(path)
    assert [c.subdir for c in classes] == ["c1", "c2", "c3"]
    assert [len(c.cases) for c in classes] == [2, 1, 1]
    assert classes[0].signature == ((3, "float16"),)
    assert classes[1].signature == ((2, "float16"),)


def test_uniform_cases_degenerate_to_single_class(tmp_path):
    path = _write(tmp_path / "cases.yaml", [
        {"case_id": i, "input_shape": [[2048, 2048]], "dtype": ["float32"]} for i in range(3)
    ])
    classes = classify_cases(path)
    assert len(classes) == 1 and classes[0].subdir == "."


def test_missing_cases_file_is_single_class(tmp_path):
    classes = classify_cases(tmp_path / "nope.yaml")
    assert len(classes) == 1 and classes[0].subdir == "."


def test_write_class_cases_only_for_split(tmp_path):
    path = _write(tmp_path / "cases.yaml", [
        {"case_id": 1, "input_shape": [[2, 3]], "dtype": ["float16"]},
        {"case_id": 2, "input_shape": [[2, 3, 4]], "dtype": ["float32"]},
    ])
    classes = classify_cases(path)
    target = tmp_path / "c1" / "cases.yaml"
    assert write_class_cases(path, classes[0], target) == target
    assert len(yaml.safe_load(target.read_text())["cases"]) == 1


class _T:
    def __init__(self, ndim, dtype):
        self._n, self.dtype = ndim, dtype

    def dim(self):
        return self._n


def test_dispatcher_routes_by_dim_and_dtype(tmp_path):
    op = tmp_path / "gelu"
    for sub in ("c1", "c2"):
        (op / sub).mkdir(parents=True)
        (op / sub / "gelu_impl.py").write_text(f"def gelu(x):\n    return ({sub!r}, x.dim())\n", encoding="utf-8")
    manifest = {"op_name": "gelu", "classes": [
        {"subdir": "c1", "signature": [[2, "float16"]]},
        {"subdir": "c2", "signature": [[3, "float32"]]},
    ]}
    path = write_dispatcher(op, manifest)
    spec = importlib.util.spec_from_file_location("gelu_entry", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.gelu(_T(2, "torch.float16")) == ("c1", 2)
    assert mod.gelu(_T(3, "torch.float32")) == ("c2", 3)
    # pypto exposes <op>_wrapper, not <op>: dispatcher must resolve it
    (op / "c1" / "gelu_impl.py").write_text("def gelu_wrapper(x):\n    return ('wrap', x.dim())\n", encoding="utf-8")
    s2 = importlib.util.spec_from_file_location("g2", write_dispatcher(op, manifest))
    m2 = importlib.util.module_from_spec(s2)
    s2.loader.exec_module(m2)
    assert m2.gelu(_T(2, "torch.float16")) == ("wrap", 2)
    try:
        mod.gelu(_T(4, "torch.int8"))
        assert False
    except ValueError:
        pass
