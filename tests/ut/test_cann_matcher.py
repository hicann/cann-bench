import sys
import types

from kernel_eval.benches.cann_matcher import OperatorMatcher
from kernel_eval.benches.cann_spec import CannTaskSpec


class FakeLoader:
    def __init__(self, operators):
        self._operators = operators

    def list_operators(self):
        return list(self._operators)


def test_load_ai_operator_uses_schema_function_name_for_digit_boundaries(monkeypatch):
    def glm_v4_5_gate(*args):
        return args

    module = types.ModuleType("cann_bench")
    module.glm_v4_5_gate = glm_v4_5_gate
    monkeypatch.setitem(sys.modules, "cann_bench", module)

    matcher = OperatorMatcher(
        FakeLoader([
            CannTaskSpec(
                task_id="level1/glm_v4_5_gate",
                name="GlmV45Gate",
                schema="glm_v4_5_gate(Tensor x) -> Tensor y",
            )
        ])
    )

    assert matcher.load_ai_operator("GlmV45Gate") is glm_v4_5_gate
