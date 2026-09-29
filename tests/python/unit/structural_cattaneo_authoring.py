"""Actual public authoring; select the intended source tree through pytest pythonpath."""
from pathlib import Path
import sys

import pops
import pytest
from pops.params import Positive, RuntimeParam

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples/migration/scientific"))
from structural_cattaneo import build_case


@pytest.mark.parametrize("variant", ["canonical", "permuted"])
def test_two_states_form_one_complete_principal_group_and_exact_local_residual(variant):
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    authored = build_case(variant=variant)
    plan = pops.resolve(pops.validate(authored.case), layout=authored.layout)
    assert sorted(len(state) for state in authored.states) == [1, 2]
    graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
    code = emit_cpp_program(plan.time, model=graph)
    assert code.count("PreparedPrincipalFlux<pops::kNativeDimension,3>") == 1
    assert code.count("resource.evaluate(") == 1
    assert code.count("_resource.publish(") == 1
    assert code.count("solve_prepared_local_nonlinear(") == 1
    assert "prepare_local_nonlinear_problem<2>" in code
    assert "Cval0[0]" in code and "Cval0[1]" in code and "Gval[" in code
    assert "parameter_sets[0]" in code and "parameter_sets[1]" in code
    assert "kFailRun" in code


@pytest.mark.parametrize("value", [0., -.1, float("nan"), float("inf")])
def test_public_runtime_parameter_declaration_refuses_invalid_tau(value):
    parameter = RuntimeParam("relaxation_time", default=.1, domain=Positive())
    with pytest.raises((TypeError, ValueError)):
        parameter.check_bind(value)
