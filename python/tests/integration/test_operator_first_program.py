"""Integration: operator-first Program shape and generated C++ route."""

import pytest

from examples.spec_final import manual_board_predictor_corrector_poisson_lorentz as manual
from pops.time import Program


def test_operator_first_program_rejects_string_operator_selectors():
    module = manual.build_model()
    program = Program("operator_selector_gate").bind_operators(module)
    U = program.state("U", block="plasma", space=module.state_spaces()["U"])

    with pytest.raises(TypeError, match="typed operator"):
        program.call("fields_from_state", U.n)


def test_operator_first_program_generates_calls_through_generated_module():
    module = manual.build_model()
    program = manual.build_program(module)
    cpp = program._emit_cpp_program_for_target(model=module)

    for op in ("fields_from_state", "explicit_rate", "implicit_operator"):
        assert "GeneratedModule::Operators::%s" % op in cpp

    body_start = cpp.index("auto generated_program_body")
    body_end = cpp.index("ctx.install", body_start)
    program_body = cpp[body_start:body_end]
    assert "ctx.rhs_into" not in program_body
    assert "ctx.solve_fields_from_state" not in program_body
    assert "ctx.source_default_into" not in program_body
    assert "ctx.neg_div_flux_default_into" not in program_body
