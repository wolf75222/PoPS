"""Receive version selection through public graphs and independent legacy routes."""
import pops
import pytest

from pops._ir.quantity import PhysicalDimension
from pops.time import FailRun
from tests.python.unit.codegen.test_integral_candidate_capture import build_feedback
from tests.python.unit.fields.test_nonlinear_mixed_field_problem import mixed_case


def test_legacy_declarations_and_pairings_keep_their_versions():
    program = pops.Program("legacy_integral_contract")
    program.integral_state("q", initial=.7)
    assert program._serialize()["version"] == 5
    _, _, program, current, *_ = mixed_case()
    assert program._serialize()["version"] == 5
    program.dot_all(current.n, current.n)
    assert program._serialize()["version"] == 7


def test_typed_declaration_alone_discriminates_its_identity_contract():
    program = pops.Program("typed_integral_contract")
    program.integral_state("q", initial=.7, units=PhysicalDimension())
    serialized = program._serialize()
    assert serialized["version"] == 8
    assert serialized["integral_units_v2"]


@pytest.mark.parametrize("pairing_first", (False, True))
def test_original_spatial_residual_version_has_order_independent_precedence(pairing_first):
    _, field, program, current, request, *_ = mixed_case()
    if pairing_first:
        program.dot_all(current.n, current.n)
    program.solve(request, solver=field.default_program_solver()).consume(action=FailRun())
    if not pairing_first:
        program.dot_all(current.n, current.n)
    serialized = program._serialize()
    assert {node["op"] for node in serialized["nodes"]} >= {"solve_spatial_field", "reduce"}
    assert serialized["version"] == 8


def test_original_spatial_residual_inside_lazy_region_selects_version_eight():
    _, field, program, current, request, *_ = mixed_case()

    def with_solve(author):
        author.solve(request, solver=field.default_program_solver()).consume(action=FailRun())
        return current.n

    program.branch(program.norm2(current.n) > 0, with_solve, lambda author: current.n)
    assert not any(node.op == "solve_spatial_field" for node in program._values)
    assert program._serialize()["version"] == 8


def test_typed_capture_and_both_scalar_expression_routes_survive_integration():
    _, _, program, _, temporal = build_feedback()
    # The legacy reduction scalar and the typed global leaf use distinct component names.
    scalar = program.dot_all(temporal.n, temporal.n)
    from pops.time.expressions import component_names
    assert component_names(scalar) == ("scalar",)
    capture = next(node for node in program._values if node.op == "integral_candidate")
    assert component_names(capture) == ("value",)
    assert program._serialize()["version"] == 8
