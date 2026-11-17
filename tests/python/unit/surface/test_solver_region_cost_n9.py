"""N9 « Le coût de la région d'un solveur » (ROM-90): a solver's region records no face expression.

A reference solve lowers its solver's body as a region whose value leaves by a raw copy into a
buffer allocated in the prelude, as the native CG's solution. b6's face-record expressions are
kept for the publication of an accepted state, so the region's own expressions were computed and
then dropped at its boundary: the cost of the relation route (about 400 times the native witness
at 64², the PR of « La référence sous les solves de relation »). These tests read the emitted C++:
inside the region, the pure field algebra, no captured face, no write bracket and no affine
expression; outside it, the recorded lowering is unchanged. The native evidence (the same values,
the bench) is in ``tests/python/integration/runtime/test_relation_solve_reference_n9_native.py``
and ``benchmarks/n9_relation_cost``.
"""
from __future__ import annotations

import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_solver_region import (
    SOLVER_REGION, emit_axpy, emit_lincomb, enter_solver_region, in_solver_region,
)
from tests.python.unit.surface.test_solver_reference_n9 import _program, _reference, _solve
from tests.python.unit.surface.test_su_t3_solves import _cpp, _resolved, _surface_l1d

# The calls of b6's face-record algebra (FieldExchangeExpressions) that a field operation emits.
RECORDED = (
    "ctx.axpy(", "ctx.lincomb(", "ctx.copy_with_provenance(", "ctx.capture_stencil_exchanges(",
    "ctx.prepare_stencil_expression_write(", "ctx.finish_stencil_expression_write(",
    "ctx.attach_affine_flux_expression(",
)


def _reference_solve(program):
    (solve,) = [value for value in program._values if value.op == "solve_linear"
                and value.attrs.get("realization", {}).get("kind") == "reference"]
    return solve


def _split(source, solve):
    """The C++ before the solve's region, the region (its captured Result) and the rest."""
    lines = source.splitlines()
    start = lines.index(next(line for line in lines if line.strip()
                             == "std::int64_t capture_status_%d = 0;" % solve.attrs["body"].id))
    end = lines.index(next(line for line in lines if line.strip()
                           == "pops::SolveReport kr%d_region_report{};" % solve.id))
    return lines[:start], lines[start:end], lines[end:]


def _calls(lines, names=RECORDED):
    return {name: sum(line.count(name) for line in lines) for name in names
            if any(name in line for line in lines)}


@pytest.fixture(scope="module")
def l1d():
    elaborated = _surface_l1d().elaborated
    resolved = _resolved(elaborated.case, elaborated.layout)
    return resolved.time, _cpp(resolved)


def test_the_region_of_a_relation_solve_records_no_face_expression(l1d):
    # L1-d: the relation's operator applies the StencilFlux of two_point() at the iterates, and the
    # CG combines them; none of it is recorded, and the algebra is the pure one of the natives.
    program, source = l1d
    before, region, after = _split(source, _reference_solve(program))
    assert _calls(region) == {}
    assert _calls(region, ("pops::PureFieldAlgebra::axpy(",))["pops::PureFieldAlgebra::axpy("] > 0
    assert any(".apply(" in line and "stencil_prepared_" in line for line in region)
    assert not any("pointwise_exchange_coverage_mask" in line for line in region)
    # Outside the region the lowering still records: the right-hand side before it, the update
    # with the solution after it, and the publication of the accepted state.
    assert {"ctx.axpy(", "ctx.prepare_stencil_expression_write("} <= set(_calls(before))
    assert {"ctx.axpy(", "ctx.prepare_stencil_expression_write("} <= set(_calls(after))
    assert any("ctx.publish_stencil_exchanges(" in line for line in after)


def test_the_solution_leaves_by_a_raw_copy_into_its_prelude_buffer(l1d):
    # The result carries no expression, as the native CG's: a buffer allocated once in the
    # prelude, written by the raw copy of the region's value and by nothing that records.
    program, source = l1d
    solve = _reference_solve(program)
    solution = "sf_sol%d" % solve.id
    allocation = ("auto %s = std::make_shared<pops::MultiFab<pops::kNativeDimension>>("
                  "ctx.alloc_scalar_field(" % solution)
    assert sum(allocation in line for line in source.splitlines()) == 1
    _, _, after = _split(source, solve)
    copy = "pops::PureFieldAlgebra::copy(*%s, (*captured_%d_s1_owner));" % (
        solution, solve.attrs["body"].id)
    assert [line.strip() for line in after].count(copy) == 1
    # No recorded call takes the solution as its destination (its first argument).
    destinations = [name + prefix + solution for name in RECORDED for prefix in ("(*", "*")]
    assert not [line for line in source.splitlines()
                if any(destination in line for destination in destinations)]


def test_a_solver_region_over_a_b6_operator_records_nothing():
    # The region of a solve over a hand-written b6 operator (set_apply) is lowered the same way.
    program, _ = _program(_reference(max_iterations=50))
    source = emit_cpp_program(program)
    _, region, _ = _split(source, _solve(program))
    assert _calls(region) == {}


def test_the_pure_algebra_is_emitted_only_inside_a_solver_region():
    # Outside a region the helpers return the recorded statements byte for byte.
    var = {}
    assert not in_solver_region(var)
    assert emit_axpy(var, "u", "c", "x") == "ctx.axpy(u, c, x);"
    assert emit_axpy(var, "u", "c", "x", "{{0, 1, 1}}") == "ctx.axpy(u, c, x, dt, {{0, 1, 1}});"
    assert emit_lincomb(var, "u", "a", "x", "b", "y") == "ctx.lincomb(u, a, x, b, y);"
    region = dict(var)
    enter_solver_region(region, 7)
    assert region[SOLVER_REGION] == 7 and region[("stencil_flux_expressions",)] is False
    assert emit_axpy(region, "u", "c", "x", "{{0, 1, 1}}") == (
        "pops::PureFieldAlgebra::axpy(u, c, x);")
    assert emit_lincomb(region, "u", "a", "x", "b", "y") == (
        "pops::PureFieldAlgebra::lincomb(u, a, x, b, y);")
    assert not in_solver_region(var)  # the region's map is a copy
