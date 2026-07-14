"""Exact prepared-boundary lowering for the matrix-free RHS Jacobian-vector product."""
from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import pytest

from pops.codegen.program_emit_solve import (
    _rhs_stage_fraction,
    _validate_matrix_free_contract,
)
from pops.codegen.program_codegen import emit_cpp_program
from pops.linalg import LinearProblem
from pops.solvers import krylov
from pops.time import FailRun, Program, StagePoint, TimePoint

from typed_program_support import solve_field, typed_field, typed_state


def _emit(*, sources, field_coupled=False):
    program = Program("rhs_jacvec_boundary_emit")
    temporal = typed_state(program, "fluid", state_name="U")
    point = (
        TimePoint(program.clock, Fraction(1, 3))
        if field_coupled else
        StagePoint(
            "implicit_stage",
            {
                "explicit": TimePoint(program.clock, Fraction(1, 3)),
                "implicit": TimePoint(program.clock, Fraction(2, 3)),
            },
        )
    )
    iterate = program.value("newton_iterate", 1 * temporal.n, at=point)
    fields = (
        solve_field(
            program, iterate, field=typed_field(program, "potential"), name="newton_fields")
        if field_coupled else None
    )
    r0 = program._rhs_primitive(
        name="frozen_rhs", state=iterate, fields=fields, flux=True, sources=sources)
    operator = program.matrix_free_operator(
        "newton_jacobian", domain="state", range_="state", ncomp=1)
    jacvec = []

    def apply(builder, out, direction):
        result = builder.rhs_jacvec(
            out,
            direction,
            iterate=iterate,
            r0=r0,
            c_dt=Fraction(2, 3) * builder.dt,
            eps=Fraction(1, 10_000_000),
            flux=True,
            sources=sources,
            field_coupled=field_coupled,
        )
        jacvec.append(result)
        return result

    operator = program.set_apply(operator, apply)
    linear_rhs = program.value("linear_rhs", -1 * iterate, at=point)
    correction = program.solve(
        LinearProblem(operator, linear_rhs, at=point),
        solver=krylov.GMRES(max_iter=4, rel_tol=1.0e-8, restart=2),
    ).consume(action=FailRun())
    endpoint = program.value("next", 1 * correction, at=temporal.next.point)
    program.commit(temporal.next, endpoint)
    field_plans = None
    if field_coupled:
        field_plans = {
            "potential": SimpleNamespace(
                name="potential",
                native_options={
                    "provider_slot": "provider::potential::sha256:exact",
                    "output_route": {"components": ("potential",)},
                },
            )
        }
    source = emit_cpp_program(program, field_plans=field_plans)
    return source, operator, jacvec[0], r0


def _names(operator, jacvec):
    suffix = "%d_%d" % (operator.id, jacvec.id)
    return {
        "point": "jac_point" + suffix,
        "has_boundary": "jac_has_boundary" + suffix,
        "r0_core": "jac_r0_core" + suffix,
        "boundary_work": "jac_boundary_work" + suffix,
        "field_slot": "jac_field_slot" + suffix,
        "cdt": "jac_cdt" + suffix,
    }


def _apply_source(source, operator):
    start = source.index("pops::ApplyFn apply_A%d" % operator.id)
    return source[start:source.index("\n  };", start) + len("\n  };")]


def test_apply_captures_point_and_only_conditionally_allocates_boundary_scratch():
    source, operator, jacvec, _ = _emit(sources=None)
    names = _names(operator, jacvec)
    apply_source = _apply_source(source, operator)

    point_allocation = source.index(
        "std::make_shared<pops::runtime::multiblock::BoundaryEvaluationPoint>()")
    apply_declaration = source.index("pops::ApplyFn apply_A%d" % operator.id)
    begin_step = source.index("ctx.begin_step(dt)")
    point_refresh = source.index("*%s = ctx.boundary_evaluation_point(" % names["point"])
    assert point_allocation < apply_declaration < begin_step < point_refresh
    assert names["point"] in apply_source.splitlines()[0]
    assert "const bool %s = ctx.has_boundary_linearization(0);" % names["has_boundary"] in source

    for scratch in (names["r0_core"], names["boundary_work"]):
        declaration = next(line for line in source.splitlines() if "auto %s =" % scratch in line)
        assert "%s ? std::make_shared<pops::MultiFab>(" % names["has_boundary"] in declaration
        assert "std::shared_ptr<pops::MultiFab>{}" in declaration
        assert apply_source.count(scratch) >= 1
    assert "std::make_shared" not in apply_source


def test_step_refresh_uses_r0_exact_explicit_stage_and_separates_boundary_residual():
    source, operator, jacvec, r0 = _emit(sources=None)
    names = _names(operator, jacvec)
    refresh = "*%s = ctx.boundary_evaluation_point(%d);" % (names["point"], r0.id)
    refresh_index = source.index(refresh)
    preceding = source[:refresh_index]
    assert preceding.rfind("ctx.set_stage_time(1, 3);") > preceding.rfind("ctx.begin_step(dt)")

    residual = (
        "ctx.boundary_residual_into_at(*%s, 0, *jac_uk%d_%d, *%s);"
        % (names["point"], operator.id, jacvec.id, names["boundary_work"])
    )
    subtract = (
        "ctx.axpy(*%s, static_cast<pops::Real>(-1), *%s);"
        % (names["r0_core"], names["boundary_work"])
    )
    assert source.index(refresh) < source.index(residual) < source.index(subtract)


@pytest.mark.parametrize(("sources", "flux_only"), [(None, "false"), ([], "true")])
def test_apply_uses_point_qualified_core_and_exact_boundary_jvp(sources, flux_only):
    source, operator, jacvec, _ = _emit(sources=sources)
    names = _names(operator, jacvec)
    apply_source = _apply_source(source, operator)

    assert (
        "ctx.rhs_core_into_at(*%s, 0, *jac_up%d_%d, *jac_rp%d_%d, %s);"
        % (names["point"], operator.id, jacvec.id, operator.id, jacvec.id, flux_only)
    ) in apply_source
    assert (
        "ctx.boundary_jvp_into_at(*%s, 0, *jac_uk%d_%d, "
        "const_cast<pops::MultiFab&>(in), *%s);"
        % (names["point"], operator.id, jacvec.id, names["boundary_work"])
    ) in apply_source
    assert "ctx.axpy(out, -*%s, *%s);" % (names["cdt"], names["boundary_work"]) \
        in apply_source
    assert "ctx.rhs_into" not in apply_source
    assert "ctx.neg_div_flux_default_into" not in apply_source
    assert "ctx.boundary_evaluation_point" not in apply_source


def test_zero_direction_has_a_positive_fallback_step_instead_of_dividing_by_zero():
    source, operator, jacvec, _ = _emit(sources=[])
    apply_source = _apply_source(source, operator)
    step_line = next(line for line in apply_source.splitlines() if "const pops::Real jh" in line)
    assert "jvn > pops::Real(0) ?" in step_line
    assert "/ jvn : static_cast<pops::Real>(" in step_line
    assert "const pops::Real jc = *jac_cdt%d_%d / jh;" % (operator.id, jacvec.id) \
        in apply_source


def test_field_coupled_apply_restores_the_frozen_provider_after_the_perturbed_rhs():
    source, operator, jacvec, _ = _emit(sources=None, field_coupled=True)
    names = _names(operator, jacvec)
    apply_source = _apply_source(source, operator)
    assert (
        'const std::string %s = "provider::potential::sha256:exact";'
        % names["field_slot"]
    ) in source
    assert names["field_slot"] in apply_source.splitlines()[0]
    perturbed_solve = (
        "ctx.solve_fields_from_state_at(*%s, %s, 0, *jac_up%d_%d);"
        % (names["point"], names["field_slot"], operator.id, jacvec.id)
    )
    frozen_solve = (
        "ctx.solve_fields_from_state_at(*%s, %s, 0, *jac_uk%d_%d);"
        % (names["point"], names["field_slot"], operator.id, jacvec.id)
    )
    perturbed_rhs = "ctx.rhs_core_into_at(*%s" % names["point"]
    boundary_jvp = "ctx.boundary_jvp_into_at(*%s" % names["point"]
    assert apply_source.count("ctx.solve_fields_from_state_at(") == 2
    assert (
        apply_source.index(perturbed_solve)
        < apply_source.index(perturbed_rhs)
        < apply_source.index(frozen_solve)
        < apply_source.index(boundary_jvp)
    )
    assert "ctx.solve_fields_from_state(0, *jac_up" not in apply_source


def test_codegen_refuses_to_invent_a_missing_r0_stage_point():
    with pytest.raises(ValueError, match="exact TimePoint or StagePoint"):
        _rhs_stage_fraction(SimpleNamespace(point=object()))


def test_codegen_defensively_rejects_a_forged_rhs_base():
    point = object()
    block = object()
    iterate = SimpleNamespace(block=block, point=point)
    other = SimpleNamespace(block=block, point=point)
    r0 = SimpleNamespace(
        op="rhs",
        inputs=(other,),
        block=block,
        point=point,
        attrs={"flux": True, "sources": None, "fluxes": None},
    )
    jacvec = SimpleNamespace(
        op="rhs_jacvec",
        inputs=(object(), object(), iterate, r0),
        attrs={"field_coupled": False, "flux": True, "sources": None},
    )
    with pytest.raises(ValueError, match="exact frozen iterate"):
        _validate_matrix_free_contract(jacvec, None)


def test_codegen_defensively_rejects_an_ambiguous_coupled_field_context():
    block = object()
    point = object()
    iterate = SimpleNamespace(block=block, point=point, id=7)
    context = SimpleNamespace(
        field=object(),
        stage_sources=((block, iterate.id), (object(), 11)),
    )
    fields = SimpleNamespace(vtype="fields", field_context=context)
    r0 = SimpleNamespace(
        op="rhs",
        inputs=(iterate, fields),
        block=block,
        point=point,
        attrs={"flux": True, "sources": None, "fluxes": None},
        field_context=context,
    )
    jacvec = SimpleNamespace(
        op="rhs_jacvec",
        inputs=(object(), object(), iterate, r0),
        attrs={"field_coupled": True, "flux": True, "sources": None},
    )
    with pytest.raises(ValueError, match="unambiguous field context"):
        _validate_matrix_free_contract(jacvec, None)
