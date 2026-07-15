"""Authoring-time authentication of the frozen residual used by rhs_jacvec."""
from __future__ import annotations

from fractions import Fraction

import pytest

from pops import model as model_api
from pops.numerics.terms import DefaultSource, Flux
from pops.runtime.amr_program_support import (
    AMRProgramSupportContext,
    amr_program_op_support,
)
from pops.time import Program, StagePoint, TimePoint

from typed_program_support import solve_field, typed_field, typed_state


def _program():
    module = model_api.Module("rhs_jacvec_contract_model")
    state_space = module.state_space("U", ("u",))
    signature = (state_space,) >> model_api.Rate(state_space)
    module.operator(
        name="flux", signature=signature, kind="grid_operator", expr="default_flux")
    named_flux = module.operator(
        name="named_flux", signature=signature, kind="grid_operator", expr="named_flux")
    state = module.state_handle(state_space)
    program = Program("rhs_jacvec_contract")
    temporal = typed_state(
        program, "fluid", state_name="U", model=module, state=state)
    point = StagePoint(
        "implicit_stage",
        {
            "explicit": TimePoint(program.clock, Fraction(1, 3)),
            "implicit": TimePoint(program.clock, Fraction(2, 3)),
        },
    )
    iterate = program.value("iterate", 1 * temporal.n, at=point)
    operator = program.matrix_free_operator(
        "J", domain="state", range_="state", ncomp=1)
    return program, iterate, operator, named_flux


def _record(program, operator, iterate, r0, *, sources=("default",), field_coupled=False):
    return program.set_apply(
        operator,
        lambda builder, out, direction: builder.rhs_jacvec(
            out,
            direction,
            iterate=iterate,
            r0=r0,
            c_dt=builder.dt,
            sources=sources,
            field_coupled=field_coupled,
        ),
    )


def test_rhs_jacvec_rejects_a_field_that_is_not_an_rhs_node():
    program, iterate, operator, _named_flux = _program()
    with pytest.raises(ValueError, match=r"exact precomputed rhs\(iterate\)"):
        _record(program, operator, iterate, iterate)


def test_rhs_jacvec_rejects_an_rhs_computed_from_another_iterate():
    program, iterate, operator, _named_flux = _program()
    other = program.value("other_iterate", 1 * iterate, at=iterate.point)
    r0 = program.rhs(name="r0", state=other, terms=[Flux(), DefaultSource()])
    with pytest.raises(ValueError, match="exact frozen iterate"):
        _record(program, operator, iterate, r0)


def test_rhs_jacvec_rejects_r0_at_another_temporal_point():
    program, iterate, operator, _named_flux = _program()
    r0 = program.rhs(name="r0", state=iterate, terms=[Flux(), DefaultSource()])
    misplaced = program.value(
        "misplaced_r0", r0, at=TimePoint(program.clock, Fraction(3, 4)))
    with pytest.raises(ValueError, match="exact block and temporal point"):
        _record(program, operator, iterate, misplaced)


def test_rhs_jacvec_rejects_a_different_default_source_selection():
    program, iterate, operator, _named_flux = _program()
    full_r0 = program.rhs(name="r0", state=iterate, terms=[Flux(), DefaultSource()])
    with pytest.raises(ValueError, match="exact same default-flux/default-source"):
        _record(program, operator, iterate, full_r0, sources=[])


def test_rhs_jacvec_rejects_a_named_flux_base_residual():
    program, iterate, operator, named_flux = _program()
    named_r0 = program.rhs(name="r0", state=iterate, terms=[Flux(named_flux)])
    with pytest.raises(ValueError, match="may not use a named flux"):
        _record(program, operator, iterate, named_r0, sources=[])


def test_rhs_jacvec_accepts_one_exact_field_context_from_the_frozen_iterate():
    program, iterate, operator, _named_flux = _program()
    field = typed_field(program, "potential")
    fields = solve_field(program, iterate, field=field, name="iterate_fields")
    r0 = program.rhs(
        name="r0", state=iterate, fields=fields, terms=[Flux(), DefaultSource()])
    completed = _record(
        program, operator, iterate, r0, field_coupled=True)
    jacvec = next(node for node in completed.attrs["apply_block"] if node.op == "rhs_jacvec")
    assert jacvec.attrs["field_coupled"] is True
    assert r0.field_context.field is field


def test_recursive_ir_exposes_field_coupled_jacvec_to_the_amr_capability_gate():
    program, iterate, operator, _named_flux = _program()
    field = typed_field(program, "potential")
    fields = solve_field(program, iterate, field=field, name="iterate_fields")
    r0 = program.rhs(
        name="r0", state=iterate, fields=fields,
        terms=[Flux(), DefaultSource()])
    _record(program, operator, iterate, r0, field_coupled=True)

    assert "rhs_jacvec" not in {node["op"] for node in program.ir_nodes()}
    recursive = program.ir_nodes(recursive=True)
    assert any(
        node["op"] == "rhs_jacvec"
        and node["attrs"].get("field_coupled") is True
        for node in recursive
    )
    context = AMRProgramSupportContext(
        refined_hierarchy=True,
        shared_block_interfaces=False,
        field_routes_validated=True,
    )
    assert amr_program_op_support(program, context=context) == {
        "fine_level_field_perturbation": "pending",
        "named_field_solve": "green",
    }


def test_rhs_jacvec_rejects_field_coupling_without_an_exact_field_context():
    program, iterate, operator, _named_flux = _program()
    r0 = program.rhs(name="r0", state=iterate, terms=[Flux(), DefaultSource()])
    with pytest.raises(ValueError, match="one unambiguous field context"):
        _record(program, operator, iterate, r0, field_coupled=True)
