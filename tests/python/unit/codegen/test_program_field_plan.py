"""Program field admission requires a connected solve graph and exact physical source mapping."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import pops
from pops.codegen.program_field_plan import capture_program_field_plans
from pops.fields import FieldDiscretization, FieldProblem
from pops.fields.methods import CellCenteredSecondOrder
from pops.math import laplacian
from pops.model import Handle, OwnerPath
from pops.solvers import CartesianCG
from pops.time._graph.base import CanonicalData


def _graph():
    case = pops.Case("program-field")
    original = Handle("potential", kind="field", owner=OwnerPath.model("equation"))
    physical = FieldProblem("potential", unknowns=(original,),
                            equations=(-laplacian(original) == 1,))
    field = case.field(physical, FieldDiscretization(
        method=CellCenteredSecondOrder(), boundaries=(), solver=CartesianCG()))
    registered = case._field_registry.resolved_registration(field)
    physical = registered.operator
    metadata = {"field_problem_identity": physical.identity.token, "field_dependencies": ()}
    load = SimpleNamespace(id=0, op="field_problem_load", attrs=dict(metadata), inputs=())
    apply = SimpleNamespace(id=1, op="field_problem_apply", attrs=dict(metadata), inputs=())
    operator = SimpleNamespace(id=2, op="matrix_free_operator",
                               attrs={"apply_block": (apply,)}, inputs=())
    solve = SimpleNamespace(id=3, op="solve_linear", inputs=(operator, load), attrs={
        "solve_request": {"physical_problem": CanonicalData({
            "field_handle": case.resolve(field).canonical_identity(),
            "field_problem": physical.to_data(),
            "unknown_components": ("potential",),
        }).to_data()}})
    program = SimpleNamespace(_values=(load, operator, solve))
    layout = Handle("default", kind="layout", owner=case.owner_path.canonical())
    layout_plan = SimpleNamespace(layout_for=lambda _field: layout)
    return case, program, layout_plan, load, apply, solve


def _capture(case, program, layout):
    return capture_program_field_plans(case, lambda value: value, target="system",
                                       layout_plan=layout, program=program)["potential"]


def test_program_field_plan_seals_independent_storage_and_actual_solve_nodes():
    case, program, layout, _load, _apply, _solve = _graph()
    plan = _capture(case, program, layout)
    assert plan.solve_node_ids == (3,)
    assert plan.storage.unknowns == plan.operator.unknowns
    assert plan.storage.unknowns[0].block_ref is None
    assert plan.coverage.rows[0].targets == ("program:solve_linear:3",)
    assert not hasattr(plan, "native_options")
    assert not hasattr(plan, "rhs_providers")
    plan.validate_program(program)


def test_unconnected_load_witness_cannot_admit_descriptor_only_field():
    case, program, layout, _load, _apply, solve = _graph()
    solve.inputs = (solve.inputs[0],)
    with pytest.raises(ValueError, match="executable load and apply"):
        _capture(case, program, layout)


def test_program_field_admission_refuses_changed_equations_or_input_reads():
    case, program, layout, load, _apply, solve = _graph()
    plan = _capture(case, program, layout)
    load.attrs["field_dependencies"] = (
        Handle("foreign", kind="state", owner=OwnerPath.model("other")),)
    with pytest.raises(ValueError, match="native input reads"):
        plan.validate_program(program)
    load.attrs["field_dependencies"] = ()
    solve.attrs["solve_request"]["physical_problem"]["field_problem"] = {}
    with pytest.raises(ValueError, match="physical equations"):
        plan.validate_program(program)


def test_program_field_admission_refuses_unconsumed_registration_and_amr_route():
    case, program, layout, _load, _apply, _solve = _graph()
    plan = _capture(case, program, layout)
    with pytest.raises(NotImplementedError, match="Uniform"):
        replace(plan, target="amr_system")
    program._values = ()
    with pytest.raises(ValueError, match="explicit Program solve"):
        _capture(case, program, layout)


def test_program_field_plan_revalidation_refuses_changed_identity():
    case, program, layout, _load, _apply, _solve = _graph()
    plan = _capture(case, program, layout)
    object.__setattr__(plan, "solve_node_ids", (9,))
    with pytest.raises(ValueError, match="identity changed"):
        plan.validate_program(program)
