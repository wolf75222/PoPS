"""Public Model Field operands and imposed Aux provider requirements; Source only."""
from fractions import Fraction

import pytest

import pops
from pops.analytic import time as analytic_time
from pops.codegen.component_provider_packs import resolve_component_provider_packs
from pops.domain import Rectangle
from pops.fields import AnalyticAux, DerivedAux
from pops.frames import Cartesian2D
from pops.lib.time import IMEX
from pops.math import ValueExpr, ddt
from pops.time._evaluation_point import evaluation_stage_fraction


def _model(name="aux_field_contract"):
    frame = Rectangle("unit", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model(name, frame=frame)
    state = model.state("U", components=("u",))
    return model, state, frame


def _program(model, state, implicit):
    explicit = model.rate("explicit", equation=ddt(state) == model.source(
        "explicit_source", on=state, value=(state[0],)))
    case = pops.Case("aux_field_case")
    block = case.block("material", model)
    return IMEX(block[state], explicit_operator=explicit, implicit_operator=implicit), explicit


@pytest.mark.parametrize("explicit_empty", [False, True], ids=["inferred", "explicit-empty"])
def test_constant_implicit_operator_remains_nullary_and_same_scheme(explicit_empty):
    model, state, _ = _model()
    body = model.local_linear_operator("implicit", on=state, matrix=((-2.0,),))
    implicit = model.operator("implicit", returns=body,
                              **({"inputs": ()} if explicit_empty else {}))
    assert implicit.signature.inputs == ()
    program, _ = _program(model, state, implicit)
    assert program.validate() is True
    solve = next(v for v in program._values if v.op == "solve_local_linear")
    assert evaluation_stage_fraction(solve, ark_partition="implicit") == Fraction(1)


@pytest.mark.parametrize("explicit_empty", [False, True], ids=["inferred", "explicit-empty"])
def test_analytic_clock_auxiliary_is_not_a_field_operand(explicit_empty):
    model, state, frame = _model()
    tau = model.auxiliary("stage_tau", frame=frame.canonical_id)
    implicit = model.operator("implicit", returns=model.local_linear_operator(
        "implicit", on=state, matrix=((-(2 + tau),),)),
        **({"inputs": ()} if explicit_empty else {}))
    assert implicit.signature.inputs == ()
    op = model.module.operator_registry().get("implicit")
    assert dict(op.requirements) == {"aux": ["stage_tau"]}
    program, _ = _program(model, state, implicit)
    module = model.module
    module.aux_provider(AnalyticAux(module.aux_handle(module.aux()["stage_tau"]),
                                  analytic_time(program.clock), frame=frame))
    packs = resolve_component_provider_packs(module)
    assert [(key.space_kind, key.space_name, key.component)
            for key in packs.by_operator["implicit"]] == [("aux", "stage_tau", "stage_tau")]
    assert program.validate() is True
    solve = next(v for v in program._values if v.op == "solve_local_linear")
    apply = next(v for v in program._values if v.op == "apply")
    assert evaluation_stage_fraction(solve, ark_partition="implicit") == 1
    assert evaluation_stage_fraction(apply) == 1


def test_declared_extra_field_operand_is_honored_and_imex_refuses_coupling():
    model, state, _ = _model()
    model.aux("electric")
    implicit = model.operator("implicit", inputs=("fields",),
        returns=model.local_linear_operator("implicit", on=state, matrix=((-2.0,),)))
    assert len(implicit.signature.inputs) == 1
    with pytest.raises(ValueError, match="field-independent"):
        _program(model, state, implicit)


def test_true_field_inference_and_explicit_empty_refusal_are_atomic():
    model, state, _ = _model()
    electric = model.aux("electric")
    implicit = model.operator("implicit", returns=model.local_linear_operator(
        "implicit", on=state, matrix=((electric,),)))
    assert len(implicit.signature.inputs) == 1
    before = model.module.module_hash()
    with pytest.raises(ValueError, match="omit a solved FieldSpace"):
        model.operator("hidden", inputs=(), returns=model.local_linear_operator(
            "hidden", on=state, matrix=((electric,),)))
    assert model.module.module_hash() == before
    assert "hidden" not in model._dsl._m._linear_sources
    assert "hidden" not in model._dsl._m._declared_linear_operator_inputs
    with pytest.raises(ValueError, match="field-independent"):
        _program(model, state, implicit)


@pytest.mark.parametrize("inputs", [("foreign",), ("fields", "fields")])
def test_invalid_declared_field_operands_do_not_mutate_registry(inputs):
    model, state, _ = _model()
    before = model.module.module_hash()
    with pytest.raises(ValueError, match="distinct declared FieldSpaces"):
        model.operator("invalid", inputs=inputs, returns=model.local_linear_operator(
            "invalid", on=state, matrix=((-2.0,),)))
    assert model.module.module_hash() == before


def test_derived_auxiliary_cannot_hide_transitive_solved_field_dependency():
    model, state, _ = _model()
    model.aux("electric")
    coefficient = model.auxiliary("coefficient")
    model.auxiliary("intermediate")
    implicit = model.operator("implicit", inputs=(), returns=model.local_linear_operator(
        "implicit", on=state, matrix=((coefficient,),)))
    assert implicit.signature.inputs == ()
    module = model.module
    middle = module.aux_handle(module.aux()["intermediate"])
    solved = module.field_handle(module.field_spaces()["fields"])
    module.aux_provider(DerivedAux(middle, ValueExpr(solved)))
    module.aux_provider(DerivedAux(module.aux_handle(module.aux()["coefficient"]),
                                  ValueExpr(middle)))
    with pytest.raises(ValueError, match="hides solved FieldSpace dependency"):
        resolve_component_provider_packs(module)


def test_multistate_infers_exact_field_operands_and_refuses_explicit_omission():
    model = pops.Model("multi_aux_field")
    model.species("first", state=("u",))
    selected = model.species("selected", state=("v",))
    model.field("coefficient", components=("rate",))
    module = model.module
    space = module.field_spaces()["coefficient"]
    rate, = module.field_symbols(space)
    body = model.local_linear_operator("implicit", on=selected, matrix=((rate,),))
    implicit = model.operator("implicit", returns=body)
    assert implicit.signature.inputs == (space,)
    before = module.module_hash()
    with pytest.raises(ValueError, match="omit a solved FieldSpace"):
        model.operator("hidden", inputs=(), returns=body)
    assert module.module_hash() == before
