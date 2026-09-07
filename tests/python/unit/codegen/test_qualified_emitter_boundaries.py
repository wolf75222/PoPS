"""Qualified source identities are translated only in private native emitter views."""
import pytest

from pops._ir.quantity import QuantityRef
from pops.codegen.component_provider_packs import require_emitter_provider_carrier
from pops.codegen.program_codegen import emit_cpp_program
from pops.model import Module
from pops.model.state_symbols import native_formula_carrier_view
from pops.physics import Model
from tests.python.support.physics_roles import FRAME, X_AXIS, Y_AXIS
from test_coupled_implicit_codegen import _program


def test_direct_formula_carrier_emission_authenticates_a_detached_view():
    model = Model("raw_qualified", frame=FRAME)
    state = model.state("U", components=("u",))
    u = QuantityRef(state, "u", space=state.space)
    pressure = model.primitive("pressure", u * u)
    model.flux("F", frame=FRAME, state=state,
               components={X_AXIS: (pressure,), Y_AXIS: (pressure,)},
               waves={X_AXIS: (1,), Y_AXIS: (1,)})
    model._dsl._model_hash()
    source = model._dsl._m
    authored_recipe = source.prim_defs["pressure"]
    authority = source._formula_source_module
    before = authority.module_hash()
    view = native_formula_carrier_view(source)
    assert view is not source
    assert view.prim_defs["pressure"].a.deps() == {"u"}
    assert source.prim_defs["pressure"] is authored_recipe
    assert isinstance(authored_recipe.a, QuantityRef)
    require_emitter_provider_carrier(view)
    require_emitter_provider_carrier(source)
    cpp = source.emit_cpp_brick(name="DirectQualified")
    assert "struct DirectQualified" in cpp
    assert "const pops::Real pressure" in cpp
    assert authority.module_hash() == before
    assert source._formula_source_module is authority
    assert source.prim_defs["pressure"] is authored_recipe
    object.__setattr__(source, "_formula_source_module", Module("foreign"))
    with pytest.raises(ValueError, match="Module authority"):
        native_formula_carrier_view(source)


def test_coupled_emission_preserves_source_quantities_and_binds_private_coordinates():
    module, program, _ = _program()
    body = module.operator_registry().get("collision").body
    authored = body["electrons"][0]
    assert isinstance(authored.a, QuantityRef)
    cpp = emit_cpp_program(program, model=None)
    assert "pops_input_0_component_0" in cpp
    assert "pops_input_1_component_0" in cpp
    assert module.operator_registry().get("collision").body["electrons"][0] is authored
    assert isinstance(authored.a, QuantityRef)


def test_coupled_emission_rejects_equal_named_quantity_from_a_foreign_owner():
    module, program, _ = _program()
    operator = module.operator_registry().get("collision")
    foreign = Module("same_physical_display")
    space = foreign.state_space("electron_state", ("ne", "pex", "pey"))
    ne, _, _ = foreign.state_symbols(space)
    altered = {name: tuple(values) for name, values in operator.body.items()}
    altered["electrons"] = (ne, *altered["electrons"][1:])
    operator.body = altered
    with pytest.raises(ValueError, match="exactly one matching input state"):
        emit_cpp_program(program, model=None)


@pytest.mark.parametrize("first", ["roles", "provided"])
def test_roe_provider_exclusivity_on_the_same_authoring_source(first):
    from pops._ir.values import StateRef
    model = Model("roe_guards", frame=FRAME)
    state = model.state("U", components=("u",))
    u, = state
    model.flux("F", frame=FRAME, state=state,
               components={X_AXIS: (u,), Y_AXIS: (u,)},
               waves={X_AXIS: (1,), Y_AXIS: (1,)})
    source = model._dsl._m
    rows = {"x": (StateRef("L", u),), "y": (StateRef("R", u),)}
    if first == "roles":
        source.enable_roe()
        with pytest.raises(ValueError, match="one single provider"):
            source.roe_dissipation(**rows)
    else:
        source.roe_dissipation(**rows)
        with pytest.raises(ValueError, match="one single provider"):
            source.enable_roe()


def test_roe_qualified_variables_require_explicit_sides_and_bind_only_at_emission():
    from pops._ir.values import StateRef
    model = Model("roe_qualified", frame=FRAME)
    state = model.state("U", components=("u",))
    u, = state
    model.flux("F", frame=FRAME, state=state,
               components={X_AXIS: (u,), Y_AXIS: (u,)},
               waves={X_AXIS: (1,), Y_AXIS: (1,)})
    source = model._dsl._m
    with pytest.raises(ValueError, match="'u' outside marker"):
        source.roe_dissipation(x=(u,), y=(StateRef("L", u),))
    assert source._roe_rows is None
    with pytest.raises(ValueError, match="nested left"):
        source.roe_dissipation(x=(StateRef("L", StateRef("R", u)),), y=(StateRef("L", u),))
    source.roe_dissipation(x=(StateRef("R", u) - StateRef("L", u),),
                           y=(StateRef("R", u) - StateRef("L", u),))
    authored = source._roe_rows["x"][0]
    model._dsl._model_hash()
    cpp = source.emit_cpp_brick(name="QualifiedRoe")
    assert "R_u - L_u" in cpp
    assert source._roe_rows["x"][0] is authored
    assert isinstance(authored.a.expr, QuantityRef)


def test_coupled_authored_name_cannot_impersonate_a_private_bound_coordinate():
    from pops._ir.expr import Var
    module, program, _ = _program()
    operator = module.operator_registry().get("collision")
    altered = {name: tuple(values) for name, values in operator.body.items()}
    altered["electrons"] = (Var("pops_input_0_component_0", "cons"), *altered["electrons"][1:])
    operator.body = altered
    with pytest.raises(ValueError, match="collides with a private input binding"):
        emit_cpp_program(program, model=None)
