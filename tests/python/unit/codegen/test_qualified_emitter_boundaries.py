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


@pytest.mark.parametrize("formula_kind", ["bare", "qualified", "mixed"])
def test_physical_component_resembling_private_coordinate_binds_its_exact_input(formula_kind):
    from pops import model as types, time
    from pops._ir.expr import Var
    from pops.solvers.nonlinear import LocalNewton
    from typed_program_support import typed_state
    module = Module("private_looking_physical_names")
    left = module.state_space("left", ("pops_input_1_component_0",))
    right = module.state_space("right", ("rho",))
    qualified_left, = module.state_symbols(left)
    qualified_right, = module.state_symbols(right)
    bare_left = Var(left.components[0], "cons")
    bare_right = Var(right.components[0], "cons")
    if formula_kind == "bare":
        first, second = bare_left, bare_right
        first_symbol, second_symbol = left.components[0], right.components[0]
    elif formula_kind == "qualified":
        first, second = qualified_left, qualified_right
        first_symbol, second_symbol = "pops_input_0_component_0", "pops_input_1_component_0_"
    else:
        first, second = bare_left, qualified_right
        first_symbol, second_symbol = left.components[0], "pops_input_1_component_0_"
    exchange = module.operator(
        "exchange", signature=types.Signature((left, right), types.RateBundle({
            "left": types.Rate(left), "right": types.Rate(right)})),
        kind="coupled_rate", expr={"left": (first + second,), "right": (first - second,)})
    program = time.Program("private_looking_input")
    left_n = typed_state(program, "left", space=left, model=module, state=module.state_handle(left))
    right_n = typed_state(program, "right", space=right, model=module, state=module.state_handle(right))
    solved = program.solve(time.CoupledImplicitEuler(exchange, (left_n, right_n)),
                           solver=LocalNewton()).consume(action=time.RejectAttempt())
    left_next = typed_state(program, "left", state_name=left.name, space=left, model=module,
                            state=module.state_handle(left)).next
    right_next = typed_state(program, "right", state_name=right.name, space=right, model=module,
                             state=module.state_handle(right)).next
    program.commit_many({left_next: solved[left_n.block], right_next: solved[right_n.block]})
    cpp = emit_cpp_program(program, model=None)
    assert "const pops::Real %s = Ueval[0];" % first_symbol in cpp
    assert "const pops::Real %s = Ueval[1];" % second_symbol in cpp
    assert "const pops::Real %s = Ueval[1];" % first_symbol not in cpp
    assert "const pops::Real %s = Ueval[0];" % second_symbol not in cpp
    # Distinct sentinel values expose the selected source coordinates for every
    # authoring spelling, using the same lookup that emits the kernel locals.
    from pops.codegen.program_emit_model_kernels import _component_sources
    reads = _component_sources({first_symbol, second_symbol},
                               {left_n.block: left_n, right_n.block: right_n},
                               lambda state, index: 7 if state is left_n else 2)
    assert (reads[first_symbol] + reads[second_symbol],
            reads[first_symbol] - reads[second_symbol]) == (9, 5)


def test_private_coordinate_selection_avoids_all_components_and_legacy_aliases():
    from pops.model import StateSpace
    from pops.model.state_symbols import native_input_state_component_symbols, state_component_symbol
    left = StateSpace("left", ("pops_input_1_component_0", "pops_input_1_component_0_"))
    right = StateSpace("right", ("rho",))
    coordinates = native_input_state_component_symbols((left, right))
    assert coordinates[1] == ("pops_input_1_component_0__",)
    occupied = {component for space in (left, right) for component in space.components}
    occupied.update(state_component_symbol(space, component)
                    for space in (left, right) for component in space.components)
    flattened = [symbol for row in coordinates for symbol in row]
    assert not (set(flattened) & occupied)
    assert len(set(flattened)) == len(flattened)
    assert native_input_state_component_symbols((left, right)) == coordinates


def test_unknown_bare_private_looking_variable_is_not_an_input_declaration():
    from pops._ir.expr import Const, Var
    module, program, _ = _program()
    operator = module.operator_registry().get("collision")
    operator.body = {
        "electrons": (Var("pops_input_0_component_0", "cons"), Const(0), Const(0)),
        "ions": (Const(0), Const(0), Const(0)),
    }
    with pytest.raises(NotImplementedError, match="component of no input state"):
        emit_cpp_program(program, model=None)


def test_frozen_multistate_compiler_binding_reuses_exact_source_without_module_cache_mutation():
    from pops.codegen._compiler_lowering import require_compiler_lowering
    from pops.codegen.component_provider_packs import resolve_component_provider_packs
    from pops.codegen.module_lowering import _module_to_model
    from pops.math import ddt, div
    from pops._ir.expr import Const
    model = Model("frozen_multistate", frame=FRAME)
    left = model.species("left", state=("u",))
    right = model.species("right", state=("v",))
    for name, state in (("Fleft", left), ("Fright", right)):
        value, = state
        flux = model.flux(name, frame=FRAME, state=state,
                   components={X_AXIS: (value,), Y_AXIS: (value,)},
                   waves={X_AXIS: (Const(1),), Y_AXIS: (Const(1),)})
        model.rate("rhs_" + name, equation=ddt(state) == -div(flux))
    source = model.module
    model.freeze()
    assert model._dsl._module_cache is None
    source_manifest = source.manifest().to_dict()
    source_flux = model._dsl._m._flux
    lowering = require_compiler_lowering(model)
    assert lowering.source_module is source
    packs = resolve_component_provider_packs(source)
    lowering.bind_component_provider_packs(packs)
    lowering.bind_component_provider_packs(packs)
    assert model._dsl._module_cache is None
    assert model._dsl._m._formula_source_module is source
    assert model._dsl._m._flux is source_flux
    require_emitter_provider_carrier(model._dsl)
    require_emitter_provider_carrier(model._dsl._m)
    for state_name in (left.space.name, right.space.name):
        emitter = _module_to_model(source, state_space=state_name)
        emitter.__pops_bind_component_provider_packs__(packs)
        assert emitter._m._formula_native_bound is True
        first = emitter._m.emit_cpp_brick(name="Frozen_" + state_name)
        second = emitter._m.emit_cpp_brick(name="Frozen_" + state_name)
        assert first == second
        assert emitter._m._formula_source_module is source
    assert source.manifest().to_dict() == source_manifest
    assert model._dsl._module_cache is None
    assert model._dsl._m._flux is source_flux
    with pytest.raises(ValueError, match="Module owner differs"):
        model._dsl.__pops_retain_compiler_source__(Module("foreign_source"))


def test_recovery_qualified_predicate_is_retained_and_only_privately_bound():
    from pops.codegen.module_lowering import lower_and_validate
    model = Model("qualified_recovery", frame=FRAME)
    state = model.state("population", components=("q",))
    q, = state
    model.flux("F", frame=FRAME, state=state,
               components={X_AXIS: (q,), Y_AXIS: (q,)},
               waves={X_AXIS: (1,), Y_AXIS: (1,)})
    predicate = q > 0
    model.recovery_admissibility(q=predicate)
    source_module = model.module
    model.freeze()
    emitter, source = lower_and_validate(model, facade=model)
    generated = emitter._m.emit_cpp_brick(name="QualifiedRecovery")
    assert "const pops::Real q = P[0];" in generated
    assert "if (!((q > " in generated
    assert emitter._m._recovery_admissibility["q"].eval({"q": 1})
    assert not emitter._m._recovery_admissibility["q"].eval({"q": -1})
    assert model._dsl._m._recovery_admissibility["q"] is predicate
    assert predicate.a is q
    assert model._primitive_state_values[0] is q
    assert emitter._m._recovery_admissibility["q"].a.deps() == {"q"}
    assert source is source_module is model.module


@pytest.mark.parametrize("foreign_kind", ["owner", "state", "physical_type"])
def test_recovery_rejects_foreign_qualified_coordinate_even_with_same_component(foreign_kind):
    model = Model("recovery_owner", frame=FRAME)
    state = model.state("U", components=("q",))
    foreign = (Module("other") if foreign_kind == "owner" else
               Module(model.name, owner=model.owner_path))
    space = foreign.state_space("other" if foreign_kind == "state" else "U", ("q",))
    q, = foreign.state_symbols(space)
    with pytest.raises(ValueError, match="outside the owned selected primitive state"):
        model.recovery_admissibility(q=q > 0)
    assert model._dsl._m._recovery_admissibility == {}
    owned, = state
    model.recovery_admissibility(q=owned > 0)


def test_recovery_derived_primitive_homonym_does_not_authorize_excluded_state_coordinate():
    model = Model("recovery_homonym", frame=FRAME)
    state = model.state("U", components=("q",))
    conservative_q, = state
    primitive_q = model.primitive("q", 2 * conservative_q)
    model.primitive_state(primitive_q, conservative=(primitive_q / 2,))
    assert model._primitive_state_values[0] is primitive_q
    with pytest.raises(ValueError, match="outside the owned selected primitive state"):
        model.recovery_admissibility(q=conservative_q > 0)
    assert model._dsl._m._recovery_admissibility == {}
    model.recovery_admissibility(q=primitive_q > 0)


def test_recovery_primitive_coordinate_selection_rolls_back_with_failed_declaration(monkeypatch):
    model = Model("recovery_atomic", frame=FRAME)
    state = model.state("U", components=("q",))
    q, = state
    p = model.primitive("p", 2 * q)
    original = model._primitive_state_values
    with monkeypatch.context() as patch:
        def fail_inverse(expressions):
            model._dsl._m.cons_from = list(expressions)
            raise RuntimeError("injected inverse failure")
        patch.setattr(model._dsl, "conservative_from", fail_inverse)
        with pytest.raises(RuntimeError, match="injected inverse failure"):
            model.primitive_state(p, conservative=(p / 2,))
    assert len(model._primitive_state_values) == len(original)
    assert all(value is previous for value, previous in
               zip(model._primitive_state_values, original, strict=True))
    assert model._dsl._m.prim_state == ["q"]
    model.primitive_state(p, conservative=(p / 2,))
    with pytest.raises(ValueError, match="outside the owned selected primitive state"):
        model.recovery_admissibility(p=q > 0)
    model.recovery_admissibility(p=p > 0)


def _provider_refinement_model():
    from pops.fields import FieldOutput, GradientOutput
    from pops.math import ddt, div, laplacian
    model = Model("provider_refinement", frame=FRAME)
    state = model.state("U", components=("u",))
    u, = state
    flux = model.flux("F", frame=FRAME, state=state,
                     components={X_AXIS: (u,), Y_AXIS: (u,)},
                     waves={X_AXIS: (1,), Y_AXIS: (1,)})
    field = model.field("fields")
    model.field_operator("fields", unknown=field, equation=-laplacian(field) == u,
                         outputs=(FieldOutput("phi", field), GradientOutput("grad", field)))
    gx = model.aux("grad_x")
    source = model.source("force", on=state, value=(u * gx,))
    model.rate("rhs", equation=ddt(state) == -div(flux) + source)
    return model


def test_private_emitters_bind_two_authenticated_access_plans_without_source_contamination():
    from dataclasses import replace
    from pops.codegen.component_provider_packs import (
        bind_emitter_provider_packs, emitter_carrier_snapshot, resolve_component_provider_packs,
    )
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.resolved_operations import build_resolved_operations

    model = _provider_refinement_model()
    module = model.module
    default = resolve_component_provider_packs(module)
    model._dsl.__pops_retain_compiler_source__(module)
    model._dsl.__pops_bind_component_provider_packs__(default)
    narrow = build_resolved_operations(module)
    source_operation = next(op for op in narrow.operations if op.identity.endswith("::force"))
    rate_operation = next(op for op in narrow.operations if op.identity.endswith("::rhs"))
    source_fields = tuple(read for read in source_operation.inputs if read.kind == "field")
    assert source_fields
    broad = build_resolved_operations(module, constructions=tuple(
        replace(op, inputs=(*op.inputs, *source_fields)) if op is rate_operation else op
        for op in narrow.operations))
    for plan in (narrow, broad):
        for operation in plan.operations:
            plan.require_native(operation.identity, module=module)
    narrow_packs, broad_packs = (plan.require_provider_packs(module) for plan in (narrow, broad))
    assert len(default.by_operator["rhs"]) > len(broad_packs.by_operator["rhs"]) > 0
    assert len(narrow_packs.by_operator["rhs"]) == 0
    assert narrow_packs.complete.to_data() == broad_packs.complete.to_data() == default.complete.to_data()
    assert narrow_packs.physical_flux.to_data() == default.physical_flux.to_data()
    assert narrow_packs.by_operator["force"].to_data() == default.by_operator["force"].to_data()
    before_facade = emitter_carrier_snapshot(model._dsl)
    before_model = emitter_carrier_snapshot(model._dsl._m)
    # This compiler-only annotation can remain on a reused source facade. A fresh
    # emitter must receive the nominated plan instead of inheriting that old one.
    object.__setattr__(model._dsl, "_resolved_operations", broad)
    first, _ = lower_and_validate(model, resolved_operations=narrow)
    second, _ = lower_and_validate(model, resolved_operations=broad)
    canonical, canonical_source = lower_and_validate(module, resolved_operations=narrow)
    assert canonical_source is module
    assert canonical._m._formula_native_bound
    assert canonical._component_operator_provider_packs["rhs"].to_data() == narrow_packs.by_operator["rhs"].to_data()
    assert first is not second and first._m is not second._m
    assert first._resolved_operations is narrow and second._resolved_operations is broad
    assert model._dsl._resolved_operations is broad
    assert first._component_operator_provider_packs["rhs"].to_data() == narrow_packs.by_operator["rhs"].to_data()
    assert second._component_operator_provider_packs["rhs"].to_data() == broad_packs.by_operator["rhs"].to_data()
    for emitter in (first, second, canonical):
        bind_emitter_provider_packs(emitter)
        emitter._model_hash()
        require_emitter_provider_carrier(emitter)
        require_emitter_provider_carrier(emitter._m)
    assert emitter_carrier_snapshot(model._dsl) == before_facade
    assert emitter_carrier_snapshot(model._dsl._m) == before_model
    with pytest.raises(ValueError, match="conflicting component-provider pack"):
        broad_packs.attach(first)
    # A new view cannot launder a tampered source's binding metadata.
    object.__setattr__(model._dsl, "_component_flux_consumer_plan", ({"forged": True},))
    with pytest.raises(ValueError, match="does not match its exact attachment witness"):
        model.__pops_compiler_lowering__()
