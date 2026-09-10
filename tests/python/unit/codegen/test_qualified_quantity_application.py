"""Qualified capture and applications must survive the real legacy lowering boundary."""
from fractions import Fraction

import pytest

from pops._ir.application import ApplicationContext, ApplicationEvaluation
from pops._ir.quantity import PhysicalDimension, PhysicalSupport, QuantityRef
from pops._ir.visitors import _key
from pops.codegen.module_lowering import _module_to_model
from pops.model.bundles import ProductSpace, RateBundle
from pops.model.module import Module
from pops.model.signatures import Signature
from pops.model.spaces import FieldSpace, Rate, StateSpace
from pops.params import ConstParam, RuntimeParam


def _source(name="same"):
    module = Module(name)
    state = module.state_space("U", ["rho"])
    rho, = module.state_symbols(state)
    source = module.operator("source", signature=state >> Rate(state),
                             kind="local_source", expr=[rho + 1])
    return module, state, rho, source


def test_live_quantity_identity_is_distinct_but_module_structure_is_shareable():
    left, _, a, _ = _source()
    right, _, b, _ = _source()
    assert isinstance(a, QuantityRef)
    assert a.handle != b.handle
    assert _key(a) != _key(b)
    assert left.module_hash() == right.module_hash()
    assert a.declaration_references() == (a.handle,)
    resolved = a.resolve_references(lambda handle: handle._resolved())
    assert resolved.handle.is_resolved
    assert resolved.component == "rho"
    assert resolved.to_data()["kind"] == "quantity_ref"
    with pytest.raises(TypeError, match="binding"):
        a.to_cpp()


def test_qualified_quantities_reach_native_expression_binding_and_refuse_foreign_owner():
    module, state, rho, _ = _source()
    lowered = _module_to_model(module)
    assert lowered._m._source_terms["source"][0].deps() == {"rho"}
    other, _, foreign, _ = _source()
    module.operator("foreign", signature=state >> Rate(state), kind="local_source", expr=[foreign])
    with pytest.raises(ValueError, match="foreign qualified"):
        _module_to_model(module)
    assert other is not module


def test_support_representation_sampling_and_unknown_dimensions_do_not_alias():
    physical = PhysicalSupport((("x", "physical"), ("y", "physical")))
    phase = PhysicalSupport((("x", "physical"), ("v", "velocity")))
    unknown = StateSpace("U", ["rho"], support=physical)
    unitless = StateSpace("U", ["rho"], support=physical, units=[PhysicalDimension()])
    assert unknown != unitless
    assert unknown.units == (None,)
    assert unitless.to_data()["units"][0]["powers"] == []
    assert unknown != StateSpace("U", ["rho"], support=phase)
    assert unknown != StateSpace("U", ["rho"], support=physical, sampling="cell_average")
    assert unknown != StateSpace("U", ["rho"], support=physical, representation="integrated")
    assert Rate(unitless).units == unitless.units
    module = Module("typed")
    target = module.state_space("U", ["rho"], support=physical)
    rhs = module.operator("rhs", signature=target >> Rate(target), kind="local_source", expr=[0])
    foreign = Module("phase")
    phase_state = foreign.state_space("U", ["rho"], support=phase)
    with pytest.raises(TypeError, match="support/representation"):
        module.apply(rhs, foreign.state_symbols(phase_state))


def test_callable_is_captured_once_and_native_lowering_contains_no_callback():
    module = Module("capture")
    state = module.state_space("U", ["rho"])
    calls = []
    @module.operator("source", signature=state >> Rate(state), kind="local_source",
                     specialization={"scale": Fraction(3, 2)})
    def source(u, *, scale):
        calls.append(scale)
        return [scale * u[0]]
    assert calls == [Fraction(3, 2)]
    assert not callable(module._registry.get("source").body)
    assert isinstance(module._registry.get("source").body, tuple)
    _module_to_model(module)
    _module_to_model(module)
    assert calls == [Fraction(3, 2)]
    assert source.signature.output == Rate(state)


def test_joint_projection_shares_one_context_and_preserves_multiplicity():
    module = Module("joint")
    a = module.state_space("a", ["rho"])
    b = module.state_space("b", ["rho", "energy"])
    signature = Signature((a, b), RateBundle({"a": Rate(a), "b": Rate(b)}))
    @module.operator("exchange", signature=signature, kind="coupled_rate")
    def exchange(ua, ub):
        return {"a": [ua[0] - ub[0]], "b": [ub[0] - ua[0], ub[1]]}
    ua, ub = module.state_symbols(a), module.state_symbols(b)
    application = module.apply(exchange, ua, ub, context=ApplicationContext(stage="first"))
    left = application["a"][0]
    right = application["b"][0]
    assert left.application is right.application
    context = ApplicationEvaluation({ua[0].qualified_id: 4,
                                     ub[0].qualified_id: 1, ub[1].qualified_id: 7})
    assert left.eval(context) == 3
    assert right.eval(context) == -3
    assert (left + left).eval(context) == 6
    assert len(context._applications) == 1
    other_stage = module.apply(exchange, ua, ub, context=ApplicationContext(stage="second"))
    assert _key(application) != _key(other_stage)
    with pytest.raises(TypeError, match="truth value"):
        bool(left)


def test_parameter_spelling_does_not_select_eos_but_explicit_constitutive_does():
    module, _, _, _ = _source("non_gas")
    module.param(RuntimeParam("gamma", default=3))
    assert _module_to_model(module)._m.gamma is None
    gas, _, _, _ = _source("gas")
    ratio = gas.param(ConstParam("heat_capacity_ratio", Fraction(7, 5)))
    gas.constitutive(gamma=ratio)
    assert _module_to_model(gas)._m.gamma == Fraction(7, 5)
    assert gas._constitutive["parameter"] == "heat_capacity_ratio"
    assert gas.module_hash() != module.module_hash()


def test_product_output_is_representable_without_claiming_a_native_field_route():
    one = FieldSpace("phi", ["phi"])
    two = FieldSpace("psi", ["psi", "gradient"])
    signature = Signature((), ProductSpace({"one": one, "two": two}))
    assert signature.to_data()["output"]["kind"] == "product_space"
    with pytest.raises(TypeError, match="output must be a FieldSpace"):
        module = Module("joint_fields")
        state = module.state_space("U", ["rho"])
        module.operator("joint", signature=Signature((state,), signature.output),
                        kind="field_operator", expr=[0])


def test_typed_metadata_roundtrips_without_collapsing_unknown_dimensions():
    density = PhysicalDimension((("mass", Fraction(1)), ("length", Fraction(-3))))
    support = PhysicalSupport((("x", "physical"),))
    state = StateSpace("U", ["rho"], units=[density], support=support, sampling="cell_average")
    data = state.to_data()
    restored = StateSpace(**{key: value for key, value in data.items() if key != "kind"})
    assert restored == state
    assert restored.units[0] != PhysicalDimension()


def test_distinct_effectful_applications_are_not_common_subexpressions():
    module, state, rho, _ = _source("fallible")
    operator = module.operator("checked", signature=state >> Rate(state), kind="local_source",
                               expr=[rho], capabilities={"effects": ("may_fail",)})
    first = module.apply(operator, [rho])
    second = module.apply(operator, [rho])
    assert _key(first) != _key(second)
    assert first.effects == ("may_fail",)
    module.operator("consumer", signature=state >> Rate(state), kind="local_source",
                    expr=first["value"])
    with pytest.raises(ValueError, match="effectful application"):
        _module_to_model(module)


def test_callable_truth_test_fails_during_capture_without_partial_registration():
    module = Module("unsafe_python_branch")
    state = module.state_space("U", ["rho"])
    with pytest.raises(TypeError, match="truth value"):
        @module.operator("unsafe", signature=state >> Rate(state), kind="local_source")
        def unsafe(u):
            return [u[0] if u[0] else 0]
    assert module.operator_registry().names() == []


def test_blackboard_formula_view_preserves_typed_source_and_primitive_realization():
    from pops.physics import Model
    from tests.python.support.physics_roles import FRAME, X_AXIS, Y_AXIS
    model = Model("board_qualified", frame=FRAME)
    state = model.state("U", components=["u"])
    u = QuantityRef(state, "u", space=state.space)
    pressure = model.primitive("pressure", u * u)
    model.flux("F", frame=FRAME, state=state,
               components={X_AXIS: [pressure], Y_AXIS: [pressure]},
               waves={X_AXIS: [1], Y_AXIS: [1]})
    source = model._dsl._m.prim_defs["pressure"]
    assert isinstance(source.a, QuantityRef)
    lowering = model.__pops_compiler_lowering__()
    assert lowering.source_module is model.module
    assert lowering.emit_model is not model._dsl
    assert lowering.emit_model.check() is True
    assert lowering.emit_model.eval_flux([2], [], "x") == [4]
    assert model._dsl._m.prim_defs["pressure"] is source
    assert lowering.emit_model._m.prim_defs["pressure"].a.deps() == {"u"}


def test_first_species_alias_is_explicitly_bound_without_erasing_scientific_identity():
    from pops.physics import Model
    from tests.python.support.physics_roles import FRAME, X_AXIS, Y_AXIS
    model = Model("first_species", frame=FRAME)
    state = model.species("electrons", state=["rho"])
    rho = QuantityRef(state, "rho", space=state.space)
    model.flux("F", frame=FRAME, state=state,
               components={X_AXIS: [rho], Y_AXIS: [rho]},
               waves={X_AXIS: [1], Y_AXIS: [1]})
    lowering = model.__pops_compiler_lowering__()
    assert lowering.emit_model.eval_flux([5], [], "x") == [5]
    assert model._dsl._m._flux["x"][0].handle.local_id == "electrons"
    retained_flux = model.module.operator_registry().get("flux_default").body
    assert retained_flux["x"][0] is model._dsl._m._flux["x"][0]
    assert retained_flux["x"][0].handle == state
    with pytest.raises(TypeError):
        retained_flux["x"] = ()


def test_expression_product_capture_keeps_heterogeneous_support_and_shared_projections():
    module = Module("cross_support")
    physical = PhysicalSupport((("x", "physical"),))
    phase = PhysicalSupport((("x", "physical"), ("v", "velocity")))
    density = PhysicalDimension((("mass", 1), ("length", -3)))
    state = module.state_space("U", ["rho"], support=physical, units=[density])
    velocity_samples = module.field_space("f", ["f"], support=phase,
                                          sampling="point_value", units=[None])
    response = module.field_space("response", ["value", "sensitivity"], support=phase,
                                   sampling="point_value", units=[None, PhysicalDimension()])
    output = ProductSpace({"density": state, "response": response})
    calls = []
    @module.operator("joint_response", signature=Signature((state, velocity_samples), output),
                     kind="expression", specialization={"scale": Fraction(2)})
    def joint_response(u, f, *, scale):
        calls.append(scale)
        return {"density": u, "response": (scale * f[0], scale)}
    rho, = module.state_symbols(state)
    f, = module.field_symbols(velocity_samples)
    application = module.apply(joint_response, (rho,), (f,))
    response_value, sensitivity = application["response"]
    density_value, = application["density"]
    context = ApplicationEvaluation({rho.qualified_id: 3, f.qualified_id: 4})
    assert (density_value.eval(context), response_value.eval(context), sensitivity.eval(context)) == (3, 8, 2)
    assert density_value.application is response_value.application is sensitivity.application
    assert len(context._applications) == 1
    assert calls == [Fraction(2)]
    signature_data = joint_response.signature.to_data()
    assert signature_data["output"]["outputs"][1]["space"] == response.to_data()
    assert signature_data["inputs"][0]["support"] != signature_data["inputs"][1]["support"]
    assert signature_data["output"]["outputs"][1]["space"]["units"] == [
        None, PhysicalDimension().to_data()]
    from pops.model import ModuleManifest
    import json
    manifest = module.manifest().to_dict()
    assert ModuleManifest.from_json(json.dumps(manifest)).to_dict() == manifest
    with pytest.raises(TypeError):
        module._registry.get("joint_response").body["density"] = ()
    with pytest.raises(ValueError, match="no codegen lowering"):
        _module_to_model(module)
    assert calls == [Fraction(2)]


def test_expression_projection_composition_checks_its_declared_output_space():
    module = Module("projection_types")
    state = module.state_space("U", ["rho"])
    physical = module.field_space("physical", ["value"], support=PhysicalSupport((("x", "physical"),)))
    phase = module.field_space("phase", ["value"], support=PhysicalSupport((("v", "velocity"),)))
    calls = []
    def author(u):
        calls.append(1)
        return {"physical": u, "phase": u}
    joint = module.operator("joint", signature=Signature((state,), ProductSpace({
        "physical": physical, "phase": phase})), kind="expression", expr=author)
    @module.operator("consume", signature=Signature((physical,), physical), kind="expression")
    def consume(x):
        return x
    rho, = module.state_symbols(state)
    application = module.apply(joint, (rho,))
    composed = module.apply(consume, application["physical"])
    assert composed["value"][0].eval(ApplicationEvaluation({rho.qualified_id: 5})) == 5
    with pytest.raises(TypeError, match="support/representation"):
        module.apply(consume, application["phase"])
    assert calls == [1]
    with pytest.raises(ValueError, match="every ProductSpace output"):
        module.operator("incomplete", signature=joint.signature, kind="expression",
                        expr={"physical": [rho]})
    assert "incomplete" not in module.operator_registry().names()


def test_retained_joint_local_rate_signature_does_not_relax_grid_or_source_routes():
    module = Module("joint_rate_types")
    state = module.state_space("U", ["rho"])
    other = module.state_space("V", ["rho"])
    field = module.field_space("E", ["field"])
    signature = Signature((state, other, field), Rate(state))
    retained = module.operator("joint_rate", signature=signature, kind="local_rate", expr=[0])
    assert retained.signature.inputs == (state, other, field)
    for kind in ("grid_operator", "local_source"):
        with pytest.raises(TypeError, match="incompatible signature"):
            module.operator(kind, signature=signature, kind=kind, expr=[0])
    with pytest.raises(TypeError, match="first StateSpace"):
        module.operator("wrong_target", signature=Signature(signature.inputs, Rate(other)),
                        kind="local_rate", expr=[0])


def test_facade_module_retains_authoritative_bodies_for_every_expression_route():
    from pops.physics._facade import Model
    facade = Model("retained_formulas")
    rho, = facade.conservative_vars("rho")
    facade.flux(x=[rho], y=[rho])
    facade.flux_term("transport", x=[2 * rho], y=[3 * rho])
    facade.source([rho])
    facade.source_term("forcing", [2 * rho])
    facade.linear_source("relaxation", [[-1]])
    facade.local_transform("shift", [rho + 1], valid_if=rho > 0)
    facade.aux("phi")
    facade.elliptic_rhs(rho)
    facade.elliptic_field("secondary", rho + 1, aux=("psi",))
    facade.projection([rho])
    module = facade.module
    registry = module.operator_registry()
    expected = {
        "flux_default": facade._m._flux,
        "transport": facade._m._flux_terms["transport"],
        "source_default": facade._m._source,
        "forcing": facade._m._source_terms["forcing"],
        "relaxation": facade._m._linear_sources["relaxation"],
        "shift": facade._m._local_transforms["shift"],
        "fields_from_state": facade._m._elliptic,
        "secondary": facade._m._elliptic_fields["secondary"]["rhs"],
        "projection": facade._m._proj,
    }
    from collections.abc import Mapping
    def body_key(value):
        if isinstance(value, Mapping):
            return {name: body_key(item) for name, item in value.items()}
        if isinstance(value, (tuple, list)):
            return tuple(body_key(item) for item in value)
        return _key(value)
    for name, body in expected.items():
        assert body_key(registry.get(name).body) == body_key(body), name
    assert registry.get("flux_default").body["x"][0] is facade._m._flux["x"][0]
    with pytest.raises(TypeError):
        registry.get("shift").body["valid_if"] = 1
    assert facade.module is module
    assert facade._m.operator_registry() is registry


def test_facade_module_projection_retains_complete_state_qualification():
    from pops.physics._facade import Model
    facade = Model("qualified_facade")
    facade.conservative_vars("rho")
    metadata = facade._m._state_space_metadata
    metadata.update(support=PhysicalSupport((("x", "physical"),)),
                    sampling="cell_average", value_shape=(), domain="positive",
                    units=[PhysicalDimension()])
    original = facade._m.state_space()
    projected = facade.module.state_spaces()["U"]
    assert projected == original
    assert projected.to_data() == original.to_data()
    assert projected.support is metadata["support"]
    assert projected.units == (PhysicalDimension(),)
