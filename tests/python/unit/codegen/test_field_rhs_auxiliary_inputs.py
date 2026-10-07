"""Field RHS mathematical inputs retain separate State and qualified Auxiliary values."""
import pytest

import pops
from pops.analytic import x
from pops.codegen.component_provider_packs import resolve_component_provider_packs
from pops.codegen.module_lowering import lower_and_validate
from pops.domain import Rectangle
from pops.fields import AnalyticAux, DerivedAux, FieldOutput
from pops.frames import Cartesian2D
from pops.math import Reaction, ValueExpr, laplacian
from pops.model import Module, Signature
from pops.params import RuntimeParam


def _field_model(*, name="aux_field_rhs", auxiliary=True, primitive=False):
    frame = Rectangle("field_rhs_box", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model(name, frame=frame)
    state = model.state("density", components=("rho",))
    coefficient = model.auxiliary("imposed", frame=frame.canonical_id) if auxiliary else 2
    if primitive:
        coefficient = model.primitive("coefficient", 3 * coefficient)
    parameter = model.param(RuntimeParam("strength", default=1.5))
    potential = model.field("potential")
    operator = model.field_operator("recover", unknown=potential,
        equation=-laplacian(potential) + Reaction(potential, 1)
        == model.value(parameter) * coefficient * state[0],
        outputs=(FieldOutput("observed", potential),))
    module = model.module
    if auxiliary:
        module.aux_provider(AnalyticAux(module.aux_handle(module.aux()["imposed"]),
                                      2 + x(frame), frame=frame))
    return model, state, operator


@pytest.mark.parametrize("primitive", [False, True])
def test_named_field_rhs_declares_exact_aux_and_reads_its_compact_slot(primitive):
    model, state, operator = _field_model(primitive=primitive)
    definition = model.module.operator_registry().get(operator.name)
    assert definition.requirements["aux"] == ["imposed"]
    assert definition.signature.inputs == (state.space,)
    packs = resolve_component_provider_packs(model.module)
    assert [(key.space_kind, key.space_name, key.component)
            for key in packs.by_operator[operator.name]] == [("aux", "imposed", "imposed")]
    lowered, _ = lower_and_validate(model, facade=model)
    implementation = getattr(lowered, "_m", lowered)
    from pops.codegen.module_codegen import emit_cpp_elliptic_field

    source = emit_cpp_elliptic_field(implementation, operator.name, "FieldDensity")
    assert "using State = pops::StateVec<1>" in source
    assert "elliptic_rhs(const State& U, const pops::ProviderValues<1>& a)" in source
    assert "pops::provider_value<0>(a)" in source
    assert "params.get(" in source
    assert "prepared_field_rhs_input_contract_version = 2" in source


def test_provider_free_field_rhs_keeps_physical_state_only_formula():
    model, state, operator = _field_model(auxiliary=False)
    assert "aux" not in model.module.operator_registry().get(operator.name).requirements
    assert not len(resolve_component_provider_packs(model.module).by_operator[operator.name])
    lowered, _ = lower_and_validate(model, facade=model)
    implementation = getattr(lowered, "_m", lowered)
    from pops.codegen.module_codegen import emit_cpp_elliptic_field

    source = emit_cpp_elliptic_field(implementation, operator.name, "FieldDensity")
    assert "using State = pops::StateVec<1>" in source
    assert "elliptic_rhs(const State& U)" in source
    assert "pops::provider_value<" not in source
    assert "n_providers = 0" in source


@pytest.mark.parametrize("transitive", [False, True])
def test_derived_field_rhs_requires_a_real_field_context_at_resolve(transitive):
    module = Module("hidden_field_rhs")
    state = module.state_space("S", ("s",))
    fields = module.field_space("previous_solution", ("f",))
    output = module.field_space("next_solution", ("g",))
    imposed = module.aux_field("imposed")
    target = module.aux_handle(imposed)
    dependency = module.field_handle(fields)
    if transitive:
        intermediate = module.aux_handle(module.aux_field("intermediate"))
        module.aux_provider(DerivedAux(intermediate, ValueExpr(dependency)))
        dependency = intermediate
    module.aux_provider(DerivedAux(target, ValueExpr(dependency)))
    module.operator("recover", kind="field_operator", signature=Signature((state,), output),
                    requirements={"aux": ["imposed"]}, expr=0.)
    with pytest.raises(ValueError, match="requires a FieldContext input"):
        resolve_component_provider_packs(module)


def test_v2_read_effect_authenticates_aux_and_parameters_without_changing_v1():
    from pops.codegen.state_read_extent import cell_state_read_extent

    model, _, operator = _field_model(primitive=True)
    lowered, _ = lower_and_validate(model, facade=model)
    implementation = getattr(lowered, "_m", lowered)
    rhs = implementation._elliptic_fields[operator.name]["rhs"]
    pack = implementation._component_operator_provider_packs[operator.name]
    assert cell_state_read_extent(implementation, rhs) is None  # v1 stays conservative.
    effect = cell_state_read_extent(implementation, rhs, provider_pack=pack)
    assert effect["schema"] == 2 and effect["provider_count"] == 1
    assert effect["cells"] == (0, 0)


def test_v2_read_effect_keeps_opaque_and_foreign_parameter_reads_unknown():
    from pops._ir.expr import Expr
    from pops.codegen.state_read_extent import cell_state_read_extent

    class OpaqueScalar(Expr):
        def deps(self):
            return set()

    model, _, operator = _field_model()
    lowered, _ = lower_and_validate(model, facade=model)
    implementation = getattr(lowered, "_m", lowered)
    pack = implementation._component_operator_provider_packs[operator.name]
    assert cell_state_read_extent(implementation, OpaqueScalar(), provider_pack=pack) is None
    other = pops.Model("foreign_parameter")
    handle = other.param(RuntimeParam("strength", default=1.5))
    assert cell_state_read_extent(implementation, other.value(handle), provider_pack=pack) is None
