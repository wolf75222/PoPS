"""ADC-652 model-side contracts: signatures, declarations, handles and manifests."""
from __future__ import annotations

import json

import pytest

from pops import model
from pops.params import ConstParam, RuntimeParam
from pops.provenance import ProvenanceRecord, SourceSpan


def _operator_source(owner):
    return ProvenanceRecord(
        primary=SourceSpan(__file__, 1),
        owner=owner,
        authoring_api="tests.model.Operator",
    )


@pytest.mark.parametrize("bad", [object(), True, ""])
def test_authoring_identities_refuse_implicit_stringification(bad):
    with pytest.raises(ValueError):
        model.Module(bad)
    state = model.StateSpace("U", ("rho",))
    signature = (state,) >> model.Rate(state)
    with pytest.raises(ValueError):
        model.Operator(bad, "local_rate", signature)
    with pytest.raises(ValueError):
        model.Operator("rate", bad, signature)
    with pytest.raises((TypeError, ValueError)):
        RuntimeParam(bad)
    with pytest.raises((TypeError, ValueError)):
        RuntimeParam("alpha", dtype=bad)
    with pytest.raises(ValueError):
        model.AuxSpace(bad)
    with pytest.raises(ValueError):
        model.AuxSpace("mask", kind=bad)


@pytest.mark.parametrize("bad_id", [object(), True, "", -1])
def test_manifest_constructors_do_not_coerce_identity_fields(bad_id):
    state = model.StateSpace("U", ("rho",))
    owner = model.OwnerPath.model("manifest-constructor")
    operator = model.Operator(
        "rate", "local_rate", (state,) >> model.Rate(state),
        source=_operator_source(owner))
    handle = model.OperatorHandle(
        "rate", kind="local_rate", owner=owner, signature=operator.signature)
    with pytest.raises(ValueError):
        model.OperatorManifestEntry(operator, bad_id, handle)

    valid = model.Module("valid").manifest()
    with pytest.raises(ValueError):
        model.ModuleManifest(
            name=bad_id, owner_path=model.OwnerPath.model("valid"),
            state_spaces={}, field_spaces={}, params={}, aux={},
            provider_pack={"schema_version": 1, "capacity": None, "entries": []},
            has_eigenvalues={}, operators=valid.operators, capabilities={},
            native_routes={}, native_catalog={}, abi_requirements={})


def test_operator_manifest_id_refuses_numeric_string():
    state = model.StateSpace("U", ("rho",))
    owner = model.OwnerPath.model("manifest-constructor")
    operator = model.Operator(
        "rate", "local_rate", (state,) >> model.Rate(state),
        source=_operator_source(owner))
    handle = model.OperatorHandle(
        "rate", kind="local_rate", owner=owner, signature=operator.signature)
    with pytest.raises(ValueError):
        model.OperatorManifestEntry(operator, "0", handle)


def test_operator_signature_contract_refuses_malformed_output_and_dropped_input():
    module = model.Module("contracts")
    state = module.state_space("U", ("rho",))
    fields = module.field_space("fields", ("phi",))
    parameter_like = module.aux_field("parameter_like")

    with pytest.raises(TypeError, match="output must be Rate"):
        module.operator(
            "bad_output", signature=(state,) >> fields,
            kind="local_rate", expr="bad")
    with pytest.raises(TypeError, match="StateSpace.*FieldSpace"):
        module.operator(
            "dropped_parameter",
            signature=model.Signature((state, parameter_like), model.Rate(state)),
            kind="local_source", expr="bad")

    # Registry repeats validation because Operator is an internal mutable
    # codegen record and may have been modified before registration.
    operator = model.Operator(
        "source", "local_source", (state,) >> model.Rate(state))
    operator.signature = (state,) >> fields
    with pytest.raises(TypeError, match="incompatible signature"):
        model.OperatorRegistry(owner=module.owner_path).register(operator)


def test_signature_extension_is_a_small_structural_protocol():
    class ForeignSpace:
        def __init__(self, name):
            self.name = name

        def __hash__(self):
            return hash((type(self), self.name))

        def __eq__(self, other):
            return isinstance(other, ForeignSpace) and self.name == other.name

        def to_data(self):
            return {"kind": "foreign", "name": self.name, "shape": [2, 3]}

    foreign = ForeignSpace("Q")
    assert model.Signature((foreign,), foreign).to_data() == {
        "inputs": [{"kind": "foreign", "name": "Q", "shape": [2, 3]}],
        "output": {"kind": "foreign", "name": "Q", "shape": [2, 3]},
    }
    with pytest.raises(TypeError, match="descriptor protocol"):
        model.Signature((object(),), foreign)


def test_all_descriptor_redeclarations_are_rejected_even_when_identical():
    module = model.Module("declarations")
    module.state_space("U", ("rho",), layout="face")
    with pytest.raises(ValueError, match="already declared"):
        module.state_space("U", ("rho",), layout="face")
    with pytest.raises(ValueError, match="already declared"):
        module.state_space("U", ("rho", "energy"), layout="face")

    module.field_space("fields", ("phi",))
    with pytest.raises(ValueError, match="already declared"):
        module.field_space("fields", ("phi",))
    with pytest.raises(ValueError, match="already declared"):
        module.field_space("fields", ("grad_phi",))

    module.param(ConstParam("alpha", 1.0))
    with pytest.raises(ValueError, match="already declared"):
        module.param(ConstParam("alpha", 1.0))
    with pytest.raises(ValueError, match="already declared"):
        module.param(ConstParam("alpha", 2.0))

    module.aux_field("mask", "cell_scalar")
    with pytest.raises(ValueError, match="already declared"):
        module.aux_field("mask", "cell_scalar")
    with pytest.raises(ValueError, match="already declared"):
        module.aux_field("mask", "face_vector")


def test_pure_module_declarers_return_canonical_operator_handles():
    module = model.Module("handles")
    state = module.state_space("U", ("rho",))
    fields = module.field_space("fields", ("phi",))

    field_handle = module.operator(
        "fields_from_state", signature=(state,) >> fields,
        kind="field_operator", expr="phi")

    @module.operator(
        "source", signature=(state,) >> model.Rate(state),
        kind="local_source")
    def source(_state):
        return "source"

    rate = module.rate_operator(
        "rate", state_space=module.state_handle(state), sources=[])
    for handle in (field_handle, source, rate):
        assert isinstance(handle, model.OperatorHandle)
        assert module.operator_handle(handle.name) == handle
        assert handle.owner_path == module.operator_registry().owner_path
    assert module.operator_registry().get("source").body.__name__ == "source"


def test_module_family_registries_issue_and_authenticate_all_declaration_handles():
    module = model.Module("all-handles")
    state = module.state_space("U", ("rho",))
    field = module.field_space("fields", ("phi",))
    parameter = module.param(ConstParam("alpha", 1.0))
    aux = module.aux_field("mask")

    handles = (
        module.state_handle(state),
        module.field_handle(field),
        module.param_handle(parameter),
        module.aux_handle(aux),
    )
    index = module.declaration_index()
    assert [handle.kind for handle in handles] == ["state", "field", "parameter", "aux"]
    assert all(index.authenticate(handle) is handle for handle in handles)
    assert all(handle.owner_path == module.owner_path for handle in handles)

    foreign = model.Module("all-handles")
    foreign_state = foreign.state_space("U", ("rho",))
    with pytest.raises(ValueError, match="another Module"):
        module.state_handle(foreign_state)


def test_composite_rate_infers_the_field_context_required_by_its_sources():
    module = model.Module("rate-fields")
    state = module.state_space("U", ("rho",))
    fields = module.field_space("fields", ("phi",))
    source = module.operator(
        "electric", signature=(state, fields) >> model.Rate(state),
        kind="local_source", expr="electric")

    rate = module.rate_operator(
        "explicit_rhs",
        state_space=module.state_handle(state),
        flux=False,
        sources=[source],
    )

    assert rate.signature.inputs == (state, fields)
    assert module.operator_registry().get("explicit_rhs").capabilities["requires_fields"] is True


def test_rate_retains_physical_flux_identity_when_routed_as_native_default():
    module = model.Module("default-rate-flux")
    state = module.state_space("U", ("rho",))
    signature = (state,) >> model.Rate(state)
    transport = module.operator(
        "transport", signature=signature, kind="grid_operator", expr="transport")

    rate = module.rate_operator(
        "advance",
        state_space=module.state_handle(state),
        fluxes=(transport,),
        default_flux=transport,
    )

    lowering = module.operator_registry().get(rate.name).lowering
    assert lowering["fluxes"] == ["transport"]
    assert lowering["default_flux"] == "transport"

    other = module.operator(
        "other", signature=signature, kind="grid_operator", expr="other")
    with pytest.raises(ValueError, match="sole exact operator"):
        module.rate_operator(
            "ambiguous",
            state_space=module.state_handle(state),
            fluxes=(transport, other),
            default_flux=transport,
        )


def test_module_composite_rate_rejects_string_and_foreign_references():
    module = model.Module("typed-rate")
    state = module.state_space("U", ("rho",))
    source = module.operator(
        "source", signature=(state,) >> model.Rate(state),
        kind="local_source", expr="source")

    with pytest.raises(TypeError, match="name string is not a semantic reference"):
        module.state_handle("U")
    with pytest.raises(TypeError, match="OperatorHandle"):
        module.rate_operator("bad", state_space=state, sources=["source"])

    foreign = model.Module("typed-rate")
    foreign_state = foreign.state_space("U", ("rho",))
    foreign_source = foreign.operator(
        "source", signature=(foreign_state,) >> model.Rate(foreign_state),
        kind="local_source", expr="source")
    with pytest.raises((ValueError, model.MissingOwnershipError), match="another Module|owned by"):
        module.rate_operator(
            "foreign", state_space=state, sources=[foreign_source])

    rate = module.rate_operator("good", state_space=state, sources=[source])
    assert rate.registered_operator_name == "good"


def test_module_and_registry_owner_anchors_are_read_only():
    module = model.Module("owner")
    registry = module.operator_registry()
    original = module.owner_path
    assert registry.owner_path == original
    with pytest.raises(AttributeError):
        module.owner_path = model.OwnerPath.model("other")
    with pytest.raises(AttributeError):
        registry.owner_path = model.OwnerPath.model("other")
    assert module.owner_path == registry.owner_path == original

    foreign = model.OperatorRegistry(
        owner=model.OwnerPath.fresh(model.OwnerKind.MODEL_DEFINITION, "foreign"))
    with pytest.raises(ValueError, match="another Module"):
        module.adopt_registry(foreign)


def test_manifest_is_structured_deeply_frozen_json_and_copy_out():
    module = model.Module("manifest")
    state = module.state_space(
        "U", ("rho",), roles={"rho": {"physical": "density", "aliases": ["n"]}})
    fields = module.field_space("fields", ("phi",))
    module.operator(
        "fields_from_state", signature=(state,) >> fields,
        kind="field_operator", capabilities={"routes": [{"name": "poisson"}]}, expr="phi")

    manifest = module.manifest()
    entry = manifest.operators.describe("fields_from_state")
    assert manifest.schema_version == 6
    assert entry.to_dict()["signature"] == model.Signature((state,), fields).to_data()
    assert json.loads(manifest.to_json()) == manifest.to_dict()

    original_hash = manifest.hash
    with pytest.raises(TypeError):
        manifest.state_spaces["U"]["roles"]["rho"]["physical"] = "mass"
    with pytest.raises(TypeError):
        entry.capabilities["routes"][0]["name"] = "other"
    with pytest.raises(AttributeError):
        manifest.name = "changed"
    for value, attribute in (
        (entry, "kind"),
        (manifest.operators, "_entries"),
        (manifest, "name"),
    ):
        with pytest.raises(AttributeError, match="immutable"):
            delattr(value, attribute)

    detached = manifest.to_dict()
    detached["state_spaces"]["U"]["roles"]["rho"]["physical"] = "mass"
    detached["operators"][0]["capabilities"]["routes"][0]["name"] = "other"
    assert manifest.hash == original_hash
    assert manifest.state_spaces["U"]["roles"]["rho"]["physical"] == "density"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_manifest_refuses_non_finite_float_at_construction(bad):
    module = model.Module("finite-manifest")
    state = module.state_space("U", ("rho",))
    module.operator(
        "source", signature=(state,) >> model.Rate(state),
        kind="local_source", capabilities={"enabled": True, "weight": bad}, expr="source")

    with pytest.raises(ValueError, match="non-finite float"):
        module.manifest()


def test_manifest_preserves_boolean_metadata_and_exposes_alias_identity():
    plain = model.Module("aliases")
    plain_state = plain.state_space("U", ("rho",))
    plain.operator(
        "source", signature=(plain_state,) >> model.Rate(plain_state),
        kind="local_source", capabilities={"enabled": True}, expr="source")
    plain_manifest = plain.manifest()

    aliased = model.Module("aliases")
    state = aliased.state_space("U", ("rho",))
    aliased.operator(
        "source", signature=(state,) >> model.Rate(state),
        kind="local_source", capabilities={"enabled": True}, expr="source")
    aliased.operator_registry().register_alias("readable", "source")
    manifest = aliased.manifest()

    assert manifest.operators.describe("source").capabilities["enabled"] is True
    alias_row = manifest.to_dict()["operator_aliases"]["readable"]
    assert alias_row["target"] == "source"
    assert alias_row["handle"]["registered_operator_name"] == "source"
    assert alias_row["target_handle"]["registered_operator_name"] == "source"
    assert manifest.hash != plain_manifest.hash
    aliases = manifest.to_dict()["operator_aliases"]
    aliases["readable"]["target"] = "forged"
    assert manifest.to_dict()["operator_aliases"]["readable"]["target"] == "source"


def test_manifest_abi_binding_is_functional_and_rate_inherits_base_layout():
    module = model.Module("layout")
    state = module.state_space("U", ("rho", "flux"), layout="face")
    rate = model.Rate(state)
    assert rate.layout == "face"
    assert rate.components == state.components
    module.operator(
        "shape_rate", signature=(state,) >> rate,
        kind="local_rate", expr="shape")
    from pops.numerics.terms import Flux
    from pops.problem import Case
    from pops.time import Program

    block = Case(name="shape-case").block("fluid", module)
    program = Program("rate_shape")._bind_operators(module)
    value = program.state(block[module.state_handle(state)]).n
    rate_value = program.rhs(state=value, terms=[Flux()])
    assert rate_value.logical_shape == {
        "vtype": "rhs", "space": "Rate(U)", "n_comp": 2, "layout": "face"}

    manifest = module.manifest()
    bound = manifest.with_abi_key("abi-v1")
    assert manifest.abi_requirements["abi_key"] is None
    assert bound.abi_requirements["abi_key"] == "abi-v1"
    assert bound.hash != manifest.hash

    from pops.codegen.loader import CompiledProblem

    graph = program.to_graph()
    compiled = CompiledProblem(
        "problem.so", program=program, model=module, abi_key="abi-v2",
        cxx="c++", std="c++17", module_manifest=manifest, program_graph=graph)
    assert manifest.abi_requirements["abi_key"] is None
    assert compiled.module_manifest is not manifest
    assert compiled.module_manifest.abi_requirements["abi_key"] == "abi-v2"
    assert compiled.program_graph.graph_hash == graph.graph_hash
