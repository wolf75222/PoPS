"""ADC-652 model-side contracts: signatures, declarations, handles and manifests."""
from __future__ import annotations

import json

import pytest

from pops import model


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
    with pytest.raises(ValueError):
        model.ParameterSpace(bad)
    with pytest.raises(ValueError):
        model.ParameterSpace("alpha", dtype=bad)
    with pytest.raises(ValueError):
        model.AuxSpace(bad)
    with pytest.raises(ValueError):
        model.AuxSpace("mask", kind=bad)


@pytest.mark.parametrize("bad_id", [object(), True, "", -1])
def test_manifest_constructors_do_not_coerce_identity_fields(bad_id):
    state = model.StateSpace("U", ("rho",))
    operator = model.Operator("rate", "local_rate", (state,) >> model.Rate(state))
    with pytest.raises(ValueError):
        model.OperatorManifestEntry(operator, bad_id)

    valid = model.Module("valid").manifest()
    with pytest.raises(ValueError):
        model.ModuleManifest(
            name=bad_id, state_spaces={}, field_spaces={}, params={}, aux={},
            has_eigenvalues={}, operators=valid.operators, capabilities={},
            native_routes={}, native_catalog={}, abi_requirements={})


def test_operator_manifest_id_refuses_numeric_string():
    state = model.StateSpace("U", ("rho",))
    operator = model.Operator("rate", "local_rate", (state,) >> model.Rate(state))
    with pytest.raises(ValueError):
        model.OperatorManifestEntry(operator, "0")


def test_operator_signature_contract_refuses_malformed_output_and_dropped_input():
    module = model.Module("contracts")
    state = module.state_space("U", ("rho",))
    fields = module.field_space("fields", ("phi",))
    parameter = module.param("alpha", 1.0)

    with pytest.raises(TypeError, match="output must be Rate"):
        module.operator(
            "bad_output", signature=(state,) >> fields,
            kind="local_rate", expr="bad")
    with pytest.raises(TypeError, match="StateSpace.*FieldSpace"):
        module.operator(
            "dropped_parameter", signature=model.Signature((state, parameter), model.Rate(state)),
            kind="local_source", expr="bad")

    # Registry repeats validation because Operator is an internal mutable
    # codegen record and may have been modified before registration.
    operator = model.Operator(
        "source", "local_source", (state,) >> model.Rate(state))
    operator.signature = (state,) >> fields
    with pytest.raises(TypeError, match="incompatible signature"):
        model.OperatorRegistry().register(operator)


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


def test_incompatible_redeclarations_are_rejected_and_compatible_ones_are_idempotent():
    module = model.Module("declarations")
    state = module.state_space("U", ("rho",), layout="face")
    assert module.state_space("U", ("rho",), layout="face") is state
    with pytest.raises(ValueError, match="already declared incompatibly"):
        module.state_space("U", ("rho", "energy"), layout="face")

    fields = module.field_space("fields", ("phi",))
    assert module.field_space("fields", ("phi",)) is fields
    with pytest.raises(ValueError, match="already declared incompatibly"):
        module.field_space("fields", ("grad_phi",))

    param = module.param("alpha", 1.0)
    assert module.param("alpha", 1.0) is param
    with pytest.raises(ValueError, match="already declared incompatibly"):
        module.param("alpha", 2.0)

    aux = module.aux_field("mask", "cell_scalar")
    assert module.aux_field("mask", "cell_scalar") is aux
    with pytest.raises(ValueError, match="already declared incompatibly"):
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

    rate = module.rate_operator("rate", state_space=state, sources=[])
    for handle in (field_handle, source, rate):
        assert isinstance(handle, model.OperatorHandle)
        assert module.operator_handle(handle.name) == handle
        assert handle.owner_path == module.operator_registry().owner_path
    assert module.operator_registry().get("source").body.__name__ == "source"


def test_composite_rate_infers_the_field_context_required_by_its_sources():
    module = model.Module("rate-fields")
    state = module.state_space("U", ("rho",))
    fields = module.field_space("fields", ("phi",))
    module.operator(
        "electric", signature=(state, fields) >> model.Rate(state),
        kind="local_source", expr="electric")

    rate = module.rate_operator(
        "explicit_rhs", state_space=state, flux=False, sources=["electric"])

    assert rate.signature.inputs == (state, fields)
    assert module.operator_registry().get("explicit_rhs").capabilities["requires_fields"] is True


def test_module_and_registry_owner_anchors_are_read_only():
    module = model.Module("owner")
    registry = module.operator_registry()
    original = module.owner_path
    assert registry.owner_path == original
    with pytest.raises(AttributeError):
        module.owner_path = model.OwnerPath("other")
    with pytest.raises(AttributeError):
        registry.owner_path = model.OwnerPath("other")
    assert module.owner_path == registry.owner_path == original

    foreign = model.OperatorRegistry(owner=model.OwnerPath("foreign"))
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
    assert manifest.schema_version == 3
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
    assert manifest.to_dict()["operator_aliases"] == {"readable": "source"}
    assert manifest.hash != plain_manifest.hash
    aliases = manifest.to_dict()["operator_aliases"]
    aliases["readable"] = "forged"
    assert manifest.to_dict()["operator_aliases"] == {"readable": "source"}


def test_manifest_abi_binding_is_functional_and_rate_inherits_base_layout():
    module = model.Module("layout")
    state = module.state_space("U", ("rho", "flux"), layout="face")
    rate = model.Rate(state)
    assert rate.layout == "face"
    assert rate.components == state.components
    from pops.time import Program
    program = Program("rate_shape")
    value = program.state("fluid", space=state)
    rate_value = program._rhs_legacy(state=value, sources=[])
    assert rate_value.logical_shape == {
        "vtype": "rhs", "space": "Rate(U)", "n_comp": 2, "layout": "face"}

    manifest = module.manifest()
    bound = manifest.with_abi_key("abi-v1")
    assert manifest.abi_requirements["abi_key"] is None
    assert bound.abi_requirements["abi_key"] == "abi-v1"
    assert bound.hash != manifest.hash

    from pops.codegen.loader import CompiledProblem

    compiled = CompiledProblem(
        "problem.so", program=object(), model=module, abi_key="abi-v2",
        cxx="c++", std="c++17", module_manifest=manifest)
    assert manifest.abi_requirements["abi_key"] is None
    assert compiled.module_manifest is not manifest
    assert compiled.module_manifest.abi_requirements["abi_key"] == "abi-v2"
