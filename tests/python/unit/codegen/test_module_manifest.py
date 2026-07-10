"""ADC-585: the Module / Operator manifests that replace ModelSpec as the central representation.

``Module.manifest()`` builds a self-describing, JSON-ready :class:`ModuleManifest`: the spaces,
params, aux, eigenvalue presence, the typed operator registry (each operator by its stable id, in
registration order) and the native route-registry components.  It supersedes the legacy flat
ModelSpec POD, which is now quarantined off the ``pops`` root (``pops.runtime.ModelSpec``).

Pure Python (the model layer is import-free); skips if ``pops`` is not importable.
"""

import copy
import json
from fractions import Fraction

import pytest

pytest.importorskip("pops")

import pops
from pops import model
from pops.ir.expr import Var


def _small_module():
    """A minimal single-state Module with a field, a param, an aux and one field_operator."""
    mod = model.Module("m")
    u = mod.state_space("U", ("rho", "mx", "my"), roles={"rho": "Density"})
    f = mod.field_space("fields", ("phi", "grad_x", "grad_y"))
    mod.parameters(alpha=1.0)
    mod.aux_fields(B_z="cell_scalar")
    mod.operator(
        name="fields_from_state", signature=(u,) >> f, kind="field_operator", expr="POISSON"
    )
    return mod


def _two_fluid_module():
    """(e, i) -> RateBundle{electrons: Rate(e), ions: Rate(i)} coupled_rate collision operator."""
    mod = model.Module("two_fluid")
    e = mod.state_space("electron_state", ("ne", "mex", "mey"))
    i = mod.state_space("ion_state", ("ni", "mix", "miy"))
    bundle = model.RateBundle({"electrons": model.Rate(e), "ions": model.Rate(i)})
    mod.operator(
        name="collision",
        signature=model.Signature((e, i), bundle),
        kind="coupled_rate",
        expr={
            "electrons": [Var("ni", "cons") - Var("ne", "cons")],
            "ions": [Var("ne", "cons") - Var("ni", "cons")],
        },
    )
    return mod


def test_manifest_schema_and_spaces():
    module = _small_module()
    manifest = module.manifest()
    assert manifest.schema_version == model.manifest.SCHEMA_VERSION == 4
    assert manifest.name == "m"
    assert manifest.to_dict()["owner_path"] == module.owner_path.canonical().to_data()
    assert manifest.state_spaces["U"]["components"] == ("rho", "mx", "my")
    assert manifest.state_spaces["U"]["roles"] == {"rho": "Density"}
    assert manifest.field_spaces["fields"]["components"] == ("phi", "grad_x", "grad_y")
    assert manifest.params["alpha"]["default"] == {
        "kind": "binary64",
        "value": (1.0).hex(),
        "target": "real",
    }
    assert manifest.aux["B_z"]["aux_kind"] == "cell_scalar"
    assert manifest.has_eigenvalues == {"x": False, "y": False}

    data = manifest.to_dict()
    declarations = (
        (data["state_spaces"], "U", "state"),
        (data["field_spaces"], "fields", "field"),
        (data["params"], "alpha", "parameter"),
        (data["aux"], "B_z", "aux"),
    )
    for family, name, kind in declarations:
        row = family[name]
        handle = model.Handle.from_canonical_identity(row["handle"])
        assert handle.owner_path == module.owner_path.canonical()
        assert handle.local_id == name
        assert handle.kind == kind
        assert row["qid"] == handle.qualified_id
    print("OK  manifest carries schema_version, spaces, params, aux")


def test_manifest_does_not_mutate_module():
    mod = _small_module()
    before = mod.module_hash()
    mod.manifest()
    assert mod.module_hash() == before
    print("OK  build_module_manifest is read-only (module hash unchanged)")


def test_operator_entries_in_registration_order_with_ids():
    mod = model.Module("m")
    u = mod.state_space("U", ("rho",))
    f = mod.field_space("fields", ("phi",))
    mod.operator(name="op_a", signature=(u, f) >> model.Rate(u), kind="local_rate", expr="A")
    mod.operator(name="op_b", signature=(u,) >> f, kind="field_operator", expr="B")
    entries = list(mod.manifest().operators)
    assert [e.name for e in entries] == ["op_a", "op_b"]
    assert [e.id for e in entries] == [0, 1]
    assert entries[0].kind == "local_rate" and entries[1].kind == "field_operator"
    assert entries[0].inputs == ("U", "fields")
    first_handle = model.Handle.from_canonical_identity(entries[0].to_dict()["handle"])
    assert isinstance(first_handle, model.OperatorHandle)
    assert entries[0].qid == first_handle.qualified_id
    assert first_handle.registered_operator_name == "op_a"
    assert entries[0].to_dict()["signature"] == {
        "inputs": [u.to_data(), f.to_data()],
        "output": model.Rate(u).to_data(),
    }
    print("OK  operator entries are in registration order with stable ids")


def test_hash_is_stable_and_operator_addition_changes_it():
    m1 = _small_module()
    identical = _small_module()
    assert m1.owner_path != identical.owner_path
    assert m1.owner_path.canonical() == identical.owner_path.canonical()
    assert m1.manifest().to_dict() == identical.manifest().to_dict()
    assert m1.manifest().hash == identical.manifest().hash
    assert m1.manifest().hash == m1.manifest().hash
    m2 = _small_module()
    u = m2.state_spaces()["U"]
    m2.operator(name="extra_rate", signature=(u,) >> model.Rate(u), kind="local_rate", expr="R")
    assert m1.manifest().hash != m2.manifest().hash
    # the operator-registry manifest hash is likewise sensitive
    assert m1.manifest().operators.hash != m2.manifest().operators.hash
    print("OK  manifest hash is stable and an added operator changes it")


def test_to_json_round_trips_through_json_loads():
    manifest = _small_module().manifest()
    blob = manifest.to_json()
    restored = json.loads(blob)
    assert restored["schema_version"] == 4
    assert restored["name"] == "m"
    assert restored["operators"][0]["name"] == "fields_from_state"
    assert restored == manifest.to_dict()
    rebuilt = model.ModuleManifest.from_json(blob)
    assert rebuilt.to_dict() == manifest.to_dict()
    assert rebuilt.hash == manifest.hash
    print("OK  to_json round-trips through json.loads")


def test_strict_json_round_trip_rejects_schema_and_identity_tampering():
    data = _small_module().manifest().to_dict()

    extra = copy.deepcopy(data)
    extra["legacy_owner"] = "m"
    with pytest.raises(TypeError, match="exactly"):
        model.ModuleManifest.from_dict(extra)

    wrong_schema = copy.deepcopy(data)
    wrong_schema["schema_version"] = 3
    with pytest.raises(ValueError, match="schema_version"):
        model.ModuleManifest.from_dict(wrong_schema)

    wrong_name = copy.deepcopy(data)
    wrong_name["name"] = "other"
    with pytest.raises(ValueError, match="does not match owner_path"):
        model.ModuleManifest.from_dict(wrong_name)

    forged_qid = copy.deepcopy(data)
    forged_qid["state_spaces"]["U"]["qid"] = "forged"
    with pytest.raises(ValueError, match="qid"):
        model.ModuleManifest.from_dict(forged_qid)

    with pytest.raises(ValueError, match="duplicate object key"):
        model.ModuleManifest.from_json('{"schema_version":4,"schema_version":4}')
    with pytest.raises(ValueError, match="non-finite"):
        model.ModuleManifest.from_json('{"schema_version":NaN}')


def test_alias_manifest_carries_authenticated_alias_and_target_identities():
    module = _small_module()
    module.operator_registry().register_alias("fields", "fields_from_state")
    manifest = module.manifest()
    row = manifest.to_dict()["operator_aliases"]["fields"]

    assert row["name"] == "fields"
    assert row["target"] == "fields_from_state"
    alias = model.Handle.from_canonical_identity(row["handle"])
    target = model.Handle.from_canonical_identity(row["target_handle"])
    assert isinstance(alias, model.OperatorHandle)
    assert isinstance(target, model.OperatorHandle)
    assert alias.local_id == "fields"
    assert alias.registered_operator_name == "fields_from_state"
    assert target.local_id == target.registered_operator_name == "fields_from_state"
    assert row["qid"] == alias.qualified_id
    assert row["target_qid"] == target.qualified_id

    rebuilt = model.ModuleManifest.from_dict(manifest.to_dict())
    assert rebuilt.operators.aliases() == manifest.operators.aliases()

    forged = manifest.to_dict()
    forged["operator_aliases"]["fields"]["target"] = "forged"
    with pytest.raises(ValueError, match="unknown operator"):
        model.ModuleManifest.from_dict(forged)


def test_manifest_preserves_an_exact_rational_parameter_default():
    mod = model.Module("exact")
    mod.param("third", Fraction(1, 3), dtype="real64")

    assert mod.manifest().params["third"]["default"] == {
        "kind": "rational",
        "numerator": "1",
        "denominator": "3",
        "target": "real64",
    }


def test_native_routes_and_abi_slot_present():
    manifest = _small_module().manifest()
    routes = manifest.native_routes
    assert set(routes) == {"version", "hash", "signature"}
    from pops.runtime.routes import route_registry_hash, route_registry_signature

    assert routes["hash"] == route_registry_hash()
    assert routes["signature"] == route_registry_signature()
    # abi_key is a compile-time fact: the Module manifest leaves the slot open for the compile seam.
    assert manifest.abi_requirements["abi_key"] is None
    assert manifest.abi_requirements["route_registry_signature"] == route_registry_signature()
    print("OK  native route components + ABI-requirements slot present")


def test_coupled_rate_multi_output_names_bundle_without_new_flat_field():
    manifest = _two_fluid_module().manifest()
    entry = manifest.operators.describe("collision")
    assert entry.kind == "coupled_rate"
    # inputs are the participating state spaces; the output is the RateBundle, named by its blocks
    assert entry.inputs == ("electron_state", "ion_state")
    assert entry.output.startswith("RateBundle{")
    assert "electrons" in entry.output and "ions" in entry.output
    # no new flat per-block field on the entry: the manifest row exposes only the frozen slots
    row = entry.to_dict()
    assert set(row) == {
        "id",
        "name",
        "kind",
        "qid",
        "handle",
        "signature",
        "inputs",
        "output",
        "capabilities",
        "requirements",
        "lowering_route",
    }
    print("OK  a coupled_rate names its RateBundle output with no new flat field")


def test_describe_errors_cite_operator_and_registry_contents():
    manifest = _two_fluid_module().manifest()
    with pytest.raises(KeyError) as excinfo:
        manifest.operators.describe("radiation")
    message = str(excinfo.value)
    assert "radiation" in message  # the requested operator
    assert "collision" in message  # the registry contents, not a historical tag
    print("OK  describe() on an unknown operator cites operator + registry contents")


def test_modelspec_is_quarantined_off_the_pops_root():
    assert not hasattr(pops, "ModelSpec")
    from pops.runtime import ModelSpec  # the legacy native-bridge POD's quarantined home

    assert ModelSpec is not None
    print("OK  pops.ModelSpec removed; pops.runtime.ModelSpec available")


def main():
    test_manifest_schema_and_spaces()
    test_manifest_does_not_mutate_module()
    test_operator_entries_in_registration_order_with_ids()
    test_hash_is_stable_and_operator_addition_changes_it()
    test_to_json_round_trips_through_json_loads()
    test_strict_json_round_trip_rejects_schema_and_identity_tampering()
    test_alias_manifest_carries_authenticated_alias_and_target_identities()
    test_native_routes_and_abi_slot_present()
    test_coupled_rate_multi_output_names_bundle_without_new_flat_field()
    test_describe_errors_cite_operator_and_registry_contents()
    test_modelspec_is_quarantined_off_the_pops_root()
    print("OK  test_module_manifest")


if __name__ == "__main__":
    main()
