"""Uniform field packages preserve resolved output ownership and RHS bindings."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import pops
from pops.codegen import Production
from pops.codegen._compile_emit import _normalize_native_amr_field_roles
from pops.codegen._orchestration_compile import _resolved_native_amr_field_roles
from pops.codegen.module_lowering import lower_and_validate
from pops.identity import canonical_bytes


@pytest.fixture(scope="module")
def uniform_multiphysics():
    path = Path(__file__).resolve().parents[4] / (
        "examples/final/EXEMPLE_SPEC_FINALE_MULTIPHYSIQUE_CORE.py"
    )
    spec = importlib.util.spec_from_file_location("_uniform_field_roles_example", path)
    assert spec is not None and spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = example
    spec.loader.exec_module(example)
    target = example.build_final_case()
    resolved = pops.resolve(
        target.authoring.case,
        layout=target.layout_plan,
        layout_providers={target.layout_handle: target.layout_provider},
        backend=Production(),
    )
    assert resolved.target == "system"
    assert {block.name for block in resolved.blocks} == {"electrons", "ions"}
    return resolved


def test_uniform_projection_keeps_one_output_owner_and_two_exact_charge_bindings(
    uniform_multiphysics,
) -> None:
    resolved = uniform_multiphysics
    roles = _resolved_native_amr_field_roles(resolved)
    assert set(roles) == {"electrons", "ions"}
    field_plan, = resolved.field_plans.values()
    options = field_plan.native_install_data()
    output = options["output_route"]
    assert output["owner_block"] == "electrons"
    assert output["gradient_sign"] == -1
    assert {key["owner_qid"] for key in output["component_keys"]} == {
        output["owner_identity"]
    }
    output_roles = [role for block_roles in roles.values() for role in block_roles
                    if role["kind"] == "output"]
    assert output_roles == [{
        "kind": "output",
        "field": options["provider_slot"],
        "block": "electrons",
        "output_keys": tuple(output["component_keys"]),
        "gradient_sign": -1,
    }]
    assert [role["kind"] for role in roles["electrons"]] == ["output", "rhs"]
    assert [role["kind"] for role in roles["ions"]] == ["rhs"]

    bindings = []
    for ordinal, provider in enumerate(options["provider_pack"]):
        block = provider["owner_block"]
        rhs, = (role for role in roles[block] if role["kind"] == "rhs")
        assert rhs == {
            "kind": "rhs",
            "field": options["provider_slot"],
            "block": block,
            "binding_ordinal": ordinal,
            "binding_identity": canonical_bytes(provider["provider_identity"]).hex(),
            "provider_key": provider["key"],
            "coefficient": float(provider["coefficient"]),
        }
        assert rhs["provider_key"] == {
            "electrons": "electron_charge", "ions": "ion_charge",
        }[block]
        bindings.append(rhs["binding_identity"])
    assert len(bindings) == len(set(bindings)) == 2


def test_uniform_total_compile_forwards_each_resolved_block_role(
    uniform_multiphysics, monkeypatch,
) -> None:
    from pops.codegen import _orchestration_compile as orchestration

    resolved = uniform_multiphysics
    expected = _resolved_native_amr_field_roles(resolved)
    observed = {}

    def fake_compile(block_name, model, backend, target, options, **kwargs):
        del model, backend, options
        assert target == "system"
        observed[block_name] = kwargs
        return block_name

    monkeypatch.setattr(orchestration, "compile_install_model", fake_compile)
    assert orchestration.compile_install_models(resolved, {}) == {
        "electrons": "electrons", "ions": "ions",
    }
    for block in resolved.blocks:
        assert observed[block.name]["native_field_roles"] == expected[block.name]
        assert observed[block.name]["consumer_owner_qid"] == block.instance_owner_qid
        assert observed[block.name]["resolved_operations"] is block.resolved_operations


@pytest.mark.parametrize("block_name", ("electrons", "ions"))
def test_uniform_native_source_emits_only_the_exact_rhs_attachment(
    uniform_multiphysics, block_name: str,
) -> None:
    resolved = uniform_multiphysics
    roles = _resolved_native_amr_field_roles(resolved)[block_name]
    block = next(block for block in resolved.blocks if block.name == block_name)
    state_space, = block.state_spaces
    emitter, _ = lower_and_validate(
        block.model, state_space=state_space, resolved_operations=block.resolved_operations,
    )
    source = emitter._m.emit_cpp_native_loader(
        name="UniformField" + block_name,
        target="system",
        native_field_roles=roles,
        consumer_owner_qid=block.instance_owner_qid,
    )
    install = source[source.index("void pops_install_native("):]
    rhs, = (role for role in roles if role["kind"] == "rhs")
    assert install.count("NativeEllipticAttachmentRole::rhs_only") == 1
    assert "attachment.field = %s;" % json.dumps(rhs["provider_key"]) in install
    assert "attachment.field_slot = %s;" % json.dumps(rhs["field"]) in install
    assert "attachment.binding_identity = %s;" % json.dumps(rhs["binding_identity"]) in install
    assert 'attachment.rhs_identity = "' in install
    assert '/%s";' % rhs["provider_key"] in install
    assert "attachment.rhs = std::move(named_elliptic_rhs_0);" in install
    # The resolved Case stages the single output route. An RHS package keeps the struct's
    # canonical empty outputs/sign=1; it must not copy the model-local field-output claim.
    assert "attachment.outputs =" not in install
    assert "attachment.output_keys =" not in install
    assert "attachment.gradient_sign =" not in install
    assert "NativeEllipticAttachmentRole::output_and_rhs" not in install
    assert "package.elliptic_attachments.push_back({" not in install
    assert "s->commit(std::move(package));" in install


def _repeated_provider_roles(resolved, block_name: str) -> tuple:
    field_name, = resolved.field_plans
    options = resolved.field_plans[field_name].native_install_data()
    provider = next(row for row in options["provider_pack"] if row["owner_block"] == block_name)
    # Derive an extra signed term from the resolved provider's exact identity. The baseline
    # fixture and its authenticated FieldInstallPlan are unchanged; only this projection input
    # contains the repeated contribution.
    repeated_options = {
        **options,
        "provider_pack": [*options["provider_pack"], {**provider, "coefficient": -0.25}],
    }
    projection_input = SimpleNamespace(
        target=resolved.target,
        blocks=resolved.blocks,
        field_plans={field_name: SimpleNamespace(native_install_data=lambda: repeated_options)},
    )
    roles = _resolved_native_amr_field_roles(projection_input)[block_name]
    rhs = [role for role in roles if role["kind"] == "rhs"]
    assert [role["coefficient"] for role in rhs] == [1.0, -0.25]
    assert len({role["binding_ordinal"] for role in rhs}) == 2
    assert len({(role["field"], role["provider_key"], role["binding_identity"])
                for role in rhs}) == 1
    assert roles == _normalize_native_amr_field_roles(roles)
    return roles


@pytest.mark.parametrize("block_name", ("electrons", "ions"))
def test_uniform_repeated_signed_provider_keeps_terms_and_emits_one_rhs_closure(
    uniform_multiphysics, block_name: str,
) -> None:
    resolved = uniform_multiphysics
    roles = _repeated_provider_roles(resolved, block_name)
    block = next(block for block in resolved.blocks if block.name == block_name)
    state_space, = block.state_spaces
    emitter, _ = lower_and_validate(
        block.model, state_space=state_space, resolved_operations=block.resolved_operations,
    )
    source = emitter._m.emit_cpp_native_loader(
        name="RepeatedUniformField" + block_name,
        target="system",
        native_field_roles=roles,
        consumer_owner_qid=block.instance_owner_qid,
    )
    install = source[source.index("void pops_install_native("):]
    assert install.count("auto named_elliptic_model_") == 1
    assert install.count("auto named_elliptic_rhs_") == 1
    assert install.count("NativeEllipticAttachmentRole::rhs_only") == 1
    assert install.count("package.elliptic_attachments.push_back(") == 1
    assert [role["coefficient"] for role in roles if role["kind"] == "rhs"] == [1.0, -0.25]


@pytest.mark.parametrize("block_name", ("electrons", "ions"))
def test_uniform_repeated_provider_refuses_conflicting_binding_identity(
    uniform_multiphysics, block_name: str,
) -> None:
    resolved = uniform_multiphysics
    roles = _repeated_provider_roles(resolved, block_name)
    conflicting = (*roles[:-1], {**roles[-1], "binding_identity": "foreign-binding"})
    block = next(block for block in resolved.blocks if block.name == block_name)
    state_space, = block.state_spaces
    emitter, _ = lower_and_validate(
        block.model, state_space=state_space, resolved_operations=block.resolved_operations,
    )
    with pytest.raises(ValueError, match="conflicting provider identities"):
        emitter._m.emit_cpp_native_loader(
            name="ConflictingUniformField" + block_name,
            target="system",
            native_field_roles=conflicting,
            consumer_owner_qid=block.instance_owner_qid,
        )


def _rhs_role() -> dict:
    return {
        "kind": "rhs", "field": "tests.uniform.field-slot", "block": "material",
        "binding_ordinal": 0, "binding_identity": "tests.uniform.binding",
        "provider_key": "charge", "coefficient": 0.1,
    }


def _default_rhs_emitter():
    from pops.physics._facade import Model

    model = Model("uniform_default_rhs")
    (rho,) = model.conservative_vars("rho")
    model.primitive_vars(rho)
    model.conservative_from([rho])
    model.flux(x=[rho], y=[0 * rho])
    model.eigenvalues(x=[1 + 0 * rho], y=[0 * rho])
    model.elliptic_rhs(2 * rho)
    emitter, _ = lower_and_validate(model, facade=model)
    assert emitter._m._elliptic is not None
    return emitter


def test_uniform_explicit_default_rhs_role_emits_one_selected_closure() -> None:
    emitter = _default_rhs_emitter()
    role = {**_rhs_role(), "provider_key": "fields_from_state", "coefficient": 1.0}
    source = emitter._m.emit_cpp_native_loader(
        name="SelectedDefaultRhs", target="system", native_field_roles=(role,),
    )
    install = source[source.index("void pops_install_native("):]
    assert install.count("auto named_elliptic_rhs_") == 1
    assert "auto named_elliptic_rhs_0 = pops::make_poisson_rhs(model);" in install
    assert install.count("NativeEllipticAttachmentRole::rhs_only") == 1
    assert install.count("package.elliptic_attachments.push_back(") == 1
    assert 'attachment.field = "fields_from_state";' in install
    assert "attachment.field_slot = %s;" % json.dumps(role["field"]) in install
    assert "attachment.binding_identity = %s;" % json.dumps(role["binding_identity"]) in install
    assert "attachment.rhs = std::move(named_elliptic_rhs_0);" in install
    assert "auto fields_from_state_rhs =" not in install
    assert "package.elliptic_attachments.push_back({" not in install


def test_uniform_empty_roles_suppress_default_rhs_but_legacy_none_preserves_it() -> None:
    emitter = _default_rhs_emitter()
    empty_source = emitter._m.emit_cpp_native_loader(
        name="UnselectedDefaultRhs", target="system", native_field_roles=(),
    )
    empty_install = empty_source[empty_source.index("void pops_install_native("):]
    assert "package.elliptic_attachments.push_back(" not in empty_install
    assert "pops::make_poisson_rhs(" not in empty_install

    legacy_source = emitter._m.emit_cpp_native_loader(
        name="LegacyDefaultRhs", target="system", native_field_roles=None,
    )
    legacy_install = legacy_source[legacy_source.index("void pops_install_native("):]
    assert "auto fields_from_state_rhs = pops::make_poisson_rhs(model);" in legacy_install
    assert legacy_install.count("package.elliptic_attachments.push_back(") == 1
    assert 'package.elliptic_attachments.push_back({"fields_from_state",' in legacy_install
    assert "NativeEllipticAttachmentRole::rhs_only" not in legacy_install


@pytest.mark.parametrize("change, exception", (
    ("missing_binding", TypeError),
    ("empty_field", TypeError),
    ("rhs_output_keys", TypeError),
    ("rhs_gradient_sign", TypeError),
    ("nonfinite_coefficient", ValueError),
    ("duplicate_binding", ValueError),
))
def test_uniform_rhs_roles_refuse_incomplete_or_output_claiming_shapes(
    change: str, exception: type[Exception],
) -> None:
    role = _rhs_role()
    roles = [role]
    if change == "missing_binding":
        del role["binding_identity"]
    elif change == "empty_field":
        role["field"] = ""
    elif change == "rhs_output_keys":
        role["output_keys"] = ()
    elif change == "rhs_gradient_sign":
        role["gradient_sign"] = -1
    elif change == "nonfinite_coefficient":
        role["coefficient"] = float("nan")
    else:
        roles.append(dict(role))
    with pytest.raises(exception):
        _normalize_native_amr_field_roles(roles)


@pytest.mark.parametrize("change", ("empty_outputs", "invalid_sign", "invalid_key"))
def test_uniform_output_roles_retain_their_exact_key_and_gradient_contract(change: str) -> None:
    role = {
        "kind": "output", "field": "tests.uniform.field-slot", "block": "material",
        "output_keys": ({"owner_qid": "tests/material", "space_kind": "field",
                         "space_name": "potential", "component": "potential"},),
        "gradient_sign": 1,
    }
    if change == "empty_outputs":
        role["output_keys"] = ()
    elif change == "invalid_sign":
        role["gradient_sign"] = 0
    else:
        del role["output_keys"][0]["owner_qid"]
    with pytest.raises((TypeError, ValueError)):
        _normalize_native_amr_field_roles((role,))


def test_system_artifact_identity_changes_with_exact_field_roles(tmp_path, monkeypatch) -> None:
    from pops.codegen import _artifact_identity as artifact_identity
    from pops.codegen import _compile_drivers as drivers

    identity_names = []
    runtime_roles = []

    class Model:
        _elliptic_fields = {}

        @staticmethod
        def _check_require_metadata(require_metadata, backend):
            del require_metadata, backend

    def artifact_spec(model, **kwargs):
        del model
        identity_names.append(kwargs["name"])
        return object(), kwargs["name"]

    def compile_native(model, path, *args, native_field_roles, **kwargs):
        del model, args
        assert kwargs["target"] == "system"
        runtime_roles.append(native_field_roles)
        return path

    monkeypatch.setattr(drivers, "_native_kokkos_compiler", lambda cxx: cxx)
    monkeypatch.setattr(drivers, "_abi_key_python", lambda *args: "test-abi")
    monkeypatch.setattr(artifact_identity, "model_artifact_spec", artifact_spec)
    monkeypatch.setattr(drivers, "_identity_cache_so_path", lambda identity: str(
        tmp_path / (hashlib.sha256(identity.encode()).hexdigest() + ".so")
    ))
    monkeypatch.setattr(drivers, "_record_artifact_identity", lambda *args: None)
    monkeypatch.setattr(drivers, "compile_native", compile_native)
    monkeypatch.setattr(drivers, "publish_staged_artifact", lambda *args, **kwargs: None)

    base = _rhs_role()
    output = {
        "kind": "output", "field": base["field"], "block": base["block"],
        "output_keys": tuple(
            {"owner_qid": "tests/material", "space_kind": "field",
             "space_name": "potential", "component": component}
            for component in ("potential", "electric_x", "electric_y")
        ),
        "gradient_sign": -1,
    }
    repeated = {**base, "binding_ordinal": 1, "coefficient": -0.25}
    variants = (
        (output, base),
        (output, {**base, "binding_identity": "tests.uniform.other-binding"}),
        (output, {**base, "field": "tests.uniform.other-field-slot"}),
        (output, {**base, "provider_key": "other_charge"}),
        (output, {**base, "coefficient": 0.2}),
        ({**output, "gradient_sign": 1}, base),
        ({**output, "output_keys": tuple(
            {**key, "owner_qid": "tests/other-material"} for key in output["output_keys"]
        )}, base),
        (output, base, repeated),
        (output, base, {**repeated, "coefficient": -0.5}),
        (output, repeated, base),
    )
    paths = []
    for roles in (*variants, variants[0]):
        paths.append(drivers.compile_model(
            Model(), include="test-headers", cxx="test-c++", std="c++23",
            target="system", name="uniform-field-package", _native_field_roles=roles,
        ))
        assert runtime_roles[-1] == _normalize_native_amr_field_roles(roles)
        assert type(runtime_roles[-1][-1]["coefficient"]) is float
    assert len(set(paths[:-1])) == len(variants)
    assert len(set(identity_names[:-1])) == len(variants)
    assert paths[-1] == paths[0]
    assert identity_names[-1] == identity_names[0]
