"""Extension contract for prepared-preconditioner compilation providers."""

from dataclasses import replace
from types import SimpleNamespace

import pytest


def test_plugin_provider_registers_resolves_and_emits_without_dispatcher_changes(tmp_path):
    from pops.codegen.program_emit_solve import _prepared_preconditioner
    from pops.solvers._prepared_preconditioner_registry import (
        prepared_preconditioner_provider_by_emitter_id,
        prepared_preconditioner_provider_by_id,
        prepared_preconditioner_provider_from_identity,
    )
    from pops.solvers import preconditioners

    include_root = tmp_path / "include"
    header = include_root / "example" / "native_preconditioner.hpp"
    header.parent.mkdir(parents=True)
    header.write_text("#pragma once\n", encoding="utf-8")

    calls = []

    def emit(node, prelude, prototype, vector_distribution_expr, provider):
        calls.append((node.id, provider.scheme))
        return preconditioners.NativeEmission(
            "example::make_preconditioner_apply(*%s, %s)" % (
                prototype,
                vector_distribution_expr,
            )
        )

    provider = preconditioners.register(
        preconditioners.Provider(
            provider_id="example.preconditioner",
            interface_version=1,
            options_schema="example.preconditioner.options@1",
            scheme="test_plugin_preconditioner",
            descriptor_name="test_plugin_preconditioner",
            display_name="TestPluginPreconditioner()",
            native_id="example::NativePreconditioner",
            validator_id="example.prepared-preconditioner.validate@1",
            planner_id="example.prepared-preconditioner.plan@1",
            emitter_id="example.prepared-preconditioner@1",
            preconditioned=True,
            prepared_buffers=1,
            use_policy=preconditioners.UsePolicy(
                "example.prepared-preconditioner.use", 1,
                {"methods": ("gmres",)}, lambda _use, _where: None,
            ),
            options=(),
            emitter=emit,
            native_component=preconditioners.HeaderOnlyComponent(
                "example.prepared-preconditioner",
                include_root=include_root,
                entry_headers=("example/native_preconditioner.hpp",),
            ),
        )
    )

    assert prepared_preconditioner_provider_by_id(provider.provider_id) is provider
    assert prepared_preconditioner_provider_by_emitter_id(provider.emitter_id) is provider
    assert prepared_preconditioner_provider_from_identity(provider.authority()) is provider

    prelude = []
    node = SimpleNamespace(
        id=17,
        op="solve_linear",
        attrs={
            "preconditioner_provider": provider.authority(),
            "preconditioner_options": {},
        },
    )
    expression = _prepared_preconditioner(
        node,
        prelude,
        "prototype_field",
        "pops::VectorDistribution::distributed()",
    )

    assert calls == [(17, "test_plugin_preconditioner")]
    assert prelude == []
    assert expression.startswith("pops::PreparedLinearPreconditioner(*prototype_field, ")
    assert "pops::PreparedLinearPreconditionerProvider::trusted_extension(" in expression
    assert 'pops::PreparedProviderIdentity{"example.prepared-preconditioner@1", 1ull}' in expression
    assert (
        "example::make_preconditioner_apply(*prototype_field, "
        "pops::VectorDistribution::distributed())" in expression
    )
    assert ", {}, example::make_preconditioner_apply(" in expression
    assert expression.endswith(
        "pops::VectorDistribution::distributed())), "
        "pops::VectorDistribution::distributed())"
    )
    from pops.codegen.program_emit_kernels import (
        _prepared_native_component_includes,
        _prepared_native_components,
    )
    from pops.fields._prepared_nullspace_registry import none_prepared_nullspace_provider
    from pops.solvers.krylov._prepared_method_registry import (
        prepared_krylov_method_provider_by_id,
    )

    nullspace_component = none_prepared_nullspace_provider().native_component
    method = prepared_krylov_method_provider_by_id("pops.krylov.gmres")
    program = SimpleNamespace(_values=(node,))
    node.attrs.update({
        "method_provider": method.authority(),
        "method_options": {"restart": 5},
        "nullspace_provider": none_prepared_nullspace_provider().authority(),
        "nullspace_contract": {
            "schema_version": 1,
            "provider": none_prepared_nullspace_provider().authority(),
            "contract": {"declaration": "nonsingular"},
        },
    })
    assert _prepared_native_components(program) == (
        method.native_component,
        provider.native_component,
        nullspace_component,
    )
    assert _prepared_native_component_includes(program) == (
        "#include <pops/numerics/elliptic/linear/generic_krylov.hpp>  "
        "// prepared native provider\n"
        "#include <example/native_preconditioner.hpp>  "
        "// prepared native provider\n"
    )


def test_provider_registry_is_append_only_and_rejects_both_identity_collisions():
    from pops.solvers._prepared_preconditioner_registry import (
        PreparedPreconditionerProvider,
        prepared_preconditioner_provider_by_id,
        register_prepared_preconditioner_provider,
    )
    from pops.native_components import PreparedNativeComponent
    from pops.solvers import preconditioners

    existing = prepared_preconditioner_provider_by_id("pops.preconditioner.identity")

    def provider(*, provider_id, emitter_id):
        return PreparedPreconditionerProvider(
            provider_id=provider_id,
            interface_version=1,
            options_schema=provider_id + ".options@1",
            scheme=provider_id,
            descriptor_name=provider_id,
            display_name=provider_id,
            native_id="example::NativePreconditioner",
            validator_id=provider_id + ".validate@1",
            planner_id=provider_id + ".plan@1",
            emitter_id=emitter_id,
            preconditioned=True,
            prepared_buffers=0,
            use_policy=preconditioners.UsePolicy(
                "pops.test.%s.use" % provider_id, 1,
                {"methods": ("gmres",)}, lambda _use, _where: None,
            ),
            options=(),
            emitter=lambda *_args: preconditioners.NativeEmission(
                "example::NativePreconditioner{}"
            ),
            native_component=PreparedNativeComponent.pops_builtin(
                "pops.test.%s" % provider_id
            ),
        )

    with pytest.raises(ValueError, match="provider .* already registered"):
        register_prepared_preconditioner_provider(
            provider(provider_id=existing.provider_id, emitter_id="example.unique-emitter@1")
        )
    with pytest.raises(ValueError, match="emitter .* already registered"):
        register_prepared_preconditioner_provider(
            provider(provider_id="example.unique", emitter_id=existing.emitter_id)
        )


def test_provider_authority_authenticates_option_and_allocation_contracts(monkeypatch):
    import pops.solvers._prepared_preconditioner_registry as registry
    from pops.native_components import PreparedNativeComponent
    from pops.solvers import preconditioners

    provider = preconditioners.register(preconditioners.Provider(
        provider_id="example.preconditioner.authority",
        interface_version=1,
        options_schema="example.preconditioner.authority.options@1",
        scheme="authority_probe",
        descriptor_name="authority_probe",
        display_name="AuthorityProbe()",
        native_id="example::AuthorityProbe",
        validator_id="example.preconditioner.authority.validate@1",
        planner_id="example.preconditioner.authority.plan@1",
        emitter_id="example.preconditioner.authority.emit@1",
        preconditioned=True,
        prepared_buffers=1,
        use_policy=preconditioners.UsePolicy(
            "example.preconditioner.authority.use", 1, {}, lambda _use, _where: None,
        ),
        options=(preconditioners.IntOption("passes", default=1, minimum=1),),
        emitter=lambda *_args: preconditioners.NativeEmission("example::apply()"),
        native_component=PreparedNativeComponent.pops_builtin(
            "example.preconditioner.authority"
        ),
        scratch_resources=(
            preconditioners.ScratchResource("vendor_workspace", 2, True, "exact vendor pool"),
        ),
    ))
    authority = provider.authority()
    assert authority["option_contracts"] == ({
        "schema_version": 1,
        "type_id": "pops.prepared-preconditioner.option.signed-int@1",
        "name": "passes",
        "default": 1,
        "minimum": 1,
        "maximum": (1 << 31) - 1,
    },)
    assert authority["allocation_plan"] == {
        "schema_version": 1,
        "planner_id": "example.preconditioner.authority.plan@1",
        "prepared_buffers": 1,
        "scratch_resources": ({
            "schema_version": 1,
            "kind": "vendor_workspace",
            "buffers": 2,
            "exact": True,
            "note": "exact vendor pool",
        },),
    }

    changed_options = replace(
        provider,
        options=(preconditioners.IntOption("passes", default=2, minimum=1),),
    )
    changed_plan = replace(
        provider,
        planner_id="example.preconditioner.authority.plan@2",
        prepared_buffers=3,
    )
    assert changed_options.authority() != authority
    assert changed_plan.authority() != authority

    # Simulate loading old IR in a newer process that accidentally reused the provider id.
    # The complete option/allocation authority makes the stale IR fail before planning/emission.
    monkeypatch.setitem(registry._providers_by_id, provider.provider_id, changed_plan)
    monkeypatch.setitem(registry._providers_by_emitter_id, provider.emitter_id, changed_plan)
    with pytest.raises(ValueError, match="identity is inconsistent"):
        registry.prepared_preconditioner_provider_from_identity(authority)


def test_unknown_provider_identity_has_no_emission_fallback():
    from pops.codegen.program_emit_solve import _prepared_preconditioner
    from pops.solvers._prepared_preconditioner_registry import (
        prepared_preconditioner_provider_by_id,
    )

    identity = prepared_preconditioner_provider_by_id(
        "pops.preconditioner.identity"
    ).authority()
    identity = dict(identity)
    identity["emitter_id"] = "unknown.plugin@1"
    node = SimpleNamespace(
        id=23,
        attrs={
            "preconditioner_provider": identity,
            "preconditioner_options": {},
        },
    )
    with pytest.raises(NotImplementedError, match="not registered"):
        _prepared_preconditioner(node, [], "prototype", "distribution")


def test_problem_compatibility_is_delegated_to_the_provider_use_policy():
    from pops.solvers._prepared_preconditioner_registry import (
        prepared_preconditioner_provider_by_id,
    )

    from pops.fields import ConstantNullspace, MeanValueGauge
    from pops.linalg import LinearOperatorProperties, LinearProblem

    none = LinearProblem(object(), object(), nullspace=None).canonical_nullspace_contract()
    constant = LinearProblem(
        object(), object(),
        properties=LinearOperatorProperties.symmetric_operator(),
        nullspace=ConstantNullspace(), gauge=MeanValueGauge(0),
    ).canonical_nullspace_contract()
    identity = prepared_preconditioner_provider_by_id("pops.preconditioner.identity")
    from pops.solvers.krylov._prepared_method_registry import (
        prepared_krylov_method_provider_by_id,
    )
    cg = prepared_krylov_method_provider_by_id("pops.krylov.cg").authority()
    gmres = prepared_krylov_method_provider_by_id("pops.krylov.gmres").authority()
    identity.validate_use(
        method_provider=cg, components=7, nullspace_contract=constant, where="test"
    )

    geometric = prepared_preconditioner_provider_by_id(
        "pops.preconditioner.geometric-mg"
    )
    geometric.validate_use(
        method_provider=gmres, components=1, nullspace_contract=none, where="test"
    )
    with pytest.raises(ValueError, match="left-preconditioning"):
        geometric.validate_use(
            method_provider=cg, components=1, nullspace_contract=none, where="test"
        )
    with pytest.raises(ValueError, match="scalar-only"):
        geometric.validate_use(
            method_provider=gmres, components=2, nullspace_contract=none, where="test"
        )
    with pytest.raises(NotImplementedError, match="nullspace contract"):
        geometric.validate_use(
            method_provider=gmres, components=1, nullspace_contract=constant, where="test"
        )


def test_header_only_component_detects_any_tree_drift_and_excludes_root_from_identity(tmp_path):
    from pops.native_components import PreparedNativeComponent

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    for root in (first_root, second_root):
        (root / "vendor").mkdir(parents=True)
        (root / "vendor" / "provider.hpp").write_text("#pragma once\n", encoding="utf-8")
        (root / "vendor" / "detail.hpp").write_text("inline constexpr int k = 1;\n", encoding="utf-8")
    first = PreparedNativeComponent.header_only(
        "vendor.provider", include_root=first_root,
        entry_headers=("vendor/provider.hpp",),
    )
    second = PreparedNativeComponent.header_only(
        "vendor.provider", include_root=second_root,
        entry_headers=("vendor/provider.hpp",),
    )
    assert first.manifest() == second.manifest()
    assert first.manifest_sha256 == second.manifest_sha256

    (first_root / "vendor" / "detail.hpp").write_text(
        "inline constexpr int k = 2;\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="changed after registration"):
        first.stage_verified(tmp_path / "staged")


@pytest.mark.parametrize(
    ("directive", "error"),
    [
        ('#include "/tmp/untracked.hpp"\n', "escapes"),
        ("#include PROVIDER_SELECTED_HEADER\n", "literal"),
        ('#include "missing/detail.hpp"\n', "absent"),
    ],
)
def test_header_only_component_refuses_unauthenticated_include_inputs(
    tmp_path, directive, error,
):
    from pops.native_components import PreparedNativeComponent

    root = tmp_path / error
    (root / "vendor").mkdir(parents=True)
    (root / "vendor" / "provider.hpp").write_text(directive, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        PreparedNativeComponent.header_only(
            "vendor.invalid.%s" % error,
            include_root=root,
            entry_headers=("vendor/provider.hpp",),
        )


def test_public_prepared_descriptor_is_registry_backed_without_scheme_dispatch(tmp_path):
    from pops.solvers import preconditioners
    from pops.time._program.solve import _lower_preconditioner

    include_root = tmp_path / "include"
    (include_root / "vendor").mkdir(parents=True)
    (include_root / "vendor" / "provider.hpp").write_text("#pragma once\n", encoding="utf-8")
    provider = preconditioners.register(preconditioners.Provider(
        provider_id="vendor.prepared-public",
        interface_version=1,
        options_schema="vendor.prepared-public.options@1",
        scheme="test_public_prepared_provider",
        descriptor_name="test_public_prepared_provider",
        display_name="VendorPrepared()",
        native_id="vendor::Prepared",
        validator_id="vendor.prepared-public.validate@1",
        planner_id="vendor.prepared-public.plan@1",
        emitter_id="vendor.prepared-public@1",
        preconditioned=True,
        prepared_buffers=0,
        use_policy=preconditioners.UsePolicy(
            "vendor.prepared-public.use", 1,
            {"methods": ("gmres",)}, lambda _use, _where: None,
        ),
        options=(preconditioners.IntOption("passes", default=1, minimum=1),),
        emitter=lambda *_args: preconditioners.NativeEmission(
            "vendor::make_preconditioner_apply()"
        ),
        native_component=preconditioners.HeaderOnlyComponent(
            "vendor.prepared-public",
            include_root=include_root,
            entry_headers=("vendor/provider.hpp",),
        ),
    ))
    descriptor = preconditioners.Prepared(provider, passes=3)
    assert descriptor.options == {"passes": 3}
    assert _lower_preconditioner(descriptor) == (provider.authority(), {"passes": 3})
