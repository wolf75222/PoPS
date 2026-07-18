"""External prepared-Krylov method compiler contract."""

from types import SimpleNamespace

import pytest


def test_external_method_uses_provider_owned_options_native_component_and_emitter(tmp_path):
    from pops.native_components import PreparedNativeComponent
    from pops.solvers import krylov

    def prepare_options(options):
        if set(options) != {"history"} or type(options["history"]) is not int:
            raise ValueError("external method requires an exact history")
        if options["history"] < 1:
            raise ValueError("external method history must be positive")
        return {"history": options["history"]}

    def validate(use, where):
        if use.preconditioned and use.components != 1:
            raise ValueError("%s external method preconditioner is scalar-only" % where)

    include_root = tmp_path / "include"
    header = include_root / "example" / "krylov_history_method.hpp"
    header.parent.mkdir(parents=True)
    header.write_text("#pragma once\n", encoding="utf-8")

    provider = krylov.register_prepared_krylov_method_provider(
        krylov.PreparedKrylovMethodProvider(
            provider_id="example.krylov.history-method",
            interface_version=1,
            options_schema="example.krylov.history-method.options@1",
            emitter_id="example.krylov.history-method@1",
            capabilities={
                "contract_version": 2,
                "preconditioning_placement": "right",
            },
            native_component=PreparedNativeComponent.header_only(
                "example.krylov.history-method",
                include_root=include_root,
                entry_headers=("example/krylov_history_method.hpp",),
            ),
            option_preparer=prepare_options,
            validator=validate,
            emitter=lambda _node, options: "example::history_method(%d)"
            % options["history"],
        )
    )

    descriptor = krylov.Prepared(
        provider,
        max_iter=12,
        method_options={"history": 5},
        name="HistoryMethod",
    )
    prepared = descriptor.prepare_program_solve()
    assert prepared.method_provider == provider.authority()
    assert prepared.method_options == {"history": 5}

    use = krylov.PreparedKrylovMethodUse(
        rel_tol=1.0e-8,
        abs_tol=0,
        max_iterations=12,
        components=1,
        input_ghosts=2,
        preconditioned=True,
        operator_properties={
            "symmetric": False,
            "positive_definite": False,
            "positive_definite_on_nullspace_complement": False,
        },
        declared_nullspace=False,
        method_options=prepared.method_options,
    )
    provider.validate_use(use, where="test")
    node = SimpleNamespace(attrs={
        "method_provider": provider.authority(),
        "method_options": prepared.method_options,
    })
    assert provider.emit_cpp(node) == "example::history_method(5)"


@pytest.mark.parametrize(
    ("provider_id", "placement"),
    (
        ("pops.krylov.cg", "none"),
        ("pops.krylov.bicgstab", "right"),
        ("pops.krylov.gmres", "left"),
        ("pops.krylov.richardson", "none"),
    ),
)
def test_builtin_method_authority_declares_native_preconditioning_placement(
    provider_id,
    placement,
):
    from pops.solvers.krylov._prepared_method_registry import (
        prepared_krylov_method_provider_by_id,
    )

    capabilities = prepared_krylov_method_provider_by_id(provider_id).authority()["capabilities"]
    assert capabilities == {
        "contract_version": 2,
        "preconditioning_placement": placement,
    }


def test_python_registry_has_no_legacy_workspace_authority():
    from pops.solvers import krylov

    assert "workspace_planner" not in krylov.PreparedKrylovMethodProvider.__dataclass_fields__
    assert not hasattr(krylov, "PreparedKrylovWorkspacePlan")


@pytest.mark.parametrize("initial_residual_field", [True, 1.0])
def test_legacy_python_workspace_plan_is_rejected_before_field_bounds(
    initial_residual_field,
):
    """Python cannot forge native workspace indices, even with int-like values."""
    from pops.native_components import PreparedNativeComponent
    from pops.solvers import krylov

    with pytest.raises(TypeError, match="workspace_planner"):
        krylov.PreparedKrylovMethodProvider(
            provider_id="example.legacy-workspace",
            interface_version=1,
            options_schema="example.legacy-workspace.options@1",
            emitter_id="example.legacy-workspace@1",
            capabilities={},
            native_component=PreparedNativeComponent.pops_builtin(
                "example.legacy-workspace",
                entry_headers=("pops/numerics/elliptic/linear/generic_krylov.hpp",),
            ),
            option_preparer=lambda options: options,
            validator=lambda _use, _where: None,
            workspace_planner=lambda _use: {
                "fields": 2,
                "initial_residual_field": initial_residual_field,
            },
            emitter=lambda _node, _options: "example::legacy_workspace_method()",
        )


def test_method_registry_is_append_only_and_unknown_authority_has_no_fallback():
    from pops.solvers.krylov._prepared_method_registry import (
        prepared_krylov_method_provider_by_id,
        prepared_krylov_method_provider_from_identity,
        register_prepared_krylov_method_provider,
    )

    provider = prepared_krylov_method_provider_by_id("pops.krylov.cg")
    with pytest.raises(ValueError, match="already registered"):
        register_prepared_krylov_method_provider(provider)

    forged = dict(provider.authority())
    forged["provider_id"] = "unknown.krylov.provider"
    with pytest.raises(NotImplementedError, match="not registered"):
        prepared_krylov_method_provider_from_identity(forged)
