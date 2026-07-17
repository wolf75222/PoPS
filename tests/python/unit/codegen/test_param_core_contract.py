"""Core contract of canonical declarations, ParamHandle ownership and module manifests."""
from __future__ import annotations

import copy

import pytest

pytest.importorskip("pops")

from pops import model
from pops._ir.expr import Const, Var
from pops._ir.values import RuntimeParamRef
from pops.math import Bool, Integer, Real
from pops.params import (
    MISSING,
    ConstParam,
    DerivedParam,
    Interval,
    ParamDefaultState,
    ParamInvalidation,
    ParamKind,
    ParamPhase,
    ParamProvenance,
    ParamStorage,
    Positive,
    RuntimeParam,
    validate_parameter_data,
)


def _derived(module: model.Module, dependency: model.ParamHandle) -> model.ParamHandle:
    expression = module.value(dependency) * module.value(dependency)
    return module.param(DerivedParam(
        "alpha2",
        expression,
        depends_on=(dependency,),
        phase=ParamPhase.PerBlock,
        storage=ParamStorage.DerivedCache,
        invalidation=ParamInvalidation.OnDependencies,
        dtype=Real,
        unit="m2/s2",
    ))


def test_runtime_default_state_is_lossless_but_not_artifact_identity():
    missing = RuntimeParam("alpha", dtype=Real)
    first = RuntimeParam("alpha", dtype=Real, default=1.0)
    second = RuntimeParam("alpha", dtype=Real, default=2.0)

    assert missing.default is MISSING
    assert missing.default_state is ParamDefaultState.Missing
    assert first.to_data()["default"]["state"] == "value"
    assert missing.artifact_data() == first.artifact_data() == second.artifact_data()
    assert missing.bind_data() != first.bind_data() != second.bind_data()


def test_const_value_remains_in_artifact_identity():
    first = ConstParam("order", 2, dtype=Integer)
    second = ConstParam("order", 3, dtype=Integer)
    assert first.kind is ParamKind.Const
    assert first.storage is ParamStorage.Inline
    assert first.artifact_data() != second.artifact_data()


def test_typed_value_domain_unit_and_provenance_are_strict_and_immutable():
    provenance = ParamProvenance("input-deck", metadata={"section": "transport"})
    param = RuntimeParam(
        "speed", dtype=Real, default=1.5, domain=Positive(), unit="m/s",
        provenance=provenance,
    )
    assert validate_parameter_data(param.to_data()) == param.to_data()
    assert param.to_data()["provenance"] == {
        "source": "input-deck", "metadata": {"section": "transport"},
    }
    with pytest.raises(AttributeError, match="immutable"):
        param.default = 2.0
    with pytest.raises(TypeError, match="requires an int"):
        ConstParam("order", 2.5, dtype=Integer)
    with pytest.raises(TypeError, match="requires a bool"):
        RuntimeParam("enabled", dtype=Bool, default=1)
    with pytest.raises(ValueError, match="compile"):
        RuntimeParam("bad", default=-1.0, domain=Positive())


def test_constraints_round_trip_without_opaque_python_objects():
    domain = Interval(0, 8)
    assert type(domain).from_data(domain.to_data()).to_data() == domain.to_data()
    with pytest.raises(TypeError, match="Constraint"):
        RuntimeParam("bad", domain=object())
    with pytest.raises(TypeError, match="typed"):
        RuntimeParam("bad", domain="positive")


def test_module_param_requires_a_canonical_declaration_and_returns_param_handle():
    module = model.Module("transport")
    declaration = RuntimeParam("alpha", default=1.0)
    handle = module.param(declaration)

    assert isinstance(handle, model.ParamHandle)
    assert handle.param_kind == "runtime"
    assert module.param_handle(declaration) is handle
    assert module.param_declaration(handle) is declaration
    assert module.params() == {"alpha": declaration}
    with pytest.raises(TypeError, match="RuntimeParam"):
        module.param("alpha")
    with pytest.raises(TypeError):
        module.param("beta", 2.0)
    with pytest.raises(ValueError, match="already declared"):
        module.param(RuntimeParam("alpha", default=2.0))


def test_param_handle_subtype_and_kind_round_trip_canonically():
    module = model.Module("transport")
    handle = module.param(ConstParam("order", 2, dtype=Integer))
    canonical = module.declaration_index().authenticate(handle)._resolved(
        module.owner_path.canonical()
    )
    data = canonical.canonical_identity()
    rebuilt = model.Handle.from_canonical_identity(data)

    assert isinstance(rebuilt, model.ParamHandle)
    assert rebuilt == canonical
    assert rebuilt.param_kind == "const"
    forged = copy.deepcopy(data)
    forged["param_kind"] = "runtime"
    with pytest.raises(ValueError, match="invalid qualified_id|payload"):
        model.Handle.from_canonical_identity(forged)


def test_runtime_ir_read_authenticates_parameter_kind_and_stable_dtype_metadata():
    module = model.Module("transport")
    runtime = module.param(RuntimeParam("alpha", dtype=Real, default=1.0))
    const = module.param(ConstParam("order", 2, dtype=Integer))

    read = RuntimeParamRef("alpha", 0.0, handle=runtime, dtype=Real)
    assert read.handle is runtime and read.dtype == "Real"
    with pytest.raises(TypeError, match="Runtime/Derived"):
        RuntimeParamRef("order", 0, handle=const, dtype=Integer)


def test_registry_rejects_foreign_dependency_and_compile_derived_from_runtime():
    first = model.Module("first")
    runtime = first.param(RuntimeParam("alpha", default=1.0))
    second = model.Module("second")
    expression = Var("alpha", "param")
    with pytest.raises(ValueError, match="not issued"):
        second.param(DerivedParam(
            "foreign", expression, depends_on=(runtime,), phase=ParamPhase.Bind,
            storage=ParamStorage.DerivedCache,
            invalidation=ParamInvalidation.OnDependencies,
        ))
    with pytest.raises(ValueError, match="Compile cannot depend on runtime"):
        first.param(DerivedParam(
            "compile_bad", expression, depends_on=(runtime,), phase=ParamPhase.Compile,
            storage=ParamStorage.Inline, invalidation=ParamInvalidation.Never,
        ))


def test_registry_authenticates_expression_dependencies_exactly():
    module = model.Module("dependencies")
    alpha = module.param(ConstParam("alpha", 2.0))
    with pytest.raises(ValueError, match="undeclared dependency"):
        module.param(DerivedParam(
            "ghost", Var("ghost", "param"), depends_on=(alpha,),
            phase=ParamPhase.Compile, storage=ParamStorage.Inline,
            invalidation=ParamInvalidation.Never,
        ))
    with pytest.raises(ValueError, match="unread parameter"):
        module.param(DerivedParam(
            "unused", Const(1.0), depends_on=(alpha,),
            phase=ParamPhase.Compile, storage=ParamStorage.Inline,
            invalidation=ParamInvalidation.Never,
        ))

    foreign_module = model.Module("foreign-same-name")
    foreign_alpha = foreign_module.param(ConstParam("alpha", 2.0))
    with pytest.raises(ValueError, match="foreign parameter handle"):
        module.param(DerivedParam(
            "foreign_read", foreign_module.value(foreign_alpha), depends_on=(alpha,),
            phase=ParamPhase.Compile, storage=ParamStorage.Inline,
            invalidation=ParamInvalidation.Never,
        ))


def test_compile_derived_result_revalidates_dtype_and_domain():
    typed = model.Module("typed-derived")
    three = typed.param(ConstParam("three", 3, dtype=Integer))
    with pytest.raises(TypeError, match="requires an int"):
        typed.param(DerivedParam(
            "half", typed.value(three) / 2, depends_on=(three,),
            phase=ParamPhase.Compile, storage=ParamStorage.Inline,
            invalidation=ParamInvalidation.Never, dtype=Integer,
        ))

    signed = model.Module("domain-derived")
    negative = signed.param(ConstParam("negative", -1.0))
    with pytest.raises(ValueError, match="outside domain"):
        signed.param(DerivedParam(
            "positive", signed.value(negative), depends_on=(negative,),
            phase=ParamPhase.Compile, storage=ParamStorage.Inline,
            invalidation=ParamInvalidation.Never, domain=Positive(),
        ))


def test_param_registry_accepts_explicit_case_authority_but_never_ownerless():
    owner = model.OwnerPath.fresh(model.OwnerKind.CASE, "case")
    registry = model.ParamRegistry(owner=owner)
    handle = registry.register(RuntimeParam("threshold", default=0.1))
    assert handle.owner_path == owner
    with pytest.raises((TypeError, ValueError)):
        model.ParamRegistry(owner=None)


def test_one_declaration_object_cannot_be_claimed_by_multiple_owners():
    declaration = RuntimeParam("alpha", default=1.0)
    first = model.Module("first")
    second = model.Module("second")
    first.param(declaration)

    assert declaration.is_owned
    assert declaration.owner_identity == str(first.owner_path)
    with pytest.raises(ValueError, match="already owned|shared owner or tie"):
        second.param(declaration)


def test_derived_contract_is_explicit_and_manifest_is_lossless():
    module = model.Module("transport")
    alpha = module.param(RuntimeParam("alpha", default=1.0, domain=Positive()))
    alpha2 = _derived(module, alpha)
    declaration = module.param_declaration(alpha2)

    assert declaration.default_state is ParamDefaultState.Derived
    assert declaration.depends_on == (alpha,)
    with pytest.raises(TypeError, match="Expr"):
        DerivedParam(
            "bad", "alpha * alpha", depends_on=(alpha,), phase=ParamPhase.PerBlock,
            storage=ParamStorage.DerivedCache,
            invalidation=ParamInvalidation.OnDependencies,
        )

    manifest = module.manifest()
    row = manifest.to_dict()["params"]["alpha2"]
    assert manifest.schema_version == 8
    assert row["kind"] == "derived"
    assert row["depends_on"] == [{"name": "alpha", "param_kind": "runtime"}]
    assert row["phase"] == "per_block"
    assert row["storage"] == "derived_cache"
    assert isinstance(model.Handle.from_canonical_identity(row["handle"]), model.ParamHandle)
    assert model.ModuleManifest.from_dict(manifest.to_dict()).to_dict() == manifest.to_dict()
