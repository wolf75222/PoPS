"""ADC-655: sealed CompiledModel parameter metadata is detached from authoring."""
from __future__ import annotations

import gc
import weakref
from types import MappingProxyType

import pytest

from pops.codegen._compiled_parameter import CompiledParameter, compiled_parameters
from pops.codegen.loader import CompiledModel
from pops.math import Real
from pops.model import Module
from pops.params import (
    DerivedParam,
    ParamInvalidation,
    ParamPhase,
    ParamStorage,
    Positive,
    RuntimeParam,
)


def _module_with_derived_parameter():
    module = Module("transport")
    speed = module.param(RuntimeParam(
        "speed", dtype=Real, default=1.5, domain=Positive(), unit="m/s"))
    module.param(DerivedParam(
        "speed2",
        module.value(speed) * module.value(speed),
        depends_on=(speed,),
        phase=ParamPhase.Bind,
        storage=ParamStorage.DerivedCache,
        invalidation=ParamInvalidation.OnDependencies,
        dtype=Real,
        unit="m2/s2",
    ))
    return module


def _compiled(params):
    return CompiledModel(
        so_path="<compiled-parameter-stub>",
        backend="aot",
        adder="add_compiled_block",
        cons_names=("u",),
        cons_roles=("Scalar",),
        prim_names=("u",),
        n_vars=1,
        gamma=None,
        n_aux=0,
        params=params,
        caps={"cpu": True},
        abi_key="abi",
        model_hash="model",
        cxx="c++",
        std="c++23",
    )


def test_projection_keeps_canonical_metadata_and_only_scalar_defaults():
    module = _module_with_derived_parameter()
    projected = compiled_parameters(module.params())

    assert isinstance(projected, MappingProxyType)
    speed = projected["speed"]
    speed2 = projected["speed2"]
    assert isinstance(speed, CompiledParameter)
    assert speed.kind == "runtime" and speed.phase == "bind"
    assert speed.has_default and speed.default == 1.5
    assert speed.domain["kind"] == "positive"
    assert speed.to_data() == module.params()["speed"].bind_data()
    assert speed2.kind == "derived" and speed2.phase == "bind"
    assert speed2.has_default is False and speed2.default is None
    assert speed2.to_data()["depends_on"] == [
        {"name": "speed", "param_kind": "runtime"}
    ]
    assert speed2.to_data()["expression"] is not None
    assert not hasattr(speed2, "expression")
    assert not hasattr(speed2, "depends_on")

    detached = speed.to_data()
    detached["domain"]["kind"] = "tampered"
    assert speed.domain["kind"] == "positive"
    with pytest.raises(TypeError):
        projected["late"] = speed
    with pytest.raises(AttributeError, match="immutable"):
        speed.kind = "const"


def test_seal_projects_params_and_preserves_runtime_compatibility():
    module = _module_with_derived_parameter()
    compiled = _compiled(module.params())
    compiled._runtime_param_names = ("speed", "speed2")
    fallback = _compiled(module.params())

    compiled._seal()
    compiled._seal()
    fallback._seal()

    assert compiled._sealed is True
    assert isinstance(compiled.params, MappingProxyType)
    assert all(isinstance(value, CompiledParameter) for value in compiled.params.values())
    assert compiled.runtime_param_names == ["speed", "speed2"]
    assert compiled.runtime_param_values() == [1.5, None]
    assert fallback.runtime_param_names == ["speed", "speed2"]
    assert fallback.runtime_param_values() == [1.5, None]
    assert compiled.arguments().params["speed"]["kind"] == "runtime"


def test_sealed_compiled_model_drops_module_registry_and_authority_graph():
    class _Canary:
        def __call__(self, _operation):
            return None

    def build():
        module = _module_with_derived_parameter()
        registry = module.param_registry()

        # ParamRegistry is slots-only and not weak-referenceable.  A weak-referenceable guard held
        # exclusively by it proves collection of that registry, while a shared authority canary
        # proves the original ParameterDeclaration objects are gone as well.
        registry_canary = _Canary()
        authority_canary = _Canary()
        object.__setattr__(registry, "_mutation_guard", registry_canary)
        object.__setattr__(registry, "_authority_token", authority_canary)
        for declaration in registry.declarations().values():
            object.__setattr__(declaration, "_authority_token", authority_canary)

        compiled = _compiled(module.params())
        compiled._seal()
        return compiled, {
            "module": weakref.ref(module),
            "registry": weakref.ref(registry_canary),
            "authority": weakref.ref(authority_canary),
        }

    compiled, refs = build()
    gc.collect()

    assert compiled.params["speed2"].to_data()["expression"] is not None
    assert {name: reference() for name, reference in refs.items()} == {
        "module": None,
        "registry": None,
        "authority": None,
    }


def test_projection_accepts_only_explicit_scalar_legacy_values():
    projected = compiled_parameters({"count": 3, "enabled": True, "weight": 0.25})

    assert projected["count"].kind == "const" and projected["count"].default == 3
    assert projected["enabled"].dtype == "Bool" and projected["enabled"].default is True
    assert projected["weight"].default == 0.25
    with pytest.raises(TypeError, match="bind_data"):
        compiled_parameters({"opaque": object()})
