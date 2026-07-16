"""Strict authoring-reference boundary for :class:`CompiledModel`."""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Any


_ATOMIC = (type(None), bool, int, float, complex, str, bytes, Decimal, Fraction, Enum)
_SEQUENCE_FIELDS = (
    "cons_names", "cons_roles", "prim_names", "aux_extra_names", "elliptic_field_names",
    "state_spaces",
)
_SCALAR_FIELDS = (
    "has_hllc", "has_roe", "has_wave_speeds", "so_path", "backend", "target",
    "n_vars", "gamma", "n_aux", "abi_key", "model_hash", "cxx", "std",
    "wave_speed_provider",
)
_CORE_FIELDS = set(_SEQUENCE_FIELDS) | set(_SCALAR_FIELDS) | {
    "params", "caps", "bind_schema", "install_plan", "definition_identity",
    "module_manifest",
    "semantic_identity", "artifact_spec_identity", "binary_identity", "artifact_identity",
}


def validate_compiled_model_result(compiled: Any, *, allow_install_plan: bool = False) -> None:
    """Reject hidden authoring objects immediately after ``model.compile()`` returns."""
    _validate_core(compiled, allow_install_plan=allow_install_plan)
    for name, value in _stored_items(type(compiled), compiled):
        if name in _CORE_FIELDS or name in ("_sealed",):
            continue
        if name == "_problem_snapshot":
            _require_snapshot(value)
            continue
        if name == "_runtime_param_names":
            _write_stored(compiled, name, _string_tuple(value, name))
            continue
        _write_stored(compiled, name, _detach_extension(value, where=name))


def seal_compiled_model(compiled: Any) -> None:
    """Normalize every retained field and prove no live authoring graph survives."""
    validate_compiled_model_result(compiled, allow_install_plan=True)
    for name in _SEQUENCE_FIELDS:
        object.__setattr__(compiled, name, _string_tuple(_core_value(compiled, name), name))
    object.__setattr__(
        compiled, "caps", _data_mapping(_core_value(compiled, "caps"), where="caps"))
    identity = _core_value(compiled, "definition_identity")
    if identity is not None:
        from pops.codegen._compiled_model_identity import validate_compiled_model_identity

        object.__setattr__(
            compiled, "definition_identity", validate_compiled_model_identity(identity))


def _validate_core(compiled: Any, *, allow_install_plan: bool) -> None:
    for name in _SCALAR_FIELDS:
        value = _core_value(compiled, name)
        if not isinstance(value, _ATOMIC):
            raise TypeError(
                "CompiledModel.%s contains non-data value %s" % (name, type(value).__name__))
    for name in _SEQUENCE_FIELDS:
        _string_tuple(_core_value(compiled, name), name)
    has_wave_speeds = _core_value(compiled, "has_wave_speeds")
    wave_speed_provider = _core_value(compiled, "wave_speed_provider")
    allowed_wave_speed_providers = {"explicit_pair", "jacobian", "pressure_derived"}
    if has_wave_speeds and wave_speed_provider not in allowed_wave_speed_providers:
        raise ValueError(
            "CompiledModel with wave speeds requires an exact detached wave_speed_provider"
        )
    if not has_wave_speeds and wave_speed_provider is not None:
        raise ValueError(
            "CompiledModel without wave speeds cannot retain wave_speed_provider"
        )
    _data_mapping(_core_value(compiled, "caps"), where="caps")
    identity = _core_value(compiled, "definition_identity")
    if identity is not None:
        from pops.codegen._compiled_model_identity import validate_compiled_model_identity

        validate_compiled_model_identity(identity)
    module_manifest = _core_value(compiled, "module_manifest")
    if module_manifest is not None:
        from pops.model import ModuleManifest

        if type(module_manifest) is not ModuleManifest:
            raise TypeError("CompiledModel.module_manifest must be an exact ModuleManifest")
        if identity is None or identity.get("module_hash") is None:
            raise ValueError(
                "CompiledModel.module_manifest requires an authenticated module_hash")
        manifest_abi = module_manifest.abi_requirements.get("abi_key")
        if manifest_abi != _core_value(compiled, "abi_key"):
            raise ValueError(
                "CompiledModel.module_manifest ABI key disagrees with the compiled model")
    from pops.identity import Identity
    expected_identity_domains = {
        "semantic_identity": "semantic",
        "artifact_spec_identity": "artifact-spec",
        "binary_identity": "binary",
        "artifact_identity": "artifact",
    }
    for name, domain in expected_identity_domains.items():
        value = _core_value(compiled, name)
        if value is not None and (not isinstance(value, Identity) or value.domain != domain):
            raise TypeError("CompiledModel.%s must be a pops.%s Identity" % (name, domain))
    bind_schema = _core_value(compiled, "bind_schema")
    if bind_schema is not None:
        from pops.model.bind_schema import BindSchema

        if type(bind_schema) is not BindSchema:
            raise TypeError("CompiledModel.bind_schema must be an immutable BindSchema")
    install_plan = _core_value(compiled, "install_plan")
    if install_plan is not None:
        if not allow_install_plan:
            raise TypeError(
                "model.compile() returned a CompiledModel with a pre-attached InstallPlan; "
                "only pops.compile may create bind authority")
        from pops.codegen._plans import InstallPlan

        if type(install_plan) is not InstallPlan:
            raise TypeError("CompiledModel.install_plan must be an immutable InstallPlan")


def _detach_extension(value: Any, *, where: str) -> Any:
    if isinstance(value, _ATOMIC):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({
            _detach_extension(key, where=where): _detach_extension(item, where=where)
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_detach_extension(item, where=where) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_detach_extension(item, where=where) for item in value)
    from pops.model.handles import Handle, OwnerPath

    if isinstance(value, (Handle, OwnerPath)):
        raise TypeError(
            "CompiledModel.%s cannot retain Handle/OwnerPath authoring identity" % where)
    freeze = getattr(value, "freeze", None)
    if not callable(freeze):
        raise TypeError(
            "CompiledModel.%s contains unsupported live object %s.%s"
            % (where, type(value).__module__, type(value).__qualname__))
    from pops.problem._detached import detached_frozen

    return detached_frozen(value)


def _data_mapping(value: Any, *, where: str) -> Any:
    if not isinstance(value, Mapping):
        raise TypeError("CompiledModel.%s must be a mapping" % where)
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError("CompiledModel.%s keys must be non-empty strings" % where)
        if not isinstance(item, _ATOMIC):
            raise TypeError(
                "CompiledModel.%s[%r] contains non-data value %s"
                % (where, key, type(item).__name__))
        result[key] = item
    return MappingProxyType(result)


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    try:
        result = tuple(value)
    except TypeError:
        raise TypeError("CompiledModel.%s must be a sequence of strings" % name) from None
    if any(not isinstance(item, str) for item in result):
        raise TypeError("CompiledModel.%s must contain only strings" % name)
    return result


def _require_snapshot(value: Any) -> None:
    from pops.problem._snapshot import AuthoringSnapshot

    if value is None:
        return
    if type(value) is not AuthoringSnapshot:
        raise TypeError("CompiledModel._problem_snapshot must be an AuthoringSnapshot")


def _stored_items(cls: type, value: Any) -> tuple[tuple[str, Any], ...]:
    data = object.__getattribute__(value, "__dict__")
    items = list(data.items())
    names = set(data)
    for owner in cls.__mro__:
        slots = owner.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name.startswith("__") and not name.endswith("__"):
                name = "_%s%s" % (owner.__name__.lstrip("_"), name)
            if name in names or name in ("__dict__", "__weakref__"):
                continue
            descriptor = owner.__dict__.get(name)
            if descriptor is None:
                continue
            try:
                stored = descriptor.__get__(value, cls)
            except AttributeError:
                continue
            names.add(name)
            items.append((name, stored))
    return tuple(items)


def _core_value(compiled: Any, name: str) -> Any:
    data = object.__getattribute__(compiled, "__dict__")
    if name not in data:
        raise TypeError("CompiledModel is missing required stored field %r" % name)
    return data[name]


def _write_stored(compiled: Any, name: str, value: Any) -> None:
    data = object.__getattribute__(compiled, "__dict__")
    if name in data:
        data[name] = value
        return
    for owner in type(compiled).__mro__:
        descriptor = owner.__dict__.get(name)
        if descriptor is not None and hasattr(descriptor, "__set__"):
            descriptor.__set__(compiled, value)
            return
    raise AttributeError("CompiledModel stored field %r disappeared during validation" % name)


__all__ = ["seal_compiled_model", "validate_compiled_model_result"]
