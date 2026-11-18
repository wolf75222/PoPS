"""Typed external AMR provider bindings.

These values select one exact native component at authoring time.  They retain no
Python callback: resolve authenticates the component, then compile carries only the immutable
native binding record to installation. Reflux is the only native AMR provider role.
The clustering policy carries authenticated core controls independently of that ABI.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from pops.identity import make_identity


@dataclass(frozen=True, slots=True)
class AMRProviderLoweringContext:
    """Resolved facts available to every open AMR provider authority.

    A reflux provider authenticates the Program clock and the layout.
    """

    layout_identity: str
    components: Any
    clock_identity: str

    def __post_init__(self) -> None:
        for name in ("layout_identity", "clock_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError("AMR provider lowering context requires a non-empty %s" % name)


@dataclass(frozen=True, slots=True)
class ResolvedAMRProviderBinding:
    """One role-qualified, data-only provider binding returned by an open authority."""

    role: str
    data: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role:
            raise TypeError("resolved AMR provider binding requires a non-empty role")
        if type(self.data) is not dict:
            raise TypeError("resolved AMR provider binding data must be an exact dict")
        installation = self.data.get("runtime_installation")
        if self.data.get("schema_version") != 1 \
                or not isinstance(self.data.get("provider_identity"), str) \
                or not self.data["provider_identity"] \
                or not isinstance(self.data.get("layout_identity"), str) \
                or not self.data["layout_identity"] \
                or not isinstance(self.data.get("native_interface"), Mapping) \
                or not isinstance(installation, Mapping) \
                or set(installation) != {"schema_version", "protocol"} \
                or installation.get("schema_version") != 1 \
                or installation.get("protocol") not in {
                    "builtin", "external_component"
                }:
            raise TypeError(
                "resolved AMR provider binding is incomplete or lacks its runtime protocol"
            )
        object.__setattr__(
            self,
            "data",
            validate_amr_provider_binding(
                role=self.role,
                frozen_binding=self.data,
                layout_identity=self.data["layout_identity"],
            ),
        )


@dataclass(frozen=True, slots=True)
class PreparedAMRProviderInstallation:
    """Validated intrinsic AMR authority retained by the native runtime."""

    role: str
    binding: dict[str, Any]


def _external_component(value: Any, *, interface: Any, where: str) -> Any:
    from pops.external import ExternalComponent

    if type(value) is not ExternalComponent:
        raise TypeError("%s requires an exact pops.external.ExternalComponent" % where)
    actual = value.component_type.interface
    if actual != interface:
        raise TypeError(
            "%s requires exact interface %s@%d, got %s@%d"
            % (where, interface.uri, interface.version, actual.uri, actual.version)
        )
    interface.require_manifest(value.component_manifest)
    interface.native_target_variants(value)
    determinism = value.component_manifest.determinism.get("classification")
    if determinism not in {"bitwise", "reproducible"}:
        raise ValueError(
            "%s requires determinism.classification 'bitwise' or 'reproducible'; "
            "every rank must derive the same hierarchy" % where
        )
    return value


def _require_component(component: Any, values: Any, *, where: str) -> None:
    from pops.external import ExternalComponent

    rows = tuple(values)
    matches = [
        value for value in rows
        if type(value) is ExternalComponent
        and value.component_manifest.component_id
        == component.component_manifest.component_id
    ]
    if len(matches) != 1 or matches[0].to_data() != component.to_data():
        raise ValueError(
            "%s requires its exact ExternalComponent in pops.resolve(..., components=...)"
            % where
        )


def _component_binding(component: Any, interface: Any) -> dict[str, Any]:
    return {
        "component_id": component.component_manifest.component_id,
        "component_manifest_identity": component.component_manifest.manifest_digest.token,
        "component": component.to_data(),
        "native_interface": interface.to_data(),
        "interface_version": interface.version,
    }


@dataclass(frozen=True, slots=True)
class RefluxProvider:
    """Bind one external local Reflux table to the conservative AMR transition."""

    component: Any
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        from pops import interfaces

        _external_component(
            self.component,
            interface=interfaces.Reflux,
            where="RefluxProvider.component",
        )

    def resolve_references(self, resolver: Any) -> RefluxProvider:
        if not callable(resolver):
            raise TypeError("RefluxProvider.resolve_references requires a callable resolver")
        return self

    def require_component_inputs(self, components: Any) -> None:
        _require_component(self.component, components, where="RefluxProvider")

    def lower_amr_provider(
        self, context: AMRProviderLoweringContext,
    ) -> ResolvedAMRProviderBinding:
        """Authenticate the component, hierarchy layout and Program clock."""
        if type(context) is not AMRProviderLoweringContext:
            raise TypeError("RefluxProvider requires an AMRProviderLoweringContext")
        self.require_component_inputs(context.components)
        data = {
            **self.runtime_binding_data(),
            "layout_identity": context.layout_identity,
            "clock_identity": context.clock_identity,
        }
        data["provider_identity"] = amr_provider_binding_identity("reflux", data)
        return ResolvedAMRProviderBinding("reflux", data)

    def runtime_binding_data(self) -> dict[str, Any]:
        from pops import interfaces

        data = {
            "schema_version": 1,
            "provider_type": "external_amr_reflux",
            "runtime_installation": {
                "schema_version": 1,
                "protocol": "external_component",
            },
            **_component_binding(self.component, interfaces.Reflux),
        }
        data["provider_identity"] = make_identity("amr-reflux-provider", data).token
        return data

    inspect = runtime_binding_data
    canonical_identity = runtime_binding_data


@dataclass(frozen=True, slots=True)
class _AMRRuntimeInterfaceProtocol:
    """Native-interface-owned validation and installation route."""

    role: str
    native_interface: Mapping[str, Any]
    builtin_provider_id: str

    @property
    def resolved_identity_namespace(self) -> str:
        return "resolved-amr-%s-provider" % self.role

    def validate_resolved_capability(self, binding: Mapping[str, Any]) -> None:
        del binding


@dataclass(frozen=True, slots=True)
class _RefluxRuntimeInterfaceProtocol(_AMRRuntimeInterfaceProtocol):
    """The local Reflux callback is qualified by the accepted Program clock."""

    def validate_resolved_capability(self, binding: Mapping[str, Any]) -> None:
        if not isinstance(binding.get("clock_identity"), str) \
                or not binding["clock_identity"]:
            raise ValueError("AMR Reflux lacks its exact Program clock authority")


def _runtime_interface_key(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, Mapping):
        raise TypeError("AMR provider binding has no native-interface protocol")
    return (
        value.get("uri"), value.get("version"), value.get("catalog_sha256"),
        value.get("protocol_abi"), value.get("cpp_table"),
    )


def _runtime_interface_matches(value: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(value, Mapping):
        return False
    from pops.identity.semantic import semantic_value

    return semantic_value(value, where="AMR provider native interface") == semantic_value(
        expected, where="AMR provider protocol interface")


def _runtime_interface_protocols() -> dict[tuple[Any, ...], _AMRRuntimeInterfaceProtocol]:
    from pops import interfaces

    protocols: tuple[_AMRRuntimeInterfaceProtocol, ...] = (
        _RefluxRuntimeInterfaceProtocol(
            role="reflux",
            native_interface=interfaces.Reflux.to_data(),
            builtin_provider_id="pops.lib.amr::flux_register_reflux",
        ),
    )
    return {_runtime_interface_key(row.native_interface): row for row in protocols}


def require_builtin_clustering(value: Any) -> Any:
    from pops.lib.amr import BergerRigoutsos

    if type(value) is not BergerRigoutsos:
        raise TypeError("AMR.clustering requires the built-in BergerRigoutsos policy")
    return value


def resolve_clustering_binding(value: Any, layout_identity: str) -> dict[str, Any]:
    binding = {**require_builtin_clustering(value).runtime_binding_data(),
               "layout_identity": layout_identity}
    binding["provider_identity"] = amr_provider_binding_identity("clustering", binding)
    return binding


def _clustering_efficiency(value: Any) -> float:
    """Decode the same finite binary64 control recorded in CanonicalOptions."""
    try:
        if (isinstance(value, Mapping) and set(value) == {"binary64"}
                and isinstance(value["binary64"], str)):
            decoded = float.fromhex(value["binary64"])
            if decoded.hex() != value["binary64"]:
                raise ValueError("AMR clustering efficiency must be canonical binary64")
        elif not isinstance(value, bool) and isinstance(value, (int, float)):
            decoded = float(value)
        else:
            raise TypeError("AMR clustering efficiency must be canonical binary64")
    except OverflowError as exc:
        raise ValueError("AMR clustering efficiency must be in (0, 1]") from exc
    if not isfinite(decoded) or not 0.0 < decoded <= 1.0:
        raise ValueError("AMR clustering efficiency must be in (0, 1]")
    return decoded


def clustering_controls(
    frozen_binding: Any, *, layout_identity: str,
) -> tuple[float, int, int]:
    """Verify the whole intrinsic policy before lowering any native control."""
    fields = {
        "schema_version", "provider_type", "provider_id", "provider_identity",
        "runtime_installation", "layout_identity",
        "minimum_efficiency", "minimum_box_size", "maximum_box_size",
    }
    if not isinstance(frozen_binding, Mapping) or set(frozen_binding) != fields:
        raise ValueError("AMR clustering binding must contain exactly the built-in policy fields")
    binding = frozen_binding
    installation = binding["runtime_installation"]
    if (type(binding["schema_version"]) is not int or binding["schema_version"] != 1
            or binding["provider_type"] != "builtin_amr_clustering"
            or binding["provider_id"] != "pops.lib.amr::berger_rigoutsos"
            or not isinstance(layout_identity, str) or not layout_identity
            or binding["layout_identity"] != layout_identity
            or not isinstance(installation, Mapping)
            or set(installation) != {"schema_version", "protocol"}
            or type(installation["schema_version"]) is not int
            or installation["schema_version"] != 1
            or installation["protocol"] != "builtin"):
        raise ValueError("AMR clustering binding is not the built-in policy of this layout")
    efficiency = _clustering_efficiency(binding["minimum_efficiency"])
    minimum = binding["minimum_box_size"]
    maximum = binding["maximum_box_size"]
    if type(minimum) is not int or type(maximum) is not int or not 1 <= minimum <= maximum:
        raise ValueError("AMR clustering box-size controls are invalid")
    if binding["provider_identity"] != amr_provider_binding_identity("clustering", binding):
        raise ValueError("AMR clustering provider_identity does not authenticate its policy")
    return efficiency, minimum, maximum


def amr_provider_binding_identity(role: str, data: Mapping[str, Any]) -> str:
    """Return the canonical identity for one resolved third-party provider binding.

    The clustering policy has an intrinsic namespace. A Reflux extension takes the namespace
    of its exact native interface.
    """
    if not isinstance(role, str) or not role or not isinstance(data, Mapping):
        raise TypeError("AMR provider identity requires one role and one binding mapping")
    if role == "clustering":
        if "native_interface" in data:
            raise ValueError("AMR clustering policy has no native substitution interface")
        namespace = "resolved-amr-clustering-provider"
    else:
        protocol = _runtime_interface_protocols().get(
            _runtime_interface_key(data.get("native_interface")))
        if protocol is None or protocol.role != role \
                or not _runtime_interface_matches(
                    data.get("native_interface"), protocol.native_interface):
            raise ValueError("AMR provider identity selects an unsupported role/interface")
        namespace = protocol.resolved_identity_namespace
    from pops.identity.semantic import semantic_value

    payload = {key: value for key, value in data.items() if key != "provider_identity"}
    return make_identity(
        namespace,
        semantic_value(payload, where="resolved AMR %s provider" % role),
    ).token


def _validate_builtin_binding(
    protocol: _AMRRuntimeInterfaceProtocol,
    binding: Mapping[str, Any],
    component_inputs: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    del component_inputs
    if binding.get("provider_type") != "builtin_amr_%s" % protocol.role \
            or binding.get("provider_id") != protocol.builtin_provider_id \
            or any(name in binding for name in (
                "component_id", "component_manifest_identity", "component",
                "interface_version",
            )):
        raise ValueError("builtin AMR %s provider is not canonical" % protocol.role)


def _validate_external_binding(
    protocol: _AMRRuntimeInterfaceProtocol,
    binding: Mapping[str, Any],
    component_inputs: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    component_id = binding.get("component_id")
    component = binding.get("component")
    manifest_identity = binding.get("component_manifest_identity")
    if binding.get("provider_type") != "external_amr_%s" % protocol.role \
            or not isinstance(component_id, str) or not component_id \
            or not isinstance(manifest_identity, str) or not manifest_identity \
            or binding.get("interface_version") != protocol.native_interface.get("version") \
            or not isinstance(component, Mapping) \
            or component.get("component_id") != component_id \
            or component.get("component_manifest") != manifest_identity \
            or not _runtime_interface_matches(
                component.get("interface"), protocol.native_interface) \
            or "provider_id" in binding:
        raise ValueError("external AMR %s provider lost exact component identity" % protocol.role)
    if component_inputs is not None:
        from pops.identity.semantic import semantic_value

        installed = component_inputs.get(component_id)
        if installed is None or semantic_value(
                installed, where="AMR component authority") != semantic_value(
                    component, where="AMR provider component authority"):
            raise ValueError(
                "external AMR %s provider differs from its component authority"
                % protocol.role)
    if binding.get("provider_identity") != amr_provider_binding_identity(
            protocol.role, binding):
        raise ValueError(
            "AMR %s provider_identity does not authenticate its resolved authority"
            % protocol.role)
    raise NotImplementedError(
        "external AMR reflux component installation requires a complete atomic AMR "
        "provider-pack authority; no external component installer is published. "
        "The executable authority remains the transactional AmrRuntime<Dim> reflux ledger"
    )


_BINDING_PROTOCOLS: dict[
    str,
    Callable[
        [_AMRRuntimeInterfaceProtocol, Mapping[str, Any],
         Mapping[str, Mapping[str, Any]] | None],
        None,
    ],
] = {
    "builtin": _validate_builtin_binding,
    "external_component": _validate_external_binding,
}


def validate_amr_provider_binding(
    *,
    role: str,
    frozen_binding: Any,
    layout_identity: str,
    component_inputs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Authenticate detached intrinsic controls or the exact Reflux provider protocol."""
    if not isinstance(frozen_binding, Mapping):
        raise TypeError("AMR %s provider binding must be a mapping" % role)
    binding = dict(frozen_binding)
    if role == "clustering":
        clustering_controls(binding, layout_identity=layout_identity)
        return binding
    protocol = _runtime_interface_protocols().get(
        _runtime_interface_key(binding.get("native_interface")))
    if protocol is None or protocol.role != role \
            or not _runtime_interface_matches(
                binding.get("native_interface"), protocol.native_interface):
        raise ValueError("AMR provider binding role disagrees with its native interface")
    installation = binding.get("runtime_installation")
    installation_protocol = (
        installation.get("protocol") if isinstance(installation, Mapping) else None)
    if binding.get("schema_version") != 1 \
            or not isinstance(binding.get("provider_identity"), str) \
            or not binding["provider_identity"] \
            or binding.get("layout_identity") != layout_identity \
            or not isinstance(installation, Mapping) \
            or set(installation) != {"schema_version", "protocol"} \
            or installation.get("schema_version") != 1 \
            or not isinstance(installation_protocol, str) \
            or not installation_protocol:
        raise ValueError("AMR %s provider binding is incomplete or unauthenticated" % role)
    validator = _BINDING_PROTOCOLS.get(installation_protocol)
    if validator is None:
        raise NotImplementedError("AMR %s provider binding protocol is not implemented" % role)
    validator(protocol, binding, component_inputs)
    protocol.validate_resolved_capability(binding)
    if binding["provider_identity"] != amr_provider_binding_identity(role, binding):
        raise ValueError(
            "AMR %s provider_identity does not authenticate its resolved authority" % role)
    return binding


def _prepare_builtin_provider(
    protocol: _AMRRuntimeInterfaceProtocol,
    binding: dict[str, Any],
    **_: Any,
) -> PreparedAMRProviderInstallation:
    if binding.get("provider_id") != protocol.builtin_provider_id \
            or any(name in binding for name in (
                "component_id", "component_manifest_identity", "component")):
        raise ValueError("builtin AMR %s provider is not canonical" % protocol.role)
    return PreparedAMRProviderInstallation(protocol.role, binding)


_INSTALLATION_PROTOCOLS: dict[str, Callable[..., PreparedAMRProviderInstallation]] = {
    "builtin": _prepare_builtin_provider,
}


def prepare_amr_provider_installation(
    *,
    role: str,
    frozen_binding: Any,
    layout_identity: str,
) -> PreparedAMRProviderInstallation:
    """Retain exact core clustering controls or prepare the Reflux installation protocol."""
    if not isinstance(frozen_binding, Mapping):
        raise TypeError("AMR %s provider binding must be an immutable mapping" % role)
    binding = validate_amr_provider_binding(
        role=role,
        frozen_binding=frozen_binding,
        layout_identity=layout_identity,
    )
    if role == "clustering":
        return PreparedAMRProviderInstallation(role, binding)
    protocol = _runtime_interface_protocols()[
        _runtime_interface_key(binding.get("native_interface"))]
    installation = binding.get("runtime_installation")
    installation_protocol = (
        installation.get("protocol") if isinstance(installation, Mapping) else None)
    if not isinstance(installation, Mapping) \
            or set(installation) != {"schema_version", "protocol"} \
            or installation.get("schema_version") != 1 \
            or not isinstance(installation_protocol, str) \
            or not installation_protocol:
        raise ValueError("AMR %s provider lacks its exact runtime protocol" % role)
    lowering = _INSTALLATION_PROTOCOLS.get(installation_protocol)
    if lowering is None:
        raise NotImplementedError(
            "AMR %s provider runtime protocol is not implemented" % role)
    return lowering(
        protocol,
        binding,
    )


__all__ = [
    "AMRProviderLoweringContext",
    "amr_provider_binding_identity",
    "RefluxProvider",
    "ResolvedAMRProviderBinding",
    "validate_amr_provider_binding",
]
