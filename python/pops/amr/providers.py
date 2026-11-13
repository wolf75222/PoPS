"""Typed external AMR provider bindings.

These values select one exact native component at authoring time.  They retain no
Python callback: resolve authenticates the component, then compile carries only the immutable
native binding record to installation.  The roles are clustering and reflux; the tagger role went
with the tag graph (N12 lot 3c-2b), since a refinement criterion is generated into the Program.
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

    Providers decide which facts they consume: a reflux provider authenticates the Program
    clock while a clustering provider remains independent of it; the resolver never needs to name
    either concrete implementation.
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


@dataclass(frozen=True, slots=True)
class PreparedAMRProviderNativeConfig:
    """Provider-owned bind-time config plus options retained for component preparation."""

    role: str
    config: dict[str, Any]
    provider_options: dict[str, Any]


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
class ClusteringProvider:
    """Bind one external Clustering table to the AMR hierarchy authority."""

    component: Any
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        from pops import interfaces

        _external_component(
            self.component,
            interface=interfaces.Clustering,
            where="ClusteringProvider.component",
        )

    def resolve_references(self, resolver: Any) -> ClusteringProvider:
        if not callable(resolver):
            raise TypeError("ClusteringProvider.resolve_references requires a callable resolver")
        return self

    def require_component_inputs(self, components: Any) -> None:
        _require_component(self.component, components, where="ClusteringProvider")

    def lower_amr_provider(
        self, context: AMRProviderLoweringContext,
    ) -> ResolvedAMRProviderBinding:
        """Authenticate the component and emit its role-qualified detached binding."""
        if type(context) is not AMRProviderLoweringContext:
            raise TypeError("ClusteringProvider requires an AMRProviderLoweringContext")
        self.require_component_inputs(context.components)
        data = {
            **self.runtime_binding_data(),
            "layout_identity": context.layout_identity,
        }
        data["provider_identity"] = amr_provider_binding_identity("clustering", data)
        return ResolvedAMRProviderBinding("clustering", data)

    def runtime_binding_data(self) -> dict[str, Any]:
        from pops import interfaces

        data = {
            "schema_version": 1,
            "provider_type": "external_amr_clustering",
            "runtime_installation": {
                "schema_version": 1,
                "protocol": "external_component",
            },
            **_component_binding(self.component, interfaces.Clustering),
        }
        data["provider_identity"] = make_identity("amr-clustering-provider", data).token
        return data

    inspect = runtime_binding_data
    canonical_identity = runtime_binding_data


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

    def builtin_native_config(self, binding: Mapping[str, Any]) -> dict[str, Any]:
        del binding
        return {}


@dataclass(frozen=True, slots=True)
class _ClusteringRuntimeInterfaceProtocol(_AMRRuntimeInterfaceProtocol):
    """Clustering controls that belong to the builtin native implementation."""

    def validate_resolved_capability(self, binding: Mapping[str, Any]) -> None:
        installation = binding.get("runtime_installation")
        if not isinstance(installation, Mapping) \
                or installation.get("protocol") != "builtin":
            return
        efficiency = binding.get("minimum_efficiency")
        minimum = binding.get("minimum_box_size")
        maximum = binding.get("maximum_box_size")
        if isinstance(efficiency, Mapping) and set(efficiency) == {"binary64"} \
                and isinstance(efficiency.get("binary64"), str):
            try:
                decoded_efficiency = float.fromhex(efficiency["binary64"])
            except ValueError:
                decoded_efficiency = None
            if decoded_efficiency is not None \
                    and decoded_efficiency.hex() != efficiency["binary64"]:
                decoded_efficiency = None
        elif not isinstance(efficiency, bool) and isinstance(efficiency, (int, float)):
            decoded_efficiency = float(efficiency)
        else:
            decoded_efficiency = None
        if decoded_efficiency is None or not isfinite(decoded_efficiency) \
                or not 0.0 < decoded_efficiency <= 1.0:
            raise ValueError("builtin AMR clustering minimum_efficiency must be in (0, 1]")
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1 \
                or isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1 \
                or minimum > maximum:
            raise ValueError("builtin AMR clustering box-size controls are invalid")

    def builtin_native_config(self, binding: Mapping[str, Any]) -> dict[str, Any]:
        required = ("minimum_efficiency", "minimum_box_size", "maximum_box_size")
        if any(name not in binding for name in required):
            raise TypeError("builtin AMR clustering binding lacks exact algorithm controls")
        return {
            "cluster_min_efficiency": binding["minimum_efficiency"],
            "cluster_min_box_size": binding["minimum_box_size"],
            "cluster_max_box_size": binding["maximum_box_size"],
        }


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
        _ClusteringRuntimeInterfaceProtocol(
            role="clustering",
            native_interface=interfaces.Clustering.to_data(),
            builtin_provider_id="pops.lib.amr::berger_rigoutsos",
        ),
        _RefluxRuntimeInterfaceProtocol(
            role="reflux",
            native_interface=interfaces.Reflux.to_data(),
            builtin_provider_id="pops.lib.amr::flux_register_reflux",
        ),
    )
    return {_runtime_interface_key(row.native_interface): row for row in protocols}


def amr_provider_binding_identity(role: str, data: Mapping[str, Any]) -> str:
    """Return the canonical identity for one resolved third-party provider binding.

    The namespace belongs to the selected native interface.  Extension providers use this
    helper instead of reproducing PoPS identity details or inventing a provider-name branch.
    """
    if not isinstance(role, str) or not role or not isinstance(data, Mapping):
        raise TypeError("AMR provider identity requires one role and one binding mapping")
    protocol = _runtime_interface_protocols().get(
        _runtime_interface_key(data.get("native_interface")))
    if protocol is None or protocol.role != role \
            or not _runtime_interface_matches(
                data.get("native_interface"), protocol.native_interface):
        raise ValueError("AMR provider identity selects an unsupported role/interface")
    from pops.identity.semantic import semantic_value

    payload = {key: value for key, value in data.items() if key != "provider_identity"}
    return make_identity(
        protocol.resolved_identity_namespace,
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
    executable_authority = {
        "clustering": "BergerRigoutsosProvider<Dim>",
        "reflux": "the transactional AmrRuntime<Dim> reflux ledger",
    }[protocol.role]
    raise NotImplementedError(
        "external AMR %s component installation requires a complete atomic AMR "
        "provider-pack authority; no external component installer is published. "
        "The executable authority remains %s"
        % (protocol.role, executable_authority)
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
    """Authenticate detached provider data through its interface and installation protocols."""
    if not isinstance(frozen_binding, Mapping):
        raise TypeError("AMR %s provider binding must be a mapping" % role)
    binding = dict(frozen_binding)
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


def _builtin_native_config(
    protocol: _AMRRuntimeInterfaceProtocol,
    binding: dict[str, Any],
) -> PreparedAMRProviderNativeConfig:
    if binding.get("provider_id") != protocol.builtin_provider_id:
        raise ValueError("builtin AMR %s provider is not canonical" % protocol.role)
    return PreparedAMRProviderNativeConfig(
        protocol.role, protocol.builtin_native_config(binding), {})


_NATIVE_CONFIG_PROTOCOLS: dict[
    str, Callable[[_AMRRuntimeInterfaceProtocol, dict[str, Any]],
                  PreparedAMRProviderNativeConfig]
] = {
    "builtin": _builtin_native_config,
}


def prepare_amr_provider_native_config(
    frozen_binding: Any,
) -> PreparedAMRProviderNativeConfig:
    """Prepare bind-time config without dispatching on concrete provider names."""
    if not isinstance(frozen_binding, Mapping):
        raise TypeError("AMR provider native config requires one binding mapping")
    raw = dict(frozen_binding)
    protocols = _runtime_interface_protocols()
    protocol = protocols.get(_runtime_interface_key(raw.get("native_interface")))
    if protocol is None:
        raise ValueError("AMR provider native config selects an unsupported interface")
    layout_identity = raw.get("layout_identity")
    if not isinstance(layout_identity, str) or not layout_identity:
        raise ValueError("AMR provider native config requires one layout identity")
    binding = validate_amr_provider_binding(
        role=protocol.role,
        frozen_binding=raw,
        layout_identity=layout_identity,
    )
    installation = binding.get("runtime_installation")
    route = installation.get("protocol") if isinstance(installation, Mapping) else None
    if not isinstance(route, str) or not route:
        raise ValueError("AMR provider native config lacks its exact runtime protocol")
    lowering = _NATIVE_CONFIG_PROTOCOLS.get(route)
    if lowering is None:
        raise NotImplementedError("AMR provider native config protocol is not implemented")
    return lowering(protocol, binding)


def prepare_amr_provider_installation(
    *,
    role: str,
    frozen_binding: Any,
    layout_identity: str,
) -> PreparedAMRProviderInstallation:
    """Lower one detached provider through its native-interface protocol.

    The runtime does not infer behavior from the mapping slot or from ``provider_type``.  The
    authority carries an explicit installation protocol, while the exact native interface owns
    capability validation and the native installation route.
    """
    if not isinstance(frozen_binding, Mapping):
        raise TypeError("AMR %s provider binding must be an immutable mapping" % role)
    binding = validate_amr_provider_binding(
        role=role,
        frozen_binding=frozen_binding,
        layout_identity=layout_identity,
    )
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
    "ClusteringProvider",
    "PreparedAMRProviderNativeConfig",
    "RefluxProvider",
    "ResolvedAMRProviderBinding",
    "validate_amr_provider_binding",
]
