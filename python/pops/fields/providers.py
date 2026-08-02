"""Typed field-solver providers.

The external route is deliberately one indivisible solver/topology pair.  A topology component is
never authored or resolved on its own: the exact pair crosses resolve -> compile -> bind as one
provider authority and is matched against the explicitly supplied ``resolve(components=...)`` set.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet, RequirementSet


_EXTERNAL_PROVIDER_ID = "pops.fields.external-field-solver"
_EXTERNAL_PROVIDER_VERSION = 2
_EXTERNAL_PROVIDER_INTERFACE = "pops.prepared-field-solver-provider@1"
_EXTERNAL_RESOLVER_ID = "pops.fields.external-field-solver.resolve@2"
_EXTERNAL_INSTALLER_ID = "pops.fields.external-field-solver.install@2"
_EXTERNAL_USE_POLICY_ID = "pops.fields.external-field-solver.use"
_EXTERNAL_USE_POLICY_VERSION = 3
_EXTERNAL_ADAPTER_ID = "pops.fields.external-field-solver.system-host-serial@1"
_EXTERNAL_HIERARCHY_POLICY = {
    "policy_id": "pops.field-hierarchy.level-local",
    "interface_version": 1,
    "option_schema": "pops.field-hierarchy.options.empty@1",
    "options": {},
}


def _external_adapter_capabilities() -> dict[str, Any]:
    """Return the exact proved adapter envelope, detached for public inspection."""
    return {
        "provider_id": _EXTERNAL_PROVIDER_ID,
        "provider_version": _EXTERNAL_PROVIDER_VERSION,
        "adapter_identity": _EXTERNAL_ADAPTER_ID,
        "targets": ["system"],
        "layout_kinds": ["uniform"],
        "max_levels": 1,
        "hierarchy_policies": [_EXTERNAL_HIERARCHY_POLICY["policy_id"]],
        # FieldSolver@2 can describe a level on each patch, but the installed adapter still owns
        # one System MultiFab.  Metadata capacity is not an executable AMR provider bridge.
        "abi_patch_level_metadata": True,
        "hierarchy_materialization": False,
        "amr_provider_bridge": False,
        "execution": "host-serial-multi-patch-batch",
        "components": ["FieldTopology@2", "FieldSolver@2"],
    }


def _external_provider_authority() -> dict[str, Any]:
    """Project the provider authority without capturing the process-local registry object."""
    return {
        "schema_version": 1,
        "interface": _EXTERNAL_PROVIDER_INTERFACE,
        "provider_id": _EXTERNAL_PROVIDER_ID,
        "version": _EXTERNAL_PROVIDER_VERSION,
        "resolver_id": _EXTERNAL_RESOLVER_ID,
        "installer_id": _EXTERNAL_INSTALLER_ID,
        "use_policy": {
            "policy_id": _EXTERNAL_USE_POLICY_ID,
            "version": _EXTERNAL_USE_POLICY_VERSION,
            "capabilities": _external_adapter_capabilities(),
        },
    }


def _declared_execution(component: Any) -> dict[str, bool]:
    variants = [
        row for row in component.component_manifest.target["variants"]
        if row["dimension"] == 2 and row["scalar"] == "float64"
    ]
    return {
        "host": any(row["device"] in ("cpu", "host") for row in variants),
        "mpi": any("mpi" in row["features"] for row in variants),
        "gpu": any(row["device"] not in ("cpu", "host") for row in variants),
    }


def _component_binding(component: Any, expected: Any, *, role: str) -> dict[str, Any]:
    from pops.external import ExternalComponent

    if type(component) is not ExternalComponent:
        raise TypeError(
            "ExternalFieldSolver.%s must be an exact pops.external.ExternalComponent" % role
        )
    interface = component.component_type.interface
    if interface != expected:
        raise TypeError(
            "ExternalFieldSolver.%s must implement exact interface %s@%d, got %s@%d"
            % (role, expected.uri, expected.version, interface.uri, interface.version)
        )
    expected.require_manifest(component.component_manifest)
    expected.resolve_native_target(component)
    parameters = component.to_data()["parameters"]
    try:
        json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "ExternalFieldSolver.%s parameters must be strict JSON values" % role
        ) from exc
    return {
        "component_id": component.component_manifest.component_id,
        "component_manifest_identity": component.component_manifest.manifest_digest.token,
        "source_package_identity": component.package_identity.token,
        "native_interface": interface.to_data(),
        "interface_version": interface.version,
        "parameters": parameters,
        "declared_execution": _declared_execution(component),
    }


class ExternalFieldSolver(Descriptor):
    """One authenticated external FieldSolver coupled to its FieldTopology authority.

    ``relative_tolerance``, ``absolute_tolerance`` and ``max_iterations`` are request controls of
    the generated ``FieldSolver`` ABI.  Package/component parameters remain owned independently by
    each :class:`~pops.external.ExternalComponent` and are prepared exactly once by the native
    loader.
    """

    category = "field_solver_provider"
    provider_id = "pops.fields.external-field-solver.v2"

    def __init__(
        self,
        *,
        solver: Any,
        topology: Any,
        relative_tolerance: float = 1.0e-8,
        absolute_tolerance: float = 0.0,
        max_iterations: int = 50,
    ) -> None:
        from pops import interfaces

        # Validate both values before mutating this descriptor: construction is one authoring
        # transaction and can never leave a solver without its topology authority.
        solver_binding = _component_binding(
            solver, interfaces.FieldSolver, role="solver")
        topology_binding = _component_binding(
            topology, interfaces.FieldTopology, role="topology")
        if solver_binding["component_id"] == topology_binding["component_id"]:
            raise ValueError(
                "ExternalFieldSolver requires distinct exact FieldSolver and FieldTopology "
                "components"
            )
        for name, value in (
            ("relative_tolerance", relative_tolerance),
            ("absolute_tolerance", absolute_tolerance),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("ExternalFieldSolver.%s must be a finite real" % name)
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("ExternalFieldSolver.%s must be finite and >= 0" % name)
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise TypeError("ExternalFieldSolver.max_iterations must be an integer")
        if max_iterations < 1:
            raise ValueError("ExternalFieldSolver.max_iterations must be >= 1")
        self.solver = solver
        self.topology = topology
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.max_iterations = max_iterations

    def component_bindings(self) -> tuple[dict[str, Any], dict[str, Any]]:
        from pops import interfaces

        return (
            _component_binding(self.topology, interfaces.FieldTopology, role="topology"),
            _component_binding(self.solver, interfaces.FieldSolver, role="solver"),
        )

    def options(self) -> dict[str, Any]:
        topology, solver = self.component_bindings()
        return {
            "topology": topology,
            "solver": solver,
            "request": {
                "relative_tolerance": self.relative_tolerance,
                "absolute_tolerance": self.absolute_tolerance,
                "max_iterations": self.max_iterations,
            },
        }

    def to_data(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "provider": _external_provider_authority(),
            "options": self.options(),
        }

    def requirements(self) -> RequirementSet:
        return RequirementSet({
            "external_components": True,
            "field_topology": True,
            "field_topology_contract": "uniform_cartesian_full_material_v1",
            "field_hierarchy_policy": _EXTERNAL_HIERARCHY_POLICY["policy_id"],
            "max_levels": 1,
            "host_execution": True,
        })

    def capabilities(self) -> CapabilitySet:
        topology, solver = self.component_bindings()
        declared = {
            name: topology["declared_execution"][name]
            and solver["declared_execution"][name]
            for name in ("host", "mpi", "gpu")
        }
        adapter = {"host": True, "mpi": False, "gpu": False}
        # The component pair may declare broader targets, but this concrete adapter intentionally
        # intersects them with the runtime facts it actually implements.  It passes host views and
        # does not yet publish an inter-rank topology-consensus proof, hence serial host is the sole
        # truthful route in v2.
        provider = _external_provider_authority()
        return CapabilitySet({
            "provider": provider,
            "adapter": provider["use_policy"]["capabilities"],
            "external_field_solver_v2": True,
            "topology_provenance": True,
            "topology_contract": "uniform_cartesian_full_material_v1",
            "execution_adapter": "host_serial_multi_patch_batch_v1",
            "supports_amr": False,
            "max_levels": 1,
            "hierarchy_policy": _EXTERNAL_HIERARCHY_POLICY["policy_id"],
            "host": declared["host"] and adapter["host"],
            "mpi": declared["mpi"] and adapter["mpi"],
            "gpu": declared["gpu"] and adapter["gpu"],
            "component_pair_declares_mpi": declared["mpi"],
            "component_pair_declares_gpu": declared["gpu"],
        })

    def _prepared_field_solver(self) -> tuple[Any, dict[str, Any]]:
        return _EXTERNAL_FIELD_SOLVER_PROVIDER, self.options()


class PreparedFieldSolver(Descriptor):
    """Generic descriptor backed by one registered field-solver provider.

    Registration alone is not a native implementation: a provider must own a real native installer
    and its exact component bindings.  This descriptor only records immutable authoring options.
    """

    category = "field_solver_provider"

    def __init__(self, provider: Any, **options: Any) -> None:
        from ._prepared_field_solver_registry import (
            PreparedFieldSolverProvider,
            prepared_field_solver_provider_by_resolver_id,
        )

        if type(provider) is not PreparedFieldSolverProvider:
            raise TypeError("PreparedFieldSolver requires an exact registered Provider")
        if prepared_field_solver_provider_by_resolver_id(provider.resolver_id) is not provider:
            raise ValueError("PreparedFieldSolver provider is not the registered authority")
        self.provider = provider
        self.provider_options = dict(options)

    @property
    def name(self) -> str:
        return self.provider.provider_id

    def options(self) -> dict[str, Any]:
        return dict(self.provider_options)

    def to_data(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "provider": self.provider.authority(),
            "options": self.options(),
        }

    def _prepared_field_solver(self) -> tuple[Any, dict[str, Any]]:
        return self.provider, self.options()


def _external_resolver(options, facts, where):
    from ._prepared_field_solver_registry import PreparedFieldSolverResolution

    _validate_external_facts(facts, where)
    if not isinstance(options, Mapping) or set(options) != {"topology", "solver", "request"}:
        raise TypeError("%s external field solver options have an invalid shape" % where)
    topology = options["topology"]
    solver = options["solver"]
    request = options["request"]
    if not isinstance(topology, Mapping) or not isinstance(solver, Mapping):
        raise TypeError("%s external field solver component bindings must be mappings" % where)
    expected_request = {"relative_tolerance", "absolute_tolerance", "max_iterations"}
    if not isinstance(request, Mapping) or set(request) != expected_request:
        raise TypeError("%s external field solver request has an invalid shape" % where)
    relative = _finite_nonnegative(
        request["relative_tolerance"], where="%s relative_tolerance" % where)
    absolute = _finite_nonnegative(
        request["absolute_tolerance"], where="%s absolute_tolerance" % where)
    maximum = request["max_iterations"]
    if type(maximum) is not int or maximum < 1 or maximum > (1 << 31) - 1:
        raise ValueError("%s max_iterations must be one positive native integer" % where)
    return PreparedFieldSolverResolution(
        {
            "schema_identity": "pops.external.field-solver-request@2",
            "provider_id": _EXTERNAL_PROVIDER_ID,
            "provider_version": _EXTERNAL_PROVIDER_VERSION,
            "adapter_identity": _EXTERNAL_ADAPTER_ID,
            "options": {
                "relative_tolerance": relative,
                "absolute_tolerance": absolute,
                "max_iterations": maximum,
            },
        },
        {
            "provider_id": "pops.external.field-topology",
            "version": 1,
            "adapter_identity": _EXTERNAL_ADAPTER_ID,
            "topology_identity": facts.layout["topology_identity"],
            "layout": {
                "kind": facts.layout["kind"],
                "levels": facts.layout["levels"],
            },
            "hierarchy_policy": dict(facts.hierarchy),
            "component": dict(topology),
        },
        (dict(topology), dict(solver)),
    )


def _finite_nonnegative(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a finite real" % where)
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("%s must be finite and nonnegative" % where)
    return result


def _validate_external_facts(facts: Any, where: str) -> None:
    hierarchy = facts.hierarchy
    requested_policy = hierarchy.get("policy_id", "<missing>")
    if facts.target != "system":
        raise ValueError(
            "%s provider %s has no AMR provider bridge: target=%r, layout=%r, levels=%r, "
            "hierarchy_policy=%r; FieldSolver@2 patch-level metadata is only a carrier until an "
            "AmrFieldSolverProvider adapter materializes and solves the complete hierarchy"
            % (
                where,
                _EXTERNAL_PROVIDER_ID,
                facts.target,
                facts.layout.get("kind"),
                facts.layout.get("levels"),
                requested_policy,
            )
        )
    if facts.layout.get("kind") != "uniform" or facts.layout.get("levels") != 1:
        raise ValueError(
            "%s provider %s adapter %s requires one uniform level, got kind=%r levels=%r"
            % (
                where,
                _EXTERNAL_PROVIDER_ID,
                _EXTERNAL_ADAPTER_ID,
                facts.layout.get("kind"),
                facts.layout.get("levels"),
            )
        )
    if (
        requested_policy != _EXTERNAL_HIERARCHY_POLICY["policy_id"]
        or hierarchy.get("interface_version") != _EXTERNAL_HIERARCHY_POLICY["interface_version"]
        or hierarchy.get("option_schema") != _EXTERNAL_HIERARCHY_POLICY["option_schema"]
        or dict(hierarchy.get("options", {})) != _EXTERNAL_HIERARCHY_POLICY["options"]
    ):
        raise ValueError(
            "%s provider %s adapter %s supports only hierarchy policy %s, got %r"
            % (
                where,
                _EXTERNAL_PROVIDER_ID,
                _EXTERNAL_ADAPTER_ID,
                _EXTERNAL_HIERARCHY_POLICY["policy_id"],
                requested_policy,
            )
        )
    if facts.layout.get("embedded_boundary") or facts.layout.get("adaptive"):
        raise ValueError(
            "%s external FieldSolver@2 requires a full-material non-adaptive topology" % where
        )
    if facts.operator.get("screened"):
        raise ValueError(
            "%s external FieldSolver@2 has no reaction-coefficient carrier" % where
        )
    if facts.boundary.get("dynamic") or facts.boundary.get("dependent"):
        raise ValueError(
            "%s external FieldSolver@2 carries only an immutable boundary contract" % where
        )
    if facts.nonlinear:
        raise ValueError(
            "%s external FieldSolver@2 has no shared nonlinear iterate/JVP protocol" % where
        )


def _validate_external_use(use, where):
    facts = use.facts
    _validate_external_facts(facts, where)
    bindings = use.resolution.component_bindings
    if len(bindings) != 2 or any(
        not binding.get("declared_execution", {}).get("host") for binding in bindings
    ):
        raise ValueError(
            "%s external field components require compatible 2D float64 CPU targets" % where
        )


def _install_external(context: Any, binding: Any) -> None:
    context.install_component(binding)


from ._prepared_field_solver_registry import (  # noqa: E402
    PreparedFieldSolverBinding as Binding,
    PreparedFieldSolverFacts as Facts,
    PreparedFieldSolverProvider as Provider,
    PreparedFieldSolverResolution as Resolution,
    PreparedFieldSolverUse as Use,
    PreparedFieldSolverUsePolicy as UsePolicy,
    register_prepared_field_solver_provider as register,
)


_EXTERNAL_FIELD_SOLVER_PROVIDER = register(Provider(
    provider_id=_EXTERNAL_PROVIDER_ID,
    version=_EXTERNAL_PROVIDER_VERSION,
    resolver_id=_EXTERNAL_RESOLVER_ID,
    installer_id=_EXTERNAL_INSTALLER_ID,
    use_policy=UsePolicy(
        _EXTERNAL_USE_POLICY_ID,
        _EXTERNAL_USE_POLICY_VERSION,
        _external_adapter_capabilities(),
        _validate_external_use,
    ),
    resolver=_external_resolver,
    native_installer=_install_external,
))
if _EXTERNAL_FIELD_SOLVER_PROVIDER.authority() != _external_provider_authority():
    raise RuntimeError("external field solver provider authority projection is inconsistent")


__all__ = [
    "Binding",
    "ExternalFieldSolver",
    "Facts",
    "PreparedFieldSolver",
    "Provider",
    "Resolution",
    "Use",
    "UsePolicy",
    "register",
]
