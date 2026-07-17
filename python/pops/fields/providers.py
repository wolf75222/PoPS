"""Typed field-solver providers.

The external route is deliberately one indivisible solver/topology pair.  A topology component is
never authored or resolved on its own: the exact pair crosses resolve -> compile -> bind as one
provider authority and is matched against the explicitly supplied ``resolve(components=...)`` set.
"""
from __future__ import annotations

import json
import math
from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet, RequirementSet


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
            "provider_id": self.provider_id,
            "provider_kind": "external_component_v1",
            "topology": topology,
            "solver": solver,
            "request": {
                "relative_tolerance": self.relative_tolerance,
                "absolute_tolerance": self.absolute_tolerance,
                "max_iterations": self.max_iterations,
            },
        }

    def to_data(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "options": self.options()}

    def requirements(self) -> RequirementSet:
        return RequirementSet({
            "external_components": True,
            "field_topology": True,
            "field_topology_contract": "uniform_cartesian_full_material_v1",
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
        return CapabilitySet({
            "external_field_solver_v2": True,
            "topology_provenance": True,
            "topology_contract": "uniform_cartesian_full_material_v1",
            "execution_adapter": "host_serial_multi_patch_batch_v1",
            "host": declared["host"] and adapter["host"],
            "mpi": declared["mpi"] and adapter["mpi"],
            "gpu": declared["gpu"] and adapter["gpu"],
            "component_pair_declares_mpi": declared["mpi"],
            "component_pair_declares_gpu": declared["gpu"],
        })

    def lower_field_solver(self, *, target: str, layout: Any) -> dict[str, Any]:
        if target != "system":
            raise ValueError(
                "ExternalFieldSolver ABI v2 supports Uniform System only; AMR requires a "
                "hierarchy-aware FieldSolver/FieldTopology interface version"
            )
        from pops.layouts import Uniform
        from pops.mesh import CartesianGrid

        if type(layout) is not Uniform or type(layout.mesh) is not CartesianGrid:
            raise ValueError(
                "ExternalFieldSolver ABI v2 requires an exact Uniform(CartesianGrid(...)) "
                "layout"
            )
        if layout.embedded_boundary is not None:
            raise ValueError(
                "ExternalFieldSolver ABI v2 currently proves full-material Cartesian topology; "
                "embedded-boundary/cut-cell material data is not lowered yet"
            )
        if layout.refine is not None:
            raise ValueError(
                "ExternalFieldSolver ABI v2 requires a non-adaptive Uniform layout; refinement "
                "criteria require the hierarchy-aware interface"
            )
        if not self.capabilities().get("host"):
            raise ValueError(
                "ExternalFieldSolver requires both component manifests to declare a compatible "
                "2D float64 CPU target for the host adapter"
            )
        return dict(self.options())


__all__ = ["ExternalFieldSolver"]
