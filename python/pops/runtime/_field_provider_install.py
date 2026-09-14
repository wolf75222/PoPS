"""Shared authenticated field-provider installation for uniform and AMR engines.

Component identity validation and registration use one ABI. Engine adapters retain
ownership of their distinct field-plan and topology publication order.
"""

from __future__ import annotations

from typing import Any


class PreparedFieldSolverInstall:
    """Resources and component registration shared by native field-solver adapters."""

    def __init__(self, engine: Any, field_plan: Any, install_plan: Any) -> None:
        self.engine = engine
        self.field_plan = field_plan
        self.install_plan = install_plan
        self.options = field_plan.native_install_data()
        self.slot = self.options["provider_slot"]

    def _register_component(self, binding: Any) -> Any:
        if self.install_plan is None:
            raise ValueError("component field providers require the authenticated InstallPlan")
        component_bindings = binding.resolution.to_data()["component_bindings"]
        if len(component_bindings) != 2:
            raise ValueError("component field provider requires exact topology and solver bindings")
        installed = []
        from pops.fields._identity import field_identity, strict_field_data
        from pops.identity import canonical_bytes

        for authority in component_bindings:
            component = self.install_plan.components.get(authority["component_id"])
            if component is None:
                raise ValueError(
                    "field %r requires installed component %r"
                    % (self.field_plan.name, authority["component_id"])
                )
            if component.component_manifest.token != authority["component_manifest_identity"]:
                raise ValueError("field component manifest identity changed before install")
            if canonical_bytes(strict_field_data(component.interface.to_data())) != canonical_bytes(
                strict_field_data(authority["native_interface"])
            ):
                raise ValueError("field component native interface identity changed before install")
            if component.native_handle is None:
                raise ValueError("field components must be loaded before native installation")
            installed.append(component.native_handle)
        import json
        from pops.runtime._component_execution_context import component_execution_data

        nullspace = self.options["nullspace_provider"]
        boundary = {
            "identity": field_identity(
                "field-boundary-contract",
                {
                    "field": self.field_plan.identity.token,
                    "faces": self.options["boundary_faces"],
                    "nullspace_provider": nullspace,
                    "topology_identity": binding.facts.layout["topology_identity"],
                },
            ).token,
            "faces": self.options["boundary_faces"],
            "nullspace_provider": nullspace,
            "topology_identity": binding.facts.layout["topology_identity"],
        }
        request = binding.resolution.native_contract["options"]
        exact = self.engine.register_field_solver_provider(
            self.slot,
            installed[0],
            installed[1],
            component_bindings[0],
            component_bindings[1],
            json.dumps(
                component_bindings[0]["parameters"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            json.dumps(
                component_bindings[1]["parameters"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            self.install_plan.artifact.layout_plan.qualified_id,
            binding.facts.layout["topology_identity"],
            json.dumps(strict_field_data(boundary), sort_keys=True, separators=(",", ":")),
            request["relative_tolerance"],
            request["absolute_tolerance"],
            request["max_iterations"],
            component_execution_data(self.install_plan.execution_context),
        )
        return exact


class PreparedFieldNullspaceInstall:
    """Publish a resolved nullspace through the shared native engine protocol."""

    def __init__(self, engine: Any, slot: str) -> None:
        self.engine = engine
        self.slot = slot

    def install_registered_nullspace(self, binding: Any) -> None:
        contract = binding.resolution.to_data()["native_contract"]
        self.engine.set_field_nullspace(
            self.slot,
            contract["provider_route"],
            contract["schema_identity"],
            contract["options"],
        )
