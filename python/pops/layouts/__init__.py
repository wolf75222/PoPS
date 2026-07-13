"""Public mesh-organization descriptors.

A layout owns mesh structure only.  Physics, numerical methods, temporal programs and runtime
controls remain separate authorities.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors_report import CapabilitySet, RequirementSet
from pops.mesh._descriptor import Availability, MeshDescriptor
from pops.mesh.layouts import Uniform


class AMR(MeshDescriptor):
    """One complete adaptive-layout authority.

    Accuracy, formal order and halo depth are deliberately absent: they are derived from the
    selected numerical and transfer providers during resolution.
    """

    category = "layout"

    def __init__(
        self,
        *,
        grid: Any,
        hierarchy: Any,
        tagging: Any,
        regrid: Any,
        transfer: Any,
        execution: Any,
    ) -> None:
        # Structural snapshots consume ``options()``.  Keeping authorities private prevents the
        # generic snapshotter from recursively treating Schedule implementation helpers as public
        # authoring state in addition to that canonical projection.
        self._grid = grid
        self._hierarchy = hierarchy
        self._tagging = tagging
        self._regrid = regrid
        self._transfer = transfer
        self._execution = execution

    @property
    def grid(self) -> Any:
        return self._grid

    @property
    def hierarchy(self) -> Any:
        return self._hierarchy

    @property
    def tagging(self) -> Any:
        return self._tagging

    @property
    def regrid(self) -> Any:
        return self._regrid

    @property
    def transfer(self) -> Any:
        return self._transfer

    @property
    def execution(self) -> Any:
        return self._execution

    def _validate_authorities(self) -> None:
        from pops.amr import (
            AMRExecution,
            AMRHierarchy,
            AMRRegrid,
            AMRTagging,
            AMRTransfer,
        )

        expected = (
            (self.hierarchy, AMRHierarchy, "hierarchy"),
            (self.tagging, AMRTagging, "tagging"),
            (self.regrid, AMRRegrid, "regrid"),
            (self.transfer, AMRTransfer, "transfer"),
            (self.execution, AMRExecution, "execution"),
        )
        for value, kind, name in expected:
            if type(value) is not kind:
                raise TypeError("AMR.%s must be an exact %s" % (name, kind.__name__))
        for method in ("validate", "capabilities", "requirements", "options", "to_dict"):
            if not callable(getattr(self.grid, method, None)):
                raise TypeError("AMR.grid must implement %s()" % method)
        if self.hierarchy.max_levels < 2:
            raise ValueError("AMR hierarchy requires at least one coarse/fine transition")

    def requirements(self) -> RequirementSet:
        return RequirementSet({
            "amr_runtime": True,
            "reflux": True,
            "transactional_regrid": True,
            "tag_reduction": True,
        })

    def capabilities(self) -> CapabilitySet:
        self._validate_authorities()
        grid = self.grid.capabilities().to_dict()
        ratio = self.hierarchy.uniform_ratio
        return CapabilitySet({
            "layout": "amr",
            "supports_amr": True,
            "dim": grid.get("dim"),
            "max_levels": self.hierarchy.max_levels,
            "ratio": ratio if ratio is not None else 1,
            "transition_ratios": list(self.hierarchy.ratios),
            "execution": self.execution.mode,
        })

    def options(self) -> dict[str, Any]:
        self._validate_authorities()
        return {
            "grid": self.grid.to_dict(),
            "hierarchy": self.hierarchy.to_data(),
            "tagging": self.tagging.inspect(),
            "regrid": self.regrid.to_data(),
            "transfer": self.transfer.inspect(),
            "execution": self.execution.to_data(),
        }

    def available(self, context: Any = None) -> Availability:
        del context
        self._validate_authorities()
        dimension = self.grid.capabilities().get("dim")
        if dimension != 2:
            return Availability.no(
                "the installed native AMR provider supports exactly two spatial dimensions",
                alternatives=["select a registered AMR hierarchy provider for this dimension"],
            )
        ratio = self.hierarchy.uniform_ratio
        if ratio is None:
            return Availability.no(
                "the installed native hierarchy provider cannot preserve heterogeneous "
                "transition ratios",
                alternatives=["select a provider advertising per-transition ratios"],
            )
        if ratio != 2:
            return Availability.no(
                "the installed native transfer providers support refinement ratio 2",
                alternatives=["AMRHierarchy(..., ratios=(2, ...))"],
            )
        return Availability.yes()

    def validate(self, context: Any = None) -> bool:
        self._validate_authorities()
        self.grid.validate(context)
        return super().validate(context)

    def resolve_for_case(self, resolver: Any) -> AMR:
        """Return a detached descriptor with every declaration Handle authenticated."""
        if not callable(resolver):
            raise TypeError("AMR.resolve_for_case requires a callable Handle resolver")

        def resolved(value: Any) -> Any:
            if getattr(value, "is_resolved", False):
                return value
            return resolver(value)

        return type(self)(
            grid=self.grid,
            hierarchy=self.hierarchy,
            tagging=self.tagging.resolve_references(resolved),
            regrid=self.regrid,
            transfer=self.transfer.resolve_references(resolved),
            execution=self.execution,
        )

    def resolve_amr_authorities(self, context: Any) -> Any:
        """Resolve via the same open layout protocol available to extension descriptors."""
        from pops.amr import resolve_amr_authorities

        return resolve_amr_authorities(
            hierarchy=self.hierarchy,
            tagging=self.tagging,
            regrid=self.regrid,
            transfer=self.transfer,
            execution=self.execution,
            context=context,
        )

    def runtime_layout_data(self) -> dict[str, Any]:
        """Project exact geometry/cadence/execution facts for a runtime provider."""
        self._validate_authorities()
        return {
            "schema_version": 1,
            "layout_type": "adaptive_cartesian",
            "grid": self.grid.to_dict(),
            "regrid": self.regrid.to_data(),
            "execution": self.execution.to_data(),
        }


__all__ = ["AMR", "Uniform"]
