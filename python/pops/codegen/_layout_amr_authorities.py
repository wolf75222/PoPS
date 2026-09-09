"""Exact local AMR authorities joined to an authenticated parent layout projection."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pops.mesh import LayoutPlan
    from pops.amr._resolution import ResolvedAMRAuthorities


@dataclass(frozen=True, slots=True)
class ResolvedLayoutAMRAuthorities:
    layout_plan: LayoutPlan
    authorities: ResolvedAMRAuthorities

    def __post_init__(self):
        from pops.mesh import LayoutPlan
        from pops.amr._resolution import ResolvedAMRAuthorities
        from pops.initial import InitialConditionPlan
        from pops.mesh._amr import BootstrapPlan, ResolvedHierarchy
        from pops.mesh._amr.transfer import ResolvedAMRTransfer
        from pops.amr import AMRExecution
        if type(self.layout_plan) is not LayoutPlan or len(self.layout_plan.layouts) != 1 \
                or not self.layout_plan.layouts[0].adaptive:
            raise TypeError("layout AMR authority requires one exact adaptive LayoutPlan")
        if type(self.authorities) is not ResolvedAMRAuthorities:
            raise TypeError("layout AMR authority requires exact ResolvedAMRAuthorities")
        a = self.authorities
        for value, kind in zip((a.hierarchy, a.transfer, a.initial_conditions, a.bootstrap,
                                a.execution),
                               (ResolvedHierarchy, ResolvedAMRTransfer, InitialConditionPlan,
                                BootstrapPlan, AMRExecution), strict=True):
            if type(value) is not kind:
                raise TypeError("layout AMR authority contains a non-exact authority")
        if any(value.layout_plan_id != self.layout_plan.qualified_id
               for value in (a.transfer, a.initial_conditions, a.bootstrap)):
            raise ValueError("layout AMR authorities authenticate another projection")
        if a.bootstrap.hierarchy_identity != a.hierarchy.identity \
                or a.bootstrap.transfer_identity != a.transfer.identity \
                or a.bootstrap.initial_identity != a.initial_conditions.identity:
            raise ValueError("layout bootstrap does not authenticate its exact authorities")

    @property
    def layout_id(self):
        return self.layout_plan.layouts[0].handle.qualified_id

    def canonical_identity(self):
        return {"schema_version": 1, "layout_id": self.layout_id,
                "layout_plan": self.layout_plan.canonical_identity(),
                "authorities": self.authorities.canonical_identity()}


@dataclass(frozen=True, slots=True)
class AMRAuthorityValidationContext:
    """The inputs consumed by the shared AMR validation, not a simulation plan."""
    layout_plan: Any
    resolved_hierarchy: Any
    amr_transfer: Any
    initial_condition_plan: Any
    bootstrap_plan: Any
    amr_execution: Any
    amr_providers: Any
    component_inputs: tuple
    blocks: tuple
    target: str = "amr_system"

    @classmethod
    def from_layout(cls, row, *, component_inputs=(), blocks=()):
        a = row.authorities
        return cls(row.layout_plan, a.hierarchy, a.transfer, a.initial_conditions,
                   a.bootstrap, a.execution, a.providers, tuple(component_inputs), tuple(blocks))


def validate_layout_amr_authorities(layout_plan, authorities):
    if not isinstance(authorities, Mapping):
        raise TypeError("layout_amr_authorities must be a mapping")
    if not authorities:
        return
    expected = tuple(row.handle.qualified_id for row in layout_plan.layouts if row.adaptive)
    if tuple(authorities) != expected:
        raise ValueError("layout AMR authorities must exactly cover adaptive layouts in plan order")
    for layout_id, value in authorities.items():
        if type(value) is not ResolvedLayoutAMRAuthorities:
            raise TypeError("layout AMR authorities must contain exact resolved layout authorities")
        handle = next(row.handle for row in layout_plan.layouts
                      if row.handle.qualified_id == layout_id)
        projected = layout_plan.project(handle)
        if value.layout_id != layout_id or value.layout_plan != projected:
            raise ValueError("layout AMR authority does not authenticate the parent projection")
