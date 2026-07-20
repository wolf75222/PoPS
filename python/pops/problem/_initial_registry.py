"""Case-owned initial-condition declarations and AMR authority lowering."""
from __future__ import annotations

from typing import Any

from pops.model import Handle, OwnerKind, OwnerPath
from pops.problem._registry_freeze import FreezableRegistry
from pops._report import ReportTree


class InitialConditionRegistry(FreezableRegistry):
    """Register each qualified physical state exactly once."""

    family = "initial"

    def __init__(self, owner: Any, resolver: Any) -> None:
        self._owner_path = OwnerPath.coerce(owner).require_authoring_root(
            OwnerKind.CASE, where="InitialConditionRegistry owner")
        if not callable(resolver):
            raise TypeError("InitialConditionRegistry resolver must be callable")
        self._resolver = resolver
        self._conditions: dict[str, Any] = {}

    @property
    def owner_path(self) -> OwnerPath:
        return self._owner_path

    def add(self, initial: Any) -> Any:
        from pops.initial import InitialCondition

        self._guard_frozen("add an initial condition")
        if type(initial) is not InitialCondition:
            raise TypeError("case.initials.add requires an exact InitialCondition")
        canonical = self._resolver(initial.state)
        if not isinstance(canonical, Handle) or canonical.kind != "state" \
                or not canonical.is_resolved:
            raise TypeError(
                "InitialConditionRegistry resolver must return a canonical state Handle")
        key = canonical.qualified_id
        if key in self._conditions:
            raise ValueError("initial condition for %s is already declared" % key)
        self._conditions[key] = initial
        return initial

    def __len__(self) -> int:
        return len(self._conditions)

    def __iter__(self) -> Any:
        return iter(self._conditions.values())

    def _freezable_members(self) -> Any:
        return tuple(self._conditions.values())

    def resolved(self) -> tuple[Any, ...]:
        return tuple(
            self._conditions[key].resolve_references(self._resolver)
            for key in sorted(self._conditions)
        )

    def validate(self, context: Any = None) -> ReportTree:
        report = ReportTree(
            phase="validation",
            severity="info",
            code="validation.initial.root",
            source=self.family,
        )
        for key, initial in sorted(self._conditions.items()):
            try:
                resolved = initial.resolve_references(self._resolver)
                resolved.canonical_identity()
            except Exception as exc:  # noqa: BLE001 - aggregate exact descriptor refusal
                report = report.error(
                    self.family,
                    "invalid_initial_condition",
                    str(exc),
                    context={"state": key},
                )
        return report

    def inspect(self) -> list[dict[str, Any]]:
        return [initial.inspect() for initial in self._conditions.values()]

    def resolve_plan(self, *, layout_plan: Any, expected_subjects: Any) -> Any:
        """Resolve one exact initialization plan for any uniform or adaptive layout."""
        if not self._conditions:
            raise ValueError(
                "initial-condition resolution requires at least one Case initial condition")
        from pops.initial import InitialConditionPlanBuilder

        builder = InitialConditionPlanBuilder(layout_plan, expected_subjects)
        for key in sorted(self._conditions):
            authored = self._conditions[key]
            initial = authored.resolve_references(self._resolver)
            builder.add(
                initial.state,
                initial.source(self.owner_path),
                authoring_alias=(
                    authored.state if not authored.state.is_resolved else None
                ),
            )
        return builder.resolve()

    def resolve_amr(
        self,
        *,
        layout_plan: Any,
        transfers: Any,
        hierarchy: Any,
        tagging: Any,
        constraints: Any = (),
    ) -> Any:
        """Derive the exact low-level IC and bootstrap plans from registered bricks."""
        if not self._conditions:
            raise ValueError("AMR resolution requires at least one Case initial condition")
        from pops.initial import InitialConditionAuthorities
        from pops.mesh._amr import (
            BootstrapOrdering,
            BootstrapSelection,
            resolve_bootstrap,
        )
        from pops.mesh._amr.bootstrap import _physical_initial_subjects

        authored = tuple(
            self._conditions[key] for key in sorted(self._conditions)
        )
        resolved = tuple(
            initial.resolve_references(self._resolver) for initial in authored
        )
        phase_orders = {initial.bootstrap_phases for initial in resolved}
        if len(phase_orders) != 1:
            raise ValueError(
                "initial projections require incompatible bootstrap phase orderings")
        selections = []
        for initial in resolved:
            selections.append(BootstrapSelection(initial.state, initial.bootstrap_method()))
        initial_plan = self.resolve_plan(
            layout_plan=layout_plan,
            expected_subjects=_physical_initial_subjects(transfers),
        )
        bootstrap_plan = resolve_bootstrap(
            layout_plan=layout_plan,
            hierarchy=hierarchy,
            transfers=transfers,
            initial_conditions=initial_plan,
            tagging=tagging,
            selections=tuple(selections),
            ordering=BootstrapOrdering(next(iter(phase_orders))),
            constraints=constraints,
        )
        return InitialConditionAuthorities(initial_plan, bootstrap_plan)


__all__ = ["InitialConditionRegistry"]
