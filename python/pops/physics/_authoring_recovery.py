"""Primitive-recovery policy authoring for symbolic physical models."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops._ir import _children, _wrap
from pops._ir.quantity import QuantityRef

if TYPE_CHECKING:
    from ._model_contract import _HyperbolicModel
else:
    _HyperbolicModel = object


class _RecoveryMixin(_HyperbolicModel):
    """Declare physical constraints consumed by native variable recovery."""

    def recovery_admissibility(self, **constraints: Any) -> None:
        """Require named primitive components to satisfy symbolic predicates.

        Keys identify components of the already-declared primitive state.  Values are typed
        symbolic Boolean expressions over that primitive state, for example
        ``rho=rho > 0`` or ``p=p >= p_floor``.  The generated C++ brick reports the first failing
        primitive component, and the prepared recovery chain refuses to publish that candidate.
        """
        self._declare_recovery_admissibility(constraints)

    def _declare_recovery_admissibility(
        self, constraints: Any, *, primitive_quantities: Any = (),
    ) -> None:
        """Retain predicates after checking explicitly nominated primitive coordinates.

        The public blackboard nominates its selected coordinate objects. Their qualified
        references remain in the scientific formulas until the private emitter binding;
        component spelling alone cannot establish this authority.
        """
        self._guard_mutable("declare recovery admissibility")
        if not self.prim_state:
            raise ValueError(
                "recovery_admissibility: call primitive_vars(...) first so constraints have "
                "a typed component layout"
            )
        if not constraints:
            raise ValueError("recovery_admissibility: declare at least one named constraint")
        if self._recovery_admissibility:
            raise ValueError(
                "recovery_admissibility: policy already declared; author the complete policy "
                "in one call"
            )

        primitive_names = set(self.prim_state)
        unknown_components = sorted(set(constraints) - primitive_names)
        if unknown_components:
            raise ValueError(
                "recovery_admissibility: unknown primitive components %s; declared layout is %s"
                % (unknown_components, list(self.prim_state))
            )

        quantities = tuple(primitive_quantities)
        if any(not isinstance(value, QuantityRef)
               or value.handle.kind != "state"
               or value.handle.owner_path != self.owner_path
               or value.component not in primitive_names for value in quantities):
            raise ValueError("recovery primitive quantities require owned selected state coordinates")

        prepared = {}
        for component in self.prim_state:
            if component not in constraints:
                continue
            predicate = _wrap(constraints[component])
            if not callable(getattr(predicate, "resolve_for_amr_predicate", None)):
                raise TypeError(
                    "recovery_admissibility[%r] requires a typed symbolic Boolean expression"
                    % component
                )
            qualified_dependencies = set()
            stack = [predicate]
            while stack:
                node = stack.pop()
                if isinstance(node, QuantityRef):
                    if not any(node.handle == value.handle and node.space == value.space
                               and node.index == value.index for value in quantities):
                        raise ValueError(
                            "recovery_admissibility[%r] reads a qualified quantity outside the "
                            "owned selected primitive state: %s" % (component, node.qualified_id))
                    qualified_dependencies.add(node.qualified_id)
                stack.extend(_children(node))
            unknown_dependencies = sorted(
                set(predicate.deps()) - primitive_names - qualified_dependencies)
            if unknown_dependencies:
                raise ValueError(
                    "recovery_admissibility[%r] reads values outside the primitive state: %s"
                    % (component, unknown_dependencies)
                )
            prepared[component] = predicate

        self._recovery_admissibility = prepared
        self._invalidate_authoring_views()
