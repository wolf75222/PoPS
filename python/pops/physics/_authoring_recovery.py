"""Primitive-recovery policy authoring for symbolic physical models."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops._ir import _wrap

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
            unknown_dependencies = sorted(set(predicate.deps()) - primitive_names)
            if unknown_dependencies:
                raise ValueError(
                    "recovery_admissibility[%r] reads values outside the primitive state: %s"
                    % (component, unknown_dependencies)
                )
            prepared[component] = predicate

        self._recovery_admissibility = prepared
