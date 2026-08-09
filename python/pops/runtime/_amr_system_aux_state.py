"""Exact owner-qualified auxiliary routes for :class:`AmrSystem`.

AMR auxiliary storage is a sealed native registry of ``InputAux`` and
``DerivedAux`` providers.  Python passes an exact ``ComponentKey`` through that
registry; it never picks a block-local carrier component or a legacy ``aux:N``
slot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pops.runtime._amr_system_contract import _AmrSystem
else:
    _AmrSystem = object


class _AmrSystemAuxState(_AmrSystem):
    """ComponentKey-only AMR auxiliary uploads and reads."""

    @staticmethod
    def _auxiliary_key(value: Any) -> Any:
        from pops.model.provider_pack import ComponentKey

        if type(value) is ComponentKey:
            return value
        if isinstance(value, dict):
            try:
                return ComponentKey(**value)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "auxiliary component mapping must carry owner_qid, space_kind, "
                    "space_name, component"
                ) from exc
        raise TypeError("auxiliary input requires an exact pops.model.ComponentKey")

    def stage_auxiliary_input(self, key: Any, values: Any) -> None:
        """Stage one declared owner-qualified ``InputAux`` component.

        Native code authenticates the producer, resolves its storage group, and
        publishes it transactionally.  ``DerivedAux`` and field outputs cannot
        be overwritten through this route.
        """
        import numpy as np

        exact = self._auxiliary_key(key)
        self._s.stage_auxiliary_input(
            exact.owner_qid,
            exact.space_kind,
            exact.space_name,
            exact.component,
            np.asarray(values, dtype=float),
        )

    def auxiliary_component(self, key: Any) -> Any:
        """Read one published owner-qualified auxiliary component."""
        exact = self._auxiliary_key(key)
        return self._s.auxiliary_component(
            exact.owner_qid,
            exact.space_kind,
            exact.space_name,
            exact.component,
        )


__all__ = ["_AmrSystemAuxState"]
