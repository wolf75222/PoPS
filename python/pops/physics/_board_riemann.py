"""Capability-checked, atomic Riemann selection for the board facade."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._board_contract import atomic_attrs, require_name
from .board_handles import _roles_for

if TYPE_CHECKING:
    from ._model_contract import _BoardModel
else:
    _BoardModel = object


_GENERIC_CAPABILITY_ENABLERS = {"hllc": "enable_hllc", "roe": "enable_roe"}
_PRESSURE_ROLE_FLUXES = frozenset({"hllc", "roe", "euler_hllc", "euler_roe"})
_EULER_LAYOUT_FLUXES = frozenset({"euler_hllc", "euler_roe"})


class _RiemannAuthoringMixin(_BoardModel):
    """Select a Riemann provider only after all required capabilities validate."""

    def riemann(self, name: Any, flux: Any = None, pressure: Any = None,
                velocity: Any = None, sound_speed: Any = None, wave_speeds: Any = None,
                contact_speed: Any = None, star_state: Any = None) -> Any:
        scheme = getattr(name, "scheme", name)
        kind = require_name(scheme, "Riemann scheme").lower()
        self._validate_riemann_capabilities(kind, pressure, wave_speeds)
        hyp = self._dsl._m
        with atomic_attrs(
                (hyp, "aux_names"), (hyp, "aux_extra_names"), (hyp, "_hllc"), (hyp, "_roe"),
                (hyp, "_riemann_hook_forms"), (self, "_riemann")):
            enabler = _GENERIC_CAPABILITY_ENABLERS.get(kind)
            if enabler is not None:
                getattr(self._dsl, enabler)()
            self._dsl.set_riemann_hooks(
                pressure=self._to_expr(pressure) if pressure is not None else None,
                sound_speed=self._to_expr(sound_speed) if sound_speed is not None else None,
                contact_speed=self._to_expr(contact_speed) if contact_speed is not None else None,
                star_state=self._to_expr(star_state) if star_state is not None else None,
            )
            self._riemann = name
        return name

    def _validate_riemann_capabilities(self, kind: str, pressure: Any,
                                       wave_speeds: Any) -> None:
        hyp = self._dsl._m
        roles = set(_roles_for(hyp))
        has_pressure = "p" in hyp.prim_defs or pressure is not None
        fluid = {"Density", "MomentumX", "MomentumY"}
        if kind in _PRESSURE_ROLE_FLUXES:
            if not has_pressure:
                raise ValueError(
                    "riemann %s requires model capability 'pressure' for state %r: declare a "
                    "primitive m.primitive('p', ...) or pass m.riemann(..., pressure=...)"
                    % (kind.upper(), self._state_name()))
            missing = fluid - roles
            if missing:
                raise ValueError(
                    "riemann %s requires model capability 'hllc_star_state' for state %r: the "
                    "fluid roles %s are needed (declare m.state(..., roles={...})); missing %s"
                    % (kind.upper(), self._state_name(), sorted(fluid), sorted(missing)))
            if kind in _EULER_LAYOUT_FLUXES and len(hyp.cons_names) != 4:
                raise ValueError(
                    "riemann %s requires a canonical 4-variable Euler layout (rho, rho_u, rho_v, E) "
                    "for state %r; use riemann='hllc'/'roe' for a generic model"
                    % (kind.upper(), self._state_name()))
        elif kind == "hll":
            if (wave_speeds is None and not hyp._eig and hyp._wave_speeds is None
                    and hyp._ws_jacobian is None):
                raise ValueError(
                    "riemann HLL requires model capability 'wave_speeds': declare m.flux(..., "
                    "waves=...) or pass m.riemann('hll', wave_speeds=...)")

    def _state_name(self) -> str:
        return next(iter(self._states), "U")


__all__ = ["_RiemannAuthoringMixin"]
