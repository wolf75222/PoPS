"""``cfl_limit(flux, cfl=, max_dt=None)``: master's ``AdaptiveCFL`` as a step rule of the surface
(spec 0.3, 2.6.9 gallery, SD44, SD70; lot D-E of N15).

``pops.Advance(method, dt=cfl_limit(advection, cfl=0.45, max_dt=1e-2))`` takes each step as
cfl x min_a h_a / max_a s_a over the cells, s_a the wave speed that b6 records on each cell for
the fluxes of the model, capped by ``max_dt``; on a uniform layout this is master's
cfl . h_min / max speed (``include/pops/runtime/system.hpp:872-874@bb9``). The run lands on its
frontiers. ``cfl_limit([advection, drift], cfl=)`` keeps the smallest of their limits, which b6's
selection gives when the list names every flux of the model.

It is a bridge of MG9 (``tests/gates/mg9_bridges.toml``), as ``cg()`` of ``pops.lib.solve``: it
subclasses the surface's ``StepRuleValue`` and names b6's ``AdaptiveCFL``, until N8 elaborates
``dt=`` as a function of ``cell`` and the step rules of 5.11 are written with the surface. The value is a ``pops.StepRuleValue``; ``realize`` builds the
step rule of N8, ``StepRule(cfl, max_dt, strategy=AdaptiveCFL(cfl, max_dt))``: the compiled run of
``System::run_advance`` executes the rule, and ``pops.run`` on the same artifact runs master's
``AdaptiveCFL``. The pairing is this word's; the core never reads its options (CF2).
"""
from __future__ import annotations

import math as _math
from typing import Any

from pops.surface import StepRuleValue
from pops.time import AdaptiveCFL


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("cfl_limit(flux, %s=...) takes a positive number, got %r" % (name, value))
    value = float(value)
    if not _math.isfinite(value) or value <= 0.0:
        raise ValueError("cfl_limit(flux, %s=...) takes a positive finite number, got %r"
                         % (name, value))
    return value


class _CflLimit(StepRuleValue):
    """The value of ``cfl_limit(flux, cfl=, max_dt=None)``."""

    word = "cfl_limit"
    __slots__ = ("cfl", "max_dt")

    def __init__(self, fluxes: Any, cfl: float, max_dt: float | None) -> None:
        self.cfl = cfl
        self.max_dt = max_dt
        super().__init__(fluxes, (("cfl", cfl), ("max_dt", max_dt)))

    def realize(self) -> Any:
        # Simulation.run takes the rule; pops.run on the same artifact takes master's AdaptiveCFL.
        return self.rule(self.cfl, self.max_dt, strategy=AdaptiveCFL(self.cfl, self.max_dt))


def cfl_limit(flux: Any = None, *, cfl: Any = None, max_dt: Any = None, **earlier: Any) -> Any:
    """``cfl_limit(flux, cfl=, max_dt=None)``, or a list of fluxes (2.6.9)."""
    if "number" in earlier:
        raise TypeError("cfl_limit(flux, cfl=...) takes master's keyword cfl=, not number= "
                        "(SD70)")
    if earlier:
        raise TypeError("cfl_limit has no option %s" % ", ".join(sorted(earlier)))
    if flux is None:
        raise TypeError("cfl_limit(flux, cfl=...) takes the flux whose speeds bound the step")
    if cfl is None:
        raise TypeError("cfl_limit(flux, cfl=...) needs cfl=, the Courant number")
    return _CflLimit(flux, _positive(cfl, "cfl"),
                    None if max_dt is None else _positive(max_dt, "max_dt"))


__all__ = ["cfl_limit"]
