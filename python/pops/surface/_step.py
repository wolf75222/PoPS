"""A step rule of a library, ``dt=`` of ``pops.Advance`` (spec 0.3, 2.6.2, 2.6.9; lot D-E of N15).

``pops.Advance(method, dt=rule)`` takes a number, a fixed proposal, or a step rule that a library
writes over the fluxes of the model, such as the ``cfl_limit`` of ``pops.lib.time``. The core
knows the rule by the fluxes it reads and by the realization the library gives (``realize``), the
step rule of N8 that the compiled run executes; it does not know which formula the library
names, as it does not know the name of a scheme (CF2).

A rule reads the speeds that b6 records for the whole model on each cell (R9, SD44), so the
elaborator checks that it names every transport flux of the model, that each flux has the
derived wave speeds of R9, and that the model has no diffusive flux, whose limit b6 adds to the
same selection; a rule outside these limits names the lot that elaborates it.
"""
from __future__ import annotations

from typing import Any

from ._model import DiffusiveFlux, Flux
from ._refusal import user_site


class StepRuleValue:
    """A step rule of a library: the fluxes whose speeds bound each step, the options the
    library states (shown by ``repr`` and ``pops.inspect``) and ``realize``, the step rule of
    N8 that the compiled run executes. A library subclasses it and names it with ``word``."""

    word = "step rule"
    __slots__ = ("fluxes", "options", "site")

    def __init__(self, fluxes: Any, options: tuple, *, site: Any = None) -> None:
        listed = tuple(fluxes) if isinstance(fluxes, (list, tuple)) else (fluxes,)
        if not listed:
            raise TypeError("%s(...) takes a flux of the model or a list of them" % self.word)
        for flux in listed:
            if isinstance(flux, DiffusiveFlux) or not isinstance(flux, Flux):
                raise TypeError("%s(...) takes transport fluxes of the model, model.flux(of=u, "
                                "value=...), got %r" % (self.word, flux))
        if len({id(flux) for flux in listed}) != len(listed):
            raise ValueError("%s(...) names each flux once" % self.word)
        models = {id(flux.model) for flux in listed}
        if len(models) != 1:
            raise ValueError("%s(...) takes fluxes of one model" % self.word)
        self.fluxes = listed
        self.options = tuple(options)
        self.site = user_site(3) if site is None else site

    def realize(self) -> Any:
        """The step rule of N8 that the compiled run executes (``pops.time``); a library
        returns ``self.rule(...)``."""
        raise NotImplementedError("%s.realize()" % type(self).__name__)

    @staticmethod
    def rule(fraction: float, max_dt: float | None, *, strategy: Any) -> Any:
        """N8's compiled step rule: each step is b6's selection at ``fraction`` of the stable
        step of the model's wave speeds, capped by ``max_dt``; ``strategy`` is the step
        strategy that ``pops.run`` takes on the same artifact, which the library pairs with the
        rule (CF2: the core names neither)."""
        from pops.time._program.elaboration import StepRule

        return StepRule(fraction, max_dt, strategy=strategy)

    def __repr__(self) -> str:
        from ._expr import text

        names = [text(flux) for flux in self.fluxes]
        first = names[0] if len(names) == 1 else "[%s]" % ", ".join(names)
        options = ["%s=%r" % (name, value) for name, value in self.options if value is not None]
        return "%s(%s)" % (self.word, ", ".join([first, *options]))
