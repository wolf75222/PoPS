"""Elaboration of the temporal program onto the internal builder (spec 0.3, section 5.2).

``Advance(method)`` is one attempt loop of a specialized method, ``Proposal(dt)`` its attempt
parameters and ``Retry`` its budgets. ``elaborate`` traces the method once against a fresh builder
(``pops.time.Program``), commits the candidate at the endpoint of the state it evolves and attaches
the builder to the case: the builder is the elaborator, and the case never calls it.

This lot elaborates the trial of a method whose candidate reaches ``B.time + p.dt``, with the
default acceptance (``finite()``, written ``accept=None``) and one attempt. That acceptance
proposes the same proposal again (T28), a fixed step, which the builder runs as ``FixedDt``; the
finiteness check itself, ``Retry`` budgets, other assessments and the compiled step loop are
lowered by the compiled control of N8. Every construct of section 5 outside this lot is refused
with ``NotYetElaborated``, which names the lot that lowers it.
"""
from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any

from pops.time._program.api import Program as _Builder
from pops.time._program.traced_method import (
    MethodTrace, NotYetElaborated, SpecializedMethod, _fraction,
)
from pops.time._step.strategy import FixedDt


class Proposal:
    """``Proposal(dt, request=None, **choices)``: the attempt parameters of a step (5.2)."""

    __slots__ = ("dt", "request", "choices")

    def __init__(self, dt: Any, request: Any = None, **choices: Any) -> None:
        if isinstance(dt, bool) or not isinstance(dt, (int, float, Fraction, Decimal)):
            raise TypeError("Proposal(dt): dt is a positive finite duration, got %r" % (dt,))
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError("Proposal(dt): dt is a positive finite duration, got %r" % (dt,))
        _fraction(dt)
        if request is not None:
            raise NotYetElaborated("Proposal(request=...): solve requests (5.5, SV3) are "
                                   "elaborated with the implicit relations of a later lot of N8")
        if choices:
            raise NotYetElaborated("Proposal(%s): attempt choices are elaborated by the compiled "
                                   "control of N8" % ", ".join(sorted(choices)))
        self.dt: Any = dt
        self.request = request
        self.choices = dict(choices)


class Retry:
    """``Retry(max_attempts=1, max_transforms=0)``: the budgets of one attempt loop (T38)."""

    __slots__ = ("max_attempts", "max_transforms")

    def __init__(self, max_attempts: int = 1, max_transforms: int = 0) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("Retry(max_attempts=n) needs an integer n >= 1")
        if isinstance(max_transforms, bool) or not isinstance(max_transforms, int) \
                or max_transforms < 0:
            raise ValueError("Retry(max_transforms=n) needs an integer n >= 0")
        self.max_attempts = max_attempts
        self.max_transforms = max_transforms


class Advance:
    """``Advance(method, accept=finite(), retry=None, clock=None, time_map=None)`` (5.2).

    ``accept=None`` is the default acceptance ``finite()``; ``retry=None`` is ``Retry()``.
    """

    __slots__ = ("method", "accept", "retry", "clock", "time_map")

    def __init__(self, method: Any, accept: Any = None, retry: Any = None, clock: Any = None,
                 time_map: Any = None) -> None:
        if not isinstance(method, SpecializedMethod):
            raise TypeError("Advance(method): a method specialized with its components, "
                            "written m(component=...), got %r" % (method,))
        if accept is not None:
            raise NotYetElaborated("Advance(accept=...): an assessment other than the default "
                                   "finite() (T28) is lowered by the compiled control of N8")
        if retry is not None and not isinstance(retry, Retry):
            raise TypeError("Advance(retry=...) takes a Retry")
        if retry is not None and (retry.max_attempts, retry.max_transforms) != (1, 0):
            raise NotYetElaborated("Advance(retry=Retry(%d, %d)): retry budgets (T38) are lowered "
                                   "by the compiled control of N8"
                                   % (retry.max_attempts, retry.max_transforms))
        if clock is not None or time_map is not None:
            raise NotYetElaborated("Advance(clock=..., time_map=...): clocks and time maps (T23, "
                                   "T36) are elaborated in Phase 2 of N8")
        self.method = method
        self.accept = accept
        self.retry = Retry() if retry is None else retry
        self.clock = clock
        self.time_map = time_map


class Elaboration:
    """The result of ``elaborate``: the builder attached to the case and the method trace."""

    __slots__ = ("advance", "proposal", "program", "trace")

    def __init__(self, advance: Advance, proposal: Proposal, program: Any,
                 trace: MethodTrace) -> None:
        self.advance = advance
        self.proposal = proposal
        self.program = program
        self.trace = trace


def elaborate(advance: Advance, *, case: Any, state: Any, proposal: Proposal) -> Elaboration:
    """Elaborate ``advance`` for the block-qualified ``state`` of ``case``, from ``proposal``.

    The builder is named after the method; its value and stage names are presentation (CC2).
    """
    if not isinstance(advance, Advance):
        raise TypeError("elaborate(advance): an Advance, got %r" % (advance,))
    if not isinstance(proposal, Proposal):
        raise TypeError("elaborate(proposal=...): the first Proposal of the run, got %r"
                        % (proposal,))
    program = _Builder(advance.method.name)
    temporal = program.state(state)
    trace = advance.method.trace(program, temporal, proposal)
    program.commit(temporal.next, trace.candidate)
    program.step_strategy(FixedDt(proposal.dt))
    case.program(program)
    return Elaboration(advance, proposal, program, trace)
