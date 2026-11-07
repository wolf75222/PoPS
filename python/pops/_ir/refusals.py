"""Typed refusals of the region analyses (spec 0.3 IR13, IR17, IR19, IR21, D5, the refusal table of
§14.1).

Each refusal names the phase that raises it and what it refuses: a region refusal names the
region and the first map, coefficient or construct that blocks it, a coverage refusal the term.
They are ``ValueError`` subclasses, so a caller that already catches the refusals of a phase
keeps catching them.
"""
from __future__ import annotations


class RegionRefusal(ValueError):
    """A refusal about one region of the plan, with the map or coefficient that blocks it."""

    rule = ""
    phase = ""

    def __init__(self, message, *, region, blocking=None):
        self.region = region
        self.blocking = blocking
        super().__init__("%s (%s, %s): %s" % (type(self).__name__, self.rule, self.phase, message))


class PropertyNotDerived(RegionRefusal):
    """A selection or a required claim needs a property that the analysis does not prove."""

    rule = "IR13"
    phase = "resolve"

    def __init__(self, message, *, region, property, blocking=None):
        self.property = property
        super().__init__(message, region=region, blocking=blocking)


class ConservationNotStructural(PropertyNotDerived):
    """A conservation claim without a proved identity Sum_k w_k = 0 (IR17)."""

    rule = "IR17"
    phase = "resolve"


class ClassNotReached(RegionRefusal):
    """Classification gives a region another class than the one its builder expects."""

    rule = "IR21"
    phase = "compile"

    def __init__(self, message, *, region, expected, reached, blocking=None):
        self.expected = expected
        self.reached = reached
        super().__init__(message, region=region, blocking=blocking)


class DeliveryNotStructural(RegionRefusal):
    """An owned ``Use`` leaves a region by a path that IR19 does not type."""

    rule = "IR19"
    phase = "resolve"

    def __init__(self, message, *, region, construct, term=None):
        self.construct = construct
        self.term = term
        self.detail = message
        if term is not None:
            message = "%s; term %s" % (message, term)
        super().__init__(message, region=region, blocking=construct)


class CoverageNotExact(ValueError):
    """A term is covered twice or not at all on a success path (D5)."""

    rule = "D5"
    phase = "resolve"

    def __init__(self, message, *, term, count):
        self.term = term
        self.count = count
        times = "0 times" if count == 0 else "at least twice"
        super().__init__("%s (%s, %s): %s; term %s is delivered %s on a success path"
                         % (type(self).__name__, self.rule, self.phase, message, term, times))


__all__ = ["ClassNotReached", "ConservationNotStructural", "CoverageNotExact",
           "DeliveryNotStructural", "PropertyNotDerived", "RegionRefusal"]
