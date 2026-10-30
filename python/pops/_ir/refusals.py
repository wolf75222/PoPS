"""Typed refusals of the index-map analyses (spec 0.3 IR13, IR21, the refusal table of §14.1).

Each refusal names the region and the first map or coefficient that blocks it, and the phase
that raises it. They are ``ValueError`` subclasses, so a caller that already catches the
refusals of a phase keeps catching them.
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
    """A selection or a required claim needs a property that the analysis answers Unknown."""

    rule = "IR13"
    phase = "resolve"

    def __init__(self, message, *, region, property, blocking=None):
        self.property = property
        super().__init__(message, region=region, blocking=blocking)


class ClassNotReached(RegionRefusal):
    """Classification gives a region another class than the one its builder expects."""

    rule = "IR21"
    phase = "compile"

    def __init__(self, message, *, region, expected, reached, blocking=None):
        self.expected = expected
        self.reached = reached
        super().__init__(message, region=region, blocking=blocking)


__all__ = ["ClassNotReached", "PropertyNotDerived", "RegionRefusal"]
