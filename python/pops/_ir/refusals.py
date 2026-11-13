"""Typed refusals of the region analyses (spec 0.3 IR13, IR17, IR19, IR21, D5, R7, the refusal table
of §14.1), and of the elaboration of a named body (IR6).

Each refusal names the phase that raises it and what it refuses: a region refusal names the
region and the first map, coefficient or construct that blocks it, a coverage refusal the term,
and a balance refusal the term and the construct that holds it.
They are ``ValueError`` subclasses, so a caller that already catches the refusals of a phase
keeps catching them.
"""
from __future__ import annotations

from typing import Any


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


class GeneralAccessOnPatches(RegionRefusal, TypeError):
    """DM5 (strict mode): a table or CSR gather reaches a field leaf, an array on ``Patches``.

    On ``Patches`` no table or CSR gather reaches an array unless the case requests it, and phase 1
    has no request form (N10 lot 3, addendum §3). The general accesses are ``at`` with an index
    that is not the argument of its map, the distributed local access, the imports by requests
    or by CSR, and the distributed scatter. A field leaf was already refused there with a
    ``TypeError``; the refusal is also a ``TypeError``, so callers that catch it keep catching it.
    """

    rule = "DM5"
    phase = "tracing"

    def __init__(self, message, *, region, op):
        self.op = op
        super().__init__(message, region=region, blocking=op)


class CoverageNotExact(ValueError):
    """A term is covered twice or not at all on a success path (D5)."""

    rule = "D5"
    phase = "resolve"

    term: Any = None
    count: Any = None
    terms: tuple[Any, ...] = ()

    def __init__(self, message, *, term=None, count=None, terms=None):
        if term is None and count is None and terms is None:
            # A cooperative subclass (the surface's row) has written its message and set its
            # attributes; the class defaults stand for any it left out.
            super().__init__(message)
            return
        self.term = term
        self.count = count
        # Every term in the same state, when one refusal names several (O6); ``term`` is the first.
        self.terms = (term,) if terms is None else tuple(terms)
        times = "0 times" if count == 0 else "at least twice"
        super().__init__("%s (%s, %s): %s; term %s is delivered %s on a success path"
                         % (type(self).__name__, self.rule, self.phase, message, term, times))


class ScheduleIllegal(ValueError):
    """A storage constraint written by a case that the plan cannot satisfy (IR22, DM3).

    The plan chooses every storage layout and distribution; a case MAY constrain the choice, and
    ``pops.compile`` refuses a constraint that the runtime cannot run, before any native work
    (phase ``compile`` of the refusal table of §14.1). At this commit the one storage constraint
    a case can write is the b6 form ``PatchLayout(distribute_coarse=...)``, which spec 0.3 §21.6
    maps to the plan's distribution; the ``storage_hint`` directive of IR22 comes with the
    schedules of N7.
    """

    rule = "IR22"
    phase = "compile"

    def __init__(self, message, *, subject, constraint):
        self.subject = subject
        self.constraint = constraint
        super().__init__("%s (%s, %s): %s; constraint %s on %s"
                         % (type(self).__name__, self.rule, self.phase, message, constraint,
                            subject))


class TermNotAdditive(ValueError):
    """A typed term of a balance that is not a summand of its right-hand side (R7, §14.1).

    R7 decomposes the right-hand side of a balance at its top-level additive structure: ``+``,
    ``-`` and a product or a quotient by an exact number. A term (the divergence of a flux, a
    declared source, a delivery, a nonconservative product) under any other node, or used as one
    component of a vector, is not a summand: its coverage, delivery and accounting could not be
    derived exactly. The refusal is raised while the equation is traced, and names the term, the
    node that holds it, the line that wrote it and the constructs to write instead.

    The constructor is cooperative: every argument is optional and keyword-only, and a ``site``
    that a subclass set before calling it is kept, so a refusal family with its own sites (the
    user surface) can subclass it.
    """

    rule = "R7"
    phase = "tracing"
    term: Any = None
    construct: Any = None
    site: Any = None

    def __init__(self, message=None, *, term=None, construct=None, site=None):
        # An argument given wins; otherwise a value a subclass set before this call stays.
        for name, value in (("term", term), ("construct", construct), ("site", site)):
            if value is not None:
                setattr(self, name, value)
        if message is not None and term is None and construct is None and site is None:
            # A cooperative subclass has already written its message and set its site.
            super().__init__(message)
            return
        if message is None:
            message = "a term of a balance is not a summand of its right-hand side"
        where = ("" if getattr(self.site, "file", None) is None
                 else "; written at %s:%d" % (self.site.file, self.site.line))
        held = "" if self.term is None else "; term %s in %s" % (self.term, self.construct)
        super().__init__("%s (%s, %s): %s%s%s"
                         % (type(self).__name__, self.rule, self.phase, message, held, where))


class RecursiveGraphFunction(ValueError):
    """A named body whose elaboration reaches its own elaboration again (IR6, §14.1).

    ``cycle`` holds (name, Python origin) for each elaboration of the cycle, the first and the last
    naming the same definition. A recursion is written as ``iterate``.
    """

    rule = "IR6"
    phase = "validate"

    def __init__(self, cycle: tuple) -> None:
        self.cycle = tuple(cycle)
        path = " -> ".join("%s (%s)" % (name, origin) for name, origin in self.cycle)
        super().__init__("RecursiveGraphFunction (IR6, validate): %s reaches its own elaboration "
                         "again: %s; a recursion is written as iterate" % (self.cycle[0][0], path))


__all__ = ["ClassNotReached", "ConservationNotStructural", "CoverageNotExact",
           "DeliveryNotStructural", "GeneralAccessOnPatches", "PropertyNotDerived",
           "RecursiveGraphFunction", "RegionRefusal", "ScheduleIllegal", "TermNotAdditive"]

