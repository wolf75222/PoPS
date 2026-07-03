"""pops.params.constraints -- typed parameter domains (Spec 5 sec.5.12).

A constraint is a typed object (``domain=Positive()``), not a string
(``domain="positive"``). It can ``check`` a value and describe itself; the codegen /
runtime can also enforce it. Inert.
"""
from pops.descriptors import Descriptor


class Constraint(Descriptor):
    category = "constraint"

    def check(self, value, who="value"):
        """Raise ValueError if @p value violates the constraint (override in subclasses)."""
        return True


class Positive(Constraint):
    def check(self, value, who="value"):
        if not (value is None or value > 0):
            raise ValueError("%s must be > 0 (got %r)" % (who, value))
        return True


class NonNegative(Constraint):
    def check(self, value, who="value"):
        if not (value is None or value >= 0):
            raise ValueError("%s must be >= 0 (got %r)" % (who, value))
        return True


class Range(Constraint):
    def __init__(self, lo, hi):
        if lo > hi:
            raise ValueError("Range: lo must be <= hi (got lo=%r hi=%r)" % (lo, hi))
        self.lo = lo
        self.hi = hi

    def options(self):
        return {"lo": self.lo, "hi": self.hi}

    def check(self, value, who="value"):
        if value is not None and not (self.lo <= value <= self.hi):
            raise ValueError("%s must be in [%r, %r] (got %r)" % (who, self.lo, self.hi, value))
        return True


class In(Constraint):
    """Membership in a fixed set of allowed values."""

    def __init__(self, *allowed):
        self.allowed = tuple(allowed)

    def options(self):
        return {"allowed": self.allowed}

    def check(self, value, who="value"):
        if value is not None and value not in self.allowed:
            raise ValueError("%s must be one of %r (got %r)" % (who, self.allowed, value))
        return True


class Interval(Range):
    """A closed interval domain ``[lo, hi]`` (ADC-541: the readable alias of :class:`Range`).

    ``Interval(0.0, 1.0)`` is the same closed-interval constraint as ``Range(0.0, 1.0)``; the name
    reads more naturally on a param's ``domain=``. It is a thin subclass so an ``isinstance(...,
    Range)`` check keeps recognising it, while its own ``name`` distinguishes it in diagnostics.
    """


class OneOf(In):
    """A fixed allowed-value set (ADC-541: the readable alias of :class:`In`).

    ``OneOf("roe", "hll")`` is the same membership constraint as ``In("roe", "hll")``; the name
    reads more naturally on a param's ``domain=``. A thin subclass so ``isinstance(..., In)`` keeps
    recognising it, with its own ``name`` for diagnostics.
    """


__all__ = ["Constraint", "Positive", "NonNegative", "Range", "In", "Interval", "OneOf"]
