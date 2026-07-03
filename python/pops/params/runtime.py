"""pops.params.runtime -- typed scalar parameters (Spec 5 sec.5.12).

A parameter declares whether it is compile-time or runtime, its typed dtype (from
:mod:`pops.math`), an optional default, and an optional typed domain constraint -- instead
of the string form ``Param(kind="runtime")`` / ``domain="positive"``. These are inert
descriptors; the codegen / runtime consume them (a runtime param appears in
``compiled.arguments()``; a const param participates in the cache key).
"""
from pops.descriptors import Descriptor, reject_string_selector
from pops.math import Real


def _check_domain(name, domain):
    """Refuse a bare-string ``domain=`` (Spec 5 sec.7); pass a typed constraint / None through.

    ``domain="positive"`` is the anti-pattern a typed domain replaces: it is rejected via
    :func:`reject_string_selector` naming the typed alternative (``pops.params.Positive()`` ...).
    A typed :class:`~pops.params.constraints.Constraint` or ``None`` passes through unchanged.
    """
    if isinstance(domain, str):
        reject_string_selector(
            domain, "%s domain" % name,
            "a typed pops.params domain (Positive() / NonNegative() / Interval(lo, hi) / "
            "OneOf(...)), not a string")  # always raises
    return domain


def _domain_error(name, domain, value, phase):
    """The 4-part domain-violation message (ADC-541): param / expected domain / value / phase.

    Every domain refusal -- at ``compile`` (the declared default) or at ``bind`` (a supplied value)
    -- reads the same shape so a caller can see WHICH param, its EXPECTED domain, the RECEIVED value
    and the PHASE the check ran in. Returns the message string (the caller raises)."""
    expected = getattr(domain, "options", lambda: {})() or getattr(domain, "name", domain)
    return ("param %r: value %r is outside the expected domain %s (%s) at the %s phase"
            % (name, value, getattr(domain, "name", domain), expected, phase))


class RuntimeParam(Descriptor):
    """A runtime parameter: changeable without recompilation if the ABI is unchanged.

    ``RuntimeParam("alpha", dtype=Real, default=1.0, domain=Positive())``. It appears in
    ``compiled.arguments()`` and is set at bind time; it does NOT participate in the codegen
    hash (changing it must not force a recompile while the ABI holds). A bare-string ``domain=``
    is refused at construction (Spec 5 sec.7).
    """

    category = "runtime_param"

    def __init__(self, name, dtype=Real, default=None, domain=None):
        self._name = str(name)
        self.dtype = dtype
        self.default = default
        self.domain = _check_domain(self._name, domain)

    @property
    def name(self):
        return self._name

    def options(self):
        return {"name": self._name, "dtype": getattr(self.dtype, "name", self.dtype),
                "default": self.default,
                "domain": self.domain.name if self.domain is not None else None}

    def capabilities(self):
        return {"runtime": True, "compile_time": False}

    def validate(self, context=None):
        """Validate the declared DEFAULT against the domain (the ``compile`` phase)."""
        super().validate(context)  # honour the explainable available() route check too
        if self.domain is not None and self.default is not None:
            try:
                self.domain.check(self.default, who="%s default" % self._name)
            except ValueError:
                raise ValueError(_domain_error(self._name, self.domain, self.default, "compile"))
        return True

    def check_bind(self, value):
        """Validate a value SUPPLIED at bind time against the domain (ADC-541, the ``bind`` phase).

        A runtime param is set at ``System.install`` / bind; a bound value outside the declared
        domain is refused HERE (before the runtime uses it) with the 4-part diagnostic (param /
        expected domain / received value / phase). A missing value (``None``) with no default is
        refused too -- a runtime param with no default MUST be supplied. Returns True on success.
        """
        if value is None:
            if self.default is None:
                raise ValueError(
                    "param %r: a value is required at the bind phase (no default declared); supply "
                    "it at System.install / bind" % self._name)
            value = self.default
        if self.domain is not None:
            try:
                self.domain.check(value, who=self._name)
            except ValueError:
                raise ValueError(_domain_error(self._name, self.domain, value, "bind"))
        return True


class ConstParam(Descriptor):
    """A compile-time constant: frozen into the generated code; in the cache key.

    ``ConstParam("gamma", value=5.0/3.0)``. Changing it can require a recompile (it changes
    the codegen hash). Use a :class:`RuntimeParam` for values that must change at run time.
    """

    category = "const_param"

    def __init__(self, name, value, dtype=Real):
        self._name = str(name)
        self.value = value
        self.dtype = dtype

    @property
    def name(self):
        return self._name

    def options(self):
        return {"name": self._name, "value": self.value,
                "dtype": getattr(self.dtype, "name", self.dtype)}

    def capabilities(self):
        return {"runtime": False, "compile_time": True, "in_cache_key": True}


class DerivedParam(Descriptor):
    """A parameter derived from others by a PoPS expression (computed in C++, not Python)."""

    category = "derived_param"

    def __init__(self, name, expression):
        self._name = str(name)
        self.expression = expression

    @property
    def name(self):
        return self._name

    def options(self):
        return {"name": self._name,
                "expression": getattr(self.expression, "name", repr(self.expression))}


__all__ = ["RuntimeParam", "ConstParam", "DerivedParam"]
