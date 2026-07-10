"""pops.time scheduler annotations (Spec 3 unified scheduler).

``Schedule`` is an inert IR annotation deciding WHEN a node is due and what to do when it is
not; the module helpers (``always`` / ``every`` / ``when`` / ``on_start`` / ``on_end`` /
``subcycle``) build the kinds. Authoring only.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pops.time.value_metadata import _freeze_attr


def _freeze_param(value: Any) -> Any:
    # Runtime output policies deliberately accept a callable predicate; Program schedules reject
    # such predicates at their explicit lowerability gate. Every data leaf is otherwise strict.
    if callable(value):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_param(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_param(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_param(item) for item in value)
    return _freeze_attr(value)


class Schedule:
    """When a Program node is due, and what to do when it is not (Spec 3 unified scheduler).

    A Schedule is an inert IR annotation recorded on a node (``ProgramValue.attrs['schedule']``). The
    ``kind`` decides WHEN the node is due (``always`` every step, ``every(N)``, ``when(cond)``,
    ``on_start`` / ``on_end``, ``subcycle``); the ``policy`` decides what happens when it is NOT
    due (``recompute`` the default, ``hold`` the cached value, ``skip``, ``zero``,
    ``accumulate_dt``, or ``error``). Build a kind with the module helpers and set the policy by
    chaining: ``every(10).hold()``.

    Only ``always()`` runs at ``sim.step`` today: the runtime that honors a non-trivial schedule
    (the typed cache, ``accumulate_dt``, the checkpoint) is the C++ part of ADC-458, so a node
    carrying a non-always schedule is recorded and inspectable but refuses to lower (it is never
    silently ignored). See ``docs/sphinx/reference/program-scheduler.md``.
    """

    _KINDS = ("always", "every", "when", "on_start", "on_end", "subcycle")
    _POLICIES = ("recompute", "hold", "skip", "zero", "accumulate_dt", "error")
    # policies that reuse a stored value, so the operator must be cacheable
    _CACHING = ("hold", "accumulate_dt")
    __pops_ir_immutable__ = True
    # ADC-642: each kind decoded ONCE. so_lowerable = a compiled sim.step(dt) loop can evaluate the
    # due-test (on_end cannot: no end-of-run signal reaches the .so). host_cadence = the host output
    # driver can fire it (subcycles are internal to the native macro step, invisible to the run-loop
    # hook). Codegen and the output driver READ these instead of re-listing kind strings; the per-kind
    # C++ / host bodies stay in their own layers.
    _KIND_FACTS = {
        "always":   {"so_lowerable": True,  "host_cadence": True},
        "every":    {"so_lowerable": True,  "host_cadence": True},
        "when":     {"so_lowerable": True,  "host_cadence": True},
        "on_start": {"so_lowerable": True,  "host_cadence": True},
        "on_end":   {"so_lowerable": False, "host_cadence": True},
        "subcycle": {"so_lowerable": True,  "host_cadence": False},
    }

    __slots__ = ("kind", "policy", "params")

    def __init__(self, kind: Any, policy: Any = "recompute", **params: Any) -> None:
        if kind not in Schedule._KINDS:
            raise ValueError("schedule kind %r must be one of %s"
                             % (kind, ", ".join(Schedule._KINDS)))
        if policy not in Schedule._POLICIES:
            raise ValueError("schedule policy %r must be one of %s"
                             % (policy, ", ".join(Schedule._POLICIES)))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "params", _freeze_param(params))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Schedule is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Schedule is immutable")

    def is_always(self) -> Any:
        """True for the default cadence (every step, recompute) -- the only schedule that lowers."""
        return self.kind == "always" and self.policy == "recompute"

    def needs_cache(self) -> Any:
        """True if the policy reuses a stored value (so the operator must be cacheable)."""
        return self.policy in Schedule._CACHING

    def so_lowerable(self) -> bool:
        """True when a compiled sim.step(dt) loop can evaluate this kind's due-test (ADC-642)."""
        return Schedule._KIND_FACTS[self.kind]["so_lowerable"]

    def host_cadence(self) -> bool:
        """True when the host output run-loop can honor this kind's cadence (ADC-642)."""
        return Schedule._KIND_FACTS[self.kind]["host_cadence"]

    def _with_policy(self, policy: Any) -> Any:
        return Schedule(self.kind, policy=policy, **self.params)

    def recompute(self) -> Any:
        """A copy whose off-cadence policy re-evaluates the node (the default)."""
        return self._with_policy("recompute")

    def hold(self) -> Any:
        """A copy whose off-cadence policy reuses the cached value (needs a cacheable op)."""
        return self._with_policy("hold")

    def skip(self) -> Any:
        """A copy whose off-cadence policy skips the node entirely."""
        return self._with_policy("skip")

    def zero(self) -> Any:
        """A copy whose off-cadence policy substitutes a zero value."""
        return self._with_policy("zero")

    def accumulate_dt(self) -> Any:
        """A copy whose off-cadence policy accumulates dt until the node is next due."""
        return self._with_policy("accumulate_dt")

    def error(self) -> Any:
        """A copy whose off-cadence policy raises: the node must never run off cadence."""
        return self._with_policy("error")

    def __repr__(self) -> str:
        if self.kind == "every":
            base = "every(%r)" % (self.params.get("n"),)
        elif self.kind == "subcycle":
            base = "subcycle(%r)" % (self.params.get("count"),)
        elif self.kind == "when":
            base = "when(...)"
        else:
            base = "%s()" % self.kind
        return base if self.policy == "recompute" else "%s.%s()" % (base, self.policy)


# A new kind that forgets its facts row fails loudly at import, not silently at a consumer (ADC-642).
assert set(Schedule._KIND_FACTS) == set(Schedule._KINDS)


def always() -> Any:
    """Due every step, recomputed -- the default cadence (the only schedule that runs today)."""
    return Schedule("always")


def every(n: Any) -> Any:
    """Due every ``n`` macro-steps (``n`` a positive int)."""
    if isinstance(n, bool) or not (isinstance(n, int) and n > 0):
        raise ValueError("every(n): n must be a positive int, got %r" % (n,))
    return Schedule("every", n=n)


def when(cond: Any) -> Any:
    """Due when the runtime condition ``cond`` holds (a Program Bool value or a callable)."""
    return Schedule("when", cond=cond)


def on_start() -> Any:
    """Due only at the first step."""
    return Schedule("on_start")


def on_end() -> Any:
    """Due only at the last step."""
    return Schedule("on_end")


def subcycle(count: Any, dt: Any = None) -> Any:
    """Structured sub-cycling: ``count`` inner steps (of ``dt`` each, default ``macro_dt/count``)."""
    if isinstance(count, bool) or not (isinstance(count, int) and count > 0):
        raise ValueError("subcycle(count): count must be a positive int, got %r" % (count,))
    return Schedule("subcycle", count=count, dt=dt)
