"""Outputs of an elaborated Advance (spec 0.3 section 5.2, CC14, T33): ``Output``, ``Product``.

``Output(value, when=None, at=None, write=None)`` describes what a run publishes after the
publications of its accepted steps. ``value(B, interval)`` is a closed body (T33) of the accepted
continuation ``B``; it returns a ``Product`` of arrays and scalars. ``when=every(k)`` makes the
Output due after each publication whose accepted macro step is a multiple of k; ``at`` is a tuple
of physical times or ``every_dt(dt)``, the times k * dt, k = 1, 2, ..., and each of these times is
a frontier that the run lands on (CC14). A run evaluates each of its Outputs once at its start,
on the state from which it starts, then after each publication that is due, once for that
publication, with the achieved time of the publication.

``elaborate`` traces each ``value`` once, into a read-only region of the builder, as it traces
the dt bound of b6: the scalars of a record are generated C++ (CF3), evaluated by the compiled
run outside every step, and the arrays are the accepted state of the block, gathered from every
rank by the core machinery that copies it. The record and the schedule of each Output enter the
Program identity, and the artifact manifest publishes the schedule with the compiled control.

This lot elaborates records of the accepted state and of its cell reductions (``cell_sum``,
``cell_max``, ``cell_min``) and scalar arithmetic on them, with ``when=every(k)`` and ``at``. The
last accepted interval (its ``TimeSpan``, its start state, its increment: ``pops.step_change``),
``B.time`` in a record, other ``when`` bodies, writers of ``pops.lib.output`` and the callbacks
of PF1 (``write=``) are refused with ``NotYetElaborated``, which names the lot that lowers them.
"""
from __future__ import annotations

import keyword
import math
from collections.abc import Callable, Iterator, Mapping
from decimal import Decimal, localcontext
from types import MappingProxyType
from typing import Any

from pops.time._program.traced_method import ClosedBodyViolation, NotYetElaborated
from pops.time.values import ProgramValue

_OUTPUT_READONLY_OPS = frozenset({"state", "reduce", "scalar_op"})
"""Operations of a record: the accepted state, its cell reductions and scalar arithmetic."""


class Product(Mapping):
    """``Product(name=value, ...)``: a record of named values, in the order written (IR1)."""

    __slots__ = ("_fields",)

    def __init__(self, **fields: Any) -> None:
        for name in fields:
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError("Product field names are identifiers, got %r" % (name,))
        object.__setattr__(self, "_fields", MappingProxyType(dict(fields)))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Product is immutable")

    def __getitem__(self, name: str) -> Any:
        return self._fields[name]

    def __getattr__(self, name: str) -> Any:
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(name) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def __repr__(self) -> str:
        return "Product(%s)" % ", ".join("%s=%r" % item for item in self._fields.items())


def _time(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s is a physical time, an int or a float, got %r" % (where, value))
    result = float(value)
    if not math.isfinite(result) or (type(value) is int and result != value):
        raise ValueError("%s must be a finite binary64 time, got %r" % (where, value))
    return result


class Every:
    """``every(k)``: a ``when`` body that is true after each k-th accepted step (CC14, DM13b).

    The count is the accepted macro step of the publication, as b6's ``Every`` reads its
    accepted-step coordinate: a later run continues the cadence of the earlier one.
    """

    __slots__ = ("steps",)

    def __init__(self, steps: int) -> None:
        if isinstance(steps, bool) or type(steps) is not int or not 1 <= steps <= 2**62:
            raise ValueError("every(k) needs an integer k >= 1")
        object.__setattr__(self, "steps", steps)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("every(k) is immutable")

    def __eq__(self, other: object) -> bool:
        return type(other) is Every and other.steps == self.steps

    def __hash__(self) -> int:
        return hash((Every, self.steps))

    def __repr__(self) -> str:
        return "every(%d)" % self.steps


class EveryDt:
    """``every_dt(dt)``: the output times k * dt, k = 1, 2, ..., each a frontier (CC14).

    The time of index k is the binary64 rounding of k times the shortest decimal spelling of
    ``dt``, as b6's ``EveryDt`` lattice (``_every_dt_lattice_time``): ``every_dt(0.1)`` lands on
    the binary64 value written ``0.3``, not on ``3 * 0.1``.
    """

    __slots__ = ("dt",)

    def __init__(self, dt: Any) -> None:
        dt = _time(dt, "every_dt(dt)")
        if not dt > 0.0:
            raise ValueError("every_dt(dt) needs a positive duration, got %r" % (dt,))
        object.__setattr__(self, "dt", dt)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("every_dt(dt) is immutable")

    def __eq__(self, other: object) -> bool:
        return type(other) is EveryDt and other.dt == self.dt

    def __hash__(self) -> int:
        return hash((EveryDt, self.dt))

    def __repr__(self) -> str:
        return "every_dt(%r)" % self.dt

    def time(self, index: int) -> float:
        with localcontext() as context:
            context.prec = 50
            return float(Decimal(repr(self.dt)) * index)

    def times(self, after: float, until: float) -> tuple[float, ...]:
        """The times of the lattice that are after ``after`` and not after ``until``."""
        quotient = after / self.dt
        if not math.isfinite(quotient):
            raise ValueError("every_dt(%r) is below the binary64 resolution of the clock at t=%r"
                             % (self.dt, after))
        index = max(1, math.floor(quotient))
        # The rounded quotient is within a few indices of the first time after `after`, unless
        # the lattice is below the resolution of the clock there.
        for _ in range(4):
            if self.time(index) > after:
                break
            index += 1
        else:
            raise ValueError("every_dt(%r) is below the binary64 resolution of the clock at t=%r"
                             % (self.dt, after))
        times = []
        while True:
            value = self.time(index)
            if not math.isfinite(value) or value > until:
                return tuple(times)
            if times and not value > times[-1]:
                raise ValueError("every_dt(%r) is below the binary64 resolution of the clock "
                                 "at t=%r" % (self.dt, value))
            times.append(value)
            index += 1


def every(k: int) -> Every:
    """``every(k)``: the ``when`` body of an Output that is due after each k-th accepted step."""
    return Every(k)


def every_dt(dt: Any) -> EveryDt:
    """``every_dt(dt)``: the times k * dt, k = 1, 2, ..., for the ``at`` of an Output."""
    return EveryDt(dt)


class Output:
    """``Output(value, when=None, at=None, write=None)``: what a run publishes (CC14)."""

    __slots__ = ("value", "when", "at", "write")

    def __init__(self, value: Callable, when: Any = None, at: Any = None,
                 write: Any = None) -> None:
        if not callable(value):
            raise TypeError("Output(value): a closed body value(B, interval) -> Product, got %r"
                            % (value,))
        if when is not None and type(when) is not Every:
            raise NotYetElaborated(
                "Output(when=%r): a when body other than every(k), a closed Bool body of the "
                "accepted continuation, is elaborated with the predicates of a later lot of N8"
                % (when,))
        if at is not None and type(at) is not EveryDt:
            if isinstance(at, (str, bytes)) or not isinstance(at, (tuple, list)):
                raise TypeError("Output(at=...) is a tuple of physical times or every_dt(dt), "
                                "got %r" % (at,))
            times = tuple(_time(item, "Output(at=...)") for item in at)
            if not times:
                raise ValueError("Output(at=()) names no time; leave at=None instead")
            # Equal times are one time: a publication that reaches it evaluates the Output once.
            at = tuple(sorted(set(times)))
        if write is not None:
            raise NotYetElaborated(
                "Output(write=%r): the writers of pops.lib.output and the callbacks of PF1, "
                "through the native writer interface of section 15.2, are lowered by a later lot "
                "of N8; without write=, result.outputs keeps the published values (CC14)"
                % (write,))
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "when", when)
        object.__setattr__(self, "at", at)
        object.__setattr__(self, "write", write)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Output is immutable")

    def __repr__(self) -> str:
        return "Output(%s, when=%r, at=%r)" % (
            getattr(self.value, "__name__", repr(self.value)), self.when, self.at)


# --- tracing -----------------------------------------------------------------------------------


class _OutputTrace:
    __slots__ = ("program", "active", "region")

    def __init__(self, program: Any, region: int) -> None:
        self.program = program
        self.active = True
        self.region = region

    def require_active(self, what: str) -> None:
        if not self.active:
            raise ClosedBodyViolation(
                "%s of an Output body is read outside the trace of that body (T33)" % what)


class OutputField:
    """The accepted state of a block inside an Output body: published as an array, or reduced."""

    __slots__ = ("_trace", "_value")

    def __init__(self, trace: _OutputTrace, value: ProgramValue) -> None:
        self._trace = trace
        self._value = value

    def _refuse(self, *_: Any) -> Any:
        raise NotYetElaborated(
            "a record publishes the accepted state or a cell reduction of it; an expression of "
            "fields is a definition of the model (2.6), published by a later lot of N8")

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = _refuse
    __neg__ = __pow__ = _refuse

    def __bool__(self) -> bool:
        from pops._ir.symbolic import SymbolicTruthValueError
        raise SymbolicTruthValueError(
            self, suggestion="the truth of a traced state is undecided at elaboration (E1): a "
            "record publishes values; a when body other than every(k) comes with a later lot")

    def __repr__(self) -> str:
        return "OutputField(%s)" % self._value.name


class _OutputContinuation:
    """``B`` inside an Output body: the accepted continuation after the publication (T33)."""

    __slots__ = ("_trace", "_state")

    def __init__(self, trace: _OutputTrace, state: OutputField) -> None:
        self._trace = trace
        self._state = state

    @property
    def state(self) -> OutputField:
        self._trace.require_active("the accepted continuation")
        return self._state

    @property
    def time(self) -> Any:
        raise NotYetElaborated(
            "B.time in a record: each published value already carries the achieved time of its "
            "publication (CC14); a record that computes with the time comes with a later lot "
            "of N8")

    @property
    def memory(self) -> Any:
        raise NotYetElaborated("method memory (section 9, T27) is elaborated by a later lot of N8")


class _Interval:
    """The ``interval`` of an Output body, which this lot does not elaborate."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        raise NotYetElaborated(
            "interval.%s: the last accepted interval of an Output (its TimeSpan, its start state "
            "and its increment, pops.step_change) is elaborated by a later lot of N8" % name)

    def __bool__(self) -> bool:
        raise NotYetElaborated(
            "the last accepted interval of an Output (None at the initial time) is elaborated by "
            "a later lot of N8")


_ACTIVE: list[_OutputTrace] = []


def _field(field: Any, where: str) -> OutputField:
    if type(field) is not OutputField:
        raise TypeError("%s reduces the accepted state of an Output body, B.state; got %r"
                        % (where, field))
    if not _ACTIVE or field._trace is not _ACTIVE[-1]:
        raise ClosedBodyViolation(
            "%s reads a state outside the Output body that traced it (T33)" % where)
    return field


def _component(component: Any, where: str) -> int:
    if isinstance(component, bool) or type(component) is not int or component < 0:
        raise ValueError("%s(component=...) is an integer >= 0, got %r" % (where, component))
    return component


def cell_sum(field: Any, component: int = 0) -> ProgramValue:
    """The sum over the owned active cells of one component of the accepted state (collective)."""
    field = _field(field, "cell_sum")
    return field._trace.program.sum_component(field._value, _component(component, "cell_sum"))


def _extremum(kind: str, field: Any, component: Any) -> ProgramValue:
    field = _field(field, "cell_" + kind)
    program = field._trace.program
    value = field._value
    return program._new("scalar", "reduce", (value,),
                        {"kind": kind, "comp": _component(component, "cell_" + kind)}, None,
                        value.block)


def cell_max(field: Any, component: int = 0) -> ProgramValue:
    """The largest value over the cells of one component of the accepted state (collective)."""
    return _extremum("max", field, component)


def cell_min(field: Any, component: int = 0) -> ProgramValue:
    """The smallest value over the cells of one component of the accepted state (collective)."""
    return _extremum("min", field, component)


class OutputRegion:
    """One traced Output: its builder region and its record, ``(name, "array", block)`` for the
    accepted state of a block and ``(name, "scalar", value)`` for a scalar of the region."""

    __slots__ = ("output", "sub", "record")

    def __init__(self, output: Output, sub: list, record: tuple) -> None:
        self.output = output
        self.sub = sub
        self.record = record

    def scalars(self) -> tuple[ProgramValue, ...]:
        return tuple(item for _, kind, item in self.record if kind == "scalar")

    def record_data(self) -> list[dict[str, Any]]:
        """The record shape: each array names its block, each scalar its index in the entry."""
        rows, scalars = [], 0
        for name, kind, item in self.record:
            if kind == "array":
                rows.append({"name": name, "array": item})
            else:
                rows.append({"name": name, "scalar": scalars})
                scalars += 1
        return rows

    def schedule_data(self) -> dict[str, Any]:
        """The schedule and the record shape that the compiled control publishes."""
        at = self.output.at
        return {
            "when": None if self.output.when is None else {"every": self.output.when.steps},
            "at": (None if at is None else {"every_dt": at.dt.hex()} if type(at) is EveryDt
                   else {"times": [time.hex() for time in at]}),
            "record": self.record_data(),
        }


def trace_output(program: Any, temporal: Any, output: Output) -> OutputRegion:
    """Trace ``output.value`` once into a read-only region of ``program``."""
    from pops._ir.symbolic import SymbolicTruthValueError
    from pops.time._authoring import readonly_authoring_query
    from pops.time._program.dt_bound import readonly_query_nodes
    from pops.time._program.value_validation import require_region
    from pops.time.references import block_name

    if type(output) is not Output:
        raise TypeError("elaborate(outputs=...) holds Output values, got %r" % (output,))
    state = program._current_time_value(temporal)
    sub: list = []
    with readonly_authoring_query(program):
        program._recording.append(sub)
        trace = _OutputTrace(program, program._region_for_block(sub))
        _ACTIVE.append(trace)
        try:
            result = output.value(_OutputContinuation(trace, OutputField(trace, state)),
                                  _Interval())
        except (TypeError, ValueError) as error:
            if isinstance(error, (ClosedBodyViolation, SymbolicTruthValueError)):
                raise
            raise type(error)("Output body %s: %s" % (
                getattr(output.value, "__name__", "value"), error)) from error
        finally:
            _ACTIVE.pop()
            trace.active = False
            program._recording.pop()
        if type(result) is not Product:
            raise TypeError("an Output body returns a Product of arrays and scalars, got %r"
                            % (result,))
        if not len(result):
            raise ValueError("an Output body returns a Product with at least one value")
        record = []
        for name, value in result.items():
            if type(value) is OutputField:
                if value._trace is not trace:
                    raise ClosedBodyViolation(
                        "Output record %r holds a state of another body (T33)" % name)
                record.append((name, "array", block_name(value._value.block)))
            elif isinstance(value, ProgramValue) and value.vtype == "scalar":
                require_region(program, value, trace.region, "Output record %r" % name,
                               vtype="scalar")
                record.append((name, "scalar", value))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                raise TypeError("Output record %r is the number %r: a record publishes values "
                                "of the accepted state; a constant is not one" % (name, value))
            else:
                raise ClosedBodyViolation(
                    "Output record %r holds %r, which is neither the accepted state nor a "
                    "scalar that the body computed from it (T33)" % (name, value))
        readonly_query_nodes(program, sub, where="Output", ops=_OUTPUT_READONLY_OPS)
    return OutputRegion(output, sub, tuple(record))


__all__ = ["Every", "EveryDt", "Output", "OutputField", "OutputRegion", "Product", "cell_max",
           "cell_min", "cell_sum", "every", "every_dt", "trace_output"]
