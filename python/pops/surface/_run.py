"""Compile, bind, run and publish a case of the surface (spec 0.3, 2.6.2, 2.6.6, 2.6.8; §5.2, CC14, Q3).

Lot G of N15 (the run of the vertical slice of Phase 1, OD31). It elaborates, on the compiled run
of N8 (``pops.runtime._simulation``), these constructs of the surface:

* ``pops.compile(case, target=pops.OpenMP())``: b6's ``compile(resolve(validate(case)))`` of the
  elaborated case. ``target=`` has no default (2.6.9 lists none) and is refused when it is left
  out; ``pops.CUDA()`` and ``mpi=`` are later lots (G13, Q03a, N10);
* ``pops.bind(artifact, params={nu: v}, data={u0: array}, resources=pops.Resources(threads=n),
  directory=d)``: each array is given in the order of the support's axes, ``[i, j]`` being the
  cell (x_i, y_j), and is transposed to b6's layout (Q3); the threads are fixed by the first bind
  of a process, 1 when it is left out, as master (§2.6.5, §12.1): a later bind that asks for
  another count runs with the first and reports both counts in ``diagnostics``;
* ``simulation.run(until=t, max_steps=n)``: N8's ``Simulation.run`` from the first proposal of
  the advance, ``max_steps`` 1_000_000 when it is left out; ``result.state[u]`` is a NumPy array in
  the order of the support's axes; a failed run raises ``pops.RunFailed``, or its variant
  ``pops.BudgetExhausted`` or ``pops.SolveFailed``, whose ``result`` is the surface ``RunResult``;
* ``pops.Output(pops.record(u=u), when=schedule, write=writer)`` and ``pops.every(time=h)`` or
  ``pops.every(steps=k)``: an Output of N8 whose body reads the accepted state. A time or a list
  of times gives ``at=`` and the run publishes at those times only; ``every(time=h)`` publishes at
  the start t0 of the run and at t0 + k h, ``every(steps=k)`` at the start and after every k-th
  accepted step. N8 lowers both on a lattice fixed when the case is compiled, k h and the k-th
  macro step from the start of the simulation, which is the lattice of the spec only when the
  run starts on it: a later run whose start is off the lattice is refused before it runs
  (``NotYetElaborated``, naming the N8 lot 'run-anchored schedules', ROM-89, which owes the
  anchoring t0 + k h before the exit of phase 1), and a later run that starts on it publishes
  its times and steps after its start, the start being the end of the run before it, which that run published
  (review of lot G, F1). ``write=None`` keeps the publications in ``result.outputs`` (2.6.9); a writer of
  ``pops.lib.output`` writes the whole series of the simulation after each run, outside every step
  (CF9), relative to ``directory=``.

Each other option of these constructs refuses with ``NotYetElaborated`` and the lot that
elaborates it.
"""
from __future__ import annotations

import math as _math
import os
from typing import Any

import numpy as _np

from ._model import Datum, Model, Parameter, Quantity
from ._refusal import AmbiguousElaboration, NotYetElaborated, surface_entry, user_site

_GPU = "a later lot of N15 with the device storage of G13 and the GPU run of Q03a"
_MPI = "a later lot of N15 with the distributed runs of N10"
_RESTART = "a later lot of N15 with the continuation and restart of G07 (CN11)"
_DEFINITIONS = "a later lot of N15 with the definitions of R16 and the reductions of N8"
_PREDICATE = "a later lot of N8 with the predicates of the when bodies (CC14)"
_DEFAULT_MAX_STEPS = 1_000_000
_LATTICE = ("the N8 lot 'run-anchored schedules' (ROM-89), owed before the exit of phase 1: N8 "
            "fixes the grid of every(time=h) and every(steps=k) at compile time, from the start "
            "of the simulation, not of the run (spec 2.6.2)")


# --- records and schedules ----------------------------------------------------------------------

class Record:
    """``pops.record(a=x, b=y)``: named values, in the order written (IR1, SU14)."""

    __slots__ = ("_fields", "site")

    def __init__(self, fields: dict, site: Any) -> None:
        self._fields = tuple(fields.items())
        self.site = site

    def items(self) -> tuple:
        return self._fields

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "site":
            raise AttributeError(name)
        fields = object.__getattribute__(self, "_fields")
        for key, value in fields:
            if key == name:
                return value
        raise AttributeError("the record has the fields %s; .%s is none of them"
                             % (", ".join(key for key, _ in fields), name))

    def __repr__(self) -> str:
        return "pops.record(%s)" % ", ".join("%s=%r" % item for item in self._fields)


@surface_entry
def record(**fields: Any) -> Record:
    """``pops.record(a=x, b=y)``: a record of named values (IR1)."""
    if not fields:
        raise TypeError("pops.record(...) takes at least one field, pops.record(u=u)")
    for name in fields:
        if name.startswith("_"):
            raise ValueError("pops.record(%s=...): a field name does not start with _" % name)
    return Record(dict(fields), user_site(2))


class Every:
    """``pops.every(time=h)`` or ``pops.every(steps=k)``: the schedule of an Output (SD43)."""

    __slots__ = ("time", "steps")

    def __init__(self, time: Any, steps: Any) -> None:
        self.time = time
        self.steps = steps

    def __repr__(self) -> str:
        return ("pops.every(time=%r)" % self.time if self.time is not None
                else "pops.every(steps=%r)" % self.steps)


@surface_entry
def every(*, time: Any = None, steps: Any = None) -> Every:
    """``pops.every(time=h)``: at the start t0 of a run and at t0 + k h; ``pops.every(steps=k)``:
    at the start and after every k-th accepted step (2.6.2, CC14)."""
    if (time is None) == (steps is None):
        raise TypeError("pops.every takes time=h or steps=k, one of them")
    if time is not None:
        if isinstance(time, bool) or not isinstance(time, (int, float)) \
                or not _math.isfinite(float(time)) or not float(time) > 0.0:
            raise ValueError("pops.every(time=h) takes a positive finite duration, got %r"
                             % (time,))
        return Every(float(time), None)
    if isinstance(steps, bool) or type(steps) is not int or steps < 1:
        raise ValueError("pops.every(steps=k) takes an integer k >= 1, got %r" % (steps,))
    return Every(None, steps)


def _time(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s is a physical time, a number, got %r" % (where, value))
    result = float(value)
    if not _math.isfinite(result):
        raise ValueError("%s is a finite time, got %r" % (where, value))
    return result


class Output:
    """``pops.Output(pops.record(u=u), when=schedule, write=writer)`` (2.6.2, CC14)."""

    __slots__ = ("record", "when", "write", "site", "times", "every_time", "every_steps")

    @surface_entry
    def __init__(self, value: Any = None, *, when: Any = None, write: Any = None,
                 **earlier: Any) -> None:
        self.site = user_site(2)
        if "at" in earlier:
            from ._refusal import SpellingRefusal

            raise SpellingRefusal("when=t, when=[t1, t2] or when=pops.every(time=h)").refuse(
                "pops.Output(at=...)", "pops.Output")
        if earlier:
            raise TypeError("pops.Output has no option %s" % ", ".join(sorted(earlier)))
        if callable(value) and not isinstance(value, Record):
            from ._refusal import SpellingRefusal

            raise SpellingRefusal("pops.Output(pops.record(u=u), when=...), a record of what "
                                  "the run publishes").refuse("pops.Output(<a function>)",
                                                              "pops.Output")
        if not isinstance(value, Record):
            raise TypeError("pops.Output takes a record of what the run publishes, "
                            "pops.Output(pops.record(u=u), when=...), got %r" % (value,))
        for name, item in value.items():
            if isinstance(item, Quantity):
                continue
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                raise TypeError("pops.Output(pops.record(%s=%r)): an output publishes quantities "
                                "of the model; a number is the metadata= of a writer"
                                % (name, item))
            raise NotYetElaborated("pops.Output(pops.record(%s=%r)), a record of something "
                                   "other than an evolved quantity" % (name, item), _DEFINITIONS)
        if when is None:
            raise AmbiguousElaboration(
                "pops.Output(...) has no schedule", site=self.site,
                write="when=t, when=[t1, t2], when=pops.every(time=h) or "
                      "when=pops.every(steps=k)")
        self.times = self.every_time = self.every_steps = None
        if isinstance(when, Every):
            self.every_time, self.every_steps = when.time, when.steps
        elif isinstance(when, list):
            if not when:
                raise ValueError("pops.Output(when=[...]) takes at least one time")
            if any(isinstance(item, (Every, list)) for item in when):
                raise NotYetElaborated("pops.Output(when=[...]) with a list of schedules",
                                       "a later lot of N15 with the run of N8")
            self.times = tuple(sorted({_time(item, "pops.Output(when=[...])") for item in when}))
        elif isinstance(when, tuple):
            raise TypeError("pops.Output(when=...) takes a list of times, when=[t1, t2]; a tuple "
                            "is a point, a size or an interval (SU14)")
        elif callable(when):
            raise NotYetElaborated("pops.Output(when=<a predicate>)", _PREDICATE)
        else:
            self.times = (_time(when, "pops.Output(when=t)"),)
        if write is not None and not (callable(getattr(write, "write_series", None))
                                      and callable(getattr(write, "check", None))):
            raise TypeError("pops.Output(write=...) takes a writer of pops.lib.output, such as "
                            "npz(path), got %r" % (write,))
        if write is not None and callable(getattr(write, "check_record", None)):
            write.check_record(value)
        self.record = value
        self.when = when
        self.write = write

    def quantities(self) -> tuple:
        return tuple(item for _, item in self.record.items())

    def lowered(self, block: Any) -> Any:
        """The Output of N8: its body reads the accepted state of ``block`` once per field."""
        from pops.time._program import outputs as _n8

        names = tuple(name for name, _ in self.record.items())

        def value(B: Any, interval: Any) -> Any:
            return _n8.Product(**{name: B.state for name in names})

        value.__name__ = "output_%s" % "_".join(names)
        if self.every_steps is not None:
            return _n8.Output(value, when=_n8.every(self.every_steps))
        if self.every_time is not None:
            return _n8.Output(value, at=_n8.every_dt(self.every_time))
        return _n8.Output(value, at=self.times)

    def keeps(self, time: float, first_run: bool, publication_index: int) -> bool:
        """Whether a publication of N8 at ``time`` is one of this Output's. N8 publishes once at
        the start of each run: the start of the first run of a simulation is its start, which
        a schedule ``every`` keeps and a list of times keeps when it holds that time; the start of
        a later run is the end of the run before it, published already or off the schedule."""
        if publication_index == 0 and not first_run:
            return False
        if self.times is None:
            return True
        return time in self.times

    def __repr__(self) -> str:
        return "pops.Output(%r, when=%r%s)" % (self.record, self.when, "" if self.write is None
                                                 else ", write=%r" % (self.write,))


# --- targets, resources and the artifact --------------------------------------------------------

class OpenMP:
    """``pops.OpenMP()``: the CPU target, Kokkos on its host execution space."""

    __slots__ = ()

    @surface_entry
    def __init__(self, *, mpi: Any = None, substitutions: Any = None) -> None:
        if mpi is not None:
            raise NotYetElaborated("pops.OpenMP(mpi=...)", _MPI)
        if substitutions is not None:
            raise NotYetElaborated("pops.OpenMP(substitutions=...), the qualified substitutions "
                                   "of §11.5", "a later lot of N15 with N9")

    def __repr__(self) -> str:
        return "pops.OpenMP()"


class CUDA:
    """``pops.CUDA()``: the GPU target; refused until the GPU run of Q03a."""

    __slots__ = ()

    @surface_entry
    def __init__(self, *, mpi: Any = None, substitutions: Any = None) -> None:
        raise NotYetElaborated("pops.CUDA(), the GPU target", _GPU)


class Resources:
    """``pops.Resources(threads=n)``: the threads of the run (§12.1)."""

    __slots__ = ("threads",)

    @surface_entry
    def __init__(self, *, threads: Any = None, history: Any = None, communicator: Any = None,
                 capacity: Any = None) -> None:
        if communicator is not None:
            raise NotYetElaborated("pops.Resources(communicator=...)", _MPI)
        if history is not None:
            raise NotYetElaborated("pops.Resources(history=...) (B3)", _RESTART)
        if capacity is not None:
            raise NotYetElaborated("pops.Resources(capacity=...) (B5)",
                                   "a later lot of N15 with the capacities of G02")
        if threads is not None and (isinstance(threads, bool) or type(threads) is not int
                                    or threads < 1):
            raise ValueError("pops.Resources(threads=n) takes an integer n >= 1, got %r"
                             % (threads,))
        self.threads = 1 if threads is None else threads

    def __repr__(self) -> str:
        return "pops.Resources(threads=%d)" % self.threads


class Artifact:
    """What ``pops.compile`` returns: the compiled artifact of b6 and the case it compiles."""

    __slots__ = ("case", "compiled", "target")

    def __init__(self, case: Any, compiled: Any, target: Any) -> None:
        self.case = case
        self.compiled = compiled
        self.target = target

    def __repr__(self) -> str:
        return "<artifact of case %r for %r>" % (self.case.model.name, self.target)


@surface_entry
def compile(case: Any = None, *, target: Any = None, schedules: Any = None) -> Artifact:  # noqa: A001
    """``pops.compile(case, target=pops.OpenMP())``: validate, resolve and compile the case."""
    from ._program import Case

    if not isinstance(case, Case):
        raise TypeError("pops.compile takes a pops.Case, got %r" % (case,))
    if schedules is not None:
        raise NotYetElaborated("pops.compile(schedules=...), the schedules of §11",
                               "a later lot of N15 with the lowering classes of N7")
    if target is None:
        raise AmbiguousElaboration("pops.compile(case) has no target", site=user_site(2),
                                   write="pops.compile(case, target=pops.OpenMP())")
    if not isinstance(target, OpenMP):
        raise TypeError("pops.compile(target=...) takes pops.OpenMP(), got %r" % (target,))
    from pops._api import compile as _compile
    from pops._api import resolve, validate

    elaborated = case.elaborated
    plan = resolve(validate(elaborated.case), layout=elaborated.layout)
    return Artifact(case, _compile(plan), target)


# --- bind ---------------------------------------------------------------------------------------

def _first_threads(threads: int) -> int:
    """Fix the threads of this process at ``threads`` unless its execution space has started; the
    count it runs with."""
    from pops.runtime import _threading

    native = _threading._selected_native_module()
    started = _threading._first_system_built if native is None else getattr(
        native, "kokkos_is_initialized", lambda: _threading._first_system_built)()
    if not started:
        _threading.set_threads(threads)
        return threads
    return int(os.environ.get("KOKKOS_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
               or threads)


def to_layout(array: Any, cells: tuple, where: str) -> Any:
    """An array in the order of the support's axes, as b6 stores a state of one component: the
    component first, then the axes from the last to the first (Q3)."""
    value = _np.asarray(array, dtype=_np.float64)
    if value.shape != tuple(cells):
        raise ValueError("%s has the shape %r; the cells of its support are %r, in the order of "
                         "its axes" % (where, value.shape, tuple(cells)))
    return _np.ascontiguousarray(_np.transpose(value)[_np.newaxis, ...])


def from_layout(array: Any, cells: tuple) -> Any:
    """The inverse of ``to_layout``: a state of one component of b6 in the order of the axes."""
    value = _np.asarray(array, dtype=_np.float64)
    expected = (1,) + tuple(reversed(tuple(cells)))
    if value.shape != expected:
        if value.ndim != 1 or value.size != _math.prod(expected):
            raise ValueError("a state of b6 has the shape %r or is flat; got %r"
                             % (expected, value.shape))
        value = value.reshape(expected)
    return _np.ascontiguousarray(_np.transpose(value[0]))


@surface_entry
def bind(artifact: Any = None, *, params: Any = None, data: Any = None, resources: Any = None,
         directory: Any = None, restart: Any = None, migrate: Any = None) -> Simulation:
    """``pops.bind(artifact, params={nu: v}, data={u0: array}, resources=..., directory=d)``."""
    if not isinstance(artifact, Artifact):
        raise TypeError("pops.bind takes what pops.compile returns, got %r" % (artifact,))
    if restart is not None or migrate is not None:
        raise NotYetElaborated("pops.bind(%s=...)" % ("restart" if restart is not None
                                                       else "migrate"), _RESTART)
    if resources is None:
        resources = Resources()
    if not isinstance(resources, Resources):
        raise TypeError("pops.bind(resources=...) takes pops.Resources(threads=n), got %r"
                        % (resources,))
    directory = os.path.abspath(os.getcwd() if directory is None else os.fspath(directory))
    case = artifact.case
    elaborated = case.elaborated
    model: Model = case.model
    params = {} if params is None else params
    if not isinstance(params, dict):
        raise TypeError("pops.bind(params=...) takes a dictionary keyed by parameters, "
                        "params={nu: value}")
    bound_params = {}
    for key, value in params.items():
        if isinstance(key, str):
            raise TypeError("pops.bind(params=...) is keyed by handles, not by the string %r "
                            "(SU8)" % key)
        if not isinstance(key, Parameter) or key.model is not model:
            raise TypeError("pops.bind(params=...) is keyed by the parameters of the case's "
                            "model, got %r" % (key,))
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("pops.bind(params={%s: v}) takes a number, got %r" % (key, value))
        bound_params[elaborated.block[elaborated.params[key]]] = float(value)
    data = {} if data is None else data
    if not isinstance(data, dict):
        raise TypeError("pops.bind(data=...) takes a dictionary keyed by data, data={u0: array}")
    (support, uniform), = case.layout
    cells = tuple(uniform.cells)
    initial: dict = {}
    for key, value in data.items():
        if isinstance(key, str):
            raise TypeError("pops.bind(data=...) is keyed by handles, not by the string %r (SU8)"
                            % key)
        if not isinstance(key, Datum) or key.model is not model:
            raise TypeError("pops.bind(data=...) is keyed by the data of the case's model, got %r"
                            % (key,))
        quantities = [quantity for quantity, datum in case.program.initial if datum is key]
        if not quantities:
            raise NotYetElaborated("pops.bind(data={%s: ...}) for a datum that no initial value "
                                   "reads" % key, "a later lot of N15 with the data fields of Q3")
        initial[key] = to_layout(value, cells, "pops.bind(data={%s: array})" % key)
    supplied: dict = {}
    bindings = tuple(artifact.compiled.plan.initial_condition_plan.bindings)
    (quantity,) = tuple(elaborated.states)          # one evolved quantity at this level
    for binding in bindings:
        datum = case.program.initial_of(quantity)
        if datum not in initial:
            raise AmbiguousElaboration(
                "the datum %s, the initial value of %s, is not supplied" % (datum, quantity),
                write="pops.bind(artifact, data={%s: array})" % datum, site=user_site(2))
        supplied[binding.subject] = initial[datum]
    threads = _first_threads(resources.threads)
    from pops._api import bind as _bind
    from pops.runtime._simulation import Simulation as _Simulation

    runtime = _bind(artifact.compiled, initial_values=supplied, params=bound_params)
    return Simulation(artifact, _Simulation(runtime), directory,
                      {"asked": resources.threads, "running": threads}, cells)


# --- the run ------------------------------------------------------------------------------------

class RunFailed(RuntimeError):
    """A run whose result holds a failure (T39); ``result`` is the surface ``RunResult``."""

    def __init__(self, result: RunResult, message: str) -> None:
        super().__init__(message)
        self.result = result


class BudgetExhausted(RunFailed):
    """``max_steps`` of the run or the attempts of a step were spent (T39)."""


class SolveFailed(RunFailed):
    """A solve of the trial failed at a fixed failure site (T39)."""


_VARIANTS = {"BudgetExhausted": BudgetExhausted, "SolveFailed": SolveFailed}


class Publication:
    """One publication of an Output: its achieved time, its macro step and its record."""

    __slots__ = ("time", "step", "values")

    def __init__(self, time: float, step: int, values: dict) -> None:
        self.time = time
        self.step = step
        self.values = values

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def __repr__(self) -> str:
        return "Publication(time=%r, step=%d, %s)" % (self.time, self.step,
                                                      ", ".join(self.values))


class RunResult:
    """``result.state[u]``, ``result.time``, ``result.steps``, ``result.outputs``,
    ``result.diagnostics`` and ``result.failure`` (CC14)."""

    __slots__ = ("state", "time", "steps", "rejections", "outputs", "diagnostics", "failure",
                 "continuation")

    def __init__(self, **fields: Any) -> None:
        for name in self.__slots__:
            object.__setattr__(self, name, fields[name])

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("RunResult is immutable")

    def __repr__(self) -> str:
        return "RunResult(time=%r, steps=%d, failure=%r)" % (self.time, self.steps, self.failure)


class _State:
    """``result.state[u]``: a table keyed by the evolved quantities (SU14)."""

    __slots__ = ("_values",)

    def __init__(self, values: tuple) -> None:
        self._values = values

    def __getitem__(self, quantity: Any) -> Any:
        if isinstance(quantity, str):
            raise TypeError("result.state is keyed by handles, not by the string %r (SU8)"
                            % quantity)
        for key, value in self._values:
            if key is quantity:
                return value
        raise KeyError("result.state holds %s; %r is none of them"
                       % (", ".join(str(key) for key, _ in self._values), quantity))

    def __iter__(self) -> Any:
        return iter(key for key, _ in self._values)

    def __len__(self) -> int:
        return len(self._values)


class Simulation:
    """What ``pops.bind`` returns; ``simulation.run(until=t, max_steps=n)`` runs it (§5.2)."""

    __slots__ = ("artifact", "_simulation", "directory", "_threads", "_cells", "_series",
                 "_written", "_runs")

    def __init__(self, artifact: Artifact, simulation: Any, directory: str, threads: dict,
                 cells: tuple) -> None:
        self.artifact = artifact
        self._simulation = simulation
        self.directory = directory
        self._threads = threads
        self._cells = cells
        outputs = artifact.case.outputs
        self._series: dict = {output: [] for output in outputs}
        self._written: set = set()
        self._runs = 0

    @surface_entry
    def run(self, *, until: Any = None, max_steps: Any = None, **earlier: Any) -> RunResult:
        """Run until the accepted time is ``until``; ``max_steps`` accepted steps at most."""
        if "proposal" in earlier:
            from ._refusal import SpellingRefusal

            raise SpellingRefusal("simulation.run(until=t, max_steps=n); the first step is dt= "
                                  "of pops.Advance").refuse("simulation.run(proposal=...)",
                                                            "simulation.run")
        if earlier:
            raise TypeError("simulation.run has no option %s" % ", ".join(sorted(earlier)))
        if until is None:
            raise TypeError("simulation.run(until=t) needs the time to reach")
        until = _time(until, "simulation.run(until=t)")
        steps = _DEFAULT_MAX_STEPS if max_steps is None else max_steps
        from pops.runtime import _simulation as _n8

        for output in self._series:
            if output.write is not None and output not in self._written:
                output.write.check(self.directory)
        self._check_lattices()
        first_run = self._runs == 0
        self._runs += 1
        failure_error = None
        try:
            result = self._simulation.run(until, max_steps=steps)
        except _n8.RunFailed as error:
            failure_error = error
            result = error.result
        surface = self._result(result, first_run)
        self._write()
        if failure_error is not None:
            variant = _VARIANTS.get(type(failure_error).__name__, RunFailed)
            raise variant(surface, str(failure_error)).with_traceback(None) from failure_error
        return surface

    def _check_lattices(self) -> None:
        """Refuse a run whose start is off the lattice of an ``every`` schedule (F1): N8's
        lattice, k h and every k-th macro step, is fixed at compile time, and it is t0 + k h only
        when the run starts on it."""
        runtime = self._simulation._runtime
        start, step = float(runtime.time()), int(runtime.macro_step())
        for output in self._series:
            if output.every_time is not None:
                from pops.time._program.outputs import EveryDt

                lattice = EveryDt(output.every_time)
                index = round(start / output.every_time)
                near = lattice.time(index)
                if abs(near - start) > 4.0 * max(_math.ulp(start), _math.ulp(near)):
                    raise NotYetElaborated(
                        "a run of %r that starts at t0 = %r, off the lattice k h of "
                        "pops.every(time=%r), where the run would publish t0 + k h"
                        % (output, start, output.every_time), _LATTICE)
            if output.every_steps is not None and step % output.every_steps:
                raise NotYetElaborated(
                    "a run of %r that starts after %d accepted steps, which is not a multiple of "
                    "pops.every(steps=%d)" % (output, step, output.every_steps), _LATTICE)

    def _result(self, result: Any, first_run: bool) -> RunResult:
        case = self.artifact.case
        elaborated = case.elaborated
        cells = self._cells
        states = []
        for quantity in elaborated.states:
            block = case.model.name             # the block of the elaborated case (SU8)
            states.append((quantity, from_layout(result.state[block], cells)))
        outputs = {}
        regions = self.artifact.case.outputs
        for output, lowered in zip(regions, elaborated.outputs, strict=True):
            rows = []
            for index, publication in enumerate(result.outputs.get(lowered, ())):
                if not output.keeps(publication.time, first_run, index):
                    continue
                values = {name: from_layout(publication.values[name], cells)
                          for name, _ in output.record.items()}
                rows.append(Publication(publication.time, publication.macro_step, values))
            self._series[output].extend(rows)
            outputs[output] = tuple(rows)
        diagnostics = dict(result.diagnostics)
        diagnostics["threads"] = dict(self._threads)
        return RunResult(state=_State(tuple(states)), time=result.time, steps=result.steps,
                         rejections=result.rejections, outputs=outputs, diagnostics=diagnostics,
                         failure=result.failure, continuation=result.continuation)

    def _write(self) -> None:
        for output, series in self._series.items():
            if output.write is None:
                continue
            first = output not in self._written
            output.write.write_series(self.directory, output.record, tuple(series),
                                      first=first)
            self._written.add(output)

    def __repr__(self) -> str:
        return "<simulation of case %r>" % self.artifact.case.model.name


__all__ = ["Artifact", "BudgetExhausted", "CUDA", "Every", "OpenMP", "Output", "Publication",
           "Record", "Resources", "RunFailed", "RunResult", "Simulation", "SolveFailed", "bind",
           "compile", "every", "from_layout", "record", "to_layout"]
