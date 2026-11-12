"""The compiled run of an elaborated Advance (spec 0.3 sections 5.2 and 14.1, CC14, PF1).

``Simulation(runtime)`` takes the runtime that ``pops.bind`` returns for an artifact whose Program
elaborates an ``Advance`` (``pops.time._program.elaboration``). ``Simulation.run(until,
proposal=None, max_steps=None)`` is one call of ``System::run_advance``: the attempts, the
assessments of the candidate, the rejections, the retries and the landing on ``until`` run as
compiled C++, and no Python runs before that call returns (PF1, CF9). The run lands on ``until``
as master does (section 14.1): the rules of master's FixedDt, which absorb the rounding of
repeated additions, with the frontier published exactly (``runtime/program/advance_run.hpp``).
An Advance elaborated from a ``StepRule`` (N8 « Règle de pas compilée ») takes each step's
proposal from b6's CFL selection of ``System::step_cfl``, then lands with the same rules; it
refuses ``proposal=``. On one artifact, ``pops.run`` (master's ``AdaptiveCFL``) caps its last step
at ``t_end - t`` instead, so the two can differ by one sliver step at a frontier: when a CFL step
ends within four ulp short of it, or when ``t + (t_end - t)`` is not ``t_end``.
The run returns a ``RunResult``; a run whose result holds a failure raises its T39 variant, a
subclass of ``RunFailed`` whose ``result`` is that ``RunResult``, on every rank after the same
step.

The Outputs of the Advance (``pops.time._program.outputs``, CC14) are evaluated by the same native
call: once at the start of the run, on the state from which it starts, then after each publication
they are due on, with its achieved time. Each time of their ``at`` that lies ahead of the run, up
to ``until``, is a frontier that the run lands on; a time within four units in the last place below
``until`` is ``until``, as master merges a consumer deadline into the end of its run. Their values
come back in ``RunResult.outputs``, one ``Publication`` per evaluation.

The failure sites of the field requests of a method (T38, N8 lot 5) are classified: a retryable
site (phase ``trial``) rejects the attempt, as an assessment does, and the next attempt repeats the
proposal; when every attempt is spent the run raises ``BudgetExhausted`` tagged ``trial``, whose
payload holds the ``SolveFailed`` of the last attempt as its ``cause``. A fixed site (phase
``trial_failure``), whose dependency set excludes the proposal, ends the step after its first
attempt, which is not counted as a rejection: the run raises ``SolveFailed`` tagged ``trial``. Both
leave the accepted state of the step. The admissibility guards of a method body (IR4, N8 lot 6)
follow the same rule with the variant ``NotAdmissible``, which the site's row of the compiled
control names. The Auxiliary memory of the method (T27) is read after a run
as ``continuation.memory``, slot one of the History ring of each member (``MemoryView``).

This lot runs a Uniform artifact of one layout to the frontier ``until``. Writers and callbacks of
Outputs, assessments other than ``finite()``, transforms, the classification of the other trial
failures (a native error that no site classifies still leaves the run as that error, the accepted
state restored), the continuation codec and the AMR and multi-layout routes are later lots of N8.
"""
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

from pops.time._program.control import AdvanceControl, memory_rings
from pops.time._program.elaboration import Proposal, StepRule
from pops.time._program.memory import MemoryView
from pops.time._program.outputs import EveryDt, OutputRegion, Product

DEFAULT_MAX_STEPS = 1_000_000
"""``max_steps`` when a run leaves it out: master's (``_runtime_instance.py:1598@bb9``), CC14."""

_REACHED, _STEP_BUDGET, _ATTEMPT_BUDGET, _TRIAL_FAILURE = 0, 1, 2, 3
_TRIAL_PHASES = ("trial", "trial_failure")


class RunFailed(RuntimeError):
    """A run whose ``RunResult`` holds a failure (T39); ``result`` is that ``RunResult``."""

    def __init__(self, result: RunResult) -> None:
        failure = result.failure if type(result) is RunResult else None
        if failure is None or failure.variant is not type(self):
            raise TypeError("RunFailed is raised with the RunResult whose failure is its variant")
        super().__init__(failure.describe())
        self.result = result


class BudgetExhausted(RunFailed):
    """A budget of the run (``max_steps``) or of a step (``Retry.max_attempts``) was spent."""


class SolveFailed(RunFailed):
    """A field solve of the trial failed at a fixed failure site, which a retry cannot change
    (T38, T39): the step ends after its first attempt."""


class NotAdmissible(RunFailed):
    """An admissibility guard of the trial failed (S5, IR4, T39): at a fixed failure site, which
    a retry cannot change, the step ends after its first attempt (T38)."""


class Failure:
    """The failure that ended a run: its T39 variant, its origin tag and its payload.

    The origin of a spent attempt budget is the phase of the rejections of its step:
    ``"assessment"`` when the assessments rejected every attempt, ``"trial"`` when a retryable
    failure site of the trial did (the payload's ``cause`` is the ``SolveFailed`` of the last
    attempt). A ``SolveFailed`` at a fixed site is tagged ``"trial"``. ``max_steps``, a budget of
    the run rather than of an attempt, has the origin ``None``.
    """

    __slots__ = ("variant", "origin", "payload")

    def __init__(self, variant: type, origin: str | None, payload: Mapping[str, Any]) -> None:
        if not (isinstance(variant, type) and issubclass(variant, RunFailed)) \
                or variant is RunFailed:
            raise TypeError("Failure.variant is a subclass of RunFailed")
        if origin not in (None, "trial", "assessment", "transform", "child"):
            raise ValueError("Failure.origin is trial, assessment, transform, child or None")
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "payload", MappingProxyType(dict(payload)))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Failure is immutable")

    def describe(self) -> str:
        payload = self.payload
        if self.variant is BudgetExhausted and payload["budget"] == "max_steps":
            return ("%s: max_steps=%d accepted steps ended the run at t=%r before until=%r"
                    % (self.variant.__name__, payload["max_steps"], payload["time"],
                       payload["until"]))
        if self.variant in (SolveFailed, NotAdmissible):
            if "attempts" not in payload:
                return "%s at the trial site %s: %s" % (
                    self.variant.__name__, payload["site"], payload["reason"])
            return ("%s: the fixed failure site %s ended the step from t=%r after %d attempt(s), "
                    "which a retry cannot change (T38): %s"
                    % (self.variant.__name__, payload["site"], payload["time"],
                       payload["attempts"], payload["reason"]))
        rejecting = ("the assessments" if self.origin == "assessment"
                     else "the trial site %s" % payload["cause"].payload["site"])
        return ("%s: %s rejected all %d attempt(s) of the step from t=%r; last: %s"
                % (self.variant.__name__, rejecting, payload["attempts"], payload["time"],
                   payload["rejection"]))

    def __repr__(self) -> str:
        return "Failure(%s, origin=%r, payload=%r)" % (
            self.variant.__name__, self.origin, dict(self.payload))


class Continuation:
    """The last accepted continuation: its time, its macro step, the next proposal (5.2) and the
    Auxiliary memory of the method (T27), a ``MemoryView`` of no member for a method without
    memory."""

    __slots__ = ("time", "macro_step", "proposal", "memory")

    def __init__(self, time: float, macro_step: int, proposal: Proposal | StepRule,
                 memory: MemoryView | None = None) -> None:
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "macro_step", macro_step)
        object.__setattr__(self, "proposal", proposal)
        object.__setattr__(self, "memory", memory)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Continuation is immutable")

    def __repr__(self) -> str:
        if type(self.proposal) is StepRule:
            return "Continuation(time=%r, macro_step=%d, proposal=%r)" % (
                self.time, self.macro_step, self.proposal)
        return "Continuation(time=%r, macro_step=%d, proposal=Proposal(dt=%r))" % (
            self.time, self.macro_step, self.proposal.dt)


class Publication:
    """One evaluation of an Output: the achieved time and the macro step of the publication it
    followed (or of the start of the run) and the values of its record (CC14)."""

    __slots__ = ("time", "macro_step", "values")

    def __init__(self, time: float, macro_step: int, values: Product) -> None:
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "macro_step", macro_step)
        object.__setattr__(self, "values", values)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Publication is immutable")

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def __repr__(self) -> str:
        return "Publication(time=%r, macro_step=%d, values=%s)" % (
            self.time, self.macro_step, ", ".join(self.values))


def _ahead(times: tuple, start: float, until: float) -> list:
    """The output times that are frontiers of a run from ``start`` to ``until``.

    A time within four units in the last place below ``until`` is ``until`` itself, as master
    merges a consumer deadline into the end of its run. The merge is one-sided on purpose, as in
    master, whose run loop takes a deadline only when it is at most ``t_end``, then lands on
    ``t_end`` when ``_same_physical_time`` holds (``_runtime_instance.py:1782-1787``, « a threshold
    one ULP above t_end is a future occurrence, not an end-of-run sample »): a tolerance never
    extends the requested run, so a time after ``until``, however close, stays ahead for a later
    run, which lands on it.
    """
    frontiers = []
    for time in times:
        if not time > start or time > until:
            continue
        if until - time <= 4.0 * max(math.ulp(time), math.ulp(until)):
            time = until
        if not frontiers or time > frontiers[-1]:
            frontiers.append(time)
    return frontiers


class RunState(Mapping):
    """The accepted state at the time of a result, block by block, in the layout of Q3.

    Reading a block gathers it from every rank, so every rank reads the same blocks in the same
    order. A later run of the simulation supersedes this state; reading it then is refused.
    """

    __slots__ = ("_runtime", "_time", "_macro_step", "_blocks")

    def __init__(self, runtime: Any, time: float, macro_step: int) -> None:
        self._runtime = runtime
        self._time = time
        self._macro_step = macro_step
        self._blocks = tuple(runtime.block_names())

    def __getitem__(self, block: str) -> Any:
        if block not in self._blocks:
            raise KeyError(block)
        if self._runtime.time() != self._time or self._runtime.macro_step() != self._macro_step:
            raise RuntimeError(
                "this state at t=%r was superseded by a later run; read it before running again"
                % self._time)
        return self._runtime.state_global(block)

    def __iter__(self) -> Iterator[str]:
        return iter(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)


class RunResult:
    """What a run returns (CC14): ``state``, ``time``, ``steps``, ``rejections``, ``outputs``,
    ``diagnostics``, ``continuation`` and ``failure``."""

    __slots__ = ("state", "time", "steps", "rejections", "outputs", "diagnostics",
                 "continuation", "failure")

    def __init__(self, *, state: RunState, time: float, steps: int, rejections: int,
                 diagnostics: Mapping[str, Any], continuation: Continuation,
                 failure: Failure | None, outputs: Mapping[Any, tuple] | None = None) -> None:
        values = {
            "state": state, "time": time, "steps": steps, "rejections": rejections,
            # Each Output of the Advance maps to its Publications in run order (CC14).
            "outputs": MappingProxyType(dict(outputs or {})),
            "diagnostics": MappingProxyType(dict(diagnostics)),
            "continuation": continuation, "failure": failure,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("RunResult is immutable")

    def __repr__(self) -> str:
        return "RunResult(time=%r, steps=%d, rejections=%d, failure=%r)" % (
            self.time, self.steps, self.rejections, self.failure)


def _binary64(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s is an int or a float, got %r" % (where, value))
    result = float(value)
    if not math.isfinite(result) or (type(value) is int and result != value):
        raise ValueError("%s must be a finite binary64 value, got %r" % (where, value))
    return result


def _trial_site(report: Mapping[str, Any]) -> dict:
    """The site and the reason of the trial rejection of ``report``: its detail is the message of
    the generated guard, ``<site> failed: <reason>`` (codegen/program_emit_solve.py for a solve,
    program_emit_ops.py for an admissibility guard)."""
    site, separator, reason = report["rejection_detail"].partition(" failed: ")
    if report["rejection_phase"] not in _TRIAL_PHASES or not separator:
        raise RuntimeError("a trial rejection names its failure site: %r"
                           % report["rejection_detail"])
    return {"site": site, "reason": reason}


_VARIANTS = {"SolveFailed": SolveFailed, "NotAdmissible": NotAdmissible}


def _trial_variant(control: Any, site: str) -> type:
    """The T39 variant of the failure site ``site``, from its row of the compiled control."""
    variants = {row["site"]: row["variant"] for row in control.to_data().get("trial_failures", ())}
    if variants.get(site) not in _VARIANTS:
        raise RuntimeError("the failure site %r has no classified row in the compiled control"
                           % site)
    return _VARIANTS[variants[site]]


def _run_data(start: float, until: float, dt: float, rule: Any, max_steps: int) -> dict:
    data: dict[str, Any] = {"start": start.hex(), "until": until.hex()}
    if rule is None:
        data["proposal"] = dt.hex()
    else:
        data["rule"] = {"fraction": rule[0].hex(),
                        "max_dt": None if rule[1] is None else rule[1].hex()}
    data["max_steps"] = max_steps
    return data


class Simulation:
    """A bound artifact whose Program elaborates an Advance, run by ``System::run_advance``."""

    __slots__ = ("_runtime", "_control", "_continuation", "_run_native", "_outputs")
    _outputs: tuple[OutputRegion, ...]

    def __init__(self, runtime: Any) -> None:
        from pops.runtime._runtime_instance import RuntimeInstance
        if type(runtime) is not RuntimeInstance:
            raise TypeError("Simulation takes the RuntimeInstance returned by pops.bind")
        from pops.time._program.api import Program
        artifact = runtime._install_plan.artifact
        program = getattr(getattr(artifact, "program", None), "program", None)
        control = getattr(program, "_advance_control", None) \
            if isinstance(program, Program) else None
        if type(control) is not AdvanceControl:
            raise TypeError(
                "Simulation runs an artifact whose Program elaborates an Advance "
                "(pops.time._program.elaboration.elaborate); this one has no compiled control")
        if runtime._consumer_graph.nodes:
            raise NotImplementedError(
                "the compiled run publishes the Outputs of its Advance (CC14), not the b6 "
                "ConsumerGraph; this artifact declares consumers")
        regions: tuple[OutputRegion, ...] = tuple(getattr(program, "_advance_outputs", ()))
        if len(regions) != len(control.outputs):
            raise RuntimeError("the Program and its compiled control list different Outputs")
        native = getattr(runtime._executor, "_s", None)
        if native is None or not callable(getattr(native, "_run_advance", None)):
            raise NotImplementedError(
                "the compiled run supports the Uniform runtime of one layout; the AMR and "
                "multi-layout routes are later lots of N8")
        if native._advance_max_attempts() != control.max_attempts:
            raise RuntimeError("the installed Program does not export the attempt budget of its "
                               "compiled control")
        self._runtime = runtime
        self._control = control
        self._continuation: Continuation | None = None
        self._run_native = native._run_advance
        self._outputs = regions

    @property
    def control(self) -> AdvanceControl:
        """The compiled control that the artifact manifest publishes."""
        return self._control

    def state(self) -> RunState:
        """The accepted state between runs, in the form of ``RunResult.state``."""
        runtime = self._runtime
        return RunState(runtime, runtime.time(), runtime.macro_step())

    def run(self, until: Any, proposal: Any = None, max_steps: Any = None) -> RunResult:
        """Run attempts until the accepted time is ``until``; see the module docstring.

        Without ``proposal``, the run starts from the next proposal of the last continuation,
        or from the first proposal of the Advance for the first run. ``max_steps`` bounds the
        accepted steps of the run, ``DEFAULT_MAX_STEPS`` when it is left out.
        """
        runtime = self._runtime
        until = _binary64(until, "Simulation.run(until=...)")
        rule = self._control.step_rule
        if rule is not None:
            # The step rule chooses every proposal (N8 « Règle de pas compilée »); the native run
            # does not read the proposal argument.
            if proposal is not None:
                raise TypeError("Simulation.run(proposal=...): the step rule of this Advance "
                                "chooses every proposal")
            dt = 0.0
        elif proposal is None:
            dt = (self._control.first_dt if self._continuation is None
                  else float(self._continuation.proposal.dt))
        elif type(proposal) is Proposal:
            # An exact Proposal keeps its value; the run takes its nearest binary64, as FixedDt.
            dt = float(proposal.dt)
        else:
            raise TypeError("Simulation.run(proposal=...) takes a Proposal, got %r" % (proposal,))
        if max_steps is None:
            max_steps = DEFAULT_MAX_STEPS
        elif isinstance(max_steps, bool) or type(max_steps) is not int or max_steps < 0:
            raise ValueError("Simulation.run(max_steps=n) needs an integer n >= 0")
        start, start_step = runtime.time(), runtime.macro_step()
        last = self._continuation
        if last is not None and (last.time, last.macro_step) != (start, start_step):
            raise RuntimeError("the runtime advanced outside this Simulation since its last run")
        schedule = []
        for region in self._outputs:
            output = region.output
            at = output.at
            times = () if at is None else at.times(start, until) if type(at) is EveryDt else at
            schedule.append((0 if output.when is None else output.when.steps,
                             _ahead(times, start, until),
                             [item for _, kind, item in region.record if kind == "array"]))
        try:
            report = self._run_native(until, dt, max_steps, schedule, None if rule is None else (
                rule[0], math.inf if rule[1] is None else rule[1]))
        finally:
            # A failure of a trial leaves the accepted state of its step; the next proposal is
            # the one of this run (T28, next=None).
            time, macro_step = runtime.time(), runtime.macro_step()
            self._continuation = Continuation(
                time, macro_step, Proposal(dt) if rule is None else StepRule(*rule),
                MemoryView(runtime, time, macro_step, memory_rings(self._control)))
        time = runtime.time()
        report = dict(report)
        published: dict[Any, list] = {region.output: [] for region in self._outputs}
        for sample in report.pop("outputs"):
            region = self._outputs[sample["output"]]
            scalars, arrays = iter(sample["scalars"]), iter(sample["arrays"])
            values = Product(**{name: next(arrays) if kind == "array" else float(next(scalars))
                                for name, kind, _ in region.record})
            published[region.output].append(
                Publication(sample["time"], sample["macro_step"], values))
        stop = report["stop"]
        failure = None
        if stop == _STEP_BUDGET:
            failure = Failure(BudgetExhausted, None, {
                "budget": "max_steps", "max_steps": max_steps, "steps": report["steps"],
                "time": time, "until": until})
        elif stop == _ATTEMPT_BUDGET:
            phase = report["rejection_phase"]
            payload = {"budget": "max_attempts", "max_attempts": self._control.max_attempts,
                       "attempts": report["attempts"], "time": time,
                       "rejection": report["rejection"]}
            if phase == "trial":
                site = _trial_site(report)
                payload["cause"] = Failure(_trial_variant(self._control, site["site"]), "trial",
                                           site)
            elif phase != "assessment":
                raise RuntimeError("System::run_advance spent an attempt budget on a rejection "
                                   "of phase %r" % phase)
            failure = Failure(BudgetExhausted, phase, payload)
        elif stop == _TRIAL_FAILURE:
            if report["rejection_phase"] != "trial_failure":
                raise RuntimeError("System::run_advance ended a step at a fixed failure site "
                                   "without its phase")
            site = _trial_site(report)
            failure = Failure(_trial_variant(self._control, site["site"]), "trial", dict(
                site, time=time, attempts=report["attempts"]))
        elif stop != _REACHED or time != until:
            raise RuntimeError("System::run_advance returned without reaching its frontier")
        result = RunResult(
            state=RunState(runtime, time, runtime.macro_step()), time=time,
            steps=report["steps"], rejections=report["rejections"],
            diagnostics={
                "control": self._control.to_data(),
                "run": _run_data(start, until, dt, rule, max_steps),
                "native": report,
                "outputs": [len(rows) for rows in published.values()],
            },
            continuation=self._continuation, failure=failure,
            outputs={output: tuple(rows) for output, rows in published.items()})
        if failure is not None:
            raise failure.variant(result)
        return result


__all__ = ["BudgetExhausted", "Continuation", "DEFAULT_MAX_STEPS", "Failure", "Publication",
           "NotAdmissible", "RunFailed", "RunResult", "RunState", "Simulation", "SolveFailed"]
