"""The compiled run of an elaborated Advance (spec 0.3 sections 5.2 and 14.1, CC14, PF1).

``Simulation(runtime)`` takes the runtime that ``pops.bind`` returns for an artifact whose Program
elaborates an ``Advance`` (``pops.time._program.elaboration``). ``Simulation.run(until,
proposal=None, max_steps=None)`` is one call of ``System::run_advance``: the attempts, the
assessments of the candidate, the rejections, the retries and the landing on ``until`` run as
compiled C++, and no Python runs before that call returns (PF1, CF9). The run lands on ``until``
as master does (section 14.1): the rules of master's FixedDt, which absorb the rounding of
repeated additions, with the frontier published exactly (``runtime/program/advance_run.hpp``).
The run returns a ``RunResult``; a run whose result holds a failure raises its T39 variant, a
subclass of ``RunFailed`` whose ``result`` is that ``RunResult``, on every rank after the same
step.

This lot runs a Uniform artifact of one layout to the frontier ``until``. Outputs and their times,
assessments other than ``finite()``, transforms, the T38 classification of trial failures (a
trial failure still leaves the run as the native error that raised it, the accepted state
restored), the continuation codec and the AMR and multi-layout routes are later lots of N8.
"""
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any

from pops.time._program.control import AdvanceControl
from pops.time._program.elaboration import Proposal

DEFAULT_MAX_STEPS = 1_000_000
"""``max_steps`` when a run leaves it out: master's (``_runtime_instance.py:1598@bb9``), CC14."""

_REACHED, _STEP_BUDGET, _ATTEMPT_BUDGET = 0, 1, 2


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


class Failure:
    """The failure that ended a run: its T39 variant, its origin tag and its payload.

    The origin is ``"assessment"`` when the assessments of one step rejected every attempt of
    its budget, and ``None`` for ``max_steps``, a budget of the run rather than of an attempt.
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
        if payload["budget"] == "max_steps":
            return ("%s: max_steps=%d accepted steps ended the run at t=%r before until=%r"
                    % (self.variant.__name__, payload["max_steps"], payload["time"],
                       payload["until"]))
        return ("%s: the assessments rejected all %d attempt(s) of the step from t=%r; last: %s"
                % (self.variant.__name__, payload["attempts"], payload["time"],
                   payload["rejection"]))

    def __repr__(self) -> str:
        return "Failure(%s, origin=%r, payload=%r)" % (
            self.variant.__name__, self.origin, dict(self.payload))


class Continuation:
    """The last accepted continuation: its time, its macro step and the next proposal (5.2)."""

    __slots__ = ("time", "macro_step", "proposal")

    def __init__(self, time: float, macro_step: int, proposal: Proposal) -> None:
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "macro_step", macro_step)
        object.__setattr__(self, "proposal", proposal)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Continuation is immutable")

    def __repr__(self) -> str:
        return "Continuation(time=%r, macro_step=%d, proposal=Proposal(dt=%r))" % (
            self.time, self.macro_step, self.proposal.dt)


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
                 failure: Failure | None) -> None:
        values = {
            "state": state, "time": time, "steps": steps, "rejections": rejections,
            # pops.Output is a later lot of N8: a run of this lot publishes no output.
            "outputs": MappingProxyType({}),
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


class Simulation:
    """A bound artifact whose Program elaborates an Advance, run by ``System::run_advance``."""

    __slots__ = ("_runtime", "_control", "_continuation", "_run_native")

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
                "the compiled run publishes no output yet (pops.Output is a later lot of N8); "
                "this artifact declares consumers")
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
        if proposal is None:
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
        try:
            report = self._run_native(until, dt, max_steps)
        finally:
            # A failure of a trial leaves the accepted state of its step; the next proposal is
            # the one of this run (T28, next=None).
            self._continuation = Continuation(runtime.time(), runtime.macro_step(),
                                              Proposal(dt))
        time = runtime.time()
        stop = report["stop"]
        failure = None
        if stop == _STEP_BUDGET:
            failure = Failure(BudgetExhausted, None, {
                "budget": "max_steps", "max_steps": max_steps, "steps": report["steps"],
                "time": time, "until": until})
        elif stop == _ATTEMPT_BUDGET:
            failure = Failure(BudgetExhausted, "assessment", {
                "budget": "max_attempts", "max_attempts": self._control.max_attempts,
                "attempts": report["attempts"], "time": time, "rejection": report["rejection"]})
        elif stop != _REACHED or time != until:
            raise RuntimeError("System::run_advance returned without reaching its frontier")
        result = RunResult(
            state=RunState(runtime, time, runtime.macro_step()), time=time,
            steps=report["steps"], rejections=report["rejections"],
            diagnostics={
                "control": self._control.to_data(),
                "run": {"start": start.hex(), "until": until.hex(), "proposal": dt.hex(),
                        "max_steps": max_steps},
                "native": dict(report),
            },
            continuation=self._continuation, failure=failure)
        if failure is not None:
            raise failure.variant(result)
        return result


__all__ = ["BudgetExhausted", "Continuation", "DEFAULT_MAX_STEPS", "Failure", "RunFailed",
           "RunResult", "RunState", "Simulation"]
