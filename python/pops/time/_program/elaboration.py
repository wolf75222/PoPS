"""Elaboration of the temporal program onto the internal builder (spec 0.3, section 5.2).

``Advance(method)`` is one attempt loop of a specialized method, ``Proposal(dt)`` its attempt
parameters and ``Retry`` its budgets. ``elaborate`` traces the method once against a fresh builder
(``pops.time.Program``), commits the candidate at the endpoint of the state it evolves and attaches
the builder to the case: the builder is the elaborator, and the case never calls it.

Lot 3 adds the Outputs of the Advance (``outputs.py``, CC14), which the compiled run evaluates after
publication, and whose times it lands on.

This lot elaborates the trial of a method whose candidate reaches ``B.time + p.dt``, with the
default acceptance and a budget of attempts. The default acceptance ``finite()``, written
``accept=None``, accepts a candidate whose every value is finite: the candidate passes, before its
commit, an assessment whose condition is ``isfinite(norm_inf_all(candidate))``, the norm counting
a NaN as infinite. A rejection proposes the same proposal again (T28, ``next=None``), a fixed
step, which the builder runs as ``FixedDt`` with the nearest binary64 value of ``Proposal.dt``;
the trace never reads that value (T33). ``Retry(max_attempts=n)`` bounds the attempts of a step
(T38). The Program also records an ``AdvanceControl`` (``control.py``) that the compiled run of
``System::run_advance`` executes and that the artifact manifest publishes. Every construct of
section 5 outside this lot is refused with ``NotYetElaborated``, which names the lot that lowers
it.
"""
from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any

from pops.time._program.api import Program as _Builder
from pops.time._program.control import AdvanceControl
from pops.time._program.outputs import trace_output
from pops.time._program.traced_method import (
    MethodTrace, NotYetElaborated, SpecializedMethod, _fraction,
)
from pops.time._step.strategy import FixedDt


class Proposal:
    """``Proposal(dt, request=None, **choices)``: the attempt parameters of a step (5.2).

    ``dt`` keeps the value written, exact for a ``Fraction`` or a ``Decimal``; the step that the
    runtime takes is its nearest binary64 value.
    """

    __slots__ = ("dt", "request", "choices")

    def __init__(self, dt: Any, request: Any = None, **choices: Any) -> None:
        if isinstance(dt, bool) or not isinstance(dt, (int, float, Fraction, Decimal)):
            raise TypeError("Proposal(dt): dt is a positive finite duration, got %r" % (dt,))
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError("Proposal(dt): dt is a positive finite duration, got %r" % (dt,))
        _fraction(dt)
        if request is not None:
            from pops.time._program.relations import Request

            if not isinstance(request, Request):
                raise TypeError("Proposal(request=...) takes a Request(rtol=..., atol=...), the "
                                "request of the solves of the attempt (SV3), got %r" % (request,))
        if choices:
            raise NotYetElaborated("Proposal(%s): attempt choices (T28), which Accept and Reject "
                                   "carry, are lowered with the assessments of N8 phase 1"
                                   % ", ".join(sorted(choices)))
        self.dt: Any = dt
        self.request = request
        self.choices = dict(choices)


class Retry:
    """``Retry(max_attempts=1, max_transforms=0)``: the budgets of one attempt loop (T38)."""

    __slots__ = ("max_attempts", "max_transforms")

    def __init__(self, max_attempts: int = 1, max_transforms: int = 0) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("Retry(max_attempts=n) needs an integer n >= 1")
        if isinstance(max_transforms, bool) or not isinstance(max_transforms, int) \
                or max_transforms < 0:
            raise ValueError("Retry(max_transforms=n) needs an integer n >= 0")
        self.max_attempts = int(max_attempts)
        self.max_transforms = int(max_transforms)


class Advance:
    """``Advance(method, accept=finite(), retry=None, clock=None, time_map=None)`` (5.2).

    ``accept=None`` is the default acceptance ``finite()``; ``retry=None`` is ``Retry()``.
    """

    __slots__ = ("method", "accept", "retry", "clock", "time_map")

    def __init__(self, method: Any, accept: Any = None, retry: Any = None, clock: Any = None,
                 time_map: Any = None) -> None:
        if not isinstance(method, SpecializedMethod):
            raise TypeError("Advance(method): a method specialized with its components, "
                            "written m(component=...), got %r" % (method,))
        if accept is not None:
            raise NotYetElaborated("Advance(accept=...): an assessment other than the default "
                                   "finite() (T28) is lowered with the assessments of N8 phase 1")
        if retry is not None and not isinstance(retry, Retry):
            raise TypeError("Advance(retry=...) takes a Retry")
        if retry is not None and retry.max_transforms != 0:
            raise NotYetElaborated("Advance(retry=Retry(max_transforms=%d)): transforms (T28) are "
                                   "lowered with the assessments of N8 phase 1"
                                   % retry.max_transforms)
        if clock is not None or time_map is not None:
            raise NotYetElaborated("Advance(clock=..., time_map=...): clocks and time maps (T23, "
                                   "T36) are elaborated in Phase 2 of N8")
        self.method = method
        self.accept = accept
        self.retry = Retry() if retry is None else retry
        self.clock = clock
        self.time_map = time_map


class Elaboration:
    """The result of ``elaborate``: the builder attached to the case, the method trace and the
    compiled control of the Advance."""

    __slots__ = ("advance", "proposal", "program", "trace", "control")

    def __init__(self, advance: Advance, proposal: Proposal, program: Any,
                 trace: MethodTrace, control: AdvanceControl) -> None:
        self.advance = advance
        self.proposal = proposal
        self.program = program
        self.trace = trace
        self.control = control


# The lot that classifies the field_evaluation guards of a request (T39 PartialOperation, IR4).
_GUARD_LOT = "N8 field_evaluation guards"


def _type_identity(program: Any, kind: Any, node: Any) -> dict:
    """The type identity of a memory member: its space, or for a field observation its field
    authority, never a node id (CN9; this lot computes no schema_id, which is G07's)."""
    from pops.time._program.memory import State

    row = program._serialize_node(node, include_provenance=False)
    if kind is State:
        return {"kind": "state", "space": row.get("space")}
    attrs = row.get("attrs", {})
    return {"kind": "field_observation", "authority": attrs.get("field_authority"),
            "space": row.get("space")}


def _lower_memory(program: Any, method: Any, trace: MethodTrace) -> tuple:
    """Lower each member of the Candidate's memory to a History ring of depth one (CN11), after
    the assessment and the commit, and return its control rows (T27, CN1, CN2)."""
    if not trace.memory:
        return ()
    rows = []
    for name, kind, node in trace.memory:
        if name in program._histories:
            raise ValueError("the memory member %r of %r names a History ring that the Program "
                             "already holds" % (name, method.name))
        program.store_history(name, node, depth=1)
        rows.append({"path": "%s/%s" % (method.name, name), "role": "auxiliary",
                     "type": _type_identity(program, kind, node), "writer": method.name,
                     "readers": [], "ring": name})
    return tuple(rows)


def _trial_failures(program: Any, trace: MethodTrace) -> tuple:
    """The failure sites of the trace (T38): for each field request its solve, classified, and the
    field_evaluation guards of its load, coefficients and reads, listed unclassified; then each
    admissibility guard of the body (IR4), classified."""
    rows = []
    for request in trace.requests:
        retry = not request.fixed
        for guard in ("load", "coefficients"):
            rows.append({"site": "%s/%s" % (request.site, guard), "guard": "field_evaluation",
                         "variant": None, "retry_can_change": retry, "classified": False,
                         "lot": _GUARD_LOT})
        rows.append({"site": request.site + "/solve", "guard": "solve", "variant": "SolveFailed",
                     "retry_can_change": retry, "classified": True, "lot": None})
        for _, unknown in request.reads:
            rows.append({"site": "%s/read:%s" % (request.site, unknown.name),
                         "guard": "field_evaluation", "variant": None, "retry_can_change": retry,
                         "classified": False, "lot": _GUARD_LOT})
    for site in trace.solves:
        rows.append({"site": site.site, "guard": "solve", "variant": "SolveFailed",
                     "retry_can_change": not site.fixed, "classified": True, "lot": None})
    for guard in trace.guards:
        rows.append({"site": guard.site, "guard": "admissibility", "variant": "NotAdmissible",
                     "retry_can_change": not guard.fixed, "classified": True, "lot": None})
    return tuple(rows)


def elaborate(advance: Advance, *, case: Any, state: Any, proposal: Proposal,
              name: str | None = None, outputs: Any = (), realization: Any = None) -> Elaboration:
    """Elaborate ``advance`` for the block-qualified ``state`` of ``case``, from ``proposal``.

    ``outputs`` is a tuple of ``Output`` (``outputs.py``, CC14): each body is traced once into a
    read-only region of the builder, and the compiled control lists its schedule and record.

    The builder is named ``name``, by default after the method; that name and the value and stage
    names are presentation (CC2), though b6 hashes them into the program identity (C47).

    ``realization`` maps the path of each solve of the method (``relations.py``: the method name,
    ``method/name`` for a named solve, ``method/*`` for every solve of the method) to its solver
    reference, the ``Realization`` of 5.5; a key that names no solve is refused with
    ``UnknownSolvePath``, which lists the solves.
    """
    if not isinstance(advance, Advance):
        raise TypeError("elaborate(advance): an Advance, got %r" % (advance,))
    if not isinstance(proposal, Proposal):
        raise TypeError("elaborate(proposal=...): the first Proposal of the run, got %r"
                        % (proposal,))
    program = _Builder(advance.method.name if name is None else name)
    temporal = program.state(state)
    if realization is not None and not isinstance(realization, dict):
        raise TypeError("elaborate(realization=...) maps solve paths to solvers, got %r"
                        % (realization,))
    trace = advance.method.trace(program, temporal, proposal, realization)
    unused = sorted(set(realization or {}) - {
        key for key in (realization or {})
        if any(key in (site.path, "%s/*" % advance.method.name) for site in trace.solves)})
    if unused:
        from pops.time._program.relations import UnknownSolvePath

        raise UnknownSolvePath(
            "the realization keys %s name no solve of %r, whose solves are %s (5.5)"
            % (", ".join(map(repr, unused)), advance.method.name,
               ", ".join(repr(site.path) for site in trace.solves) or "none"))
    # finite(): every value of the candidate is finite, the norm counting a NaN as infinite.
    candidate = program._assess_candidate(
        "finite", trace.candidate, program.isfinite(program.norm_inf_all(trace.candidate)))
    program.commit(temporal.next, candidate)
    if isinstance(outputs, (str, bytes)) or not isinstance(outputs, (tuple, list)):
        raise TypeError("elaborate(outputs=...) is a tuple of Output, got %r" % (outputs,))
    if len({id(output) for output in outputs}) != len(outputs):
        raise ValueError("elaborate(outputs=...) names each Output once")
    regions = tuple(trace_output(program, temporal, output) for output in outputs)
    if regions:
        program._set_advance_outputs(regions)
    memory = _lower_memory(program, advance.method, trace)
    dt = proposal.dt
    strategy = FixedDt(float(dt) if isinstance(dt, (Fraction, Decimal)) else dt)
    program.step_strategy(strategy)
    control = AdvanceControl(assessments=(("finite", "default"),),
                             max_attempts=advance.retry.max_attempts,
                             max_transforms=advance.retry.max_transforms, first_dt=strategy.dt,
                             outputs=tuple(region.schedule_data() for region in regions),
                             memory=memory, trial_failures=_trial_failures(program, trace))
    program._set_advance_control(control)
    case.program(program)
    return Elaboration(advance, proposal, program, trace, control)
