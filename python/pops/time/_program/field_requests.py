"""Field requests of a traced method (spec 0.3 T6, T7, T38; N8 lot 5).

``field.at(U)`` is the field that an evaluation needs, solved for the data of its stage (T6). It is
legal inside a traced method only:

- in a component evaluation, at the state that the evaluation was given, at the stage point of
  that evaluation, which the trace knows;
- in the method body, at a state of the trace materialized at a stage, or ``B.state``, at the point
  of its materialization.

A request lowers onto b6's general linear field route: ``program.solve(field, values={U's state:
U}, at=point)``, consumed with ``RejectAttempt()``, then ``field.observe``. On that route the data of
a request are its state alone (the binding takes exactly the problem's state dependencies, the
boundary relations are periodic or homogeneous Neumann, the reactions constant, and no physical
time enters), so the dependency set of a request is its field and its state. Two requests of one
attempt with the same field registration, the same exact expansion of the state and the same stage
value of it have identical dependency sets: the second returns the first request and adds no node
(T7, exact reuse). The stage value enters the key because b6 ties an observation to the stage
value that its solve read (the field context of a publication, the stage of a stencil field): the
same state read at another time of the attempt is a copy at another stage, and a request there is
solved again. Its reuse goes with the linear recombination of T7, in the candidate-guard lot.

The reads of a request (``request[unknown]``, ``request.gradient``, ``request.publish``) are b6's,
and ``request[unknown]`` creates its node once. Every node of a request is recorded in the trace:
the rules of a component (``traced_method._require_rate``) exempt them from the time rule, since
PoPS places them at the stage of the evaluation, and count the nodes of a reused request as given.

T38. A request whose state is ``B.state`` alone, with no coefficient in ``p.dt``, has a dependency
set that excludes the proposal and the candidate: its solve is a fixed failure site, which ends the
step with ``SolveFailed`` (phase ``trial_failure``) and consumes no attempt. Any other request reads
the proposal through a rate of its state: a retryable site (phase ``trial``), counted against
``Retry.max_attempts``. The solve outcome carries ``trial_site`` with the stable site name.
"""
from __future__ import annotations

import contextlib
from typing import Any

from pops.time._program.traced_method import (
    ClosedBodyViolation, NotYetElaborated, TracedField, _active_trace, _refuse_foreign,
)
from pops.time.values import ProgramValue

_LINEAR_METHODS = ("CellCenteredSecondOrder", "CellCenteredGeneralCoupled")


def _identity(handle: Any) -> str:
    """The canonical identity of ``handle``, as a hashable key."""
    import json

    from pops.time.references import canonical_handle

    return json.dumps(canonical_handle(handle).canonical_identity(), sort_keys=True)


def _expansion_key(trace: Any, field: TracedField) -> tuple:
    return tuple(sorted((atom.label, poly) for atom, poly in field._expansion))


def _fixed(trace: Any, field: TracedField) -> bool:
    """T38: the state reads neither a rate nor ``p.dt``, so the request excludes the proposal."""
    return all(atom is trace.state_atom and all(power == 0 for power, _ in poly)
               for atom, poly in field._expansion)


class FieldRequest:
    """One field request of an attempt: its solution and its memoized reads (T6, T7)."""

    __slots__ = ("_trace", "_field", "_value", "_solution", "_reads", "site", "fixed", "nodes")

    def __init__(self, trace: Any, field: Any, value: Any, site: str, fixed: bool) -> None:
        self._trace = trace
        self._field = field
        self._value = value  # the stage value that the request is solved for
        self._solution: Any = None
        self._reads: dict = {}
        self.site = site
        self.fixed = fixed
        self.nodes: set = set()

    @contextlib.contextmanager
    def _recording(self) -> Any:
        program = self._trace.program
        first = program._next_id
        try:
            yield
        finally:
            ids = set(range(first, program._next_id))
            self.nodes |= ids
            self._trace.request_ids |= ids

    def _require_active(self) -> None:
        self._trace.require_active("the field request %s" % self.site)

    def __getitem__(self, unknown: Any) -> ProgramValue:
        """The observation of ``unknown``, one node for every read of this request."""
        self._require_active()
        key = _identity(unknown)
        if key not in self._reads:
            with self._recording():
                value = self._solution[unknown]
            self._reads[key] = (value, unknown)
            self._trace.request_reads[value.id] = (self, unknown)
        return self._reads[key][0]

    @property
    def reads(self) -> tuple:
        """``((observation, unknown), ...)`` of the reads of this request, in order."""
        return tuple(sorted(self._reads.values(), key=lambda item: item[0].id))

    def gradient(self, unknown: Any, *, dimension: int) -> ProgramValue:
        """b6's cell-centred gradient of ``unknown``, reading the memoized observation."""
        self._require_active()
        if type(dimension) is not int or dimension not in (1, 2, 3):
            raise TypeError("field gradient requires an explicit physical dimension")
        value = self[unknown]
        with self._recording():
            return self._solution._gradient_of(value, unknown, dimension=dimension)

    def publish(self, bindings: Any, *, states: Any = None) -> ProgramValue:
        """b6's publication of observations of this request to physics field inputs."""
        self._require_active()
        with self._recording():
            return self._solution.publish(bindings, states=states)

    def __repr__(self) -> str:
        return "<FieldRequest %s>" % self.site


def _route(field: Any, value: ProgramValue) -> Any:
    """The registration of ``field``, checked to be the general linear route of one state."""
    from pops.fields.problem import FieldProblem

    registry = getattr(field, "_field_registry", None)
    if registry is None:
        raise TypeError("field.at(U) is a request of a field registered by its Case")
    registration = registry.resolved_registration(field)
    problem = registration.operator
    method = type(registration.discretization.method).__name__
    if type(problem) is not FieldProblem or method not in _LINEAR_METHODS:
        raise NotYetElaborated(
            "field.at(U) on %s: this lot elaborates the general linear field route "
            "(CellCenteredSecondOrder, CellCenteredGeneralCoupled); nonlinear, temporal and "
            "provider routes are elaborated by a later lot of N8" % (field.name,))
    dependencies = problem.dependencies()
    if len(dependencies) != 1 or any(handle.kind != "state" for handle in dependencies) or (
            value.state_ref is None or _identity(dependencies[0]) != _identity(value.state_ref)):
        raise NotYetElaborated(
            "field.at(U) on %s: the data of its problem read %s; this lot elaborates a field "
            "whose data read the requested state only, so that its dependency set is its field "
            "and its state (T7); other fields are elaborated by a later lot of N8"
            % (field.name, "no state" if not dependencies else "other states"))
    return registry, problem


def at(field: Any, state: Any) -> FieldRequest:
    """``field.at(U)``: the request of ``field`` at the state ``U`` of a traced method (T6)."""
    from pops.time._program import outputs

    if outputs._ACTIVE:
        raise NotYetElaborated(
            "field.at(U) in an Output body: a field request at the accepted state of an Output "
            "(T8, T33) is elaborated with the Assess bodies of a later lot of N8")
    trace = _active_trace()
    if trace is None:
        raise ClosedBodyViolation(
            "field.at(U) is a field request outside a method: an evaluation requests the field "
            "that it needs inside a traced method (T6)")
    evaluation = trace.evaluation
    if evaluation is not None:
        if state is not evaluation.given:
            if isinstance(state, ProgramValue) and state.prog is trace.program \
                    and state.id >= evaluation.first:
                raise NotYetElaborated(
                    "the component %r requests a field at a state that it computed; this lot "
                    "elaborates a request at the state the evaluation is given (T6), the others "
                    "come with a later lot of N8" % evaluation.component)
            raise ClosedBodyViolation(
                "the component %r requests a field at a value it was not given, such as a state "
                "of another stage: a component evaluates at the state it is given (T33, T4)"
                % evaluation.component)
        traced, value, point = evaluation.state, evaluation.given, evaluation.point
        owner = "%s#%d" % (evaluation.component, evaluation.index)
    else:
        if not isinstance(state, TracedField):
            _refuse_foreign(state, "field.at(U)")
            raise TypeError("field.at(U) in a method body takes a traced state of the body")
        if state._trace is not trace:
            raise ClosedBodyViolation("field.at(U) reads a state of another method trace (T33)")
        if state._value is None:
            raise NotYetElaborated(
                "field.at(U) at a combination that no evaluation materialized: a request is "
                "made at a state of a stage, and the body never adds a stage")
        traced, value, point = state, state._value, state._value.point
        owner = None
    registry, problem = _route(field, value)
    # T7 on b6: the stage value enters the key, since b6 ties an observation to the stage value
    # that its solve read (a publication's field context, a stencil's field stage).
    key = (_identity(registry.canonicalize(field)), _expansion_key(trace, traced), value.id)
    request = trace.requests.get(key)
    if request is not None:
        if evaluation is not None:
            evaluation.reused |= request.nodes
        return request
    if owner is None:
        owner = "body#%d" % trace.body_requests
        trace.body_requests += 1
    fixed = _fixed(trace, traced)
    request = FieldRequest(trace, field, value, "%s/%s/%s" % (trace.name, owner, problem.name),
                           fixed)
    from pops.time.solve_outcome import RejectAttempt

    program = trace.program
    with request._recording():
        outcome = program.solve(field, values={value.state_ref: value}, at=point)
        consumed = outcome._consume(RejectAttempt(), {"trial_site": {
            "site": request.site + "/solve", "retry_can_change": not fixed}})
        request._solution = field.observe(consumed)
    for node in program._issued_values.values():
        if node.id in request.nodes and node.op in ("field_problem_load",
                                                    "field_problem_coefficients") \
                and "temporal_tau" in repr(node.attrs.get("expressions", ())):
            raise NotYetElaborated(
                "field.at(U) on %s: its data read a stage time (temporal_tau), which enters the "
                "dependency set of the request (T7); elaborated by a later lot of N8" % field.name)
    trace.requests[key] = request
    if evaluation is not None:
        evaluation.reused |= request.nodes
    return request


__all__ = ["FieldRequest", "at"]
