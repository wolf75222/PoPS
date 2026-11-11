"""Explicit consumed outcome contract for Program solve nodes."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar

from pops.time._authoring import authoring_transaction
from pops.time.residual_common import CanonicalDescriptor, residual_names

SOLVE_STATUSES = (
    "singular", "breakdown", "iteration_limit",
    "invalid_evaluation", "capability_failure", "invalid_input", "incompatible_rhs",
    "inadmissible_candidate", "safeguard_failure",
)
_STATUS_SET = frozenset(SOLVE_STATUSES)


def _statuses(values: Iterable[Any]) -> tuple[str, ...]:
    statuses = residual_names(values, "SolveAction statuses", nonempty=True)
    unknown = tuple(status for status in statuses if status not in _STATUS_SET)
    if unknown:
        raise ValueError("unknown solve status(es): %s" % ", ".join(unknown))
    return statuses


@dataclass(frozen=True, slots=True)
class SolveAction(CanonicalDescriptor):
    """Base class for explicit runtime disposition of a non-solved solve result."""

    statuses: tuple[str, ...] = SOLVE_STATUSES
    kind: ClassVar[str] = "solve_action"

    def __post_init__(self) -> None:
        object.__setattr__(self, "statuses", _statuses(self.statuses))


@dataclass(frozen=True, slots=True)
class FailRun(SolveAction):
    """Abort the run when a solve reports any configured non-solved status."""

    kind: ClassVar[str] = "fail_run"


@dataclass(frozen=True, slots=True)
class RejectAttempt(SolveAction):
    """Reject the current attempt/step when a solve reports a configured status."""

    kind: ClassVar[str] = "reject_attempt"


@dataclass(frozen=True, slots=True)
class ResidualSolution:
    """Ordered, typed projections of one consumed residual solve outcome."""

    values: tuple[Any, ...]
    names: tuple[str, ...] = ()
    unknowns: tuple[Any, ...] = ()
    problem_identity: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "unknowns", tuple(self.unknowns))
        if self.names and (len(self.names) != len(self.values)
                           or len(set(self.names)) != len(self.names)):
            raise ValueError("ResidualSolution names must identify every projection exactly once")
        if self.unknowns and len(self.unknowns) != len(self.values):
            raise ValueError("ResidualSolution unknowns must identify every projection")

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: Any) -> Any:
        from pops.time.solve_request import SolveUnknown

        if isinstance(index, str):
            if index not in self.names:
                raise KeyError(index)
            index = self.names.index(index)
        elif isinstance(index, SolveUnknown):
            identities = tuple(item.identity for item in self.unknowns)
            if index.identity not in identities:
                raise KeyError(index.name)
            index = identities.index(index.identity)
        return self.values[index]


class SolveOutcome:
    """Non-readable Program solve token; call ``consume(action=...)`` to project values."""

    __slots__ = ("_program", "_token", "_factory", "_name", "_result", "_node", "_evidence")

    def __init__(self, program: Any, token: Any, factory: Any, name: Any) -> None:
        self._program = program
        self._token = token
        self._factory = factory
        self._name = name
        self._result = None
        self._node = None
        self._evidence = None

    @property
    def token(self) -> Any:
        if "solve_request" in self._token.attrs:
            from pops.time.solve_request import SolveRequestError

            raise SolveRequestError(
                "unconsumed_result", "call consume(action=...) before reading a solve result")
        raise TypeError(
            "SolveOutcome is not readable; call outcome.consume(action=FailRun(...) or "
            "RejectAttempt(...)) before using solved values")

    def consume(self, *, action: SolveAction | None = None) -> Any:
        return self._consume(action, {})

    def _consume(self, action: Any, attrs: Any) -> Any:
        """``consume`` with extra outcome attributes, such as the ``trial_site`` of a field
        request of a traced method (``pops.time._program.field_requests``)."""
        if not isinstance(action, SolveAction):
            if "solve_request" in self._token.attrs:
                from pops.time.solve_request import SolveRequestError

                raise SolveRequestError(
                    "missing_failure_disposition", "consume requires FailRun(...) or RejectAttempt(...)")
            raise TypeError("SolveOutcome.consume requires action=FailRun(...) or RejectAttempt(...)")
        if self._result is not None:
            raise RuntimeError("SolveOutcome has already been consumed")
        if "solve_request" in self._token.attrs:
            self._program._validate_solve_request_node(self._token)
        with authoring_transaction(self._program):
            node = self._program._new(
                "solve_outcome", "solve_outcome", (self._token,),
                {"action": action, **attrs}, "%s_outcome" % self._name, self._token.block,
                space=self._token.space, point=self._token.point)
            result = self._factory(node)
        self._result = result
        self._node = node
        return result

    @property
    def evidence(self) -> "SolveEvidence":
        """The evidence of a consumed solve (spec 0.3 SV5): its iterations and the norm of its last
        physical residual, in the request's norm. It reports what the realization did and never
        decides success: the gates of SV8 decide, and a failed solve does not continue past its
        consumption, so the evidence that a Program reads is that of a solved candidate."""
        if self._node is None:
            raise TypeError(
                "SolveOutcome evidence is readable only after outcome.consume(action=...) (SV5)")
        if self._token.op not in _EVIDENCE_OPS:
            raise NotImplementedError(
                "SolveOutcome evidence of a %r solve is not elaborated yet; the prepared Krylov "
                "solve_linear route has it (N9 lot 4)" % self._token.op)
        if self._evidence is None:
            self._evidence = SolveEvidence(self._program, self._node)
        return self._evidence

    def __iter__(self):
        raise TypeError("SolveOutcome is not iterable; call consume(action=...) first")

    def __len__(self) -> int:
        raise TypeError("SolveOutcome has no length; call consume(action=...) first")

    def __getitem__(self, _index: int) -> Any:
        raise TypeError("SolveOutcome is not indexable; call consume(action=...) first")

    def __repr__(self) -> str:
        return "<SolveOutcome %s: unconsumed>" % self._name


_EVIDENCE_OPS = frozenset(("solve_linear",))
SOLVE_EVIDENCE_FIELDS = ("iterations", "residual_norm")


class SolveEvidence:
    """The evidence of one consumed solve (SV5), as Program values read after the consumption.

    ``iterations`` is an Int64 and ``residual_norm`` a Scalar, both identical on every rank (the
    count is local to an identical recurrence and the norm is already reduced). ``reason`` is
    "solved": a consumed outcome whose report is not solved fails the attempt or the run at its
    consumption. ``correction_norm`` is not measured by either route yet."""

    __slots__ = ("_program", "_outcome", "_values")

    def __init__(self, program: Any, outcome: Any) -> None:
        self._program = program
        self._outcome = outcome
        self._values: dict[str, Any] = {}

    def _value(self, field: str) -> Any:
        if field not in self._values:
            from pops.time._program.structured_control import new_typed, scalar_signature

            kind = "int64" if field == "iterations" else "scalar"
            with authoring_transaction(self._program):
                self._values[field] = new_typed(
                    self._program, scalar_signature(kind), "solve_evidence", (self._outcome,),
                    {"field": field}, point=self._outcome.point,
                    name="%s_%s" % (self._outcome.name, field))
        return self._values[field]

    @property
    def iterations(self) -> Any:
        return self._value("iterations")

    @property
    def residual_norm(self) -> Any:
        return self._value("residual_norm")

    @property
    def reason(self) -> str:
        return "solved"

    @property
    def correction_norm(self) -> Any:
        raise NotImplementedError(
            "SolveOutcome evidence correction_norm (SV5) is not measured by the prepared Krylov "
            "route or the reference CG yet; a later lot of N9 adds it")

    def __repr__(self) -> str:
        return "<SolveEvidence of %s>" % self._outcome.name


class FieldSolveOutcome(SolveOutcome):
    """Non-readable field-solve token.

    A field value is publishable only after :meth:`consume` records an explicit
    :class:`FailRun` or :class:`RejectAttempt` action.  The distinct public type keeps field
    materialization visible in signatures and error messages while reusing the exact one-shot
    consumption transaction of every other solve.
    """

    def __repr__(self) -> str:
        return "<FieldSolveOutcome %s: unconsumed>" % self._name


__all__ = [
    "FailRun", "FieldSolveOutcome", "RejectAttempt", "ResidualSolution", "SOLVE_EVIDENCE_FIELDS",
    "SOLVE_STATUSES", "SolveAction", "SolveEvidence", "SolveOutcome",
]
