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
class SolveFailureDisposition:
    """One row of the table of solve failures (spec 0.3 SV8, SV10, T38, T39; N9 lot « La
    référence sous les solves de relation », §3.2): the status that a failure of ``source`` gives,
    how its action is chosen, and the variant of T39 that a trial site reports.

    ``action`` is ``"filter"`` (the consumer's status filter decides), ``"refusal"`` (the
    consumer's choice for ``safeguard_failure``, as a refused SV8 gate) or ``"fail_run"`` (the
    failure keeps its own terminal action, whatever the consumer). ``variant`` is None when the run
    fails: an invalid evaluation of the realization leaves as the untyped error of today until
    IR24 types it (phase 2)."""

    status: str
    action: str
    variant: str | None

    def resolve(self, kind: str, statuses: tuple[str, ...]) -> str:
        """The action, ``"reject_attempt"`` or ``"fail_run"``, under a consumer ``kind`` whose
        filter holds ``statuses``."""
        if self.action == "fail_run" or kind != "reject_attempt":
            return "fail_run"
        filtered = self.status if self.action == "filter" else "safeguard_failure"
        return "reject_attempt" if filtered in statuses else "fail_run"


# The table, one code on both realizations: the native's KrylovFailureActions and its SV8 gate
# read it, the reference's report reads it for each Result of its region, and a trial site reads
# its variant. A failure that it does not list is an error of the generator, not a status.
_SOLVE_FAILURES = {
    ("body", "iteration_limit"): SolveFailureDisposition(
        "iteration_limit", "filter", "BudgetExhausted"),
    ("body", "breakdown"): SolveFailureDisposition("breakdown", "filter", "SolveFailed"),
    ("body", "singular"): SolveFailureDisposition("singular", "filter", "SolveFailed"),
    ("body", "invalid_evaluation"): SolveFailureDisposition(
        "invalid_evaluation", "fail_run", None),
    ("gate", "safeguard_failure"): SolveFailureDisposition(
        "safeguard_failure", "refusal", "SolveFailed"),
    ("gate", "invalid_evaluation"): SolveFailureDisposition(
        "invalid_evaluation", "refusal", "SolveFailed"),
}
# The statuses whose action the consumer's filter chooses on the native recurrence, in the order
# of ``pops::KrylovFailureActions`` (krylov_workspace.hpp).
FILTERED_SOLVE_STATUSES = ("singular", "breakdown", "iteration_limit")


# The variants of T39 that a solve site of a traced method can report (§3.6): a row of the compiled
# control lists them, and a rejected attempt maps its status through the table.
SOLVE_SITE_VARIANTS = ("BudgetExhausted", "SolveFailed")


def solve_status_variant(status: str) -> str | None:
    """The variant of T39 that a solve site reports for an attempt rejected with ``status``, or
    None when no row of the table rejects an attempt with it: ``iteration_limit`` gives
    ``BudgetExhausted``, every other rejected status ``SolveFailed`` (SV10). The table holds one
    variant per status, whatever the source, so the status alone decides."""
    variants = {row.variant for row in _SOLVE_FAILURES.values()
                if row.status == status and row.variant is not None}
    if len(variants) > 1:
        raise RuntimeError("the table of solve failures gives %r two variants" % status)
    return variants.pop() if variants else None


def solve_failure_disposition(source: str, status: str) -> SolveFailureDisposition:
    """The row of the table of solve failures for a failure of ``source`` (``"body"``, the
    realization's own failure, or ``"gate"``, the SV8 gate of the problem) with ``status``."""
    row = _SOLVE_FAILURES.get((source, status))
    if row is None:
        raise ValueError("no solve failure %r of source %r: the table lists %s" % (
            status, source, ", ".join("%s/%s" % key for key in _SOLVE_FAILURES)))
    return row


def solve_realization_kind(node: Any) -> str:
    """``"native"`` or ``"reference"``: the realization of a ``solve_linear`` node (SV12). A
    native node keeps its provider attributes where they were, so that no native identity moves;
    a reference node carries ``realization`` and its region (``body_block``, ``body``)."""
    realization = node.attrs.get("realization")
    return "native" if realization is None else realization["kind"]


def state_operand_native_refusal(operator: Any) -> str:
    """The refusal of a native solve of a State-operand ``matrix_free_operator`` (IR26, N9 « La
    référence sous les solves de relation » §3.4, the pilot's condition on Q4): no C++ apply is
    emitted from its declaration, and its native realization is the operator that G10 generates."""
    return ("the native realization of the State-operand operator %s is the operator generated by "
            "G10" % (getattr(operator, "name", operator),))


def native_realization(node: Any) -> Any:
    """The provider attributes of a native ``solve_linear`` node. A pass that reads the native
    provider reads it here, so that it never reads a native attribute from a reference node: a
    reference node is refused by name, and so is a native node of a State-operand operator, whose
    native realization is G10's."""
    if solve_realization_kind(node) != "native":
        raise ValueError(
            "solve_linear %r is realized by a reference solver (SV12): it has no native provider; "
            "its region is its realization" % (getattr(node, "name", node),))
    operator = node.inputs[0] if getattr(node, "inputs", ()) else None
    if operator is not None and operator.attrs.get("apply_operand") == "state":
        raise NotImplementedError(state_operand_native_refusal(operator))
    return node.attrs


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
    def evidence(self) -> SolveEvidence:
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
            kind = "int64" if field == "iterations" else "scalar"
            with authoring_transaction(self._program):
                self._values[field] = self._program._new(
                    kind, "solve_evidence", (self._outcome,), {"field": field},
                    "%s_%s" % (self._outcome.name, field), None, space=None, field_context=None,
                    state_ref=None, point=self._outcome.point, inherit_state_ref=False)
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
        # A NotImplementedError: pops.time core does not import the elaborator's NotYetElaborated.
        raise NotImplementedError(
            "SolveOutcome evidence correction_norm (SV5) has no elaborated form yet on the "
            "prepared Krylov route or the reference CG; the later N9 lot « Évidence des solves » "
            "elaborates it")

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
    "FILTERED_SOLVE_STATUSES", "FailRun", "FieldSolveOutcome", "RejectAttempt",
    "ResidualSolution", "SOLVE_EVIDENCE_FIELDS", "SOLVE_SITE_VARIANTS", "SOLVE_STATUSES",
    "SolveAction", "SolveEvidence", "SolveFailureDisposition", "SolveOutcome",
    "native_realization", "solve_failure_disposition", "solve_realization_kind",
    "solve_status_variant", "state_operand_native_refusal",
]
