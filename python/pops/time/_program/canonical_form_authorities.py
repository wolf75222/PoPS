"""Stock producers with a projection closed over their witnessed inputs.

Registering a public extension or using a frozen dataclass does not enroll its
serializer here. These are the exact types whose implementations read their
own recorded fields, registered Case/definition authorities, and the other
closed values below. A Program containing any other provider projects normally.
"""
from __future__ import annotations

from functools import cache


@cache
def _stock_projection_authorities() -> tuple[dict[int, type], dict[int, type]]:
    from pops._ir import balance, control_expr, expr, quantity
    from pops.identity.scalar import ScalarLiteral
    from pops.model import handles, ownership, signatures, spaces
    from pops.model.global_quantity import GlobalQuantityHandle, GlobalQuantityRef
    from pops.problem import handles as case_handles
    from pops.provenance import ProvenanceRecord, SourceSpan
    from pops.time import field_context, handles as time_handles, points, solve_outcome, values
    from pops.time._history import contracts, persistence, policy
    from pops.time._schedule import api as schedule
    from pops.time._schedule import domains
    from pops.time._step import transaction
    from pops.time.stencil import StencilAccess
    from pops.time.value_metadata import CoeffPolynomial
    from .control import AdvanceControl

    # Node metadata and symbolic values are read by the canonical SSA/DAG projectors.
    # Traversal of each recorded field also authenticates every nested object or mapping.
    closed = (
        values.ProgramValue, values._Coeff, CoeffPolynomial, StencilAccess,
        ScalarLiteral, ProvenanceRecord, SourceSpan,
        balance.Accumulation, balance.Balance, balance.BalanceOccurrence, balance.BalanceView,
        control_expr.Where, expr.Abs, expr.Add, expr.BooleanOr, expr.Compare, expr.Const,
        expr.Div, expr.Exp, expr.Expm1, expr.Maximum, expr.Minimum, expr.Mul, expr.Neg,
        expr.Partial, expr.Pow, expr.Sqrt, expr.Sub, expr.Var,
        quantity.PhysicalDimension, quantity.PhysicalSupport, quantity.QuantityRef,
        # Compilers project signatures, physical declarations and typed spaces from their fields.
        ownership.OwnerPath, ownership.OwnerSegment, signatures.Signature,
        spaces.FieldSpace, spaces.RateSpace, spaces.StateSpace, GlobalQuantityRef,
        # These time contracts project recorded clock, state and transaction relationships.
        points.Clock, points.TimePoint, points.StagePoint,
        field_context.FieldContext, field_context.FieldReadProvenance, AdvanceControl,
        contracts.HistoryContract, contracts.HistoryValidity, contracts._CanonicalInterpolation,
        persistence.Dense, persistence.Interval, persistence.Revolve, policy.CopyCurrent,
        solve_outcome.FailRun, solve_outcome.RejectAttempt, transaction.AcceptanceGuard,
        # Schedule's projector reads each concrete component, including every child under a
        # stock parent. The stock serializers read their fields and immutable identities.
        schedule.Schedule, schedule.Always, schedule.Every, schedule.EveryDt,
        schedule.AtStart, schedule.AtEnd, schedule.When, schedule.Hold, schedule.Skip,
        schedule.Zero, schedule.AccumulateDt, schedule.Error,
        domains.AcceptedStep, domains.Attempt, domains.Stage, domains.ClockTick,
        domains.AMRLevel, domains.Event, domains.EventHandle, domains.WallOutput,
    )
    # Typed Handles use the Case and definition witnesses instead of traversing their registry
    # back references. The enum payloads below are fixed scalar literals.
    leaves = (
        ownership.OwnerKind, transaction.GuardRole, transaction.ProvisionalStore,
        handles.Handle, handles.OperatorHandle, handles.ParamHandle, handles.StateHandle,
        case_handles.BlockHandle, case_handles.FieldHandle, case_handles.OperatorHandle,
        case_handles.StateHandle, GlobalQuantityHandle,
        time_handles.HistoryHandle, time_handles.StageHandle, time_handles.StateEndpointHandle,
        time_handles.TimeState,
    )
    return ({id(kind): kind for kind in closed}, {id(kind): kind for kind in leaves})


def projection_authorities() -> tuple[dict[int, type], dict[int, type]]:
    """Include the exact frozen registry classes issued by the physical seal."""
    from pops._projection_authority import _declared_projection_types
    from pops.model.operators import Operator
    from pops.model.registry import OperatorRegistry
    from pops.physics._freeze import _FROZEN_CLASSES

    closed, leaves = _stock_projection_authorities()
    declared, declared_references = _declared_projection_types()
    sealed = tuple(_FROZEN_CLASSES[base] for base in (OperatorRegistry, Operator)
                   if base in _FROZEN_CLASSES)
    return ({**closed, **declared, **{id(kind): kind for kind in sealed}},
            {**leaves, **declared_references})
