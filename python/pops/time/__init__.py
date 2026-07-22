"""pops.time -- compiled time-program DSL (the temporal LANGUAGE, builder-mode IR).

A ``Program`` is a restricted numerical-program description for one time step. Python only
BUILDS a typed IR; it never executes a numerical stage. Compiler and runtime layers consume this
package through explicit adapters, while :mod:`pops.time` remains an inert authoring authority.

This package is the time LANGUAGE only: ``Program``, the ``ProgramValue`` /
``StageStateSet`` values, the ``Schedule`` scheduler (``always`` / ``every`` / ``every_dt`` /
``when`` / ``on_start`` / ``on_end`` / ``subcycle``) and the IR-optimizer wrappers
(``eliminate_*`` / ``optimize``). ``every_dt`` is the ConsumerGraph physical-time cadence; it
does not schedule compiled Program stages. The READY time-stepping schemes live in
``pops.lib.time`` (Spec 4 s6 / s14),
called by their explicit names (no ``std`` bundle, Spec 4 s7); import them from there.

cf. docs/sphinx/reference/time-program.md (Phase 8) and the ADC-399 epic.
"""
from pops.time.handles import (  # noqa: F401
    HistoryHandle, StageHandle, StateEndpointHandle, TimeState,
)
from pops.time._history.policy import CopyCurrent  # noqa: F401
from pops.time._history.persistence import (  # noqa: F401
    Dense, HistoryPersistence, Interval, Revolve,
)
from pops.time._graph import (  # noqa: F401
    Branch, Commit, Loop, OperatorCall, ProgramGraph, ProgramValue as GraphProgramValue,
    Region, RegionCapture, Solve, StateRead, Synchronize,
    Unknown, ValueRef,
)
from pops.time._methods.properties import (  # noqa: F401
    AdditiveMethodCertificate, AdditiveMethodProperties, MethodCertificate,
    MethodProperties, ProgramMethodCertificate, SSPCertificate, UnknownOrder,
    certify_program_graph,
)
from pops.time._methods.tableau import (  # noqa: F401
    AdditiveRungeKuttaTableau, RungeKuttaTableau,
)
from pops.time._program.pass_api import (  # noqa: F401
    eliminate_common_subexpressions,
    eliminate_dead_nodes,
    eliminate_redundant_field_solves,
    optimize,
)
from pops.time._program.api import Program
from pops.time.solve_outcome import (  # noqa: F401
    FailRun, FieldSolveOutcome, RejectAttempt, ResidualSolution, SOLVE_STATUSES, SolveAction,
    SolveOutcome,
)
from pops.time._step.strategy import (  # noqa: F401
    AdaptiveCFL, ErrorControlledDt, ExternalTimeGrid, FixedDt, StepStrategy,
)
from pops.time.solve_problem import (  # noqa: F401
    CoupledImplicitEuler, LocalLinear, LocalResidual,
)
from pops.time._step.transaction import (  # noqa: F401
    ALL_PROVISIONAL_STORES, AcceptanceGuard, BlockProjection, GuardRole,
    ProjectAndRecheck, ProvisionalStore, StepTransactionPlan, StepTransactionReport,
)
from pops.time.points import Clock, StagePoint, TimePoint  # noqa: F401
from pops.time._schedule.api import (  # noqa: F401
    AMRLevel, AcceptedStep, AccumulateDt, Always, AtEnd, AtStart, Attempt,
    ClockTick, Domain, Error, Event, EventHandle, Every, EveryDt, Hold, OffPolicy,
    Schedule, ScheduleAction, ScheduleComment, ScheduleDomainIR, ScheduleDueIR,
    ScheduleDueKind, ScheduleLoweringIR, ScheduleOffIR, ScheduleTimeline,
    Skip, Stage, Trigger, WallOutput, When, Zero,
    always, every, every_dt, on_end, on_start, when,
)
from pops.time._schedule.synchronization import (  # noqa: F401
    SampleAndHold, SynchronizationRelation,
)
from pops.time._schedule.protocol import UnresolvedScheduleCondition  # noqa: F401
from pops.time.value_collections import StageStateSet  # noqa: F401
from pops.time.values import ProgramValue  # noqa: F401
from pops.time.stencil import StencilAccess  # noqa: F401

__all__ = ["Program", "ProgramValue", "StageStateSet", "StencilAccess", "ResidualSolution",
           "CoupledImplicitEuler", "LocalLinear", "LocalResidual",
           "SolveOutcome", "FieldSolveOutcome", "SolveAction", "FailRun", "RejectAttempt",
           "SOLVE_STATUSES", "Schedule",
           "StepStrategy", "FixedDt", "AdaptiveCFL", "ErrorControlledDt", "ExternalTimeGrid",
           "ALL_PROVISIONAL_STORES", "AcceptanceGuard", "BlockProjection", "GuardRole",
           "ProjectAndRecheck", "ProvisionalStore", "StepTransactionPlan", "StepTransactionReport",
           "ProgramGraph", "GraphProgramValue", "StateRead", "Unknown", "OperatorCall",
           "Solve", "Branch", "Loop", "Region", "RegionCapture",
           "Synchronize", "Commit", "ValueRef",
           "Clock", "TimePoint", "StagePoint",
           "RungeKuttaTableau", "AdditiveRungeKuttaTableau",
           "MethodCertificate", "MethodProperties", "AdditiveMethodCertificate",
           "AdditiveMethodProperties", "ProgramMethodCertificate", "SSPCertificate",
           "UnknownOrder", "certify_program_graph",
           "SampleAndHold", "SynchronizationRelation",
           "TimeState", "StageHandle", "HistoryHandle", "StateEndpointHandle", "CopyCurrent",
           "HistoryPersistence", "Dense", "Interval", "Revolve",
           "Domain", "AcceptedStep", "Attempt", "Stage", "ClockTick", "AMRLevel",
           "EventHandle", "Event", "WallOutput", "Trigger", "Always", "Every", "EveryDt",
           "AtStart", "AtEnd", "When", "OffPolicy", "Hold", "Skip", "Zero",
           "AccumulateDt", "Error",
            "ScheduleTimeline", "ScheduleDueKind", "ScheduleAction", "ScheduleComment",
            "ScheduleDomainIR", "ScheduleDueIR", "ScheduleOffIR", "ScheduleLoweringIR",
            "UnresolvedScheduleCondition",
            "always", "every", "every_dt", "when", "on_start", "on_end",
           "eliminate_dead_nodes", "eliminate_common_subexpressions",
           "eliminate_redundant_field_solves", "optimize"]
