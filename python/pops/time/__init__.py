"""pops.time -- compiled time-program DSL (the temporal LANGUAGE, builder-mode IR).

A ``Program`` is a restricted, COMPILED numerical program describing one time step. Python
only BUILDS a typed IR; it never executes a numerical stage. The C++ lowering lives in
``pops.codegen.program_codegen`` and is reached through ``Program.emit_cpp_program`` (a lazy
delegator), so this package imports only ``pops.ir`` / ``pops.model`` -- never ``pops.codegen``,
``_pops``, or ``pops.lib`` at module scope (Spec 4 acyclic graph: time -> {ir, model}).

This package is the time LANGUAGE only: ``Program``, the ``ProgramValue`` /
``StageStateSet`` values, the ``Schedule`` scheduler (``always`` / ``every`` / ``when`` /
``on_start`` / ``on_end`` / ``subcycle``) and the IR-optimizer wrappers (``eliminate_*`` /
``optimize``). The READY time-stepping schemes live in ``pops.lib.time`` (Spec 4 s6 / s14),
called by their explicit names (no ``std`` bundle, Spec 4 s7); import them from there.

cf. docs/sphinx/reference/time-program.md (Phase 8) and the ADC-399 epic.
"""
from pops.time.handles import (  # noqa: F401
    HistoryHandle, StageHandle, StateEndpointHandle, TimeState,
)
from pops.time.history import CopyCurrent  # noqa: F401
from pops.time.history_persistence import (  # noqa: F401
    Dense, HistoryPersistence, Interval, Revolve,
)
from pops.time.graph import (  # noqa: F401
    Branch, Commit, Loop, OperatorCall, ProgramGraph, ProgramValue as GraphProgramValue,
    Region, RegionCapture, Solve, StateRead, Synchronize,
    Unknown, ValueRef,
)
from pops.time.method_properties import (  # noqa: F401
    AdditiveMethodCertificate, AdditiveMethodProperties, MethodCertificate,
    MethodProperties, ProgramMethodCertificate, SSPCertificate, UnknownOrder,
    certify_program_graph,
)
from pops.time.method_tableau import (  # noqa: F401
    AdditiveRungeKuttaTableau, RungeKuttaTableau,
)
from pops.time.passes_facade import (  # noqa: F401
    eliminate_common_subexpressions,
    eliminate_dead_nodes,
    eliminate_redundant_field_solves,
    optimize,
)
from pops.time.program import Program
from pops.time.solve_outcome import (  # noqa: F401
    FailRun, FieldSolveOutcome, RejectAttempt, ResidualSolution, SOLVE_STATUSES, SolveAction,
    SolveOutcome,
)
from pops.time.step_strategy import (  # noqa: F401
    AdaptiveCFL, ErrorControlledDt, ExternalTimeGrid, FixedDt, StepStrategy,
)
from pops.time.solve_problem import (  # noqa: F401
    CoupledImplicitEuler, LocalLinear, LocalResidual,
)
from pops.time.step_transaction import (  # noqa: F401
    ALL_PROVISIONAL_STORES, AcceptanceGuard, BlockProjection, GuardRole,
    ProjectAndRecheck, ProvisionalStore, StepTransactionPlan, StepTransactionReport,
)
from pops.time.points import Clock, StagePoint, TimePoint  # noqa: F401
from pops.time.schedule import (  # noqa: F401
    AMRLevel, AcceptedStep, AccumulateDt, Always, AtEnd, AtStart, Attempt,
    ClockTick, Domain, Error, Event, EventHandle, Every, Hold, OffPolicy,
    Schedule, Skip, Stage, Trigger, WallOutput, When, Zero,
    always, every, on_end, on_start, when,
)
from pops.time.synchronization import (  # noqa: F401
    SampleAndHold, SynchronizationRelation,
)
from pops.time.values import StageStateSet, ProgramValue  # noqa: F401

__all__ = ["Program", "ProgramValue", "StageStateSet", "ResidualSolution",
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
           "EventHandle", "Event", "WallOutput", "Trigger", "Always", "Every",
           "AtStart", "AtEnd", "When", "OffPolicy", "Hold", "Skip", "Zero",
           "AccumulateDt", "Error",
           "always", "every", "when", "on_start", "on_end",
           "eliminate_dead_nodes", "eliminate_common_subexpressions",
           "eliminate_redundant_field_solves", "optimize"]
