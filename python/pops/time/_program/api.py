"""pops.time.Program -- the compiled time-program authoring class (builder-mode IR).

A ``Program`` BUILDS a typed SSA IR for one time step; Python never executes a numerical stage.
It is an inert authoring/IR authority: compiler lowering is owned exclusively by
``pops.codegen`` and runtime materialization by ``pops.runtime``. The class is composed from
focused authoring mixins and this package never imports either consumer layer.

cf. docs/sphinx/reference/time-program.md (Phase 8) and the ADC-399 epic.
"""
from __future__ import annotations

from typing import Any

from pops.model.ownership import OwnerKind, OwnerPath
from pops.time._program.contract import register_program_type
from pops.time._program.authoring import _ProgramAuthoring
from pops.time._program.condensed import _ProgramCondensed
from pops.time._program.operations import _ProgramCore
from pops.time._program.dt_bound import _ProgramDtBound
from pops.time._program.history import _ProgramHistory
from pops.time._program.inspection import _ProgramInspect
from pops.time._program.local import _ProgramLocal
from pops.time._program.passes import _ProgramPasses
from pops.time._program.solve import _ProgramSolve
from pops.time._program.time_handles import _ProgramTimeHandles
from pops.time.references import bind_program_block, block_name
from pops.time._step.transaction import (
    ALL_PROVISIONAL_STORES,
    AcceptanceGuard,
    GuardRole,
    ProvisionalStore,
    StepTransactionPlan,
    ensure_step_strategy,
)
from pops.time.values import _Coeff, ProgramValue  # noqa: F401  (ProgramValue used by mixins via prog ref)


@register_program_type
class Program(_ProgramTimeHandles, _ProgramCore, _ProgramLocal, _ProgramCondensed,
              _ProgramHistory, _ProgramSolve,
              _ProgramAuthoring, _ProgramDtBound, _ProgramPasses, _ProgramInspect):
    """A compiled time program (builder mode). Holds the SSA value list and the committed
    blocks. The Python object only BUILDS the IR; it is never executed numerically during
    ``sim.step``. Authoring and pure IR inspection methods come from the mixins.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "name" and hasattr(self, "name"):
            raise AttributeError(
                "pops.time.Program name is an immutable identity anchor; construct a new Program")
        if getattr(self, "_frozen", False):
            if name == "_frozen" and value is not True:
                raise RuntimeError("pops.time.Program freeze is irreversible")
            if name != "_frozen" or value is not True:
                raise RuntimeError("pops.time.Program is frozen: cannot change %s" % name)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if name == "name":
            raise AttributeError(
                "pops.time.Program name is an immutable identity anchor; construct a new Program")
        if getattr(self, "_frozen", False):
            raise RuntimeError("pops.time.Program is frozen: cannot delete %s" % name)
        object.__delattr__(self, name)

    def __init__(self, name: Any) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("Program name must be a non-empty string")
        self.name = name
        self._owner_path = OwnerPath.fresh(OwnerKind.CONSUMER, name)
        # Exact live CASE authority selected by the first block. Runtime block indices have meaning
        # only inside that one assembly; equal local names from another Case must never alias them.
        self._case_owner_path = None
        self._init_time_handle_tables()
        # De-stringing is the ONE public path (Spec 5 sec.15, ADC-479 criteria 23 + 27): the public
        # Callable operator handles and the public P.rhs require typed declarations
        # Free operator names and flux/source selector kwargs never enter Program IR. Callable
        # handles and typed RHS terms lower through one private native projection only.
        self._values = []
        self._issued_values = {}  # id -> strong identity, including stale immutable replacement records
        self._next_id = 0
        self._commits = {}      # qualified state Handle -> State value
        self._recording = []    # stack of sub-block lists (a control-flow body); see _new / while_
        self._next_region = 1
        self._recording_regions = {}  # id(list) -> (strong list ref, exact authoring-region token)
        self._region_imports = {}  # destination region -> explicitly sanctioned source regions
        # Qualified state Handle -> StateSpace or None. A block may instantiate several declared
        # state families; their semantic state_ref, never the block/name alone, owns the type contract.
        self._state_spaces = {}
        self._histories = {}    # name -> max declared lag (multistep histories; ADC-406a)
        self._history_spaces = {}  # full-state history name -> StateSpace or None
        self._history_blocks = {}  # full-state history name -> qualified block or None
        self._history_state_refs = {}  # full-state history name -> qualified state Handle
        # name -> slot ncomp for a NARROW (non-full-state) history ring (ADC-427). Only names read with
        # an explicit P.history(ncomp=1) appear here (the condensed-Schur phi^n carry); a full-state
        # multistep ring is absent (the codegen emits the historical 2-arg register_history for it).
        self._histories_ncomp = {}
        # name -> (depth, HistoryPersistence) per keep_history ring (ADC-626). The checkpoint persists
        # the policy-selected slots; the compile-time pass validates coherence + program-determinism per
        # ring. Empty -> every ring persists Dense (the historical whole ring, no recomputation).
        self._history_persistence = {}
        # OPTIONAL dt bound (spec s18 / ADC-417): a recorded scalar sub-program (cfl -> Scalar) the
        # generated .so exports as pops_program_dt_bound; None = no bound (the native CFL is used).
        self._dt_bound = None        # (block, scalar_value) once set; the block is the scalar sub-block
        self.dt = _Coeff({1: 1})     # symbolic time step; participates in coefficient arithmetic
        # Operator registries are indexed by their exact authoring OwnerPath. A coupled Program may
        # bind several models with homonymous operators; there is deliberately no "current" registry.
        self._operator_registries = {}
        # OPTIONAL debug capture of the authoring source location per IR node (ADC-530). DEFAULT OFF:
        # a stack walk per node is too costly for the normal build path, and the location is
        # INSPECTION-ONLY (never serialized into the IR / the hash). Toggle with
        # capture_source_locations(True) before building to populate ProgramValue.source_location.
        self._capture_source = False
        # Temporary data-only context installed while a pops.lib.time factory expands. It contains
        # SourceSpan values and an API string, never a callable or live frame.
        self._provenance_context = None
        # ADC-666: explicit attempt controller. Runtime kwargs are validated against this descriptor;
        # a run-time CFL/dt/error-control option never silently selects a strategy.
        self._step_strategy = None
        self._transaction_stores = ALL_PROVISIONAL_STORES
        self._acceptance_guards = ()
        # ADC-563 freeze: a Program is MUTABLE while authored and FROZEN by pops.compile. After
        # freeze, adding an IR node (via _new) RAISES -- a compiled artifact is frozen to exactly the
        # program it was compiled from. Emission / hashing are pure reads and stay allowed.
        self._frozen = False

    @property
    def owner_path(self) -> OwnerPath:
        """Stable authoring identity used to qualify this Program's declaration handles."""
        return self._owner_path

    def freeze(self) -> Any:
        """Deep-freeze the Program IR and detach every pre-freeze container reference.

        ``pops.compile`` freezes the time Program it lowers; graph inspection and the IR hash are
        pure reads and remain allowed. Every owned list/dict/set is replaced by an
        immutable copy, so a stale authoring reference cannot alter the compiled identity. Idempotent.
        """
        if self._frozen:
            return self
        from pops.time._program.freeze import freeze_program_tables
        freeze_program_tables(self)
        return self

    def to_graph(self) -> Any:
        """Return the detached immutable ProgramGraph snapshot of this authoring Program."""
        from pops.time._program.graph_conversion import program_to_graph

        return program_to_graph(self)

    def _guard_mutable(self, operation: Any) -> None:
        """Reject every authoring mutation after ``freeze()``, including non-node metadata writes."""
        if self._frozen:
            raise RuntimeError(
                "pops.time.Program %r is frozen: cannot %s" % (self.name, operation))

    def _region_for_block(self, block: Any) -> int:
        """Return the deterministic region token for one recorded sub-block list."""
        key = id(block)
        entry = self._recording_regions.get(key)
        if entry is None:
            region = self._next_region
            self._next_region += 1
            self._recording_regions[key] = (block, region)
            return region
        if entry[0] is not block:
            raise RuntimeError("internal authoring-region identity collision")
        return entry[1]

    def _current_region(self) -> int:
        return self._region_for_block(self._recording[-1]) if self._recording else 0

    def _allow_region_capture(self, source: int, destination: int) -> None:
        """Declare one explicit loop-carried edge between two sibling sub-block regions."""
        self._region_imports.setdefault(destination, set()).add(source)

    def _new(self, vtype: Any, op: Any, inputs: Any, attrs: Any, name: Any, block: Any,
             **metadata: Any) -> Any:
        """Guard the single IR-append choke point against a post-freeze mutation (ADC-563)."""
        self._guard_mutable("add IR node %r" % op)
        if block is not None:
            bind_program_block(self, block, where="IR op %r" % op)
        return super()._new(vtype, op, inputs, attrs, name, block, **metadata)

    def capture_source_locations(self, enabled: Any = True) -> Any:
        """Enable (or disable) recording each IR node's authoring source location (ADC-530).

        When enabled, every subsequently built :class:`pops.time.values.ProgramValue` captures the file and
        line of the authoring call site into its ``source_location`` (a debug aid: which macro line
        emitted a node). It is INSPECTION-ONLY -- excluded from ``_serialize`` / ``_ir_hash`` -- so it
        never changes a compiled-artifact cache key or a trajectory. Off by default (the stack walk is
        skipped on the normal build path). Returns ``self`` for chaining."""
        self._guard_mutable("change source-location capture")
        self._capture_source = bool(enabled)
        return self

    def step_strategy(
        self,
        strategy: Any,
        *,
        stores: Any = ALL_PROVISIONAL_STORES,
    ) -> Any:
        """Attach the explicit StepStrategy/transaction contract to this Program.

        This is authoring metadata, not a runtime kwargs bag.  The native controller must validate run
        controls against the selected strategy before the first attempt and must report staged,
        committed, or rolled-back effects through StepTransactionReport.
        """
        self._guard_mutable("set step strategy")
        strategy = ensure_step_strategy(strategy)
        stores = tuple(stores)
        if not stores or any(type(store) is not ProvisionalStore for store in stores):
            raise TypeError("Program.step_strategy stores must contain ProvisionalStore values")
        if len(set(stores)) != len(stores):
            raise ValueError("Program.step_strategy stores cannot contain duplicates")
        self._step_strategy = strategy
        self._transaction_stores = stores
        return self

    def _register_acceptance_guard(self, guard: AcceptanceGuard) -> None:
        self._guard_mutable("register acceptance guard %r" % guard.name)
        if any(existing.name == guard.name for existing in self._acceptance_guards):
            raise ValueError("acceptance guard %r is already declared" % guard.name)
        self._acceptance_guards = self._acceptance_guards + (guard,)

    def transaction_plan(self) -> Any:
        """Return the frozen transaction plan, or None when the Program has no controller contract."""
        if self._step_strategy is None:
            return None
        from pops.time._step.strategy import ErrorControlledDt
        if type(self._step_strategy) is ErrorControlledDt and not any(
                guard.role is GuardRole.ERROR_ESTIMATE for guard in self._acceptance_guards):
            raise ValueError(
                "ErrorControlledDt requires a lowered AcceptanceGuard with role=GuardRole.ERROR_ESTIMATE")
        return StepTransactionPlan(
            self._step_strategy, self._transaction_stores, self._acceptance_guards)

    def validate_runtime_controls(self, controls: Any = None) -> bool:
        """Validate run controls against the explicit StepStrategy selected by this Program."""
        if self._step_strategy is None:
            controls = {} if controls is None else dict(controls)
            if controls:
                raise ValueError(
                    "runtime controls require Program.step_strategy(...); got %s"
                    % ", ".join(sorted(controls)))
            return True
        self._step_strategy.validate_runtime_controls(controls)
        return True

    def __str__(self) -> str:
        """Short, deterministic, array-free summary -- never the full SSA IR.

        Prints the program name, the op count and the committed block names (Spec 5 sec.12.1):
        a one-line header, not a node-by-node dump.
        """
        return "Program(name=%r, ops=%d, blocks=%s)" % (
            self.name, len(self._values),
            sorted(block_name(state.block_ref) for state in self._commits))


__all__ = ["Program"]
