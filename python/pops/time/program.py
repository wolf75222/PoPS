"""pops.time.Program -- the compiled time-program authoring class (builder-mode IR).

A ``Program`` BUILDS a typed SSA IR for one time step; Python never executes a numerical stage.
The C++ lowering (``emit_cpp_program``) is a thin delegator to
``pops.codegen.program_codegen`` (lazy import), so this package keeps a strictly acyclic
graph (it imports only ``pops.ir`` / ``pops.model`` and never ``pops.codegen`` / ``_pops`` at
module scope). The class is composed from focused authoring mixins.

cf. docs/sphinx/reference/time-program.md (Phase 8) and the ADC-399 epic.
"""
from __future__ import annotations

from typing import Any

from pops.ir.literals import exact_numeric_scalar
from pops.model.ownership import OwnerKind, OwnerPath
from pops.time.program_authoring import _ProgramAuthoring
from pops.time.program_condensed import _ProgramCondensed
from pops.time.program_core import _ProgramCore
from pops.time.program_dt_bound import _ProgramDtBound
from pops.time.program_history import _ProgramHistory
from pops.time.program_inspect import _ProgramInspect
from pops.time.program_local import _ProgramLocal
from pops.time.program_passes import _ProgramPasses
from pops.time.program_solve import _ProgramSolve
from pops.time.program_time_handles import _ProgramTimeHandles
from pops.time.references import bind_program_block, block_name
from pops.time.values import _Coeff, ProgramValue  # noqa: F401  (ProgramValue used by mixins via prog ref)


class Program(_ProgramTimeHandles, _ProgramCore, _ProgramLocal, _ProgramCondensed,
              _ProgramHistory, _ProgramSolve,
              _ProgramAuthoring, _ProgramDtBound, _ProgramPasses, _ProgramInspect):
    """A compiled time program (builder mode). Holds the SSA value list and the committed
    blocks. The Python object only BUILDS the IR; it is never executed numerically during
    ``sim.step``. Authoring methods come from the mixins; C++ emission is delegated to
    ``pops.codegen.program_codegen`` via :meth:`emit_cpp_program`.
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
        # P.call requires a typed operator handle and the public P.rhs requires the typed terms= list
        # (the legacy string operator name / flux=/sources= form is REFUSED). The byte-identical
        # builders survive ONLY as the internal _call / _rhs_legacy, which the typed front doors and
        # the pops.lib.time macros lower through -- no opt-in flag, no second public path.
        self._values = []
        self._issued_values = {}  # id -> strong identity, including stale immutable replacement records
        self._next_id = 0
        self._commits = {}      # qualified state Handle -> State value
        self._recording = []    # stack of sub-block lists (a control-flow body); see _new / while_
        self._next_region = 1
        self._recording_regions = {}  # id(list) -> (strong list ref, exact authoring-region token)
        self._region_imports = {}  # destination region -> explicitly sanctioned source regions
        self._state_spaces = {}  # block -> StateSpace or None; typed/untyped mixing is forbidden
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
        self._default_state_spaces = {}
        self._default_field_spaces = {}
        # OPTIONAL debug capture of the authoring source location per IR node (ADC-530). DEFAULT OFF:
        # a stack walk per node is too costly for the normal build path, and the location is
        # INSPECTION-ONLY (never serialized into the IR / the hash). Toggle with
        # capture_source_locations(True) before building to populate ProgramValue.source_location.
        self._capture_source = False
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

        ``pops.compile`` freezes the time Program it lowers; emission (``emit_cpp_program``) and the
        IR hash are pure reads and remain allowed. Every owned list/dict/set is replaced by an
        immutable copy, so a stale authoring reference cannot alter the compiled identity. Idempotent.
        """
        if self._frozen:
            return self
        from pops.time.program_freeze import freeze_program_tables
        freeze_program_tables(self)
        return self

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

    def __str__(self) -> str:
        """Short, deterministic, array-free summary -- never the full SSA IR.

        Prints the program name, the op count and the committed block names (Spec 5 sec.12.1):
        a one-line header, not a node-by-node dump.
        """
        return "Program(name=%r, ops=%d, blocks=%s)" % (
            self.name, len(self._values),
            sorted(block_name(state.block_ref) for state in self._commits))

    # --- C++ codegen (lowering to a problem.so source) lives in pops.codegen; the authoring
    # Program delegates via a LAZY import so pops.time stays free of any codegen/_pops edge. ---
    def emit_cpp_program(self, model: Any = None, target: Any = "system") -> Any:
        """Generate the C++ source of a problem.so implementing this Program (codegen).

        Thin authoring entry point: delegates to the free function
        :func:`pops.codegen.program_codegen.emit_cpp_program`, imported lazily so the
        ``pops.time`` package never imports ``pops.codegen`` / ``_pops`` at module scope. See
        the codegen function for the full lowering contract.

        @param target ``"system"`` (default) emits ``pops_install_program`` (installable on
            ``System``); ``"amr_system"`` ALSO emits ``pops_install_program_amr`` (the AMR
            install entry ``AmrSystem::install_program`` resolves), epic ADC-511 / ADC-508.
        """
        from pops.codegen import program_codegen as _pcg
        return _pcg.emit_cpp_program(self, model=model, target=target)

    def _check_lowerable(self, model: Any = None) -> Any:
        """Raise if the IR uses a construct the codegen cannot lower (delegates to
        :func:`pops.codegen.program_codegen._check_lowerable`, lazy import).
        """
        from pops.codegen import program_codegen as _pcg
        return _pcg._check_lowerable(self, model)

    def _check_schedules_lowerable(self) -> Any:
        """Raise if a node carries a schedule the codegen cannot lower (delegates to
        :func:`pops.codegen.program_codegen._check_schedules_lowerable`, lazy import).
        """
        from pops.codegen import program_codegen as _pcg
        return _pcg._check_schedules_lowerable(self)

    def _emit_body(self, model: Any = None) -> Any:
        """Lower the install-function body to ``(prelude, body)`` C++ (delegates to
        :func:`pops.codegen.program_codegen._emit_body`, lazy import). Exposed for the codegen
        tests that assert the body shape directly.
        """
        from pops.codegen import program_codegen as _pcg
        return _pcg._emit_body(self, model)


class CompiledTime:
    """Record of a compiled `Program`'s macro-step cadence (`substeps` / `stride`).

    A compiled Program OWNS the whole step body: it is installed via `sim.install_program` and driven
    by `sim.step(dt)`. Its cadence is applied to the System with `sim.set_program_cadence(substeps,
    stride)` (call it after `install_program`); a `CompiledTime` just records those values. The
    compiled program is NOT attached via `sim.add_equation(time=CompiledTime(...))` -- that path is
    rejected with an explicit error (the transport policy passed to `add_equation` is a native
    `pops.Explicit`/etc.; the compiled program is installed separately). `substeps` and
    `stride` are wired (ADC-411) as a SYSTEM-level orchestration AROUND the opaque program closure
    (`System.set_program_cadence`, mirroring the native per-block advance loop): `substeps=n` runs the
    program n times over `eff_dt/n`; `stride=M` runs the whole program once per M macro-steps with
    `eff_dt = M*dt` (GLOBAL hold-then-catch-up, the clock still ticks every macro-step).

    Two semantic limits to keep in mind (cf. system_stepper.hpp):
      - `substeps > 1` is bit-exact vs native `pops.Explicit(substeps=n)` ONLY for an UNCOUPLED /
        transport-only program: `program_step_(h)` re-runs the WHOLE program (its `solve_fields`
        included), whereas native substeps subdivides ONLY the transport (solve_fields runs once).
      - `stride` here is GLOBAL (a compiled program is one whole-system closure), so it equals native
        per-block stride only for a single-block system (or all blocks sharing the stride).

    A NUMERIC `cfl` (e.g. `cfl=0.5`) is wired: it is applied at RUNTIME by `sim.run(cfl=...)`, whose
    `cfl` defaults to the installed cadence's `cfl` when the caller passes none, so a bare
    `sim.run(t_end)` after `bind(..., cadence=CompiledTime(cfl=0.5))` advances at `cfl=0.5` (the
    per-block CFL `dt` is computed in PoPS and drives the installed Program). A self-computed `cfl`
    SUB-PROGRAM (`cfl="program"`) is still deferred (the Program would export its own dt bound); it
    fails loud rather than being silently ignored."""

    def __init__(self, substeps: Any = 1, stride: Any = 1, cfl: Any = "default") -> None:
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError("CompiledTime: substeps must be a positive int (got %r)" % (substeps,))
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError("CompiledTime: stride must be a positive int (got %r)" % (stride,))
        # A numeric cfl is wired (applied at runtime via sim.run(cfl=)); only a non-numeric,
        # non-"default" cfl (e.g. cfl="program", a self-computed dt sub-program) is still deferred.
        if cfl != "default":
            if isinstance(cfl, str):
                raise NotImplementedError(
                    "CompiledTime: a self-computed cfl sub-program (cfl=%r) is deferred (ADC-401 "
                    "Phase 2c); pass a positive numeric cfl=<value> or an explicit dt to sim.step(dt)"
                    % (cfl,))
            cfl = exact_numeric_scalar(cfl, where="CompiledTime cfl")
            if cfl <= 0:
                raise ValueError("CompiledTime: cfl must be > 0 (got %r)" % (cfl,))
        self.substeps = substeps
        self.stride = stride
        self.cfl = cfl
        self.kind = "compiled"

    def __repr__(self) -> str:
        return "CompiledTime(substeps=%d, stride=%d, cfl=%r)" % (self.substeps, self.stride, self.cfl)
