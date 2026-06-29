"""pops.time.Program -- the compiled time-program authoring class (builder-mode IR).

A ``Program`` BUILDS a typed SSA IR for one time step; Python never executes a numerical stage.
The C++ lowering (``emit_cpp_program``) is a thin delegator to
``pops.codegen.program_codegen`` (lazy import), so this package keeps a strictly acyclic
graph (it imports only ``pops.ir`` / ``pops.model`` and never ``pops.codegen`` / ``_pops`` at
module scope). The class is composed from focused authoring mixins.

cf. docs/sphinx/reference/time-program.md (Phase 8) and the ADC-399 epic.
"""
from pops.time.program_authoring import _ProgramAuthoring
from pops.time.program_core import _ProgramCore
from pops.time.program_inspect import _ProgramInspect
from pops.time.program_local import _ProgramLocal
from pops.time.program_passes import _ProgramPasses
from pops.time.program_solve import _ProgramSolve
from pops.time.values import _Coeff, Value  # noqa: F401  (Value used by mixins via prog ref)


class Program(_ProgramCore, _ProgramLocal, _ProgramSolve, _ProgramAuthoring,
              _ProgramPasses, _ProgramInspect):
    """A compiled time program (builder mode). Holds the SSA value list and the committed
    blocks. The Python object only BUILDS the IR; it is never executed numerically during
    ``sim.step``. Authoring methods come from the mixins; C++ emission is delegated to
    ``pops.codegen.program_codegen`` via :meth:`emit_cpp_program`.
    """

    def __init__(self, name):
        self.name = name
        # De-stringing is the ONE public path (Spec 5 sec.15, ADC-479 criteria 23 + 27): the public
        # P.call requires a typed operator handle and the public P.rhs requires the typed terms= list
        # (the legacy string operator name / flux=/sources= form is REFUSED). The byte-identical
        # builders survive ONLY as the internal _call / _rhs_legacy, which the typed front doors and
        # the pops.lib.time macros lower through -- no opt-in flag, no second public path.
        self._values = []
        self._next_id = 0
        self._commits = {}      # block -> State value
        self._commit_fields = {}  # block -> optional FieldContext value associated with the commit
        self._recording = []    # stack of sub-block lists (a control-flow body); see _new / while_
        self._histories = {}    # name -> max declared lag (multistep histories; ADC-406a)
        # OPTIONAL dt bound (spec s18 / ADC-417): a recorded scalar sub-program (cfl -> Scalar) the
        # generated .so exports as pops_program_dt_bound; None = no bound (the native CFL is used).
        self._dt_bound = None        # (block, scalar_value) once set; the block is the scalar sub-block
        self.dt = _Coeff({1: 1.0})   # symbolic time step; participates in coefficient arithmetic
        # OPTIONAL bound operator registry (Spec 2, operator-first): set by bind_operators so P.call
        # can resolve and type-check operators at build time. None = legacy PDE-shortcut-only Program.
        self._registry = None
        # Per-emit scratch names of coupled_rate blocks, keyed by (coupled node id, block): the
        # coupled_rate kernel fills them and each coupled_rate_out projection aliases its block's
        # scratch (ADC-457). Populated during _emit_op; harmless to keep across emits (keys are unique
        # per node id).
        self._coupled_scratch = {}

    def __str__(self):
        """Short, deterministic, array-free summary -- never the full SSA IR.

        Prints the program name, the op count and the committed block names (Spec 5 sec.12.1):
        a one-line header, not a node-by-node dump.
        """
        return "Program(name=%r, ops=%d, blocks=%s)" % (
            self.name, len(self._values), sorted(self._commits))

    # --- C++ codegen (lowering to a problem.so source) lives in pops.codegen; the authoring
    # Program delegates via a LAZY import so pops.time stays free of any codegen/_pops edge. ---
    def emit_cpp_program(self, model=None, *, layout=None):
        """Generate the C++ source of a problem.so implementing this Program (inspection).

        Thin authoring entry point: delegates to the free function
        :func:`pops.codegen.program_codegen.emit_cpp_program`, imported lazily so the
        ``pops.time`` package never imports ``pops.codegen`` / ``_pops`` at module scope.

        ``target=`` is intentionally not a public argument. Pass ``layout=AMR(...)`` to inspect
        the AMR install ABI; omit ``layout`` for the uniform System ABI. The internal codegen driver
        calls :meth:`_emit_cpp_program_for_target` after deriving that target from the Case layout.
        """
        target = self._target_from_layout(layout)
        return self._emit_cpp_program_for_target(model=model, target=target)

    def _emit_cpp_program_for_target(self, model=None, target="system"):
        """Internal codegen seam using the native ABI token derived from a typed layout."""
        if target not in ("system", "amr_system"):
            raise ValueError("_emit_cpp_program_for_target: target must be 'system' or 'amr_system'")
        from pops.codegen import program_codegen as _pcg
        return _pcg.emit_cpp_program(self, model=model, target=target)

    @staticmethod
    def _target_from_layout(layout):
        if layout is None:
            return "system"
        from pops.mesh.layouts import AMR, Uniform
        if isinstance(layout, AMR):
            return "amr_system"
        if isinstance(layout, Uniform):
            return "system"
        raise TypeError(
            "Program.emit_cpp_program: layout must be a typed pops.mesh.layouts.Uniform(...) "
            "or AMR(...) descriptor, not %r" % type(layout).__name__)

    def _check_lowerable(self, model=None):
        """Raise if the IR uses a construct the codegen cannot lower (delegates to
        :func:`pops.codegen.program_codegen._check_lowerable`, lazy import).
        """
        from pops.codegen import program_codegen as _pcg
        return _pcg._check_lowerable(self, model)

    def _check_schedules_lowerable(self):
        """Raise if a node carries a schedule the codegen cannot lower (delegates to
        :func:`pops.codegen.program_codegen._check_schedules_lowerable`, lazy import).
        """
        from pops.codegen import program_codegen as _pcg
        return _pcg._check_schedules_lowerable(self)

    def _emit_body(self, model=None):
        """Lower the install-function body to ``(prelude, body)`` C++ (delegates to
        :func:`pops.codegen.program_codegen._emit_body`, lazy import). Exposed for the codegen
        tests that assert the body shape directly.
        """
        from pops.codegen import program_codegen as _pcg
        return _pcg._emit_body(self, model)
