"""AmrSystem install/bind mixin (ADC-619 split).

The low-level ``pops.bind`` install seam of :class:`pops.runtime.amr_system.AmrSystem`:
``_install_compiled`` (the native / compiled install orchestration) plus its field-solver,
named-elliptic-field and aux helpers (``_install_solver`` / ``_declared_elliptic_fields`` /
``_install_aux``). Split out of ``amr_system`` for the 500-line cap; mixed into ``AmrSystem``
via inheritance and operating on ``self._s`` (the native facade), ``self._aux_field_index`` and
the other AmrSystem methods (``add_equation`` / ``set_density`` / ``set_poisson`` /
``_finish_program_install``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.runtime._system_unified_install import validate_install_arguments

if TYPE_CHECKING:
    from pops.runtime._amr_system_contract import _AmrSystem
else:
    _AmrSystem = object


class _AmrSystemInstall(_AmrSystem):
    """``pops.bind`` install seam for :class:`AmrSystem` (mixed in; operates on ``self``)."""

    def _install_compiled(self, compiled: Any = None, *, instances: Any = None, params: Any = None,
                          aux: Any = None, solvers: Any = None, cadence: Any = None,
                          outputs: Any = None, diagnostics: Any = None) -> Any:
        """INTERNAL low-level install seam on the AMR hierarchy (Spec 5 sec.11) -- signature parity
        with ``System._install_compiled``. NOT the public entry point: author the run with
        ``pops.bind(...)``, which dispatches System / AmrSystem and calls this seam.

        Runs the SAME early bind-input validation (``validate_install_arguments``: reject -- BEFORE
        any native mutation -- an install missing a REQUIRED argument the artifact declares, with one
        clear actionable error), then lowers to the AMR layer:

          - NATIVE install (``compiled=None``): wires each instance with ``add_equation`` (native
            bricks / a CompiledModel target='amr_system'), sets the field solvers (``set_poisson``),
            the aux inputs (``set_magnetic_field`` / ``set_aux_field``) and each instance's initial
            density (``set_density``). This is the real AMR add path; a full run is Kokkos-gated.
          - COMPILED install (a ``compiled`` handle carrying a time Program, epic ADC-511 / ADC-508):
            the same wiring, then ``install_program(so_path)`` installs the compiled Program on the AMR
            hierarchy (the .so must export ``pops_install_program_amr``: compile it with
            ``target='amr_system'``). The runtime params (``params=``) route to ``set_program_params``
            and the cadence (``cadence=``) to ``set_program_cadence`` -- the AMR counterparts of the
            System routes. The per-level Lie/Strang macro-step DRIVER is Kokkos-gated (the .so fails
            loud at install until the AmrProgramContext seam lands), so a full run is a ROMEO step.

        @param compiled a compiled time-Program handle, or ``None`` for a native AMR install.
        @param instances dict {name: {"initial": array, "spatial": <brick>, "model": <model>,
            "time": <policy>}}; the block is bound by the dict KEY.
        @param params runtime parameters of a COMPILED time Program (dsl.Param kind='runtime'),
            routed to ``set_program_params`` per PROGRAM block. A native AMR install (``compiled=None``)
            has no ``set_block_params`` (the native AMR .so loader does not transport runtime params),
            so a non-empty ``params=`` there raises rather than dropping them silently.
        @param aux dict {field_name: array}: "B_z" -> set_magnetic_field, "T_e" rejected (derived),
            any other -> set_aux_field on the declaring block.
        @param solvers dict {field: <solver>}: lowered to set_poisson (default Poisson field only).
        @param cadence optional pops.time.CompiledTime(substeps=, stride=): the compiled Program's GLOBAL
            macro-step cadence, applied with ``set_program_cadence`` AFTER install_program. A native
            AMR install has no Program, so a non-None cadence there raises (set substeps / stride on
            the native time policy instead).
        """
        # RUNTIME FREEZE (ADC-592): a second install on an already-bound AMR engine is a re-composition
        # and is refused explicitly -- the compiled artifact is bound exactly once.
        from pops.runtime._lifecycle import guard_assembling
        guard_assembling(self, "_install_compiled")
        instances = instances or {}
        params = params or {}
        aux = aux or {}
        solvers = solvers or {}

        # (0) EARLY VALIDATION (shared with System._install_compiled): reject a compiled install missing a
        # required declared argument BEFORE any native mutation. Inert (reads arguments() metadata).
        validate_install_arguments(self, compiled, instances, params, aux, solvers)

        # COMPILED vs NATIVE. COMPILED: `compiled` carries a .so_path time Program (installed in step 5,
        # with the section-24 .so validation) + a PHYSICAL model (the per-block model an instance falls
        # back on). NATIVE: `compiled is None` -- each instance carries its OWN native model. Validate the
        # handle up front, BEFORE any native mutation (no half-configured AMR hierarchy).
        so_path = None
        compiled_model = None
        if compiled is not None:
            so_path = getattr(compiled, "so_path", None)
            if so_path is None:
                raise TypeError(
                    "pops.bind: compiled handle has no .so_path (got %r); pass a compile_problem(...) "
                    "result (target='amr_system'), or compiled=None for a native AMR install (each "
                    "instance carries its own native model)." % type(compiled).__name__)
            compiled_model = getattr(compiled, "model", None)
        # (7) OUTPUT / CHECKPOINT policies and DIAGNOSTIC measures flow onto the AMR engine exactly
        # like the Uniform System (ADC-542 / addendum C.1): AmrSystem.run() fires each at its cadence
        # through the AMR per-level output driver + the composite-reduction diagnostics path. Stored
        # here; the run-loop hook lives on AmrSystem.
        if outputs:
            self._output_policies = list(outputs)
        if diagnostics:
            self._diagnostic_measures = list(diagnostics)

        # (1) FIELD SOLVERS first (parity with System: set_poisson before adding blocks AND before
        # install_program -- the section-24 solver requirement reads the configured solver). The DECLARED
        # named elliptic fields (ADC-428), collected from the per-instance models, widen the accepted
        # solver-field set beyond the default Poisson names: a solver selection for a model-declared named
        # field routes (the native loader wired register_elliptic_field), a typo is rejected against the
        # declared set. Mirror of System._install with _declared_elliptic_fields.
        declared_fields = self._declared_elliptic_fields(instances)
        for field, solver_brick in solvers.items():
            self._install_solver(field, solver_brick, declared_fields)

        # (2) INSTANCES: add each named block (add_equation, binds the Program block of that name), then
        # set its initial density. The block model is the per-instance "model" if given, else the
        # PHYSICAL model carried by the compiled handle (compiled.model) -- NOT the handle itself (the
        # time Program .so installed in step 5).
        for name, spec in instances.items():
            if not isinstance(spec, dict):
                raise TypeError("pops.bind: instances[%r] must be a dict "
                                "(initial/spatial/time/model); got %r"
                                % (name, type(spec).__name__))
            model = spec.get("model", compiled_model)
            if model is None:
                raise ValueError(
                    "pops.bind: instance %r has no block model -- supply "
                    "instances[%r]['model'] (an pops.Model(...) / a target='amr_system' "
                    "CompiledModel), or pass a compiled handle that carries one "
                    "(compile_problem(model=...))." % (name, name))
            spatial = spec.get("spatial")
            time = spec.get("time")
            self.add_equation(name, model, spatial=spatial, time=time)

        # (3) AUX fields: B_z -> set_magnetic_field; named -> set_aux_field. After the blocks exist
        # (a named aux resolves against the block's declared aux table) and BEFORE install_program.
        for field_name, field in aux.items():
            self._install_aux(field_name, field)

        # (4) INITIAL state per instance (set_density on the AMR coarse base level).
        for name, spec in instances.items():
            initial = spec.get("initial")
            if initial is not None:
                self.set_density(name, initial)

        # (5/5b/6) COMPILED time Program: install_program on the AMR hierarchy, route runtime params and
        # apply the global cadence (or reject params= / cadence= on a NATIVE install). Extracted into the
        # _AmrSystemProgram mixin (_finish_program_install) to keep this module under the line budget.
        self._finish_program_install(compiled, so_path, params, cadence)

        # (7) FREEZE (ADC-592): the AMR composition is fully lowered -- build the BoundSnapshot manifest
        # of WHAT was bound (build_amr_snapshot, in _bound_snapshot), then _finalize_bind marks the
        # runtime 'bound' as the LAST act. The AMR route installs no whole-system Program, so
        # program_hash / abi_key / cache_key are None; each block's own CompiledModel hash lands in the
        # per-block snapshot row.
        from pops.runtime._bound_snapshot import build_amr_snapshot
        snapshot = build_amr_snapshot(instances, solvers, aux, params)
        self._finalize_bind(snapshot)  # freeze (ADC-592): _finalize_bind lives on _LifecycleMixin

    # Field names the default AMR Poisson route already serves (the shared coarse elliptic solve).
    _DEFAULT_POISSON_FIELDS = ("phi", "poisson", "charge_density", "default")

    def _install_solver(self, field: Any, solver_brick: Any,
                        declared_fields: Any = frozenset()) -> Any:
        """Lower a field-solver selection to set_poisson (AMR, ADC-428).

        The default Poisson field and any NAMED elliptic field a block's model DECLARES (via
        m.elliptic_field, collected into @p declared_fields) are accepted: the named field's RHS is wired
        by the native AMR loader (register_elliptic_field + set_block_elliptic_field) and solved by the
        AmrRuntime engine each solve_fields, while the solver selection routes through set_poisson for
        both (the AMR solver is always geometric_mg). A field name that is NEITHER the default Poisson
        field NOR a declared named field is a TYPO -- rejected LOUD, naming the declared set. Mirror of
        System._install_solver, minus the System-only solver options the AMR set_poisson lacks."""
        if field not in self._DEFAULT_POISSON_FIELDS and field not in declared_fields:
            declared = ", ".join(sorted(declared_fields)) or "(none declared)"
            raise ValueError(
                "pops.bind: solver selection names field %r, which is neither the default Poisson "
                "field (%s) nor a named elliptic field any installed model declares (declared: %s). "
                "Declare it with m.elliptic_field(%r, rhs=...), or fix the field name."
                % (field, ", ".join(self._DEFAULT_POISSON_FIELDS), declared, field))
        token = solver_brick if isinstance(solver_brick, str) else (
            getattr(solver_brick, "scheme", None) or getattr(solver_brick, "name", None))
        if token is None:
            raise TypeError("pops.bind: solver must be a token string or an "
                            "pops.solvers.<Solver>(...) descriptor; got %r"
                            % type(solver_brick).__name__)
        self.set_poisson(solver=token)

    @staticmethod
    def _declared_elliptic_fields(instances: Any) -> Any:
        """Collect the NAMED elliptic fields declared by the per-instance models (ADC-428). Reads each
        model's declared names WITHOUT compiling: a target='amr_system' CompiledModel exposes
        ``elliptic_field_names``; a raw physics/dsl Model exposes the ``_elliptic_fields`` mapping.
        Returns a set (empty when no model declares a named field). Mirror of
        System._declared_elliptic_fields (no compiled whole-system handle on the AMR path)."""
        names = set()
        for spec in (instances or {}).values():
            if not isinstance(spec, dict):
                continue
            model = spec.get("model")
            if model is None:
                continue
            explicit = getattr(model, "elliptic_field_names", None)
            if explicit is not None:
                names.update(explicit)
                continue
            raw = getattr(model, "_elliptic_fields", None)
            if raw:
                names.update(raw)
        return names

    def _install_aux(self, field_name: Any, field: Any) -> Any:
        """Lower an aux entry on AMR: 'B_z' -> set_magnetic_field; 'T_e' rejected (derived); any
        other name -> set_aux_field on the block that declares it. Mirror of System._install_aux."""
        if field_name == "B_z":
            self.set_magnetic_field(field)
            return
        if field_name == "T_e":
            raise ValueError(
                "pops.bind: aux 'T_e' is DERIVED from a fluid block, not a static aux "
                "field; use set_electron_temperature_from(block).")
        block = None
        for blk, table in self._aux_field_index.items():
            if field_name in table:
                block = blk
                break
        if block is None:
            raise ValueError(
                "pops.bind: aux field %r is not declared by any installed instance; add the "
                "instance with a model declaring m.aux_field(%r)." % (field_name, field_name))
        self.set_aux_field(block, field_name, field)
