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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pops.runtime._system_unified_install import validate_install_arguments

if TYPE_CHECKING:
    from pops.runtime._amr_system_contract import _AmrSystem
else:
    _AmrSystem = object


class _AmrSystemInstall(_AmrSystem):
    """``pops.bind`` install seam for :class:`AmrSystem` (mixed in; operates on ``self``)."""

    def _install_compiled(self, compiled: Any = None, *, instances: Any = None, params: Any = None,
                          aux: Any = None, solvers: Any = None, field_plans: Any = None,
                          cadence: Any = None,
                          outputs: Any = None, diagnostics: Any = None,
                          bind_schema: Any = None, initial_values: Any = (),
                          bootstrap_plan: Any = None, amr_transfer: Any = None) -> Any:
        """INTERNAL low-level install seam on the AMR hierarchy (Spec 5 sec.11) -- signature parity
        with ``System._install_compiled``. NOT the public entry point: author the run with
        ``pops.bind(...)``, which dispatches System / AmrSystem and calls this seam.

        Runs the SAME early bind-input validation (``validate_install_arguments``: reject -- BEFORE
        any native mutation -- an install missing a REQUIRED argument the artifact declares, with one
        clear actionable error), then lowers to the AMR layer:

          - NATIVE install (``compiled=None``): wires each InstallPlan ``CompiledModel`` with
            ``add_equation``, sets the field solvers (``set_poisson``),
            the aux inputs (``set_magnetic_field`` / ``set_aux_field``) and each instance's initial
            density (``set_density``). This is the real AMR add path; a full run is Kokkos-gated.
          - COMPILED install (a ``compiled`` handle carrying a time Program, epic ADC-511 / ADC-508 /
            ADC-634): the same wiring, then ``install_program(so_path)`` installs the compiled Program
            on the AMR hierarchy (the .so must export ``pops_install_program_amr``: compile it with
            ``target='amr_system'``). The runtime params (``params=``) route to ``set_program_params``
            and the cadence (``cadence=``) to ``set_program_cadence`` -- the AMR counterparts of the
            System routes. The per-level macro-step driver is the AmrProgramContext seam (ADC-508); a
            Program using a deferred op (Schur / history / named-flux) compiles against it and throws
            the honest AmrProgramContext backstop only when that op is reached at run.

        @param compiled a compiled time-Program handle, or ``None`` for a native AMR install.
        @param instances dict {name: {"initial": array, "spatial": <brick>, "model": <CompiledModel>,
            "time": <policy>}}; the block is bound by the dict KEY.
        @param params canonical block-qualified runtime values resolved by BindSchema and routed to
            ``set_program_params`` per PROGRAM block. A native AMR install (``compiled=None``)
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
        params = {} if params is None else params
        aux = aux or {}
        solvers = solvers or {}
        field_plans = field_plans or {}

        # (0) EARLY VALIDATION (shared with System._install_compiled): reject a compiled install missing a
        # required declared argument BEFORE any native mutation. Inert (reads arguments() metadata).
        validate_install_arguments(
            self, compiled, instances, params, aux, solvers or field_plans)
        if amr_transfer is not None:
            self._install_bootstrap_routes(amr_transfer)

        # COMPILED vs NATIVE. COMPILED: `compiled` carries a .so_path time Program (installed in step 5,
        # with the section-24 .so validation). Every block model comes exclusively from InstallPlan;
        # neither route falls back to compiled.model or a live authoring builder.
        so_path = None
        if compiled is not None:
            so_path = getattr(compiled, "so_path", None)
            if so_path is None:
                raise TypeError(
                    "pops.bind: compiled handle has no .so_path (got %r); pass a compile_problem(...) "
                    "result (target='amr_system'), or compiled=None for a native AMR install (each "
                    "instance carries its own native model)." % type(compiled).__name__)
        if outputs or diagnostics:
            raise ValueError(
                "native AMR install does not accept free output/diagnostic lists; "
                "declare exact ConsumerGraph nodes on the compiled plan"
            )

        # (1) FIELD SOLVERS first (parity with System: set_poisson before adding blocks AND before
        # install_program -- the section-24 solver requirement reads the configured solver). The DECLARED
        # named elliptic fields (ADC-428), collected from the per-instance models, widen the accepted
        # solver-field set beyond the default Poisson names: a solver selection for a model-declared named
        # field routes (the native loader wired register_elliptic_field), a typo is rejected against the
        # declared set. Mirror of System._install with _declared_elliptic_fields.
        declared_fields = self._declared_elliptic_fields(instances)
        if field_plans and solvers:
            raise ValueError("pops.bind: field_plans and legacy solvers cannot both be installed")
        for field, field_plan in field_plans.items():
            self._install_field_plan(field, field_plan)
        for field, solver_brick in solvers.items():
            self._install_solver(field, solver_brick, declared_fields)

        # (2) INSTANCES: add each named block (add_equation, binds the Program block of that name), then
        # set its initial density. The per-instance detached CompiledModel is mandatory.
        # resolved_models is reused by step-4b (ADC-514) to route
        # the native per-block runtime params to set_block_params: it maps each instance name to the
        # detached CompiledModel add_equation received (target='amr_system' metadata exposes
        # runtime_param_names).
        from pops.codegen.loader import CompiledModel

        resolved_models = {}
        for name, spec in instances.items():
            if not isinstance(spec, Mapping):
                raise TypeError("pops.bind: instances[%r] must be a mapping "
                                "(initial/spatial/time/model); got %r"
                                % (name, type(spec).__name__))
            model = spec.get("model")
            if not isinstance(model, CompiledModel):
                raise TypeError(
                    "pops.bind: instances[%r] must carry a detached target='amr_system' "
                    "CompiledModel from InstallPlan; rebuild with pops.compile(...)" % name
                )
            spatial = spec.get("spatial")
            time = spec.get("time")
            self.add_equation(name, model, spatial=spatial, time=time)
            resolved_models[name] = model

        # (3) AUX fields: B_z -> set_magnetic_field; named -> set_aux_field. After the blocks exist
        # (a named aux resolves against the block's declared aux table) and BEFORE install_program.
        for field_name, field in aux.items():
            self._install_aux(field_name, field)

        # (4) INITIAL state: legacy density or the canonical full conservative-state manifest.
        initial_rows = tuple(initial_values)
        if initial_rows and any(spec.get("initial") is not None for spec in instances.values()):
            raise ValueError("pops.bind: duplicate legacy and InitialConditionPlan state authorities")
        for name, spec in instances.items():
            initial = spec.get("initial")
            if initial is not None:
                self.set_density(name, initial)
        seen_initial = set()
        for subject_id, name, initial, space, centering, method, source in initial_rows:
            if method == "analytic":
                route = source.get("native_route")
                if route == "constant_field":
                    components = [
                        float.fromhex(value["binary64"])
                        if isinstance(value, Mapping) and "binary64" in value else float(value)
                        for value in source.get("components", ())
                    ]
                    self._s._register_analytic_constant(
                        subject_id, name or "", space, centering, components
                    )
                elif route == "gaussian_field":
                    center = source.get("center", {})
                    if space != "cell" or set(center) != {"x", "y"}:
                        raise ValueError(
                            "pops.bind: gaussian_field requires one cell state and x/y center"
                        )
                    def native_float(value: Any) -> float:
                        return float.fromhex(value["binary64"]) \
                            if isinstance(value, Mapping) and "binary64" in value \
                            else float(value)

                    self._s._register_analytic_gaussian(
                        subject_id, name or "", native_float(center["x"]),
                        native_float(center["y"]), native_float(source["background"]),
                        native_float(source["amplitude"]),
                        native_float(source["inverse_width"]),
                    )
                else:
                    raise NotImplementedError(
                        "pops.bind: no native analytic provider for route %r" % route)
                continue
            if space == "cell":
                if name not in instances:
                    raise ValueError("pops.bind: initial state targets unknown block %r" % name)
                if name in seen_initial:
                    raise ValueError(
                        "pops.bind: multiple initial physical states target block %r" % name
                    )
                seen_initial.add(name)
                self._s._bind_bootstrap_block_subject(subject_id, name)
                self.set_conservative_state(name, initial)
            elif space in {"face", "node"}:
                self._s._register_bootstrap_array(subject_id, centering, initial)
            else:
                raise NotImplementedError(
                    "pops.bind: native bootstrap has no payload carrier for space %r" % space
                )

        # (4b) BindSchema supplies complete block-qualified vectors. Install them on the native
        # block carrier; a compiled Program receives the same values on its own carrier below.
        if bind_schema is None and compiled is not None:
            bind_schema = getattr(compiled, "bind_schema", None)
        if bind_schema is not None:
            self._install_block_params(resolved_models, bind_schema, params)
        elif params:
            raise ValueError(
                "pops.bind: parameter values require a compiled artifact carrying BindSchema"
            )
        for field_plan in field_plans.values():
            self._install_field_boundary_parameters(field_plan, params, compiled=compiled)

        if bootstrap_plan is not None:
            from pops.runtime._amr_bootstrap_execution import execute_native_bootstrap

            self._bootstrap_execution = execute_native_bootstrap(
                self, bootstrap_plan, initial_rows
            )

        # (5/5b/6) COMPILED time Program: install_program on the AMR hierarchy, route the REMAINING runtime
        # params and apply the global cadence (or reject a leftover params= / cadence= on a NATIVE install).
        # Extracted into the _AmrSystemProgram mixin (_finish_program_install) to keep this module small.
        self._finish_program_install(compiled, so_path, bind_schema, params, cadence)

        # (7) FREEZE (ADC-592): the AMR composition is fully lowered -- build the BoundSnapshot manifest
        # of WHAT was bound (build_amr_snapshot, in _bound_snapshot), then _finalize_bind marks the
        # runtime 'bound' as the LAST act. If this route installed a whole-system Program, its
        # program/cache/ABI identity and cadence are retained alongside each block-model hash.
        from pops.runtime._bound_snapshot import build_amr_snapshot
        snapshot = build_amr_snapshot(
            self, compiled, instances, solvers or field_plans, cadence, aux, params
        )
        self._finalize_bind(snapshot)  # freeze (ADC-592): _finalize_bind lives on _LifecycleMixin

    def _install_block_params(self, resolved_models: Any, schema: Any, params: Any) -> None:
        """Install complete owner-qualified block vectors from BindSchema."""
        from pops.runtime._install_param_routing import route_block_params
        per_block = route_block_params(resolved_models, schema, params)
        for name, values in per_block.items():
            self.set_block_params(name, values)

    def _install_bootstrap_routes(self, registry: Any) -> None:
        from pops.mesh.amr.transfer import ApplyTransferProvider, ResolvedAMRTransfer

        if type(registry) is not ResolvedAMRTransfer:
            raise TypeError("pops.bind: amr_transfer must be an exact AMRTransfer")
        face_vectors = set()
        for entry in registry.entries:
            action = entry.action
            provider = action.provider
            provider_options = provider.options.to_data()
            paired = provider_options.get("paired_subjects")
            if paired is not None:
                pair = tuple(paired)
                if len(pair) != 2 or any(not isinstance(value, str) for value in pair):
                    raise ValueError("pops.bind: paired face provider has an invalid subject manifest")
                face_vectors.add(pair)
            key = entry.key.to_data()
            if type(action) is ApplyTransferProvider:
                options = action.route.options.to_data()
                capabilities = action.capabilities
                order, ghost = capabilities.order, capabilities.ghost_depth
            else:
                options = provider_options
                order, ghost = 1, (0,)
            dimensions = {row.accuracy.dimension for row in entry.requirements}
            if len(dimensions) != 1:
                raise ValueError("pops.bind: one native transfer route cannot mix dimensions")
            ratios = {
                tuple(row.accuracy.refinement_ratio) for row in entry.requirements
            }
            if len(ratios) != 1 or len(set(next(iter(ratios)))) != 1:
                raise ValueError(
                    "pops.bind: one native transfer route requires one isotropic ratio"
                )
            self._s._register_bootstrap_transfer_route(
                entry.key.identity.token,
                [row.subject.qualified_id for row in entry.requirements],
                provider.qualified_id,
                key["space"]["name"],
                key["centering"]["name"],
                key["representation"]["name"],
                key["storage"]["name"],
                key["operation"]["name"],
                options["native_route"],
                order,
                ghost,
                next(iter(dimensions)),
                next(iter(ratios))[0],
            )
        for pair in sorted(face_vectors):
            self._s._register_bootstrap_face_vector(pair)

    # Field names the default AMR Poisson route already serves (the shared coarse elliptic solve).
    _DEFAULT_POISSON_FIELDS = ("phi", "poisson", "charge_density", "default")

    def _install_field_plan(self, field: Any, field_plan: Any) -> None:
        """Install the complete resolved AMR field route before native block loaders run."""
        from pops.codegen.field_install import ResolvedFieldInstallPlan
        if not isinstance(field_plan, ResolvedFieldInstallPlan):
            raise TypeError("install field_plans must contain ResolvedFieldInstallPlan values")
        if field_plan.name != field or field_plan.target != "amr_system":
            raise ValueError("resolved AMR field install plan identity/target mismatch")
        field_plan.__post_init__()
        options = field_plan.native_options
        routes = options["provider_pack"]
        output_route = options["output_route"]
        from pops.identity import canonical_bytes
        solver = field_plan.discretization.solver
        if options["solver"] != "geometric_mg":
            raise ValueError("AMR field plan requires GeometricMG")
        mg_fn = getattr(solver, "mg_options", None)
        mg = mg_fn() if callable(mg_fn) else {}
        from pops.solvers._numeric import native_float
        mg_args = {
            "rel_tol": 1.0e-8, "max_cycles": 50, "min_coarse": 2,
            "pre_smooth": 2, "post_smooth": 2, "bottom_sweeps": 50,
            "coarse_threshold": 0, **dict(mg),
        }
        slot = options["provider_slot"]
        self._s.set_field_solver_plan(
            slot, options["provider_identity_text"],
            canonical_bytes(output_route["owner_identity"]).decode("utf-8"),
            output_route["owner_block"],
            output_route["key"],
            [canonical_bytes(route["provider_identity"]).decode("utf-8")
             for route in routes],
            [route["owner_block"] for route in routes],
            [route["key"] for route in routes],
            [route["coefficient"] for route in routes],
            options["solver"], options["hierarchy"],
            native_float(mg.get("abs_tol", 0.0),
                         where="AMR field plan absolute tolerance"),
            native_float(mg_args["rel_tol"],
                         where="AMR field plan relative tolerance"),
            mg_args["max_cycles"], mg_args["min_coarse"], mg_args["pre_smooth"],
            mg_args["post_smooth"], mg_args["bottom_sweeps"],
            mg_args["coarse_threshold"])
        faces = options["boundary_faces"]
        if faces is not None:
            self._s.set_field_boundary_plan(
                slot, [face["type"] for face in faces],
                [face["alpha"] for face in faces],
                [face["beta"] for face in faces],
                [face["value"] for face in faces])
        dependencies = options["boundary_dependencies"]
        self._s.set_field_boundary_dependencies(
            slot,
            [row["owner_block"] for row in dependencies["states"]],
            [row["component"] for row in dependencies["states"]],
            [], [], [])
        self._s.set_field_nullspace(
            slot, options["nullspace"] == "constant", options["gauge"] == "mean_zero")
        nonlinear = options.get("nonlinear")
        if nonlinear is not None:
            field_plan.nonlinear_provider.install(self._s, slot)

    def _install_field_boundary_parameters(self, field_plan: Any, params: Any, *,
                                           compiled: Any) -> None:
        if not field_plan.native_options.get("boundary_kernel_required"):
            return
        if compiled is None:
            raise ValueError(
                "dynamic AMR field boundaries require a compiled artifact that owns their "
                "generated device launchers")
        handles = field_plan.boundary_parameter_handles()
        missing = [handle.qualified_id for handle in handles if handle not in params]
        if missing:
            raise ValueError(
                "dynamic AMR field boundary parameter pack is incomplete: %s" %
                ", ".join(missing))
        from pops.solvers._numeric import native_float
        values = [native_float(params[handle], where="dynamic AMR field boundary parameter %s" %
                               handle.qualified_id) for handle in handles]
        self._s.set_field_boundary_parameters(
            field_plan.native_options["provider_slot"], values)

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
        # ADC-645: GeometricMG(amr_composite=CompositeFAC(...)) opts the AMR FIELD solve into the
        # native composite FAC path. None (default) forwards NOTHING extra, so the native call is
        # byte-identical to the historical set_poisson(solver=token) (Option A).
        composite = getattr(solver_brick, "amr_composite", None)
        if composite is not None:
            self.set_poisson(solver=token, **composite.set_poisson_kwargs())
        else:
            self.set_poisson(solver=token)

    @staticmethod
    def _declared_elliptic_fields(instances: Any) -> Any:
        """Collect named fields exclusively from detached per-block CompiledModel metadata."""
        from pops.codegen.loader import CompiledModel

        names = set()
        for block_name, spec in (instances or {}).items():
            if not isinstance(spec, Mapping):
                raise TypeError("pops.bind: instances[%r] must be a mapping" % block_name)
            model = spec.get("model")
            if not isinstance(model, CompiledModel):
                raise TypeError(
                    "pops.bind: instances[%r] must carry a detached CompiledModel from InstallPlan"
                    % block_name
                )
            declared = getattr(model, "elliptic_field_names", None)
            if declared is None:
                raise ValueError(
                    "pops.bind: CompiledModel for block %r lacks elliptic_field_names metadata; "
                    "rebuild it with pops.compile(...)" % block_name
                )
            names.update(declared)
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
