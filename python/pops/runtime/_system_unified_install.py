"""System unified-install mixin (Spec-4 PR-F): the INTERNAL ``_install_compiled`` seam.

``_install_compiled`` (the low-level seam that lowers to add_equation / set_poisson /
set_magnetic_field / set_aux_field / install_program) plus its private
lowering helpers. It is NOT the public entry point (Spec 5 sec.11): authors call
``pops.bind(artifact, initial_state=..., params=..., aux=..., resources=..., initial_values=...)``;
binding dispatches to the private System / AmrSystem engine and calls this seam. Mixed into
``System`` via inheritance; methods operate on ``self`` (calling the
other mixins' methods) and ``self._s``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pops.runtime._install_param_routing import route_block_params, route_program_params
from pops.runtime._engine_descriptors import Spatial

from pops.runtime._bind_validation import (
    collect_missing_arguments as _collect_missing_arguments_impl,
    validate_install_arguments as _validate_install_arguments_impl,
)

if TYPE_CHECKING:
    from pops.runtime._system_contract import _System
else:
    _System = object

# ADC-613: the GeometricMG V-cycle kwargs set_poisson accepts, minus abs_tol (routed separately so
# the historical abs_tol path keeps working when no typed descriptor is present).
_MG_SET_POISSON_KEYS = ("rel_tol", "max_cycles", "min_coarse", "pre_smooth", "post_smooth",
                        "bottom_sweeps", "coarse_threshold")


def _mg_set_poisson_kwargs(mg_options: Any) -> Any:
    """Translate a GeometricMG.mg_options() dict into set_poisson keyword args (ADC-613).

    Empty in -> empty out, so a string-token / lib-descriptor solver selection leaves set_poisson at
    its native V-cycle defaults (bit-identical). Only the keys the resolver produced are forwarded."""
    from pops.solvers._numeric import native_float
    result = {k: mg_options[k] for k in _MG_SET_POISSON_KEYS if k in mg_options}
    if "rel_tol" in result:
        result["rel_tol"] = native_float(result["rel_tol"], where="GeometricMG relative tolerance")
    return result


class _SystemUnifiedInstall(_System):
    """The internal ``_install_compiled`` lowering seam of System (driven by ``pops.bind``)."""

    def _install_compiled(self, compiled=None, *, instances=None, params=None, aux=None,
                          solvers=None, field_plans=None, install_plan=None):
        """INTERNAL low-level install seam (Spec 5 sec.11): wire a compiled handle + per-instance
        state/spatial + params + aux + field solvers in ONE call, then install the compiled time
        Program. NOT the public entry point: author the run with ``pops.bind(artifact,
        initial_state=..., params=..., aux=..., resources=..., initial_values=...)``. Binding
        dispatches to the private System / AmrSystem engine and calls this seam. This
        method is undocumented on the public surface (it carries no ``install`` alias) and may change.

        It LOWERS to the existing lower-layer calls
        (add_equation / set_poisson / set_magnetic_field / set_aux_field /
        install_program) -- there is NO parallel runtime (Spec section 3). The lower-layer calls stay
        available and unchanged; this seam just sequences them in the right order so the
        install-time validation (section 24) sees a fully-configured simulation.

        install() is the ONE entry for BOTH runtime modes (Spec 4 amendment): a COMPILED-program sim
        (pass the compiled Program handle as ``compiled``) and a per-block native sim
        (``compiled=None``; each InstallPlan instance still carries a detached CompiledModel).

        @param compiled the compiled problem handle (compile_problem(...) result) carrying ``so_path``,
            installed via install_program after every instance/solver/aux is wired. Pass ``None`` for a
            native per-block sim: no Program is installed; each instance must still supply its own
            InstallPlan ``CompiledModel`` and optional ``"time"`` policy.
        @param instances dict {name: {"initial": array, "spatial": <descriptor>,
            "model": <CompiledModel>, "time": <private engine policy>}}. The block is bound by the
            dict KEY @p name (Spec criterion
            23), not a "state" field. Each entry adds the named block (add_equation), sets its
            "initial" state (if given) and lowers the "spatial" brick to the add_equation spatial args.
            The block model is always the per-instance ``"model"`` from InstallPlan. Public
            ``spatial`` authoring uses ``pops.numerics.FiniteVolume(...)``; an already-lowered
            private ``Spatial`` adapter is accepted only inside the install pipeline.
        @param params complete mapping from canonical, block-qualified ParamHandle values to their
            resolved runtime values. BindSchema has already applied defaults and derived values.
        @param aux dict {field_name: array}: "B_z" -> set_magnetic_field, "T_e" -> rejected (it is
            DERIVED, use set_electron_temperature_from), any other -> set_aux_field on the instance
            declaring it. Set BEFORE install_program so the section-24 aux requirement check sees it.
        @param solvers dict {field: <pops.solvers.GeometricMG(...)>}: lowered to
            set_poisson(solver=...). The default Poisson field ("phi"/"charge_density"/"poisson") and
            any NAMED elliptic field a block's model DECLARES (m.elliptic_field) are accepted and route
            through the shared system elliptic solver; a field name no model declares raises (typo).
        @throws the verbatim Spec section-24 errors at install (missing aux / solver / block instance /
            Riemann capability). A disallowed schedule is rejected earlier, at Program compile.
        """
        # RUNTIME FREEZE (ADC-592): a second install on an already-bound engine is refused explicitly.
        from pops.runtime._lifecycle import guard_assembling
        guard_assembling(self, "_install_compiled")
        instances = instances or {}
        params = {} if params is None else params
        aux = aux or {}
        solvers = solvers or {}
        field_plans = field_plans or {}
        if solvers and field_plans:
            raise ValueError("install received both legacy solvers and resolved field_plans")
        validation_solvers = solvers or field_plans

        # (0) EARLY VALIDATION (Spec 5 sec.10): in the COMPILED path, read the artifact's DECLARED bind
        # inputs (compiled.arguments()) and reject BEFORE any native call an install missing a REQUIRED
        # argument (instance / param / aux / solver). Inert (reads metadata); enforces only 'required',
        # so a valid install is unchanged.
        self._validate_install_arguments(
            compiled, instances, params, aux, validation_solvers, field_plans=field_plans)

        # (1) FIELD SOLVERS first: set_poisson must run before install_program (the C++ section-24
        # solver requirement reads poisson_solver()). The DECLARED named elliptic fields (from the
        # handle + per-instance models) widen the accepted solver-field set beyond the default Poisson
        # names (C1-System), while a typo is rejected against the declared set.
        declared_fields = self._declared_elliptic_fields(compiled, instances)
        if field_plans:
            for field, field_plan in field_plans.items():
                self._install_field_plan(field, field_plan, declared_fields)
        else:
            for field, solver_brick in solvers.items():
                self._install_solver(field, solver_brick, declared_fields)

        # (2) INSTANCES: add each named block (binds the Program block of that name, criterion 23),
        # lower its spatial brick and set its initial state. Every instance comes from InstallPlan and
        # carries its own detached CompiledModel; bind never consults compiled.model or a PDE builder.
        so_path = None
        if compiled is not None:
            so_path = getattr(compiled, "so_path", None)
            if so_path is None:
                raise TypeError(
                    "install: compiled handle has no .so_path (got %r); pass a compile_problem(...) "
                    "result, or compiled=None for a native sim (each instance carries its own native "
                    "model)." % type(compiled).__name__)
        resolved_models = {}
        lowered_instances = {}
        for name, spec in instances.items():
            if not isinstance(spec, Mapping):
                raise TypeError("install: instances[%r] must be a mapping (initial/spatial/time/model); "
                                "got %r" % (name, type(spec).__name__))
            model = spec.get("model")
            if model is None:
                raise ValueError(
                    "install: instance %r has no CompiledModel from InstallPlan; resolve and "
                    "compile the Case before binding" % name)
            model = self._resolve_instance_model(model)
            resolved_models[name] = model
            spatial = self._lower_spatial(spec.get("spatial"))
            time = spec.get("time")
            self._validate_riemann_capability(model, spatial)
            lowered_instances[name] = (spec, model, spatial, time)

        # Resolve all complete vectors before constructing the first native closure. There is no
        # post-install mutable parameter channel.
        bind_schema = getattr(compiled, "bind_schema", None) if compiled is not None else None
        if bind_schema is not None:
            per_block_params = self._route_block_params(resolved_models, bind_schema, params)
        elif params:
            raise ValueError(
                "install: parameter values require a compiled artifact carrying BindSchema"
            )
        else:
            per_block_params = {}

        for name, (spec, model, spatial, time) in lowered_instances.items():
            self.add_equation(
                name, model, spatial=spatial, time=time,
                _bind_params=per_block_params.get(name, []),
            )
            initial = spec.get("initial")
            if initial is not None:
                self.set_state(name, initial)

        # The final FieldOperator owns the solve name while its provider operators own only RHS
        # closures. Attach the resolved solve to the exact FieldSpace storage route after block
        # loaders have installed those closures; no legacy m.elliptic_field name is inferred.
        for field_plan in field_plans.values():
            self._register_field_plan_output(field_plan, resolved_models)

        # (3) AUX fields: B_z -> set_magnetic_field; named -> set_aux_field. Before install_program.
        for field_name, field in aux.items():
            self._install_aux(field_name, field)

        # (4) Boundary-kernel parameters are independent from model package parameters, which crossed
        # the package ABI during block installation above.
        for field_plan in field_plans.values():
            self._install_field_boundary_parameters(field_plan, params, compiled=compiled)

        # (5) COMPILED mode only: install the compiled time Program (binds blocks by name + runs the
        # section-24 .so requirement validation: aux / solver / block instance, verbatim messages). In
        # NATIVE mode (compiled=None) there is no Program -- the step-2 blocks drive the native loop.
        if so_path is not None:
            self.install_program(so_path)
            # (5a) HISTORY-PERSISTENCE POLICIES (ADC-626): the compiled Program records a per-ring
            # persistence policy (Dense / Interval / Revolve) on program._history_persistence. Attach the
            # name -> policy map to the System so the checkpoint stores only the policy-selected slots and
            # the restart replays the gaps. Absent -> Dense (the whole ring), byte-compatible with v1.
            program = getattr(compiled, "program", None)
            program = getattr(program, "program", program)
            persistence = getattr(program, "_history_persistence", None) if program else None
            set_persistence = getattr(self, "set_history_persistence", None)
            if persistence and set_persistence is not None:
                set_persistence(
                    {name: policy for name, (_depth, policy) in persistence.items()})
            # (5b) Program carriers were emitted with neutral values. Always install the complete
            # BindSchema projection after loading, including declaration defaults.
            self._install_program_params(compiled, bind_schema, params)
            component = getattr(compiled, "program", None)
            authored = getattr(component, "program", component)
            self._step_strategy = getattr(authored, "_step_strategy", None)
            self._step_transaction_plan = (
                authored.transaction_plan() if authored is not None else None)
            if authored is not None:
                self._temporal_restart_state.configure_program(
                    authored.temporal_manifest(),
                    time=self.time(), macro_step=self.macro_step())

        # Shared NumericalFlux routes need both endpoint MultiFabs and the installed Program, but
        # remain structural bind authorities. Materialize them here, after block construction and
        # before the lifecycle snapshot/freeze; no post-bind mutation seam is introduced.
        if install_plan is not None:
            from pops.runtime._runtime_authorities import finalize_runtime_authorities
            finalize_runtime_authorities(self, install_plan)

        # (8) FREEZE (ADC-592): the composition is fully lowered -- snapshot WHAT was bound, then
        # _finalize_bind marks the runtime 'bound' as the LAST act (nothing above ran frozen, so the
        # install sequence never trips its own guards).
        from pops.runtime._bound_snapshot import build_uniform_snapshot
        snapshot = build_uniform_snapshot(
            self, compiled, resolved_models, instances, validation_solvers,
            aux, params)
        self._finalize_bind(snapshot)  # _finalize_bind lives on _LifecycleMixin

    def explain_bind(self, compiled: Any) -> Any:
        """A printable :class:`pops.codegen.inspect_report.BindReport` of @p compiled vs this sim
        (Spec 5 sec.12.1, criterion #15). INERT: reads the artifact's DECLARED bind inputs
        (``compiled.arguments()``) and the blocks / named aux ALREADY wired on this System, then
        reuses the ADC-463 :func:`collect_missing_arguments` to compute, per group
        (instances / params / aux / solvers), which inputs are PROVIDED vs still REQUIRED. It binds
        nothing and mutates nothing -- the read-only counterpart of the install seam's early
        validation."""
        from pops.codegen.inspect_report import build_bind_report
        return build_bind_report(self, compiled)

    def _validate_install_arguments(self, compiled: Any, instances: Any, params: Any, aux: Any,
                                    solvers: Any, *, field_plans: Any = None) -> Any:
        """Early bind-input validation (Spec 5 sec.10): reject a COMPILED install missing a REQUIRED
        argument the artifact declares, BEFORE any native mutation. Thin wrapper around the shared
        private ``_bind_validation.validate_install_arguments`` implementation."""
        _validate_install_arguments_impl(
            self, compiled, instances, params, aux, solvers, field_plans=field_plans)

    # Host-testable alias of the pure core (mirrors _route_block_params: callable as
    # System._collect_missing_arguments without building a System).
    _collect_missing_arguments = staticmethod(_collect_missing_arguments_impl)

    def _lower_spatial(self, spatial: Any) -> Any:
        """Lower ``pops.numerics.FiniteVolume`` to the private ``Spatial`` engine adapter.

        An already-lowered ``Spatial`` value and ``None`` are accepted only within this private
        install pipeline.
        """
        if spatial is None:
            return Spatial()
        if isinstance(spatial, Spatial):
            return spatial
        runtime_spatial = getattr(spatial, "runtime_spatial", None)
        if callable(runtime_spatial):
            first, second = runtime_spatial(), runtime_spatial()
            if type(first) is not Spatial or type(second) is not Spatial:
                raise TypeError("runtime_spatial() must return an exact private Spatial value")
            if first != second:
                raise ValueError("runtime_spatial() must be deterministic")
            return first
        raise TypeError(
            "install: spatial must implement the pops.numerics finite-volume lowering protocol; "
            "got %r" % type(spatial).__name__)

    def _resolve_instance_model(self, model: Any) -> Any:
        """Accept only a runtime-ready model emitted into ``InstallPlan``.

        Compiling a PDE builder during bind made the runtime a second compiler and reintroduced live
        authoring authority. Public ``pops.compile`` now builds every block loader up front.
        """
        from pops.codegen.loader import CompiledModel
        if isinstance(model, CompiledModel):
            return model
        raise TypeError(
            "install: instance model must be a detached CompiledModel from InstallPlan, got %s; "
            "compile the Case before binding"
            % type(model).__name__
        )

    def _validate_riemann_capability(self, model: Any, spatial: Any) -> Any:
        """Section 24 capability check: reject the selected Riemann flux when a compiled model does
        not back it. Delegates to the SHARED gate pops.runtime.routes.check_riemann_capability
        (ADC-642) -- the SAME predicate System.add_equation / AmrSystem.add_equation call -- plus the
        HLL wave-speed cross-checks; one source, three call sites, zero divergence. A private native
        ``ModelSpec`` skips it because the C++ requires-gate validates at first use."""
        from pops.codegen.loader import CompiledModel  # late import (codegen <-> __init__ cycle)
        if not isinstance(model, CompiledModel):
            return
        from pops.runtime.routes import check_riemann_capability
        check_riemann_capability(spatial.flux, model, "install")
        flux = getattr(spatial, "flux", "rusanov")
        if flux == "hll":
            provider = getattr(spatial, "waves_provider", None)
            if provider is not None:
                from pops.numerics.riemann.waves import check_hll_waves
                check_hll_waves(provider, model, "install")
            if not getattr(model, "has_wave_speeds", True):
                raise ValueError(
                    "install: riemann 'hll' requires signed wave speeds: declare "
                    "m.wave_speeds(x=(smin, smax), y=(smin, smax)) (without pressure), or a primitive "
                    "'p' (m.primitive('p', ...)); otherwise use riemann='rusanov'.")

    # Field names the default native Poisson route already serves (the shared system elliptic solve).
    _DEFAULT_POISSON_FIELDS = ("phi", "poisson", "charge_density", "default")

    def _install_field_plan(self, field: Any, field_plan: Any,
                            declared_fields: Any = frozenset()) -> None:
        """Consume every resolve-time field-plan property at the native boundary."""
        from pops.codegen.field_install import ResolvedFieldInstallPlan
        if not isinstance(field_plan, ResolvedFieldInstallPlan):
            raise TypeError("install field_plans must contain ResolvedFieldInstallPlan values")
        if field_plan.name != field or field_plan.target != "system":
            raise ValueError("resolved field install plan identity/target mismatch")
        # Re-run canonical construction verification before touching the native engine.
        field_plan.__post_init__()
        options = field_plan.native_options
        solver_brick = field_plan.discretization.solver
        token = self._solver_token(solver_brick)
        if token != options["solver"]:
            raise ValueError("field plan solver token drifted after resolve")
        mg = self._solver_mg_options(solver_brick)
        from pops.solvers._numeric import native_float
        slot = options["provider_slot"]
        routes = options["provider_pack"]
        output_route = options["output_route"]
        from pops.identity import canonical_bytes
        mg_args = _mg_set_poisson_kwargs(mg)
        mg_args = {
            "rel_tol": 1.0e-8, "max_cycles": 50, "min_coarse": 2,
            "pre_smooth": 2, "post_smooth": 2, "bottom_sweeps": 50,
            "coarse_threshold": 0, **mg_args,
        }
        self._s.set_field_solver_plan(
            slot, options["provider_identity_text"],
            canonical_bytes(output_route["owner_identity"]).hex(),
            output_route["owner_block"],
            output_route["key"],
            [canonical_bytes(route["provider_identity"]).hex()
             for route in routes],
            [route["owner_block"] for route in routes],
            [route["key"] for route in routes],
            [route["coefficient"] for route in routes], token,
            native_float(mg.get("abs_tol", 0.0),
                         where="field plan absolute tolerance"),
            mg_args["rel_tol"], mg_args["max_cycles"], mg_args["min_coarse"],
            mg_args["pre_smooth"], mg_args["post_smooth"],
            mg_args["bottom_sweeps"], mg_args["coarse_threshold"])
        faces = options["boundary_faces"]
        if faces is not None:
            self._s.set_field_boundary_plan(
                slot,
                [face["type"] for face in faces],
                [face["alpha"] for face in faces],
                [face["beta"] for face in faces],
                [face["value"] for face in faces])
        dependencies = options["boundary_dependencies"]
        self._s.set_field_boundary_dependencies(
            slot,
            [row["owner_block"] for row in dependencies["states"]],
            [row["component"] for row in dependencies["states"]],
            [row["owner_block"] for row in dependencies["fields"]],
            [row["output_key"] for row in dependencies["fields"]],
            [row["component"] for row in dependencies["fields"]])
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
                "dynamic field boundaries require a compiled artifact that owns their generated "
                "device launchers")
        handles = field_plan.boundary_parameter_handles()
        missing = [handle.qualified_id for handle in handles if handle not in params]
        if missing:
            raise ValueError(
                "dynamic field boundary parameter pack is incomplete: %s" % ", ".join(missing))
        from pops.solvers._numeric import native_float
        values = [native_float(params[handle], where="dynamic field boundary parameter %s" %
                               handle.qualified_id) for handle in handles]
        self._s.set_field_boundary_parameters(
            field_plan.native_options["provider_slot"], values)

    def _register_field_plan_output(self, field_plan: Any, models: Any) -> None:
        route = field_plan.native_options["output_route"]
        block = route["owner_block"]
        model = models.get(block)
        if model is None:
            raise ValueError("field output route names unknown block %r" % block)
        from pops.physics.aux import aux_component_index

        declared = tuple(getattr(model, "aux_extra_names", ()) or ())
        components = tuple(route["components"])
        try:
            indices = [aux_component_index(component, declared) for component in components]
        except ValueError as error:
            raise ValueError(
                "field output route %r is absent from block %r native aux layout: %s"
                % (field_plan.name, block, ", ".join(components))
            ) from error
        indices.extend([-1] * (3 - len(indices)))
        self._s.register_elliptic_field(
            block, route["key"], indices[0], indices[1], indices[2])

    def _install_solver(self, field: Any, solver_brick: Any,
                        declared_fields: Any = frozenset()) -> Any:
        """Lower a field-solver selection to set_poisson (C1-System).

        The default Poisson field and any NAMED elliptic field a block's model DECLARES (via
        m.elliptic_field, collected into @p declared_fields) are accepted: the named field's RHS is
        wired by the native loader (register_elliptic_field + set_block_elliptic_field), and its solve
        reuses the shared system elliptic solver, so the solver selection routes through set_poisson
        for both. A field name that is NEITHER the default Poisson field NOR a declared named field is a
        TYPO -- rejected LOUD, naming the declared set (never a silent drop)."""
        if field not in self._DEFAULT_POISSON_FIELDS and field not in declared_fields:
            declared = ", ".join(sorted(declared_fields)) or "(none declared)"
            raise ValueError(
                "install: solver selection names field %r, which is neither the default Poisson "
                "field (%s) nor a named elliptic field any installed model declares (declared: %s). "
                "Declare it with m.elliptic_field(%r, rhs=...), or fix the field name."
                % (field, ", ".join(self._DEFAULT_POISSON_FIELDS), declared, field))
        token = self._solver_token(solver_brick)
        opts = self._solver_option_dict(solver_brick)
        mg = self._solver_mg_options(solver_brick)  # ADC-613: resolved V-cycle scalars (or {})
        from pops.solvers._numeric import native_float
        # Solver plans are already resolved and carry native route tokens. Keep that representation
        # behind the private seam; public set_poisson accepts typed bc/wall descriptors only.
        self._set_poisson_native(
            rhs=opts.get("rhs", "charge_density"), solver=token,
            bc=opts.get("bc", "auto"), wall=opts.get("wall", "none"),
            wall_radius=float(opts.get("wall_radius", 0.0)),
            epsilon=float(opts.get("epsilon", 1.0)),
            abs_tol=native_float(
                mg.get("abs_tol", opts.get("abs_tol", 0.0)),
                where="GeometricMG absolute tolerance"),
            **_mg_set_poisson_kwargs(mg))

    @staticmethod
    def _solver_option_dict(solver_brick: Any) -> Any:
        """The plain-dict option bag of a solver selection, or ``{}``.

        A lib BrickDescriptor carries scheme options as a ``.options`` DICT ATTRIBUTE; a typed
        pops.solvers descriptor exposes ``options`` as a METHOD (a bound method is not a mapping),
        so only a genuine dict is read here -- never the method object (the pre-613 code read the
        bound method by mistake, so no typed knob ever flowed)."""
        opts = getattr(solver_brick, "options", None)
        return dict(opts) if isinstance(opts, Mapping) else {}

    @staticmethod
    def _solver_mg_options(solver_brick: Any) -> Any:
        """The RESOLVED native GeometricMG V-cycle scalars of a typed descriptor (ADC-613), or ``{}``.

        A typed pops.solvers.elliptic.GeometricMG exposes ``mg_options()`` (rel_tol / max_cycles /
        min_coarse / pre_smooth / post_smooth / bottom_sweeps, tolerance descriptor already mapped).
        A string token or a lib descriptor has none -> ``{}`` -> set_poisson keeps its native
        defaults, bit-identical."""
        mg_fn = getattr(solver_brick, "mg_options", None)
        if callable(mg_fn):
            resolved = mg_fn()
            if isinstance(resolved, Mapping):
                return dict(resolved)
        return {}

    @staticmethod
    def _declared_elliptic_fields(compiled: Any, instances: Any) -> Any:
        """Collect named elliptic fields exclusively from InstallPlan CompiledModel metadata."""
        del compiled  # a whole-program handle is never a field-declaration authority
        from pops.codegen.loader import CompiledModel

        names = set()
        for block_name, spec in (instances or {}).items():
            if not isinstance(spec, Mapping):
                raise TypeError("install: instances[%r] must be a mapping" % block_name)
            model = spec.get("model")
            if not isinstance(model, CompiledModel):
                raise TypeError(
                    "install: instances[%r] must carry a detached CompiledModel from InstallPlan"
                    % block_name
                )
            declared = getattr(model, "elliptic_field_names", None)
            if declared is None:
                raise ValueError(
                    "install: CompiledModel for block %r lacks elliptic_field_names metadata; "
                    "re-run pops.resolve(case) and pops.compile(plan)" % block_name
                )
            names.update(declared)
        return names

    @staticmethod
    def _solver_token(solver_brick: Any) -> Any:
        """Resolve a field-solver selection to its set_poisson token. Accepts a string, or a
        descriptor carrying ``scheme`` (pops.solvers.GeometricMG -> 'geometric_mg')."""
        if isinstance(solver_brick, str):
            return solver_brick
        token = getattr(solver_brick, "scheme", None) or getattr(solver_brick, "name", None)
        if token is None:
            raise TypeError("install: solver must be a token string or an pops.solvers.<Solver>(...) "
                            "descriptor; got %r" % type(solver_brick).__name__)
        return token

    def _install_aux(self, field_name: Any, field: Any) -> Any:
        """Lower an aux entry: 'B_z' -> set_magnetic_field; 'T_e' rejected (derived); any other name
        -> set_aux_field on the block that declares it."""
        if field_name == "B_z":
            self.set_magnetic_field(field)
            return
        if field_name == "T_e":
            raise ValueError(
                "install: aux 'T_e' is DERIVED from a fluid block via "
                "set_electron_temperature_from(block), not set as a static aux field.")
        block = self._block_declaring_aux(field_name)
        if block is None:
            raise ValueError(
                "install: aux field %r is not declared by any installed instance; add the instance "
                "with a model declaring m.aux_field(%r)." % (field_name, field_name))
        self.set_aux_field(block, field_name, field)

    def _block_declaring_aux(self, field_name: Any) -> Any:
        """The block whose named-aux table declares @p field_name, or None."""
        for block, table in self._aux_field_index.items():
            if field_name in table:
                return block
        return None

    # Host-testable pure core (P7-b block-param routing, ADC-514 shares it with the AMR path): callable
    # as System._route_block_params without building a System. Extracted to _install_param_routing so the
    # Uniform and AMR install seams both delegate to ONE routing implementation.
    _route_block_params = staticmethod(route_block_params)

    # Host-testable pure core (ADC-510 program-param routing, mirror of _route_block_params): callable
    # as System._route_program_params without building a System.
    _route_program_params = staticmethod(route_program_params)

    def _install_program_params(self, compiled: Any, schema: Any, params: Any) -> None:
        """Install complete owner-qualified Program vectors from BindSchema."""
        per_block = self._route_program_params(compiled, schema, params)
        for blk, values in per_block.items():
            self._s.set_program_params(blk, values)
