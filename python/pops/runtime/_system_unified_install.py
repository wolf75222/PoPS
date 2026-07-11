"""System unified-install mixin (Spec-4 PR-F): the INTERNAL ``_install_compiled`` seam.

``_install_compiled`` (the low-level seam that lowers to add_equation / set_poisson /
set_magnetic_field / set_aux_field / set_block_params / install_program) plus its private
lowering helpers. It is NOT the public entry point (Spec 5 sec.11): authors call
``pops.bind(compiled, state=, params=, aux=, solvers=)``, which dispatches System / AmrSystem and
calls this seam. Mixed into ``System`` via inheritance; methods operate on ``self`` (calling the
other mixins' methods) and ``self._s``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops._bootstrap import ModelSpec
from pops.runtime._install_param_routing import route_block_params, route_program_params
from pops.runtime.bricks import Spatial

# The two sec.10 install-argument validators moved to ``_bind_validation`` (ADC-550), their natural
# home beside the other bind-time gates, and are re-imported so the historical
# ``from pops.runtime._system_unified_install import validate_install_arguments`` path (and the AMR
# install seam / tests that use it) is unchanged.
from pops.runtime._bind_validation import (  # noqa: F401
    collect_missing_arguments,
    validate_install_arguments,
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
                          solvers=None, cadence=None, outputs=None, diagnostics=None):
        """INTERNAL low-level install seam (Spec 5 sec.11): wire a compiled handle + per-instance
        state/spatial + params + aux + field solvers in ONE call, then install the compiled time
        Program. NOT the public entry point: author the run with ``pops.bind(compiled, state=,
        params=, aux=, solvers=)``, which dispatches System / AmrSystem and calls this seam. This
        method is undocumented on the public surface (it carries no ``install`` alias) and may change.

        It LOWERS to the existing lower-layer calls
        (add_equation / set_poisson / set_magnetic_field / set_aux_field / set_block_params /
        install_program) -- there is NO parallel runtime (Spec section 3). The lower-layer calls stay
        available and unchanged; this seam just sequences them in the right order so the
        install-time validation (section 24) sees a fully-configured simulation.

        install() is the ONE entry for BOTH runtime modes (Spec 4 amendment): a COMPILED-program sim
        (pass the compile_problem(...) handle as ``compiled``) and a NATIVE sim (``compiled=None``;
        each instance carries its own native model + native time policy, no compiled .so).

        @param compiled the compiled problem handle (compile_problem(...) result) carrying ``so_path``,
            installed via install_program after every instance/solver/aux is wired. Pass ``None`` for a
            NATIVE sim: no Program is installed; each instance must supply its own native ``"model"``
            and (optionally) ``"time"`` policy, and the native per-block advance loop drives stepping.
        @param instances dict {name: {"initial": array, "spatial": <brick>, "model": <pops.Model>,
            "time": <pops.Explicit/IMEX>}}. The block is bound by the dict KEY @p name (Spec criterion
            23), not a "state" field. Each entry adds the named block (add_equation), sets its
            "initial" state (if given) and lowers the "spatial" brick to the add_equation spatial args.
            The block model is the per-instance ``"model"`` if given, else ``compiled`` (single-
            instance case). ``spatial`` is an pops.FiniteVolume(...) / pops.Spatial(...) OR an
            pops.numerics.spatial.FiniteVolume(...) descriptor.
        @param params complete mapping from canonical, block-qualified ParamHandle values to their
            resolved runtime values. BindSchema has already applied defaults and derived values.
        @param aux dict {field_name: array}: "B_z" -> set_magnetic_field, "T_e" -> rejected (it is
            DERIVED, use set_electron_temperature_from), any other -> set_aux_field on the instance
            declaring it. Set BEFORE install_program so the section-24 aux requirement check sees it.
        @param solvers dict {field: <pops.solvers.GeometricMG(...)/pops.GeometricMG(...)>}: lowered to
            set_poisson(solver=...). The default Poisson field ("phi"/"charge_density"/"poisson") and
            any NAMED elliptic field a block's model DECLARES (m.elliptic_field) are accepted and route
            through the shared system elliptic solver; a field name no model declares raises (typo).
        @param cadence optional pops.CompiledTime(substeps=, stride=): the compiled Program's macro-step
            cadence, applied with set_program_cadence AFTER install_program. A compiled Program is ONE
            whole-system closure, so its cadence is GLOBAL (one program-level value). A numeric
            cadence.cfl is applied at runtime by sim.run(cfl=) (the cadence pins it on the System so a
            bare sim.run(t_end) uses it), not by the install.
        @param outputs optional list of pops.output.OutputPolicy / CheckpointPolicy (C4 / ADC-509)
            stored so sim.run(output_dir=) fires each at its cadence via the existing write/checkpoint.
        @throws the verbatim Spec section-24 errors at install (missing aux / solver / block instance /
            Riemann capability). A disallowed schedule is rejected earlier, at Program compile.
        """
        # RUNTIME FREEZE (ADC-592): a second install on an already-bound engine is refused explicitly.
        from pops.runtime._lifecycle import guard_assembling
        guard_assembling(self, "_install_compiled")
        instances = instances or {}
        params = params or {}
        aux = aux or {}
        solvers = solvers or {}

        # (0) EARLY VALIDATION (Spec 5 sec.10): in the COMPILED path, read the artifact's DECLARED bind
        # inputs (compiled.arguments()) and reject BEFORE any native call an install missing a REQUIRED
        # argument (instance / param / aux / solver). Inert (reads metadata); enforces only 'required',
        # so a valid install is unchanged.
        self._validate_install_arguments(compiled, instances, params, aux, solvers)

        # (1) FIELD SOLVERS first: set_poisson must run before install_program (the C++ section-24
        # solver requirement reads poisson_solver()). The DECLARED named elliptic fields (from the
        # handle + per-instance models) widen the accepted solver-field set beyond the default Poisson
        # names (C1-System), while a typo is rejected against the declared set.
        declared_fields = self._declared_elliptic_fields(compiled, instances)
        for field, solver_brick in solvers.items():
            self._install_solver(field, solver_brick, declared_fields)

        # (2) INSTANCES: add each named block (binds the Program block of that name, criterion 23),
        # lower its spatial brick and set its initial state. The block model is the per-instance "model"
        # if given, else the PHYSICAL model on the handle (not the handle, which is the step-5 .so).
        # COMPILED: a compile_problem(...) handle with a .so_path time Program. NATIVE: compiled is None
        # (each instance carries its own model + time policy, step 5 skipped). Validate the handle first.
        so_path = None
        compiled_model = None
        if compiled is not None:
            so_path = getattr(compiled, "so_path", None)
            if so_path is None:
                raise TypeError(
                    "install: compiled handle has no .so_path (got %r); pass a compile_problem(...) "
                    "result, or compiled=None for a native sim (each instance carries its own native "
                    "model)." % type(compiled).__name__)
            compiled_model = getattr(compiled, "model", None)
        resolved_models = {}  # instance name -> RESOLVED (CompiledModel), reused by the params step
        for name, spec in instances.items():
            if not isinstance(spec, dict):
                raise TypeError("install: instances[%r] must be a dict (initial/spatial/time/model); "
                                "got %r" % (name, type(spec).__name__))
            model = spec.get("model", compiled_model)
            if model is None:
                raise ValueError(
                    "install: instance %r has no block model -- supply instances[%r]['model'] "
                    "(an pops.Model(...) / CompiledModel), or pass a compiled handle that carries one "
                    "(compile_problem(model=...))." % (name, name))
            model = self._resolve_instance_model(model)
            resolved_models[name] = model
            spatial = self._lower_spatial(spec.get("spatial"))
            time = spec.get("time")
            # Capability check (section 24): the selected Riemann flux must be backed by the model.
            self._validate_riemann_capability(model, spatial)
            self.add_equation(name, model, spatial=spatial, time=time)
            initial = spec.get("initial")
            if initial is not None:
                self.set_state(name, initial)

        # (3) AUX fields: B_z -> set_magnetic_field; named -> set_aux_field. Before install_program.
        for field_name, field in aux.items():
            self._install_aux(field_name, field)

        # (4) PARAMS: BindSchema already resolved every supplied/default/derived value to a canonical
        # ParamHandle. Project the complete block vectors; no name broadcast and no default fallback.
        bind_schema = getattr(compiled, "bind_schema", None) if compiled is not None else None
        if bind_schema is not None:
            self._install_params(resolved_models, bind_schema, params)
        elif params:
            raise ValueError(
                "install: parameter values require a compiled artifact carrying BindSchema"
            )

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
            persistence = getattr(program, "_history_persistence", None) if program else None
            set_persistence = getattr(self, "set_history_persistence", None)
            if persistence and set_persistence is not None:
                set_persistence(
                    {name: policy for name, (_depth, policy) in persistence.items()})
            # (5b) Program carriers were emitted with neutral values. Always install the complete
            # BindSchema projection after loading, including declaration defaults.
            self._install_program_params(compiled, bind_schema, params)

        # (6) PROGRAM CADENCE (substeps / stride): a compiled Program is ONE whole-system closure, so
        # its macro-step cadence is GLOBAL. Apply it AFTER install_program (the cadence wraps the
        # installed closure); a native sim sets substeps / stride on its time policy instead.
        if cadence is not None:
            if so_path is None:
                raise ValueError(
                    "install(cadence=): a cadence applies to a compiled time Program; a native sim "
                    "(compiled=None) has no Program -- set substeps / stride on the native time policy "
                    "(pops.Explicit(substeps=, stride=)) instead.")
            self._install_cadence(cadence)

        if outputs:  # (7) OUTPUT / CHECKPOINT policies (C4): run() fires each at its cadence
            self._output_policies = list(outputs)
        if diagnostics:  # (7b) DIAGNOSTIC measures (ADC-542): run() fires each at its cadence
            self._diagnostic_measures = list(diagnostics)

        # (8) FREEZE (ADC-592): the composition is fully lowered -- snapshot WHAT was bound, then
        # _finalize_bind marks the runtime 'bound' as the LAST act (nothing above ran frozen, so the
        # install sequence never trips its own guards).
        from pops.runtime._bound_snapshot import build_uniform_snapshot
        snapshot = build_uniform_snapshot(self, compiled, resolved_models, instances, solvers,
                                          cadence, aux, params)
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
                                    solvers: Any) -> Any:
        """Early bind-input validation (Spec 5 sec.10): reject a COMPILED install missing a REQUIRED
        argument the artifact declares, BEFORE any native mutation. Thin wrapper around the shared
        module-level :func:`validate_install_arguments` (reused by ``AmrSystem._install_compiled``
        for parity)."""
        validate_install_arguments(self, compiled, instances, params, aux, solvers)

    # Host-testable alias of the pure core (mirrors _route_block_params: callable as
    # System._collect_missing_arguments without building a System).
    _collect_missing_arguments = staticmethod(collect_missing_arguments)

    def _install_cadence(self, cadence: Any) -> Any:
        """Apply a CompiledTime macro-step cadence to the installed program (set_program_cadence).

        set_program_cadence is a SYSTEM-level orchestration around the opaque program closure:
        substeps=n re-runs the whole program over eff_dt/n; stride=M runs it once per M macro-steps. A
        NUMERIC cadence.cfl is NOT consumed here; it is stored on the System so a bare sim.run(t_end)
        defaults sim.run(cfl=) to it. A self-computed cfl sub-program (cfl='program') is rejected
        upstream by CompiledTime, so it never reaches here."""
        from pops.time.program import CompiledTime
        if not isinstance(cadence, CompiledTime):
            raise TypeError("install(cadence=): expected a pops.CompiledTime(substeps=, stride=), "
                            "got %r" % type(cadence).__name__)
        if cadence.cfl != "default":
            # Pin the numeric cfl so run() with no explicit cfl= uses it (not a silent no-op).
            from pops.solvers._numeric import native_float
            self._program_cadence_cfl = native_float(
                cadence.cfl, where="CompiledTime cfl")
        self.set_program_cadence(cadence.substeps, cadence.stride)

    def _lower_spatial(self, spatial: Any) -> Any:
        """Lower a spatial selection to an pops.Spatial consumed by add_equation. Accepts an
        pops.Spatial / pops.FiniteVolume (returned as-is), an pops.numerics.spatial.FiniteVolume(...)
        BrickDescriptor (read its riemann/reconstruction/positivity_floor options), or None (default
        Spatial)."""
        if spatial is None:
            return Spatial()
        if isinstance(spatial, Spatial):
            return spatial
        # A lib BrickDescriptor carries the scheme options as STRING tokens in .options. Lower them
        # to the canonical Spatial tokens directly (Spatial._from_tokens bypasses the public typed-
        # descriptor guard, which the runtime FiniteVolume now enforces -- Spec 5 sec.7).
        opts = getattr(spatial, "options", None)
        if isinstance(opts, dict):
            limiter = opts.get("reconstruction", opts.get("limiter", "minmod"))
            riemann = opts.get("riemann", opts.get("flux", "rusanov"))
            variables = opts.get("variables", opts.get("recon", "conservative"))
            return Spatial._from_tokens(
                limiter, riemann, variables,
                positivity_floor=opts.get("positivity_floor"),
                wave_speed_cache=bool(opts.get("wave_speed_cache", False)))
        raise TypeError("install: spatial must be an pops.FiniteVolume / pops.Spatial or an "
                        "pops.numerics.spatial.FiniteVolume(...) descriptor; got %r"
                        % type(spatial).__name__)

    def _resolve_instance_model(self, model: Any) -> Any:
        """Resolve an instance's block model to something add_equation accepts. A ModelSpec
        (pops.Model(...)) or a dsl.CompiledModel passes through unchanged. A dsl.Model (the PDE
        builder, e.g. carried by compile_problem(model=...)) is compiled to a CompiledModel so the
        block is added on the real System context.

        Backend choice (P7-b): a dsl.Model declaring RUNTIME params is compiled via AOT, because the
        production/native backend FREEZES runtime params at their declaration value (so
        install(params=...) -> set_block_params would raise 'block ... has no runtime parameter'); a
        const-only model keeps the native production path (no .so dlopen). The AOT block gates its OWN
        time integrator to SSPRK2 + backward-Euler, harmless here (the compiled time Program drives the
        step). A runtime-param instance must use an AOT-compatible ``time`` (Explicit()==SSPRK2 fine)."""
        # Late imports (the codegen/physics modules import this package: avoid the cycle).
        from pops.codegen.loader import CompiledModel
        from pops.physics.facade import Model
        if isinstance(model, (ModelSpec, CompiledModel)):
            return model
        if isinstance(model, Model):
            has_runtime = any(getattr(p, "kind", "const") == "runtime"
                              for p in model.params.values())
            return model.compile(backend="aot" if has_runtime else "production")
        return model  # unknown -> let add_equation raise its own clear error

    def _validate_riemann_capability(self, model: Any, spatial: Any) -> Any:
        """Section 24 capability check: reject the selected Riemann flux when a compiled model does
        not back it. Delegates to the SHARED gate pops.runtime.routes.check_riemann_capability
        (ADC-642) -- the SAME predicate System.add_equation / AmrSystem.add_equation call -- plus the
        HLL wave-speed cross-checks; one source, three call sites, zero divergence. A composed native
        pops.Model(...) skips (the C++ requires-gate validates at first use)."""
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
        self.set_poisson(rhs=opts.get("rhs", "charge_density"), solver=token,
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
        return opts if isinstance(opts, dict) else {}

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
            if isinstance(resolved, dict):
                return resolved
        return {}

    @staticmethod
    def _declared_elliptic_fields(compiled: Any, instances: Any) -> Any:
        """Collect the NAMED elliptic fields declared by the compiled handle's model and the
        per-instance models (C1-System). Reads each model's declared names WITHOUT compiling: a
        CompiledModel exposes ``elliptic_field_names``; a raw physics/dsl Model exposes the
        ``_elliptic_fields`` mapping. Returns a set (empty when no model declares a named field)."""
        names = set()

        def _names_of(model):
            if model is None:
                return ()
            explicit = getattr(model, "elliptic_field_names", None)
            if explicit is not None:
                return list(explicit)
            raw = getattr(model, "_elliptic_fields", None)
            return list(raw) if raw else ()

        names.update(_names_of(getattr(compiled, "model", None)))
        for spec in (instances or {}).values():
            if isinstance(spec, dict):
                names.update(_names_of(spec.get("model")))
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

    def _install_params(self, resolved_models: Any, schema: Any, params: Any) -> None:
        """Install complete owner-qualified block vectors from BindSchema."""
        per_block = self._route_block_params(resolved_models, schema, params)
        for name, values in per_block.items():
            self._s.set_block_params(name, values)

    # Host-testable pure core (ADC-510 program-param routing, mirror of _route_block_params): callable
    # as System._route_program_params without building a System.
    _route_program_params = staticmethod(route_program_params)

    def _install_program_params(self, compiled: Any, schema: Any, params: Any) -> None:
        """Install complete owner-qualified Program vectors from BindSchema."""
        per_block = self._route_program_params(compiled, schema, params)
        for blk, values in per_block.items():
            self._s.set_program_params(blk, values)
