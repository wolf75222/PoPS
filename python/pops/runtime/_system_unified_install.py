"""System unified-install mixin: public ``install`` over one compiled-problem seam.

``install`` is the explicit-runtime entry point used by examples that manually construct
``System``. All public runtime wiring should enter through ``sim.install(compiled, ...)``.
The lowering seam routes the combined ``CompiledProblem`` into the native runtime while keeping
block instantiation, field solvers, aux inputs, params and outputs validated in one place.
"""

from pops._bootstrap import ModelSpec
from pops.runtime._install_param_routing import route_program_params
from pops.runtime.bricks import Spatial


def collect_missing_arguments(args, provided_blocks, provided_params, provided_aux,
                              provided_solvers):
    """Return actionable missing required bind inputs; pure metadata check, no engine calls."""
    missing = []
    for name, spec in sorted(getattr(args, "instances", {}).items()):
        if spec.get("required") and name not in provided_blocks:
            missing.append("instance %r (a state block the program advances); supply its initial "
                           "state via sim.install(instances={%r: {'initial': <array>, ...}})"
                           % (name, name))
    for name, spec in sorted(getattr(args, "params", {}).items()):
        if spec.get("required") and name not in provided_params:
            missing.append("runtime param %r; pass sim.install(params={%r: <value>})"
                           % (name, name))
    for name, spec in sorted(getattr(args, "aux", {}).items()):
        if spec.get("required") and name not in provided_aux:
            missing.append("aux field %r; pass sim.install(aux={%r: <array>})" % (name, name))
    for name, spec in sorted(getattr(args, "solvers", {}).items()):
        if spec.get("required") and name not in provided_solvers:
            missing.append("solver for field %r; pass sim.install(solvers={%r: <Solver>})"
                           % (name, name))
    return missing


def validate_install_arguments(sim, compiled, instances, params, aux, solvers):
    """Reject missing required compiled-bind inputs before mutating System/AmrSystem."""
    if compiled is None or not hasattr(compiled, "arguments"):
        return
    try:
        args = compiled.arguments()
    except Exception:  # noqa: BLE001 -- introspection must never break a valid install
        return
    provided_blocks = set(instances)
    try:
        provided_blocks |= set(sim.block_names())
    except Exception:  # noqa: BLE001 -- block_names is a convenience; absence is not a failure
        pass
    # Named aux already declared on the sim (B_z has no queryable trace, so it must come via aux=).
    provided_named_aux = set()
    for table in getattr(sim, "_aux_field_index", {}).values():
        provided_named_aux |= set(table)
    missing = collect_missing_arguments(
        args, provided_blocks, set(params), set(aux) | provided_named_aux, set(solvers))
    if missing:
        raise ValueError("install: the compiled artifact is missing required argument(s):\n  "
                         + "\n  ".join(missing))


class _SystemUnifiedInstall:
    """Unified install lowering for System.

    ``install(...)`` is the documented runtime entry point for scripts that explicitly build a
    ``System``.
    """

    def _install_problem_so(self, so_path):
        """Install a combined compiled-problem shared object through the native runtime.

        This wrapper is the private compiled-problem loader used by ``sim.install``. The generated
        shared object may still export the historical C ABI symbol it was built around, but the
        Python/native binding seam is a compiled-problem attach, not a public Program route.
        """
        return self._s.install_problem(so_path)

    def install(self, compiled=None, *, instances=None, params=None, aux=None,
                solvers=None, cadence=None, outputs=None):
        """Public Spec-4 install entry point.

        Wires a ``CompiledProblem`` plus its block instances, runtime params, aux fields and field
        solvers in one validated call, then installs the compiled problem artifact. Pass
        ``compiled=None`` for the native per-block route. This is a thin public wrapper over the
        single lowering seam; it exists so examples and user code do not call private
        ``_install_*`` helpers.
        """
        return self._install_compiled(
            compiled,
            instances=instances,
            params=params,
            aux=aux,
            solvers=solvers,
            cadence=cadence,
            outputs=outputs,
        )

    def _install_compiled(self, compiled=None, *, instances=None, params=None, aux=None,
                          solvers=None, cadence=None, outputs=None):
        """Shared install seam for compiled and native System routes.

        Wires instances, runtime params, aux fields, field solvers, output policies and the optional
        compiled problem artifact in one validated order. Public callers enter through
        ``sim.install``; this method keeps the low-level sequencing in one place.
        """
        instances = instances or {}
        params = params or {}
        aux = aux or {}
        solvers = solvers or {}

        # (0) EARLY VALIDATION (Spec 5 sec.10): in the COMPILED path, read the artifact's DECLARED
        # bind inputs (compiled.arguments()) and reject BEFORE any native call an install missing a
        # REQUIRED argument (instance / runtime param / aux / solver). It enforces only 'required';
        # an input the artifact marks optional (a const param, an unrequired solver) is never demanded.
        # Inert: it reads metadata and compares dicts (no compile / bind / allocation). It never
        # rejects an install that supplies everything required, so a valid install is unchanged.
        self._validate_install_arguments(compiled, instances, params, aux, solvers)

        # (1) FIELD SOLVERS first: the native loader validates field-solver requirements when the
        # compiled problem is attached, so configure solvers before that attach. Declared named
        # elliptic fields (from the compiled handle + per-instance models) widen the accepted
        # solver-field set beyond the default Poisson names; typos are rejected against that set.
        declared_fields = self._declared_elliptic_fields(compiled, instances)
        for field, solver_brick in solvers.items():
            self._install_solver(field, solver_brick, declared_fields)

        # (2) INSTANCES: add each named block (binds the Program block of that name, criterion 23),
        # lower its spatial brick and set its initial state. The block model is the per-instance
        # "model" if given, else the physical Module carried by the CompiledProblem. COMPILED mode:
        # `compiled` is a compile_problem(...) handle carrying a combined artifact attached in step 5.
        # NATIVE mode: `compiled is None`; each instance carries its own native model + time policy
        # (runtime Explicit / Strang), step 5 is skipped, and the native per-block loop drives
        # stepping. Validate the handle up front, BEFORE any System mutation (no half-configured
        # System).
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
                    "(a pops.model.Module / CompiledModel), or pass a compiled handle that carries one "
                    "(compile_problem(model=...))." % (name, name))
            model = self._resolve_instance_model(model)
            resolved_models[name] = model
            spatial = self._lower_spatial(spec.get("spatial"))
            time = spec.get("time")
            # Capability check (section 24): the selected Riemann flux must be backed by the model.
            self._validate_riemann_capability(model, spatial)
            self._add_equation(name, model, spatial=spatial, time=time)
            initial = spec.get("initial")
            if initial is not None:
                self._set_state(name, initial)

        # (3) AUX fields: B_z -> set_magnetic_field; named -> set_aux_field. Before artifact attach.
        for field_name, field in aux.items():
            self._install_aux(field_name, field)

        # (4) PARAMS (AOT-block path, P7-b): route each runtime param to the instance whose RESOLVED
        # CompiledModel declares it (set_block_params), so a block's compiled residual (ctx.rhs_into) sees
        # the runtime value. Native mode rejects an unknown name; the compiled-program path defers (an
        # unconsumed name may be a Program-lowered param routed in 5b) and subtracts the consumed ones.
        program_params_left = dict(params)
        if params:
            consumed = self._install_params(resolved_models, params,
                                            reject_unknown=(compiled is None))
            for name in consumed:
                program_params_left.pop(name, None)

        # (5) COMPILED mode only: attach the combined compiled problem (binds blocks by name + runs
        # native requirement validation: aux / solver / block instance, verbatim messages). In NATIVE
        # mode (compiled=None) there is no compiled artifact; the step-2 blocks drive the native loop.
        if so_path is not None:
            self._install_problem_so(so_path)
            # (5b) COMPILED-PROBLEM RUNTIME PARAMS (ADC-510, Spec 5 C5): route the REMAINING params
            # (the ones no AOT instance consumed in 4) to the per-block runtime parameter table seeded
            # by the native attach. A runtime param read by generated kernels reaches them via the
            # System-owned per-block RuntimeParams (no recompile). A name declared by neither an AOT
            # instance nor a generated kernel raises here (no silent drop).
            if program_params_left:
                self._install_problem_params(compiled, program_params_left)

        # (6) COMPILED-PROBLEM CADENCE (substeps / stride): the artifact is one whole-system closure,
        # so its macro-step cadence is GLOBAL (not per-block). Apply it AFTER artifact attach. A
        # native sim sets substeps / stride on its native time policy instead.
        if cadence is not None:
            if so_path is None:
                raise ValueError(
                    "install(cadence=): a cadence applies to a compiled problem artifact; a native sim "
                    "(compiled=None) has no Program -- set substeps / stride on the native time policy "
                    "instead.")
            self._install_cadence(cadence)

        if outputs:  # (7) OUTPUT / CHECKPOINT policies (C4): run() fires each at its cadence
            self._output_policies = list(outputs)

    def explain_bind(self, compiled):
        """A printable :class:`pops.codegen.inspect_report.BindReport` of @p compiled vs this sim
        (Spec 5 sec.12.1, criterion #15). INERT: reads the artifact's DECLARED bind inputs
        (``compiled.arguments()``) and the blocks / named aux ALREADY wired on this System, then
        reuses the ADC-463 :func:`collect_missing_arguments` to compute, per group
        (instances / params / aux / solvers), which inputs are PROVIDED vs still REQUIRED. It binds
        nothing and mutates nothing -- the read-only counterpart of the install seam's early
        validation."""
        from pops.codegen.inspect_report import build_bind_report
        return build_bind_report(self, compiled)

    def _validate_install_arguments(self, compiled, instances, params, aux, solvers):
        """Early bind-input validation (Spec 5 sec.10): reject a COMPILED install missing a REQUIRED
        argument the artifact declares, BEFORE any native mutation. Thin wrapper around the shared
        module-level :func:`validate_install_arguments` (reused by ``AmrSystem._install_compiled``
        for parity)."""
        validate_install_arguments(self, compiled, instances, params, aux, solvers)

    # Host-testable alias of the pure core (mirrors _route_block_params: callable as
    # System._collect_missing_arguments without building a System).
    _collect_missing_arguments = staticmethod(collect_missing_arguments)

    def _set_problem_cadence(self, substeps, stride):
        """Private native cadence attach used by the compiled-problem install seam."""
        return self._s.set_program_cadence(substeps, stride)

    def _install_cadence(self, cadence):
        """Apply a compiled-problem macro-step cadence to the installed artifact.

        The native cadence is a SYSTEM-level orchestration around the installed generated closure:
        substeps=n re-runs the whole artifact over eff_dt/n; stride=M runs it once per M
        macro-steps. A NUMERIC cadence.cfl is NOT consumed here (set_program_cadence carries only
        substeps / stride); instead it is stored on the System so a bare sim.run(t_end) defaults
        sim.run(cfl=) to it (System::step_cfl routes the resulting per-block CFL dt through the
        installed artifact). ``cfl='program'`` pins the run wrapper to call ``step_cfl(1.0)``; the
        installed C++ dt_bound hook then tightens the step inside ``System::step_cfl``."""
        from pops.runtime._compiled_cadence import CompiledProgramCadence
        if not isinstance(cadence, CompiledProgramCadence):
            raise TypeError("install(cadence=): expected an internal CompiledProgramCadence "
                            "(substeps=, stride=), got %r" % type(cadence).__name__)
        if isinstance(cadence.cfl, (int, float)):
            # Pin the numeric cfl so run() with no explicit cfl= uses it (not a silent no-op).
            self._program_cadence_cfl = float(cadence.cfl)
        elif cadence.cfl == "program":
            self._program_cadence_cfl = "program"
        self._set_problem_cadence(cadence.substeps, cadence.stride)

    def _lower_spatial(self, spatial):
        """Lower a spatial selection to an pops.Spatial consumed by add_equation. Accepts an
        runtime Spatial / FiniteVolume (returned as-is), an pops.numerics.spatial.FiniteVolume(...)
        BrickDescriptor (read its riemann/reconstruction/positivity_floor options), or None (default
        Spatial)."""
        if spatial is None:
            return Spatial()
        if isinstance(spatial, Spatial):
            return spatial
        # A spatial BrickDescriptor carries already-validated canonical C++ routing tokens in
        # .options. Lower them directly (Spatial._from_tokens bypasses the public typed-descriptor
        # guard because this path is after descriptor validation -- Spec 5 sec.7).
        opts = getattr(spatial, "options", None)
        if isinstance(opts, dict):
            limiter = opts.get("reconstruction", opts.get("limiter", "minmod"))
            riemann = opts.get("riemann", opts.get("flux", "rusanov"))
            variables = opts.get("variables", opts.get("recon", "conservative"))
            return Spatial._from_tokens(
                limiter, riemann, variables,
                positivity_floor=opts.get("positivity_floor"),
                wave_speed_cache=bool(opts.get("wave_speed_cache", False)))
        raise TypeError("install: spatial must be a runtime FiniteVolume / Spatial or an "
                        "pops.numerics.spatial.FiniteVolume(...) descriptor; got %r"
                        % type(spatial).__name__)

    def _resolve_instance_model(self, model):
        """Resolve a block model to a ModelSpec/CompiledModel accepted by add_equation."""
        # Late imports (the codegen/physics modules import this package: avoid the cycle).
        from pops.codegen.loader import CompiledModel
        from pops.codegen import AOT, Production
        from pops.codegen.module_view import ModuleCodegenView, compile_module_for_runtime
        from pops.model import Module
        if isinstance(model, (ModelSpec, CompiledModel)):
            return model
        if not isinstance(model, Module) and hasattr(model, "_m"):
            raise TypeError(
                "install: legacy physics/codegen facades carrying private _m are not accepted. "
                "Pass a pops.model.Module, a CompiledModel, or a modern pops.physics.Model lowered "
                "with to_module().")
        if not isinstance(model, Module) and hasattr(model, "to_module"):
            model = model.to_module()
        if isinstance(model, Module):
            backend = AOT() if ModuleCodegenView(model).has_runtime_params() else Production()
            return compile_module_for_runtime(model, backend=backend)
        return model  # unknown -> let add_equation raise its own clear error

    def _validate_riemann_capability(self, model, spatial):
        """Section 24 capability check: reject the selected Riemann flux when the model does not back
        it, with the verbatim spec message ``riemann <FLUX> requires capability '<cap>'``. Lowered
        from the model's emitted capabilities (CompiledModel.has_hllc / has_roe / has_wave_speeds);
        a native composed model carries the capability in its bricks (the C++ requires-gate
        is the backstop), so we only gate the compiled (.so) path here."""
        from pops.codegen.loader import CompiledModel  # late import (codegen <-> __init__ cycle)
        flux = getattr(spatial, "flux", "rusanov")
        if not isinstance(model, CompiledModel):
            return  # native composed model: the C++ requires-gate validates at first use
        if flux == "hllc" and not (getattr(model, "has_hllc", False)
                                   or "p" in getattr(model, "prim_names", [])):
            raise RuntimeError("riemann HLLC requires capability 'hllc_star_state'")
        if flux == "roe" and not (getattr(model, "has_roe", False)
                                  or "p" in getattr(model, "prim_names", [])):
            raise RuntimeError("riemann Roe requires capability 'roe_dissipation'")
        if flux == "hll" and not getattr(model, "has_wave_speeds", True):
            raise RuntimeError("riemann HLL requires capability 'wave_speeds'")

    # Field names the default native Poisson route already serves (the shared system elliptic solve).
    _DEFAULT_POISSON_FIELDS = ("phi", "poisson", "charge_density", "default")

    def _install_solver(self, field, solver_brick, declared_fields=frozenset()):
        """Lower a declared field solver to set_poisson; reject typos before runtime."""
        if field not in self._DEFAULT_POISSON_FIELDS and field not in declared_fields:
            declared = ", ".join(sorted(declared_fields)) or "(none declared)"
            raise ValueError(
                "install: solver selection names field %r, which is neither the default Poisson "
                "field (%s) nor a named elliptic field any installed model declares (declared: %s). "
                "Declare it with m.elliptic_field(%r, rhs=...), or fix the field name."
                % (field, ", ".join(self._DEFAULT_POISSON_FIELDS), declared, field))
        token = self._solver_token(solver_brick)
        opts = getattr(solver_brick, "options", {}) or {}
        if callable(opts):
            opts = opts()
        opts = opts or {}
        self._set_poisson(rhs=opts.get("rhs", "charge_density"), solver=token,
                          bc=opts.get("bc", "auto"), wall=opts.get("wall", "none"),
                          wall_radius=float(opts.get("wall_radius", 0.0)),
                          epsilon=float(opts.get("epsilon", 1.0)),
                          abs_tol=float(opts.get("abs_tol", 0.0)))

    @staticmethod
    def _declared_elliptic_fields(compiled, instances):
        """Collect named elliptic fields declared by compiled and per-instance models."""
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
    def _solver_token(solver_brick):
        """Resolve a typed field-solver descriptor to its native Poisson token."""
        if isinstance(solver_brick, str):
            raise TypeError(
                "install: solver selections must be typed descriptors such as "
                "pops.solvers.GeometricMG(); got legacy token %r" % solver_brick)
        token = getattr(solver_brick, "scheme", None) or getattr(solver_brick, "name", None)
        if token is None:
            raise TypeError("install: solver must be a pops.solvers.<Solver>(...) descriptor; got %r"
                            % type(solver_brick).__name__)
        return token

    def _install_aux(self, field_name, field):
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
        self._set_aux_field(block, field_name, field)

    def _block_declaring_aux(self, field_name):
        """The block whose named-aux table declares @p field_name, or None."""
        for block, table in self._aux_field_index.items():
            if field_name in table:
                return block
        return None

    @staticmethod
    def _route_block_params(resolved_models, params):
        """Pure routing core of _install_params (no engine call -> host-testable). Map a flat
        {param_name: value} to {block: sorted runtime-param value vector} using each RESOLVED model's
        runtime_param_names (declaration defaults for unspecified names), and return the param names
        declared by no instance. @p resolved_models maps each instance name to its RESOLVED
        CompiledModel: the raw dsl.Model has no runtime_param_names accessor, so a model passed
        UNRESOLVED here contributes no params (the bug install's resolve step prevents -- see
        install step (2)). @return (per_block, unknown), per_block only listing blocks with params."""
        consumed = set()
        per_block = {}
        for name, model in resolved_models.items():
            # runtime_param_names is a @property (list); runtime_param_values is a method.
            rt_names = list(getattr(model, "runtime_param_names", []) or [])
            if not rt_names:
                continue
            values_fn = getattr(model, "runtime_param_values", None)
            defaults = list(values_fn()) if callable(values_fn) else [None] * len(rt_names)
            values = []
            for k, pname in enumerate(rt_names):
                if pname in params:
                    values.append(float(params[pname]))
                    consumed.add(pname)
                else:
                    values.append(float(defaults[k]) if defaults[k] is not None else 0.0)
            per_block[name] = values
        unknown = sorted(set(params) - consumed)
        return per_block, unknown

    def _install_params(self, resolved_models, params, reject_unknown=True):
        """Route flat {param_name: value} to set_block_params per instance: build each instance's
        sorted runtime-param vector (declaration defaults for unspecified names) and push it. @p
        resolved_models maps each instance name to its RESOLVED CompiledModel. @p reject_unknown (native
        mode): raise on a name declared by no instance (no silent drop); the compiled-problem path
        passes False (an unconsumed name may be a generated-kernel param routed in 5b). Returns the
        CONSUMED names so the caller can subtract them from the artifact-param remainder."""
        per_block, unknown = self._route_block_params(resolved_models, params)
        for name, values in per_block.items():
            self._s.set_block_params(name, values)
        if unknown and reject_unknown:
            raise ValueError("install: params %s declared by no instance's runtime parameters"
                             % (unknown,))
        return set(params) - set(unknown)  # the names an AOT instance consumed

    # Host-testable pure core (ADC-510 artifact-param routing, mirror of _route_block_params): callable
    # as System._route_program_params without building a System.
    _route_program_params = staticmethod(route_program_params)

    def _install_problem_params(self, compiled, params):
        """Route flat {param_name: value} to native runtime params for a compiled problem (ADC-510).

        Reads the compiled handle's declared routing (runtime_param_routes), builds each block's
        complete value vector (declaration defaults for unspecified names), and pushes it to the
        System-owned per-block RuntimeParams the generated kernels read. A name declared by no
        generated kernel raises (no silent drop).
        """
        routes_fn = getattr(compiled, "runtime_param_routes", None)
        routes, defaults = routes_fn() if callable(routes_fn) else ({}, {})
        per_block, unknown = self._route_program_params(routes, defaults, params)
        for blk, values in per_block.items():
            self._s.set_program_params(blk, values)
        if unknown:
            raise ValueError(
                "install: params %s declared by no runtime parameter of the compiled problem "
                "(a runtime param must be read by generated kernels and declared as a runtime param)"
                % (unknown,))
