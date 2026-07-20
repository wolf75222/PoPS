"""AmrSystem install/bind mixin (ADC-619 split).

The low-level ``pops.bind`` install seam of :class:`pops.runtime._amr_system.AmrSystem`:
``_install_compiled`` (the native / compiled install orchestration) plus its resolved-field-plan
and aux helpers (``_install_field_plan`` / ``_install_aux``). Split out of ``amr_system`` for the
500-line cap; mixed into ``AmrSystem``
via inheritance and operating on ``self._s`` (the native facade), ``self._aux_field_index`` and
the other AmrSystem methods (``add_equation`` / ``set_density`` / ``set_poisson`` /
``_finish_program_install``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pops.runtime._bind_validation import validate_install_arguments

if TYPE_CHECKING:
    from pops.runtime._amr_system_contract import _AmrSystem
else:
    _AmrSystem = object


class _PreparedAmrFieldSolverInstall:
    """AMR native primitives consumed by provider-owned field-solver installers."""

    def __init__(self, engine: Any, field_plan: Any) -> None:
        self.engine = engine
        self.field_plan = field_plan
        self.options = field_plan.native_install_data()
        self.slot = self.options["provider_slot"]

    def install_configured(self, binding: Any) -> None:
        contract = binding.resolution.to_data()["native_contract"]
        routes = self.options["provider_pack"]
        output = self.options["output_route"]
        hierarchy_policy = self.options["hierarchy_policy"]
        if not isinstance(hierarchy_policy, Mapping) or set(hierarchy_policy) != {
            "policy_id",
            "interface_version",
            "option_schema",
            "options",
        }:
            raise TypeError("resolved AMR hierarchy-policy authority has an invalid shape")
        from pops.identity import canonical_bytes

        self.engine.set_field_solver_plan(
            self.slot,
            self.field_plan.identity.token,
            self.options["provider_identity_text"],
            canonical_bytes(output["owner_identity"]).hex(),
            output["owner_block"],
            output["key"],
            [canonical_bytes(route["provider_identity"]).hex() for route in routes],
            [route["owner_block"] for route in routes],
            [route["key"] for route in routes],
            [route["coefficient"] for route in routes],
            contract["factory_route"],
            hierarchy_policy["policy_id"],
            hierarchy_policy["interface_version"],
            hierarchy_policy["option_schema"],
            hierarchy_policy["options"],
            contract["schema_identity"],
            contract["options"],
        )
        topology = binding.resolution.to_data()["topology_contract"]
        self.engine._set_field_topology_authority(
            self.slot,
            topology["provider_id"],
            binding.identity,
            topology["topology_identity"],
        )

    def install_component(self, _binding: Any) -> None:
        raise RuntimeError(
            "component field solver reached AMR after its provider policy rejected the use"
        )


class _PreparedAmrFieldNullspaceInstall:
    def __init__(self, engine: Any, slot: str) -> None:
        self.engine = engine
        self.slot = slot

    def install_registered_nullspace(self, binding: Any) -> None:
        contract = binding.resolution.to_data()["native_contract"]
        self.engine.set_field_nullspace(
            self.slot,
            contract["provider_route"],
            contract["schema_identity"],
            contract["options"],
        )


class _AmrSystemInstall(_AmrSystem):
    """``pops.bind`` install seam for :class:`AmrSystem` (mixed in; operates on ``self``)."""

    def _install_compiled(self, compiled: Any = None, *, instances: Any = None, params: Any = None,
                          aux: Any = None, field_plans: Any = None,
                          bind_schema: Any = None, initial_values: Any = (),
                          bootstrap_plan: Any = None, amr_transfer: Any = None,
                          install_plan: Any = None) -> Any:
        """INTERNAL low-level install seam on the AMR hierarchy (Spec 5 sec.11) -- signature parity
        with ``System._install_compiled``. NOT the public entry point: author the run with
        ``pops.bind(...)``, which dispatches System / AmrSystem and calls this seam.

        Runs the SAME early bind-input validation (``validate_install_arguments``: reject -- BEFORE
        any native mutation -- an install missing a REQUIRED argument the artifact declares, with one
        clear actionable error), then lowers to the AMR layer:

          - NATIVE install (``compiled=None``): wires each InstallPlan ``CompiledModel`` with
            ``add_equation``, installs each resolved field plan,
            the aux inputs (``set_magnetic_field`` / ``set_aux_field``) and each instance's initial
            density (``set_density``). This is the real AMR add path; a full run is Kokkos-gated.
          - COMPILED install (a ``compiled`` handle carrying a time Program, epic ADC-511 / ADC-508 /
            ADC-634): the same wiring, then ``install_program(so_path)`` installs the compiled Program
            on the AMR hierarchy (the .so must export ``pops_install_program_amr``: compile it with
            ``target='amr_system'``). The runtime params (``params=``) route to ``set_program_params``
            through the same Program transaction contract as Uniform. The per-level macro-step driver
            is the AmrProgramContext seam (ADC-508); a
            Program using a deferred op (Schur / history / named-flux) compiles against it and throws
            the honest AmrProgramContext backstop only when that op is reached at run.

        @param compiled a compiled time-Program handle, or ``None`` for a native AMR install.
        @param instances dict {name: {"initial": array, "spatial": <brick>, "model": <CompiledModel>,
            "time": <policy>}}; the block is bound by the dict KEY.
        @param params canonical block-qualified runtime values resolved by BindSchema. Complete
            per-package vectors are fixed before native closures are built; Program-owned values
            route independently through ``set_program_params``.
        @param aux dict {field_name: array}: "B_z" -> set_magnetic_field, "T_e" rejected (derived),
            any other -> set_aux_field on the declaring block.
        @param field_plans resolved compile-time field discretizations keyed by field name.
        """
        # RUNTIME FREEZE (ADC-592): a second install on an already-bound AMR engine is a re-composition
        # and is refused explicitly -- the compiled artifact is bound exactly once.
        from pops.runtime._lifecycle import guard_assembling
        guard_assembling(self, "_install_compiled")
        if install_plan is not None:
            from pops.runtime._bound_snapshot import _require_exact_install_inputs

            install_plan = _require_exact_install_inputs(
                self, compiled, instances, field_plans, aux, params, install_plan)
            if bind_schema is not install_plan.artifact.bind_schema:
                raise ValueError("AMR bind schema must be the exact value from the InstallPlan")
            if bootstrap_plan is not install_plan.bootstrap_plan:
                raise ValueError("AMR bootstrap plan must be the exact value from the InstallPlan")
            if amr_transfer is not install_plan.amr_transfer:
                raise ValueError("AMR transfer must be the exact value from the InstallPlan")
            compiled = install_plan.artifact
            instances = install_plan.instances
            params = install_plan.params
            aux = install_plan.aux
            field_plans = install_plan.artifact.plan.field_plans
        else:
            instances = instances or {}
            params = {} if params is None else params
            aux = aux or {}
            field_plans = field_plans or {}

        # (0) EARLY VALIDATION (shared with System._install_compiled): reject a compiled install missing a
        # required declared argument BEFORE any native mutation. Inert (reads arguments() metadata).
        validate_install_arguments(
            self, compiled, instances, params, aux, field_plans=field_plans)
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
        # (1) RESOLVED FIELD PLANS first (parity with System: configure native solvers before
        # adding blocks and before install_program). Field identity, provider and hierarchy policy
        # were resolved at compile time; bind only materializes that immutable plan.
        for field, field_plan in field_plans.items():
            self._install_field_plan(field, field_plan)

        # (2) INSTANCES: resolve every package first, then project complete BindSchema vectors before
        # installing any block. The per-instance detached CompiledModel is mandatory.
        from pops.codegen.loader import CompiledModel

        resolved_models = {}
        lowered_instances = {}
        for name, spec in instances.items():
            if not isinstance(spec, Mapping):
                raise TypeError("pops.bind: instances[%r] must be a mapping "
                                "(initial/spatial/time/model); got %r"
                                % (name, type(spec).__name__))
            model = spec.get("model")
            if not isinstance(model, CompiledModel):
                raise TypeError(
                    "pops.bind: instances[%r] must carry a detached target='amr_system' "
                    "CompiledModel from InstallPlan; re-run pops.resolve(case) and "
                    "pops.compile(plan)" % name
                )
            spatial = spec.get("spatial")
            time = spec.get("time")
            resolved_models[name] = model
            lowered_instances[name] = (model, spatial, time)

        if bind_schema is None and compiled is not None:
            bind_schema = getattr(compiled, "bind_schema", None)
        if bind_schema is not None:
            from pops.runtime._install_param_routing import route_block_params
            per_block_params = route_block_params(resolved_models, bind_schema, params)
        elif params:
            raise ValueError(
                "pops.bind: parameter values require a compiled artifact carrying BindSchema"
            )
        else:
            per_block_params = {}

        for name, (model, spatial, time) in lowered_instances.items():
            self.add_equation(
                name, model, spatial=spatial, time=time,
                _bind_params=per_block_params.get(name, []),
            )

        for field_plan in field_plans.values():
            self._install_field_method_runtime(field_plan, resolved_models, params)

        # (3) AUX fields: B_z -> set_magnetic_field; named -> set_aux_field. After the blocks exist
        # (a named aux resolves against the block's declared aux table) and BEFORE install_program.
        for field_name, field in aux.items():
            self._install_aux(field_name, field)

        # (4) INITIAL state: AMR has one typed InitialConditionPlan authority. Uniform
        # ``initial_state`` block tables never enter this installer.
        initial_rows = tuple(initial_values)
        if any(spec.get("initial") is not None for spec in instances.values()):
            raise ValueError(
                "AMR installation accepts no initial_state block table; use the resolved "
                "InitialConditionPlan and initial_values"
            )
        seen_initial = set()
        for subject_id, name, initial, space, centering, method, source in initial_rows:
            if method == "analytic":
                from pops.runtime._initial_source_lowering import (
                    native_binary64,
                    validate_initial_source,
                )

                route = validate_initial_source(source, where="AMR initial source")
                if route == "constant_field":
                    components = [
                        native_binary64(
                            value, where="AMR initial source.components[%d]" % index,
                        )
                        for index, value in enumerate(source["components"])
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
                    self._s._register_analytic_gaussian(
                        subject_id, name or "",
                        native_binary64(center["x"], where="AMR Gaussian center.x"),
                        native_binary64(center["y"], where="AMR Gaussian center.y"),
                        native_binary64(
                            source["background"], where="AMR Gaussian background"),
                        native_binary64(
                            source["amplitude"], where="AMR Gaussian amplitude"),
                        native_binary64(
                            source["inverse_width"], where="AMR Gaussian inverse_width"),
                    )
                elif route == "analytic_expression":
                    projection = source.get("projection", {})
                    if space != "cell" or centering != "cell" \
                            or not isinstance(projection, Mapping) \
                            or projection.get("projection") != "conservative_cell_average":
                        raise ValueError(
                            "pops.bind: analytic_expression requires the cell-centred "
                            "ConservativeCellAverage projection")
                    from pops.runtime._analytic_expression_lowering import (
                        lower_analytic_components,
                    )

                    lowered = lower_analytic_components(
                        source.get("components"),
                        frame_id=source.get("frame_id"),
                        bindings=params,
                    )
                    self._s._register_analytic_expression(
                        subject_id,
                        name or "",
                        space,
                        centering,
                        [list(opcodes) for opcodes, _ in lowered],
                        [list(literals) for _, literals in lowered],
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

        # (4b) Boundary-kernel parameters are independent from package parameters fixed in step 2.
        for field_plan in field_plans.values():
            self._install_field_boundary_parameters(field_plan, params, compiled=compiled)

        if bootstrap_plan is not None:
            from pops.runtime._amr_bootstrap_execution import execute_native_bootstrap

            self._bootstrap_execution = execute_native_bootstrap(
                self,
                bootstrap_plan,
                initial_rows,
                {
                    name: field_plan.native_options["provider_slot"]
                    for name, field_plan in field_plans.items()
                },
            )

        # (5/5b/6) COMPILED time Program: install_program on the AMR hierarchy, route the remaining
        # runtime params and attach the typed step-transaction contract.
        # Extracted into the _AmrSystemProgram mixin (_finish_program_install) to keep this module small.
        self._finish_program_install(compiled, so_path, bind_schema, params)

        # The shared-interface scheduler authenticates the materialized per-level MultiFabs. Keep
        # that structural install inside the bind transaction: after lazy runtime construction, before
        # the BoundSnapshot and native lifecycle freeze.
        if install_plan is not None:
            from pops.runtime._runtime_authorities import finalize_runtime_authorities
            finalize_runtime_authorities(self, install_plan)

        # (7) FREEZE (ADC-592): the AMR composition is fully lowered -- build the BoundSnapshot manifest
        # of WHAT was bound (build_amr_snapshot, in _bound_snapshot), then _finalize_bind marks the
        # runtime 'bound' as the LAST act. If this route installed a whole-system Program, its
        # program/cache/ABI identity and transaction plan are retained alongside each block-model hash.
        from pops.runtime._bound_snapshot import build_amr_snapshot
        snapshot = build_amr_snapshot(
            self, compiled, instances, field_plans, aux, params,
            install_plan=install_plan,
        )
        self._finalize_bind(snapshot)  # freeze (ADC-592): _finalize_bind lives on _LifecycleMixin

    def _install_bootstrap_routes(self, registry: Any) -> None:
        from pops.mesh._amr.transfer import (
            NativeAMRMaterializationKind,
            ResolvedAMRTransfer,
        )

        if type(registry) is not ResolvedAMRTransfer:
            raise TypeError("pops.bind: amr_transfer must be an exact AMRTransfer")
        face_vectors = set()
        for entry in registry.entries:
            native = entry.native_materialization
            provider_options = native.provider_identity.to_data().get("options", {})
            paired = provider_options.get("paired_subjects")
            if paired is not None:
                pair = tuple(paired)
                if len(pair) != 2 or any(not isinstance(value, str) for value in pair):
                    raise ValueError("pops.bind: paired face provider has an invalid subject manifest")
                face_vectors.add(pair)
            key = entry.key.to_data()
            if native.materialization is NativeAMRMaterializationKind.PHYSICAL:
                options = native.options.to_data()
                capabilities = native.capabilities.transfer
                if capabilities is None:
                    raise ValueError(
                        "pops.bind: physical AMR descriptor omitted transfer capabilities"
                    )
                order, ghost = capabilities.order, capabilities.ghost_depth
            else:
                options = native.options.to_data()
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
                entry.identity.token,
                [row.subject.qualified_id for row in entry.requirements],
                native.provider_qualified_id,
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

    def _install_field_plan(self, field: Any, field_plan: Any) -> None:
        """Install the complete resolved AMR field route before native block loaders run."""
        from pops.codegen.field_install import ResolvedFieldInstallPlan
        if not isinstance(field_plan, ResolvedFieldInstallPlan):
            raise TypeError("install field_plans must contain ResolvedFieldInstallPlan values")
        if field_plan.name != field or field_plan.target != "amr_system":
            raise ValueError("resolved AMR field install plan identity/target mismatch")
        field_plan.__post_init__()
        options = field_plan.native_install_data()
        from pops.fields._prepared_field_solver_registry import (
            prepared_field_solver_binding_from_data,
            prepared_field_solver_provider_from_identity,
        )

        binding = prepared_field_solver_binding_from_data(options["solver_provider"])
        provider = prepared_field_solver_provider_from_identity(binding.provider)
        provider.install(_PreparedAmrFieldSolverInstall(self._s, field_plan), binding)
        slot = options["provider_slot"]
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
        self._install_field_nullspace(slot, field_plan)
        nonlinear = options.get("nonlinear")
        if nonlinear is not None:
            field_plan.nonlinear_provider.install(self._s, slot)

    def _install_field_nullspace(self, slot: str, field_plan: Any) -> None:
        from pops.fields._prepared_field_nullspace_registry import (
            prepared_field_nullspace_binding_from_data,
            prepared_field_nullspace_provider_from_identity,
        )

        binding = prepared_field_nullspace_binding_from_data(
            field_plan.native_install_data()["nullspace_provider"]
        )
        provider = prepared_field_nullspace_provider_from_identity(binding.provider)
        provider.install(_PreparedAmrFieldNullspaceInstall(self._s, slot), binding)

    def _install_field_boundary_parameters(self, field_plan: Any, params: Any, *,
                                           compiled: Any) -> None:
        if not field_plan.native_options.get("boundary_kernel_required"):
            return
        if compiled is None:
            raise ValueError(
                "dynamic AMR field boundaries require a compiled artifact that owns their "
                "generated device launchers")
        handles = field_plan.provider_parameter_handles("boundary-kernel")
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

    def _install_field_method_runtime(
        self, field_plan: Any, models: Any, params: Any,
    ) -> None:
        """Offer opaque target resources to the method provider's authenticated installer."""
        from pops.fields import PreparedFieldRuntimeInstallContext

        field_plan.install_runtime(
            PreparedFieldRuntimeInstallContext(
                target=field_plan.target,
                engine=self._s,
                resources={"models": models},
                slot=field_plan.native_options["provider_slot"],
            ),
            params,
        )

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
