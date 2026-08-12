"""AmrSystem install/bind mixin (ADC-619 split).

The low-level ``pops.bind`` install seam of :class:`pops.runtime._amr_system.AmrSystem`:
``_install_compiled`` (the native / compiled install orchestration) plus its resolved-field-plan
and aux helpers (``_install_field_plan`` / ``_install_aux``). Split out of ``amr_system`` for the
500-line cap; mixed into ``AmrSystem``
via inheritance and operating on ``self._s`` (the native facade) and the other AmrSystem methods
(``add_equation`` / ``set_density`` / ``set_poisson`` /
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


def _constant_analytic_program(
    components: Any,
    *,
    native_binary64: Any,
) -> tuple[list[list[str]], list[list[float]]]:
    """Lower a uniform source to the canonical scalar postfix ABI."""
    return (
        [["constant"] for _ in components],
        [
            [native_binary64(value, where="AMR initial source.components[%d]" % index)]
            for index, value in enumerate(components)
        ],
    )


def _gaussian_analytic_program(
    source: Any, *, native_binary64: Any, ranked_gaussian_center: Any
) -> tuple[list[list[str]], list[list[float]]]:
    """Lower the resolved Gaussian to the same canonical postfix ABI as all analytic data."""
    center = ranked_gaussian_center(source, where="AMR Gaussian")
    if not center:
        raise ValueError("AMR Gaussian must have an exact native-rank centre")
    inverse_width = native_binary64(source["inverse_width"], where="AMR Gaussian inverse_width")
    if not inverse_width > 0.0:
        raise ValueError("AMR Gaussian inverse_width must be strictly positive")
    opcodes: list[str] = []
    literals: list[float] = []
    for axis, coordinate in enumerate(("x", "y", "z")[: len(center)]):
        opcodes.extend((coordinate, "constant", "sub", coordinate, "constant", "sub", "mul"))
        literals.extend((0.0, center[axis], 0.0, 0.0, center[axis], 0.0, 0.0))
        if axis:
            opcodes.append("add")
            literals.append(0.0)
    opcodes.extend(("constant", "mul", "neg", "exp", "constant", "mul", "constant", "add"))
    literals.extend(
        (
            inverse_width,
            0.0,
            0.0,
            0.0,
            native_binary64(source["amplitude"], where="AMR Gaussian amplitude"),
            0.0,
            native_binary64(source["background"], where="AMR Gaussian background"),
            0.0,
        )
    )
    return [opcodes], [literals]


class _PreparedAmrFieldSolverInstall:
    """AMR native primitives consumed by provider-owned field-solver installers."""

    def __init__(self, engine: Any, field_plan: Any, install_plan: Any) -> None:
        self.engine = engine
        self.field_plan = field_plan
        self.install_plan = install_plan
        self.options = field_plan.native_install_data()
        self.slot = self.options["provider_slot"]

    def _install_common_plan(self, binding: Any, provider_route: str) -> None:
        if type(provider_route) is not str or not provider_route:
            raise TypeError("native AMR field solver provider route must be non-empty")
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
            provider_route,
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

    def install_configured(self, binding: Any) -> None:
        contract = binding.resolution.to_data()["native_contract"]
        self._install_common_plan(binding, contract["factory_route"])

    def install_component(self, binding: Any) -> None:
        if self.install_plan is None:
            raise ValueError("component field providers require the authenticated InstallPlan")
        component_bindings = binding.resolution.to_data()["component_bindings"]
        if len(component_bindings) != 2:
            raise ValueError("component field provider requires exact topology and solver bindings")
        installed = []
        from pops.fields._identity import field_identity, strict_field_data
        from pops.identity import canonical_bytes

        for authority in component_bindings:
            component = self.install_plan.components.get(authority["component_id"])
            if component is None:
                raise ValueError(
                    "field %r requires installed component %r"
                    % (self.field_plan.name, authority["component_id"])
                )
            if component.component_manifest.token != authority["component_manifest_identity"]:
                raise ValueError("field component manifest identity changed before install")
            if canonical_bytes(strict_field_data(component.interface.to_data())) != canonical_bytes(
                strict_field_data(authority["native_interface"])
            ):
                raise ValueError("field component native interface identity changed before install")
            if component.native_handle is None:
                raise ValueError("field components must be loaded before native installation")
            installed.append(component.native_handle)

        import json
        from pops.runtime._component_execution_context import component_execution_data

        nullspace = self.options["nullspace_provider"]
        boundary = {
            "identity": field_identity(
                "field-boundary-contract",
                {
                    "field": self.field_plan.identity.token,
                    "faces": self.options["boundary_faces"],
                    "nullspace_provider": nullspace,
                    "topology_identity": binding.facts.layout["topology_identity"],
                },
            ).token,
            "faces": self.options["boundary_faces"],
            "nullspace_provider": nullspace,
            "topology_identity": binding.facts.layout["topology_identity"],
        }
        request = binding.resolution.native_contract["options"]
        exact = self.engine.register_field_solver_provider(
            self.slot,
            installed[0],
            installed[1],
            component_bindings[0],
            component_bindings[1],
            json.dumps(
                component_bindings[0]["parameters"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            json.dumps(
                component_bindings[1]["parameters"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            self.install_plan.artifact.layout_plan.qualified_id,
            binding.facts.layout["topology_identity"],
            json.dumps(strict_field_data(boundary), sort_keys=True, separators=(",", ":")),
            request["relative_tolerance"],
            request["absolute_tolerance"],
            request["max_iterations"],
            component_execution_data(self.install_plan.execution_context),
        )
        if type(exact) is not str or not exact:
            raise RuntimeError("native AMR component field solver returned no exact identity")
        if exact != self.slot:
            raise RuntimeError("native AMR component field solver changed its provider route")
        self._install_common_plan(binding, exact)


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

    def _install_compiled(
        self,
        compiled: Any = None,
        *,
        instances: Any = None,
        params: Any = None,
        aux: Any = None,
        field_plans: Any = None,
        bind_schema: Any = None,
        initial_values: Any = (),
        bootstrap_plan: Any = None,
        amr_transfer: Any = None,
        install_plan: Any = None,
    ) -> Any:
        """INTERNAL low-level install seam on the AMR hierarchy (Spec 5 sec.11) -- signature parity
        with ``System._install_compiled``. NOT the public entry point: author the run with
        ``pops.bind(...)``, which dispatches System / AmrSystem and calls this seam.

        Runs the SAME early bind-input validation (``validate_install_arguments``: reject -- BEFORE
        any native mutation -- an install missing a REQUIRED argument the artifact declares, with one
        clear actionable error), then lowers to the AMR layer:

          - NATIVE install (``compiled=None``): wires each InstallPlan ``CompiledModel`` with
            ``add_equation``, installs each resolved field plan, stages exact
            ``ComponentKey`` ``InputAux`` values, and registers each instance's initial density.
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
        @param aux dict {ComponentKey: array}: declared ``InputAux`` values only.  Derived and
            field-output components have no upload path.
        @param field_plans resolved compile-time field discretizations keyed by field name.
        """
        # RUNTIME FREEZE (ADC-592): a second install on an already-bound AMR engine is a re-composition
        # and is refused explicitly -- the compiled artifact is bound exactly once.
        from pops.runtime._lifecycle import guard_assembling

        guard_assembling(self, "_install_compiled")
        if install_plan is not None:
            from pops.runtime._bound_snapshot import _require_exact_install_inputs

            install_plan = _require_exact_install_inputs(
                self, compiled, instances, field_plans, aux, params, install_plan
            )
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
        validate_install_arguments(self, compiled, instances, params, aux, field_plans=field_plans)
        if install_plan is not None:
            from pops.runtime._runtime_authorities import (
                _validate_shared_interface_implicit_execution_before_install,
            )

            _validate_shared_interface_implicit_execution_before_install(install_plan)
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
                    "instance carries its own native model)." % type(compiled).__name__
                )
        # (1) RESOLVED FIELD PLANS first (parity with System: configure native solvers before
        # adding blocks and before install_program). Field identity, provider and hierarchy policy
        # were resolved at compile time; bind only materializes that immutable plan.
        for field, field_plan in field_plans.items():
            self._install_field_plan(field, field_plan, install_plan=install_plan)

        # (2) INSTANCES: resolve every package first, then project complete BindSchema vectors before
        # installing any block. The per-instance detached CompiledModel is mandatory.
        from pops.codegen.loader import CompiledModel

        resolved_models = {}
        lowered_instances = {}
        for name, spec in instances.items():
            if not isinstance(spec, Mapping):
                raise TypeError(
                    "pops.bind: instances[%r] must be a mapping "
                    "(initial/spatial/time/model); got %r" % (name, type(spec).__name__)
                )
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
                name,
                model,
                spatial=spatial,
                time=time,
                _bind_params=per_block_params.get(name, []),
            )

        for field_plan in field_plans.values():
            self._install_field_method_runtime(field_plan, resolved_models, params)

        # (3) Stage authenticated InputAux values only.  The native registry resolves each exact
        # owner-qualified ComponentKey; there is no block/name or raw-component fallback.
        for key, field in aux.items():
            self._install_aux(key, field)

        # (4) INITIAL state: register every bootstrap authority before hierarchy materialization.
        # Cell arrays are staged on the native block descriptors here; this does not build the
        # hierarchy, so the compiled Program still installs at the unique pre-build boundary below.
        initial_rows = tuple(initial_values)
        if any(spec.get("initial") is not None for spec in instances.values()):
            raise ValueError(
                "AMR installation accepts no initial_state block table; use the resolved "
                "InitialConditionPlan and initial_values"
            )
        seen_initial = set()
        for subject_id, name, initial, space, centering, method, source in initial_rows:
            if (
                not isinstance(subject_id, str)
                or not subject_id
                or not isinstance(name, str)
                or not name
            ):
                raise ValueError(
                    "pops.bind: AMR initial state requires one qualified subject and block"
                )
            from pops.runtime._initial_source_lowering import validate_initial_source

            route = validate_initial_source(source, where="AMR initial source")
            self._s._bind_bootstrap_subject(subject_id, name, route)
            if method == "analytic":
                from pops.runtime._initial_source_lowering import (
                    native_binary64,
                    ranked_gaussian_center,
                )

                if route == "constant_field":
                    opcodes, literals = _constant_analytic_program(
                        source["components"],
                        native_binary64=native_binary64,
                    )
                elif route == "gaussian_field":
                    if space != "cell":
                        raise ValueError("pops.bind: gaussian_field requires one cell state")
                    opcodes, literals = _gaussian_analytic_program(
                        source,
                        native_binary64=native_binary64,
                        ranked_gaussian_center=ranked_gaussian_center,
                    )
                elif route == "analytic_expression":
                    projection = source.get("projection", {})
                    if (
                        space != "cell"
                        or centering != "cell"
                        or not isinstance(projection, Mapping)
                        or projection.get("projection") != "conservative_cell_average"
                    ):
                        raise ValueError(
                            "pops.bind: analytic_expression requires the cell-centred "
                            "ConservativeCellAverage projection"
                        )
                    from pops.runtime._analytic_expression_lowering import (
                        lower_analytic_components,
                    )

                    lowered = lower_analytic_components(
                        source.get("components"),
                        frame_id=source.get("frame_id"),
                        bindings=params,
                    )
                    self._s._stage_bootstrap_analytic_state(
                        subject_id,
                        name,
                        space,
                        centering,
                        "conservative_cell_average",
                        [list(component_opcodes) for component_opcodes, _ in lowered],
                        [list(component_literals) for _, component_literals in lowered],
                    )
                    continue
                else:
                    raise NotImplementedError(
                        "pops.bind: no native analytic provider for route %r" % route
                    )
                self._s._stage_bootstrap_analytic_state(
                    subject_id,
                    name,
                    space,
                    centering,
                    "conservative_cell_average",
                    opcodes,
                    literals,
                )
                continue
            if route != "bound_level_zero":
                raise ValueError(
                    "pops.bind: array initial state requires the bound_level_zero source route"
                )
            if space == "cell":
                if name not in instances:
                    raise ValueError("pops.bind: initial state targets unknown block %r" % name)
                if name in seen_initial:
                    raise ValueError(
                        "pops.bind: multiple initial physical states target block %r" % name
                    )
                seen_initial.add(name)
                self._s._stage_bootstrap_array(subject_id, name, space, centering, initial)
            else:
                raise NotImplementedError(
                    "pops.bind: native bootstrap has no payload carrier for space %r" % space
                )

        # (4b) Boundary-kernel parameters are independent from package parameters fixed in step 2.
        for field_plan in field_plans.values():
            self._install_field_boundary_parameters(field_plan, params, compiled=compiled)

        # (5/5b/6) COMPILED time Program: install_program on the AMR hierarchy, route the remaining
        # runtime params and attach the typed step-transaction contract. The native loader
        # materializes the hierarchy itself, after every block/field/aux/bootstrap descriptor exists
        # but before any cell payload is copied into that hierarchy.
        # Extracted into the _AmrSystemProgram mixin (_finish_program_install) to keep this module small.
        self._finish_program_install(compiled, so_path, bind_schema, params)

        # Authenticate and install the level-zero shared-interface routes before bootstrap.  The
        # clustering proper-nesting proof may reach a face deliberately omitted from a block's
        # physical-boundary plan; only an already prepared exact interface route may own that face.
        # The same incremental finalizer runs after every successful level creation so each newly
        # materialized parent owns its exact shared-face route before the next transition is tagged.
        if install_plan is not None:
            from pops.runtime._runtime_authorities import finalize_runtime_authorities

            finalize_runtime_authorities(self, install_plan)

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
                on_level_materialized=(
                    None
                    if install_plan is None
                    else lambda: finalize_runtime_authorities(self, install_plan)
                ),
            )

        # Extend the already authenticated interface registry to the complete materialized level
        # prefix. Keep that structural install inside the bind transaction, before the BoundSnapshot
        # and native lifecycle freeze.
        if install_plan is not None:
            from pops.runtime._runtime_authorities import finalize_runtime_authorities

            finalize_runtime_authorities(self, install_plan, complete=True)

            from pops.runtime._checkpoint_resource_budget import (
                install_amr_checkpoint_resource_budget,
            )

            install_amr_checkpoint_resource_budget(self, install_plan)

        # (7) FREEZE (ADC-592): the AMR composition is fully lowered -- build the BoundSnapshot manifest
        # of WHAT was bound (build_amr_snapshot, in _bound_snapshot), then _finalize_bind marks the
        # runtime 'bound' as the LAST act. If this route installed a whole-system Program, its
        # program/cache/ABI identity and transaction plan are retained alongside each block-model hash.
        from pops.runtime._bound_snapshot import build_amr_snapshot

        snapshot = build_amr_snapshot(
            self,
            compiled,
            instances,
            field_plans,
            aux,
            params,
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
        oriented_face_groups: dict[tuple[str, ...], int] = {}
        for entry in registry.entries:
            native = entry.native_materialization
            provider_options = native.provider_identity.to_data().get("options", {})
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
            dimension = next(iter(dimensions))
            oriented = provider_options.get("oriented_subjects")
            if oriented is not None:
                if isinstance(oriented, (str, bytes)):
                    raise TypeError(
                        "pops.bind: oriented face provider subjects must be an ordered sequence"
                    )
                group = tuple(oriented)
                if (
                    len(group) != dimension
                    or len(set(group)) != dimension
                    or any(not isinstance(value, str) or not value for value in group)
                    or key["space"]["name"] != "face"
                    or key["operation"]["name"] != "prolongation"
                    or options.get("native_route") != "divergence_preserving_face"
                ):
                    raise ValueError(
                        "pops.bind: oriented face provider requires exactly one ordered "
                        "divergence-preserving subject per native axis"
                    )
                previous_dimension = oriented_face_groups.setdefault(group, dimension)
                if previous_dimension != dimension:
                    raise ValueError(
                        "pops.bind: oriented face provider group mixes native dimensions"
                    )
            elif options.get("native_route") == "divergence_preserving_face":
                raise ValueError(
                    "pops.bind: divergence-preserving face provider omitted oriented_subjects"
                )
            ratios = {tuple(row.accuracy.refinement_ratio) for row in entry.requirements}
            if len(ratios) != 1 or len(next(iter(ratios))) != dimension:
                raise ValueError(
                    "pops.bind: one native transfer route requires one exact-ranked ratio"
                )
            ghost_depth = tuple(ghost)
            if len(ghost_depth) == 1:
                ghost_depth *= dimension
            if len(ghost_depth) != dimension:
                raise ValueError(
                    "pops.bind: one native transfer route requires exact-ranked ghost depth"
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
                ghost_depth,
                next(iter(ratios)),
            )
        for group in sorted(oriented_face_groups):
            self._s._register_bootstrap_oriented_face_subjects(group)

    def _install_field_plan(self, field: Any, field_plan: Any, *, install_plan: Any = None) -> None:
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
        provider.install(_PreparedAmrFieldSolverInstall(self._s, field_plan, install_plan), binding)
        slot = options["provider_slot"]
        faces = options["boundary_faces"]
        if faces is not None:
            self._s.set_field_boundary_plan(
                slot,
                [face["type"] for face in faces],
                [face["alpha"] for face in faces],
                [face["beta"] for face in faces],
                [face["value"] for face in faces],
            )
        dependencies = options["boundary_dependencies"]
        self._s.set_field_boundary_dependencies(
            slot,
            [row["owner_block"] for row in dependencies["states"]],
            [row["component"] for row in dependencies["states"]],
            [row["owner_block"] for row in dependencies["fields"]],
            [row["output_key"] for row in dependencies["fields"]],
            [row["component"] for row in dependencies["fields"]],
        )
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

    def _install_field_boundary_parameters(
        self, field_plan: Any, params: Any, *, compiled: Any
    ) -> None:
        if not field_plan.native_options.get("boundary_kernel_required"):
            return
        if compiled is None:
            raise ValueError(
                "dynamic AMR field boundaries require a compiled artifact that owns their "
                "generated device launchers"
            )
        handles = field_plan.provider_parameter_handles("boundary-kernel")
        missing = [handle.qualified_id for handle in handles if handle not in params]
        if missing:
            raise ValueError(
                "dynamic AMR field boundary parameter pack is incomplete: %s" % ", ".join(missing)
            )
        from pops.solvers._numeric import native_float

        values = [
            native_float(
                params[handle],
                where="dynamic AMR field boundary parameter %s" % handle.qualified_id,
            )
            for handle in handles
        ]
        self._s.set_field_boundary_parameters(field_plan.native_options["provider_slot"], values)

    def _install_field_method_runtime(
        self,
        field_plan: Any,
        models: Any,
        params: Any,
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

    def _install_aux(self, key: Any, field: Any) -> None:
        """Stage one exact externally supplied ``InputAux`` component."""
        self.stage_auxiliary_input(key, field)
