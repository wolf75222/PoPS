"""System unified-install mixin (Spec-4 PR-F): the INTERNAL ``_install_compiled`` seam.

``_install_compiled`` (the low-level seam that stages all native packages, seals their one global
ProviderPack carrier, then installs blocks/fields/program) plus its private
lowering helpers. It is NOT the public entry point (Spec 5 sec.11): authors call
``pops.bind(artifact, initial_state=..., params=..., aux=..., resources=..., initial_values=...)``;
binding dispatches to the private System / AmrSystem engine and calls this seam. Mixed into
``System`` via inheritance; methods operate on ``self`` (calling the
other mixins' methods) and ``self._s``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pops.runtime._bricks_scheme import Spatial
from pops.runtime._install_param_routing import route_block_params, route_program_params

from pops.runtime._bind_validation import (
    collect_missing_arguments as _collect_missing_arguments_impl,
    validate_install_arguments as _validate_install_arguments_impl,
)

if TYPE_CHECKING:
    from pops.runtime._system_contract import _System
else:
    _System = object


class _PreparedSystemFieldSolverInstall:
    """Native primitives offered to provider-owned installers on the uniform System."""

    def __init__(self, engine: Any, field_plan: Any, install_plan: Any) -> None:
        self.engine = engine
        self.field_plan = field_plan
        self.install_plan = install_plan
        self.options = field_plan.native_install_data()
        self.slot = self.options["provider_slot"]

    def _install_common_plan(self, provider_route: str) -> None:
        if type(provider_route) is not str or not provider_route:
            raise TypeError("native field solver provider route must be non-empty")
        routes = self.options["provider_pack"]
        output = self.options["output_route"]
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
        )

    def _install_topology_authority(self, binding: Any) -> None:
        topology = binding.resolution.to_data()["topology_contract"]
        self.engine._set_field_topology_authority(
            self.slot,
            topology["provider_id"],
            binding.identity,
            topology["topology_identity"],
        )

    def install_configured(self, binding: Any) -> None:
        contract = binding.resolution.to_data()["native_contract"]
        exact = self.engine.register_configured_field_solver_provider(
            contract["factory_route"],
            self.slot,
            contract["schema_identity"],
            contract["options"],
        )
        if type(exact) is not str or not exact:
            raise RuntimeError("native configured field solver returned no exact identity")
        # The topology authority is attached to the resolved field-plan slot.  Registering the
        # backend provider alone does not create that slot, so the common plan must precede it.
        self._install_common_plan(self.slot)
        self._install_topology_authority(binding)

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
            raise RuntimeError("native component field solver returned no exact identity")
        self._install_common_plan(self.slot)


class _PreparedSystemFieldNullspaceInstall:
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


class _SystemUnifiedInstall(_System):
    """The internal ``_install_compiled`` lowering seam of System (driven by ``pops.bind``)."""

    def _install_initial_source(
        self,
        name: str,
        row: Mapping[str, Any],
        *,
        params: Any,
    ) -> None:
        source = row.get("source")
        from pops.runtime._initial_source_lowering import (
            native_binary64,
            ranked_gaussian_center,
            validate_initial_source,
        )

        if not isinstance(source, Mapping):
            raise TypeError("uniform initial source must be a canonical mapping")
        route = validate_initial_source(source, where="uniform initial source")
        if route == "bound_level_zero":
            value = row.get("value")
            if value is None:
                raise ValueError("uniform BindArray initial source has no bound value")
            self.set_state(name, value)
            return
        projection = source.get("projection", {})
        if (
            not isinstance(projection, Mapping)
            or projection.get("projection") != "conservative_cell_average"
            or row.get("space") != "cell"
            or row.get("centering") != "cell"
        ):
            raise ValueError(
                "uniform analytic initials require the cell-centred "
                "ConservativeCellAverage projection"
            )
        if route == "constant_field":
            components = tuple(source.get("components", ()))
            if not components:
                raise ValueError("uniform constant initial source has no components")
            opcodes = [["constant"] for _ in components]
            literals = [
                [
                    native_binary64(
                        value,
                        where="uniform initial source.components[%d]" % index,
                    )
                ]
                for index, value in enumerate(components)
            ]
            self._s._set_analytic_expression_state(
                name, "cell", "cell", "conservative_cell_average", opcodes, literals
            )
            return
        if route == "gaussian_field":
            self._s._set_analytic_gaussian_state(
                name,
                ranked_gaussian_center(source, where="uniform Gaussian"),
                native_binary64(source["background"], where="uniform Gaussian background"),
                native_binary64(source["amplitude"], where="uniform Gaussian amplitude"),
                native_binary64(source["inverse_width"], where="uniform Gaussian inverse_width"),
            )
            return
        if route == "analytic_expression":
            from pops.runtime._analytic_expression_lowering import lower_analytic_components

            lowered = lower_analytic_components(
                source["components"],
                frame_id=source["frame_id"],
                bindings=params,
            )
            self._s._set_analytic_expression_state(
                name,
                "cell",
                "cell",
                "conservative_cell_average",
                [list(opcodes) for opcodes, _ in lowered],
                [list(literals) for _, literals in lowered],
            )
            return
        if route == "field_mapped_analytic_expression":
            from pops.runtime._analytic_expression_lowering import lower_analytic_components

            seed = lower_analytic_components(
                source["seed_components"],
                frame_id=source["frame_id"],
                bindings=params,
            )
            self._s._set_analytic_expression_state(
                name,
                "cell",
                "cell",
                "conservative_cell_average",
                [list(opcodes) for opcodes, _ in seed],
                [list(literals) for _, literals in seed],
            )
            self._s.solve_fields()
            mapped = lower_analytic_components(
                source["components"],
                frame_id=source["frame_id"],
                bindings=params,
            )
            inputs = sorted(source["inputs"], key=lambda row: row["value_id"])
            input_bindings = []
            for row in inputs:
                if row["source"] == "state":
                    input_bindings.append(
                        {
                            "source": "state",
                            "component": int(row["component"]),
                        }
                    )
                elif row["source"] == "provider":
                    input_bindings.append(
                        {
                            "source": "provider",
                            "key": dict(row["key"]),
                        }
                    )
                else:  # schema validation must already have rejected this branch.
                    raise ValueError(
                        "field-mapped analytic input has no exact state/provider source"
                    )
            self._s._set_analytic_mapped_state(
                name,
                [list(opcodes) for opcodes, _ in mapped],
                [list(literals) for _, literals in mapped],
                input_bindings,
                source["consumer_qid"],
            )
            return
        raise NotImplementedError("uniform initial source route %r is not native" % route)

    def _install_compiled(
        self,
        compiled=None,
        *,
        instances=None,
        params=None,
        aux=None,
        field_plans=None,
        install_plan=None,
        initial_sources=None,
        _layout_checkpoint_install=None,
        authority_plan=None,
    ):
        """INTERNAL low-level install seam (Spec 5 sec.11): wire a compiled handle + per-instance
        state/spatial + params + aux + resolved field plans in one call, then install the compiled time
        Program. NOT the public entry point: author the run with ``pops.bind(artifact,
        initial_state=..., params=..., aux=..., resources=..., initial_values=...)``. Binding
        dispatches to the private System / AmrSystem engine and calls this seam. This
        method is undocumented on the public surface (it carries no ``install`` alias) and may change.

        It uses one three-phase native transaction: stage every package and its
        routes; seal the global ProviderPack carrier; then install every block,
        field plan and Program.  There is no per-block sealing or name-based
        auxiliary installation path.

        The seam supports a compiled-Program runtime and a per-block native runtime. Both are reached
        exclusively through the public lifecycle; neither exposes a second authoring entry point.

        @param compiled the compiled problem handle (compile_problem(...) result) carrying ``so_path``,
            installed via install_program after every instance/field-plan/aux route is wired. Pass ``None`` for a
            spatial-only native assembly: no Program is installed, so stepping remains unavailable
            until an explicit Program authority is installed. Each instance must still supply its own
            InstallPlan ``CompiledModel`` and optional spatial metadata.
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
        @param aux dict {ComponentKey: array}: externally supplied InputAux values.  DerivedAux and
            field-output components have no Python upload route.
        @param field_plans complete resolve-time field installation plans. Solver, boundary,
            nullspace, hierarchy and output authority are already fixed and authenticated.
        @throws the verbatim Spec section-24 errors at bind (missing aux / field plan / block instance /
            Riemann capability). A disallowed schedule is rejected earlier, at Program compile.
        """
        # RUNTIME FREEZE (ADC-592): a second install on an already-bound engine is refused explicitly.
        from pops.runtime._lifecycle import guard_assembling

        guard_assembling(self, "_install_compiled")
        if install_plan is not None and _layout_checkpoint_install is not None:
            raise ValueError("uniform bind has competing checkpoint resource authorities")
        if install_plan is not None:
            from pops.runtime._bound_snapshot import _require_exact_install_inputs

            install_plan = _require_exact_install_inputs(
                self, compiled, instances, field_plans, aux, params, install_plan
            )
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

        # (0) EARLY VALIDATION (Spec 5 sec.10): in the COMPILED path, read the artifact's DECLARED bind
        # inputs (compiled.arguments()) and reject BEFORE any native call an install missing a REQUIRED
        # argument (instance / param / aux). Inert (reads metadata); enforces only 'required',
        # so a valid install is unchanged.
        self._validate_install_arguments(compiled, instances, params, aux, field_plans=field_plans)

        # (1) Stage every package.  Route registration happens inside each staged DSO; no block may
        # capture provider storage until the following one global finalization.
        # lower its spatial brick and set its initial state. Every instance comes from InstallPlan and
        # carries its own detached CompiledModel; bind never consults compiled.model or a PDE builder.
        so_path = None
        if compiled is not None:
            so_path = getattr(compiled, "so_path", None)
            if so_path is None:
                raise TypeError(
                    "pops.bind: compiled handle has no .so_path (got %r); pass a compile_problem(...) "
                    "result, or compiled=None for a native sim (each instance carries its own native "
                    "model)." % type(compiled).__name__
                )
        resolved_models = {}
        lowered_instances = {}
        for name, spec in instances.items():
            if not isinstance(spec, Mapping):
                raise TypeError(
                    "pops.bind: instances[%r] must be a mapping (initial/spatial/time/model); "
                    "got %r" % (name, type(spec).__name__)
                )
            model = spec.get("model")
            if model is None:
                raise ValueError(
                    "pops.bind: instance %r has no CompiledModel from InstallPlan; resolve and "
                    "compile the Case before binding" % name
                )
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
                "pops.bind: parameter values require a compiled artifact carrying BindSchema"
            )
        else:
            per_block_params = {}

        pending_initials = []
        self._batch_native_packages = True
        try:
            for name, (spec, model, spatial, time) in lowered_instances.items():
                self.add_equation(
                    name,
                    model,
                    spatial=spatial,
                    time=time,
                    _bind_params=per_block_params.get(name, []),
                )
                initial = spec.get("initial")
                source = None if initial_sources is None else initial_sources.get(name)
                if initial is not None and source is not None:
                    raise ValueError("uniform block has competing initial_state and InitialCondition")
                pending_initials.append((name, initial, source))

            # (2) Resolve the complete field-plan/provider image before native package finalization.
            # Provider-owned output installers stage their exact routes while packages are pending;
            # finalization materializes those outputs against the sealed detached auxiliary/block image,
            # attaches package RHS closures, then runs prepared-boundary installers. No field backend is
            # registered twice and no prepared boundary can observe a missing named field.
            for field, field_plan in field_plans.items():
                self._install_field_plan(field, field_plan, install_plan=install_plan)

            # Boundary parameters and method-bound coefficients are part of the exact field plan consumed
            # during backend construction, so install them while the plan remains unmaterialized.
            for field_plan in field_plans.values():
                self._install_field_boundary_parameters(field_plan, params, compiled=compiled)

            for field_plan in field_plans.values():
                self._install_field_method_runtime(field_plan, resolved_models, params)

            if self._pending_native_packages:
                self._s._finalize_native_packages()
                self._pending_native_packages = 0
        finally:
            self._batch_native_packages = False

        # (3) External InputAux values are staged only after the global registry has authenticated
        # their exact ComponentKeys.  A derived or field-output key is rejected natively.
        for key, field in aux.items():
            self._install_aux(key, field)

        # Initial conditions run after field/aux/boundary providers exist, because coupled analytic
        # profiles may materialize a seed state, solve a field, then map state+aux into the final
        # conservative vector. Still before install_program/mark_bound, so this remains bind-time data.
        for name, initial, source in pending_initials:
            if source is not None:
                self._install_initial_source(name, source, params=params)
            elif initial is not None:
                self.set_state(name, initial)

        # (5) COMPILED mode only: install the compiled time Program (binds blocks by name + runs the
        # section-24 .so requirement validation: aux / solver / block instance, verbatim messages). In
        # NATIVE mode (compiled=None) deliberately installs no temporal authority. The blocks are
        # inspectable spatial carriers, but step/advance fail closed until a Program is installed.
        if so_path is not None:
            component = getattr(compiled, "program", None)
            authored = getattr(component, "program", component)
            from pops.runtime._program_cadence_install import install_program_cadence

            install_program_cadence(self, authored)
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
                set_persistence({name: policy for name, (_depth, policy) in persistence.items()})
            # (5b) Program carriers were emitted with neutral values. Always install the complete
            # BindSchema projection after loading, including declaration defaults.
            self._install_program_params(compiled, bind_schema, params)
            self._step_strategy = getattr(authored, "_step_strategy", None)
            self._step_transaction_plan = (
                authored.transaction_plan() if authored is not None else None
            )
            if authored is not None:
                self._temporal_restart_state.configure_program(
                    authored.temporal_manifest(), time=self.time(), macro_step=self.macro_step()
                )

        # Shared NumericalFlux routes need both endpoint MultiFabs and the installed Program, but
        # remain structural bind authorities. Materialize them here, after block construction and
        # before the lifecycle snapshot/freeze; no post-bind mutation seam is introduced.
        if install_plan is not None:
            from pops.runtime._runtime_authorities import finalize_runtime_authorities

            finalize_runtime_authorities(self, install_plan)

            from pops.runtime._checkpoint_resource_budget import (
                install_uniform_checkpoint_resource_budget,
            )

            install_uniform_checkpoint_resource_budget(self, install_plan)
        elif _layout_checkpoint_install is not None:
            if (
                type(_layout_checkpoint_install) is not tuple
                or len(_layout_checkpoint_install) != 5
            ):
                raise TypeError("layout bind has an invalid checkpoint resource authority")
            from pops.runtime._runtime_authorities import finalize_layout_runtime_authorities

            finalize_layout_runtime_authorities(self, authority_plan)
            from pops.runtime._checkpoint_resource_budget import (
                install_layout_checkpoint_resource_budget,
            )

            program, block_names, artifact_identity, bind_identity, layout_plan = (
                _layout_checkpoint_install
            )
            install_layout_checkpoint_resource_budget(
                self,
                program=program,
                block_names=block_names,
                artifact_identity=artifact_identity,
                bind_identity=bind_identity,
                install_plan=layout_plan,
            )

        # (8) FREEZE (ADC-592): the composition is fully lowered -- snapshot WHAT was bound, then
        # _finalize_bind marks the runtime 'bound' as the LAST act (nothing above ran frozen, so the
        # install sequence never trips its own guards).
        from pops.runtime._bound_snapshot import build_uniform_snapshot

        snapshot = build_uniform_snapshot(
            self,
            compiled,
            resolved_models,
            instances,
            field_plans,
            aux,
            params,
            install_plan=install_plan,
        )
        self._finalize_bind(snapshot)  # _finalize_bind lives on _LifecycleMixin

    def explain_bind(self, compiled: Any) -> Any:
        """A printable :class:`pops.codegen.inspect_report.BindReport` of @p compiled vs this sim
        (Spec 5 sec.12.1, criterion #15). INERT: reads the artifact's DECLARED bind inputs
        (``compiled.arguments()``) and the blocks / named aux ALREADY wired on this System, then
        reuses the ADC-463 :func:`collect_missing_arguments` to compute, per group
        (instances / params / aux), which inputs are PROVIDED vs still REQUIRED. It binds
        nothing and mutates nothing -- the read-only counterpart of the install seam's early
        validation."""
        from pops.codegen.inspect_report import build_bind_report

        return build_bind_report(self, compiled)

    def _validate_install_arguments(
        self, compiled: Any, instances: Any, params: Any, aux: Any, *, field_plans: Any = None
    ) -> Any:
        """Early bind-input validation (Spec 5 sec.10): reject a COMPILED install missing a REQUIRED
        argument the artifact declares, BEFORE any native mutation. Thin wrapper around the shared
        private ``_bind_validation.validate_install_arguments`` implementation."""
        _validate_install_arguments_impl(
            self, compiled, instances, params, aux, field_plans=field_plans
        )

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
            "pops.bind: spatial must implement the pops.numerics finite-volume lowering protocol; "
            "got %r" % type(spatial).__name__
        )

    def _resolve_instance_model(self, model: Any) -> Any:
        """Accept only a runtime-ready model emitted into ``InstallPlan``.

        Compiling a PDE builder during bind made the runtime a second compiler and reintroduced live
        authoring authority. Public ``pops.compile`` now builds every block loader up front.
        """
        from pops.codegen.loader import CompiledModel

        if isinstance(model, CompiledModel):
            return model
        raise TypeError(
            "pops.bind: instance model must be a detached CompiledModel from InstallPlan, got %s; "
            "compile the Case before binding" % type(model).__name__
        )

    def _validate_riemann_capability(self, model: Any, spatial: Any) -> Any:
        """Section 24 capability check: reject the selected Riemann flux when a compiled model does
        not back its descriptor-owned requirements. The same capability-predicate protocol is used
        by System, AMR and pre-runtime availability, including third-party descriptors. A private
        native ``ModelSpec`` skips it because the C++ requires-gate validates at first use."""
        from pops.codegen.loader import CompiledModel  # late import (codegen <-> __init__ cycle)

        if not isinstance(model, CompiledModel):
            return
        from pops.runtime.routes import check_riemann_requirement_contract

        check_riemann_requirement_contract(
            spatial.riemann_capability_contract,
            model,
            "pops.bind",
            flux=spatial.flux,
        )

    def _install_field_plan(self, field: Any, field_plan: Any, *, install_plan: Any = None) -> None:
        """Consume every resolve-time field-plan property at the native boundary."""
        from pops.codegen.field_install import ResolvedFieldInstallPlan

        if not isinstance(field_plan, ResolvedFieldInstallPlan):
            raise TypeError("install field_plans must contain ResolvedFieldInstallPlan values")
        if field_plan.name != field or field_plan.target != "system":
            raise ValueError("resolved field install plan identity/target mismatch")
        # Re-run canonical construction verification before touching the native engine.
        field_plan.__post_init__()
        options = field_plan.native_install_data()
        from pops.fields._prepared_field_solver_registry import (
            prepared_field_solver_binding_from_data,
            prepared_field_solver_provider_from_identity,
        )

        binding = prepared_field_solver_binding_from_data(options["solver_provider"])
        provider = prepared_field_solver_provider_from_identity(binding.provider)
        provider.install(
            _PreparedSystemFieldSolverInstall(self._s, field_plan, install_plan), binding
        )
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
        # Provider-owned nullspace installation follows after the exact boundary topology is set.
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
        provider.install(_PreparedSystemFieldNullspaceInstall(self._s, slot), binding)

    def _install_field_boundary_parameters(
        self, field_plan: Any, params: Any, *, compiled: Any
    ) -> None:
        if not field_plan.native_options.get("boundary_kernel_required"):
            return
        if compiled is None:
            raise ValueError(
                "dynamic field boundaries require a compiled artifact that owns their generated "
                "device launchers"
            )
        handles = field_plan.provider_parameter_handles("boundary-kernel")
        missing = [handle.qualified_id for handle in handles if handle not in params]
        if missing:
            raise ValueError(
                "dynamic field boundary parameter pack is incomplete: %s" % ", ".join(missing)
            )
        from pops.solvers._numeric import native_float

        values = [
            native_float(
                params[handle], where="dynamic field boundary parameter %s" % handle.qualified_id
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
        """Stage one exact external input; names and block-local component indices are invalid."""
        self.stage_auxiliary_input(key, field)

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
