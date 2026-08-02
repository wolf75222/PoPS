"""Runtime provider selection for the unified installed instance.

The selected provider is derived from normalized ``LayoutPlan`` capabilities. Compile target
strings and public ``System``/``AmrSystem`` classes are not runtime dispatch authorities. The
multi-layout coordinator lives in :mod:`pops.runtime._multi_layout_executor`; this module owns only
provider selection and the single-layout native installation seams.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, cast

from pops.codegen._plans import require_install_plan


class RuntimeExecutorProvider(ABC):
    @abstractmethod
    def supports(self, install_plan: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def install(self, install_plan: Any, runtime_plan: Any = None) -> Any:
        raise NotImplementedError


def _adaptive(plan: Any) -> bool | None:
    values = {row.adaptive for row in plan.artifact.layout_plan.layouts}
    if len(values) != 1:
        return None
    return next(iter(values))


def _require_native_geometry(plan: Any) -> None:
    """Require the one-geometry invariant of a single native facade."""
    snapshots = [row.descriptor_snapshot for row in plan.artifact.layout_plan.layouts]
    if snapshots and any(value != snapshots[0] for value in snapshots[1:]):
        raise NotImplementedError(
            "RuntimeInstance received heterogeneous native geometries; the LayoutPlan is retained "
            "exactly, but this installed native provider cannot execute them in one kernel domain"
        )


def _native_runtime_facts() -> dict[str, Any]:
    from pops.runtime_environment import runtime_environment_report

    return runtime_environment_report()


def _uniform_initial_sources(plan: Any) -> dict[str, dict[str, Any]]:
    initial_plan = plan.initial_condition_plan
    if initial_plan is None:
        return {}
    by_id = {handle.qualified_id: value for handle, value in plan.initial_values.items()}
    result: dict[str, dict[str, Any]] = {}
    for binding in initial_plan.bindings:
        subject = binding.subject
        if subject.kind != "state" or subject.block_ref is None:
            raise NotImplementedError(
                "uniform native initials currently accept block-qualified state Handles only")
        block = subject.block_ref.local_id
        if block in result:
            raise ValueError("uniform native initials contain multiple states for block %r" % block)
        declaration = getattr(subject, "declaration_ref", None)
        space = getattr(declaration, "space", None)
        route = binding.source.options.to_data()
        value = by_id.get(subject.qualified_id)
        if route.get("native_route") == "bound_level_zero" and value is None:
            raise ValueError("uniform BindArray initial source has no authenticated value")
        if route.get("native_route") != "bound_level_zero" and value is not None:
            raise ValueError("uniform analytic initial source cannot be overridden at bind")
        result[block] = {
            "source": route,
            "value": value,
            "space": getattr(space, "layout", None),
            "centering": getattr(space, "centering", None),
        }
    return result


def _require_supported_execution_context(
    plan: Any, native_facts: dict[str, Any] | None = None
) -> None:
    """Refuse every resource the native engines cannot consume before constructing one."""
    from pops._platform_contracts import ExecutionContext

    context = plan.execution_context
    if type(context) is not ExecutionContext:
        raise TypeError("runtime provider requires an exact ExecutionContext")
    if context.datatype.identity != "float64":
        raise NotImplementedError(
            "native RuntimeInstance providers require exact float64"
        )
    facts = _native_runtime_facts() if native_facts is None else native_facts
    expected_device = facts.get("kokkos_device")
    expected_memory = facts.get("field_memory_space")
    expected_backend = facts.get("kokkos_backend")
    expected_shared = facts.get("kokkos_shared_space")
    expected_stream = facts.get("kokkos_stream")
    if context.device.identity != expected_device:
        raise ValueError(
            "ExecutionContext device differs from the installed Kokkos DefaultExecutionSpace"
        )
    spaces = tuple(context.backend.memory_spaces.require("runtime.memory_spaces"))
    if spaces != (expected_memory,):
        raise ValueError("ExecutionContext memory space differs from installed Kokkos SharedSpace")
    exact_capabilities = {
        "execution_backend": expected_backend,
        "shared_space": expected_shared,
        "stream_identity": expected_stream,
    }
    for name, expected in exact_capabilities.items():
        actual = context.backend.capabilities[name].require("runtime.%s" % name)
        if actual != expected:
            raise ValueError("ExecutionContext %s differs from installed Kokkos" % name)
    from pops.runtime._platform_manifest import validate_native_device_resource

    validate_native_device_resource(context)
    communicator = context.communicator
    if communicator.identity == "serial":
        if communicator.handle is not None:
            raise ValueError("the serial ExecutionContext cannot carry a communicator handle")
        if context.datatype.handle is not None:
            raise ValueError("the serial ExecutionContext cannot carry an MPI datatype handle")
        if facts.get("mpi_active") is not False:
            raise NotImplementedError(
                "the serial ExecutionContext requires native MPI to be inactive"
            )
    elif communicator.identity == "MPI_COMM_WORLD":
        if facts.get("mpi_compiled") is not True or facts.get("mpi_active") is not True \
                or facts.get("communicator") != "MPI_COMM_WORLD":
            raise NotImplementedError(
                "MPI_COMM_WORLD execution requires an MPI-enabled native module in an active "
                "MPI world launch"
            )
        from pops._native_collectives import require_world

        native = require_world(communicator.handle)
        if not native.is_float64_datatype(context.datatype.handle):
            raise ValueError(
                "MPI_COMM_WORLD execution requires the native float64 datatype resource"
            )
        if int(native.rank) != int(facts.get("mpi_rank", -1)) or int(
                native.size) != int(facts.get("mpi_ranks", -1)):
            raise ValueError(
                "ExecutionContext MPI_COMM_WORLD does not match the native runtime rank/size"
            )
    else:
        raise NotImplementedError(
            "native RuntimeInstance providers support only serial or exact MPI_COMM_WORLD; got %r"
            % communicator.identity
        )


def _require_runtime_determinism(
    plan: Any, runtime_plan: Any, native_facts: dict[str, Any]
) -> None:
    """Consume the plan's determinism guarantee against current native facts."""
    context = plan.execution_context
    communication = runtime_plan.communication
    planned = runtime_plan.determinism.assumptions
    provider_facts = {
        "rank_count": native_facts.get("mpi_ranks"),
        "device": native_facts.get("kokkos_device"),
        "communicator": native_facts.get("communicator"),
        "execution_backend": native_facts.get("kokkos_backend"),
        "shared_space": native_facts.get("kokkos_shared_space"),
        "stream_identity": native_facts.get("kokkos_stream"),
        "reduction_order": [
            row.identity.token for row in communication.collectives
        ],
        "reduction_strategy": [
            "%s:%s" % (row.operation, row.strategy)
            for row in communication.collectives
        ],
    }
    actual = {}
    for name in planned:
        if name in provider_facts:
            actual[name] = provider_facts[name]
            continue
        proof = context.backend.capabilities.get(name)
        actual[name] = (
            None
            if proof is None or not proof.known
            else proof.require("runtime.%s" % name)
        )
    runtime_plan.determinism.require_assumptions(actual)


def _require_supported_runtime_actions(runtime_plan: Any) -> None:
    """Refuse derived actions for which no native execution owner exists yet."""
    unsupported = (
        ("buffer allocations", runtime_plan.resources.buffers),
        ("cross-memory fences", runtime_plan.communication.fences),
        ("clock joins", runtime_plan.communication.clock_joins),
    )
    for label, rows in unsupported:
        if rows:
            raise NotImplementedError(
                "native RuntimeInstance has no execution owner for planned %s" % label
            )


def _compiled_spatial_ghost_depth(block: Any) -> int:
    spatial = getattr(block, "spatial", None)
    value = (
        spatial.get("ghost_depth")
        if isinstance(spatial, Mapping)
        else getattr(spatial, "ghost_depth", None)
    )
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError(
            "compiled block %r has no exact positive spatial ghost depth"
            % getattr(block, "name", None)
        )
    return value


def _require_single_layout_runtime_plan(plan: Any, runtime_plan: Any) -> None:
    """Require the exact call/layout projection consumed by one native engine."""
    layout_plan = plan.artifact.layout_plan
    if len(layout_plan.layouts) != 1:
        raise ValueError("single-layout native provider requires exactly one resolved layout")
    layout_id = layout_plan.layouts[0].handle.qualified_id
    assignments = {
        row.subject.local_id: (row.subject_id, row.layout.qualified_id)
        for row in layout_plan.assignments
        if row.subject_kind == "block"
    }
    expected_calls = tuple(assignments[block.name] for block in plan.artifact.blocks)
    actual_calls = tuple((row.block_id, row.layout_id) for row in runtime_plan.calls)
    if actual_calls != expected_calls:
        raise ValueError(
            "RuntimePlanBundle calls differ from the single-layout InstallPlan projection"
        )
    if runtime_plan.communication.transfers:
        raise ValueError("single-layout native provider cannot consume layout Transfers")
    if runtime_plan.resources.mapping_provider_ids:
        raise ValueError("single-layout native provider cannot consume mapping providers")
    if any(row.layout_id != layout_id for row in runtime_plan.communication.halos):
        raise ValueError("RuntimePlanBundle halo differs from the installed single layout")
    calls = {row.identity.token: row for row in runtime_plan.calls}
    if len(calls) != len(runtime_plan.calls):
        raise ValueError("RuntimePlanBundle contains duplicate RuntimeCall identities")
    block_names = {subject_id: name for name, (subject_id, _) in assignments.items()}
    compiled = {row.name: row for row in plan.artifact.blocks}
    if set(compiled) != set(assignments):
        raise ValueError("compiled block set differs from the single-layout plan")
    for halo in runtime_plan.communication.halos:
        call = calls.get(halo.call_id)
        if call is None or call.block_id not in block_names:
            raise ValueError("RuntimePlanBundle halo has no installed block owner")
        block = compiled[block_names[call.block_id]]
        available = _compiled_spatial_ghost_depth(block)
        if halo.depth > available:
            raise ValueError(
                "RuntimePlanBundle halo depth %d exceeds compiled block %r ghost depth %d"
                % (halo.depth, block.name, available)
            )


class _UniformNativeProvider(RuntimeExecutorProvider):
    def supports(self, install_plan: Any) -> bool:
        return _adaptive(install_plan) is False

    def install(self, install_plan: Any, runtime_plan: Any = None) -> Any:
        plan = require_install_plan(install_plan)
        if len(plan.artifact.layout_plan.layouts) > 1:
            if runtime_plan is None:
                raise TypeError("multi-layout install requires its authenticated RuntimePlanBundle")
            from pops.runtime._multi_layout_executor import install_multi_layout_uniform

            return install_multi_layout_uniform(plan, runtime_plan)

        _require_single_layout_runtime_plan(plan, runtime_plan)
        _require_native_geometry(plan)
        from pops.runtime._runtime_mesh_lowering import (
            install_uniform_embedded_boundary,
            system_config_from_layout,
        )
        from pops.runtime._system import System

        config = system_config_from_layout(plan.layout)
        engine = System(config)
        cast(Any, engine)._execution_context = plan.execution_context
        normalized_layout, = plan.artifact.layout_plan.layouts
        install_uniform_embedded_boundary(engine, normalized_layout)
        from pops.runtime._runtime_authorities import install_runtime_authorities

        install_runtime_authorities(engine, plan)
        artifact = plan.artifact
        assert artifact.program is not None, \
            "resolved single-layout Uniform artifact lost its compiled Program"
        engine._install_compiled(
            artifact,
            instances=plan.instances,
            params=plan.params,
            aux=plan.aux,
            field_plans=artifact.plan.field_plans,
            install_plan=plan,
            initial_sources=_uniform_initial_sources(plan),
        )
        return engine


class _AdaptiveNativeProvider(RuntimeExecutorProvider):
    def supports(self, install_plan: Any) -> bool:
        return _adaptive(install_plan) is True

    def install(self, install_plan: Any, runtime_plan: Any = None) -> Any:
        plan = require_install_plan(install_plan)
        _require_single_layout_runtime_plan(plan, runtime_plan)
        _require_native_geometry(plan)
        if plan.initial_condition_plan is None or plan.bootstrap_plan is None:
            raise ValueError(
                "adaptive runtime installation requires the resolved InitialConditionPlan and "
                "its authenticated AMR bootstrap plan"
            )
        from pops.runtime._amr_bind_lowering import amr_config_from_layout
        from pops.runtime._system import AmrSystem

        artifact = plan.artifact
        assert artifact.program is not None, \
            "resolved single-layout AMR artifact lost its compiled Program"
        engine = AmrSystem(amr_config_from_layout(plan.layout, hierarchy=plan.resolved_hierarchy))
        engine._execution_context = plan.execution_context
        from pops.runtime._runtime_authorities import install_runtime_authorities

        install_runtime_authorities(engine, plan)
        schema = artifact.bind_schema
        by_id = {handle.qualified_id: value for handle, value in plan.initial_values.items()}
        initial_rows = []
        from pops.mesh._amr import AnalyticReprojection

        selections = {
            row.subject.qualified_id: row.method for row in plan.bootstrap_plan.selections
        }
        physical = {
            requirement.subject.qualified_id: requirement
            for entry in plan.amr_transfer.entries
            for requirement in entry.requirements
            if requirement.materialization == "physical"
        }
        for binding in plan.initial_condition_plan.bindings:
            subject = binding.subject
            if subject.kind != "state":
                raise NotImplementedError(
                    "RuntimeInstance adaptive bootstrap currently accepts state Handles only"
                )
            requirement = physical[subject.qualified_id]
            key = requirement.key.to_data()
            block = subject.block_ref.local_id if subject.block_ref is not None else None
            initial_rows.append(
                (
                    subject.qualified_id,
                    block,
                    by_id.get(subject.qualified_id),
                    key["space"]["name"],
                    key["centering"]["name"],
                    "analytic"
                    if type(selections[subject.qualified_id]) is AnalyticReprojection
                    else "prolong",
                    binding.source.options.to_data(),
                )
            )
        engine._install_compiled(
            compiled=artifact,
            instances=plan.instances,
            params=plan.params,
            aux=plan.aux,
            field_plans=artifact.plan.field_plans,
            bind_schema=schema,
            initial_values=tuple(initial_rows),
            bootstrap_plan=plan.bootstrap_plan,
            amr_transfer=plan.amr_transfer,
            install_plan=plan,
        )
        return engine


_PROVIDERS: tuple[RuntimeExecutorProvider, ...] = (
    _UniformNativeProvider(),
    _AdaptiveNativeProvider(),
)


def install_runtime_executor(install_plan: Any, runtime_plan: Any = None) -> Any:
    plan = require_install_plan(install_plan)
    from pops.runtime._runtime_planning import require_runtime_plan_bundle

    runtime_plan = require_runtime_plan_bundle(plan, runtime_plan)
    _require_supported_runtime_actions(runtime_plan)
    native_facts = _native_runtime_facts()
    _require_runtime_determinism(plan, runtime_plan, native_facts)
    _require_supported_execution_context(plan, native_facts)
    matches = tuple(provider for provider in _PROVIDERS if provider.supports(plan))
    if len(matches) != 1:
        raise ValueError(
            "LayoutPlan must select exactly one RuntimeExecutorProvider; matched %d" % len(matches)
        )
    return matches[0].install(plan, runtime_plan)


__all__ = ["RuntimeExecutorProvider", "install_runtime_executor"]
