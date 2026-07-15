"""Runtime provider selection for the unified installed instance.

The selected provider is derived from normalized ``LayoutPlan`` capabilities. Compile target
strings and public ``System``/``AmrSystem`` classes are not runtime dispatch authorities. The
multi-layout coordinator lives in :mod:`pops.runtime._multi_layout_executor`; this module owns only
provider selection and the single-layout native installation seams.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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


def _require_supported_execution_context(plan: Any) -> None:
    """Refuse every resource the native engines cannot consume before constructing one."""
    from pops._platform_contracts import ExecutionContext

    context = plan.execution_context
    if type(context) is not ExecutionContext:
        raise TypeError("runtime provider requires an exact ExecutionContext")
    if context.datatype.identity != "float64":
        raise NotImplementedError(
            "native RuntimeInstance providers require exact float64"
        )
    if context.device.identity not in ("host", "cpu") or context.device.handle is not None:
        raise NotImplementedError(
            "native RuntimeInstance providers require host/cpu without a device handle"
        )
    facts = _native_runtime_facts()
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
        try:
            from mpi4py import MPI
        except ImportError as exc:
            raise RuntimeError(
                "MPI_COMM_WORLD execution requires mpi4py for the explicit communicator handle"
            ) from exc
        if not isinstance(communicator.handle, MPI.Comm) or MPI.Comm.Compare(
                communicator.handle, MPI.COMM_WORLD) != MPI.IDENT:
            raise ValueError(
                "the native provider consumes only the exact mpi4py.MPI.COMM_WORLD handle; "
                "custom communicators are unsupported"
            )
        if context.datatype.handle is not MPI.DOUBLE:
            raise ValueError(
                "MPI_COMM_WORLD execution requires the exact mpi4py.MPI.DOUBLE datatype handle"
            )
        if int(communicator.handle.Get_rank()) != int(facts.get("mpi_rank", -1)) or int(
                communicator.handle.Get_size()) != int(facts.get("mpi_ranks", -1)):
            raise ValueError(
                "ExecutionContext MPI_COMM_WORLD does not match the native runtime rank/size"
            )
    else:
        raise NotImplementedError(
            "native RuntimeInstance providers support only serial or exact MPI_COMM_WORLD; got %r"
            % communicator.identity
        )
    backend = str(facts.get("kokkos_backend", "")).lower()
    if any(token in backend for token in ("cuda", "hip", "sycl", "openmptarget")):
        raise NotImplementedError(
            "native Kokkos execution space %r is incompatible with host ExecutionContext"
            % facts.get("kokkos_backend")
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

        _require_native_geometry(plan)
        from pops.runtime._runtime_mesh_lowering import system_config_from_layout
        from pops.runtime._system import System

        config = system_config_from_layout(plan.layout)
        if any(block.boundaries for block in plan.artifact.plan.blocks):
            # GhostProducerPlan is the topology authority. Physical transport boundaries must not
            # leave unrelated native storage periodic through the legacy mesh flag.
            config.periodic = False
        engine = System(config)
        engine._execution_context = plan.execution_context
        from pops.runtime._runtime_authorities import install_runtime_authorities

        install_runtime_authorities(engine, plan)
        artifact = plan.artifact
        if artifact.program is None:
            raise ValueError("RuntimeInstance uniform execution requires the compiled Program")
        engine._install_compiled(
            artifact,
            instances=plan.instances,
            params=plan.params,
            aux=plan.aux,
            field_plans=artifact.plan.field_plans,
            install_plan=plan,
        )
        return engine


class _AdaptiveNativeProvider(RuntimeExecutorProvider):
    def supports(self, install_plan: Any) -> bool:
        return _adaptive(install_plan) is True

    def install(self, install_plan: Any, runtime_plan: Any = None) -> Any:
        del runtime_plan
        plan = require_install_plan(install_plan)
        _require_native_geometry(plan)
        from pops.runtime._amr_bind_lowering import amr_config_from_layout
        from pops.runtime._runtime_mesh_lowering import flow_amr_layout
        from pops.runtime._system import AmrSystem

        artifact = plan.artifact
        if artifact.program is None:
            raise ValueError("RuntimeInstance adaptive execution requires the compiled Program")
        engine = AmrSystem(amr_config_from_layout(plan.layout, hierarchy=plan.resolved_hierarchy))
        engine._execution_context = plan.execution_context
        from pops.runtime._runtime_authorities import install_runtime_authorities

        install_runtime_authorities(engine, plan)
        schema = artifact.bind_schema
        by_id = {handle.qualified_id: value for handle, value in plan.initial_values.items()}
        initial_rows = []
        if plan.initial_condition_plan is not None:
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
        if plan.bootstrap_plan is None:
            flow_amr_layout(
                engine,
                plan.layout,
                n_blocks=len(plan.instances),
                bind_schema=schema,
                params=plan.params,
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
    _require_supported_execution_context(plan)
    matches = tuple(provider for provider in _PROVIDERS if provider.supports(plan))
    if len(matches) != 1:
        raise ValueError(
            "LayoutPlan must select exactly one RuntimeExecutorProvider; matched %d" % len(matches)
        )
    return matches[0].install(plan, runtime_plan)


__all__ = ["RuntimeExecutorProvider", "install_runtime_executor"]
