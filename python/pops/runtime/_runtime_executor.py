"""Narrow execution-provider seam for the unified runtime instance.

The selected provider is derived from the normalized ``LayoutPlan`` capabilities.  Compile target
strings and public ``System``/``AmrSystem`` classes are not runtime dispatch authorities.
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
    def install(self, install_plan: Any) -> Any:
        raise NotImplementedError


def _adaptive(plan: Any) -> bool | None:
    values = {row.adaptive for row in plan.artifact.layout_plan.layouts}
    if len(values) != 1:
        return None
    return next(iter(values))


def _require_native_geometry(plan: Any) -> None:
    """The current native facade owns one geometry; aliases may share it exactly."""
    snapshots = [row.descriptor_snapshot for row in plan.artifact.layout_plan.layouts]
    if snapshots and any(value != snapshots[0] for value in snapshots[1:]):
        raise NotImplementedError(
            "RuntimeInstance received heterogeneous native geometries; the LayoutPlan is retained "
            "exactly, but this installed native provider cannot execute them in one kernel domain")


class _UniformNativeProvider(RuntimeExecutorProvider):
    def supports(self, install_plan: Any) -> bool:
        return _adaptive(install_plan) is False

    def install(self, install_plan: Any) -> Any:
        plan = require_install_plan(install_plan)
        _require_native_geometry(plan)
        from pops.runtime.system import System
        from pops.runtime._runtime_mesh_lowering import system_config_from_layout

        engine = System(system_config_from_layout(plan.layout))
        engine._execution_context = plan.execution_context
        artifact = plan.artifact
        if artifact.program is None:
            raise ValueError("RuntimeInstance uniform execution requires the compiled Program")
        engine._install_compiled(
            artifact,
            instances=plan.instances,
            params=plan.params,
            aux=plan.aux,
            field_plans=artifact.plan.field_plans,
            cadence=None,
            outputs=(),
            diagnostics=(),
        )
        return engine


class _AdaptiveNativeProvider(RuntimeExecutorProvider):
    def supports(self, install_plan: Any) -> bool:
        return _adaptive(install_plan) is True

    def install(self, install_plan: Any) -> Any:
        plan = require_install_plan(install_plan)
        _require_native_geometry(plan)
        from pops.runtime.system import AmrSystem
        from pops.runtime._amr_bind_lowering import amr_config_from_layout
        from pops.runtime._runtime_mesh_lowering import (
            flow_amr_layout,
            flow_bootstrap_tagging,
        )

        artifact = plan.artifact
        if artifact.program is None:
            raise ValueError("RuntimeInstance adaptive execution requires the compiled Program")
        engine = AmrSystem(amr_config_from_layout(
            plan.layout, hierarchy=plan.resolved_hierarchy))
        engine._execution_context = plan.execution_context
        schema = artifact.bind_schema
        by_id = {handle.qualified_id: value for handle, value in plan.initial_values.items()}
        initial_rows = []
        if plan.initial_condition_plan is not None:
            from pops.mesh.amr import AnalyticReprojection

            selections = {
                row.subject.qualified_id: row.method
                for row in plan.bootstrap_plan.selections
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
                        "RuntimeInstance adaptive bootstrap currently accepts state Handles only")
                requirement = physical[subject.qualified_id]
                key = requirement.key.to_data()
                block = subject.block_ref.local_id if subject.block_ref is not None else None
                initial_rows.append((
                    subject.qualified_id,
                    block,
                    by_id.get(subject.qualified_id),
                    key["space"]["name"],
                    key["centering"]["name"],
                    "analytic" if type(selections[subject.qualified_id]) is AnalyticReprojection
                    else "prolong",
                    binding.source.options.to_data(),
                ))
        if plan.bootstrap_plan is None:
            flow_amr_layout(
                engine,
                plan.layout,
                n_blocks=len(plan.instances),
                bind_schema=schema,
                params=plan.params,
            )
        else:
            flow_bootstrap_tagging(engine, plan.bootstrap_plan, plan.params)
        engine._install_compiled(
            compiled=artifact,
            instances=plan.instances,
            params=plan.params,
            aux=plan.aux,
            field_plans=artifact.plan.field_plans,
            cadence=None,
            outputs=(),
            diagnostics=(),
            bind_schema=schema,
            initial_values=tuple(initial_rows),
            bootstrap_plan=plan.bootstrap_plan,
            amr_transfer=plan.amr_transfer,
        )
        return engine


_PROVIDERS: tuple[RuntimeExecutorProvider, ...] = (
    _UniformNativeProvider(),
    _AdaptiveNativeProvider(),
)


def install_runtime_executor(install_plan: Any) -> Any:
    plan = require_install_plan(install_plan)
    matches = tuple(provider for provider in _PROVIDERS if provider.supports(plan))
    if len(matches) != 1:
        raise ValueError(
            "LayoutPlan must select exactly one RuntimeExecutorProvider; matched %d" % len(matches))
    return matches[0].install(plan)


__all__ = ["RuntimeExecutorProvider", "install_runtime_executor"]
