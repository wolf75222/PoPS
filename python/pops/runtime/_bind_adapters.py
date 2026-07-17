"""Final install entry from an exact :class:`InstallPlan` to ``RuntimeInstance``.

The historical Uniform/AMR adapter hierarchy intentionally no longer exists.
Backend providers select from normalized ``LayoutPlan`` capabilities inside
``RuntimeInstance``; this boundary only authenticates bind inputs and constructs
the single public runtime object.
"""
from __future__ import annotations

from typing import Any


def install_plan(install_plan: Any) -> Any:
    """Validate and install one exact plan without target-string dispatch."""
    from pops.codegen._plans import require_install_plan
    from pops.runtime._bind_validation import run_bind_gates
    from pops.runtime._runtime_instance import RuntimeInstance

    plan = require_install_plan(install_plan)
    artifact = plan.artifact
    inputs = plan.bind_inputs
    run_bind_gates(
        artifact,
        plan.layout,
        inputs.initial_state,
        plan.params,
        plan.aux,
        initial_values=plan.initial_values,
        platform_manifest=artifact.platform_manifest,
        execution_context=plan.execution_context,
    )
    return RuntimeInstance(plan)


__all__ = ["install_plan"]
