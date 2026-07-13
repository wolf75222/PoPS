"""Install resolved runtime authorities before native block construction.

This seam is intentionally protocol-driven: layout selection stays in ``_runtime_executor`` while
authorities describe the data the chosen engine must install.  A provider that cannot execute an
authority rejects it here, before native blocks freeze their configuration.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any


def install_runtime_authorities(engine: Any, install_plan: Any) -> None:
    """Install every pre-build authority carried by one normalized install plan."""
    adaptive = {row.adaptive for row in install_plan.artifact.layout_plan.layouts}
    if adaptive == {False}:
        return
    if adaptive != {True}:
        raise ValueError("runtime authorities require one coherent layout capability")

    execution = install_plan.amr_execution
    protocol = getattr(execution, "runtime_execution_data", None)
    if not callable(protocol):
        raise TypeError("adaptive execution authority must implement runtime_execution_data()")
    first, second = protocol(), protocol()
    if type(first) is not dict or first != second \
            or first.get("schema_version") != 1 \
            or first.get("authority_type") != "amr_execution":
        raise TypeError("AMR runtime_execution_data() must return one deterministic v1 dict")
    if first.get("mode") != "subcycled":
        raise NotImplementedError(
            "the installed native AMR provider executes subcycled levels; synchronous execution "
            "is retained as an exact authority and refused instead of being silently subcycled"
        )
    engine._amr_execution_authority = MappingProxyType(dict(first))

    if install_plan.bootstrap_plan is not None:
        from pops.runtime._runtime_mesh_lowering import flow_bootstrap_tagging

        flow_bootstrap_tagging(
            engine, install_plan.bootstrap_plan, install_plan.params)


__all__ = ["install_runtime_authorities"]
