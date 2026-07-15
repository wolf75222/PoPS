"""Private parameter-aware AMR threshold reference resolution."""
from __future__ import annotations

from typing import Any
from collections.abc import Callable


def resolve_refine_threshold(
    threshold: Any,
    resolver: Any,
    resolve_handle: Callable[[Any, Any, str], Any],
) -> Any:
    """Resolve owned parameter reads and reject ownerless declarations."""
    from pops._ir import ValueExpr
    from pops.model import ParamHandle
    from pops.params import ParameterDeclaration

    if isinstance(threshold, ParamHandle):
        return resolve_handle(threshold, resolver, "Refine threshold")
    if isinstance(threshold, ValueExpr):
        resolved = threshold.resolve_references(resolver)
        if not isinstance(resolved, ValueExpr):
            raise TypeError("Refine threshold resolution must preserve ValueExpr")
        if not isinstance(resolved.handle, ParamHandle):
            raise TypeError("Refine threshold ValueExpr must read a ParamHandle")
        return resolved
    if isinstance(threshold, ParameterDeclaration):
        raise ValueError(
            "Refine runtime threshold %r has no owner; register it with "
            "Problem.param(...) and pass its ParamHandle or Problem.value(handle)"
            % threshold.name
        )
    return threshold


__all__ = ["resolve_refine_threshold"]
