"""Duck-typed compile-context queries for field validation."""
from __future__ import annotations

from typing import Any


def context_value(context: Any, key: Any) -> Any:
    if context is None:
        return None
    if isinstance(context, dict):
        return context.get(key)
    return getattr(context, key, None)


def context_flag(context: Any, key: Any) -> bool:
    return bool(context_value(context, key))


def context_is_amr_layout(context: Any) -> bool:
    """Return whether a context advertises the AMR layout capability."""
    if context is None:
        return False
    layout = context_value(context, "layout")
    if layout is None:
        layout = context
    capabilities = getattr(layout, "capabilities", None)
    if not callable(capabilities):
        return False
    try:
        declared = capabilities()
    except Exception:  # the absence of a readable capability is not a known incompatibility
        return False
    return hasattr(declared, "get") and declared.get("layout") == "amr"


__all__ = ["context_flag", "context_is_amr_layout", "context_value"]
