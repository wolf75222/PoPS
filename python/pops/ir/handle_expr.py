"""Symbolic adapter from Boolean-identity declaration handles to Expr graphs."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .expr import Const, Expr


@runtime_checkable
class _DeclarationHandle(Protocol):
    """Minimal dependency-inversion protocol implemented structurally by model handles."""

    local_id: str
    qualified_id: str
    expression_readable: bool

    def canonical_identity(self) -> dict[str, Any]: ...


class ValueExpr(Expr):
    """Explicit symbolic view of a declaration handle.

    Handle equality stays Boolean; callers opt into semantic algebra through this
    node.  Its generic visitor hooks make it usable by traversal, CSE, lowering and
    optional differentiation without adding a closed-world branch to each visitor.
    """

    def __init__(self, handle: Any) -> None:
        if not isinstance(handle, _DeclarationHandle):
            raise TypeError("ValueExpr requires a declaration Handle")
        if handle.expression_readable is not True:
            raise TypeError(
                "ValueExpr requires a readable value Handle; %s is commit-only or callable"
                % type(handle).__name__)
        self.handle = handle

    def eval(self, env: Any) -> Any:
        key = self.handle.qualified_id
        if key not in env:
            raise KeyError("handle value %r missing from the environment" % key)
        return env[key]

    def deps(self) -> Any:
        return {self.handle.qualified_id}

    def __pops_ir_children__(self) -> tuple:
        return ()

    def __pops_ir_key__(self, recurse: Any) -> Any:
        return ("handle_value", self.handle.qualified_id)

    def __pops_ir_diff__(self, *, recurse: Any, target: Any, definitions: Any) -> Any:
        target_handle = target.handle if isinstance(target, ValueExpr) else target
        if isinstance(target_handle, _DeclarationHandle):
            # Differentiation must use the same LIVE declaration identity as CSE.
            # ``canonical_identity`` intentionally omits the authoring token for
            # reproducible manifests; using it here would conflate two distinct
            # same-named owners authored in the same process.
            return Const(1 if self.handle.qualified_id == target_handle.qualified_id else 0)
        return Const(0)

    def to_cpp(self) -> str:
        raise TypeError(
            "ValueExpr has no context-free C++ spelling; lower it through an owner-aware binding")

    def _str(self) -> str:
        return "value(%s)" % self.handle.qualified_id


__all__ = ["ValueExpr"]
