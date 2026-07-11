"""Central parameter-kind validation for semantic use sites.

Parameter *kind* and parameter *use* are independent decisions.  A runtime
coefficient is perfectly valid in a kernel expression, but the same value cannot
choose an array shape, an AMR refinement ratio, a stencil width, or a compiled
backend.  Those choices have already changed storage or generated code before a
runtime slot can be read.

This module is deliberately independent from the concrete parameter classes.  It
reads their small descriptor protocol (``param_kind`` with ``category`` as the
authoring fallback, plus ``phase`` / ``value`` / ``default``) so the validation
boundary does not grow one ``isinstance`` branch for each consumer.
"""
from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any


class ParamUse(str, Enum):
    """Closed semantic uses whose storage/phase requirements differ."""

    RUNTIME_VALUE = "runtime_value"
    MESH_EXTENT = "mesh_extent"
    SHAPE = "shape"
    MESH_TOPOLOGY = "mesh_topology"
    AMR_HIERARCHY = "amr_hierarchy"
    STENCIL = "stencil"
    GHOST_DEPTH = "ghost_depth"
    ORDER = "order"
    ABI = "abi"
    BACKEND = "backend"
    SCHEDULE = "schedule"
    REGRID_SCHEDULE = "regrid_schedule"


class InvalidParamUseSite(TypeError):
    """A parameter kind or evaluation phase cannot satisfy a consumer."""

    def __init__(
        self,
        message: str,
        *,
        param_kind: str,
        use: ParamUse,
        where: str,
        phase: str | None = None,
    ) -> None:
        super().__init__(message)
        self.param_kind = param_kind
        self.use = use
        self.where = where
        self.phase = phase


# Matrix cells name the action at the consumer boundary.  ``reject`` means the
# value is structurally too late, ``unwrap`` means the declaration must provide a
# concrete compile-time value, and ``preserve`` means a later plan/runtime phase
# owns evaluation and the descriptor must not be silently coerced to a literal.
_STRUCTURAL_ROW = MappingProxyType({
    "runtime": "reject",
    "const": "unwrap",
    "derived": "unwrap_compile",
})

PARAM_USE_MATRIX: Mapping[ParamUse, Mapping[str, str]] = MappingProxyType({
    ParamUse.RUNTIME_VALUE: MappingProxyType({
        "runtime": "preserve",
        "const": "unwrap",
        "derived": "preserve",
    }),
    ParamUse.MESH_EXTENT: _STRUCTURAL_ROW,
    ParamUse.SHAPE: _STRUCTURAL_ROW,
    ParamUse.MESH_TOPOLOGY: _STRUCTURAL_ROW,
    ParamUse.AMR_HIERARCHY: _STRUCTURAL_ROW,
    ParamUse.STENCIL: _STRUCTURAL_ROW,
    ParamUse.GHOST_DEPTH: _STRUCTURAL_ROW,
    ParamUse.ORDER: _STRUCTURAL_ROW,
    ParamUse.ABI: _STRUCTURAL_ROW,
    ParamUse.BACKEND: _STRUCTURAL_ROW,
    ParamUse.SCHEDULE: _STRUCTURAL_ROW,
    ParamUse.REGRID_SCHEDULE: _STRUCTURAL_ROW,
})

_KIND_ALIASES = {
    "runtime": "runtime",
    "runtime_param": "runtime",
    "runtimeparam": "runtime",
    "const": "const",
    "const_param": "const",
    "constparam": "const",
    "derived": "derived",
    "derived_param": "derived",
    "derivedparam": "derived",
}
_COMPILE_PHASES = frozenset(("compile", "compile_time", "compiletime"))
_MISSING = object()


def _token(value: Any) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    if not isinstance(raw, str):
        raw = getattr(value, "name", raw)
    if not isinstance(raw, str):
        return None
    return raw.strip().lower().replace("-", "_")


def _param_kind(value: Any) -> str | None:
    raw = getattr(value, "param_kind", None)
    if raw is None:
        raw = getattr(value, "category", None)
    token = _token(raw)
    return _KIND_ALIASES.get(token) if token is not None else None


def _param_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return repr(name) if isinstance(name, str) and name else repr(value)


def _phase(value: Any) -> str | None:
    return _token(getattr(value, "phase", None))


def _is_param_handle(value: Any) -> bool:
    """Recognise the small identity-only ParamHandle protocol without a model import."""
    return (
        getattr(value, "kind", None) == "parameter"
        and hasattr(value, "param_kind")
        and hasattr(value, "qualified_id")
    )


def _declared_payload(value: Any, *, derived: bool = False) -> Any:
    """Return a declared concrete value without evaluating callbacks or symbolic IR."""
    resolved = getattr(value, "resolved_value", _MISSING)
    if resolved is not _MISSING:
        return resolved

    declared = getattr(value, "value", _MISSING)
    if declared is not _MISSING:
        return declared

    has_default = getattr(value, "has_default", _MISSING)
    default = getattr(value, "default", _MISSING)
    if has_default is True and default is not _MISSING:
        return default
    if has_default is _MISSING and default is not _MISSING and default is not None:
        return default

    # A compile-phase DerivedParam may already carry a literal expression.  Do
    # not call/evaluate arbitrary Python or symbolic graph objects here: their
    # resolver must expose ``resolved_value`` instead.
    if derived:
        expression = getattr(value, "expression", _MISSING)
        if expression is None or isinstance(expression, (bool, int, float, str, tuple)):
            return expression
    return _MISSING


def _coerce_use(use: Any) -> ParamUse:
    if isinstance(use, ParamUse):
        return use
    try:
        return ParamUse(use)
    except (TypeError, ValueError):
        raise TypeError("parameter use must be a ParamUse, got %r" % (use,)) from None


def resolve_param_use(value: Any, use: Any, *, where: str) -> Any:
    """Validate one parameter at a consumer and return its usable representation.

    Non-parameter values pass through unchanged.  Compile-structural consumers
    explicitly unwrap ``ConstParam`` and compile-phase ``DerivedParam`` values;
    they reject ``RuntimeParam`` *before* an ``int``/``float``/``bool`` coercion
    could erase its storage class.  Runtime consumers preserve runtime/derived
    descriptors for later plan evaluation.
    """
    semantic_use = _coerce_use(use)
    if not isinstance(where, str) or not where:
        raise TypeError("parameter use-site where= must be a non-empty string")

    kind = _param_kind(value)
    if kind is None:
        return value

    action = PARAM_USE_MATRIX[semantic_use][kind]
    name = _param_name(value)
    if action == "reject":
        raise InvalidParamUseSite(
            "%s: %s %s cannot be used for %s because that choice is compile-structural; "
            "use ConstParam(...) or a DerivedParam evaluated at compile phase"
            % (where, type(value).__name__, name, semantic_use.value),
            param_kind=kind,
            use=semantic_use,
            where=where,
            phase=_phase(value),
        )
    if action == "preserve":
        return value

    if _is_param_handle(value):
        raise InvalidParamUseSite(
            "%s: ParamHandle %s carries identity and param_kind=%r but no declaration value or "
            "phase; resolve it through its owning ParamRegistry or resolved plan before "
            "compile-structural use %s"
            % (where, name, kind, semantic_use.value),
            param_kind=kind,
            use=semantic_use,
            where=where,
            phase=None,
        )

    phase = _phase(value)
    if action == "unwrap_compile" and phase not in _COMPILE_PHASES:
        detail = "has no explicit phase" if phase is None else "has phase %r" % phase
        raise InvalidParamUseSite(
            "%s: DerivedParam %s %s; %s requires phase=compile"
            % (where, name, detail, semantic_use.value),
            param_kind=kind,
            use=semantic_use,
            where=where,
            phase=phase,
        )

    payload = _declared_payload(value, derived=kind == "derived")
    if payload is _MISSING:
        raise InvalidParamUseSite(
            "%s: %s %s has no concrete value for compile-structural use %s; resolve it before "
            "constructing this descriptor"
            % (where, type(value).__name__, name, semantic_use.value),
            param_kind=kind,
            use=semantic_use,
            where=where,
            phase=phase,
        )
    return payload


def validate_param_use(value: Any, use: Any, *, where: str) -> bool:
    """Validate a use without discarding its resolved value API counterpart."""
    resolve_param_use(value, use, where=where)
    return True


__all__ = [
    "InvalidParamUseSite",
    "PARAM_USE_MATRIX",
    "ParamUse",
    "resolve_param_use",
    "validate_param_use",
]
