"""Shared validation of the authenticated prepared-Krylov IR footprint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops._ir.literals import CPP_INT_MAX, PREPARED_GMRES_MAX_RESTART, exact_cpp_int


_KRYLOV_METHODS = frozenset({"cg", "bicgstab", "gmres", "richardson"})
_PRECONDITIONED_METHODS = frozenset({"bicgstab", "gmres"})
_PRECONDITIONERS = frozenset({"identity", "geometric_mg"})
_FOOTPRINT_KEYS = frozenset(
    {
        "components",
        "input_ghosts",
        "restart",
        "preconditioned",
    }
)
_OPERATOR_PROPERTY_KEYS = frozenset(
    {
        "symmetric",
        "positive_definite",
        "positive_definite_on_nullspace_complement",
    }
)


def _exact_int(
    value: Any,
    *,
    label: str,
    minimum: int,
    maximum: int = CPP_INT_MAX,
) -> int:
    return exact_cpp_int(
        value,
        where="solve_linear Krylov footprint %s" % label,
        minimum=minimum,
        maximum=maximum,
    )


def _authenticated_operator_footprint(operator: Any) -> tuple[int, int]:
    """Return the independently authored operator shape consumed by Krylov.

    The solve node duplicates these two values so scratch inspection remains self-contained, but
    that duplicate is not an authority: codegen must bind it back to the typed operator declaration
    before allocating native fields.  Keep the import lazy so importing :mod:`pops.codegen` does not
    eagerly import the time DSL.
    """
    operator_attrs = getattr(operator, "attrs", None)
    if getattr(operator, "op", None) != "matrix_free_operator" or not isinstance(
        operator_attrs, Mapping
    ):
        raise ValueError("solve_linear requires an authenticated matrix_free_operator input")
    operator_components = _exact_int(
        operator_attrs.get("ncomp"), label="operator component count", minimum=1
    )

    from pops.time.stencil import StencilAccess

    stencil_access = operator_attrs.get("stencil_access")
    if type(stencil_access) is not StencilAccess:
        raise ValueError("solve_linear operator has no authenticated StencilAccess")
    operator_ghosts = _exact_int(
        stencil_access.required_ghost_depth, label="operator input_ghosts", minimum=0
    )
    return operator_components, operator_ghosts


def _validated_nullspace_and_gauge(
    attrs: Mapping[str, Any],
    *,
    components: int,
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Authenticate the complete author-declared nullspace/gauge pair.

    This validator deliberately does not inspect a stencil name, boundary condition, or mesh.
    ``LinearProblem`` is the sole mathematical authority; codegen only checks that its canonical
    snapshot reached the IR intact.
    """
    nullspace = attrs.get("nullspace_contract")
    gauge = attrs.get("gauge_contract")
    if (
        not isinstance(nullspace, Mapping)
        or set(nullspace) != {"schema_version", "kind"}
        or type(nullspace.get("schema_version")) is not int
        or nullspace.get("schema_version") != 1
    ):
        raise ValueError("solve_linear requires an exact nullspace_contract")
    nullspace_kind = nullspace.get("kind")
    if nullspace_kind == "none":
        if not isinstance(gauge, Mapping) or dict(gauge) != {"schema_version": 1, "kind": "none"}:
            raise ValueError(
                "solve_linear nonsingular nullspace contract requires gauge_contract=none"
            )
        return (
            {"schema_version": 1, "kind": "none"},
            {"schema_version": 1, "kind": "none"},
            False,
        )
    if nullspace_kind != "constant":
        raise ValueError("solve_linear nullspace_contract kind must be 'none' or 'constant'")
    if components != 1:
        raise ValueError(
            "solve_linear constant nullspace is scalar-only; no component basis is inferred"
        )
    from pops._ir.literals import ScalarLiteral

    if (
        not isinstance(gauge, Mapping)
        or set(gauge) != {"schema_version", "kind", "value"}
        or type(gauge.get("schema_version")) is not int
        or gauge.get("schema_version") != 1
        or gauge.get("kind") != "mean_value"
        or type(gauge.get("value")) is not ScalarLiteral
    ):
        raise ValueError(
            "solve_linear constant nullspace requires an exact MeanValueGauge snapshot"
        )
    return (
        {"schema_version": 1, "kind": "constant"},
        {"schema_version": 1, "kind": "mean_value", "value": gauge["value"]},
        True,
    )


def _validated_operator_properties(
    attrs: Mapping[str, Any],
    *,
    declared_nullspace: bool,
    method: str,
) -> dict[str, bool]:
    properties = attrs.get("operator_properties")
    if not isinstance(properties, Mapping) or set(properties) != _OPERATOR_PROPERTY_KEYS:
        raise ValueError("solve_linear requires exactly three operator-property booleans")
    if any(type(properties[key]) is not bool for key in _OPERATOR_PROPERTY_KEYS):
        raise ValueError("solve_linear operator-property certificates must be exact booleans")
    from pops.linalg import LinearOperatorProperties

    try:
        certificate = LinearOperatorProperties(**dict(properties))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "solve_linear operator properties are incoherent or unauthenticated"
        ) from exc
    if declared_nullspace and not certificate.symmetric:
        raise ValueError(
            "solve_linear constant nullspace requires a symmetric operator certificate"
        )
    if declared_nullspace and certificate.positive_definite:
        raise ValueError(
            "solve_linear constant-nullspace operator cannot be globally positive definite"
        )
    if not declared_nullspace and certificate.positive_definite_on_nullspace_complement:
        raise ValueError(
            "solve_linear complement-positive certificate requires a declared nullspace"
        )
    if method == "cg" and not certificate.certifies_cg(declared_nullspace=declared_nullspace):
        raise ValueError(
            "solve_linear CG operator certificate disagrees with its nullspace contract"
        )
    return certificate.canonical_data()


def validated_prepared_problem_contract(
    attrs: Mapping[str, Any],
    *,
    operator: Any,
) -> dict[str, Any]:
    """Return canonical prepared-problem metadata or reject an unauthenticated IR node."""
    method = attrs.get("method")
    if method not in _KRYLOV_METHODS:
        raise ValueError("solve_linear has an unauthenticated Krylov method %r" % (method,))

    # Serialized/tampered IR bypasses descriptor construction and Program authoring validation.
    # Authenticate native integer controls again before any C++ token or workspace size is emitted.
    _exact_int(attrs.get("max_iter"), label="max_iter", minimum=1)
    operator_components, _ = _authenticated_operator_footprint(operator)
    components = _exact_int(attrs.get("ncomp"), label="operator component count", minimum=1)
    if components != operator_components:
        raise ValueError("solve_linear component count disagrees with its authenticated operator")
    nullspace, gauge, declared_nullspace = _validated_nullspace_and_gauge(
        attrs, components=components
    )
    properties = _validated_operator_properties(
        attrs, declared_nullspace=declared_nullspace, method=method
    )
    return {
        "nullspace_contract": nullspace,
        "gauge_contract": gauge,
        "operator_properties": properties,
    }


def validated_krylov_footprint(attrs: Mapping[str, Any], *, operator: Any) -> dict[str, Any]:
    """Return the exact canonical footprint or reject a malformed/tampered solve node.

    Code emission and inert scratch inspection consume this one validator so neither can coerce
    booleans/strings into plausible counts or silently disagree about method, restart, or actual
    preconditioner presence.
    """
    method = attrs.get("method")
    if method not in _KRYLOV_METHODS:
        raise ValueError("solve_linear has an unauthenticated Krylov method %r" % (method,))

    operator_components, operator_ghosts = _authenticated_operator_footprint(operator)
    components = _exact_int(attrs.get("ncomp"), label="operator component count", minimum=1)
    if components != operator_components:
        raise ValueError("solve_linear component count disagrees with its authenticated operator")
    # Authentication of the problem-level mathematical contract is inseparable from allocation:
    # malformed/tampered metadata must fail before code emission or scratch accounting can proceed.
    validated_prepared_problem_contract(attrs, operator=operator)
    preconditioner = attrs.get("preconditioner")
    if preconditioner not in _PRECONDITIONERS:
        raise ValueError(
            "solve_linear has an unauthenticated prepared preconditioner %r" % (preconditioner,)
        )
    preconditioned = preconditioner != "identity"
    if preconditioned and method not in _PRECONDITIONED_METHODS:
        raise ValueError("solve_linear preconditioning is unavailable for %s" % method)

    raw_restart = attrs.get("restart")
    if method == "gmres":
        restart = _exact_int(
            raw_restart,
            label="GMRES restart (MPI Arnoldi reduction count requires restart + 1)",
            minimum=1,
            maximum=PREPARED_GMRES_MAX_RESTART,
        )
    else:
        if raw_restart is not None:
            raise ValueError(
                "solve_linear restart belongs only to GMRES; got %r for %s" % (raw_restart, method)
            )
        restart = 0

    footprint = attrs.get("krylov_footprint")
    if not isinstance(footprint, Mapping) or set(footprint) != _FOOTPRINT_KEYS:
        raise ValueError("solve_linear requires an exact typed Krylov footprint")
    footprint_components = _exact_int(footprint["components"], label="components", minimum=1)
    if footprint_components != components:
        raise ValueError("solve_linear Krylov footprint component count is unauthenticated")
    input_ghosts = _exact_int(footprint["input_ghosts"], label="input_ghosts", minimum=0)
    if input_ghosts != operator_ghosts:
        raise ValueError(
            "solve_linear Krylov footprint input_ghosts disagrees with its authenticated operator"
        )
    footprint_restart = _exact_int(
        footprint["restart"], label="restart", minimum=0, maximum=PREPARED_GMRES_MAX_RESTART
    )
    if footprint_restart != restart:
        raise ValueError("solve_linear Krylov footprint restart disagrees with method controls")
    footprint_preconditioned = footprint["preconditioned"]
    if not isinstance(footprint_preconditioned, bool):
        raise ValueError("solve_linear Krylov footprint preconditioned must be a boolean")
    if footprint_preconditioned != preconditioned:
        raise ValueError(
            "solve_linear Krylov footprint disagrees with prepared preconditioner presence"
        )

    return {
        "components": components,
        "input_ghosts": input_ghosts,
        "restart": restart,
        "preconditioned": preconditioned,
    }
