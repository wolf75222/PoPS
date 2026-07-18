"""Shared validation of the authenticated prepared-Krylov IR footprint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops.identity.scalar import CPP_INT_MAX, exact_cpp_int, scalar_literal
from pops.fields._prepared_nullspace_registry import (
    prepared_nullspace_contracts_from_attrs,
)
from pops.solvers._prepared_preconditioner_registry import prepared_preconditioner_provider_from_attrs
from pops.solvers.krylov._prepared_method_registry import (
    PreparedKrylovMethodUse,
    prepared_krylov_method_provider_from_attrs,
)

_FOOTPRINT_KEYS = frozenset({"components", "input_ghosts", "preconditioned"})
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


def _validated_operator_properties(
    attrs: Mapping[str, Any],
    *,
    declared_nullspace: bool,
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
    if declared_nullspace and certificate.positive_definite:
        raise ValueError(
            "solve_linear singular operator cannot be globally positive definite"
        )
    if not declared_nullspace and certificate.positive_definite_on_nullspace_complement:
        raise ValueError(
            "solve_linear complement-positive certificate requires a declared nullspace"
        )
    return certificate.canonical_data()


def validated_prepared_problem_contract(
    attrs: Mapping[str, Any],
    *,
    operator: Any,
) -> dict[str, Any]:
    """Return canonical prepared-problem metadata or reject an unauthenticated IR node."""
    prepared_krylov_method_provider_from_attrs(attrs)

    # Serialized/tampered IR bypasses descriptor construction and Program authoring validation.
    # Authenticate native integer controls again before any C++ token or workspace size is emitted.
    _exact_int(attrs.get("max_iter"), label="max_iter", minimum=1)
    operator_components, _ = _authenticated_operator_footprint(operator)
    components = _exact_int(attrs.get("ncomp"), label="operator component count", minimum=1)
    if components != operator_components:
        raise ValueError("solve_linear component count disagrees with its authenticated operator")
    nullspace_provider, nullspace_contracts = prepared_nullspace_contracts_from_attrs(
        attrs
    )
    declared_nullspace = nullspace_provider.singular
    properties = _validated_operator_properties(
        attrs, declared_nullspace=declared_nullspace
    )
    nullspace_provider.validate_use(
        contracts=nullspace_contracts,
        components=components,
        operator_properties=properties,
        where="solve_linear nullspace provider %r" % nullspace_provider.provider_id,
    )
    nullspace, gauge = nullspace_contracts.detached()
    return {
        "nullspace_contract": nullspace_provider.enveloped_contract(
            nullspace_contracts
        ),
        "gauge_contract": gauge,
        "operator_properties": properties,
    }


def validated_krylov_footprint(attrs: Mapping[str, Any], *, operator: Any) -> dict[str, Any]:
    """Return the exact canonical footprint or reject a malformed/tampered solve node.

    Code emission and inert scratch inspection consume this one validator so neither can coerce
    booleans/strings into plausible counts or silently disagree about method, restart, or actual
    preconditioner presence.
    """
    method_provider = prepared_krylov_method_provider_from_attrs(attrs)

    operator_components, operator_ghosts = _authenticated_operator_footprint(operator)
    components = _exact_int(attrs.get("ncomp"), label="operator component count", minimum=1)
    if components != operator_components:
        raise ValueError("solve_linear component count disagrees with its authenticated operator")
    # Authentication of the problem-level mathematical contract is inseparable from allocation:
    # malformed/tampered metadata must fail before code emission or scratch accounting can proceed.
    problem_contract = validated_prepared_problem_contract(attrs, operator=operator)
    preconditioner_provider = prepared_preconditioner_provider_from_attrs(attrs)
    preconditioned = preconditioner_provider.preconditioned
    preconditioner_provider.validate_use(
        method_provider=method_provider.authority(),
        components=components,
        nullspace_contract=problem_contract["nullspace_contract"],
        where="solve_linear preconditioner %r" % preconditioner_provider.scheme,
    )

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
    footprint_preconditioned = footprint["preconditioned"]
    if not isinstance(footprint_preconditioned, bool):
        raise ValueError("solve_linear Krylov footprint preconditioned must be a boolean")
    if footprint_preconditioned != preconditioned:
        raise ValueError(
            "solve_linear Krylov footprint disagrees with prepared preconditioner presence"
        )

    try:
        rel_tol = scalar_literal(attrs.get("tol")).to_python()
        abs_tol = scalar_literal(attrs.get("abs_tol")).to_python()
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("solve_linear has invalid prepared method scalar controls") from exc
    use = PreparedKrylovMethodUse(
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        max_iterations=_exact_int(attrs.get("max_iter"), label="max_iter", minimum=1),
        components=components,
        input_ghosts=input_ghosts,
        preconditioned=preconditioned,
        operator_properties=problem_contract["operator_properties"],
        declared_nullspace=prepared_nullspace_contracts_from_attrs(attrs)[0].singular,
        method_options=attrs.get("method_options"),
    )
    method_provider.validate_use(use, where="solve_linear prepared method")

    return {
        "components": components,
        "input_ghosts": input_ghosts,
        "preconditioned": preconditioned,
    }
