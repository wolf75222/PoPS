"""Exact immutable provider evidence for the generic HLLC/Roe pipeline.

The native routes remain :class:`pops::HLLCFlux` and :class:`pops::RoeFlux`.
This module records *which* model-side provider satisfies those routes and the
typed entropy policy used by Roe.  The evidence survives model compilation so
runtime availability and inspection never infer a provider from a truthy flag.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any

from pops.identity.scalar import scalar_literal


HLLC_FLUID_ROLES = "fluid_roles_v1"
ROE_FLUID_ROLES = "fluid_roles_v1"
ROE_DIRECT_ACTION = "direct_action_v1"
ROE_FLUX_JACOBIAN = "flux_jacobian_v1"

ENTROPY_HARTEN = "harten_v1"
ENTROPY_NONE = "none"
ENTROPY_PROVIDER_OWNED = "provider_owned"


def _exact_positive_delta(value: Any, *, where: str) -> Any:
    try:
        literal = scalar_literal(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise type(exc)("%s: %s" % (where, exc)) from exc
    if literal.unit is not None or literal.target is not None:
        raise TypeError("%s cannot carry a unit or target annotation" % where)
    try:
        exact = literal.to_python()
    except TypeError as exc:
        raise TypeError(
            "%s requires an exact int, Fraction, Decimal, or finite float" % where
        ) from exc
    if not exact > 0:
        raise ValueError("%s must be strictly positive (got %r)" % (where, exact))
    try:
        lowered = float(exact)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OverflowError("%s cannot be represented by native pops::Real" % where) from exc
    if not math.isfinite(lowered) or not lowered > 0.0:
        raise OverflowError("%s underflows or overflows the positive pops::Real range" % where)
    return exact


def _delta_token(value: Any) -> str:
    return json.dumps(
        scalar_literal(value).to_data(), sort_keys=True, separators=(",", ":")
    )


def _delta_from_token(token: Any) -> Any:
    if not isinstance(token, str) or not token:
        raise ValueError("Roe Harten entropy evidence requires a canonical scalar token")
    try:
        data = json.loads(token)
    except (TypeError, ValueError) as exc:
        raise ValueError("Roe entropy delta is not canonical scalar JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("Roe entropy delta must be canonical scalar JSON")
    kind = data.get("kind")
    try:
        if kind == "integer" and set(data) == {"kind", "value"}:
            value: Any = int(data["value"])
        elif kind == "rational" and set(data) == {"kind", "numerator", "denominator"}:
            value = Fraction(int(data["numerator"]), int(data["denominator"]))
        elif kind == "decimal" and set(data) == {"kind", "value"}:
            value = Decimal(data["value"])
        elif kind == "binary64" and set(data) == {"kind", "value"}:
            value = float.fromhex(data["value"])
        else:
            raise ValueError
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("Roe entropy delta has an unsupported scalar encoding") from exc
    value = _exact_positive_delta(value, where="compiled Roe entropy delta")
    if _delta_token(value) != token:
        raise ValueError("Roe entropy delta token is not canonical")
    return value


@dataclass(frozen=True, slots=True)
class RoeEntropyPolicy:
    """Typed entropy correction selected by a Roe model-side provider."""

    kind: str
    delta: Any = None
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if self.kind == ENTROPY_HARTEN:
            if self.delta is None:
                raise ValueError("Harten entropy policy requires delta")
            object.__setattr__(
                self,
                "delta",
                _exact_positive_delta(self.delta, where="Harten.delta"),
            )
            return
        if self.kind == ENTROPY_NONE:
            if self.delta is not None:
                raise ValueError("NoEntropyFix cannot carry delta")
            return
        raise ValueError("unknown Roe entropy policy %r" % (self.kind,))

    @property
    def delta_token(self) -> str | None:
        return _delta_token(self.delta) if self.kind == ENTROPY_HARTEN else None

    def to_data(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind}
        if self.delta_token is not None:
            data["delta"] = json.loads(self.delta_token)
        return data


def Harten(delta: Any = 0.1) -> RoeEntropyPolicy:
    """Harten's quadratic entropy correction with an exact positive ``delta``."""

    return RoeEntropyPolicy(ENTROPY_HARTEN, delta)


def NoEntropyFix() -> RoeEntropyPolicy:
    """Use the unmodified absolute eigenvalue / matrix absolute value."""

    return RoeEntropyPolicy(ENTROPY_NONE)


def require_entropy_policy(value: Any, *, default: RoeEntropyPolicy, where: str) -> RoeEntropyPolicy:
    """Normalize an optional policy while refusing untyped scalar magic."""

    selected = default if value is None else value
    if type(selected) is not RoeEntropyPolicy:
        raise TypeError(
            "%s requires riemann.Harten(delta) or riemann.NoEntropyFix(), got %s"
            % (where, type(selected).__name__)
        )
    return selected


@dataclass(frozen=True, slots=True)
class RiemannProviderEvidence:
    """Detached exact evidence for the model-side HLLC and Roe providers."""

    hllc_provider: str | None = None
    roe_provider: str | None = None
    roe_entropy_policy: str | None = None
    roe_entropy_delta: str | None = None

    def __post_init__(self) -> None:
        if self.hllc_provider not in (None, HLLC_FLUID_ROLES):
            raise ValueError("unknown HLLC provider %r" % (self.hllc_provider,))
        if self.roe_provider not in (
            None,
            ROE_FLUID_ROLES,
            ROE_DIRECT_ACTION,
            ROE_FLUX_JACOBIAN,
        ):
            raise ValueError("unknown Roe provider %r" % (self.roe_provider,))
        if self.roe_provider is None:
            if self.roe_entropy_policy is not None or self.roe_entropy_delta is not None:
                raise ValueError("Roe entropy evidence requires an exact Roe provider")
            return
        if self.roe_provider == ROE_DIRECT_ACTION:
            if self.roe_entropy_policy != ENTROPY_PROVIDER_OWNED:
                raise ValueError("direct-action Roe requires provider_owned entropy evidence")
            if self.roe_entropy_delta is not None:
                raise ValueError("direct-action Roe cannot carry a framework entropy delta")
            return
        if self.roe_entropy_policy == ENTROPY_HARTEN:
            _delta_from_token(self.roe_entropy_delta)
            return
        if self.roe_entropy_policy == ENTROPY_NONE:
            if self.roe_entropy_delta is not None:
                raise ValueError("Roe entropy policy 'none' cannot carry delta")
            return
        raise ValueError(
            "Roe provider %r requires exact harten_v1 or none entropy evidence"
            % self.roe_provider
        )


def _authoring_model(model: Any) -> Any:
    inner = getattr(model, "_dsl", model)
    inner = getattr(inner, "_m", inner)
    if hasattr(inner, "_roe") or hasattr(inner, "_hllc"):
        return inner
    return None


def authoring_provider_evidence(model: Any) -> RiemannProviderEvidence:
    """Derive exact provider evidence from one authoring model, without inference."""

    inner = _authoring_model(model)
    if inner is None:
        return RiemannProviderEvidence()
    hllc_provider = HLLC_FLUID_ROLES if bool(getattr(inner, "_hllc", False)) else None
    providers = (
        bool(getattr(inner, "_roe", False)),
        getattr(inner, "_roe_rows", None) is not None,
        getattr(inner, "_roe_jacobian", None) is not None,
    )
    if sum(providers) > 1:
        raise ValueError("model declares competing Roe providers")
    policy = getattr(inner, "_roe_entropy_policy", None)
    if providers[0]:
        if type(policy) is not RoeEntropyPolicy:
            raise ValueError("fluid-role Roe is missing its typed entropy policy")
        return RiemannProviderEvidence(
            hllc_provider,
            ROE_FLUID_ROLES,
            policy.kind,
            policy.delta_token,
        )
    if providers[1]:
        if policy is not None:
            raise ValueError("direct-action Roe cannot carry a framework entropy policy")
        return RiemannProviderEvidence(
            hllc_provider,
            ROE_DIRECT_ACTION,
            ENTROPY_PROVIDER_OWNED,
            None,
        )
    if providers[2]:
        if type(policy) is not RoeEntropyPolicy:
            raise ValueError("flux-Jacobian Roe is missing its typed entropy policy")
        stored_delta = inner._roe_jacobian.get("entropy_fix")
        expected_delta = policy.delta if policy.kind == ENTROPY_HARTEN else None
        if stored_delta != expected_delta:
            raise ValueError("flux-Jacobian Roe entropy policy disagrees with emitted delta")
        return RiemannProviderEvidence(
            hllc_provider,
            ROE_FLUX_JACOBIAN,
            policy.kind,
            policy.delta_token,
        )
    if policy is not None:
        raise ValueError("Roe entropy policy exists without a Roe provider")
    return RiemannProviderEvidence(hllc_provider=hllc_provider)


def compiled_provider_evidence(model: Any) -> RiemannProviderEvidence:
    """Read and validate detached evidence, including legacy-flag parity."""

    evidence = RiemannProviderEvidence(
        getattr(model, "hllc_provider", None),
        getattr(model, "roe_provider", None),
        getattr(model, "roe_entropy_policy", None),
        getattr(model, "roe_entropy_delta", None),
    )
    if bool(getattr(model, "has_hllc", False)) != (evidence.hllc_provider is not None):
        raise ValueError("CompiledModel has_hllc disagrees with exact HLLC provider evidence")
    if bool(getattr(model, "has_roe", False)) != (evidence.roe_provider is not None):
        raise ValueError("CompiledModel has_roe disagrees with exact Roe provider evidence")
    return evidence


def provider_evidence_of(model: Any) -> RiemannProviderEvidence:
    """Return exact authoring or detached provider evidence; never guess from booleans."""

    if all(hasattr(model, name) for name in ("hllc_provider", "roe_provider")):
        return compiled_provider_evidence(model)
    return authoring_provider_evidence(model)


__all__ = [
    "ENTROPY_HARTEN",
    "ENTROPY_NONE",
    "ENTROPY_PROVIDER_OWNED",
    "HLLC_FLUID_ROLES",
    "ROE_DIRECT_ACTION",
    "ROE_FLUID_ROLES",
    "ROE_FLUX_JACOBIAN",
    "Harten",
    "NoEntropyFix",
    "RiemannProviderEvidence",
    "RoeEntropyPolicy",
    "authoring_provider_evidence",
    "compiled_provider_evidence",
    "provider_evidence_of",
    "require_entropy_policy",
]
