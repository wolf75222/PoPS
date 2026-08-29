"""Fail-closed validation for stochastic Program operations.

The current v5 execution-service table deliberately has no RNG service.  A stochastic operation
therefore cannot participate in accepted-step rollback, rank redistribution, regrid, or restart.
Reject it while the Program is still an authoring value instead of allowing a generated artifact
to borrow process-global randomness.
"""
from __future__ import annotations

from typing import Any


_STOCHASTIC_TOKENS = ("rng", "random", "stochastic")
_NONDETERMINISM_CAPABILITY = "nondeterministic"
# ABI v5 has deliberately not made an RNG a ProgramExecutionServices authority.  This is a
# capability boundary, rather than a default-provider choice: enabling a future service requires
# one authenticated descriptor plus transaction, checkpoint, migration, and rank-redistribution
# state.  Until that versioned contract exists, *every* claimed RNG is refused -- including an
# opaque object which says it is one -- so authoring cannot create a false positive capability.
_V5_HAS_TRANSACTIONAL_RNG_SERVICE = False
_RNG_AUTHORITY_ATTRS = (
    "rng", "rng_provider", "random_provider", "stochastic_provider",
    "transactional_rng", "transactional_rng_provider",
)
_NESTED_BLOCKS = (
    "cond_block",
    "body_block",
    "true_block",
    "false_block",
    "apply_block",
    "residual_block",
)


def _walk(values: Any):
    for value in tuple(values or ()):
        yield value
        attrs = getattr(value, "attrs", None) or {}
        for key in _NESTED_BLOCKS:
            nested = attrs.get(key)
            if nested:
                yield from _walk(nested)


def _declares_nondeterminism(value: Any) -> bool:
    attrs = getattr(value, "attrs", None) or {}
    for candidate in attrs.values():
        capabilities = getattr(candidate, "capabilities", None)
        if not callable(capabilities):
            continue
        try:
            supports = getattr(capabilities(), "supports", None)
            if callable(supports) and supports(_NONDETERMINISM_CAPABILITY):
                return True
        except Exception:  # noqa: BLE001 - an opaque descriptor grants no RNG authority
            return True
    return False


def _claims_rng_authority(value: Any) -> bool:
    """Whether an operation explicitly asks an attribute to supply RNG authority.

    A provider object is intentionally not inspected or trusted here.  On ABI v5 there is no
    native service to bind it to the accepted-step transaction, so even a genuine provider cannot
    make the operation lowerable.  Detecting the claim separately makes an opaque provider fail
    deterministically rather than accidentally looking like an ordinary deterministic attribute.
    """
    attrs = getattr(value, "attrs", None) or {}
    return any(key in attrs for key in _RNG_AUTHORITY_ATTRS)


def validate_stochastic_authority(program: Any) -> None:
    """Refuse stochastic execution until a transactional RNG service is resolved.

    This validator intentionally has no positive escape hatch.  Adding one requires extending the
    common execution-service descriptor and the transaction/checkpoint registries together; an
    arbitrary attribute on an IR node is not sufficient authority.
    """
    for value in _walk(getattr(program, "_values", ())):
        operation = getattr(value, "op", None)
        spelling = operation.lower() if isinstance(operation, str) else ""
        stochastic_spelling = any(token in spelling for token in _STOCHASTIC_TOKENS)
        rng_claim = _claims_rng_authority(value)
        if stochastic_spelling or rng_claim or _declares_nondeterminism(value):
            if rng_claim and not _V5_HAS_TRANSACTIONAL_RNG_SERVICE:
                raise ValueError(
                    "stochastic Program operation %r claims an RNG provider, but ABI v5 "
                    "execution services expose no transactional RNG authority" % operation
                )
            raise ValueError(
                "stochastic Program operation %r requires a declared transactional RNG provider; "
                "the v5 execution services expose no RNG authority" % operation
            )


__all__ = ["validate_stochastic_authority"]
