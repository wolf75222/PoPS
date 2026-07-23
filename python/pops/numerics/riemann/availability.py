"""pops.numerics.riemann.availability -- pre-runtime Riemann-flux refusals (Spec 5 sec.6/sec.7).

A model-aware capability check for the numerical flux descriptors, surfaced through the typed
``available(context)`` / ``validate(context)`` protocol so a mismatch (HLL without signed wave
speeds, HLLC without the star-state structure, Roe without a declared dissipation, an explicit
Euler route on a non-Euler layout) is reported BEFORE the runtime is touched.

SINGLE SOURCE. The refusal logic is NOT duplicated here: :func:`flux_available` /
:func:`flux_validate` delegate to the same route and descriptor-requirement predicates used by
``System.add_equation`` / ``AmrSystem.add_equation`` and only translate a raised ``ValueError``
into a structured :class:`~pops.descriptors.Availability`. Builtin and external descriptors use
the same capability contract.

The predicates read the model from the @p context (a plain ``dict`` with a ``"model"`` /
``"compiled"`` key, or the compiled/authoring model itself). NO FALSE POSITIVE: a context that
carries no model cannot know the capability, so the flux stays :meth:`Availability.yes` -- the
hard gate still fires at install when a real model is present.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Availability

def _model_of(context: Any) -> Any:
    """Extract the compiled / authoring model from a validate/available @p context, or ``None``.

    Accepts a plain ``dict`` carrying the model under ``"model"`` or ``"compiled"``, an object
    exposing a ``.model`` / ``.compiled`` attribute, or the model itself (duck-typed: it exposes
    at least one capability flag the predicates read). ``None`` means "no model in scope" and the
    caller then does NOT refuse (no false positive).
    """
    if context is None:
        return None
    if isinstance(context, dict):
        model = context.get("model", context.get("compiled"))
    else:
        model = getattr(context, "model", None) or getattr(context, "compiled", None)
    if model is None:
        # The context may itself be the model (it carries the capability flags the predicates read).
        capability_flags = ("has_hllc", "has_roe", "has_wave_speeds", "n_vars", "prim_names")
        if any(hasattr(context, flag) for flag in capability_flags):
            return context
        return None
    return model


def flux_validate(flux: Any, context: Any = None) -> bool:
    """Raise the exact install-time refusal when @p flux cannot serve the model in @p context.

    Delegates to :func:`pops.runtime.routes.check_riemann_requirement_contract` for every
    descriptor-owned model requirement. This is the SAME predicate ``add_equation`` runs. A
    context with no model, or a provider with no model requirement, passes (no false positive).

    Returns ``True`` when the flux is usable; re-raises the predicate's ``ValueError`` otherwise.
    """
    model = _model_of(context)
    if model is None:
        return True
    from pops.numerics.riemann._contract import riemann_capability_contract
    from pops.runtime.routes import check_riemann_requirement_contract

    check_riemann_requirement_contract(
        riemann_capability_contract(flux), model, "validate", flux=flux)
    return True


def flux_available(flux: Any, context: Any = None) -> Any:
    """The structured :class:`Availability` of @p flux against the model in @p context.

    The ``no`` path carries the predicate's precise reason so a report / negative test can read
    the missing capability without catching an exception. A context with no model, or a flux with
    no model requirement, is :meth:`Availability.yes`.
    """
    model = _model_of(context)
    try:
        flux_validate(flux, context)
    except ValueError as err:
        from pops.numerics.riemann._contract import riemann_capability_contract
        from pops.runtime.routes import riemann_missing_capabilities
        missing = riemann_missing_capabilities(riemann_capability_contract(flux), model)
        alternatives = ["pops.numerics.riemann.Rusanov()"]
        return Availability.no(str(err), missing=missing, alternatives=alternatives)
    return Availability.yes()


__all__ = ["flux_available", "flux_validate"]
