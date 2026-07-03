"""pops.numerics.riemann.availability -- pre-runtime Riemann-flux refusals (Spec 5 sec.6/sec.7).

A model-aware capability check for the numerical flux descriptors, surfaced through the typed
``available(context)`` / ``validate(context)`` protocol so a mismatch (HLL without signed wave
speeds, HLLC without the star-state structure, Roe without a declared dissipation, an explicit
Euler route on a non-Euler layout) is reported BEFORE the runtime is touched.

SINGLE SOURCE. The refusal logic is NOT duplicated here: :func:`flux_available` /
:func:`flux_validate` delegate to the exact install-time predicates
(:func:`pops.runtime.routes.check_riemann_capability` and
:func:`pops.runtime.routes.check_wave_speed_provider`) that ``System.add_equation`` /
``AmrSystem.add_equation`` already call, and only translate a raised ``ValueError`` into a
structured :class:`~pops.descriptors.Availability`. So the descriptor surface and the install
guard can never diverge -- they run the SAME check.

The predicates read the model from the @p context (a plain ``dict`` with a ``"model"`` /
``"compiled"`` key, or the compiled/authoring model itself). NO FALSE POSITIVE: a context that
carries no model cannot know the capability, so the flux stays :meth:`Availability.yes` -- the
hard gate still fires at install when a real model is present.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Availability

# The generic-hooks fluxes and the explicit canonical-Euler routes both refuse through
# check_riemann_capability; HLL refuses through the wave-speed predicate. A rusanov / user flux
# has no model capability requirement, so it is unconditionally available.
_CAPABILITY_FLUXES = ("hllc", "roe", "euler_hllc", "euler_roe")


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


def _scheme_of(flux: Any) -> Any:
    """The lowered flux token of a riemann descriptor (its ``.scheme``), or ``None``."""
    return getattr(flux, "scheme", None)


def flux_validate(flux: Any, context: Any = None) -> bool:
    """Raise the exact install-time refusal when @p flux cannot serve the model in @p context.

    Delegates to :func:`pops.runtime.routes.check_riemann_capability` (hllc / roe / euler_hllc /
    euler_roe) and to :func:`pops.runtime.routes.check_wave_speed_provider` (hll) -- the SAME
    predicates ``add_equation`` runs -- so the descriptor and the install guard never diverge. A
    context with no model, or a rusanov / user / undeclared flux, passes (no false positive).

    Returns ``True`` when the flux is usable; re-raises the predicate's ``ValueError`` otherwise.
    """
    model = _model_of(context)
    if model is None:
        return True
    scheme = _scheme_of(flux)
    if scheme in _CAPABILITY_FLUXES:
        # Lazy: numerics must not import pops.runtime at module scope (acyclic layering).
        from pops.runtime.routes import check_riemann_capability
        check_riemann_capability(scheme, model, "validate")
    elif scheme == "hll":
        from pops.runtime.routes import check_wave_speed_provider
        requested = (getattr(flux, "options", {}) or {}).get("waves")
        provider = None
        if requested is not None:
            from pops.numerics.riemann.waves import provider_of
            actual = provider_of(model)
            provider = actual.kind if actual is not None else None
        # A bare HLL (no pinned provider) still needs the model to emit signed wave speeds; the
        # predicate raises when has_wave_speeds is False. The estimate providers pass any signed
        # source, so a None requested kind + has_wave_speeds True is accepted, matching install.
        check_wave_speed_provider(requested, model, "validate", actual_provider=provider)
    return True


def flux_available(flux: Any, context: Any = None) -> Any:
    """The structured :class:`Availability` of @p flux against the model in @p context.

    The ``no`` path carries the predicate's precise reason so a report / negative test can read
    the missing capability without catching an exception. A context with no model, or a flux with
    no model requirement, is :meth:`Availability.yes`.
    """
    try:
        flux_validate(flux, context)
    except ValueError as err:
        scheme = _scheme_of(flux)
        missing = {"hllc": ["hllc_star_state"], "roe": ["roe_dissipation"],
                   "euler_hllc": ["euler_2d_layout"], "euler_roe": ["euler_2d_layout"],
                   "hll": ["wave_speeds"]}.get(scheme, [])
        alternatives = ["pops.numerics.riemann.Rusanov()"]
        return Availability.no(str(err), missing=missing, alternatives=alternatives)
    return Availability.yes()


__all__ = ["flux_available", "flux_validate"]
