"""pops.numerics.reconstruction -- the spatial-reconstruction brick catalog (Spec 3 / Spec 5).

FirstOrder / MUSCL / WENO5 / WENO5Z selectors. The slope limiters are catalogued
separately in :mod:`pops.numerics.reconstruction.limiters`.

An external reconstruction is deliberately not exposed by this catalog.  A
reconstruction policy is a compile-time, device-callable Kokkos type; the legacy
external-brick manifest carries only a loaded shared-library id and cannot
authenticate the provider source, stencil extent, or formal order.  Treating that
id as an executable reconstruction would therefore be a host-only illusion on
CUDA and would make halo allocation unverifiable.  A future external route must
enter as an authenticated source component compiled into the generated native
artifact, not through a ``dlopen`` function pointer.

pops::Weno5 IS the WENO5-Z reconstruction (it wraps weno5z()); WENO5 and WENO5Z both
select it. MUSCL is reconstruction-by-limiter and preserves the selected native limiter route.
"""
from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType, SimpleNamespace
from typing import Any

from pops.descriptors import BrickDescriptor
from pops.params.use_sites import ParamUse, resolve_param_use
from .limiters import Minmod, _native_reconstruction_descriptor, limiters

# Spec 5 sec.7 / criterion 11: the GHOST (halo) depth a reconstruction stencil NEEDS, by its
# lowered scheme token. A first-order scheme reads the cell mean (1 ghost); a second-order
# MUSCL/slope-limited reconstruction reads one neighbour (2); the fifth-order WENO5(-Z) stencil
# reads two neighbours each side (3). Keyed on the token so the runtime Spatial brick (which
# carries the lowered token, not the descriptor) can look the requirement up.
REQUIRED_GHOST_DEPTH = MappingProxyType({
    "none": 1,
    "minmod": 2,
    "vanleer": 2,
    "weno5": 3,
})

#: The conservative second-order-MUSCL ghost depth the INSPECTION surface assumes for a memory
#: estimate (``pops.codegen.inspect_compiled._ghost_depth``). NOTE this is NOT a hard block
#: limit: the native runtime GROWS each block's halo to match its reconstruction
#: (``include/pops/runtime/system.hpp`` ``block_n_ghost(lim)`` -> 3 for weno5, 2 for MUSCL), so
#: WENO5 is served today. The ghost-depth validation therefore checks the reconstruction's
#: DECLARED requirement only against an EXPLICITLY-constrained block depth (a fixed / external
#: halo a caller passes), never against this assumption -- rejecting WENO5 by default would be a
#: FALSE POSITIVE that breaks a working problem.
INSPECT_GHOST_DEPTH_ASSUMPTION = max(
    REQUIRED_GHOST_DEPTH[token] for token in ("minmod", "vanleer")
)


def authenticated_reconstruction_route(
    descriptor: Any, *, require_muscl_limiter: bool = False
) -> Any:
    """Authenticate a reconstruction descriptor against the generated native catalogue.

    A scheme spelling alone is not authority: external manifests and user-created descriptors
    cannot borrow a builtin token.  The descriptor must carry the native brick type, an accepted
    category, the exact native C++ entry, and the catalogue-owned stencil contract.
    """
    from pops.runtime.routes import resolve

    if type(descriptor) is not BrickDescriptor:
        raise TypeError(
            "reconstruction selection requires a native BrickDescriptor from "
            "pops.numerics.reconstruction"
        )
    if descriptor.category not in ("reconstruction", "limiter"):
        raise TypeError(
            "reconstruction selection expects a reconstruction / limiter descriptor, got %r"
            % descriptor.category
        )
    scheme = descriptor.scheme
    if not isinstance(scheme, str):
        raise ValueError("reconstruction descriptor has a non-canonical scheme %r" % (scheme,))
    try:
        route = resolve("limiter", scheme, context="reconstruction descriptor")
    except ValueError as error:
        raise ValueError(
            "reconstruction descriptor scheme %r is not a known limiter scheme" % scheme
        ) from error
    if descriptor.brick_type != "native":
        raise ValueError(
            "reconstruction descriptor %r uses brick_type=%r; builtin limiter route %s requires "
            "an authenticated native descriptor. External reconstruction providers require a "
            "source-compiled device-callable Kokkos contract."
            % (descriptor.name, descriptor.brick_type, route.id)
        )
    if descriptor.native_id != route.native_entry:
        raise ValueError(
            "reconstruction descriptor %r claims %s but native_id=%r; generated catalogue "
            "requires %r"
            % (descriptor.name, route.id, descriptor.native_id, route.native_entry)
        )
    expected = {
        "formal_order": route.metadata["formal_order"],
        "ghost_depth": route.metadata["n_ghost"],
        "muscl_compatible": route.metadata["muscl_compatible"],
    }
    carried = descriptor.options
    mismatches = [
        "%s=%r (expected %r)" % (key, carried.get(key), value)
        for key, value in expected.items() if carried.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "reconstruction descriptor %r does not match the generated %s contract: %s"
            % (descriptor.name, route.id, ", ".join(mismatches))
        )
    if require_muscl_limiter:
        if descriptor.category != "limiter" or not route.metadata["muscl_compatible"]:
            raise ValueError(
                "MUSCL(limiter=) requires a catalogue route with muscl_compatible=true; got %s"
                % route.id
            )
    return route


def _weno5(name: str, epsilon: Any = None) -> Any:
    """The WENO5(-Z) descriptor with an optional smoothness regulariser ``epsilon`` (ADC-645).

    ``None`` (the default) keeps the native ``kWenoEpsilon`` literal -- the descriptor options are
    unchanged (omit-when-default) and the emitted stencil is bit-identical. A finite positive value
    is carried in the descriptor options and threaded to the native ``Weno5::eps`` by ``add_block``.
    On AMR, descriptor availability is conditional on the resolved coarse/fine authority: it must
    certify order 5 and ghost depth 3. The builtin capability family selects its conservative
    order-5 route from that resolved requirement; an insufficient external provider is refused
    before artifact creation rather than silently lowering the interface order."""
    from pops.runtime.routes import LIMITER_WENO5

    if epsilon is None:
        return _native_reconstruction_descriptor(
            LIMITER_WENO5, category="reconstruction", name=name)
    where = "reconstruction.%s(epsilon=)" % ("WENO5" if name == "weno5" else "WENO5Z")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float, Decimal, Fraction)):
        raise TypeError("%s must be an exact Python numeric scalar" % where)
    if isinstance(epsilon, float) and not math.isfinite(epsilon):
        raise ValueError("%s must be finite" % where)
    if isinstance(epsilon, Decimal) and not epsilon.is_finite():
        raise ValueError("%s must be finite" % where)
    if epsilon <= 0:
        raise ValueError("%s must be a positive number or None; got %r" % (where, epsilon))
    return _native_reconstruction_descriptor(
        LIMITER_WENO5, category="reconstruction", name=name, epsilon=epsilon)


def _muscl(limiter: Any = None) -> Any:
    """Second-order MUSCL reconstruction with one typed limiter authority.

    The limiter determines the native reconstruction token. Formal order and ghost depth are
    properties of this descriptor; callers never repeat them in an AMR or halo policy.
    """
    selected = Minmod() if limiter is None else limiter
    if isinstance(selected, str) or getattr(selected, "category", None) != "limiter":
        raise TypeError(
            "MUSCL(limiter=) requires a typed limiter descriptor such as Minmod() or VanLeer()"
        )
    route = authenticated_reconstruction_route(selected, require_muscl_limiter=True)
    return _native_reconstruction_descriptor(
        route, category="reconstruction", name="muscl", limiter=selected)


_EXTERNAL_RECONSTRUCTION_ERROR = (
    "reconstruction.User is not an executable PoPS route: reconstruction policies are "
    "compile-time device-callable Kokkos types, while the legacy external-brick manifest "
    "carries only a shared-library id and cannot authenticate provider source, formal_order, "
    "or ghost_depth. Supply a native reconstruction descriptor; an external reconstruction "
    "will require an authenticated source-compiled Kokkos provider contract."
)


class _ReconstructionCatalog(SimpleNamespace):
    """The ready-to-lower reconstruction catalog.

    ``User`` used to fabricate a descriptor that no native lowering could execute.  Keep an
    actionable attribute error for callers migrating from that surface, but do not publish a
    selector whose result would be non-executable.
    """

    def __getattr__(self, name: str) -> Any:
        if name == "User":
            raise AttributeError(_EXTERNAL_RECONSTRUCTION_ERROR)
        raise AttributeError(name)


def _first_order() -> Any:
    from pops.runtime.routes import LIMITER_NONE

    return _native_reconstruction_descriptor(
        LIMITER_NONE, category="reconstruction", name="firstorder")


reconstruction = _ReconstructionCatalog(
    FirstOrder=_first_order,
    MUSCL=_muscl,
    WENO5=lambda epsilon=None: _weno5("weno5", epsilon),
    WENO5Z=lambda epsilon=None: _weno5("weno5z", epsilon),
)


def required_ghost_depth(reconstruction_or_token: Any) -> Any:
    """The ghost depth a reconstruction NEEDS (Spec 5 sec.7 / criterion 11).

    Accepts an authenticated native reconstruction descriptor or a canonical lowered scheme token
    (``"none"`` / ``"minmod"`` / ``"vanleer"`` / ``"weno5"``). Returns ``None`` when the
    requirement is not declared/known -- the caller then does NOT reject (a missing requirement is
    not a known incompatibility; no false positive).
    """
    if isinstance(reconstruction_or_token, str):
        from pops.runtime.routes import resolve

        try:
            route = resolve(
                "limiter", reconstruction_or_token, context="required_ghost_depth")
        except ValueError:
            return None
        return route.metadata["n_ghost"]
    descriptor = reconstruction_or_token
    declared = (getattr(descriptor, "options", None) or {}).get("ghost_depth")
    declared = resolve_param_use(
        declared, ParamUse.STENCIL, where="reconstruction(ghost_depth=)")
    if type(descriptor) is BrickDescriptor:
        return authenticated_reconstruction_route(descriptor).metadata["n_ghost"]
    if isinstance(declared, int) and not isinstance(declared, bool):
        return declared
    return None


def validate_ghost_depth(reconstruction_or_token: Any, available: Any = None,
                         block: Any = None) -> bool:
    """Reject a reconstruction whose DECLARED ghost depth exceeds an EXPLICIT block depth.

    Spec 5 sec.7 / criterion 11: a high-order stencil (WENO5 needs 3 ghost cells) reading past a
    too-thin halo is a correctness bug, so the requirement must be checked before runtime. This
    raises a clear, actionable error when the requirement is KNOWN and exceeds @p available.

    The OVERRIDING discipline is NO FALSE POSITIVE. The native runtime GROWS each block's halo to
    match its reconstruction (``block_n_ghost(lim)`` in include/pops/runtime/system.hpp: 3 for
    weno5, 2 for MUSCL), so WENO5 is served today on a default block. Hence:

    * @p available defaults to ``None`` == "the block halo is allocated to match the scheme" ->
      the check NEVER fires (rejecting WENO5 by default would break a working problem);
    * the check fires ONLY when a caller passes an EXPLICIT @p available that constrains the
      block below the requirement (a fixed / external halo);
    * an undeclared reconstruction (``required_ghost_depth`` is ``None``) is never rejected.

    Args:
        reconstruction_or_token: A reconstruction descriptor or its lowered scheme token.
        available: An EXPLICIT block ghost depth, or ``None`` to defer to the scheme-matched
            runtime allocation (no rejection).
        block: Optional block name, woven into the message ("block 'plasma' has ghost_depth=2").

    Returns:
        bool: ``True`` when the depth is sufficient, undeclared, or scheme-matched.

    Raises:
        ValueError: When the reconstruction's declared ghost depth exceeds an explicit
            @p available.
    """
    needed = required_ghost_depth(reconstruction_or_token)
    if available is None:
        return True  # runtime grows the halo to the authenticated scheme.
    available = resolve_param_use(
        available, ParamUse.GHOST_DEPTH, where="validate_ghost_depth(available=)")
    if needed is None or needed <= available:
        return True
    if isinstance(reconstruction_or_token, str):
        name = reconstruction_or_token.upper()
    else:
        name = getattr(reconstruction_or_token, "name",
                       type(reconstruction_or_token).__name__).upper()
    where = ("block %r has" % block) if block is not None else "the block provides"
    raise ValueError(
        "%s requires ghost_depth >= %d, %s ghost_depth=%d; use a lower-order reconstruction "
        "(pops.numerics.reconstruction.MUSCL()) or a block with a deeper halo."
        % (name, needed, where, available))

# Spec 5: expose the schemes at module scope (``from pops.numerics.reconstruction import MUSCL``).
FirstOrder = reconstruction.FirstOrder
MUSCL = reconstruction.MUSCL
WENO5 = reconstruction.WENO5
WENO5Z = reconstruction.WENO5Z

__all__ = ["reconstruction", "limiters", "FirstOrder", "MUSCL", "WENO5", "WENO5Z",
           "REQUIRED_GHOST_DEPTH", "INSPECT_GHOST_DEPTH_ASSUMPTION", "required_ghost_depth",
           "validate_ghost_depth"]


def __getattr__(name: str) -> Any:
    """Refuse the retired non-executable ``User`` selector with an actionable explanation."""
    if name == "User":
        raise AttributeError(_EXTERNAL_RECONSTRUCTION_ERROR)
    raise AttributeError(name)
