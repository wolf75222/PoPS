"""Scheme bricks : inter-species couplings, spatial scheme, explicit time (Spec-4 PR-F).

Inter-species couplings (Ionization / Collision / ThermalExchange), the spatial discretization
(Spatial / FiniteVolume) and the plain ``Explicit`` time treatment. The implicit / split time
policies (IMEX / SourceImplicit / IMEXRK / Implicit / Role / CondensedSchur / Split / Strang)
live in ``_bricks_time`` (split out for the 500-line cap). ``pops.runtime.bricks`` re-exports
these together with the model bricks in ``_bricks_model`` and the time policies in ``_bricks_time``.
"""
from __future__ import annotations

from typing import Any

from pops.runtime.routes import (
    LIMITER_MINMOD, LIMITER_NONE, LIMITER_VANLEER, LIMITER_WENO5,
    RECON_CONSERVATIVE, RECON_PRIMITIVE,
    RIEMANN_EULER_HLLC, RIEMANN_EULER_ROE,
    RIEMANN_HLL, RIEMANN_HLLC, RIEMANN_ROE, RIEMANN_RUSANOV,
    TIME_EULER, TIME_EXPLICIT, TIME_SSPRK3,
    resolve as _resolve_route,
)


# --- Inter-species couplings (operator-split): objects passed to sim.add_coupling ---
class Ionization:
    """Ionization n_g -> n_i + n_e (rate k n_e n_g). Mass transferred from the neutral to the ion."""

    def __init__(self, electron: Any, ion: Any, neutral: Any, rate: Any) -> None:
        self.electron = electron
        self.ion = ion
        self.neutral = neutral
        self.rate = rate


class Collision:
    """Inter-species friction: force k (u_a - u_b), momentum conserved. Fluid blocks (>= 3 var)."""

    def __init__(self, a: Any, b: Any, rate: Any) -> None:
        self.a = a
        self.b = b
        self.rate = rate


class ThermalExchange:
    """Thermal exchange k (T_a - T_b), energy conserved. Euler blocks (4 var)."""

    def __init__(self, a: Any, b: Any, rate: Any) -> None:
        self.a = a
        self.b = b
        self.rate = rate


# --- Spatial scheme + time treatment (per block) ------------------------
# Spec 5 sec.7: the spatial scheme is chosen with TYPED descriptors, never bare strings. The
# tables below map each accepted descriptor scheme to the TYPED NATIVE ROUTE (ADC-584) the
# lowering layer carries -- a pops.runtime.routes.Route, whose str value IS the canonical token
# the C++ ABI consumes, so the wire crossing stays byte-identical while the identity/requirements
# become typed. reject_string_selector names the typed alternative a rejected string should point
# at. The descriptor category gates which slot a descriptor may fill (a riemann flux in the
# limiter slot is a clear error, not a silent swap).
_LIMITER_SCHEMES = {  # reconstruction / limiter descriptor scheme -> Spatial.limiter route
    "none": LIMITER_NONE, "firstorder": LIMITER_NONE,
    "minmod": LIMITER_MINMOD, "vanleer": LIMITER_VANLEER,
    "weno5": LIMITER_WENO5, "weno5z": LIMITER_WENO5,
}
_FLUX_SCHEMES = {  # riemann descriptor scheme -> Spatial.flux route
    # "user" stays a plain token: an EXTERNAL C++ flux brick resolves through the external-brick
    # catalog manifest (pops.descriptors), not the native route registry.
    "rusanov": RIEMANN_RUSANOV, "hll": RIEMANN_HLL, "hllc": RIEMANN_HLLC, "roe": RIEMANN_ROE,
    # Explicit canonical Euler 2D routes (ADC-590): EulerHLLC2D() / EulerRoe2D() descriptors.
    "euler_hllc": RIEMANN_EULER_HLLC, "euler_roe": RIEMANN_EULER_ROE,
    "user": "user",
}
_RECON_SCHEMES = {  # variables descriptor scheme -> Spatial.recon route
    "conservative": RECON_CONSERVATIVE, "primitive": RECON_PRIMITIVE,
}
_LIMITER_SUGGEST = ("pops.numerics.reconstruction.limiters.Minmod() / .VanLeer(), "
                    "pops.numerics.reconstruction.FirstOrder() / WENO5() / MUSCL(...)")
_FLUX_SUGGEST = "pops.numerics.riemann.Rusanov() / HLL() / HLLC() / Roe()"
_RECON_SUGGEST = "pops.numerics.variables.Conservative() / Primitive()"


def _lower_selector(value: Any, *, param: Any, schemes: Any, suggestion: Any, categories: Any) -> Any:
    """Lower a typed spatial-scheme descriptor to its canonical C++ token (Spec 5 sec.7).

    @p value is a typed descriptor (``BrickDescriptor`` / ``Descriptor``) carrying ``.scheme`` and
    ``.category``. A bare ``str`` is REJECTED via :func:`reject_string_selector` -- Spec 5 forbids
    naming a scheme with a string; the message points at the typed @p suggestion. A descriptor of
    the wrong category (a Riemann flux passed for the limiter slot) is a clear ``TypeError``. An
    unknown scheme is rejected rather than silently passed to the C++ boundary.
    """
    from pops.descriptors import reject_string_selector
    if value is None:
        return None
    if isinstance(value, str):
        reject_string_selector(value, param, suggestion)  # always raises
    category = getattr(value, "category", None)
    scheme = getattr(value, "scheme", None)
    if category is None or scheme is None:
        raise TypeError(
            "Spatial: %s must be a typed pops.numerics descriptor (got %r). Use %s."
            % (param, type(value).__name__, suggestion))
    if category not in categories:
        raise TypeError(
            "Spatial: %s expects a %s descriptor, got a %r descriptor (%s). Use %s."
            % (param, " / ".join(categories), category, scheme, suggestion))
    token = schemes.get(scheme)
    if token is None:
        raise ValueError(
            "Spatial: %s descriptor scheme %r is not a known %s scheme (%s). Use %s."
            % (param, scheme, param, ", ".join(sorted(schemes)), suggestion))
    return token


class Spatial:
    """Spatial discretization: reconstruction (limiter) + numerical Riemann flux.

    Spec 5 sec.7: every scheme is chosen with a TYPED ``pops.numerics`` descriptor; a bare string is
    rejected with a message naming the typed object. The boolean shortcuts (none=/minmod=/vanleer=/
    weno5=/primitive=) stay as typed-flag sugar.

    - ``limiter`` (Spec 5 sec.14.1 alias: ``reconstruction``): a reconstruction / limiter descriptor
      lowering to "none" | "minmod" | "vanleer" | "weno5".
      ``pops.numerics.reconstruction.FirstOrder()`` -> none, ``.limiters.Minmod()`` /
      ``.VanLeer()``, ``.WENO5()`` / ``.WENO5Z()`` -> weno5, ``.MUSCL(limiter=...)`` -> its limiter.
      weno5 = WENO5-Z, order 5 in smooth regions, 5-point stencil (3 ghosts), oscillation-free
      capture near a front; only the native ``add_block`` path exposes it (the compiled .so paths
      allocate 2 ghosts -> explicit rejection).
    - ``flux``: a ``pops.numerics.riemann`` descriptor lowering to "rusanov" | "hll" | "hllc" |
      "roe" | "euler_hllc" | "euler_roe".
      Rusanov() = minimal generic (requires only max_wave_speed, any model).
      HLL() = generic with signed waves (requires model.wave_speeds: native isothermal/compressible
      model, or a DSL model declaring a primitive 'p'); less diffusive than rusanov, without
      requiring a pressure or n_vars == 4. This is the recommended path for a NON Euler model with
      signed waves (moment system, isothermal): HLL() + Minmod().
      HLLC() / Roe() = GENERIC-ONLY contact-resolving (HLLC) and Roe-linearized solvers (ADC-590):
      the model MUST supply the hooks HasHLLCStructure / HasRoeDissipation (DSL m.enable_hllc()/
      m.enable_roe(); the native Euler brick provides them). There is no implicit Euler fallback.
      EulerHLLC2D() / EulerRoe2D() = the EXPLICIT canonical 2D Euler routes (4 variables
      rho/rho_u/rho_v/E + ideal-gas pressure), pinning EulerHLLCFlux2D / EulerRoeFlux2D; never a
      fallback.
    - ``recon``: a ``pops.numerics.variables`` descriptor lowering to "conservative" | "primitive"
      (reconstructed variables; primitive more robust for Euler: positivity of rho and p; shortcut
      primitive=).
    - ``positivity_floor``: DENSITY floor of the reconstructed face states (positivity limiter
      Zhang-Shu, ADC-76): conservative scaling of the face state toward the cell mean
      so that rho_face >= floor. 0/None (default) = inactive, bit-identical path.
      Motivated by the top-hat jump of contrast 1e6 in the Hoffart diocotron, where WENO5 reconstructs a
      negative density -> NaN. Requires a model exposing the Density role.
    - ``wave_speed_cache``: flux=HLL() + explicit time ONLY. Pre-computes model.wave_speeds ONCE per
      cell and direction (instead of per face) then bounds each face by min/max of the two neighbor
      cells. Net gain when wave_speeds is expensive (moment hierarchy). With limiter=FirstOrder() +
      recon=Conservative() it is BIT-IDENTICAL to the per-face path; with a 2nd-order+ limiter it is a
      Davis bound on the cell values (different result, opt-in assumed). False (default) = per-face path
      unchanged. Wired on the FULL cartesian advance only: refused if flux != HLL(), IMEX time, polar
      geometry, or a staircase/cutcell disc transport mode is active (set_disc_domain / set_geometry_mode).
    """

    def __init__(self, limiter: Any = None, flux: Any = None, recon: Any = None, *, none: bool = False,
                 minmod: bool = False, vanleer: bool = False, weno5: bool = False, primitive: bool = False,
                 positivity_floor: Any = None, wave_speed_cache: bool = False, reconstruction: Any = None,
                 _tokens: Any = None) -> None:
        # Spec 5 sec.14.1 names the reconstruction/limiter slot ``reconstruction=``; keep ``limiter=``
        # working and accept ``reconstruction=`` as an alias (only one of the two at a time).
        if reconstruction is not None:
            if limiter is not None:
                raise TypeError("Spatial: pass limiter= or reconstruction= (the alias), not both")
            limiter = reconstruction
        # Private fast path: _tokens = (limiter, flux, recon) already-lowered canonical strings.
        # Used by Spatial._from_tokens (the lib-descriptor lowering, whose options are strings).
        # Each token is resolved to its TYPED route (ADC-584): an unknown token is refused here,
        # before the C++ boundary, instead of drifting through as a free string.
        if _tokens is not None:
            lim_tok, flux_tok, recon_tok = _tokens
            if lim_tok is not None:
                lim_tok = _resolve_route("limiter", lim_tok, context="Spatial")
            if flux_tok is not None and flux_tok != "user":
                flux_tok = _resolve_route("riemann", flux_tok, context="Spatial")
            if recon_tok is not None:
                recon_tok = _resolve_route("recon", recon_tok, context="Spatial")
        else:
            lim_tok = _lower_selector(
                limiter, param="limiter", schemes=_LIMITER_SCHEMES,
                suggestion=_LIMITER_SUGGEST, categories=("reconstruction", "limiter"))
            flux_tok = _lower_selector(
                flux, param="flux", schemes=_FLUX_SCHEMES,
                suggestion=_FLUX_SUGGEST, categories=("riemann",))
            recon_tok = _lower_selector(
                recon, param="recon", schemes=_RECON_SCHEMES,
                suggestion=_RECON_SUGGEST, categories=("variables",))
        # Wave-speed provider ride-along (ADC-552): a flux descriptor built with
        # HLL(waves=<WaveSpeedProvider>) carries the provider kind in options["waves"]. Record it
        # on the Spatial so the install guard can cross-check the requested provider against the
        # compiled model's actual wave-speed source (least-invasive: read the descriptor object
        # here, the lowered route token stays byte-identical). None when no provider was pinned.
        self.waves_provider = None
        if _tokens is None and flux is not None and not isinstance(flux, str):
            self.waves_provider = getattr(flux, "options", {}).get("waves")
        # Boolean shortcuts (typed flags, not strings): override the limiter / recon slot. They
        # stay as convenience sugar -- only the bare-string selectors are forbidden (Spec 5 sec.7).
        if none:
            lim_tok = LIMITER_NONE
        elif minmod:
            lim_tok = LIMITER_MINMOD
        elif vanleer:
            lim_tok = LIMITER_VANLEER
        elif weno5:
            lim_tok = LIMITER_WENO5
        if primitive:
            recon_tok = RECON_PRIMITIVE
        # Canonical defaults (mirror the historical minmod + rusanov + conservative). Every slot
        # holds a TYPED Route (ADC-584) whose str value is the historical token, byte-identical.
        self.limiter = lim_tok if lim_tok is not None else LIMITER_MINMOD
        self.flux = flux_tok if flux_tok is not None else RIEMANN_RUSANOV
        self.recon = recon_tok if recon_tok is not None else RECON_CONSERVATIVE
        pf = 0.0 if positivity_floor is None else float(positivity_floor)
        if not (pf >= 0.0):
            raise ValueError("Spatial: positivity_floor >= 0 (0/None = inactive; received %r)"
                             % (positivity_floor,))
        self.positivity_floor = pf
        self.wave_speed_cache = bool(wave_speed_cache)

    def __str__(self) -> Any:
        # Spec 5 sec.12.1: a SHORT, deterministic one-line summary of the chosen scheme (the
        # default object repr leaks a memory address, so print() was unreadable). Only the
        # non-default knobs (positivity floor, wave-speed cache) are appended, to keep the line
        # tight on the common path. __repr__ is intentionally left as the default for debug.
        body = "limiter=%s, flux=%s, recon=%s" % (self.limiter, self.flux, self.recon)
        if self.positivity_floor:
            body += ", positivity_floor=%g" % self.positivity_floor
        if self.wave_speed_cache:
            body += ", wave_speed_cache=True"
        return "Spatial(%s)" % body

    def routes(self) -> Any:
        """The typed native routes chosen by this spatial scheme (ADC-584 inspection).

        A structured dict slot -> route manifest (family, id, token, native entry point,
        requirements, limitations). The ``user`` external-flux token has no native route (it
        resolves through the external-brick catalog manifest) and reports a minimal entry.
        """
        def _manifest(slot_route: Any) -> Any:
            if hasattr(slot_route, "manifest"):
                return slot_route.manifest()
            return {"family": "riemann", "id": "riemann.user", "token": str(slot_route),
                    "native_entry": "external brick (pops.descriptors catalog)",
                    "requirements": [], "limitations": []}
        return {"limiter": _manifest(self.limiter), "riemann": _manifest(self.flux),
                "recon": _manifest(self.recon)}

    @classmethod
    def _from_tokens(cls, limiter: Any, flux: Any, recon: Any, *, positivity_floor: Any = None, wave_speed_cache: bool = False) -> Any:
        """Build a Spatial from ALREADY-canonical string tokens (internal lowering only).

        The spatial brick-catalog descriptor (``pops.numerics.spatial.FiniteVolume``) carries its scheme
        choice as string options; ``System._lower_spatial`` resolves those to the canonical tokens and calls
        this to bypass the typed-descriptor guard. Not part of the public API -- public callers pass
        typed descriptors to ``Spatial`` / ``FiniteVolume``.
        """
        return cls(_tokens=(limiter, flux, recon),
                   positivity_floor=positivity_floor, wave_speed_cache=wave_speed_cache)

    def validate(self, ghost_depth: Any = None, block: Any = None) -> Any:
        """Reject a reconstruction whose ghost depth exceeds an EXPLICIT block halo (Spec 5 sec.7).

        The fifth-order WENO5 stencil needs a 3-cell halo; reading past a too-thin halo is a
        correctness bug (criterion 11). This checks the chosen reconstruction's DECLARED
        requirement and raises a clear, actionable error when an EXPLICIT @p ghost_depth
        constrains the block below it.

        The discipline is NO FALSE POSITIVE. The native runtime GROWS each block's halo to match
        its reconstruction (``block_n_ghost(lim)``: 3 for weno5), so WENO5 on a default block is
        a VALID problem -- and ``ghost_depth=None`` (the default) means exactly that and never
        rejects. A MUSCL / minmod / vanleer scheme (requirement <= 2) passes at any depth >= 2,
        and an undeclared reconstruction is never rejected.

        Args:
            ghost_depth: An EXPLICIT block ghost (halo) depth to check against, or ``None`` to
                defer to the scheme-matched runtime allocation (no rejection).
            block: Optional block name woven into the error message.

        Returns:
            bool: ``True`` when the reconstruction fits the (explicit or scheme-matched) depth.

        Raises:
            ValueError: When an explicit @p ghost_depth is below the reconstruction's requirement.
        """
        from pops.numerics.reconstruction import validate_ghost_depth

        available = None if ghost_depth is None else int(ghost_depth)
        return validate_ghost_depth(self.limiter, available=available, block=block)


def FiniteVolume(*args: Any, **kwargs: Any) -> Any:
    """Re-export of the composite finite-volume surface homed in ``pops.numerics.spatial`` (ADC-533).

    Spec 5 criterion 7 homes the ``FiniteVolume(riemann=HLL(...), reconstruction=MUSCL(...))``
    composite in :mod:`pops.numerics.spatial`; this site re-exports it so every existing
    ``pops.FiniteVolume`` / ``pops.runtime._bricks_scheme.FiniteVolume`` import path keeps working.
    It returns a :class:`Spatial` (consumed as-is by add_block / add_equation). The lazy import
    keeps ``pops.numerics`` free of a module-scope ``pops.runtime`` edge (acyclic layering)."""
    from pops.numerics.spatial import FiniteVolume as _FiniteVolume
    return _FiniteVolume(*args, **kwargs)


class Explicit:
    """Explicit time treatment.

    substeps=N: the block advances N times per macro-step, each substep of length dt/N
                 (fast electrons: substeps=10). Default 1 = historical behavior.
    stride=M   : block cadence, HOLD-THEN-CATCH-UP semantics (catch-up at the END of the window).
                 The block is HELD (not advanced) while (macro_step + 1) % M != 0, then advances by an
                 effective step M*dt at the macro-step where (macro_step + 1) % M == 0, i.e. at the end of each
                 window of M macro-steps (slow block, e.g. neutrals: stride=20). It thus stays
                 temporally CONSISTENT with the fast blocks (never advanced "into the future"). Default
                 1 = every macro-step, bit-identical to the historical behavior. substeps and stride are ORTHOGONAL:
                 stride=M, substeps=N -> N substeps of M*dt/N once at the end of the window.
                 POISSON COUPLING: between two catch-ups, the held block contributes to the right-hand side of the
                 system Poisson (and to the coupled sources) with its STALE state -- its last advanced
                 density/charge, frozen until the next catch-up. step_cfl honors the cadence: the stable
                 step includes the stride factor (dt <= cfl*h*substeps / (stride*w)).
                 NB: the 'aot' backend (System.add_equation on a CompiledModel backend='aot') does NOT
                 carry the cadence and REJECTS stride > 1 (explicit path, no silent ignore);
                 add_block (native) and backend='production' support the stride.
    method     : "ssprk2" (default, Shu-Osher 2-stage order 2) | "ssprk3" (3-stage order 3,
                 less dissipative, to pair with weno5) | "euler" (ForwardEuler, order 1: fidelity
                 to first-order references, validation only). Shortcut ssprk3=True.
    """

    def __init__(self, substeps: int = 1, method: str = "ssprk2", stride: int = 1, *, ssprk3: bool = False) -> None:
        if ssprk3:
            method = "ssprk3"
        if method not in ("ssprk2", "ssprk3", "euler"):
            raise ValueError("Explicit: method 'ssprk2' | 'ssprk3' | 'euler' (received %r)" % (method,))
        if int(substeps) < 1:
            raise ValueError("Explicit: substeps >= 1 (received %r)" % (substeps,))
        if int(stride) < 1:
            raise ValueError("Explicit: stride >= 1 (received %r)" % (stride,))
        self.substeps = int(substeps)
        self.stride = int(stride)
        self.method = method
        # kind passed to the compiled facade: the TYPED time route (ADC-584) whose str value is
        # the historical token -- "explicit" (SSPRK2, bit-identical default), "ssprk3" or "euler"
        # (order 1, fidelity to first-order references -- validation, never default).
        self.kind = (TIME_SSPRK3 if method == "ssprk3"
                     else TIME_EULER if method == "euler" else TIME_EXPLICIT)

    def routes(self) -> Any:
        """The typed native routes chosen by this time treatment (ADC-584 inspection)."""
        return {"time": self.kind.manifest()}
