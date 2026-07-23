"""Values consumed only at the native-engine boundary.

Inter-species couplings, the lowered ``Spatial`` value and the plain ``Explicit`` engine policy
live here. Public spatial and temporal authoring lives in :mod:`pops.numerics` and
:mod:`pops.lib.time`; this module deliberately provides no competing finite-volume constructor.
Private native-engine adapters use :mod:`pops.runtime._engine_descriptors` as their cycle-safe
aggregate.
"""
from __future__ import annotations

from typing import Any

from pops.runtime._numeric import exact_real, positive_int, strict_bool
from pops.runtime.routes import (
    RECON_CONSERVATIVE, RECON_PRIMITIVE,
    RIEMANN_EULER_HLLC, RIEMANN_EULER_ROE,
    RIEMANN_HLL, RIEMANN_HLLC, RIEMANN_ROE, RIEMANN_RUSANOV,
    TIME_EULER, TIME_EXPLICIT, TIME_SSPRK3,
)


# --- Inter-species couplings (operator-split): objects passed to sim.add_coupling ---
class Ionization:
    """Ionization n_g -> n_i + n_e (rate k n_e n_g). Mass transferred from the neutral to the ion."""

    def __init__(self, electron: Any, ion: Any, neutral: Any, rate: Any) -> None:
        self.electron = electron
        self.ion = ion
        self.neutral = neutral
        self.rate = exact_real(rate, where="Ionization.rate")


class Collision:
    """Inter-species friction: force k (u_a - u_b), momentum conserved. Fluid blocks (>= 3 var)."""

    def __init__(self, a: Any, b: Any, rate: Any) -> None:
        self.a = a
        self.b = b
        self.rate = exact_real(rate, where="Collision.rate")


class ThermalExchange:
    """Thermal exchange k (T_a - T_b), energy conserved. Euler blocks (4 var)."""

    def __init__(self, a: Any, b: Any, rate: Any) -> None:
        self.a = a
        self.b = b
        self.rate = exact_real(rate, where="ThermalExchange.rate")


# --- Spatial scheme + time treatment (per block) ------------------------
# Spec 5 sec.7: the spatial scheme is chosen with TYPED descriptors, never bare strings. The
# Generated limiter descriptors and the bounded flux/variables tables below lower each accepted
# selector to the TYPED NATIVE ROUTE (ADC-584) the lowering layer carries -- a
# pops.runtime.routes.Route, whose str value IS the canonical token
# the C++ ABI consumes, so the wire crossing stays byte-identical while the identity/requirements
# become typed. reject_string_selector names the typed alternative a rejected string should point
# at. The descriptor category gates which slot a descriptor may fill (a riemann flux in the
# limiter slot is a clear error, not a silent swap).
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


def _lower_reconstruction_selector(value: Any) -> Any:
    """Lower only a catalogue-authenticated native reconstruction descriptor."""
    from pops.descriptors import reject_string_selector
    from pops.numerics.reconstruction import authenticated_reconstruction_route

    if value is None:
        return None
    if isinstance(value, str):
        reject_string_selector(value, "limiter", _LIMITER_SUGGEST)  # always raises
    try:
        return authenticated_reconstruction_route(value)
    except TypeError as error:
        category = getattr(value, "category", None)
        scheme = getattr(value, "scheme", None)
        if category not in ("reconstruction", "limiter"):
            raise TypeError(
                "Spatial: limiter expects a reconstruction / limiter descriptor, got a %r "
                "descriptor (%s). Use %s."
                % (category, scheme, _LIMITER_SUGGEST)
            ) from error
        raise


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
    - ``wave_speed_cache``: flux=HLL() + explicit time ONLY. Pre-computes model.wave_speeds once for
      every exact reconstructed face-trace pair, then reuses that interval from both adjacent residual
      cells. Net gain when wave_speeds is expensive (moment hierarchy). It is BIT-IDENTICAL to the
      direct HLL path for FirstOrder(), MUSCL and WENO reconstruction. False (default) = direct path
      unchanged. Wired on the FULL cartesian advance only: refused if flux != HLL(), IMEX time, polar
      geometry, or a staircase/cutcell disc transport mode is active (set_disc_domain / set_geometry_mode).
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False) and name != "_frozen":
            raise RuntimeError(
                "Spatial is frozen by AuthoringSnapshot: cannot change %r; author a new "
                "pops.numerics.FiniteVolume descriptor and resolve/compile again" % name
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_frozen", False):
            raise RuntimeError(
                "Spatial is frozen by AuthoringSnapshot: cannot delete %r" % name
            )
        object.__delattr__(self, name)

    def freeze(self) -> Any:
        """Seal the complete spatial selection after the authoring boundary."""
        object.__setattr__(self, "_frozen", True)
        return self

    def to_data(self) -> dict[str, Any]:
        """Return the closed, exact identity payload of this finite-volume choice.

        Native routes are identified by their canonical wire tokens, never by their richer
        process-local objects.  Real controls remain scalar literals until the explicit native
        lowering boundary, so Decimal, Fraction and binary64 authoring values cannot collapse to
        the same accidental float identity.
        """
        from pops.identity.scalar import scalar_literal

        return {
            "schema_version": 1,
            "family": "finite_volume",
            "reconstruction": str(self.limiter),
            "riemann": {
                "route": str(self.flux),
                "external_id": self.external_flux_id,
                "capability_contract": self.riemann_capability_contract.to_data(),
                **({
                    "external_library_sha256": self.external_flux_library_sha256,
                    "external_abi_key": self.external_flux_abi_key,
                    "external_native_abi_key": self.external_flux_native_abi_key,
                    "external_model_identity": self.external_flux_model_identity,
                } if self.external_flux_id is not None else {}),
            },
            "variables": str(self.recon),
            "positivity_floor": scalar_literal(self.positivity_floor).to_data(),
            "wave_speed_cache": self.wave_speed_cache,
            "waves_provider": self.waves_provider,
            "weno_epsilon": (
                None if self.weno_epsilon is None
                else scalar_literal(self.weno_epsilon).to_data()
            ),
        }

    def identity(self) -> Any:
        """Authenticated stable identity of :meth:`to_data`."""
        from pops.identity import make_identity

        return make_identity("spatial", self.to_data(), schema_version=1)

    def __eq__(self, other: Any) -> bool:
        return type(other) is type(self) and self.to_data() == other.to_data()

    def __init__(self, limiter: Any = None, flux: Any = None, recon: Any = None, *, none: bool = False,
                 minmod: bool = False, vanleer: bool = False, weno5: bool = False, primitive: bool = False,
                 positivity_floor: Any = None, wave_speed_cache: bool = False,
                 reconstruction: Any = None) -> None:
        for label, flag in (("none", none), ("minmod", minmod), ("vanleer", vanleer),
                            ("weno5", weno5), ("primitive", primitive)):
            strict_bool(flag, where="Spatial.%s" % label)
        # Spec 5 sec.14.1 names the reconstruction/limiter slot ``reconstruction=``; keep ``limiter=``
        # working and accept ``reconstruction=`` as an alias (only one of the two at a time).
        if reconstruction is not None:
            if limiter is not None:
                raise TypeError("Spatial: pass limiter= or reconstruction= (the alias), not both")
            limiter = reconstruction
        from pops.numerics.reconstruction import FirstOrder, WENO5
        from pops.numerics.reconstruction.limiters import Minmod, VanLeer

        enabled_limiter_shortcuts = [
            (label, factory)
            for label, flag, factory in (
                ("none", none, FirstOrder),
                ("minmod", minmod, Minmod),
                ("vanleer", vanleer, VanLeer),
                ("weno5", weno5, WENO5),
            )
            if flag
        ]
        if len(enabled_limiter_shortcuts) > 1:
            raise TypeError(
                "Spatial: limiter shortcuts are mutually exclusive (received %s)"
                % ", ".join(label for label, _ in enabled_limiter_shortcuts)
            )
        if enabled_limiter_shortcuts:
            if limiter is not None:
                raise TypeError(
                    "Spatial: pass limiter=/reconstruction= or one limiter shortcut, not both"
                )
            limiter = enabled_limiter_shortcuts[0][1]()
        lim_tok = _lower_reconstruction_selector(limiter)
        flux_tok = _lower_selector(
            flux, param="flux", schemes=_FLUX_SCHEMES,
            suggestion=_FLUX_SUGGEST, categories=("riemann",))
        recon_tok = _lower_selector(
            recon, param="recon", schemes=_RECON_SCHEMES,
            suggestion=_RECON_SUGGEST, categories=("variables",))
        # Preserve the descriptor-owned capability contract across the private runtime lowering.
        # Runtime installation consumes this value; it never recognises a flux class, factory name,
        # or wire token. External C++ descriptors use the same requirements mapping.
        from pops.numerics.riemann._contract import (
            RiemannCapabilityContract,
            riemann_capability_contract,
        )

        self.riemann_capability_contract = (
            RiemannCapabilityContract(tuple(RIEMANN_RUSANOV.requirements))
            if flux is None
            else riemann_capability_contract(flux)
        )
        self.waves_provider = self.riemann_capability_contract.wave_speed_provider
        self.external_flux_id = None
        self.external_flux_library_path = None
        self.external_flux_library_sha256 = None
        self.external_flux_abi_key = None
        self.external_flux_native_abi_key = None
        self.external_flux_model_identity = None
        self.external_flux_supported_layouts = ()
        if flux is not None and not isinstance(flux, str):
            if getattr(flux, "scheme", None) == "user":
                self.external_flux_id = getattr(flux, "name", None)
                options = getattr(flux, "options", None)
                required = {
                    "library_path", "library_sha256", "abi_version", "abi_key", "native_abi_key",
                    "supported_layouts",
                    "model_identity",
                }
                if not isinstance(options, dict) or set(options) != required:
                    raise ValueError(
                        "external Riemann descriptor has no authenticated loaded-library authority; "
                        "create it with pops.lib.load_cpp_library(...) then riemann.User(id)"
                    )
                if options["abi_version"] != 2 \
                        or options["abi_key"] != \
                        "pops.external-riemann/v2;scalar=f64;index=i32;periodicity=xy":
                    raise ValueError("external Riemann descriptor carries an incompatible ABI")
                self.external_flux_library_path = options["library_path"]
                self.external_flux_library_sha256 = options["library_sha256"]
                self.external_flux_abi_key = options["abi_key"]
                self.external_flux_native_abi_key = options["native_abi_key"]
                self.external_flux_supported_layouts = tuple(options["supported_layouts"])
                self.external_flux_model_identity = options["model_identity"]
        # ADC-645 ride-along (mirror of waves_provider): a reconstruction descriptor built with
        # WENO5(epsilon=...) carries the WENO-Z regulariser in options["epsilon"]. None (the default)
        # keeps the native kWenoEpsilon -> nothing forwarded, byte-identical.
        self.weno_epsilon = None
        if limiter is not None and not isinstance(limiter, str):
            self.weno_epsilon = getattr(limiter, "options", {}).get("epsilon")
        if primitive and recon is not None:
            raise TypeError("Spatial: pass recon= or primitive=True, not both")
        if primitive:
            recon_tok = RECON_PRIMITIVE
        # Canonical defaults (mirror the historical minmod + rusanov + conservative). Every slot
        # holds a TYPED Route (ADC-584) whose str value is the historical token, byte-identical.
        if lim_tok is None:
            lim_tok = _lower_reconstruction_selector(Minmod())
        self.limiter = lim_tok
        self.flux = flux_tok if flux_tok is not None else RIEMANN_RUSANOV
        self.recon = recon_tok if recon_tok is not None else RECON_CONSERVATIVE
        self.positivity_floor = (0.0 if positivity_floor is None else exact_real(
            positivity_floor, where="Spatial.positivity_floor", minimum=0))
        self.wave_speed_cache = strict_bool(
            wave_speed_cache, where="Spatial.wave_speed_cache")

    def __str__(self) -> Any:
        # Spec 5 sec.12.1: a SHORT, deterministic one-line summary of the chosen scheme (the
        # default object repr leaks a memory address, so print() was unreadable). Only the
        # non-default knobs (positivity floor, wave-speed cache) are appended, to keep the line
        # tight on the common path. __repr__ is intentionally left as the default for debug.
        body = "limiter=%s, flux=%s, recon=%s" % (self.limiter, self.flux, self.recon)
        if self.positivity_floor:
            body += ", positivity_floor=%s" % self.positivity_floor
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
                    "requirements": list(
                        self.riemann_capability_contract.required_capabilities),
                    "limitations": []}
        return {"limiter": _manifest(self.limiter), "riemann": _manifest(self.flux),
                "recon": _manifest(self.recon)}

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
                 Production compiled models and native blocks both carry this cadence explicitly.
    method     : "ssprk2" (default, Shu-Osher 2-stage order 2) | "ssprk3" (3-stage order 3,
                 less dissipative, to pair with weno5) | "euler" (ForwardEuler, order 1: fidelity
                 to first-order references, validation only). Shortcut ssprk3=True.
    """

    def __init__(self, substeps: int = 1, method: str = "ssprk2", stride: int = 1, *, ssprk3: bool = False) -> None:
        strict_bool(ssprk3, where="Explicit.ssprk3")
        if ssprk3:
            method = "ssprk3"
        if not isinstance(method, str) or method not in ("ssprk2", "ssprk3", "euler"):
            raise ValueError("Explicit: method 'ssprk2' | 'ssprk3' | 'euler' (received %r)" % (method,))
        self.substeps = positive_int(substeps, where="Explicit.substeps")
        self.stride = positive_int(stride, where="Explicit.stride")
        self.method = method
        # kind passed to the compiled facade: the TYPED time route (ADC-584) whose str value is
        # the historical token -- "explicit" (SSPRK2, bit-identical default), "ssprk3" or "euler"
        # (order 1, fidelity to first-order references -- validation, never default).
        self.kind = (TIME_SSPRK3 if method == "ssprk3"
                     else TIME_EULER if method == "euler" else TIME_EXPLICIT)

    def routes(self) -> Any:
        """The typed native routes chosen by this time treatment (ADC-584 inspection)."""
        return {"time": self.kind.manifest()}
