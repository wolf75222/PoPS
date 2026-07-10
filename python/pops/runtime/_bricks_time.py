"""Time policies bricks : implicit / split temporal treatments (Spec-4 PR-F).

The per-block time treatments beyond plain ``Explicit``: ``IMEX`` / ``SourceImplicit`` /
``SourceImplicitBE`` / ``IMEXRK``, the physical ``Role`` enum, and the Schur-condensed source
stage + splitting policies (``CondensedSchur`` / ``Split`` / ``Strang``), plus the mask
normalization helpers. ``pops.runtime.bricks`` re-exports all of them.

This module owns ``Role`` and ``CondensedSchur``; the IMEX / source-implicit policies and the mask
helpers live in ``_bricks_time_imex`` and the operator-splitting policies in ``_bricks_time_split``
(both split out for the 500-line cap, ADC-550, and re-imported here so no public path changes). The
``Split`` transport stage type ``Explicit`` comes from ``_bricks_scheme``.
"""
from __future__ import annotations

from typing import Any

from pops.runtime._numeric import exact_real, optional_positive_int, strict_bool
from pops.runtime.routes import SOURCE_STAGE_ELECTROSTATIC_LORENTZ
from pops.runtime.defaults import PHYSICAL_DEFAULT_ALPHA

# The IMEX / source-implicit policies (IMEX / SourceImplicit / SourceImplicitBE / IMEXRK) and the
# implicit-mask helpers are split into ``_bricks_time_imex`` for the 500-line cap (ADC-550) and
# re-imported here so ``pops.runtime.bricks`` (and pops.__init__) re-export them unchanged.
from pops.runtime._bricks_time_imex import (  # noqa: F401
    IMEX,
    IMEXRK,
    SourceImplicit,
    SourceImplicitBE,
    _norm_implicit,
    _role_to_stable,
)


class Role:
    """PHYSICAL roles of a model's components (cf. VariableRole on the C++ side / variable_roles).

    Lets you address a component by its MEANING in pops.CondensedSchur(density=pops.Role.Density,
    momentum=(pops.Role.MomentumX, pops.Role.MomentumY), energy=pops.Role.Energy) rather than by a literal
    name. The values are the STABLE keys expected by the C++ (role_from_name: snake_case). The
    role -> component RESOLUTION is done on the C++ side (the block reads its own VariableRole): these
    constants serve to EXPRESS the intent in the formula and to validate that a required role is requested.
    """

    Density = "density"
    MomentumX = "momentum_x"
    MomentumY = "momentum_y"
    MomentumZ = "momentum_z"
    Energy = "energy"
    VelocityX = "velocity_x"
    VelocityY = "velocity_y"
    VelocityZ = "velocity_z"
    Pressure = "pressure"
    Temperature = "temperature"
    Scalar = "scalar"


class CondensedSchur:
    """SOURCE stage condensed by Schur (Hoffart et al., arXiv:2510.11808; cf.
    docs/SCHUR_CONDENSATION_DESIGN.md). NAMES the algorithm of the implicit source coupling potential /
    velocity / Lorentz and MAPS the fields onto the block's physical roles. This is the `source=` of an
    pops.Split temporal policy (EXPLICIT / IMPLICIT splitting).

    kind="electrostatic_lorentz" (only one for now) selects ElectrostaticLorentzCondensation:
    the stage assembles the condensed elliptic operator A = I + theta^2 dt^2 alpha rho B^{-1}, solves it
    (MG-preconditioned BiCGStab), reconstructs the velocity v = B^{-1}(v^n - theta dt grad phi) and extrapolates
    to the full step. Everything is in C++ (CondensedSchurSourceStepper, #126): NO per-cell Python callback.

    The block must expose the Density / MomentumX / MomentumY roles (Energy optional) and a B_z field
    (set_magnetic_field) -- a missing role / B_z raises an EXPLICIT error at add_equation. Works for
    a native-brick model as well as for a compiled DSL model that declares these roles (electrons).

    GEOMETRY: wired in CARTESIAN (System(mesh=pops.CartesianMesh(...))) AND in POLAR
    (System(mesh=pops.PolarMesh(...)), ring (r, theta), Track A step 2c). The choice of the condensed stepper
    (cartesian CondensedSchurSourceStepper / polar PolarCondensedSchurSourceStepper) is made on the C++ side
    according to the System geometry: the SAME pops.CondensedSchur(...) is used in both cases. The
    polar counterpart is MULTI-RANK-SAFE (correct collectives under MPI) but the facade still builds
    ONE global box (on the owner rank): correct and bit-identical to single-rank, without
    effective parallelism at this level -- the facade theta decomposition is a dedicated follow-up (update
    audit 2026-06; the old mention "n_ranks>1 raises" was stale).

    WHEN TO USE IT (CondensedSchur GLOBAL vs pops.SourceImplicit LOCAL). CondensedSchur is a
    GLOBAL implicit: it COUPLES the whole domain via the condensed tensor elliptic operator
    (solved by Krylov BiCGStab), for non-local stiff Lorentz / electrostatic coupling. If the
    stiff source is purely LOCAL (couples only the components of a single cell, without spatial
    coupling: relaxation, reactions, friction), prefer pops.SourceImplicit instead: it is cheaper
    and there is then NO elliptic solve to do.

    - ``theta``: theta-scheme in (0, 1] (0.5 = Crank-Nicolson, 1 = backward Euler).
    - ``alpha``: electrostatic coupling constant of the source subsystem
      (d_t(-Lap phi) = -alpha div(rho v)).
    - ``density`` / ``momentum`` / ``energy`` / ``magnetic_field`` / ``potential``: role / field
      descriptors. They EXPRESS the intent; the role -> component resolution is done on the C++ side
      (the block reads its own VariableRole). They accept pops.Role.* (recommended), a stable role name,
      or a variable name of the block. momentum is a pair (x, y).
    - ``krylov_tol`` / ``krylov_max_iters``: tolerance and budget of the stage's Krylov (BiCGStab)
      solve. None (defaults) = historical constants (1e-10; 400 in cartesian, 600 in polar),
      made configurable by the 2026-06 audit (explicit numerical constants).
    """

    def __init__(self, kind: str = "electrostatic_lorentz", theta: Any = 0.5,
                 alpha: Any = PHYSICAL_DEFAULT_ALPHA,
                 density: Any = Role.Density, momentum: Any = (Role.MomentumX, Role.MomentumY),
                 energy: Any = None, magnetic_field: str = "B_z", potential: str = "phi",
                 krylov_tol: Any = None, krylov_max_iters: Any = None,
                 fac_max_iters: Any = None, fac_fine_sweeps: Any = None, fac_tol: Any = None,
                 fac_coarse_rel_tol: Any = None, fac_coarse_cycles: Any = None,
                 fac_verbose: bool = False, n_precond_vcycles: Any = None,
                 polar_precond: Any = None) -> None:
        self.krylov_tol = (0.0 if krylov_tol is None else exact_real(
            krylov_tol, where="CondensedSchur.krylov_tol", minimum=0, minimum_open=True,
            maximum=1, maximum_open=True))
        self.krylov_max_iters = optional_positive_int(
            krylov_max_iters, where="CondensedSchur.krylov_max_iters")
        # ADC-645: Krylov-preconditioner knobs of the stage. n_precond_vcycles = MG V-cycles per
        # BiCGStab-preconditioner application on the CARTESIAN (and AMR) stage; the steppers accept
        # 1 or 2 (None = the historical ONE V-cycle, wire sentinel 0). polar_precond selects the
        # POLAR stage's preconditioner ('radial_line' | 'jacobi'; None = the historical RadialLine,
        # wire sentinel ""). Cross-geometry misuse refuses at the native seam, never silently.
        self.n_precond_vcycles = optional_positive_int(
            n_precond_vcycles, where="CondensedSchur.n_precond_vcycles")
        if n_precond_vcycles is not None and self.n_precond_vcycles not in (1, 2):
            raise ValueError("CondensedSchur: n_precond_vcycles must be 1 or 2 (got %r)"
                             % (n_precond_vcycles,))
        if polar_precond is not None and not isinstance(polar_precond, str):
            raise TypeError("CondensedSchur.polar_precond must be a string or None")
        self.polar_precond = polar_precond if polar_precond is not None else ""
        if polar_precond is not None and self.polar_precond not in ("radial_line", "jacobi"):
            raise ValueError("CondensedSchur: polar_precond must be 'radial_line' or 'jacobi' "
                             "(got %r)" % (polar_precond,))
        # ADC-614: composite-FAC knobs of the MULTI-LEVEL condensed Schur solve on AMR (the coarse
        # uniform stage uses only the Krylov knobs above). None (defaults) = the kFAC* constants,
        # bit-identical; refused out-of-domain (never silently clamped). Inert on the uniform System.
        self.fac_max_iters = optional_positive_int(
            fac_max_iters, where="CondensedSchur.fac_max_iters")
        self.fac_fine_sweeps = optional_positive_int(
            fac_fine_sweeps, where="CondensedSchur.fac_fine_sweeps")
        self.fac_tol = (0.0 if fac_tol is None else exact_real(
            fac_tol, where="CondensedSchur.fac_tol", minimum=0, minimum_open=True,
            maximum=1, maximum_open=True))
        self.fac_coarse_rel_tol = (0.0 if fac_coarse_rel_tol is None else exact_real(
            fac_coarse_rel_tol, where="CondensedSchur.fac_coarse_rel_tol", minimum=0,
            minimum_open=True, maximum=1, maximum_open=True))
        self.fac_coarse_cycles = optional_positive_int(
            fac_coarse_cycles, where="CondensedSchur.fac_coarse_cycles")
        self.fac_verbose = strict_bool(fac_verbose, where="CondensedSchur.fac_verbose")
        if not isinstance(kind, str) or kind != "electrostatic_lorentz":
            raise ValueError(
                "CondensedSchur: kind 'electrostatic_lorentz' (only one supported); got %r" % (kind,))
        theta_exact = exact_real(
            theta, where="CondensedSchur.theta", minimum=0, minimum_open=True, maximum=1)
        # momentum must be a pair (role_x, role_y); a bare string (iterable of characters)
        # is rejected explicitly (otherwise tuple("xy") would give two components by accident).
        if isinstance(momentum, str):
            raise ValueError(
                "CondensedSchur: momentum must be a pair (role_x, role_y), not a string (got %r)"
                % (momentum,))
        try:
            mom = tuple(momentum)
        except TypeError:
            raise ValueError(
                "CondensedSchur: momentum must be a pair (role_x, role_y) (got %r)" % (momentum,))
        if len(mom) != 2:
            raise ValueError(
                "CondensedSchur: momentum must be a pair (role_x, role_y) (got %r)" % (momentum,))
        # Role / field descriptors CARRIED in the C++ ABI (audit wave 2): density /
        # momentum / energy accept an pops.Role.* (stable role name) OR a variable name of the
        # block; the role-or-name -> component resolution is done on the C++ side (set_source_stage,
        # explicit error if not found). The DEFAULTS (canonical roles) keep the bit-identical
        # historical behavior. magnetic_field accepts a canonical aux field name
        # (AUX_CANONICAL: "B_z", "T_e", ...) -> carried aux component. potential stays fixed
        # to "phi" (the stage uses the system Poisson potential; another field would have
        # no solver behind it -> explicit rejection, no silent ignore).
        def _spec(v: Any) -> Any:
            return "" if v is None else str(v)
        # Canonical defaults -> EMPTY strings on the ABI side (the C++ then resolves the canonical
        # roles, historical path strictly unchanged).
        self.density_spec = "" if density == Role.Density else _spec(density)
        self.momentum_x_spec = "" if mom[0] == Role.MomentumX else _spec(mom[0])
        self.momentum_y_spec = "" if mom[1] == Role.MomentumY else _spec(mom[1])
        if energy is None:
            self.energy_spec = ""
        elif energy == Role.Energy:
            self.energy_spec = ""
        else:
            self.energy_spec = _spec(energy)
        if magnetic_field == "B_z":
            self.bz_aux_component = -1  # canonical channel (default, bit-identical)
        else:
            from pops.physics.aux import AUX_CANONICAL
            if magnetic_field not in AUX_CANONICAL:
                raise ValueError(
                    "CondensedSchur: magnetic_field=%r unknown (canonical aux fields: %s)"
                    % (magnetic_field, sorted(AUX_CANONICAL)))
            self.bz_aux_component = int(AUX_CANONICAL[magnetic_field])
        if potential != "phi":
            raise ValueError(
                "CondensedSchur: potential=%r not configurable (the source stage solves the "
                "system Poisson potential phi; another field would have no solver "
                "behind it); leave potential='phi' (default)." % (potential,))
        # Typed source-stage route (ADC-584); str value stays the historical token.
        self.kind = SOURCE_STAGE_ELECTROSTATIC_LORENTZ
        self.theta = theta_exact
        self.alpha = exact_real(alpha, where="CondensedSchur.alpha")
        self.density = density
        self.momentum = mom
        self.energy = energy
        self.magnetic_field = magnetic_field
        self.potential = potential
    def _has_field_overrides(self) -> Any:
        """True if a non-canonical descriptor is requested (AMR: explicit rejection, not wired)."""
        return bool(self.density_spec or self.momentum_x_spec or self.momentum_y_spec
                    or self.energy_spec or self.bz_aux_component >= 0)


# The typed constructor ElectrostaticLorentzSchur(...) for the (currently unique) CondensedSchur
# kind lives in _bricks_typed (Spec 5 sec.14.2.5 typed native-brick constructors), beside the typed
# native boundary bricks, to keep this module under the 500-line cap. pops.runtime.bricks re-exports
# it next to CondensedSchur.


# The operator-splitting policies (Split / Strang) are split into ``_bricks_time_split`` for the
# 500-line cap (ADC-550); import them AFTER CondensedSchur is defined (they require it) so the
# module load order stays acyclic. ``pops.runtime.bricks`` re-exports both unchanged.
from pops.runtime._bricks_time_split import Split, Strang  # noqa: E402,F401
