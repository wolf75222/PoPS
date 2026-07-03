"""Operator-splitting time policies: Lie (``Split``) and Strang (``Strang``).

Split out of :mod:`pops.runtime._bricks_time` for the 500-line cap (ADC-550): the two
explicit-transport / separate-source splitting policies. ``_bricks_time`` imports ``Split`` /
``Strang`` back (after it defines ``CondensedSchur``, which they require) and
``pops.runtime.bricks`` re-exports them, so no public import path changes.

The transport stage type ``Explicit`` comes from ``_bricks_scheme``; the condensed source stage
``CondensedSchur`` comes from ``_bricks_time`` (defined before this module is imported, so the
module load order stays acyclic).
"""

from pops.runtime._bricks_scheme import Explicit
from pops.runtime._bricks_time import CondensedSchur
from pops.runtime.routes import SPLITTING_LIE, SPLITTING_STRANG


class Split:
    """Temporal policy EXPLICIT / IMPLICIT SPLITTING: an EXPLICIT hyperbolic transport stage
    (pops.Explicit, SSPRK) followed by a separate SOURCE stage (cf. docs/SCHUR_CONDENSATION_DESIGN.md
    section 6). This is the OPT-IN of the Schur work: a block that does NOT use pops.Split keeps the
    default path (Explicit / IMEX / SourceImplicit), BIT-IDENTICAL.

    ::

        time=pops.Split(
            hyperbolic=pops.Explicit(ssprk3=True),
            source=pops.CondensedSchur(kind="electrostatic_lorentz", theta=0.5, ...),
        )

    - ``hyperbolic`` : pops.Explicit (the transport; SSPRK2/3, substeps, stride inherit from it).
    - ``source`` : pops.CondensedSchur (the condensed source stage, runs AFTER the transport). Only
      source backend wired for now.
    """

    # kind="explicit": the transport is run by the core explicit path (SSPRK), the condensed source
    # is plugged IN ADDITION via set_source_stage (cf. System.add_equation). The block is therefore
    # NOT IMEX (the local stiff source backward-Euler): its source is the condensed stage, apart.
    def __init__(self, hyperbolic=None, source=None):
        hyperbolic = hyperbolic if hyperbolic is not None else Explicit()
        if not isinstance(hyperbolic, Explicit):
            raise TypeError(
                "Split: hyperbolic must be an pops.Explicit (explicit SSPRK transport); got %r"
                % type(hyperbolic).__name__)
        if source is None:
            raise ValueError(
                "Split: source= is required (the separate source stage); e.g. "
                "pops.Split(hyperbolic=pops.Explicit(), source=pops.CondensedSchur(...))")
        if not isinstance(source, CondensedSchur):
            raise TypeError(
                "Split: source must be an pops.CondensedSchur(...) (only wired source stage); got %r"
                % type(source).__name__)
        self.hyperbolic = hyperbolic
        self.source = source
        # The transport takes the core explicit path: we relay the kind / substeps / stride of
        # the hyperbolic stage (SSPRK2/3). The condensed source is plugged separately (add_equation).
        self.kind = hyperbolic.kind
        self.method = hyperbolic.method
        self.substeps = hyperbolic.substeps
        self.stride = hyperbolic.stride
        # Splitting policy WIRED to the system stepper (set_time_scheme). pops.Split = the "lie"
        # route (Godunov, 1st order): H(dt) then S(dt) once per macro-step, BIT-IDENTICAL to the
        # history; pops.Strang overrides this attribute (cf. below). Typed route (ADC-584).
        self.scheme = SPLITTING_LIE


class Strang(Split):
    """Temporal policy STRANG SPLITTING (symmetric, 2nd order): one macro-step runs
    H(dt/2); S(dt); H(dt/2), where H is the EXPLICIT hyperbolic transport (pops.Explicit, SSPRK)
    and S the separate SOURCE stage (pops.CondensedSchur). This is the 2nd-order extension of pops.Split
    (Lie / Godunov, 1st order): same bricks (SSPRK transport + condensed source stage), only the ORDER
    and the cadence of field solves change.

    ::

        time=pops.Strang(
            hyperbolic=pops.Explicit(ssprk3=True),
            source=pops.CondensedSchur(theta=0.5, alpha=alpha),
        )

    The system stepper RE-SOLVES solve_fields BETWEEN stages (before each half-advance and before
    the source) so that the transport always reads a phi consistent with the current density (the
    SINGLE leading solve_fields, sufficient for Lie or a single transport advance to follow, does not
    suffice for the 2nd Strang half-advance). cf. docs/HOFFART_STEP_SEQUENCE.md and SystemStepper::step_strang.

    ``hyperbolic`` / ``source`` : identical to pops.Split. Wired by add_equation (which plugs the source
    stage AND calls set_time_scheme('strang') on the System)."""

    def __init__(self, hyperbolic=None, source=None):
        super().__init__(hyperbolic=hyperbolic, source=source)
        self.scheme = SPLITTING_STRANG


__all__ = ["Split", "Strang"]
