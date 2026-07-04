"""pops.solvers.elliptic -- the rich elliptic field-solver descriptors (Spec 5 sec.5.7).

The elliptic solve ``div(eps grad phi) = rhs`` is configured by a TYPED descriptor with a rich
parameter surface, not the bare string ``solver="geometric_mg"``:

* :class:`GeometricMG` -- geometric multigrid, configured by a typed smoother
  (:class:`pops.solvers.options.Chebyshev` / :class:`~pops.solvers.options.RedBlackGaussSeidel`),
  a typed coarse solver (:class:`~pops.solvers.options.DirectSmallGrid`), a typed convergence
  tolerance (:class:`pops.solvers.tolerances.Relative` / :class:`~pops.solvers.tolerances.Absolute`)
  and a V-cycle cap (``max_cycles``). It declares its capabilities (uniform / amr / mpi / gpu /
  variable_epsilon) so an unsupported route is refused before the runtime is touched.
* :class:`FFT` -- the real ``pops::PoissonFFTSolver`` (periodic BC, constant coefficient,
  power-of-two grid); ``available()`` reports ``partial`` for those route constraints (not
  because it is unimplemented) and points at :class:`GeometricMG` for the general case.

Both are inert (Spec 5 sec.6): they record the choice and answer ``available`` / ``lower`` /
``inspect``; the C++ kernel performs the multigrid V-cycles. The ``scheme`` attribute mirrors
the runtime token so the install path's solver-token resolution keeps working.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Availability, Descriptor
from pops.descriptors_report import CapabilitySet, ValidationReport
from pops.solvers.options import Chebyshev, DirectSmallGrid, RedBlackGaussSeidel
from pops.solvers.requirements import capability_map
from pops.solvers.tolerances import Absolute, Relative

_SMOOTHERS = (Chebyshev, RedBlackGaussSeidel)
_COARSE = (DirectSmallGrid,)
_TOLERANCES = (Relative, Absolute)

# ADC-613: the GeometricMG descriptor defaults reconcile to the NATIVE kMG* constants (the ADC-603
# numerical_defaults report), NOT the pre-613 descriptor literals (1e-6 / 20 / Chebyshev). Before
# ADC-613 these options never reached the runtime, so no recorded run ever used the old literals;
# matching the native defaults makes GeometricMG() reproduce today's V-cycle bit-for-bit. pops.solvers
# is a LEAF layer (it must not import pops.runtime, cf. tests/python/architecture/test_import_graph.py),
# so these are literals kept in lockstep with numerical_defaults.hpp by the pin test in
# tests/python/unit/runtime/test_geometric_mg_options.py (effective report == defaults report).
_MG_DEFAULT_REL_TOL = 1e-8
_MG_DEFAULT_MAX_CYCLES = 50
_MG_DEFAULT_MIN_COARSE = 2
_MG_DEFAULT_PRE_SMOOTH = 2
_MG_DEFAULT_POST_SMOOTH = 2
_MG_DEFAULT_BOTTOM_SWEEPS = 50


class GeometricMG(Descriptor):
    """The geometric-multigrid elliptic solver (``pops::GeometricMG``), richly typed.

    ``GeometricMG(smoother=RedBlackGaussSeidel(), coarse=DirectSmallGrid(),
    tolerance=Relative(1e-8), max_cycles=50, min_coarse=2, pre_sweeps=2, post_sweeps=2,
    bottom_sweeps=50)``. Every knob is a typed sub-descriptor -- a bare string / number is rejected
    loud (Spec 5 sec.7). The descriptor is inert: it records the configuration and answers the
    protocol; the C++ multigrid kernel runs the V-cycles.

    ADC-613 wires these knobs END TO END: ``pops.compile`` / ``pops.bind`` lower the descriptor to
    ``System.set_poisson``, which forwards the resolved scalars to the native ``GeometricMG`` ctor +
    ``solve(rel_tol, max_cycles, abs_tol)``. The defaults are the native ``kMG*`` constants, so
    ``GeometricMG()`` reproduces the historical V-cycle bit-for-bit. Only the Gauss-Seidel smoother
    and the Gauss-Seidel bottom solve are wired natively; :meth:`validate` refuses the (never-wired)
    ``Chebyshev`` smoother STRUCTURALLY rather than silently ignoring it.
    """

    category = "elliptic_solver"
    native_id = "pops::GeometricMG"
    scheme = "geometric_mg"

    def __init__(self, smoother: Any = None, coarse: Any = None, tolerance: Any = None,
                 max_cycles: int = _MG_DEFAULT_MAX_CYCLES, min_coarse: int = _MG_DEFAULT_MIN_COARSE,
                 pre_sweeps: int = _MG_DEFAULT_PRE_SMOOTH,
                 post_sweeps: int = _MG_DEFAULT_POST_SMOOTH,
                 bottom_sweeps: int = _MG_DEFAULT_BOTTOM_SWEEPS) -> None:
        # Default smoother is the natively-wired RedBlackGaussSeidel (ADC-613): the native V-cycle
        # uses a Gauss-Seidel smoother, so this keeps GeometricMG() working. Chebyshev stays a
        # selectable descriptor but validate() refuses it (no native Chebyshev smoother yet).
        self.smoother = _check(smoother, _SMOOTHERS, "smoother",
                               "pops.solvers.options.RedBlackGaussSeidel()", RedBlackGaussSeidel())
        self.coarse = _check(coarse, _COARSE, "coarse",
                             "pops.solvers.options.DirectSmallGrid()", DirectSmallGrid())
        # Default tolerance = the native relative criterion (kMGDefaultRelTol) with NO absolute
        # floor (abs_tol 0), i.e. the historical purely-relative V-cycle stop.
        self.tolerance = _check(tolerance, _TOLERANCES, "tolerance",
                                "pops.solvers.tolerances.Relative()",
                                Relative(_MG_DEFAULT_REL_TOL))
        self.max_cycles = _check_positive_int(max_cycles, "max_cycles", minimum=1)
        self.min_coarse = _check_positive_int(min_coarse, "min_coarse", minimum=1)
        self.pre_sweeps = _check_positive_int(pre_sweeps, "pre_sweeps", minimum=0)
        self.post_sweeps = _check_positive_int(post_sweeps, "post_sweeps", minimum=0)
        self.bottom_sweeps = _check_positive_int(bottom_sweeps, "bottom_sweeps", minimum=0)

    @property
    def name(self) -> str:
        return "geometric_mg"

    def capabilities(self) -> Any:
        return CapabilitySet(capability_map(uniform=True, amr=True, mpi=True, gpu=True,
                                            variable_epsilon=True, periodic_bc=True, wall_bc=True))

    def options(self) -> dict:
        return {
            "smoother": self.smoother.name,
            "coarse": self.coarse.name,
            "tolerance": self.tolerance.name,
            "max_cycles": self.max_cycles,
            "min_coarse": self.min_coarse,
            "pre_sweeps": self.pre_sweeps,
            "post_sweeps": self.post_sweeps,
            "bottom_sweeps": self.bottom_sweeps,
        }

    def mg_options(self) -> dict:
        """The RESOLVED native V-cycle scalars set_poisson forwards to C++ (ADC-613).

        Maps the typed tolerance descriptor onto the native mixed criterion
        ``residual <= max(rel_tol * r0, abs_tol)``: :class:`Relative` gives ``rel_tol`` and (via its
        optional :class:`AbsoluteFloor`) ``abs_tol``; :class:`Absolute` gives an absolute floor with
        the native relative tolerance retained so the mixed criterion is dominated by the floor. The
        sweep knobs pass straight through. Values here reproduce today's V-cycle for the defaults.
        """
        rel_tol, abs_tol = self._resolved_tolerance()
        # ADC-644: the coarse solver's total-cell coarsening ceiling. None ("governed by min_coarse")
        # lowers to the native disabled sentinel 0, so a default DirectSmallGrid() keeps the historical
        # hierarchy bit-for-bit; a positive threshold enables the ceiling.
        coarse_threshold = 0 if self.coarse.threshold is None else int(self.coarse.threshold)
        return {
            "rel_tol": rel_tol,
            "abs_tol": abs_tol,
            "max_cycles": self.max_cycles,
            "min_coarse": self.min_coarse,
            "pre_smooth": self.pre_sweeps,
            "post_smooth": self.post_sweeps,
            "bottom_sweeps": self.bottom_sweeps,
            "coarse_threshold": coarse_threshold,
        }

    def _resolved_tolerance(self) -> Any:
        """(rel_tol, abs_tol) for the native mixed criterion from the typed tolerance descriptor."""
        if isinstance(self.tolerance, Relative):
            floor = self.tolerance.floor.abs_floor if self.tolerance.floor is not None else 0.0
            return float(self.tolerance.rel), float(floor)
        # Absolute: keep the native relative tolerance so rel_tol stays > 0 (the native solver
        # requires it) and the absolute floor dominates the mixed stop max(rel_tol*r0, abs_tol).
        return _MG_DEFAULT_REL_TOL, float(self.tolerance.abs_tol)

    def validate(self, context: Any = None) -> ValidationReport:
        """Refuse the sub-options with no native realisation STRUCTURALLY (ADC-613).

        The native ``GeometricMG`` V-cycle uses a Gauss-Seidel smoother and a Gauss-Seidel bottom
        solve. A :class:`Chebyshev` smoother has no native kernel yet, so it is REFUSED here (with an
        actionable alternative) rather than silently ignored -- honouring the audit rule that an
        unsupported sub-option refuses, never drops. Out-of-domain tolerances are rejected too.
        """
        report = ValidationReport(subject=self)
        if isinstance(self.smoother, Chebyshev):
            report.error(
                "elliptic_solver", "smoother_not_wired",
                "GeometricMG smoother %r has no native C++ kernel: the native V-cycle uses a "
                "Gauss-Seidel smoother. Use RedBlackGaussSeidel()." % self.smoother.name,
                context={"smoother": self.smoother.name},
                alternatives=["pops.solvers.options.RedBlackGaussSeidel()"])
        rel_tol, abs_tol = self._resolved_tolerance()
        if rel_tol <= 0.0:
            report.error(
                "elliptic_solver", "rel_tol_out_of_domain",
                "GeometricMG relative tolerance must be > 0; got %r." % rel_tol,
                context={"rel_tol": rel_tol})
        if abs_tol < 0.0:
            report.error(
                "elliptic_solver", "abs_tol_out_of_domain",
                "GeometricMG absolute floor must be >= 0; got %r." % abs_tol,
                context={"abs_tol": abs_tol})
        return report

    def lower(self, context: Any = None) -> Any:
        # Refuse the un-wired sub-options before lowering: a lowered descriptor must be honestly
        # realisable natively (never a silent drop of Chebyshev / a degenerate tolerance).
        self.validate(context).raise_if_error()
        from pops.descriptors_report import LoweredDescriptor
        return LoweredDescriptor(
            name=self.name, category=self.category, native_id=self.native_id,
            options=self.options(), scheme=self.scheme,
            extra={"smoother": self.smoother.lower(context),
                   "coarse": self.coarse.lower(context),
                   "tolerance": self.tolerance.lower(context),
                   "mg_options": self.mg_options()})

    def inspect(self) -> Any:
        view = super().inspect()
        view["scheme"] = self.scheme
        view["available"] = True
        return view


class FFT(Descriptor):
    """An FFT-based spectral Poisson solver (``pops::PoissonFFTSolver``).

    A real, runtime-wired elliptic solver selectable today via the ``fft`` / ``fft_spectral``
    tokens (validated to ~1e-12); under MPI it routes through the remapped FFT path. Its
    availability is :meth:`available` ``partial`` because the spectral route carries genuine
    constraints, not because it is unimplemented: it requires a PERIODIC boundary, a
    CONSTANT-coefficient operator (no wall / embedded boundary) and a power-of-two grid.
    ``spectral=True`` selects the continuous symbol ``-(kx^2 + ky^2)`` (token ``fft_spectral``)
    over the discrete stencil (token ``fft``). Inert -- the C++ runs the transform.
    """

    category = "elliptic_solver"
    native_id = "pops::PoissonFFTSolver"

    def __init__(self, spectral: bool = False) -> None:
        self.spectral = bool(spectral)

    @property
    def name(self) -> str:
        return "fft"

    @property
    def scheme(self) -> str:
        return "fft_spectral" if self.spectral else "fft"

    def capabilities(self) -> Any:
        return CapabilitySet(capability_map(uniform=True, mpi=True, gpu=True, periodic_bc=True))

    def options(self) -> dict:
        return {"spectral": self.spectral}

    def available(self, context: Any = None) -> Any:
        # Spec 6 sec.8: FFT is mathematically incompatible with an AMR hierarchy (it needs a
        # single uniform periodic mesh, not a refined one). When the route's layout is AMR,
        # refuse PRECISELY -- naming the incompatibility and the general elliptic alternative --
        # never a vague "AMR unsupported".
        if _context_is_amr_layout(context):
            return Availability.no(
                "FFT requires Uniform(periodic=True), got AMR. Use GeometricMG().",
                missing=["uniform layout", "periodic boundary"],
                alternatives=["pops.solvers.elliptic.GeometricMG()"])
        return Availability.partial(
            "the FFT Poisson solver requires a periodic boundary, a constant-coefficient "
            "operator (no wall / embedded boundary) and a power-of-two grid; under MPI it uses "
            "the remapped FFT route",
            missing=["periodic BC", "constant coefficient", "power-of-two grid"],
            alternatives=["pops.solvers.elliptic.GeometricMG()"])

    def inspect(self) -> Any:
        view = super().inspect()
        view["scheme"] = self.scheme
        view["available"] = "partial"
        return view


def _context_is_amr_layout(context: Any) -> bool:
    """True when the route @p context names an AMR mesh layout (duck-typed, no mesh import).

    A compile / validate context may carry the chosen layout under a ``"layout"`` key (a dict)
    or a ``.layout`` attribute, or simply BE the layout descriptor. A mesh layout advertises its
    kind through ``capabilities()["layout"]`` (``"amr"`` / ``"uniform"``), so AMR is recognised
    here WITHOUT importing :mod:`pops.mesh` into the solvers layer (which would add a forbidden
    cross-layer edge). A context with no layout (the common ``available()`` call) returns False,
    so the FFT route keeps its plain ``partial`` status.
    """
    if context is None:
        return False
    if isinstance(context, dict):
        layout = context.get("layout")
    else:
        layout = getattr(context, "layout", None)
    if layout is None:
        layout = context  # the context may itself be the layout descriptor
    caps = getattr(layout, "capabilities", None)
    if callable(caps):
        try:
            declared: Any = caps()
        except Exception:
            # available() must always return an Availability, never raise: a context whose
            # capabilities() needs an argument or itself raises is simply "not an AMR layout".
            return False
        # ``declared`` is a typed CapabilitySet (or a plain dict): both expose ``.get`` (ADC-625).
        if hasattr(declared, "get") and declared.get("layout") == "amr":
            return True
    return False


def _check_positive_int(value: Any, param: str, minimum: int) -> int:
    """Validate a GeometricMG integer knob: a Python int (not bool) at least @p minimum.

    A degenerate cycle count is refused at construction (an out-of-domain V-cycle is a structural
    error, never a silently clamped one, per the ADC-612 audit rule)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("GeometricMG(%s=) must be a Python int; got %r" % (param, value))
    if value < minimum:
        raise ValueError("GeometricMG(%s=) must be >= %d; got %d" % (param, minimum, value))
    return int(value)


def _check(value: Any, allowed: Any, param: str, suggestion: str, default: Any) -> Any:
    """Validate a typed sub-descriptor keyword: pass it through, default None, reject a string.

    A bare string / number for a typed slot is the Spec 5 sec.7 anti-pattern; it is rejected
    with an actionable message naming the typed @p suggestion. ``None`` yields @p default.
    """
    if value is None:
        return default
    if isinstance(value, allowed):
        return value
    raise TypeError(
        "GeometricMG(%s=) must be a %s descriptor, not %r; use %s."
        % (param, " / ".join(c.__name__ for c in allowed), value, suggestion))


__all__ = ["GeometricMG", "FFT"]
