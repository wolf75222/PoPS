"""pops.solvers.elliptic -- the rich elliptic field-solver descriptors (Spec 5 sec.5.7).

The elliptic solve ``div(eps grad phi) = rhs`` is configured by a TYPED descriptor with a rich
parameter surface, not the bare string ``solver="geometric_mg"``:

* :class:`GeometricMG` -- geometric multigrid, configured by a typed smoother
  (:class:`pops.solvers.options.Chebyshev` / :class:`~pops.solvers.options.RedBlackGaussSeidel`),
  a typed coarse solver (:class:`~pops.solvers.options.DirectSmallGrid`), a typed convergence
  tolerance (:class:`pops.solvers.tolerances.Relative` / :class:`~pops.solvers.tolerances.Absolute`)
  and a V-cycle cap (``max_cycles``). It declares its capabilities (uniform / amr / mpi / gpu /
  variable_epsilon) so an unsupported route is refused before the runtime is touched.
* :class:`FFT` -- a PLANNED spectral Poisson solver (periodic BC, power-of-two grid); no native
  symbol yet, so ``available()`` reports ``partial`` and points at :class:`GeometricMG`.

Both are inert (Spec 5 sec.6): they record the choice and answer ``available`` / ``lower`` /
``inspect``; the C++ kernel performs the multigrid V-cycles. The ``scheme`` attribute mirrors
the runtime token so the install path's solver-token resolution keeps working.
"""
from pops.descriptors import Availability, Descriptor
from pops.solvers.options import Chebyshev, DirectSmallGrid, RedBlackGaussSeidel
from pops.solvers.requirements import capability_map
from pops.solvers.tolerances import Absolute, AbsoluteFloor, Relative

_SMOOTHERS = (Chebyshev, RedBlackGaussSeidel)
_COARSE = (DirectSmallGrid,)
_TOLERANCES = (Relative, Absolute)


class GeometricMG(Descriptor):
    """The geometric-multigrid elliptic solver (``pops::GeometricMG``), richly typed.

    ``GeometricMG(smoother=Chebyshev(2), coarse=DirectSmallGrid(100),
    tolerance=Relative(1e-6, AbsoluteFloor(1e-12)), max_cycles=20)``. Every knob is a typed
    sub-descriptor -- a bare string / number is rejected loud (Spec 5 sec.7). The descriptor is
    inert: it records the configuration and answers the protocol; the C++ multigrid kernel runs
    the V-cycles.
    """

    category = "elliptic_solver"
    native_id = "pops::GeometricMG"
    scheme = "geometric_mg"

    def __init__(self, smoother=None, coarse=None, tolerance=None, max_cycles=20):
        self.smoother = _check(smoother, _SMOOTHERS, "smoother",
                               "pops.solvers.options.Chebyshev()", Chebyshev())
        self.coarse = _check(coarse, _COARSE, "coarse",
                             "pops.solvers.options.DirectSmallGrid()", DirectSmallGrid())
        self.tolerance = _check(tolerance, _TOLERANCES, "tolerance",
                                "pops.solvers.tolerances.Relative()",
                                Relative(1e-6, AbsoluteFloor(1e-12)))
        if isinstance(max_cycles, bool) or not isinstance(max_cycles, int):
            raise TypeError("GeometricMG(max_cycles=) must be a Python int; got %r"
                            % (max_cycles,))
        if max_cycles < 1:
            raise ValueError("GeometricMG(max_cycles=) must be >= 1; got %d" % max_cycles)
        self.max_cycles = int(max_cycles)

    @property
    def name(self):
        return "geometric_mg"

    def capabilities(self):
        return capability_map(uniform=True, amr=True, mpi=True, gpu=True,
                              variable_epsilon=True, periodic_bc=True, wall_bc=True)

    def options(self):
        return {
            "smoother": self.smoother.name,
            "coarse": self.coarse.name,
            "tolerance": self.tolerance.name,
            "max_cycles": self.max_cycles,
        }

    def lower(self, context=None):
        rec = super().lower(context)
        rec["scheme"] = self.scheme
        rec["smoother"] = self.smoother.lower(context)
        rec["coarse"] = self.coarse.lower(context)
        rec["tolerance"] = self.tolerance.lower(context)
        rec["max_cycles"] = self.max_cycles
        return rec

    def inspect(self):
        view = super().inspect()
        view["scheme"] = self.scheme
        view["available"] = True
        return view


class FFT(Descriptor):
    """A PLANNED FFT-based spectral Poisson solver (periodic BC, power-of-two grid).

    No native symbol exists yet, so :meth:`available` reports ``partial`` (it names the slot
    and the constraints and points at :class:`GeometricMG`) and :attr:`native_id` is ``None``.
    ``spectral=True`` selects the continuous symbol ``-(kx^2 + ky^2)`` over the discrete
    stencil. Inert.
    """

    category = "elliptic_solver"
    native_id = None
    scheme = "fft"

    def __init__(self, spectral=False):
        self.spectral = bool(spectral)

    @property
    def name(self):
        return "fft"

    def capabilities(self):
        return capability_map(uniform=True, mpi=True, gpu=True, periodic_bc=True)

    def options(self):
        return {"spectral": self.spectral}

    def available(self, context=None):
        return Availability.partial(
            "the FFT spectral Poisson solver has no native C++ symbol yet; it also requires a "
            "periodic boundary and a power-of-two grid",
            missing=["pops::FFT native symbol", "periodic BC", "power-of-two grid"],
            alternatives=["pops.solvers.elliptic.GeometricMG()"])

    def inspect(self):
        view = super().inspect()
        view["scheme"] = self.scheme
        view["available"] = False
        return view


def _check(value, allowed, param, suggestion, default):
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
