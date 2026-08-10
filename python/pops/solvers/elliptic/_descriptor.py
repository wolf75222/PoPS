"""pops.solvers.elliptic -- the rich elliptic field-solver descriptors (Spec 5 sec.5.7).

The elliptic solve ``div(eps grad phi) = rhs`` is configured by a TYPED descriptor with a rich
parameter surface, not the bare string ``solver="geometric_mg"``:

* :class:`CartesianCG` -- the exact-ranked constant-coefficient Cartesian conjugate-gradient
  solver used by a uniform :class:`System`;
* :class:`GeometricMG` -- geometric multigrid for the AMR MG/FAC route, configured by a typed smoother
  (:class:`pops.solvers.options.Chebyshev` / :class:`~pops.solvers.options.RedBlackGaussSeidel`),
  a typed coarse solver (:class:`~pops.solvers.options.DirectSmallGrid`), a typed convergence
  tolerance (:class:`pops.solvers.tolerances.Relative` / :class:`~pops.solvers.tolerances.Absolute`)
  and a V-cycle cap (``max_cycles``). It declares its AMR / MPI / GPU /
  variable_epsilon) so an unsupported route is refused before the runtime is touched.
* :class:`FFT` -- the exact-ranked ``pops::PoissonFFTSolver<Dim>`` for Cartesian ranks 1, 2 and 3
  (periodic BC and constant coefficient). Radix-2 axes use the fast transform and other extents use
  the same authenticated discrete operator through an explicit direct-DFT fallback.

All are inert (Spec 5 sec.6): they record the choice and answer ``available`` / ``lower`` /
``inspect``; the C++ kernels perform the solve. The ``scheme`` attribute mirrors
the runtime token so the install path's solver-token resolution keeps working.
"""

from __future__ import annotations

from typing import Any

from pops._report import ReportTree
from pops.descriptors import Availability, Descriptor
from pops.descriptors_report import CapabilitySet
from pops.solvers.options import Chebyshev, CompositeFAC, DirectSmallGrid, RedBlackGaussSeidel
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
_MG_DEFAULT_ABS_TOL = 0.0
_MG_DEFAULT_MAX_CYCLES = 50
_MG_DEFAULT_MIN_COARSE = 2
_MG_DEFAULT_PRE_SMOOTH = 2
_MG_DEFAULT_POST_SMOOTH = 2
_MG_DEFAULT_BOTTOM_SWEEPS = 50
_MG_DEFAULT_COARSE_THRESHOLD = 0

_CARTESIAN_CG_DEFAULT_REL_TOL = 1e-10
_CARTESIAN_CG_DEFAULT_ABS_TOL = 0.0
_CARTESIAN_CG_DEFAULT_MAX_ITERATIONS = 2000


class CartesianCG(Descriptor):
    """Exact-ranked Cartesian conjugate gradient for one uniform ``System``.

    The native backend solves the constant-coefficient cell-centred Poisson operator over a
    compile-time rank in ``{1, 2, 3}``.  It is deliberately distinct from
    :class:`GeometricMG`: selecting this descriptor commits a CG algorithm and only the three
    controls that the native CG actually consumes.
    """

    category = "elliptic_solver"
    native_id = "pops::elliptic::nd::CartesianPoissonSolver<Dim>"
    scheme = "cartesian_cg"

    def __init__(
        self,
        tolerance: Any = None,
        max_iterations: int = _CARTESIAN_CG_DEFAULT_MAX_ITERATIONS,
    ) -> None:
        if tolerance is None:
            tolerance = Relative(_CARTESIAN_CG_DEFAULT_REL_TOL)
        if not isinstance(tolerance, _TOLERANCES):
            raise TypeError(
                "CartesianCG(tolerance=) must be a Relative / Absolute descriptor, not %r"
                % (tolerance,)
            )
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise TypeError(
                "CartesianCG(max_iterations=) must be a Python int; got %r" % (max_iterations,)
            )
        if max_iterations < 1:
            raise ValueError("CartesianCG(max_iterations=) must be >= 1; got %d" % max_iterations)
        self.tolerance = tolerance
        self.max_iterations = int(max_iterations)

    @property
    def name(self) -> str:
        return self.scheme

    def capabilities(self) -> Any:
        return CapabilitySet(capability_map(uniform=True, mpi=True, gpu=True, periodic_bc=True))

    def _resolved_tolerance(self) -> tuple[float, float]:
        if isinstance(self.tolerance, Relative):
            floor = self.tolerance.floor.abs_floor if self.tolerance.floor is not None else 0.0
            return float(self.tolerance.rel), float(floor)
        return _CARTESIAN_CG_DEFAULT_REL_TOL, float(self.tolerance.abs_tol)

    def cg_options(self) -> dict[str, Any]:
        rel_tol, abs_tol = self._resolved_tolerance()
        return {
            "rel_tol": rel_tol,
            "abs_tol": abs_tol,
            "max_iterations": self.max_iterations,
        }

    def native_cg_options(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "cartesian_cg_options",
            **self.cg_options(),
        }

    def _prepared_field_solver(self) -> tuple[Any, dict[str, Any]]:
        from ._prepared_field_providers import cartesian_cg_field_solver_provider

        return cartesian_cg_field_solver_provider(), self.cg_options()

    def options(self) -> dict[str, Any]:
        return {
            "tolerance": self.tolerance.name,
            "max_iterations": self.max_iterations,
        }

    def to_data(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "tolerance": {
                "type": type(self.tolerance).__name__,
                "options": self.tolerance.options(),
            },
            "max_iterations": self.max_iterations,
        }

    def validate(self, context: Any = None) -> ReportTree:
        rel_tol, abs_tol = self._resolved_tolerance()
        report = ReportTree(
            phase="validation",
            severity="info",
            code="validation.elliptic_solver.report",
            source="elliptic_solver",
            owner=self,
            evidence={"descriptor": self.name, "scheme": self.scheme},
        )
        if rel_tol <= 0.0:
            report = report.error(
                "elliptic_solver",
                "rel_tol_out_of_domain",
                "CartesianCG relative tolerance must be > 0; got %r." % rel_tol,
                context={"rel_tol": rel_tol},
            )
        if abs_tol < 0.0:
            report = report.error(
                "elliptic_solver",
                "abs_tol_out_of_domain",
                "CartesianCG absolute tolerance must be >= 0; got %r." % abs_tol,
                context={"abs_tol": abs_tol},
            )
        return report

    def lower(self, context: Any = None) -> Any:
        self.validate(context).raise_if_error()
        from pops.descriptors_report import LoweredDescriptor

        return LoweredDescriptor(
            name=self.name,
            category=self.category,
            native_id=self.native_id,
            options=self.options(),
            scheme=self.scheme,
            extra={
                "tolerance": self.tolerance.lower(context),
                "cg_options": self.native_cg_options(),
            },
        )

    def inspect(self) -> Any:
        view = super().inspect()
        view["scheme"] = self.scheme
        view["available"] = True
        return view


def native_geometric_mg_defaults() -> dict[str, Any]:
    """Closed native ``GeometricMgOptions`` carrier for routes where MG is ABI-inert.

    The solver module is the Python authoring authority for these pinned native defaults.  Field
    resolution snapshots this value once; bind/install must never reconstruct it from a live
    descriptor or from another set of fallback literals.
    """
    return {
        "schema_version": 1,
        "kind": "geometric_mg_options",
        "rel_tol": _MG_DEFAULT_REL_TOL,
        "abs_tol": _MG_DEFAULT_ABS_TOL,
        "max_cycles": _MG_DEFAULT_MAX_CYCLES,
        "min_coarse": _MG_DEFAULT_MIN_COARSE,
        "pre_smooth": _MG_DEFAULT_PRE_SMOOTH,
        "post_smooth": _MG_DEFAULT_POST_SMOOTH,
        "bottom_sweeps": _MG_DEFAULT_BOTTOM_SWEEPS,
        "coarse_threshold": _MG_DEFAULT_COARSE_THRESHOLD,
    }


class GeometricMG(Descriptor):
    """The geometric-multigrid elliptic solver (``pops::GeometricMG``), richly typed.

    ``GeometricMG(smoother=RedBlackGaussSeidel(), coarse=DirectSmallGrid(),
    tolerance=Relative(1e-8), max_cycles=50, min_coarse=2, pre_sweeps=2, post_sweeps=2,
    bottom_sweeps=50)``. Every knob is a typed sub-descriptor -- a bare string / number is rejected
    loud (Spec 5 sec.7). The descriptor is inert: it records the configuration and answers the
    protocol; the C++ multigrid kernel runs the V-cycles.

    ``pops.compile`` / ``pops.bind`` lower this descriptor only to the AMR MG/FAC provider. The
    defaults are the native ``kMG*`` constants. Only the Gauss-Seidel smoother and bottom solve are
    wired natively; :meth:`validate` refuses the (never-wired) ``Chebyshev`` smoother structurally
    rather than silently ignoring it. A uniform ``System`` uses :class:`CartesianCG` instead.
    """

    category = "elliptic_solver"
    native_id = "pops::GeometricMG"
    scheme = "geometric_mg"

    def __init__(
        self,
        smoother: Any = None,
        coarse: Any = None,
        tolerance: Any = None,
        max_cycles: int = _MG_DEFAULT_MAX_CYCLES,
        min_coarse: int = _MG_DEFAULT_MIN_COARSE,
        pre_sweeps: int = _MG_DEFAULT_PRE_SMOOTH,
        post_sweeps: int = _MG_DEFAULT_POST_SMOOTH,
        bottom_sweeps: int = _MG_DEFAULT_BOTTOM_SWEEPS,
        fac: Any = None,
    ) -> None:
        # Default smoother is the natively-wired RedBlackGaussSeidel (ADC-613): the native V-cycle
        # uses a Gauss-Seidel smoother, so this keeps GeometricMG() working. Chebyshev stays a
        # selectable descriptor but validate() refuses it (no native Chebyshev smoother yet).
        self.smoother = _check(
            smoother,
            _SMOOTHERS,
            "smoother",
            "pops.solvers.options.RedBlackGaussSeidel()",
            RedBlackGaussSeidel(),
        )
        self.coarse = _check(
            coarse, _COARSE, "coarse", "pops.solvers.options.DirectSmallGrid()", DirectSmallGrid()
        )
        # Default tolerance = the native relative criterion (kMGDefaultRelTol) with NO absolute
        # floor (abs_tol 0), i.e. the historical purely-relative V-cycle stop.
        self.tolerance = _check(
            tolerance,
            _TOLERANCES,
            "tolerance",
            "pops.solvers.tolerances.Relative()",
            Relative(_MG_DEFAULT_REL_TOL),
        )
        self.max_cycles = _check_positive_int(max_cycles, "max_cycles", minimum=1)
        self.min_coarse = _check_positive_int(min_coarse, "min_coarse", minimum=1)
        self.pre_sweeps = _check_positive_int(pre_sweeps, "pre_sweeps", minimum=0)
        self.post_sweeps = _check_positive_int(post_sweeps, "post_sweeps", minimum=0)
        self.bottom_sweeps = _check_positive_int(bottom_sweeps, "bottom_sweeps", minimum=0)
        # ADC-645: typed overrides for the AMR composite-FAC backend.  The field hierarchy policy,
        # not this slot, selects level-local versus composite execution; None leaves every FAC knob
        # at its native default.  A bare bool/string is rejected (typed slot, Spec 5 sec.7).
        if fac is not None and not isinstance(fac, CompositeFAC):
            raise TypeError(
                "GeometricMG(fac=) must be a pops.solvers.options.CompositeFAC "
                "descriptor or None, not %r; use CompositeFAC()." % (fac,)
            )
        self.fac = fac

    @property
    def name(self) -> str:
        return "geometric_mg"

    def capabilities(self) -> Any:
        return CapabilitySet(
            capability_map(
                amr=True,
                mpi=True,
                gpu=True,
                variable_epsilon=True,
                screened=True,
                periodic_bc=True,
                wall_bc=True,
            )
        )

    def _prepared_field_solver(self) -> tuple[Any, dict[str, Any]]:
        """Bind through the same authenticated provider protocol as external solvers."""
        from ._prepared_field_providers import (
            geometric_mg_field_solver_provider,
        )

        return geometric_mg_field_solver_provider(), {
            "mg": self.mg_options(),
            "fac": None if self.fac is None else self.fac.options(),
        }

    def native_mg_options(self) -> dict[str, Any]:
        """Closed, identity-bearing POD snapshot consumed by field resolve/install."""
        return {
            "schema_version": 1,
            "kind": "geometric_mg_options",
            **self.mg_options(),
        }

    def amr_fac_options(self) -> dict[str, Any] | None:
        """Canonical optional overrides for the native composite-FAC backend.

        Hierarchy selection is deliberately absent: ``CompositeHierarchySolve`` owns that
        decision.  ``None`` means the native ``CompositeFacOptions`` defaults, while an explicit
        :class:`CompositeFAC` carries only authored overrides (unconfigured members stay ``None``).
        """
        if self.fac is None:
            return None
        return {
            "schema_version": 1,
            "kind": "composite_fac",
            **self.fac.options(),
        }

    def options(self) -> dict:
        view = {
            "smoother": self.smoother.name,
            "coarse": self.coarse.name,
            "tolerance": self.tolerance.name,
            "max_cycles": self.max_cycles,
            "min_coarse": self.min_coarse,
            "pre_sweeps": self.pre_sweeps,
            "post_sweeps": self.post_sweeps,
            "bottom_sweeps": self.bottom_sweeps,
        }
        # ADC-645: present ONLY when set (omit-when-default), so an unconfigured GeometricMG()
        # options view -- and everything hashed from it -- is unchanged.
        if self.fac is not None:
            view["fac"] = self.fac.name
        return view

    def to_data(self) -> dict[str, Any]:
        """Exact structural identity consumed through the generic descriptor protocol."""
        return {
            "scheme": self.scheme,
            "smoother": {
                "type": type(self.smoother).__name__,
                "options": self.smoother.options(),
            },
            "coarse": {
                "type": type(self.coarse).__name__,
                "options": self.coarse.options(),
            },
            "tolerance": {
                "type": type(self.tolerance).__name__,
                "options": self.tolerance.options(),
            },
            "max_cycles": self.max_cycles,
            "min_coarse": self.min_coarse,
            "pre_sweeps": self.pre_sweeps,
            "post_sweeps": self.post_sweeps,
            "bottom_sweeps": self.bottom_sweeps,
            "fac": (
                None
                if self.fac is None
                else {
                    "type": type(self.fac).__name__,
                    "options": self.fac.options(),
                }
            ),
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
            floor = self.tolerance.floor.abs_floor if self.tolerance.floor is not None else 0
            return self.tolerance.rel, floor
        # Absolute: keep the native relative tolerance so rel_tol stays > 0 (the native solver
        # requires it) and the absolute floor dominates the mixed stop max(rel_tol*r0, abs_tol).
        return _MG_DEFAULT_REL_TOL, self.tolerance.abs_tol

    def validate(self, context: Any = None) -> ReportTree:
        """Refuse the sub-options with no native realisation STRUCTURALLY (ADC-613).

        The native ``GeometricMG`` V-cycle uses a Gauss-Seidel smoother and a Gauss-Seidel bottom
        solve. A :class:`Chebyshev` smoother has no native kernel yet, so it is REFUSED here (with an
        actionable alternative) rather than silently ignored -- honouring the audit rule that an
        unsupported sub-option refuses, never drops. Out-of-domain tolerances are rejected too.
        """
        report = ReportTree(
            phase="validation",
            severity="info",
            code="validation.elliptic_solver.report",
            source="elliptic_solver",
            owner=self,
            evidence={"descriptor": self.name, "scheme": self.scheme},
        )
        if isinstance(self.smoother, Chebyshev):
            report = report.error(
                "elliptic_solver",
                "smoother_not_wired",
                "GeometricMG smoother %r has no native C++ kernel: the native V-cycle uses a "
                "Gauss-Seidel smoother. Use RedBlackGaussSeidel()." % self.smoother.name,
                context={"smoother": self.smoother.name},
                alternatives=["pops.solvers.options.RedBlackGaussSeidel()"],
            )
        rel_tol, abs_tol = self._resolved_tolerance()
        if rel_tol <= 0.0:
            report = report.error(
                "elliptic_solver",
                "rel_tol_out_of_domain",
                "GeometricMG relative tolerance must be > 0; got %r." % rel_tol,
                context={"rel_tol": rel_tol},
            )
        if abs_tol < 0.0:
            report = report.error(
                "elliptic_solver",
                "abs_tol_out_of_domain",
                "GeometricMG absolute floor must be >= 0; got %r." % abs_tol,
                context={"abs_tol": abs_tol},
            )
        return report

    def lower(self, context: Any = None) -> Any:
        # Refuse the un-wired sub-options before lowering: a lowered descriptor must be honestly
        # realisable natively (never a silent drop of Chebyshev / a degenerate tolerance).
        self.validate(context).raise_if_error()
        from pops.descriptors_report import LoweredDescriptor

        return LoweredDescriptor(
            name=self.name,
            category=self.category,
            native_id=self.native_id,
            options=self.options(),
            scheme=self.scheme,
            extra={
                "smoother": self.smoother.lower(context),
                "coarse": self.coarse.lower(context),
                "tolerance": self.tolerance.lower(context),
                "mg_options": self.native_mg_options(),
                # ADC-645: exact authoring identity plus the canonical backend-option carrier.
                "fac": (self.fac.lower(context) if self.fac is not None else None),
                "fac_options": self.amr_fac_options(),
            },
        )

    def inspect(self) -> Any:
        view = super().inspect()
        view["scheme"] = self.scheme
        view["available"] = True
        return view


class FFT(Descriptor):
    """The exact-ranked Cartesian discrete FFT Poisson provider.

    The C++ engine is instantiated at the artifact's compile-time rank in ``{1, 2, 3}`` and
    distributes the last transform direction over canonical MPI slabs. The public provider inverts
    the same exact-rank discrete Cartesian operator used to authenticate its residual. There is
    deliberately no continuous-symbol selector: that engine has no matching apply/residual
    provider and cannot publish an authenticated :class:`SolveReport` for it.
    """

    category = "elliptic_solver"
    native_id = "pops::PoissonFFTSolver<Dim>"

    @property
    def name(self) -> str:
        return "fft"

    @property
    def scheme(self) -> str:
        return "fft"

    def capabilities(self) -> Any:
        return CapabilitySet(capability_map(uniform=True, mpi=True, gpu=True, periodic_bc=True))

    def _prepared_field_solver(self) -> tuple[Any, dict[str, Any]]:
        """Bind through the same authenticated provider protocol as every field solver."""
        from ._prepared_field_providers import fft_field_solver_provider

        return fft_field_solver_provider(), {}

    def options(self) -> dict:
        return {}

    def to_data(self) -> dict[str, Any]:
        return {"scheme": self.scheme}

    def available(self, context: Any = None) -> Any:
        # Spec 6 sec.8: FFT is mathematically incompatible with an AMR hierarchy (it needs a
        # single uniform periodic mesh, not a refined one). When the route's layout is AMR,
        # refuse PRECISELY -- naming the incompatibility and the general elliptic alternative --
        # never a vague "AMR unsupported".
        if _context_is_amr_layout(context):
            return Availability.no(
                "FFT requires Uniform(periodic=True), got AMR. Use GeometricMG().",
                missing=["uniform layout", "periodic boundary"],
                alternatives=["pops.solvers.elliptic.GeometricMG()"],
            )
        dimension = _context_layout_dimension(context)
        if dimension is not None and dimension not in (1, 2, 3):
            return Availability.no(
                "FFT requires a Cartesian layout with Dim in {1,2,3}; got Dim=%d. "
                "Use CartesianCG()." % dimension,
                missing=["native Cartesian rank in {1,2,3}"],
                alternatives=["pops.solvers.elliptic.CartesianCG()"],
            )
        return Availability.partial(
            "the exact-ranked FFT Poisson solver requires a periodic boundary, a "
            "constant-coefficient operator (no wall / embedded boundary) and canonical ordered "
            "MPI slabs; non-radix-2 extents use the diagnosed direct-DFT path",
            missing=[
                "periodic BC", "constant coefficient", "canonical MPI slabs",
            ],
            alternatives=["pops.solvers.elliptic.CartesianCG()"],
        )

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


def _context_layout_dimension(context: Any) -> int | None:
    """Best-effort exact layout rank for availability diagnostics; never raises."""
    if context is None:
        return None
    if isinstance(context, dict):
        layout = context.get("layout")
    else:
        layout = getattr(context, "layout", None)
    if layout is None:
        layout = context

    projection = getattr(layout, "normalized_geometry", None)
    if callable(projection):
        try:
            dimension = getattr(projection(), "dimension", None)
        except Exception:
            dimension = None
        if type(dimension) is int and dimension in (1, 2, 3):
            return dimension

    cells: Any = None
    if isinstance(layout, dict):
        cells = layout.get("cells")
        geometry = layout.get("geometry")
        if cells is None and isinstance(geometry, dict):
            cells = geometry.get("cells")
    else:
        cells = getattr(getattr(layout, "mesh", layout), "cells", None)
    if isinstance(cells, (tuple, list)) and len(cells) in (1, 2, 3):
        return len(cells)
    return None


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
        % (param, " / ".join(c.__name__ for c in allowed), value, suggestion)
    )


__all__ = ["CartesianCG", "GeometricMG", "FFT"]
