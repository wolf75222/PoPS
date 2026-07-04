"""pops.solvers.options -- typed smoother / coarse-solver sub-descriptors (Spec 5 sec.5.7).

The multigrid elliptic solver (:class:`pops.solvers.elliptic.GeometricMG`) is configured by
TYPED objects, not strings: a smoother (:class:`Chebyshev` / :class:`RedBlackGaussSeidel`) and
a coarse-grid solver (:class:`DirectSmallGrid`). These are inert descriptors -- they record
the choice and its knobs and compute nothing; the C++ multigrid kernel runs the smoother and
the coarse solve.
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor


class Chebyshev(Descriptor):
    """A Chebyshev polynomial smoother of the given ``degree`` (pre/post-smooth stage).

    Inert: it names the smoother and its degree; the C++ multigrid kernel applies it.
    """

    category = "smoother"
    native_id = None

    def __init__(self, degree: int = 2) -> None:
        if isinstance(degree, bool) or not isinstance(degree, int):
            raise TypeError("Chebyshev(degree=) must be a Python int; got %r" % (degree,))
        if degree < 1:
            raise ValueError("Chebyshev(degree=) must be >= 1; got %d" % degree)
        self.degree = int(degree)

    @property
    def name(self) -> str:
        return "chebyshev"

    def options(self) -> dict:
        return {"degree": self.degree}

    def lower(self, context: Any = None) -> dict:
        return {"kind": "chebyshev", "degree": self.degree}


class RedBlackGaussSeidel(Descriptor):
    """A red-black Gauss-Seidel smoother (color-ordered for stencil parallelism). Inert."""

    category = "smoother"
    native_id = None

    @property
    def name(self) -> str:
        return "red_black_gauss_seidel"

    def options(self) -> dict:
        return {}

    def lower(self, context: Any = None) -> dict:
        return {"kind": "red_black_gauss_seidel"}


class DirectSmallGrid(Descriptor):
    """A direct coarse-grid solve for hierarchies whose coarse level is below ``threshold``.

    ``threshold`` is the coarse-grid TOTAL unknown count (nx*ny) at or below which coarsening STOPS
    and the bottom (Gauss-Seidel) solve -- the native direct-small-grid stand-in -- runs. Distinct from
    :class:`pops.solvers.elliptic.GeometricMG`'s per-axis ``min_coarse``; when both are active,
    coarsening halts at whichever is reached first. Inert: the C++ multigrid kernel performs the coarse
    solve.

    ADC-644: ``threshold`` is now wired end to end (before, it was recorded but silently dropped). The
    default is ``None`` = "governed by min_coarse" (the native disabled sentinel 0), so an
    unconfigured ``DirectSmallGrid()`` keeps today's hierarchy bit-for-bit. A positive int enables the
    ceiling. (The old default ``100`` was never honoured, so no recorded run ever depended on it.)
    """

    category = "coarse_solver"
    native_id = None

    def __init__(self, threshold: Any = None) -> None:
        if threshold is None:
            self.threshold = None
            return
        if isinstance(threshold, bool) or not isinstance(threshold, int):
            raise TypeError(
                "DirectSmallGrid(threshold=) must be a Python int or None; got %r" % (threshold,))
        if threshold < 1:
            raise ValueError("DirectSmallGrid(threshold=) must be >= 1 (or None); got %d" % threshold)
        self.threshold = int(threshold)

    @property
    def name(self) -> str:
        return "direct_small_grid"

    def options(self) -> dict:
        return {"threshold": self.threshold}

    def lower(self, context: Any = None) -> dict:
        return {"kind": "direct_small_grid", "threshold": self.threshold}


class CompositeFAC(Descriptor):
    """The composite FAC AMR field solve (ADC-645): the fine patch REFINES the elliptic.

    Passed as ``GeometricMG(amr_composite=CompositeFAC(...))``, it opts the AMR Poisson FIELD solve
    into the native composite path (``AmrCouplerMP::set_composite_poisson``) instead of the Option A
    coarse solve + gradient injection. Scope = the coupler's: single block, 2 levels, one mono-box
    fine patch, replicated coarse; an out-of-scope hierarchy refuses at build (never a silent
    fallback). Every knob left ``None`` keeps the native ``kFAC*`` default (the 0-sentinel wire
    convention shared with ``CondensedSchur``'s ``fac_*``), so ``CompositeFAC()`` runs the default
    composite solve. Inert descriptor: the C++ FAC kernel does the work.
    """

    category = "amr_composite"
    native_id = "pops::CompositeFacPoisson"

    def __init__(self, max_iters: Any = None, fine_sweeps: Any = None, tol: Any = None,
                 coarse_rel_tol: Any = None, coarse_cycles: Any = None,
                 verbose: bool = False) -> None:
        # Mirror the CondensedSchur fac_* validation domains (pops/runtime/_bricks_time.py): a knob
        # is either None (the kFAC* default, wire sentinel 0) or in its domain -- never clamped.
        self.max_iters = int(max_iters) if max_iters is not None else 0
        self.fine_sweeps = int(fine_sweeps) if fine_sweeps is not None else 0
        self.tol = float(tol) if tol is not None else 0.0
        self.coarse_rel_tol = float(coarse_rel_tol) if coarse_rel_tol is not None else 0.0
        self.coarse_cycles = int(coarse_cycles) if coarse_cycles is not None else 0
        self.verbose = bool(verbose)
        if max_iters is not None and self.max_iters < 1:
            raise ValueError("CompositeFAC: max_iters >= 1 (got %r)" % (max_iters,))
        if fine_sweeps is not None and self.fine_sweeps < 1:
            raise ValueError("CompositeFAC: fine_sweeps >= 1 (got %r)" % (fine_sweeps,))
        if tol is not None and not (0.0 < self.tol < 1.0):
            raise ValueError("CompositeFAC: tol must be in (0, 1) (got %r)" % (tol,))
        if coarse_rel_tol is not None and not (0.0 < self.coarse_rel_tol < 1.0):
            raise ValueError("CompositeFAC: coarse_rel_tol must be in (0, 1) (got %r)"
                             % (coarse_rel_tol,))
        if coarse_cycles is not None and self.coarse_cycles < 1:
            raise ValueError("CompositeFAC: coarse_cycles >= 1 (got %r)" % (coarse_cycles,))

    @property
    def name(self) -> str:
        return "composite_fac"

    def options(self) -> dict:
        return {"max_iters": self.max_iters, "fine_sweeps": self.fine_sweeps, "tol": self.tol,
                "coarse_rel_tol": self.coarse_rel_tol, "coarse_cycles": self.coarse_cycles,
                "verbose": self.verbose}

    def set_poisson_kwargs(self) -> dict:
        """The AmrSystem.set_poisson keyword args this descriptor lowers to (0 = native default)."""
        return {"composite": True, "fac_max_iters": self.max_iters,
                "fac_fine_sweeps": self.fine_sweeps, "fac_tol": self.tol,
                "fac_coarse_rel_tol": self.coarse_rel_tol, "fac_coarse_cycles": self.coarse_cycles,
                "fac_verbose": self.verbose}

    def lower(self, context: Any = None) -> dict:
        return {"kind": "composite_fac", **self.options()}


__all__ = ["Chebyshev", "RedBlackGaussSeidel", "DirectSmallGrid", "CompositeFAC"]
