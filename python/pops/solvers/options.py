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
from pops.solvers._numeric import (
    exact_nonnegative_real, exact_open_unit_real, optional_positive_int, strict_bool,
)


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
    """Typed overrides for the native composite-FAC AMR field backend (ADC-645).

    Passed as ``GeometricMG(fac=CompositeFAC(...))``, this descriptor configures FAC but
    never selects a hierarchy. ``FieldDiscretization(hierarchy_policy=CompositeHierarchySolve())``
    owns that independent choice. Every knob left ``None`` keeps the matching native ``kFAC*``
    default; the canonical field-install carrier retains those ``None`` values until the native
    ``CompositeFacOptions`` POD is assembled. Inert descriptor: the C++ FAC kernel does the work.
    """

    category = "amr_composite"
    native_id = "pops::CompositeFacPoisson"

    def __init__(self, max_iters: Any = None, fine_sweeps: Any = None, rel_tol: Any = None,
                 abs_tol: Any = None, coarse_rel_tol: Any = None,
                 coarse_abs_tol: Any = None, coarse_cycles: Any = None,
                 verbose: bool = False) -> None:
        # ``None`` stays an explicit authoring default until field resolve snapshots the native POD.
        self.max_iters = optional_positive_int(max_iters, where="CompositeFAC(max_iters=)")
        self.fine_sweeps = optional_positive_int(fine_sweeps, where="CompositeFAC(fine_sweeps=)")
        self.rel_tol = (None if rel_tol is None else exact_open_unit_real(
            rel_tol, where="CompositeFAC(rel_tol=)"))
        self.abs_tol = (None if abs_tol is None else exact_nonnegative_real(
            abs_tol, where="CompositeFAC(abs_tol=)"))
        self.coarse_rel_tol = (None if coarse_rel_tol is None else exact_open_unit_real(
            coarse_rel_tol, where="CompositeFAC(coarse_rel_tol=)"))
        self.coarse_abs_tol = (None if coarse_abs_tol is None else exact_nonnegative_real(
            coarse_abs_tol, where="CompositeFAC(coarse_abs_tol=)"))
        self.coarse_cycles = optional_positive_int(
            coarse_cycles, where="CompositeFAC(coarse_cycles=)")
        self.verbose = strict_bool(verbose, where="CompositeFAC(verbose=)")

    @property
    def name(self) -> str:
        return "composite_fac"

    def options(self) -> dict:
        return {"max_iters": self.max_iters, "fine_sweeps": self.fine_sweeps,
                "rel_tol": self.rel_tol, "abs_tol": self.abs_tol,
                "coarse_rel_tol": self.coarse_rel_tol, "coarse_abs_tol": self.coarse_abs_tol,
                "coarse_cycles": self.coarse_cycles,
                "verbose": self.verbose}

    def lower(self, context: Any = None) -> dict:
        return {"kind": "composite_fac", **self.options()}


__all__ = ["Chebyshev", "RedBlackGaussSeidel", "DirectSmallGrid", "CompositeFAC"]
