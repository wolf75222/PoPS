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


__all__ = ["Chebyshev", "RedBlackGaussSeidel", "DirectSmallGrid"]
