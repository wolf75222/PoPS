"""pops.experimental -- NON-PRODUCTION / TESTS-ONLY prototyping helpers.

This package is NOT a stable public API. It holds host-side prototyping helpers that compute a
numpy residual in Python, which the PoPS "no public Python numeric" rule excludes from the public
``pops`` surface. The contents are for residual prototyping in tests only and may change or be
removed without notice.

Currently:

* :class:`~pops.experimental.python_flux.PythonFlux` -- a host (numpy) Rusanov residual backend
  for iterating on a novel flux without recompiling. For production (GPU/MPI), declare a physical
  flux on :class:`pops.Model` and discretize it with :class:`pops.numerics.FiniteVolume` instead.
"""
from __future__ import annotations

from pops.experimental.python_flux import PythonFlux  # noqa: F401

__experimental__ = True

__all__ = ["PythonFlux"]
