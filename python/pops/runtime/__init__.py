"""pops.runtime : the runtime layer of the PoPS bindings (Spec-4 PR-F).

This is the TOP layer and the ONLY layer permitted to import the ``_pops`` extension (via
``pops._bootstrap``). It owns the user-facing runtime objects:

- the systems ``System`` / ``AmrSystem`` (compose blocks, share a Poisson, advance the whole) ;
- the composable BRICKS (state / transport / source / elliptic) and the spatial + time policies ;
- the geometry MESH objects (CartesianMesh / PolarMesh / AuxHalo) ;
- the parallelism knobs (set_threads / has_kokkos / parallel_info) ;
- the environment doctor / capability matrix.

The host PythonFlux prototyping backend has moved to :mod:`pops.experimental` (NON-PRODUCTION /
TESTS-ONLY: it computes a numpy residual in Python).

The lower layers (ir / model / physics / time / lib / codegen) are numpy-free until first use
and never import this package. The runtime methods that need the codegen / physics / dsl layers
import them LAZILY in-method, both to avoid a cycle and to keep ``import pops`` numpy-free.

Public user code should enter this layer through ``System.install(compiled, ...)`` /
``AmrSystem.install(compiled, ...)``. Low-level block/equation setters remain private
implementation seams used by the install lowering.
"""
from pops.runtime.profile import PerformanceSummary, Profile  # noqa: F401

__all__ = ["Profile", "PerformanceSummary"]
