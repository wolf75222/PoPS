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

This layer is INTERNAL: the public front door is the typed compile / bind flow
(``pops.compile(problem, layout=..., backend=Production())`` then ``pops.bind(compiled, ...)``).
``System`` / ``AmrSystem`` are the engines ``pops.bind`` wires; they left the ``pops`` root
(ADC-545) and are reached, for advanced / native tests only, as
``pops.runtime.system.System`` / ``pops.runtime.system.AmrSystem`` (re-exported here too).
"""
from pops._bootstrap import ModelSpec  # noqa: F401  (legacy native-bridge POD, quarantined from the pops root by ADC-585)
# ADC-545: the engines are the advanced seam behind pops.bind; expose them here (and via
# pops.runtime.system) so advanced tests reach them without the removed top-level pops.System.
from pops.runtime.system import System, AmrSystem  # noqa: F401  (advanced runtime seam)
from pops.runtime.profile import PerformanceSummary, Profile  # noqa: F401
from pops.runtime.inspection import RuntimeInspectionReport  # noqa: F401
from pops.runtime.defaults import numerical_defaults_report  # noqa: F401
from pops.runtime.fallbacks import fallback_diagnostics_report, reset_fallback_diagnostics  # noqa: F401
from pops.runtime.routes import Route, route_manifest  # noqa: F401  (typed native routes, ADC-584)
from pops.runtime.brick_catalog import brick_catalog  # noqa: F401  (builtin native brick catalog, ADC-586)
from pops.runtime.platform_manifest import (  # noqa: F401
    CapabilityProof, ExecutionContext, ExecutionResource, FieldViewDescriptor,
    PlatformContractError, PlatformManifest, PrecisionPolicy, RuntimeBackendManifest,
    launch_checked, validate_launch,
)

__all__ = [
    "ModelSpec",
    "Profile", "PerformanceSummary", "RuntimeInspectionReport", "numerical_defaults_report",
    "fallback_diagnostics_report", "reset_fallback_diagnostics",
    "Route", "route_manifest",
    "brick_catalog",
    "CapabilityProof", "PrecisionPolicy", "PlatformManifest", "RuntimeBackendManifest",
    "ExecutionResource", "ExecutionContext", "FieldViewDescriptor", "PlatformContractError",
    "validate_launch", "launch_checked",
]
