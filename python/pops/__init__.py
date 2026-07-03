"""pops : Python bindings for the PoPS library.

The core exposes generic compiled BRICKS (transport, source, elliptic right-hand
side) ; a MODEL is a composition of bricks, named on the application side. Python
composes the bricks (objects), the cell-by-cell computation stays in compiled C++ (no
numpy, GPU/MPI preserved).

The front door is the typed assembly + compile/bind/run flow: author an inert
``pops.Problem`` (physics blocks, elliptic fields, a time scheme), compile it to a handle for a
mesh layout, then bind a runnable simulation::

    import pops
    from pops.codegen import Production
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import Uniform
    from pops.fields import PoissonProblem

    problem = (pops.Problem(name="plasma")
               .block("ne", physics=model)
               .field(PoissonProblem(unknown="phi", equation=eq, solver=mg))
               .time(time_program))
    compiled = pops.compile(problem, layout=Uniform(CartesianMesh(n=96, periodic=False)),
                            backend=Production())
    sim = pops.bind(compiled, initial_state={"ne": ne0})
    sim.run(t_end=0.1, cfl=0.4)

The scenario names (diocotron, electron_euler...) are compositions on the
application side (see adc_cases). No scenario name here.
"""
# Load the _pops extension (RTLD_GLOBAL so the DSL production .so resolves C++ symbols).
from pops import _bootstrap  # noqa: F401  (import side effect: loads _pops with the right flags)
from pops._bootstrap import (SystemConfig, _System,  # noqa: F401
                             AmrSystemConfig, _AmrSystem, abi_key)
# ADC-585 quarantined ModelSpec (legacy native-bridge POD) to pops.runtime.ModelSpec.
from pops._version import __version__  # noqa: F401

# Runtime layer (the ONLY importer of _pops): systems, parallelism, doctor, mesh, bricks, host flux.
from pops.runtime.system import System, AmrSystem  # noqa: F401
from pops.runtime.threading import set_threads, has_kokkos, parallel_info  # noqa: F401
from pops.runtime.doctor import doctor, capabilities  # noqa: F401
from pops.runtime.mesh import CartesianMesh, PolarMesh, AuxHalo  # noqa: F401
from pops.runtime.profile import Profile, PerformanceSummary  # noqa: F401
from pops.runtime.inspection import RuntimeInspectionReport  # noqa: F401
from pops.runtime.defaults import numerical_defaults_report  # noqa: F401
from pops.runtime.fallbacks import fallback_diagnostics_report, reset_fallback_diagnostics  # noqa: F401
from pops.runtime.bricks import (  # noqa: F401
    Scalar, FluidState, ExB, CompressibleFlux, IsothermalFlux,
    NoSource, PotentialForce, GravityForce, MagneticLorentzForce, PotentialMagneticForce,
    ChargeDensity, BackgroundDensity, GravityCoupling,
    Model, CompositeModel, _native_to_brick,
    DivEpsGrad, CompositeRhs, ChargeDensitySource, ElectricFieldFromPotential, EllipticModel,
    div_eps_grad, charge_density, composite_rhs, electric_field_from_potential, elliptic,
    EllipticSolver,
    Ionization, Collision, ThermalExchange,
    Spatial, FiniteVolume, Explicit, _role_to_stable, _norm_implicit,
    IMEX, SourceImplicit, SourceImplicitBE, IMEXRK, Role,
    CondensedSchur, ElectrostaticLorentzSchur, Split, Strang,
    Dirichlet, Neumann, Periodic,
)

__all__ = [
    "System", "SystemConfig", "AmrSystem", "AmrSystemConfig", "Model", "CompositeModel",
    "CartesianMesh", "PolarMesh", "AuxHalo",
    "Scalar", "FluidState", "ExB", "CompressibleFlux", "IsothermalFlux",
    "NoSource", "PotentialForce", "GravityForce", "MagneticLorentzForce", "PotentialMagneticForce",
    "ChargeDensity", "BackgroundDensity", "GravityCoupling",
    "Spatial", "FiniteVolume", "Explicit", "IMEX", "IMEXRK", "SourceImplicit", "SourceImplicitBE",
    "Split", "Strang", "CondensedSchur", "ElectrostaticLorentzSchur", "Role", "integrate",
    "Dirichlet", "Neumann", "Periodic",
    "elliptic", "div_eps_grad", "charge_density", "composite_rhs",
    "electric_field_from_potential", "EllipticSolver", "EllipticModel",
    "Ionization", "Collision", "ThermalExchange",
    "Profile", "PerformanceSummary", "RuntimeInspectionReport",
    "numerical_defaults_report", "fallback_diagnostics_report", "reset_fallback_diagnostics",
    "time", "model", "math", "physics", "lib", "mesh",
    "params", "output", "external", "fields", "linalg", "solvers", "experimental",
    "abi_key", "capabilities", "inspect", "inspect_capabilities", "inspect_amr", "native_capability_report",
    "runtime_environment_report", "validate_runtime_environment", "RuntimeCapabilityError",
    "set_threads", "has_kokkos", "parallel_info", "doctor",
    "CompiledArtifact", "CompiledTime",
    "compile_library", "read_library_manifest", "LibraryManifest",
    "Problem", "PhysicsModel", "compile", "bind", "RuntimePolicies",
]


# Lower / authoring layers + the moved integrate (re-exported, surface unchanged; numpy-free import).
from pops.runtime import integrate  # noqa: E402,F401  (pops.integrate name preserved; without numpy)
from . import time, model, math, lib, physics, mesh  # noqa: E402  (Spec 2/3 operator-first + board authoring + IR)
from . import params, output, external, fields, linalg, solvers  # noqa: E402  (Spec 5 typed params/output/fields/algebra/solvers)
from .problem import Problem  # noqa: E402,F401  (Spec 5 sec.5.16: top-level compilable assembly; pure stdlib)
from pops.physics import PhysicsModel  # noqa: E402,F401  (Spec 5 sec.11: alias of pops.physics.Model)
from .codegen.library import (  # noqa: E402,F401  (re-export: brick-library manifest API, Spec 3 section 21)
    LibraryManifest, compile_library, read_library_manifest)
from .time import CompiledTime  # noqa: E402,F401  (re-export: compiled-Program time policy)
from ._capabilities import (  # noqa: E402,F401  (Spec 5: descriptor-sourced matrix + native reports)
    inspect_capabilities, inspect_amr, native_capability_report)
from ._inspect import inspect  # noqa: E402,F401  (ADC-527: stable per-object inspect dispatcher)
from .runtime_environment import RuntimeCapabilityError, runtime_environment_report, validate_runtime_environment  # noqa: E402,F401,E501


# LAZY public front doors (PEP 562, ADC-523). `pops.compile` / `pops.bind` are the ONLY public
# compile/bind entry points; the low-level `compile_problem` / `CompiledProblem` leave the surface
# (reachable as `pops.codegen.*`). `pops.CompiledArtifact` (a Protocol) types the inspectable handle.
def __getattr__(name):
    if name in ("compile", "bind"):
        from .codegen import orchestration
        return getattr(orchestration, name)
    if name == "RuntimePolicies":  # ADC-562: typed runtime-policy bundle
        return output.RuntimePolicies  # noqa: E501
    if name == "CompiledArtifact":
        from .codegen.compiled_artifact import CompiledArtifact
        return CompiledArtifact
    if name in ("compile_problem", "CompiledProblem"):
        raise AttributeError(
            "pops.%s left the public surface (ADC-523): use pops.compile(...) / pops.bind(...) as "
            "the front doors; the low-level driver stays reachable as pops.codegen.%s." % (name, name))
    if name == "Case":
        raise AttributeError(
            "pops.Case was renamed to pops.Problem (ADC-553/ADC-526), no alias: use pops.Problem(...).")
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
