"""pops : Python bindings for the PoPS library.

The core exposes generic compiled BRICKS (transport, source, elliptic right-hand side); a
MODEL composes bricks (objects), the cell-by-cell computation stays in compiled C++ (no numpy,
GPU/MPI preserved). Scenario names (diocotron...) are application-side compositions (adc_cases).

The front door is the typed assembly + compile/bind/run flow: author an inert ``pops.Problem``
(physics blocks, elliptic fields, a time scheme), compile it for a mesh layout, then bind a run::

    import pops
    from pops.codegen import Production
    from pops.mesh.cartesian import CartesianMesh
    from pops.mesh.layouts import Uniform
    from pops.fields import PoissonProblem
    problem = (pops.Problem(name="plasma")
               .block("ne", physics=model)
               .field(PoissonProblem(unknown="phi", equation=eq, solver=mg))
               .time(time_program))
    validated = pops.validate(problem)
    resolved = pops.resolve(validated,
                            layout=Uniform(CartesianMesh(n=96, periodic=False)),
                            backend=Production())
    compiled = pops.compile(resolved)
    sim = pops.bind(compiled, pops.BindInputs(initial_state={"ne": ne0}))
    sim.run(t_end=0.1, cfl=0.4)
"""
from pops import _bootstrap  # noqa: F401  (loads _pops with RTLD_GLOBAL so the production .so resolves)
from pops._bootstrap import abi_key  # noqa: F401  (module ABI key: a diagnostic, not a config export)
# ADC-585 quarantined ModelSpec; ADC-545 retired System/AmrSystem/SystemConfig/AmrSystemConfig (see __getattr__).
from pops._version import __version__  # noqa: F401
# Runtime layer (the ONLY importer of _pops): parallelism, doctor, mesh, bricks, host flux.
from pops.runtime.threading import set_threads, has_kokkos, parallel_info  # noqa: F401
from pops.runtime.doctor import doctor, capabilities  # noqa: F401
from pops.runtime.mesh import CartesianMesh, PolarMesh, AuxHalo  # noqa: F401
from pops.runtime.profile import Profile, PerformanceSummary  # noqa: F401
from pops.runtime.inspection import RuntimeInspectionReport  # noqa: F401
from pops.runtime.defaults import numerical_defaults_report  # noqa: F401
from pops.runtime.fallbacks import fallback_diagnostics_report, reset_fallback_diagnostics  # noqa: F401
from pops.runtime.bricks import (  # noqa: F401
    Scalar, FluidState, ExB, CompressibleFlux, IsothermalFlux, NoSource, PotentialForce,
    GravityForce, MagneticLorentzForce, PotentialMagneticForce, ChargeDensity, BackgroundDensity,
    GravityCoupling, Model, CompositeModel, _native_to_brick, DivEpsGrad, CompositeRhs,
    ChargeDensitySource, ElectricFieldFromPotential, EllipticModel, div_eps_grad, charge_density,
    composite_rhs, electric_field_from_potential, elliptic, EllipticSolver, Ionization, Collision,
    ThermalExchange, Spatial, FiniteVolume, Explicit, _role_to_stable, _norm_implicit, IMEX,
    SourceImplicit, SourceImplicitBE, IMEXRK, Role, CondensedSchur, ElectrostaticLorentzSchur,
    Split, Strang, Dirichlet, Neumann, Periodic,
)
__all__ = [
    "Model", "CompositeModel", "CartesianMesh", "PolarMesh", "AuxHalo",
    "Scalar", "FluidState", "ExB", "CompressibleFlux", "IsothermalFlux",
    "NoSource", "PotentialForce", "GravityForce", "MagneticLorentzForce", "PotentialMagneticForce",
    "ChargeDensity", "BackgroundDensity", "GravityCoupling",
    "Spatial", "FiniteVolume", "Explicit", "IMEX", "IMEXRK", "SourceImplicit", "SourceImplicitBE",
    "Split", "Strang", "CondensedSchur", "ElectrostaticLorentzSchur", "Role", "integrate",
    "Dirichlet", "Neumann", "Periodic",
    "elliptic", "div_eps_grad", "charge_density", "composite_rhs",
    "electric_field_from_potential", "EllipticSolver", "EllipticModel",
    "Ionization", "Collision", "ThermalExchange",
    "Profile", "PerformanceSummary", "RuntimeInspectionReport", "numerical_defaults_report",
    "fallback_diagnostics_report", "reset_fallback_diagnostics",
    "time", "model", "math", "physics", "lib", "mesh", "params", "output", "external", "fields",
    "linalg", "solvers", "experimental", "abi_key", "capabilities", "inspect", "explain",
    "ReportTree", "ReportPhase", "ReportSeverity", "DiagnosticError", "SourceSpan", "ProvenanceRecord",
    "inspect_amr", "native_capability_report", "runtime_environment_report",
    "validate_runtime_environment", "RuntimeCapabilityError",
    "set_threads", "has_kokkos", "parallel_info", "doctor", "CompiledSimulationArtifact",
    "ResolvedSimulationPlan", "BindInputs", "InstallPlan",
    "Problem", "AuthoringSnapshot", "Program", "PhysicsModel",
    "validate", "resolve", "compile", "bind", "install",
    "RuntimePolicies",
]
# Lower / authoring layers + the moved integrate (re-exported, surface unchanged; numpy-free).
from pops.runtime import integrate  # noqa: E402,F401  (pops.integrate name preserved; without numpy)
from . import time, model, math, lib, physics, mesh  # noqa: E402  (Spec 2/3 operator-first + board authoring + IR)
from . import params, output, external, fields, linalg, solvers  # noqa: E402  (Spec 5 typed params/output/fields/algebra/solvers)
from .problem import AuthoringSnapshot, Problem  # noqa: E402,F401
from .time import Program  # noqa: E402,F401
from pops.physics import PhysicsModel  # noqa: E402,F401  (Spec 5 sec.11: alias of pops.physics.Model)
# ADC-545: library trio + CompiledTime left the root (homes in __getattr__); keep pops.codegen bound.
from . import codegen  # noqa: E402,F401
from .codegen import (  # noqa: E402,F401
    BindInputs, CompiledSimulationArtifact, InstallPlan, ResolvedSimulationPlan)
from ._capabilities import (  # noqa: E402,F401  (Spec 5: descriptor-sourced matrix + native reports)
    inspect_capabilities, inspect_amr, native_capability_report)
from ._report import DiagnosticError, ReportPhase, ReportSeverity, ReportTree  # noqa: E402,F401
from .provenance import ProvenanceRecord, SourceSpan  # noqa: E402,F401
from ._inspect import explain, inspect  # noqa: E402,F401  (typed explanation + explicit dict bridge)
from .runtime_environment import RuntimeCapabilityError, runtime_environment_report, validate_runtime_environment  # noqa: E402,F401,E501
# ADC-545: retired name -> advanced home; __getattr__ raises a targeted AttributeError naming both
# the front door (pops.compile / pops.bind) and this home (no silent alias).
_ADC545_HOMES = {
    "System": "pops.runtime.system.System (built by pops.compile(layout=Uniform(...)) + pops.bind)",
    "AmrSystem": "pops.runtime.system.AmrSystem (built by pops.compile(layout=AMR(...)) + pops.bind)",
    "SystemConfig": "pops._bootstrap.SystemConfig / pops.runtime.system.SystemConfig",
    "AmrSystemConfig": "pops._bootstrap.AmrSystemConfig / pops.runtime.system.AmrSystemConfig",
    "CompiledTime": "pops.time.CompiledTime (the time language)",
    "compile_library": "pops.codegen.compile_library (advanced brick-library manifest API)",
    "read_library_manifest": "pops.codegen.read_library_manifest",
    "LibraryManifest": "pops.codegen.LibraryManifest",
}
# Lazy canonical phase front doors; retired structural/low-level compiler spellings fail below.
def __getattr__(name: str):
    if name in ("validate", "resolve", "compile", "bind", "install"):
        from .codegen import orchestration
        return getattr(orchestration, name)
    if name == "RuntimePolicies":  # ADC-562: typed runtime-policy bundle
        return output.RuntimePolicies  # noqa: E501
    if name in ("CompiledArtifact", "compile_problem", "CompiledProblem"):
        raise AttributeError(
            "pops.%s is not part of the final typed phase API; use "
            "pops.validate -> pops.resolve -> pops.compile -> pops.bind." % name)
    if name == "Case":
        raise AttributeError(
            "pops.Case was renamed to pops.Problem (ADC-553/ADC-526), no alias: use pops.Problem(...).")
    if name in _ADC545_HOMES:
        raise AttributeError("pops.%s left the public surface (ADC-545): use %s." % (
            name, _ADC545_HOMES[name]))
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
