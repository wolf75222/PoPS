from __future__ import annotations


# This extension is an implementation detail of ``pops``.  ``__all__`` is the
# complete supported direct surface; the config/model/engine declarations below
# are only the typed seam consumed by ``pops._bootstrap``.
__all__ = (
    "StepAttemptRejected",
    "__version__",
    "__abi_version__",
    "__release_contract_sha256__",
    "__public_api_version__",
    "__semantic_ir_version__",
    "__normalization_version__",
    "__component_registry_version__",
    "__checkpoint_schema_version__",
    "__cxx_std__",
    "__cxx_compiler__",
    "__has_kokkos__",
    "__has_mpi__",
    "__mpi_include__",
    "__aux_named_base__",
    "__aux_max_extra__",
    "__aux_base_comps__",
    "__aux_max_comps__",
    "__max_runtime_params__",
    "__aux_canonical__",
    "abi_key",
    "my_rank",
    "n_ranks",
    "module_capabilities",
    "capability_report",
    "runtime_environment_report",
    "runtime_backend_manifest",
    "numerical_defaults_report",
    "fallback_diagnostics_report",
    "reset_fallback_diagnostics",
    "kokkos_is_initialized",
)


__version__: str
__abi_version__: int
__release_contract_sha256__: str
__public_api_version__: int
__semantic_ir_version__: int
__normalization_version__: int
__component_registry_version__: int
__checkpoint_schema_version__: int
__cxx_std__: int
__cxx_compiler__: str
__has_kokkos__: bool
__has_mpi__: bool
__mpi_include__: str
__aux_named_base__: int
__aux_max_extra__: int
__aux_base_comps__: int
__aux_max_comps__: int
__max_runtime_params__: int
__aux_canonical__: dict[str, int]


class StepAttemptRejected(RuntimeError): ...


# Internal bootstrap seam: these data PODs are re-exported from pops.runtime,
# not from the private native module's supported direct API.
class SystemConfig:
    n: int
    L: float
    periodic: bool
    geometry: str
    nr: int
    ntheta: int
    r_min: float
    r_max: float
    theta_boxes: int
    def __init__(self) -> None: ...


class AmrSystemConfig:
    n: int
    L: float
    regrid_every: int
    level_count: int
    regrid_grow: int
    regrid_margin: int
    explicit_bootstrap: bool
    periodic: bool
    distribute_coarse: bool
    coarse_max_grid: int
    cluster_min_efficiency: float
    cluster_min_box_size: int
    cluster_max_box_size: int
    def __init__(self) -> None: ...


class ModelSpec:
    transport: str
    source: str
    elliptic: str
    B0: float
    gamma: float
    cs2: float
    vacuum_floor: float
    qom: float
    q: float
    alpha: float
    n0: float
    sign: float
    four_pi_G: float
    rho0: float
    frozen: bool
    def __init__(self) -> None: ...
    def freeze(self) -> None: ...
    def _semantic_data(self) -> dict[str, str | float]: ...
    def _pops_freeze_snapshot(self, capability: object) -> bool: ...
    def _pops_freeze_restore(self, capability: object, state: bool) -> None: ...


# Internal native engines.  Their operational methods deliberately have no
# dynamic fallback in the stub: a new bootstrap use must be declared explicitly.
class System:
    def __init__(self, config: SystemConfig) -> None: ...


class AmrSystem:
    def __init__(self, config: AmrSystemConfig) -> None: ...


def abi_key() -> str: ...
def my_rank() -> int: ...
def n_ranks() -> int: ...
def module_capabilities(target: str = "module") -> dict[str, object]: ...
def capability_report(target: str = "module") -> dict[str, object]: ...
def runtime_environment_report() -> dict[str, object]: ...
def runtime_backend_manifest(backend: str, target: str) -> dict[str, object]: ...
def numerical_defaults_report() -> dict[str, object]: ...
def fallback_diagnostics_report() -> dict[str, object]: ...
def reset_fallback_diagnostics() -> None: ...
def kokkos_is_initialized() -> bool: ...


# Private native identity helpers used by the Python implementation and its
# native-parity tests.  ``object`` is intentional: C++ validates the closed
# serializable domain and raises TypeError for every unsupported value.
def _identity_canonical_bytes(value: object) -> bytes: ...
def _identity_sha256(value: object) -> str: ...
def _component_manifest_canonical_bytes(value: object) -> bytes: ...
def _component_manifest_semantic_bytes(value: object) -> bytes: ...
