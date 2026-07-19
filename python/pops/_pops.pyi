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
    "__kokkos_contract__",
    "__has_mpi__",
    "__has_parallel_hdf5__",
    "__native_loader_contract__",
    "__mpi_contract__",
    "__aux_named_base__",
    "__aux_max_extra__",
    "__aux_base_comps__",
    "__aux_max_comps__",
    "__max_runtime_params__",
    "__aux_canonical__",
    "abi_key",
    "my_rank",
    "n_ranks",
    "mpi_world",
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
__kokkos_contract__: dict[str, object] | None
__has_mpi__: bool
__has_parallel_hdf5__: bool
__native_loader_contract__: dict[str, object]
__mpi_contract__: dict[str, object] | None
__aux_named_base__: int
__aux_max_extra__: int
__aux_base_comps__: int
__aux_max_comps__: int
__max_runtime_params__: int
__aux_canonical__: dict[str, int]


class StepAttemptRejected(RuntimeError): ...


class _NativeMpiDatatype:
    """Non-constructible native MPI datatype identity owned by the process world."""

    identity: str
    fortran_handle: int


class _NativeWorldCommunicator:
    """Non-constructible exact native process-world authority."""

    rank: int
    size: int
    active: bool
    identity: str
    initialized_by_pops: bool
    atexit_finalize_registered: bool
    thread_level: int
    fortran_handle: int
    datatype_float64: _NativeMpiDatatype
    def is_float64_datatype(self, candidate: object) -> bool: ...
    def barrier(self) -> None: ...
    def broadcast_bytes(self, payload: bytes, root: int = 0) -> bytes: ...
    def allgather_bytes(self, payload: bytes) -> tuple[bytes, ...]: ...
    def gather_bytes(
        self, payload: bytes, root: int = 0
    ) -> tuple[bytes, ...] | None: ...


class _SolveReport:
    iters: int
    rel_residual: float
    reference_residual_norm: float
    residual_norm: float
    status: str
    action: str
    reason: str
    def valid(self) -> bool: ...
    def solved(self) -> bool: ...
    def solved_value_available(self) -> bool: ...
    def failed(self) -> bool: ...


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
    def solve_fields(self) -> _SolveReport: ...
    def output_state_local_pieces(
        self, block: str, level: int
    ) -> tuple[dict[str, object], ...]: ...
    def output_field_local_pieces(
        self, provider_slot: str, level: int
    ) -> tuple[dict[str, object], ...]: ...
    def output_state_root_pieces(
        self, world: _NativeWorldCommunicator, block: str, level: int
    ) -> tuple[dict[str, object], ...]: ...
    def output_field_root_pieces(
        self, world: _NativeWorldCommunicator, provider_slot: str, level: int
    ) -> tuple[dict[str, object], ...]: ...


class AmrSystem:
    def __init__(self, config: AmrSystemConfig) -> None: ...
    def materialize_program_restart_histories(
        self,
        payload: bytes,
        names: list[str],
        depths: list[int],
        ncomps: list[int],
    ) -> None: ...
    def output_state_local_pieces(
        self, block: str, level: int
    ) -> tuple[dict[str, object], ...]: ...
    def output_field_local_pieces(
        self, provider_slot: str, level: int
    ) -> tuple[dict[str, object], ...]: ...
    def output_state_root_pieces(
        self, world: _NativeWorldCommunicator, block: str, level: int
    ) -> tuple[dict[str, object], ...]: ...
    def output_field_root_pieces(
        self, world: _NativeWorldCommunicator, provider_slot: str, level: int
    ) -> tuple[dict[str, object], ...]: ...


def abi_key() -> str: ...
def my_rank() -> int: ...
def n_ranks() -> int: ...
def mpi_world() -> _NativeWorldCommunicator: ...
def module_capabilities(target: str = "module") -> dict[str, object]: ...
def capability_report(target: str = "module") -> dict[str, object]: ...
def runtime_environment_report() -> dict[str, object]: ...
def runtime_backend_manifest(
    backend: str, target: str, communicator: str
) -> dict[str, object]: ...
def numerical_defaults_report() -> dict[str, object]: ...
def fallback_diagnostics_report() -> dict[str, object]: ...
def reset_fallback_diagnostics() -> None: ...
def kokkos_is_initialized() -> bool: ...


# Private native parallel-HDF5 provider.  The world argument is the exact non-fabricable native
# authority; manifest/array descriptors are validated by the binding before the C API is entered.
def _parallel_hdf5_capability() -> dict[str, object]: ...
def _write_parallel_hdf5(
    world: _NativeWorldCommunicator,
    path: str,
    manifest_json: str,
    root_arrays: dict[str, object],
    fields: tuple[dict[str, object], ...],
) -> None: ...


# Private native identity helpers used by the Python implementation and its
# native-parity tests.  ``object`` is intentional: C++ validates the closed
# serializable domain and raises TypeError for every unsupported value.
def _identity_canonical_bytes(value: object) -> bytes: ...
def _identity_sha256(value: object) -> str: ...
def _component_manifest_canonical_bytes(value: object) -> bytes: ...
def _component_manifest_semantic_bytes(value: object) -> bytes: ...
