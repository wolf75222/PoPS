"""Source/type-only probes complement Host tests; no GPU execution is claimed."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def test_actual_execution_type_rejects_foreign_backend_and_stream(tmp_path):
    source = (ROOT / "include/pops/runtime/system/prepared_field_solver_component.hpp").read_text()
    admission = source[source.index("template <class MemorySpace>"):source.index(
        "template <class ExecutionSpace>\nExecutionSpace execution_instance")]
    probe = tmp_path / "admission.cpp"
    probe.write_text(r'''
#include <pops/runtime/config/generated_component_abi.hpp>
#include <type_traits>
#include <string_view>
#include <stdexcept>
#define KOKKOS_ENABLE_CUDA
#define KOKKOS_ENABLE_HIP
#define KOKKOS_ENABLE_SYCL
#define KOKKOS_ENABLE_OPENMPTARGET
namespace Kokkos {
struct HostSpace {}; struct DeviceSpace {}; struct DefaultHostExecutionSpace {};
struct Cuda { static const char* name() { return "Cuda"; } };
struct HIP { static const char* name() { return "HIP"; } };
namespace Experimental {
struct SYCL { static const char* name() { return "SYCL"; } };
struct OpenMPTarget { static const char* name() { return "OpenMPTarget"; } };
}
using DefaultExecutionSpace = Experimental::SYCL;
template<class Execution, class Memory> struct SpaceAccessibility {
  static constexpr bool accessible = !std::is_same_v<Execution, DefaultHostExecutionSpace>;
};
}
namespace component { void validate_execution_context(const PopsExecutionContextV1&) {} }
''' + admission + r'''
static_assert(borrows_native_stream<Kokkos::Cuda>);
static_assert(borrows_native_stream<Kokkos::HIP>);
static_assert(!borrows_native_stream<Kokkos::Experimental::SYCL>);
static_assert(!borrows_native_stream<Kokkos::Experimental::OpenMPTarget>);
int main() {
  PopsExecutionContextV1 e{};
  e.memory_space = POPS_MEMORY_SPACE_DEVICE_V1; e.backend_identity = "SYCL";
  e.device_identity = "sycl";
  e.compute_precision = e.accumulation_precision = e.reduction_precision = POPS_PRECISION_FLOAT64_V1;
  validate_storage_execution<Kokkos::DeviceSpace>(e);
  e.stream_handle = 42;
  try { validate_storage_execution<Kokkos::DeviceSpace>(e); return 1; }
  catch (const std::invalid_argument&) {}
  e.stream_handle = 0; e.backend_identity = "Cuda";
  try { validate_storage_execution<Kokkos::DeviceSpace>(e); return 2; }
  catch (const std::invalid_argument&) {}
  e.backend_identity = "SYCL"; e.device_identity = "hip";
  try { validate_storage_execution<Kokkos::DeviceSpace>(e); return 3; }
  catch (const std::invalid_argument&) {}
  e.device_identity = "sycl"; e.compute_precision = POPS_PRECISION_FLOAT32_V1;
  try { validate_storage_execution<Kokkos::DeviceSpace>(e); return 4; }
  catch (const std::invalid_argument&) {}
}
''')
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler is not None
    binary = tmp_path / "admission"
    subprocess.run([compiler, "-std=c++20", "-DPOPS_NATIVE_DIM=2", "-I" + str(ROOT / "include"),
                    str(probe), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_callback_guard_precedes_vote_and_finite_inspection():
    source = (ROOT / "include/pops/runtime/system/prepared_field_solver_component.hpp").read_text()
    solve = source[source.index("SolveReport execute_bound_solve_"):source.index("void prepare_identity_")] if "void prepare_identity_" in source else source[source.index("SolveReport execute_bound_solve_"):]
    assert solve.index("CallbackCompletion completion") < solve.index("component::solve_field")
    assert solve.index("completion.wait()") < solve.index("all_reduce_max(solve_error")
    assert "~CallbackCompletion() noexcept" in source
    assert "Kokkos::View<double*, Kokkos::HostSpace> host" in source
    assert "collective_preflight_([&] { finite = active_solution_is_finite(); }" in source
