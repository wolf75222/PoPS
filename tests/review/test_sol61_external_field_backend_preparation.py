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
    assert solve.index("completion.emplace(") < solve.index("component::solve_field")
    assert solve.index("callback preparation failed collectively") < solve.index("component::solve_field")
    assert solve.index("completion->wait()") < solve.index("completion.reset()")
    assert solve.index("completion.reset()") < solve.index("all_reduce_max(solve_error")
    assert "~CallbackCompletion() noexcept" in source
    assert "Kokkos::View<double*, Kokkos::HostSpace> host" in source
    assert "collective_preflight_([&] { finite = active_solution_is_finite(); }" in source
    assert "report preparation failed collectively" in source


def test_rank_local_callback_preparation_failure_is_voted_before_dispatch(tmp_path):
    """Execute the actual adapter prefix/guard with two Host threads and a vote simulator.

    Injection covers wrapper construction/runtime_error, wrapper allocation/bad_alloc, table
    lookup/bad_alloc and a callback throw. This is an ordering proof, not Native MPI or GPU.
    """
    source = (ROOT / "include/pops/runtime/system/prepared_field_solver_component.hpp").read_text()
    guard = source[source.index("template <class ExecutionSpace>\nExecutionSpace execution_instance"):
                   source.index("/// Inspect only active valid cells")]
    dispatch = source[source.index("  template <class FiniteCheck>\n  SolveReport execute_bound_solve_"):
                      source.index("    std::string exact_report;")]
    preflight = source[source.index("  template <class Function>\n  static void collective_preflight_"):
                       source.index("  void prepare_provider_contract_")]
    probe = tmp_path / "preparation.cpp"
    probe.write_text(r'''
#include <pops/runtime/config/generated_component_abi.hpp>
#include <array>
#include <atomic>
#include <barrier>
#include <exception>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
thread_local int rank = -1;
std::atomic<int> mode{0}, vote_calls{0}, unsafe_vote{0};
std::array<std::atomic<int>, 2> callbacks{}, fences{};
std::array<long, 2> vote_slots{};
std::barrier rendezvous(2);
long all_reduce_max(long value) {
  if (callbacks[rank] && !fences[rank]) unsafe_vote = 1;
  ++vote_calls;
  vote_slots[rank] = value;
  rendezvous.arrive_and_wait();
  const long result = vote_slots[0] > vote_slots[1] ? vote_slots[0] : vote_slots[1];
  rendezvous.arrive_and_wait();
  return result;
}
int n_ranks() { return 2; }
namespace Kokkos {
struct DefaultExecutionSpace {
  DefaultExecutionSpace() {
    if (rank == 0 && mode == 1) throw std::runtime_error("injected stream wrapping failure");
    if (rank == 0 && mode == 2) throw std::bad_alloc();
  }
  void fence() { ++fences[rank]; }
};
void fence() {} // does not substitute for the exact-instance join checked above
}
namespace field_solver_component_detail {
''' + guard + r'''
}
struct ExecutionOwner { PopsExecutionContextV1 view() const { return {}; } };
struct Component {
  template<class Api> const Api& table(int, int) {
    if (rank == 0 && mode == 4) throw std::bad_alloc();
    static Api api{}; return api;
  }
};
namespace component {
int solve_field(const PopsFieldSolverApiV2&, void*, const int&, PopsSolveReportV2&) {
  ++callbacks[rank];
  if (rank == 0 && mode == 3) throw std::runtime_error("callback enqueued work then threw");
  return 0;
}
}
struct SolveReport {};
class Fixture {
 public:
  ExecutionOwner owner;
  struct Spec { ExecutionOwner* execution; int solver_interface_version = 2; } spec_{&owner};
  Component component;
  Component* solver_component_ = &component;
  void* solver_state_ = nullptr;
  int request = 0;
  int* solver_request_ = &request;
''' + dispatch + '    return {};\n  }\n' + preflight + r'''
};
int main() {
  for (int scenario : {1, 2, 4, 3, 0}) {
    mode = scenario; vote_calls = 0; unsafe_vote = 0;
    for (int r = 0; r < 2; ++r) { callbacks[r] = 0; fences[r] = 0; }
    std::array<bool,2> failed{};
    std::array<std::thread,2> workers;
    for (int r = 0; r < 2; ++r) workers[r] = std::thread([&, r] {
      rank = r;
      Fixture fixture;
      try { fixture.execute_bound_solve_([] { return true; }); }
      catch (const std::runtime_error& error) {
        const std::string reason(error.what());
        failed[r] = reason == (scenario == 3 ? "external FieldSolver execution failed collectively"
                  : "external FieldSolver callback preparation failed collectively");
      }
    });
    for (auto& worker : workers) worker.join();
    const bool prep_failure = scenario == 1 || scenario == 2 || scenario == 4;
    if (callbacks[0] + callbacks[1] != (prep_failure ? 0 : 2)) return 10 + scenario;
    if (failed[0] != (scenario != 0) || failed[1] != (scenario != 0)) return 20 + scenario;
    if (vote_calls != (prep_failure ? 2 : 4) || unsafe_vote) return 30 + scenario;
    if (fences[1] != 1 || ((scenario == 3 || scenario == 4 || scenario == 0) && fences[0] != 1))
      return 40 + scenario;
  }
}
''')
    compiler = shutil.which("clang++") or shutil.which("c++")
    assert compiler is not None
    binary = tmp_path / "preparation"
    subprocess.run([compiler, "-std=c++20", "-pthread", "-DPOPS_NATIVE_DIM=2",
                    "-I" + str(ROOT / "include"), str(probe), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=15)
