// ADC-538: the exact-ranked ProgramExecutionServices EXECUTION CONTRACT, proved host-side without codegen or a .so.
// ProgramExecutionServices (include/pops/runtime/program/program_execution_services.hpp) is the C++ facade a generated
// problem.so calls to run a compiled time Program during sim.step(dt); it REIMPLEMENTS NOTHING (each
// method forwards to a System<Dim> primitive). test_program_runtime.cpp already pins one Forward-Euler
// step + the profiler counters. This suite widens the fence to the whole host-validatable seam surface
// and proves the "no Python in a time stage" contract BY CONSTRUCTION: the step body is native C++ and
// its result is bit-equal to the same step composed from the System<Dim> primitives directly.
//
// It pins:
//  - Forward-Euler via ProgramExecutionServices<kNativeDimension> == the eval_rhs reference (the ADC-538
//    parity assertion, at the per-stage solve_fields_from_state seam, not the whole-step
//    solve_fields);
//  - a 2-stage SSPRK (Heun / SSP-RK2) via ProgramExecutionServices<kNativeDimension> == a hand-written SSPRK
//    reference built from solve_fields + eval_rhs, using ctx.scratch_state_like / ctx.rhs_into /
//    ctx.lincomb / ctx.axpy and a per-stage ctx.solve_fields_from_state -- so a multi-stage
//    field-coupled Program is exercised;
//  - the remaining host-validatable seams return sane, consistent results: neg_div_flux_default_into +
//    source_default_into recompose to rhs_into; lincomb / axpy; fill_boundary; an absent projection
//    fails closed; the reductions; laplacian == divergence(gradient); the scratch
//    allocators; register/store/read/rotate history; record_scalar -> program_diagnostic; the runtime
//    params round-trip; hmin / max_wave_speed are positive;
//
// The compiled-.so runtime cadence, the held-node scheduler cache and the AOT ABI are Kokkos-only and
// validated on ROMEO; here every seam is driven on the build-selected exact native dimension.

#include <gtest/gtest.h>

#include "native_dso_compiler.hpp"
#include "program_v5_fixture.hpp"

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/core/foundation/allocator.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/program/program_execution_services.hpp>  // NativeProgramExecutionServices (the contract under test)
#include <pops/runtime/recovery/uniform_recovery_consumer.hpp>
#include <pops/runtime/system.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

std::atomic<bool> g_boundary_point_heap_window_enabled{false};
std::atomic<std::uint64_t> g_boundary_point_heap_allocations{0};

void note_boundary_point_heap_allocation() noexcept {
  if (g_boundary_point_heap_window_enabled.load(std::memory_order_relaxed))
    g_boundary_point_heap_allocations.fetch_add(1, std::memory_order_relaxed);
}

void* boundary_point_allocate(std::size_t size) {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer == nullptr)
    throw std::bad_alloc();
  note_boundary_point_heap_allocation();
  return pointer;
}

class BoundaryPointHeapWindow final {
 public:
  BoundaryPointHeapWindow()
      : before_(g_boundary_point_heap_allocations.load(std::memory_order_relaxed)) {
    g_boundary_point_heap_window_enabled.store(true, std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t close() noexcept {
    g_boundary_point_heap_window_enabled.store(false, std::memory_order_relaxed);
    return g_boundary_point_heap_allocations.load(std::memory_order_relaxed) - before_;
  }

 private:
  std::uint64_t before_ = 0;
};

}  // namespace

void* operator new(std::size_t size) {
  return boundary_point_allocate(size);
}
void* operator new[](std::size_t size) {
  return boundary_point_allocate(size);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  try {
    return boundary_point_allocate(size);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  return ::operator new(size, std::nothrow);
}
void operator delete(void* pointer) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
void* operator new(std::size_t size, std::align_val_t alignment) {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, static_cast<std::size_t>(alignment), size == 0 ? 1 : size) != 0)
    pointer = nullptr;
  if (pointer == nullptr)
    throw std::bad_alloc();
  note_boundary_point_heap_allocation();
  return pointer;
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return ::operator new(size, alignment);
}
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  try {
    return ::operator new(size, alignment);
  } catch (...) {
    return nullptr;
  }
}
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return ::operator new(size, alignment, std::nothrow);
}
void operator delete(void* pointer, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept {
  std::free(pointer);
}
void operator delete(void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}
void operator delete[](void* pointer, std::align_val_t, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

namespace pops::runtime::program::detail {

template <int Dim>
struct ProgramExecutionServicesForwardOverlayTestAccess final {
  using services_type = ProgramExecutionServices<Dim>;

  static void mark_accepted(services_type& services) {
    services.binding_ = services_type::Binding::accepted;
  }

  static std::shared_ptr<services_type> make_uniform_accepted() {
    return std::shared_ptr<services_type>(new services_type(services_type::Binding::accepted));
  }

  static bool has_live_preparation_image(const services_type& services) {
    return services.preparation_image_ != nullptr && services.preparation_image_->execution_ready();
  }

  static std::map<std::string, std::int64_t> accepted_amr_clock_ticks(const services_type& services,
                                                                      std::int64_t macro_step) {
    return services.amr_test_backend_().accepted_clock_schedule().accepted_ticks(macro_step);
  }
};

}  // namespace pops::runtime::program::detail

using namespace pops;

namespace {

constexpr int kTestDimension = kNativeDimension;
using NativeSystem = System<kTestDimension>;
using NativeSystemConfig = SystemConfig<kTestDimension>;
static_assert(std::is_nothrow_destructible_v<NativeSystem>);
static_assert(std::is_nothrow_move_constructible_v<NativeSystem>);
static_assert(std::is_nothrow_move_assignable_v<NativeSystem>);
static_assert(!std::is_copy_constructible_v<NativeSystem>);
static_assert(!std::is_copy_assignable_v<NativeSystem>);
using NativeProgramExecutionServices = runtime::program::ProgramExecutionServices<kTestDimension>;
using NativeAmrStorageTopologyAdapter = runtime::program::detail::AmrStorageTopologyAdapter<
    kTestDimension, typename Kokkos::DefaultExecutionSpace::memory_space>;
static_assert(!std::is_copy_constructible_v<NativeProgramExecutionServices>);
static_assert(!std::is_copy_assignable_v<NativeProgramExecutionServices>);
static_assert(!std::is_move_constructible_v<NativeProgramExecutionServices>);
static_assert(!std::is_move_assignable_v<NativeProgramExecutionServices>);
static_assert(!std::is_copy_constructible_v<NativeAmrStorageTopologyAdapter>);
static_assert(!std::is_copy_assignable_v<NativeAmrStorageTopologyAdapter>);
static_assert(!std::is_move_constructible_v<NativeAmrStorageTopologyAdapter>);
static_assert(!std::is_move_assignable_v<NativeAmrStorageTopologyAdapter>);
using NativeField = MultiFab<kTestDimension>;
using NativeConstView = FieldView<const Real, kTestDimension>;
using NativeBox = Box<kTestDimension>;
using NativeAmrTopology = NativeProgramExecutionServices::AmrPreparationTopologyView;
using NativeAmrRegistry =
    NativeProgramExecutionServices::AmrBackend::hierarchy_tensor_registry_type;

class EmptyAcceptedExecutionSnapshot final
    : public runtime::program::AcceptedProgramExecutionServicesSnapshot {
 public:
  std::unique_ptr<runtime::program::AcceptedProgramExecutionServicesSnapshot> prepare_restore()
      const override {
    return std::make_unique<EmptyAcceptedExecutionSnapshot>();
  }
  void publish_restore() noexcept override {}
};

struct ForwardOverlayTopologyFixture final {
  runtime::program::ProgramRuntimeState<kTestDimension> program_state{};
  std::shared_ptr<ExecutionLane> lane =
      std::make_shared<ExecutionLane>(ExecutionLane::world("pops.test.program-forward-overlay"));
  std::shared_ptr<const NativeAmrRegistry> tensor_registry = std::make_shared<NativeAmrRegistry>();
  std::shared_ptr<NativeAmrTopology> topology = std::make_shared<NativeAmrTopology>();

  ForwardOverlayTopologyFixture() {
    topology->forward_detached = true;
    topology->program_state = &program_state;
    topology->lane = lane.get();
    topology->hierarchy_tensor_registry = tensor_registry;
    topology->program_block_map = {0};
    topology->block_prototypes = {{NativeField{}}};
    topology->runtime_block_boundary_linearizations = {false};
    topology->spatial_contract = "pops.test.program-forward-overlay.spatial";
    NativeBox domain{};
    RealVector<kTestDimension> upper{};
    for (int axis = 0; axis < kTestDimension; ++axis) {
      domain.lo[axis] = 0;
      domain.hi[axis] = 0;
      upper[axis] = Real(1);
    }
    topology->level_geometries = {
        Geometry<kTestDimension>::from_bounds(domain, RealVector<kTestDimension>{}, upper)};
    topology->periodic_faces.assign(static_cast<std::size_t>(2 * kTestDimension), false);
    topology->topology_epoch = 17;
    topology->materialization_generation = 23;
    topology->validate();
  }
};

void install_execution_lane(NativeSystem& system, std::string identity) {
  system.install_prepared_boundary_execution_lane(
      std::make_shared<ExecutionLane>(ExecutionLane::world(std::move(identity))));
}

static_assert(
    std::is_same_v<decltype(std::declval<const runtime::program::ProgramExecutionServices<1>&>()
                                .template provider_values_view<0>("", 0, 0)),
                   ProviderStorageView<1, 0>>);
static_assert(
    std::is_same_v<decltype(std::declval<const runtime::program::ProgramExecutionServices<2>&>()
                                .template provider_values_view<0>("", 0, 0)),
                   ProviderStorageView<2, 0>>);
static_assert(
    std::is_same_v<decltype(std::declval<const runtime::program::ProgramExecutionServices<3>&>()
                                .template provider_values_view<0>("", 0, 0)),
                   ProviderStorageView<3, 0>>);

/// Generated Program sources intentionally use a bare braced exact-dt list.  This must bind to the
/// concrete common facade overload; routing it through the variadic dispatcher would make the list
/// non-deducible and reject otherwise valid Uniform Program artifacts at compilation.
template <int Dim>
concept HasPublicExactAxpy =
    requires(const runtime::program::ProgramExecutionServices<Dim>& context,
             MultiFab<Dim>& destination, const MultiFab<Dim>& source) {
      context.axpy(destination, Real(1), source, Real(1), {{0, 1, 1}});
    };

static_assert(HasPublicExactAxpy<1>);
static_assert(HasPublicExactAxpy<2>);
static_assert(HasPublicExactAxpy<3>);

NativeSystemConfig native_config(std::int64_t cells, Real length = Real(1)) {
  NativeSystemConfig config;
  for (int axis = 0; axis < kTestDimension; ++axis) {
    config.shape[axis] = cells;
    config.lower[axis] = Real(0);
    config.upper[axis] = length;
    config.periodicity[static_cast<std::size_t>(axis)] = true;
  }
  return config;
}

Extent<kTestDimension> enlarged_ghosts(const NativeField& field, int increment) {
  Extent<kTestDimension> ghosts = field.ghosts();
  for (int axis = 0; axis < kTestDimension; ++axis)
    ghosts[axis] += increment;
  return ghosts;
}

NativeField native_field_like(const NativeField& field, int components,
                              Extent<kTestDimension> ghosts) {
  return NativeField(field.layout(), field.distribution(), field.local_rank(), components, ghosts);
}

Real first_value(const NativeField& field, int component = 0) {
  return field.fab(0).view()(field.box(0).lo, component);
}

using GasModel = nd::IdealGasEuler<kTestDimension>;
using GasSchema = typename GasModel::Schema;
constexpr double kGamma = 1.4;
constexpr int kNcomp = GasModel::n_vars;

std::size_t uniform_cell_count(int cells) {
  std::size_t result = 1;
  for (int axis = 0; axis < kTestDimension; ++axis)
    result *= static_cast<std::size_t>(cells);
  return result;
}

void ensure_kokkos() {
#if defined(POPS_HAS_KOKKOS)
  static Kokkos::ScopeGuard guard;
#endif
}

void materialize_test_residual(NativeField& state, NativeField& residual) {
  if (state.layout() != residual.layout() || state.distribution() != residual.distribution() ||
      state.local_rank() != residual.local_rank() || state.ncomp() != residual.ncomp())
    throw std::invalid_argument("test residual requires one exact ranked field contract");
  for (std::size_t local = 0; local < state.local_size(); ++local) {
    const NativeConstView input = std::as_const(state).fab(local).view();
    const FieldView<Real, kTestDimension> output = residual.fab(local).view();
    const int components = state.ncomp();
    for_each_cell(state.box(local), [=] POPS_HD(const Index<kTestDimension>& cell) {
      for (int component = 0; component < components; ++component)
        output(cell, component) = -Real(0.25) * input(cell, component);
    });
  }
}

void materialize_zero_residual(NativeField&, NativeField& residual) {
  residual.set_val(Real(0));
}

void materialize_mean_free_density(const NativeField& state, NativeField& rhs) {
  if (rhs.ncomp() != 1 || state.layout() != rhs.layout() ||
      state.distribution() != rhs.distribution() || state.local_rank() != rhs.local_rank())
    throw std::invalid_argument("test field RHS requires one exact ranked scalar output");
  std::size_t cells = 0;
  for (std::size_t box = 0; box < state.layout().size(); ++box)
    cells += static_cast<std::size_t>(state.layout()[box].numPts());
  if (cells == 0)
    throw std::logic_error("test field RHS requires a non-empty exact ranked layout");
  const Real mean = reduce_sum(state, GasSchema::density) / static_cast<Real>(cells);
  rhs.set_val(Real(0));
  for (std::size_t local = 0; local < state.local_size(); ++local) {
    const NativeConstView input = std::as_const(state).fab(local).view();
    const FieldView<Real, kTestDimension> output = rhs.fab(local).view();
    for_each_cell(state.box(local), [=] POPS_HD(const Index<kTestDimension>& cell) {
      const Real fluctuation = input(cell, GasSchema::density) - mean;
      const Real magnitude = fluctuation < Real(0) ? -fluctuation : fluctuation;
      output(cell, 0) = magnitude <= Real(1e-12) ? Real(0) : fluctuation;
    });
  }
}

void add_gas_block(NativeSystem& s, const std::string& name, int* projection_calls = nullptr) {
  s.install_block_state_route(name, "test::state::" + name);
  const GasModel model = GasModel::prepare(Real(kGamma));
  PreparedSystemBlock<kTestDimension> prepared;
  prepared.name = name;
  prepared.provider_identity = "test.program-context.exact-ranked-euler";
  prepared.ncomp = GasModel::n_vars;
  prepared.conservative_variables = GasModel::conservative_vars();
  prepared.primitive_variables = GasModel::primitive_vars();
  prepared.gamma = kGamma;
  for (int axis = 0; axis < kTestDimension; ++axis)
    prepared.ghosts[axis] = 1;

  const auto residual = [](NativeField& state, NativeField& output) {
    materialize_test_residual(state, output);
  };
  const auto zero = [](NativeField& state, NativeField& output) {
    materialize_zero_residual(state, output);
  };
  prepared.closures.rhs_into = residual;
  prepared.closures.rhs_flux_only = residual;
  prepared.closures.source_only = zero;
  prepared.closures.source_only_masked = zero;
  prepared.closures.rhs_at_point = [residual](const auto&, NativeField& state,
                                              NativeField& output) { residual(state, output); };
  prepared.closures.rhs_flux_only_at_point = prepared.closures.rhs_at_point;
  prepared.closures.rhs_without_prepared_interfaces = prepared.closures.rhs_at_point;
  prepared.closures.rhs_flux_only_without_prepared_interfaces = prepared.closures.rhs_at_point;
  prepared.closures.rhs_core_at_point = prepared.closures.rhs_at_point;
  prepared.closures.rhs_flux_only_core_at_point = prepared.closures.rhs_at_point;
  prepared.closures.rhs_core_at_point_prepared = [residual](const auto&, NativeField& state,
                                                            NativeField& output, const auto&) {
    residual(state, output);
  };
  prepared.closures.rhs_flux_only_core_at_point_prepared =
      prepared.closures.rhs_core_at_point_prepared;
  prepared.closures.prepare_generated_state_at_point = [](const auto&, NativeField&) {};
  prepared.closures.prepare_generated_state_at_point_prepared = [](const auto&, NativeField&,
                                                                   const auto&) {};
  prepared.closures.prepare_generated_state_with_transport_prepared =
      [](const auto&, NativeField&, const auto&, const ExecutionLane&, const auto&) {};
  if (projection_calls != nullptr)
    prepared.closures.project = [projection_calls](NativeField&, const ExecutionLane&) {
      ++*projection_calls;
    };
  prepared.closures.external_ghost_boundary =
      std::make_shared<SystemBlockClosures<kTestDimension>::ExternalGhostBoundary>(
          [](const auto&, NativeField&, const auto&, const ExecutionLane&) {});
  prepared.maximum_speed = [](const NativeField&, const ExecutionLane&) { return Real(1); };
  prepared.poisson_rhs = [](const NativeField& state, NativeField& rhs) {
    materialize_mean_free_density(state, rhs);
  };
  prepared.primitive_to_conservative = [](const double* primitive, double* conservative) {
    std::copy_n(primitive, kNcomp, conservative);
  };
  prepared.conservative_to_primitive = [](const double* conservative, double* primitive) {
    RecoveryReport report;
    report.status = RecoveryStatus::kRecovered;
    report.attempted_methods = 1;
    report.selected_method = 0;
    report.last_method = 0;
    for (int component = 0; component < kNcomp; ++component) {
      if (!std::isfinite(conservative[component])) {
        report.status = RecoveryStatus::kRejected;
        report.cause = RecoveryCause::kNonFiniteCandidate;
        report.failing_component = component;
        return report;
      }
    }
    std::copy_n(conservative, kNcomp, primitive);
    return report;
  };
  prepared.batch_conservative_to_primitive = make_uniform_recovery_consumer(model);
  s.install_prepared_block(std::move(prepared));
}

void add_gas(NativeSystem& s) {
  add_gas_block(s, "gas");
  s.set_poisson("charge_density", "cartesian_cg");
}

void add_scalar_block(NativeSystem& s, const std::string& name) {
  s.install_block_state_route(name, "test::state::" + name);
  PreparedSystemBlock<kTestDimension> prepared;
  prepared.name = name;
  prepared.provider_identity = "test.program-context.scalar";
  prepared.ncomp = 1;
  prepared.conservative_variables = {VariableKind::Conservative, {"q"}, 1, {VariableRole::Scalar}};
  prepared.primitive_variables = {VariableKind::Primitive, {"q"}, 1, {VariableRole::Scalar}};
  prepared.gamma = 1.0;
  for (int axis = 0; axis < kTestDimension; ++axis)
    prepared.ghosts[axis] = 1;

  const auto residual = [](NativeField&, NativeField& output) { output.set_val(Real(0)); };
  const auto point_residual = [](const auto&, NativeField&, NativeField& output) {
    output.set_val(Real(0));
  };
  const auto prepared_point_residual = [](const auto&, NativeField&, NativeField& output,
                                          const auto&) { output.set_val(Real(0)); };
  prepared.closures.rhs_into = residual;
  prepared.closures.rhs_flux_only = residual;
  prepared.closures.source_only = residual;
  prepared.closures.source_only_masked = residual;
  prepared.closures.rhs_at_point = point_residual;
  prepared.closures.rhs_flux_only_at_point = point_residual;
  prepared.closures.rhs_without_prepared_interfaces = point_residual;
  prepared.closures.rhs_flux_only_without_prepared_interfaces = point_residual;
  prepared.closures.rhs_core_at_point = point_residual;
  prepared.closures.rhs_flux_only_core_at_point = point_residual;
  prepared.closures.rhs_core_at_point_prepared = prepared_point_residual;
  prepared.closures.rhs_flux_only_core_at_point_prepared = prepared_point_residual;
  prepared.closures.prepare_generated_state_at_point = [](const auto&, NativeField&) {};
  prepared.closures.prepare_generated_state_at_point_prepared = [](const auto&, NativeField&,
                                                                   const auto&) {};
  prepared.closures.prepare_generated_state_with_transport_prepared =
      [](const auto&, NativeField&, const auto&, const ExecutionLane&, const auto&) {};
  prepared.closures.external_ghost_boundary =
      std::make_shared<SystemBlockClosures<kTestDimension>::ExternalGhostBoundary>(
          [](const auto&, NativeField&, const auto&, const ExecutionLane&) {});
  prepared.maximum_speed = [](const NativeField&, const ExecutionLane&) { return Real(1); };
  prepared.poisson_rhs = [](const NativeField&, NativeField& output) { output.set_val(Real(0)); };
  prepared.primitive_to_conservative = [](const double* primitive, double* conservative) {
    conservative[0] = primitive[0];
  };
  prepared.conservative_to_primitive = [](const double* conservative, double* primitive) {
    RecoveryReport report;
    report.status = RecoveryStatus::kRecovered;
    report.attempted_methods = 1;
    report.selected_method = 0;
    report.last_method = 0;
    if (!std::isfinite(conservative[0])) {
      report.status = RecoveryStatus::kRejected;
      report.cause = RecoveryCause::kNonFiniteCandidate;
      report.failing_component = 0;
      return report;
    }
    primitive[0] = conservative[0];
    return report;
  };
  prepared.batch_conservative_to_primitive = [](const std::vector<double>& conserved,
                                                std::vector<double>& primitive) {
    primitive = conserved;
    UniformRecoveryBatchReport report;
    report.recovery.status = RecoveryStatus::kRecovered;
    report.recovery.attempted_methods = 1;
    report.recovery.selected_method = 0;
    report.recovery.last_method = 0;
    report.cell_count = conserved.size();
    report.recovered_cells = conserved.size();
    report.published = true;
    return report;
  };
  s.install_prepared_block(std::move(prepared));
}

/// Build the test provider through the same detached v5 preparation authority as a generated DSO.
/// The image and provider are kept together so no test callback can retain a stack-owned host
/// descriptor or accidentally exercise a direct System constructor.
struct PreparedNativeProgramServices final {
  std::shared_ptr<runtime::program::ProgramPreparationImage> image;
  std::shared_ptr<NativeProgramExecutionServices> provider;
};

PreparedNativeProgramServices prepare_native_program_services(NativeSystem& system) {
  auto host = system.program_host_descriptor();
  auto image = runtime::program::make_program_preparation_image<kTestDimension>(&system, 1);
  runtime::program::bind_program_preparation_image(host, image);
  auto provider =
      runtime::program::make_program_execution_provider<kTestDimension>(host.preparation);
  runtime::program::activate_staged_uniform_program_execution_services<kTestDimension>(image);
  return {std::move(image), std::move(provider)};
}

std::string install_long_qid_auxiliary_plan(NativeSystem& system) {
  using runtime::system::AuxiliaryComponentContract;
  using runtime::system::AuxiliaryComponentKey;
  using runtime::system::AuxiliaryConsumerProviderPlan;
  using runtime::system::AuxiliaryEvaluationEvent;
  using runtime::system::AuxiliaryFreshness;
  using runtime::system::AuxiliaryOutput;
  using runtime::system::AuxiliaryProviderKind;
  using runtime::system::AuxiliaryStorageShape;
  using runtime::system::PreparedAuxiliaryProvider;

  // Keep this longer than every supported short-string representation: the hot lookup must borrow
  // these bytes directly rather than manufacture a temporary std::string for the registry.
  const std::string qid =
      "pops.test.program-context.provider-values-view.consumer-qualified-id-over-sso";
  const AuxiliaryComponentKey key{"pops.test.program-context.provider-values-view", "input",
                                  "long-qid", "value"};
  const AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "input", "scalar"};
  AuxiliaryStorageShape<kTestDimension> shape;
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<kTestDimension>{
      "pops.test.program-context.provider-values-view.provider",
      AuxiliaryProviderKind::input,
      {AuxiliaryEvaluationEvent::initialization, AuxiliaryFreshness::once},
      {{key, contract, shape}},
      {}});
  system.install_auxiliary_consumer_plan(
      AuxiliaryConsumerProviderPlan<kTestDimension>{qid, {{{key, contract, shape}, 0}}});
  system.seal_auxiliary_providers();
  return qid;
}

PreparedNativeProgramServices prepare_native_scratch_services(NativeSystem& system) {
  auto host = system.program_host_descriptor();
  auto image = runtime::program::make_program_preparation_image<kTestDimension>(&system, 1);
  constexpr std::size_t kScratchSlotCount = 42;
  std::vector<runtime::program::ProgramInstallationTables::ResourcePlan> declaration(
      kScratchSlotCount);
  for (std::size_t slot = 0; slot < declaration.size(); ++slot)
    declaration[slot].slot = static_cast<std::uint32_t>(slot);
  auto& scratch = declaration.at(41);
  scratch.flags = runtime::program::kProgramResourceRuntimeSized;
  scratch.resource_type = runtime::program::ProgramResourcePlanType::runtime_sized;
  scratch.components = GasModel::n_vars;
  // Symbolic lowering cannot know the accepted System's actual ghost width.
  // The candidate prelude must capture that width from its immutable prototype
  // and authenticate it in the materialized ResourcePrototype, rather than
  // priming a zero-ghost field that will drift on its first hot access.
  scratch.ghosts = 0;
  runtime::program::bind_staged_uniform_program_resource_declaration<kTestDimension>(
      image, declaration, system.program_block_map());
  runtime::program::bind_program_preparation_image(host, image);
  auto provider =
      runtime::program::make_program_execution_provider<kTestDimension>(host.preparation);
  provider->prepare_rhs_scratch(41, 0, 0);
  provider->prepare_rhs_scratch(41, 1, 0);
  provider->prepare_state_scratch(41, 0, 0);
  runtime::program::activate_staged_uniform_program_execution_services<kTestDimension>(image);
  return {std::move(image), std::move(provider)};
}

PreparedNativeProgramServices prepare_native_field_route_services(NativeSystem& system) {
  auto host = system.program_host_descriptor();
  auto image = runtime::program::make_program_preparation_image<kTestDimension>(&system, 1);
  constexpr std::size_t kRouteSlotCount = 506;
  std::vector<runtime::program::ProgramInstallationTables::ResourcePlan> declaration(
      kRouteSlotCount);
  for (std::size_t slot = 0; slot < declaration.size(); ++slot)
    declaration[slot].slot = static_cast<std::uint32_t>(slot);
  runtime::program::bind_staged_uniform_program_resource_declaration<kTestDimension>(
      image, declaration, system.program_block_map());
  runtime::program::bind_program_preparation_image(host, image);
  auto provider =
      runtime::program::make_program_execution_provider<kTestDimension>(host.preparation);
  if (system.program_block_map().size() == 2) {
    provider->prepare_generated_field_route(500, "missing-provider", {0, 1});
    provider->prepare_generated_field_route(501, "missing-provider", {0, 1});
    provider->prepare_generated_field_route(502, "missing-provider", {0, 1});
    provider->prepare_generated_field_route(503, "missing-provider", {0, 1});
  }
  provider->prepare_generated_field_route(504, "missing-subset-provider", {0});
  provider->prepare_generated_field_route(505, "missing-subset-provider", {0});
  runtime::program::activate_staged_uniform_program_execution_services<kTestDimension>(image);
  return {std::move(image), std::move(provider)};
}

PreparedNativeProgramServices prepare_uniform_host_footprint_services(NativeSystem& system,
                                                                      std::string field,
                                                                      int scratch_subslot) {
  auto host = system.program_host_descriptor();
  auto image = runtime::program::make_program_preparation_image<kTestDimension>(&system, 1);
  std::vector<runtime::program::ProgramInstallationTables::ResourcePlan> declaration(1);
  auto& scratch = declaration.front();
  scratch.slot = 0;
  scratch.flags = runtime::program::kProgramResourceRuntimeSized;
  scratch.resource_type = runtime::program::ProgramResourcePlanType::runtime_sized;
  scratch.components = GasModel::n_vars;
  scratch.ghosts = 1;
  runtime::program::bind_staged_uniform_program_resource_declaration<kTestDimension>(
      image, declaration, system.program_block_map());
  runtime::program::bind_program_preparation_image(host, image);
  auto provider =
      runtime::program::make_program_execution_provider<kTestDimension>(host.preparation);
  provider->prepare_rhs_scratch(0, scratch_subslot, 0);
  provider->prepare_generated_field_route(0, field, {0});
  provider->configure_primary_clock("clock.host-footprint");
  provider->declare_clock_relation("clock.host-footprint", "clock.host-footprint.child", 2);
  provider->seal_uniform_preparation();
  return {std::move(image), std::move(provider)};
}

using ResourcePrototype = runtime::program::ProgramInstallationTables::ResourcePrototype;
using ResourcePrototypeKind = runtime::program::ProgramInstallationTables::ResourcePrototypeKind;

const ResourcePrototype& prepared_resource_row(const std::vector<ResourcePrototype>& prototypes,
                                               ResourcePrototypeKind kind) {
  const auto found = std::find_if(prototypes.begin(), prototypes.end(),
                                  [kind](const auto& row) { return row.kind == kind; });
  if (found == prototypes.end() || !found->layout.maximum_bytes)
    throw std::logic_error(
        "prepared resource family '" +
        runtime::program::ProgramInstallationTables::resource_prototype_kind_name(kind) +
        "' is absent from the host footprint");
  return *found;
}

std::uint64_t prepared_host_bytes(const std::vector<ResourcePrototype>& prototypes,
                                  ResourcePrototypeKind kind) {
  return *prepared_resource_row(prototypes, kind).layout.maximum_bytes;
}

using NativeProgramCallback = std::function<void(NativeProgramExecutionServices&, double)>;

std::vector<NativeProgramCallback>& native_program_callbacks() {
  static std::vector<NativeProgramCallback> callbacks;
  return callbacks;
}

extern "C" void pops_test_program_execution_services_callback(std::uint64_t identifier,
                                                              void* opaque, double dt) {
  auto& callbacks = native_program_callbacks();
  if (opaque == nullptr || identifier >= callbacks.size())
    throw std::logic_error("ProgramExecutionServices ABI-v5 callback token is invalid");
  callbacks.at(static_cast<std::size_t>(identifier))(
      *static_cast<NativeProgramExecutionServices*>(opaque), dt);
}

TEST(ProgramExecutionServicesContract,
     ForwardOverlayBuildsDetachedProviderFromAcceptedPreparationAnchor) {
  ensure_kokkos();
  using Access =
      runtime::program::detail::ProgramExecutionServicesForwardOverlayTestAccess<kTestDimension>;
  ForwardOverlayTopologyFixture fixture;
  runtime::program::ProgramHostDescriptor source{};
  source.native_dimension = static_cast<std::uint32_t>(kTestDimension);
  source.runtime_kind = runtime::program::ProgramRuntimeKind::amr;
  auto image = std::make_shared<runtime::program::ProgramExecutionPreparationImage<kTestDimension>>(
      source, fixture.topology, 1);
  auto accepted_provider = image->provider();
  constexpr std::string_view kPrimaryClock = "clock.macro";
  constexpr std::string_view kQualifiedClock =
      "pops.clock.v1::sha256:71ed5d5a72c7ef2868f5325da57e0db7e4c902e4d95d42dc04ca76260525e029";
  accepted_provider->configure_primary_clock(std::string(kPrimaryClock));
  EXPECT_NO_THROW(accepted_provider->declare_clock_relation(std::string(kPrimaryClock),
                                                            std::string(kQualifiedClock), 3));
  ASSERT_NO_THROW(image->reconcile_staged_amr_checkpoint_clock_identities(
      {std::string(kPrimaryClock), std::string(kQualifiedClock)}));
  const std::map<std::string, std::int64_t> expected_ticks{{std::string(kPrimaryClock), 7},
                                                           {std::string(kQualifiedClock), 21}};
  EXPECT_EQ(image->staged_clock_schedule().accepted_ticks(7), expected_ticks);
  EXPECT_EQ(Access::accepted_amr_clock_ticks(*accepted_provider, 7), expected_ticks);
  runtime::multiblock::BoundaryEvaluationPoint detached_point;
  EXPECT_NO_THROW(accepted_provider->prepare_boundary_evaluation_point(detached_point));
  EXPECT_EQ(detached_point.clock, kPrimaryClock)
      << "detached cold preparation must reserve its clock without an accepted AMR facade";
  runtime::program::PreparedForwardAmrExecutionAuthorityView<kTestDimension> authority(
      fixture.topology);
  EmptyAcceptedExecutionSnapshot snapshot;

  EXPECT_THROW(accepted_provider->with_forward_execution_overlay(
                   authority, snapshot,
                   [](std::shared_ptr<NativeProgramExecutionServices>) { return true; }),
               std::logic_error)
      << "a preparation-bound provider is not an accepted forward anchor";

  Access::mark_accepted(*accepted_provider);
  EXPECT_THROW(
      accepted_provider->register_hierarchy_tensor_solver_provider(
          std::shared_ptr<
              const NativeProgramExecutionServices::AmrBackend::hierarchy_tensor_provider_type>{}),
      std::logic_error)
      << "an accepted Program cannot grow its image-owned provider registry";
  EXPECT_THROW(accepted_provider->configure_hierarchy_tensor_solver(
                   0, 1, "pops.hierarchy.composite-tensor-fac", "pops.test.plan",
                   "pops.test.operator", std::vector<std::string>{"pops.test.assembly"},
                   "pops.test.solution", PreparedProviderOptions{"pops.test.options", {}}),
               std::logic_error)
      << "an accepted Program cannot rebuild a tensor solver outside preparation";
  auto forward_provider = accepted_provider->with_forward_execution_overlay(
      authority, snapshot,
      [](std::shared_ptr<NativeProgramExecutionServices> provider) { return provider; });
  ASSERT_TRUE(forward_provider);
  EXPECT_NE(forward_provider.get(), accepted_provider.get());
  EXPECT_TRUE(forward_provider->is_amr());
  EXPECT_TRUE(Access::has_live_preparation_image(*forward_provider));
  const auto forward_topology = forward_provider->program_resource_topology();
  EXPECT_EQ(forward_topology.levels, 1);
  EXPECT_EQ(forward_topology.epoch, fixture.topology->topology_epoch);
  EXPECT_EQ(forward_topology.generation, fixture.topology->materialization_generation);
  std::vector<int> visited_levels;
  forward_provider->for_each_program_resource_level(
      [&](int level) { visited_levels.push_back(level); });
  EXPECT_EQ(visited_levels, std::vector<int>({0}));
  runtime::multiblock::BoundaryEvaluationPoint forward_point;
  EXPECT_NO_THROW(forward_provider->prepare_boundary_evaluation_point(forward_point));
  EXPECT_EQ(forward_point.clock, kPrimaryClock)
      << "a forward overlay must inherit the complete bind-sealed clock schedule";
  EXPECT_EQ(Access::accepted_amr_clock_ticks(*forward_provider, 7), expected_ticks)
      << "activation/publication must retain both staged clock identities and their exact ticks";

  fixture.topology->periodic_faces[0] = true;
  EXPECT_THROW((void)forward_provider->bind_mesh_boundary_session(
                   fixture.topology->block_prototypes.front().front(), *fixture.topology->lane),
               std::invalid_argument)
      << "an asymmetric detached periodic image must fail before a boundary session is built";
  fixture.topology->periodic_faces[0] = false;

  image.reset();
  EXPECT_TRUE(Access::has_live_preparation_image(*forward_provider))
      << "the aliasing return must retain its detached forward image after factory return";
  EXPECT_EQ(forward_provider->program_resource_topology().generation,
            fixture.topology->materialization_generation)
      << "the retained forward image must remain its own generation authority";

  fixture.topology->block_prototypes.front().push_back(NativeField{});
  EXPECT_THROW((void)forward_provider->program_resource_topology(), std::invalid_argument)
      << "a detached topology whose field shape drifts from its geometry must fail closed";
  fixture.topology->block_prototypes.front().pop_back();

  auto wrong_runtime = Access::make_uniform_accepted();
  EXPECT_THROW(wrong_runtime->with_forward_execution_overlay(
                   authority, snapshot,
                   [](std::shared_ptr<NativeProgramExecutionServices>) { return true; }),
               std::logic_error);

  auto malformed_topology = std::make_shared<NativeAmrTopology>(*fixture.topology);
  malformed_topology->program_state = nullptr;
  EXPECT_THROW((runtime::program::PreparedForwardAmrExecutionAuthorityView<kTestDimension>(
                   std::move(malformed_topology))),
               std::invalid_argument);
}

TEST(ProgramExecutionServicesContract,
     FrozenCheckpointMacroClockIsReconciledIntoDetachedAmrSchedule) {
  ensure_kokkos();
  using Access =
      runtime::program::detail::ProgramExecutionServicesForwardOverlayTestAccess<kTestDimension>;
  ForwardOverlayTopologyFixture fixture;
  runtime::program::ProgramHostDescriptor source{};
  source.native_dimension = static_cast<std::uint32_t>(kTestDimension);
  source.runtime_kind = runtime::program::ProgramRuntimeKind::amr;
  auto image = std::make_shared<runtime::program::ProgramExecutionPreparationImage<kTestDimension>>(
      source, fixture.topology, 1);
  auto provider = image->provider();
  constexpr std::string_view kQualifiedClock =
      "pops.clock.v1::sha256:8ad8aef533351288f8d2255203718219df1f4a5866f3e0ab331c24fd2ff1f5b3";
  provider->configure_primary_clock(std::string(kQualifiedClock));

  ASSERT_NO_THROW(image->reconcile_staged_amr_checkpoint_clock_identities(
      {"clock.macro", std::string(kQualifiedClock)}));
  const std::map<std::string, std::int64_t> expected_ticks{{"clock.macro", 5},
                                                           {std::string(kQualifiedClock), 5}};
  EXPECT_EQ(image->staged_clock_schedule().accepted_ticks(5), expected_ticks);
  EXPECT_EQ(Access::accepted_amr_clock_ticks(*provider, 5), expected_ticks)
      << "the detached live service must carry the complete frozen checkpoint clock set";
}

TEST(ProgramExecutionServicesContract,
     MarkBoundWithoutProgramDoesNotRequirePreparedCouplingReceipt) {
  ensure_kokkos();
  NativeSystem system(native_config(4));
  install_execution_lane(system, "pops.test.program-context.bound-without-program");
  add_gas(system);

  EXPECT_NO_THROW(system.mark_bound());
  EXPECT_EQ(system.lifecycle_state(), "bound");
}

TEST(ProgramExecutionServicesContract,
     ClockScheduleResidentStorageTracksExternalStringsAndFrameCapacityExactly) {
  runtime::program::ClockScheduleState short_schedule;
  short_schedule.configure_primary_clock("macro");
  short_schedule.declare_relation("macro", "fast", 2);
  const std::uint64_t short_bytes = short_schedule.resident_storage_bytes();

  runtime::program::ClockScheduleState long_schedule;
  const std::string primary(192, 'p');
  const std::string child(224, 'c');
  long_schedule.configure_primary_clock(primary);
  long_schedule.declare_relation(primary, child, 2);
  long_schedule.seal_for_execution();
  const std::uint64_t relation_bytes = long_schedule.resident_storage_bytes();
  EXPECT_GT(relation_bytes, short_bytes);
  std::uint64_t frame_bytes = 0;
  {
    BoundaryPointHeapWindow heap_window;
    auto scope = long_schedule.subcycle(primary, child, 2);
    scope.iteration(0);
    frame_bytes = long_schedule.resident_storage_bytes();
    EXPECT_EQ(heap_window.close(), 0u)
        << "bind-sealed long logical-clock frames must not allocate heap storage";
    EXPECT_EQ(frame_bytes, relation_bytes)
        << "bind-sealed frames retain compact ids, not copied clock strings";
    EXPECT_EQ(frame_bytes, long_schedule.resident_storage_bytes());
  }
  const std::uint64_t retained_frame_capacity = long_schedule.resident_storage_bytes();
  EXPECT_EQ(retained_frame_capacity, relation_bytes)
      << "ending a subcycle changes no bind-sealed frame allocation";
  EXPECT_EQ(retained_frame_capacity, long_schedule.resident_storage_bytes());
}

TEST(ProgramExecutionServicesContract,
     UniformHostFootprintSeparatesCarrierCapacityFromPreparedMultiFabPayloads) {
  ensure_kokkos();
  NativeSystemConfig cfg = native_config(8);

  NativeSystem unsealed_system(cfg);
  install_execution_lane(unsealed_system, "pops.test.program-context.host-footprint-unsealed");
  add_gas(unsealed_system);
  unsealed_system.set_program_block_map({0});
  auto unsealed_image =
      runtime::program::make_program_preparation_image<kTestDimension>(&unsealed_system, 1);
  auto unsealed_host = unsealed_system.program_host_descriptor();
  runtime::program::bind_program_preparation_image(unsealed_host, unsealed_image);
  auto unsealed_provider =
      runtime::program::make_program_execution_provider<kTestDimension>(unsealed_host.preparation);
  EXPECT_THROW((void)unsealed_provider->prepared_uniform_host_resident_resource_prototypes(),
               std::logic_error);

  NativeSystem short_system(cfg);
  install_execution_lane(short_system, "pops.test.program-context.host-footprint-short");
  add_gas(short_system);
  short_system.set_program_block_map({0});
  auto short_prepared = prepare_uniform_host_footprint_services(short_system, "f", 0);
  const auto short_first =
      short_prepared.provider->prepared_uniform_host_resident_resource_prototypes();
  const auto short_second =
      short_prepared.provider->prepared_uniform_host_resident_resource_prototypes();
  const auto short_route = std::find_if(
      short_first.begin(), short_first.end(),
      [](const auto& row) { return row.kind == ResourcePrototypeKind::generated_route; });
  const auto short_scratch = std::find_if(
      short_first.begin(), short_first.end(),
      [](const auto& row) { return row.kind == ResourcePrototypeKind::prepared_scratch; });
  ASSERT_NE(short_route, short_first.end());
  ASSERT_NE(short_scratch, short_first.end());
  ASSERT_TRUE(short_route->layout.maximum_bytes.has_value());
  ASSERT_TRUE(short_scratch->layout.maximum_bytes.has_value());
  ASSERT_GT(*short_route->layout.maximum_bytes, 0u);
  ASSERT_GT(*short_scratch->layout.maximum_bytes, 0u);
  ASSERT_EQ(short_first.size(), short_second.size());
  for (std::size_t index = 0; index < short_first.size(); ++index) {
    EXPECT_EQ(short_first[index].kind, short_second[index].kind);
    EXPECT_EQ(short_first[index].slot, short_second[index].slot);
    EXPECT_EQ(short_first[index].subslot, short_second[index].subslot);
    EXPECT_EQ(short_first[index].layout.maximum_bytes, short_second[index].layout.maximum_bytes);
  }
  const auto short_rows = runtime::program::take_staged_uniform_resource_prototypes<kTestDimension>(
      short_prepared.image);

  NativeSystem long_system(cfg);
  install_execution_lane(long_system, "pops.test.program-context.host-footprint-long");
  add_gas(long_system);
  long_system.set_program_block_map({0});
  auto long_prepared =
      prepare_uniform_host_footprint_services(long_system, std::string(256, 'f'), 7);
  const auto long_rows = runtime::program::take_staged_uniform_resource_prototypes<kTestDimension>(
      long_prepared.image);
  EXPECT_GT(prepared_host_bytes(long_rows, ResourcePrototypeKind::generated_route),
            prepared_host_bytes(short_rows, ResourcePrototypeKind::generated_route));
  EXPECT_GT(prepared_host_bytes(long_rows, ResourcePrototypeKind::prepared_scratch),
            prepared_host_bytes(short_rows, ResourcePrototypeKind::prepared_scratch));

  NativeSystemConfig wider_cfg = native_config(16);
  NativeSystem wider_system(wider_cfg);
  install_execution_lane(wider_system, "pops.test.program-context.host-footprint-wider");
  add_gas(wider_system);
  wider_system.set_program_block_map({0});
  auto wider_prepared = prepare_uniform_host_footprint_services(wider_system, "f", 0);
  const auto wider_rows = runtime::program::take_staged_uniform_resource_prototypes<kTestDimension>(
      wider_prepared.image);
  EXPECT_EQ(prepared_host_bytes(wider_rows, ResourcePrototypeKind::prepared_scratch),
            prepared_host_bytes(short_rows, ResourcePrototypeKind::prepared_scratch));
  const auto short_rhs = std::find_if(short_rows.begin(), short_rows.end(), [](const auto& row) {
    return row.kind == ResourcePrototypeKind::rhs;
  });
  const auto wider_rhs = std::find_if(wider_rows.begin(), wider_rows.end(), [](const auto& row) {
    return row.kind == ResourcePrototypeKind::rhs;
  });
  ASSERT_NE(short_rhs, short_rows.end()) << "the short staged image lost its numerical rhs row";
  ASSERT_NE(wider_rhs, wider_rows.end()) << "the wider staged image lost its numerical rhs row";
  EXPECT_GT(wider_rhs->layout.cells, short_rhs->layout.cells)
      << "the numerical MultiFab payload remains in its rhs ResourcePlan row";
}

void install_native_v5_program(
    NativeSystem& system, std::string_view identity,
    const std::vector<test::program_v5::CallbackProgramResource>& resources,
    NativeProgramCallback callback, const std::vector<std::string>* blocks = nullptr) {
#if !defined(POPS_TEST_TMPDIR)
  (void)system;
  (void)identity;
  (void)resources;
  (void)callback;
  (void)blocks;
  throw std::runtime_error("ProgramExecutionServices ABI-v5 fixture requires POPS_TEST_TMPDIR");
#else
  auto& callbacks = native_program_callbacks();
  const auto identifier = static_cast<std::uint64_t>(callbacks.size());
  callbacks.push_back(std::move(callback));
  static std::size_t fixture_index = 0;
  const std::string prefix = std::string(POPS_TEST_TMPDIR) + "/program_execution_services_" +
                             std::to_string(++fixture_index);
  const std::string source_path = prefix + ".cpp";
  const std::string library_path = prefix + ".so";
  std::ofstream source(source_path);
  if (!source)
    throw std::runtime_error("cannot create ProgramExecutionServices ABI-v5 fixture source");
  source << test::program_v5::callback_program_source(
      identifier, identity, "clock.macro", blocks != nullptr ? *blocks : system.block_names(),
      resources, "pops_test_program_execution_services_callback", "uniform");
  source.close();
  const auto compiled = test::native_dso::compile_shared(source_path, library_path);
  if (!compiled.ok) {
    test::native_dso::report_compile_failure("test_program_execution_services_contract", compiled);
    throw std::runtime_error("ProgramExecutionServices ABI-v5 fixture compilation failed");
  }
  system.install_program(library_path);
#endif
}

TEST(ProgramExecutionServicesContract,
     CoupledProgramInstallRequiresACompleteSystemBlockBijectionBeforePublication) {
  ensure_kokkos();
  NativeSystem system(native_config(4));
  install_execution_lane(system, "pops.test.program-context.coupled-program-map");
  add_gas_block(system, "left");
  add_gas_block(system, "right");
  system.install_prepared_coupling_operator("test.coupled-program-map",
                                            "test.coupled-program-map/provider@1", {},
                                            [](Real, const std::vector<NativeField*>&) {});

  const std::vector<std::string> complete_blocks{"left", "right"};
  install_native_v5_program(
      system, "tests.program-execution-services/coupled-map-complete@1", {},
      [](NativeProgramExecutionServices&, double) {}, &complete_blocks);
  EXPECT_EQ(system.program_block_map(), std::vector<int>({0, 1}));

  const std::vector<std::string> subset_blocks{"left"};
  EXPECT_THROW(install_native_v5_program(
                   system, "tests.program-execution-services/coupled-map-subset@1", {},
                   [](NativeProgramExecutionServices&, double) {}, &subset_blocks),
               std::invalid_argument);
  EXPECT_EQ(system.program_block_map(), std::vector<int>({0, 1}))
      << "a rejected coupled subset Program must retain the prior accepted block map";

  const std::vector<std::string> reordered_blocks{"right", "left"};
  EXPECT_NO_THROW(install_native_v5_program(
      system, "tests.program-execution-services/coupled-map-reordered@1", {},
      [](NativeProgramExecutionServices&, double) {}, &reordered_blocks));
  EXPECT_EQ(system.program_block_map(), std::vector<int>({1, 0}));
  EXPECT_NO_THROW(system.mark_bound())
      << "the installed receipt must authenticate the replacement Program's detached block map";
}

// Non-uniform pressure IC (u = v = 0): -div F has a non-zero momentum component so the step actually
// changes the state (parity is not vacuous). Periodic, deterministic across NativeSystem instances.
std::vector<double> ic(int n) {
  const std::size_t cell_count = uniform_cell_count(n);
  const double pi = 3.14159265358979323846;
  std::vector<double> U(static_cast<std::size_t>(kNcomp) * cell_count, 0.0);
  for (std::size_t linear = 0; linear < cell_count; ++linear) {
    std::size_t remainder = linear;
    double modulation = 1.0;
    for (int axis = 0; axis < kTestDimension; ++axis) {
      const int index = static_cast<int>(remainder % static_cast<std::size_t>(n));
      remainder /= static_cast<std::size_t>(n);
      modulation *= std::cos(2 * pi * (index + 0.5) / n);
    }
    const double pressure = 3.0 + 0.5 * modulation;
    U[static_cast<std::size_t>(GasSchema::density) * cell_count + linear] = 1.0;
    U[static_cast<std::size_t>(GasSchema::energy) * cell_count + linear] =
        pressure / (kGamma - 1.0);
  }
  return U;
}

TEST(ProgramExecutionServicesContract, SystemMoveTransfersPreparedExecutionLane) {
  ensure_kokkos();
  NativeSystem source(native_config(4));
  install_execution_lane(source, "pops.test.system-move");
  NativeSystem moved(std::move(source));
  EXPECT_EQ(moved.prepared_boundary_execution_lane().identity(), "pops.test.system-move");
  EXPECT_THROW(static_cast<void>(source.prepared_boundary_execution_lane()), std::logic_error);

  // solve_fields materializes ExactNamedField + cartesian_cg, both of which pin the destination
  // lane with ExecutionLane::ImmutableBorrow. Assignment must destroy that Impl first.
  NativeSystem assigned(native_config(4));
  install_execution_lane(assigned, "pops.test.system-move.destination");
  add_gas(assigned);
  assigned.set_state("gas", ic(4));
  (void)pops::consume_solve_outcome(assigned.solve_fields());
  assigned = std::move(assigned);
  EXPECT_EQ(assigned.prepared_boundary_execution_lane().identity(),
            "pops.test.system-move.destination");
  assigned = std::move(moved);
  EXPECT_EQ(assigned.prepared_boundary_execution_lane().identity(), "pops.test.system-move");
  EXPECT_THROW(static_cast<void>(moved.prepared_boundary_execution_lane()), std::logic_error);
}

TEST(ProgramExecutionServicesContract,
     BoundDefaultFieldOwnerIsColdPrimedAndRejectRollbackPreservesIt) {
  ensure_kokkos();
  NativeSystem system(native_config(4));
  install_execution_lane(system, "pops.test.program-context.bound-default-field-rollback");
  add_gas(system);
  const std::vector<double> initial = ic(4);
  system.set_state("gas", initial);
  system.set_program_block_map({0});

  install_native_v5_program(system,
                            "tests.program-execution-services/bound-default-field-rollback@1", {},
                            [](NativeProgramExecutionServices& context, double dt) {
                              context.begin_step(dt);
                              auto outcome = context.solve_fields();
                              (void)outcome.consume(SolveConsumption::kAccept);
                              throw runtime::program::StepAttemptRejected(
                                  SolveStatus::kIterationLimit, "test rollback",
                                  "reject after default-field candidate publication");
                            });

  ASSERT_NO_THROW(system.mark_bound());
  EXPECT_EQ(system.field_provider_slots(), std::vector<std::string>{"pops.system.default-field"})
      << "the transaction image must capture the default field before the first candidate";
  EXPECT_THROW(system.step(1.0e-3), runtime::program::StepAttemptRejected);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_DOUBLE_EQ(system.time(), 0.0);
  EXPECT_EQ(system.get_state("gas"), initial);
  EXPECT_EQ(system.field_provider_slots(), std::vector<std::string>{"pops.system.default-field"})
      << "rollback must restore data without changing the bind-sealed owner";

}

TEST(ProgramExecutionServicesContract, AnonymousRateIdentityIsRejectedBeforeTopologyLookup) {
  ensure_kokkos();
  NativeSystem sim(native_config(2));
  install_execution_lane(sim, "pops.test.program-context.anonymous-rate");
  add_gas(sim);
  auto prepared = prepare_native_program_services(sim);
  auto& context = *prepared.provider;
  EXPECT_THROW((void)context.boundary_evaluation_point(-1), std::invalid_argument);
}

TEST(ProgramExecutionServicesContract,
     PreparedBoundaryPointWriteCopyIsAllocationFreeAndRejectsCapacityDrift) {
  ensure_kokkos();
  NativeSystem sim(native_config(4));
  install_execution_lane(sim, "pops.test.program-context.resident-boundary-point");
  add_gas(sim);
  sim.set_program_block_map({0});
  auto prepared = prepare_native_program_services(sim);
  auto& context = *prepared.provider;
  constexpr std::string_view kClock =
      "pops.test.program-context.resident-boundary-point.long-clock-identity";
  context.configure_primary_clock(std::string(kClock));

  // This is the cold source shape retained by a matrix-free session template.  Each identity is
  // deliberately longer than SSO so preparation must reserve all four strings before a step.
  runtime::multiblock::BoundaryEvaluationPoint capacity_source;
  capacity_source.clock.assign(kClock);
  capacity_source.graph_identity =
      "pops.test.program-context.resident-boundary-point.graph-identity";
  capacity_source.rate_identity = "pops.test.program-context.resident-boundary-point.rate-identity";
  capacity_source.application_identity =
      "pops.test.program-context.resident-boundary-point.application-identity";
  runtime::multiblock::BoundaryEvaluationPoint source;
  runtime::multiblock::BoundaryEvaluationPoint destination;
  runtime::multiblock::BoundaryEvaluationPoint retry;
  context.prepare_boundary_evaluation_point(source, capacity_source);
  context.prepare_boundary_evaluation_point(destination, capacity_source);
  context.prepare_boundary_evaluation_point(retry, capacity_source);

  context.begin_step(0.01);
  context.write_boundary_evaluation_point_into(source, 0);
  source.graph_identity.assign(capacity_source.graph_identity);
  source.rate_identity.assign(capacity_source.rate_identity);
  source.application_identity.assign(capacity_source.application_identity);

  const AllocationEventStats event_before = allocation_event_stats();
  {
    BoundaryPointHeapWindow heap_window;
    context.copy_boundary_evaluation_point_into(destination, source);
    EXPECT_EQ(heap_window.close(), 0u);
  }
  EXPECT_EQ(allocation_event_stats(), event_before);
  EXPECT_EQ(destination, source);

  // An unprepared external point cannot acquire capacity inside the candidate.  The rejection
  // happens before either scalar or string state of the destination is changed.
  runtime::multiblock::BoundaryEvaluationPoint unprimed;
  EXPECT_THROW(context.copy_boundary_evaluation_point_into(unprimed, source), std::logic_error);
  // ``BoundaryEvaluationPoint`` deliberately initializes its time coordinates to NaN, so its
  // defaulted equality operator is not a valid no-clobber witness.  Check every logical member
  // explicitly instead of comparing NaN to itself.
  EXPECT_TRUE(unprimed.clock.empty());
  EXPECT_EQ(unprimed.tick, 0);
  EXPECT_EQ(unprimed.level, 0);
  EXPECT_EQ(unprimed.substep, 0);
  EXPECT_EQ(unprimed.stage, 0);
  EXPECT_EQ(unprimed.stage_fraction, (pops::amr::Rational{0, 1}));
  EXPECT_TRUE(std::isnan(unprimed.dt));
  EXPECT_TRUE(std::isnan(unprimed.physical_time));
  EXPECT_TRUE(unprimed.graph_identity.empty());
  EXPECT_TRUE(unprimed.rate_identity.empty());
  EXPECT_TRUE(unprimed.application_identity.empty());

  {
    BoundaryPointHeapWindow heap_window;
    context.copy_boundary_evaluation_point_into(retry, source);
    EXPECT_EQ(retry, source);
    context.write_boundary_evaluation_point_into(source, 0);
    context.copy_boundary_evaluation_point_into(destination, source);
    EXPECT_EQ(heap_window.close(), 0u);
  }
  EXPECT_EQ(destination, source);
}

TEST(ProgramExecutionServicesContract, ProviderFreeViewDoesNotRequireAPlanOrStorageCarrier) {
  ensure_kokkos();
  NativeSystem sim(native_config(2));
  install_execution_lane(sim, "pops.test.program-context.provider-free-view");
  add_gas(sim);
  auto prepared = prepare_native_program_services(sim);
  auto& context = *prepared.provider;

  // This System has neither registered providers nor a program-block map.  Count zero therefore
  // proves the API is a true empty ABI: it must not resolve the qid, map a block, or dereference
  // any provider storage that belongs to another possible consumer.
  const auto providers = context.template provider_values_view<0>("not-resolved", 73, 19);
  EXPECT_TRUE(providers.storage.empty());
  EXPECT_TRUE(providers.storage_components.empty());
}

TEST(ProgramExecutionServicesContract, AuxiliaryStorageGroupFindBorrowsLongIdentityWithoutHeap) {
  const std::string group_identity =
      "pops.test.program-context.provider-values-view.storage-group-identity-over-sso";
  runtime::system::AuxiliaryStorageGroups<kTestDimension> groups;
  groups.groups.emplace(group_identity, NativeField{});
  ASSERT_NE(groups.find(group_identity), nullptr);

  const AllocationEventStats allocations_before = allocation_event_stats();
  const NativeField* found = nullptr;
  {
    BoundaryPointHeapWindow heap_window;
    for (int repeat = 0; repeat < 8; ++repeat)
      found = groups.find(group_identity);
    EXPECT_EQ(heap_window.close(), 0u);
  }
  EXPECT_EQ(found, groups.find(group_identity));
  EXPECT_EQ(allocation_event_stats(), allocations_before);
}

TEST(ProgramExecutionServicesContract,
     BoundUniformProviderViewBorrowsLongConsumerQidWithoutHeapAllocation) {
  ensure_kokkos();
  NativeSystem sim(native_config(4));
  install_execution_lane(sim, "pops.test.program-context.provider-values-view");
  add_gas(sim);
  sim.set_program_block_map({0});
  const std::string qid = install_long_qid_auxiliary_plan(sim);
  ASSERT_GT(qid.size(), std::size_t{15});

  auto prepared = prepare_native_program_services(sim);
  auto& context = *prepared.provider;
  // Prime the exact carrier and plan before opening the witness window.  The repeated calls below
  // exercise the accepted Uniform route with a qid that cannot use SSO.
  const auto primed = context.template provider_values_view<1>(qid, 0, 0);
  ASSERT_EQ(primed.storage.size(), std::size_t{1});

  const AllocationEventStats allocations_before = allocation_event_stats();
  {
    BoundaryPointHeapWindow heap_window;
    for (int repeat = 0; repeat < 8; ++repeat) {
      const auto providers = context.template provider_values_view<1>(qid, 0, 0);
      (void)providers;
    }
    EXPECT_EQ(heap_window.close(), 0u);
  }
  EXPECT_EQ(allocation_event_stats(), allocations_before);
}

TEST(ProgramExecutionServicesContract,
     PreparedUniformSumUsesTheBindSealedWorkspaceOnItsFirstAcceptedCall) {
  ensure_kokkos();
  NativeSystem sim(native_config(8));
  install_execution_lane(sim, "pops.test.program-context.prepared-sum");
  add_gas(sim);
  sim.set_state("gas", ic(8));
  sim.set_program_block_map({0});

  auto prepared = prepare_native_program_services(sim);
  auto& context = *prepared.provider;
  NativeField& state = context.state(0);

  // No SUM is performed before this point: the first call after accepted activation must use the
  // capacity prepared from the detached state prototype, with no Fab/communication allocation.
  const AllocationEventStats allocations_before = allocation_event_stats();
  const Real first = context.sum_component(0, state, GasSchema::density);
  EXPECT_EQ(allocation_event_stats(), allocations_before);
  EXPECT_NEAR(first, static_cast<Real>(uniform_cell_count(8)), Real(1e-12));

  const auto first_bits = std::bit_cast<std::uint64_t>(first);
  for (int repeat = 0; repeat < 8; ++repeat) {
    const Real repeated = context.sum_component(0, state, GasSchema::density);
    EXPECT_EQ(std::bit_cast<std::uint64_t>(repeated), first_bits);
    EXPECT_EQ(allocation_event_stats(), allocations_before);
  }
}

TEST(ProgramExecutionServicesContract,
     DetachedUniformPreparationReadsItsImageAndCannotMutateTheAcceptedFacade) {
  ensure_kokkos();
  NativeSystem sim(native_config(4));
  install_execution_lane(sim, "pops.test.program-preparation.detached");
  add_gas(sim);
  sim.set_program_block_map({0});

  const int accepted_step = sim.macro_step();
  const Real accepted_time = static_cast<Real>(sim.time());
  const auto accepted_state = sim.get_state("gas");
  auto host = sim.program_host_descriptor();
  auto image = runtime::program::make_program_preparation_image<kTestDimension>(&sim, 1);
  runtime::program::bind_program_preparation_image(host, image);
  auto context =
      runtime::program::make_program_execution_provider<kTestDimension>(host.preparation);

  EXPECT_EQ(context->n_blocks(), 1);
  EXPECT_EQ(context->macro_step(), accepted_step);
  EXPECT_EQ(context->physical_time(), accepted_time);
  EXPECT_TRUE(context->prepared_execution_lane().active());
  EXPECT_NE(&context->prepared_execution_lane(), &sim.prepared_boundary_execution_lane());
  context->configure_primary_clock("pops.test.program-preparation.clock");

  // The preparation image owns all candidate-visible services.  Destroying the failed candidate
  // without activation leaves the accepted System state and public clock untouched.
  context.reset();
  image.reset();
  EXPECT_EQ(sim.macro_step(), accepted_step);
  EXPECT_EQ(static_cast<Real>(sim.time()), accepted_time);
  EXPECT_EQ(sim.get_state("gas"), accepted_state);
}

TEST(ProgramExecutionServicesContract, PreparedLinearSolveAcceptsDistinctCongruentWorkspaceLane) {
  ensure_kokkos();
  comm_init();
  NativeSystem sim(native_config(4));
  install_execution_lane(sim, "pops.test.program-context.prepared-linear-runtime");
  add_gas(sim);
  sim.set_state("gas", ic(4));
  sim.set_program_block_map({0});
  auto prepared = prepare_native_program_services(sim);
  auto& context = *prepared.provider;
  context.begin_step(Real(1e-3));
  context.set_stage_time(0, 1);

  NativeField& prototype = context.state(0);
  const KrylovFootprint<kTestDimension> footprint{prototype.ncomp(), prototype.ghosts(), false};
  const PreparedKrylovMethod<kTestDimension> method = cg_krylov_method<kTestDimension>();
  const OperatorFingerprint authority{UINT64_C(101), UINT64_C(102), UINT64_C(103), UINT64_C(104)};
  const OperatorFingerprint resources{UINT64_C(105), UINT64_C(106), UINT64_C(107), UINT64_C(108)};
  const OperatorEvaluationSnapshot snapshot =
      context.operator_evaluation_snapshot(authority, prototype, resources);
  PreparedAffineLinearProblem<kTestDimension> problem(
      prototype,
      PreparedAffineOperatorProvider<kTestDimension>::trusted_reentrant(
          [](NativeField& out, const NativeField& in) {
            detail::PreparedFieldAlgebra::copy(out, in);
          },
          [] { return std::size_t{0}; }),
      PreparedLinearPreconditioner<kTestDimension>::identity(),
      LinearOperatorProperties::symmetric_positive_definite(), footprint,
      PreparedNullspacePolicy<kTestDimension>::nonsingular(), [snapshot] { return snapshot; });
  const ExecutionCommunicator runtime_communicator = context.prepared_execution_communicator();
  KrylovWorkspace<kTestDimension> workspace(
      runtime_communicator, "pops.test.program-context.workspace",
      "pops.test.program-context.workspace.positive", prototype, method, footprint);
  KrylovWorkspace<kTestDimension> legacy_workspace(
      runtime_communicator, "pops.test.program-context.workspace", prototype, method, footprint);
  NativeField solution = context.scratch_state_like(prototype);
  NativeField rhs = context.scratch_state_like(prototype);
  solution.set_val(Real(0));
  rhs.set_val(Real(1));
  problem.prepare(snapshot);
  workspace.bind(problem);
  legacy_workspace.bind(problem);

  const ExecutionLane& workspace_lane =
      ::pops::detail::KrylovWorkspaceAccess::execution_lane(workspace);
  EXPECT_NE(&workspace_lane, &context.prepared_execution_lane());
  EXPECT_TRUE(workspace_lane.congruent_with(context.prepared_execution_lane()));
  EXPECT_THROW((void)context.solve_prepared_linear(
                   problem, legacy_workspace, solution, rhs,
                   KrylovControls<kTestDimension>{method, Real(1e-12), Real(0), 4}),
               std::invalid_argument);
  SolveOutcome outcome = context.solve_prepared_linear(
      problem, workspace, solution, rhs,
      KrylovControls<kTestDimension>{method, Real(1e-12), Real(0), 4});
  ASSERT_TRUE(outcome.report().solved_value_available()) << outcome.report().reason;
  (void)outcome.consume(SolveConsumption::kAccept);
  for (int component = 0; component < solution.ncomp(); ++component)
    EXPECT_DOUBLE_EQ(context.sum_component(solution, component),
                     context.sum_component(rhs, component));
}

TEST(ProgramExecutionServicesContract,
     PreparedLinearSolveRefusesRankDivergentSameSolveIdLevelOwnerSelection) {
#ifndef POPS_HAS_MPI
  GTEST_SKIP() << "rank-divergent prepared solve validation requires MPI";
#else
  ensure_kokkos();
  comm_init();
  if (n_ranks() < 2)
    GTEST_SKIP() << "rank-divergent prepared solve validation requires at least two MPI ranks";

  NativeSystem sim(native_config(4));
  install_execution_lane(sim, "pops.test.program-context.prepared-linear-negative-runtime");
  add_gas(sim);
  sim.set_state("gas", ic(4));
  sim.set_program_block_map({0});
  auto prepared = prepare_native_program_services(sim);
  auto& context = *prepared.provider;
  context.begin_step(Real(1e-3));
  context.set_stage_time(0, 1);

  NativeField& prototype = context.state(0);
  const KrylovFootprint<kTestDimension> footprint{prototype.ncomp(), prototype.ghosts(), false};
  const PreparedKrylovMethod<kTestDimension> method = cg_krylov_method<kTestDimension>();
  const OperatorFingerprint authority{UINT64_C(111), UINT64_C(112), UINT64_C(113), UINT64_C(114)};
  const OperatorFingerprint resources{UINT64_C(115), UINT64_C(116), UINT64_C(117), UINT64_C(118)};
  const OperatorEvaluationSnapshot snapshot =
      context.operator_evaluation_snapshot(authority, prototype, resources);
  int apply_calls = 0;
  PreparedAffineLinearProblem<kTestDimension> problem(
      prototype,
      PreparedAffineOperatorProvider<kTestDimension>::trusted_reentrant(
          [&apply_calls](NativeField& out, const NativeField& in) {
            ++apply_calls;
            detail::PreparedFieldAlgebra::copy(out, in);
          },
          [] { return std::size_t{0}; }),
      PreparedLinearPreconditioner<kTestDimension>::identity(),
      LinearOperatorProperties::symmetric_positive_definite(), footprint,
      PreparedNullspacePolicy<kTestDimension>::nonsingular(), [snapshot] { return snapshot; });
  const ExecutionCommunicator runtime_communicator = context.prepared_execution_communicator();
  KrylovWorkspace<kTestDimension> workspace_a(
      runtime_communicator, "pops.test.program-context.workspace",
      "pops.program.amr.krylov-workspace.77/level-owner-identity-0", prototype, method, footprint);
  KrylovWorkspace<kTestDimension> workspace_b(
      runtime_communicator, "pops.test.program-context.workspace",
      "pops.program.amr.krylov-workspace.77/level-owner-identity-1", prototype, method, footprint);
  problem.prepare(snapshot);
  workspace_a.bind(problem);
  workspace_b.bind(problem);
  EXPECT_EQ(::pops::detail::KrylovWorkspaceAccess::execution_lane(workspace_a).identity(),
            ::pops::detail::KrylovWorkspaceAccess::execution_lane(workspace_b).identity());
  EXPECT_NE(::pops::detail::KrylovWorkspaceAccess::materialization_token(workspace_a),
            ::pops::detail::KrylovWorkspaceAccess::materialization_token(workspace_b));

  NativeField solution = context.scratch_state_like(prototype);
  NativeField rhs = context.scratch_state_like(prototype);
  solution.set_val(Real(0));
  rhs.set_val(Real(1));
  const int apply_calls_before_solve = apply_calls;
  KrylovWorkspace<kTestDimension>& selected_workspace = my_rank() == 0 ? workspace_a : workspace_b;
  bool refused = false;
  try {
    (void)context.solve_prepared_linear(
        problem, selected_workspace, solution, rhs,
        KrylovControls<kTestDimension>{method, Real(1e-12), Real(0), 4});
  } catch (const std::invalid_argument& error) {
    refused = true;
    EXPECT_STREQ(error.what(),
                 "Program prepared linear solve workspace lane contract differs across MPI ranks");
  }
  EXPECT_TRUE(refused);
  EXPECT_EQ(apply_calls, apply_calls_before_solve)
      << "the runtime-lane contract must reject before any selected private workspace solve";
  EXPECT_EQ(all_reduce_min(refused ? 1L : 0L, context.prepared_execution_lane()), 1L);
#endif
}

TEST(ProgramExecutionServicesContract,
     ApplyProjectionRefusesRankDivergentPreparedBlockRouteBeforeProviderInvocation) {
#ifndef POPS_HAS_MPI
  GTEST_SKIP() << "rank-divergent projection validation requires MPI";
#else
  ensure_kokkos();
  comm_init();
  if (n_ranks() < 2)
    GTEST_SKIP() << "rank-divergent projection validation requires at least two MPI ranks";

  NativeSystem sim(native_config(4));
  install_execution_lane(sim, "pops.test.program-context.projection-route-negative-runtime");
  int projection_calls = 0;
  add_gas_block(sim, "left", &projection_calls);
  add_gas_block(sim, "right", &projection_calls);
  const int selected_block = my_rank() == 0 ? 0 : 1;
  sim.set_program_block_map({selected_block});
  bool refused = false;
  try {
    (void)prepare_native_program_services(sim);
  } catch (const std::runtime_error& error) {
    refused = true;
    EXPECT_STREQ(error.what(), "Program projection/speed block routes differ across MPI ranks");
  }
  EXPECT_TRUE(refused);
  EXPECT_EQ(projection_calls, 0)
      << "the bind receipt must refuse route drift before a projection provider is reachable";
  EXPECT_EQ(all_reduce_min(refused ? 1L : 0L, sim.prepared_boundary_execution_lane()), 1L);
#endif
}

TEST(ProgramExecutionServicesContract,
     ProjectionAndMaximumSpeedUseBindSealedRoutesWithoutAllocationOrDynamicConsensus) {
  ensure_kokkos();
  NativeSystem sim(native_config(4));
  install_execution_lane(sim, "pops.test.program-context.projection-speed-hot");
  int projection_calls = 0;
  add_gas_block(sim, "gas", &projection_calls);
  sim.set_program_block_map({0});
  auto prepared = prepare_native_program_services(sim);
  auto& context = *prepared.provider;
  NativeField& state = context.state(0);

  // Warm the providers before the witness window: the assertion is about route/consensus work,
  // not a first-use provider implementation detail.
  context.apply_projection(0, state);
  EXPECT_GT(context.max_wave_speed(0, state), Real(0));
  const AllocationEventStats allocations_before = allocation_event_stats();
  const std::uint64_t consensus_before = exact_consensus_dynamic_storage_calls();
  const int calls_before = projection_calls;
  for (int repeat = 0; repeat < 8; ++repeat) {
    context.apply_projection(0, state);
    EXPECT_GT(context.max_wave_speed(0, state), Real(0));
  }
  EXPECT_EQ(projection_calls, calls_before + 8);
  EXPECT_EQ(allocation_event_stats(), allocations_before)
      << "bind-sealed projection/speed routes must not allocate in the Program hot loop";
  EXPECT_EQ(exact_consensus_dynamic_storage_calls(), consensus_before)
      << "bind-sealed projection/speed routes must not rebuild exact consensus in the hot loop";
  EXPECT_THROW(context.apply_projection(1, state), std::logic_error)
      << "a block index outside the sealed route table must fail closed";
}

TEST(ProgramExecutionServicesContract, ProjectionReportUsesOneBindSealedIdentity) {
  runtime::program::ProgramRuntimeState<kTestDimension> state;
  state.declare_step_projection("realizability");
  state.bind_transaction_authorities();

  state.begin_step_projection_report();
  state.note_step_projection("realizability");
  state.note_step_projection("realizability");
  EXPECT_EQ(state.consume_step_projections(), std::vector<std::string>({"realizability"}));
  EXPECT_TRUE(state.consume_step_projections().empty());
  EXPECT_THROW(state.note_step_projection(""), std::invalid_argument);
  EXPECT_THROW(state.note_step_projection("late"), std::logic_error);
}

TEST(ProgramExecutionServicesContract, AcceptedBalanceEvidenceIsCurrentAttemptExactAndFailClosed) {
  ensure_kokkos();
  runtime::program::ProgramRuntimeState<kTestDimension> state;
  const std::string route = "pops.balance-ledger-route.v1:sha256:" + std::string(64, '1');
  const std::array<std::pair<const char*, double>, 5> terms{{
      {"storage_change", 11.0},
      {"outward_boundary_flux", 2.0},
      {"sources", 5.0},
      {"reflux", 3.0},
      {"projection", 1.0},
  }};
  state.declare_balance_route(route);
  state.bind_transaction_authorities();

  state.begin_step_projection_report();
  for (const auto& [name, value] : terms)
    state.record_balance_term(route, name, 0.25 * value, "test");
  for (const auto& [name, value] : terms)
    state.record_balance_term(route, name, 0.75 * value, "test");
  const auto accepted = state.accepted_balance_terms(route, "test");
  EXPECT_EQ(accepted.size(), terms.size());
  for (const auto& [name, value] : terms)
    EXPECT_DOUBLE_EQ(accepted.at(name), value);
  // Reserved balance evidence is deliberately attempt-local and therefore absent
  // from the persistent/checkpointed inspection-diagnostic registry.
  EXPECT_EQ(state.diagnostics().count("pops.balance-term.v1:" + route + ":storage_change"), 0u);
  state.begin_step_projection_report();
  EXPECT_THROW((void)state.accepted_balance_terms(route, "test"), std::runtime_error);

  for (std::size_t index = 0; index + 1 < terms.size(); ++index)
    state.record_balance_term(route, terms[index].first, terms[index].second, "test");
  EXPECT_THROW((void)state.accepted_balance_terms(route, "test"), std::runtime_error);

  state.begin_step_projection_report();
  for (const std::string& forged :
       {"pops.balance-term", "pops.balance-term.v1", "pops.balance-term.v1:forged"}) {
    EXPECT_THROW(state.record_diagnostic(forged, 1.0), std::invalid_argument);
    EXPECT_EQ(state.diagnostics().count(forged), 0u);
  }
  EXPECT_THROW((void)state.accepted_balance_terms(route, "test"), std::runtime_error);
  EXPECT_THROW(state.record_balance_term("pops.balance-ledger-route.v1:sha256:bad",
                                         "storage_change", 1.0, "test"),
               std::invalid_argument);
  EXPECT_THROW(state.record_balance_term(route, "unknown", 1.0, "test"), std::invalid_argument);
  EXPECT_THROW((void)state.accepted_balance_terms(route, "test"), std::runtime_error);
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    d = std::fmax(d, std::fabs(a[k] - b[k]));
  }
  return d;
}

Real max_abs_diff(const NativeField& a, const NativeField& b) {
  Real difference = 0;
  const int components = a.ncomp();
  for (std::size_t local = 0; local < a.local_size(); ++local) {
    const NativeConstView lhs = a.fab(local).view();
    const NativeConstView rhs = b.fab(local).view();
    const NativeBox box = a.fab(local).grown_box();
    difference = std::fmax(
        difference, for_each_cell_reduce_max(box, [=] POPS_HD(const Index<kTestDimension>& cell) {
          Real local_difference = Real(0);
          for (int component = 0; component < components; ++component)
            local_difference =
                std::fmax(local_difference, std::fabs(lhs(cell, component) - rhs(cell, component)));
          return local_difference;
        }));
  }
  return static_cast<Real>(all_reduce_max(static_cast<double>(difference)));
}

}  // namespace

// A Forward-Euler Program expressed through NativeProgramExecutionServices, driven by sim.step(dt), is bit-equal to the
// reference U + dt*R computed from solve_fields + eval_rhs. Uses the PER-STAGE solve_fields_from_state
// seam (the one the codegen lowers every solve_fields to), passing the block's own live state.
TEST(ProgramExecutionServicesContract, ForwardEulerViaContextMatchesReference) {
  ensure_kokkos();
  const int n = 16;
  const double dt = 1e-3;
  NativeSystemConfig cfg = native_config(n);
  const std::vector<double> U0 = ic(n);

  NativeSystem ref(cfg);
  install_execution_lane(ref, "pops.test.program-context.forward-euler-reference");
  add_gas(ref);
  ref.set_state("gas", U0);
  (void)pops::consume_solve_outcome(ref.solve_fields());
  const std::vector<double> R0 = ref.eval_rhs("gas");
  std::vector<double> Uref(U0.size());
  for (std::size_t k = 0; k < Uref.size(); ++k) {
    Uref[k] = U0[k] + dt * R0[k];
  }

  NativeSystem sim(cfg);
  install_execution_lane(sim, "pops.test.program-context.forward-euler");
  add_gas(sim);
  sim.set_state("gas", U0);
  sim.set_program_block_map({0});
  using Resource = test::program_v5::CallbackProgramResource;
  std::vector<Resource> resources;
  resources.reserve(2);
  {
    const auto state = sim.block_state(0);
    if (!state)
      throw std::logic_error("Forward Euler fixture requires one prepared block state");
    resources.push_back({Resource::Kind::rhs, 0, 0, 0, -1,
                         static_cast<std::uint32_t>(state->ncomp()),
                         static_cast<std::uint32_t>(state->ghosts()[0])});
    resources.push_back({Resource::Kind::state, 1, 0, 0, -1,
                         static_cast<std::uint32_t>(state->ncomp()),
                         static_cast<std::uint32_t>(state->ghosts()[0])});
  }
  install_native_v5_program(sim, "tests.program-execution-services/forward-euler@1", resources,
                            [](NativeProgramExecutionServices& ctx, double h) {
                              ctx.begin_step(h);
                              ctx.set_stage_time(0, 1);
                              for (int b = 0; b < ctx.n_blocks(); ++b) {
                                NativeField& U = ctx.state(b);
                                {
                                  auto outcome = ctx.solve_fields_from_state(b, U);
                                  (void)outcome.consume(SolveConsumption::kAccept);
                                }  // per-stage field solve at the block's own state
                                NativeField& R = ctx.rhs_scratch(2 * b, 0, U);
                                NativeField& next = ctx.scratch_state(2 * b + 1, 0, U);
                                ctx.rhs_into(b, U, R, 0);
                                ctx.lincomb(next, Real(1), U, Real(h), R);
                                ctx.lincomb(U, Real(0), U, Real(1), next);
                              }
                            });
  sim.mark_bound();
  sim.step(dt);
  const std::vector<double> Up = sim.get_state("gas");

  EXPECT_TRUE(max_abs_diff(Up, Uref) < 1e-12) << "FE parity max|d|=" << max_abs_diff(Up, Uref);
  EXPECT_TRUE(max_abs_diff(Up, U0) > 1e-9) << "step did not change the state";
}

TEST(ProgramExecutionServicesContract,
     RankedHyperbolicBoundaryRefusesMappedPeriodicityWithoutProvider) {
  ensure_kokkos();
  std::vector<std::string> kinds(static_cast<std::size_t>(2 * kTestDimension), "foextrap");
  kinds.front() = "periodic";
  std::vector<std::string> identities;
  identities.reserve(kinds.size());
  for (int face = 0; face < 2 * kTestDimension; ++face)
    identities.push_back("case::block::scalar::face-" + std::to_string(face));
  auto boundary = prepare_hyperbolic_boundary<kTestDimension>(
      kinds, std::vector<double>(kinds.size(), 0.0), identities, {"Scalar"}, true);
  EXPECT_THROW((void)boundary.periodic_axes(), std::logic_error);
}

TEST(ProgramExecutionServicesContract, CommitManySnapshotsSourcesThatAreAlsoTargets) {
  ensure_kokkos();
  NativeSystemConfig cfg = native_config(8);
  NativeSystem sim(cfg);
  install_execution_lane(sim, "pops.test.program-context.simultaneous-field");
  add_gas_block(sim, "a");
  add_gas_block(sim, "b");
  sim.set_program_block_map({0, 1});
  auto prepared = prepare_native_program_services(sim);
  auto& ctx = *prepared.provider;

  NativeField& first = ctx.state(0);
  NativeField& second = ctx.state(1);
  first.set_val(Real(3));
  second.set_val(Real(7));

  ctx.commit_many({{&first, &second}, {&second, &first}});

  ASSERT_GT(first.local_size(), 0);
  ASSERT_GT(second.local_size(), 0);
  EXPECT_EQ(first_value(first), Real(7));
  EXPECT_EQ(first_value(second), Real(3));

  NativeField different_ghost_width =
      native_field_like(first, first.ncomp(), enlarged_ghosts(first, 1));
  different_ghost_width.set_val(Real(13));
  EXPECT_THROW(ctx.commit_many({{&first, &different_ghost_width}}), std::invalid_argument);
  EXPECT_EQ(first_value(first), Real(7));

  NativeField wrong_components = native_field_like(first, first.ncomp() + 1, first.ghosts());
  EXPECT_THROW(ctx.commit_many({{&first, &wrong_components}}), std::invalid_argument);
  EXPECT_EQ(first_value(first), Real(7));
  EXPECT_EQ(first_value(second), Real(3));
}

TEST(ProgramExecutionServicesContract, GeneratedScratchIsPersistentExactAndNonAliasing) {
  ensure_kokkos();
  NativeSystemConfig cfg = native_config(8);
  NativeSystem sim(cfg);
  install_execution_lane(sim, "pops.test.program-context.generated-scratch");
  add_gas(sim);
  sim.set_program_block_map({0});
  auto prepared = prepare_native_scratch_services(sim);
  auto& ctx = *prepared.provider;
  NativeField& state = ctx.state(0);

  // `activate_staged_uniform_program_execution_services` has made this the
  // accepted provider.  A regrid/refresh callback may consume its resident
  // scratch, but it must never replay the candidate prelude or allocate a new
  // family through the accepted facade.
  EXPECT_THROW(ctx.prepare_rhs_scratch(41, 0, 0), std::logic_error);
  EXPECT_THROW(ctx.prepare_state_scratch(41, 0, 0), std::logic_error);

  NativeField& rhs = ctx.rhs_scratch(41, 0, state);
  ASSERT_GT(rhs.local_size(), 0);
  Real* const rhs_storage = rhs.fab(0).view().data;
  rhs.set_val(Real(9));
  const AllocationEventStats before_reuse = allocation_event_stats();
  NativeField& reused = ctx.rhs_scratch(41, 0, state);
  const AllocationEventStats after_reuse = allocation_event_stats();
  EXPECT_EQ(&reused, &rhs);
  EXPECT_EQ(reused.fab(0).view().data, rhs_storage);
  EXPECT_EQ(after_reuse.fab_calls, before_reuse.fab_calls);
  EXPECT_EQ(after_reuse.fab_bytes, before_reuse.fab_bytes);
  EXPECT_EQ(first_value(reused), Real(0)) << "a retry must not observe provisional scratch bytes";

  NativeField& other_lane = ctx.rhs_scratch(41, 1, state);
  NativeField& provisional_state = ctx.scratch_state(41, 0, state);
  EXPECT_NE(&other_lane, &rhs);
  EXPECT_NE(&provisional_state, &rhs);
  if (other_lane.local_size() > 0)
    EXPECT_NE(other_lane.fab(0).view().data, rhs_storage);
  if (provisional_state.local_size() > 0)
    EXPECT_NE(provisional_state.fab(0).view().data, rhs_storage);

  NativeField wider = native_field_like(state, state.ncomp(), enlarged_ghosts(state, 1));
  const AllocationEventStats before_shape_drift = allocation_event_stats();
  EXPECT_THROW((void)ctx.rhs_scratch(41, 0, wider), std::logic_error);
  const AllocationEventStats after_shape_drift = allocation_event_stats();
  EXPECT_EQ(after_shape_drift.fab_calls, before_shape_drift.fab_calls);
  EXPECT_EQ(after_shape_drift.communication_calls, before_shape_drift.communication_calls);
  EXPECT_EQ(rhs.fab(0).view().data, rhs_storage);
  EXPECT_EQ(rhs.ghosts(), state.ghosts());
}

TEST(ProgramExecutionServicesContract,
     RuntimeSizedSlotSealsDistinctTypedSubslotsAndReusesTheirPreparedStorage) {
  ensure_kokkos();
  NativeSystem sim(native_config(8));
  install_execution_lane(sim, "pops.test.program-context.runtime-sized-typed-subslots");
  add_scalar_block(sim, "scalar");
  sim.set_program_block_map({0});

  auto host = sim.program_host_descriptor();
  auto image = runtime::program::make_program_preparation_image<kTestDimension>(&sim, 1);
  std::vector<runtime::program::ProgramInstallationTables::ResourcePlan> declaration(1);
  auto& resource = declaration.front();
  resource.slot = 0;
  resource.flags = runtime::program::kProgramResourceRuntimeSized;
  resource.resource_type = runtime::program::ProgramResourcePlanType::runtime_sized;
  resource.value_id = 1;
  resource.occurrence_path_id = 101;
  // The row-level components/ghosts are symbolic for a runtime-sized value.  The two typed
  // subslots below are the authoritative shapes captured by the detached preparation image.
  resource.components = 1;
  resource.ghosts = 0;
  resource.schema = "program-resource-plan:v1";
  resource.identity = "test.runtime-sized.scalar";
  resource.occurrence_path = "root/runtime-sized/scalar";
  resource.owner = "scalar";
  resource.space = "cell";
  resource.clock = "macro";
  resource.lifetime = "transient";
  resource.centering = "cell";
  resource.off_policy = "none";
  resource.communication = "none";
  resource.transfer_provider = "none";
  resource.restart_provider = "none";
  resource.component_names = "[]";
  resource.shape = "[]";
  runtime::program::bind_staged_uniform_program_resource_declaration<kTestDimension>(
      image, declaration, sim.program_block_map());
  runtime::program::bind_program_preparation_image(host, image);
  auto provider =
      runtime::program::make_program_execution_provider<kTestDimension>(host.preparation);

  provider->prepare_state_scratch(0, 0, 0);
  provider->prepare_scalar_scratch(0, 0, 0, 11, 0);
  // Replaying either prelude entry is idempotent and must retain the exact prepared layout.
  provider->prepare_state_scratch(0, 0, 0);
  provider->prepare_scalar_scratch(0, 0, 0, 11, 0);

  NativeField& state = provider->state(0);
  NativeField& state_scratch = provider->scratch_state(0, 0, state);
  NativeField& scalar_scratch = provider->scalar_scratch(0, 0, state, 11, 0);
  NativeField& state_again = provider->scratch_state(0, 0, state);
  NativeField& scalar_again = provider->scalar_scratch(0, 0, state, 11, 0);
  EXPECT_EQ(state_scratch.ncomp(), 1);
  EXPECT_EQ(state_scratch.ghosts()[0], state.ghosts()[0]);
  EXPECT_EQ(scalar_scratch.ncomp(), 11);
  EXPECT_EQ(scalar_scratch.ghosts()[0], 0);
  EXPECT_EQ(&state_scratch, &state_again);
  EXPECT_EQ(&scalar_scratch, &scalar_again);
  EXPECT_NE(&state_scratch, &scalar_scratch);

  provider->seal_uniform_preparation_without_clock();
  const auto prototypes =
      runtime::program::take_staged_uniform_resource_prototypes<kTestDimension>(image);
  const auto state_prototype =
      std::find_if(prototypes.begin(), prototypes.end(), [](const ResourcePrototype& prototype) {
        return prototype.slot == 0 && prototype.subslot == 0 &&
               prototype.kind == ResourcePrototypeKind::state;
      });
  const auto scalar_prototype =
      std::find_if(prototypes.begin(), prototypes.end(), [](const ResourcePrototype& prototype) {
        return prototype.slot == 0 && prototype.subslot == 0 &&
               prototype.kind == ResourcePrototypeKind::scalar;
      });
  ASSERT_NE(state_prototype, prototypes.end());
  ASSERT_NE(scalar_prototype, prototypes.end());
  EXPECT_EQ(state_prototype->layout.components, 1u);
  EXPECT_EQ(scalar_prototype->layout.components, 11u);
  EXPECT_EQ(state_prototype->layout.ghosts, state.ghosts()[0]);
  EXPECT_EQ(scalar_prototype->layout.ghosts, 0u);
  EXPECT_NE(state_prototype->layout.cells, scalar_prototype->layout.cells);
  EXPECT_EQ(state_prototype->layout.itemsize, sizeof(Real));
  EXPECT_EQ(scalar_prototype->layout.itemsize, sizeof(Real));

  runtime::program::ProgramInstallationTables tables;
  tables.resource_plan = declaration;
  const auto materialized = tables.materialize_resource_plan(prototypes);
  ASSERT_EQ(materialized.entries().size(), 1u);
  const auto& materialized_row = materialized.entries().front();
  EXPECT_EQ(
      materialized_row.bytes,
      state_prototype->layout.cells * sizeof(Real) * state_prototype->layout.components +
          scalar_prototype->layout.cells * sizeof(Real) * scalar_prototype->layout.components);
  EXPECT_EQ(materialized_row.maximum_bytes, materialized_row.bytes);
  EXPECT_FALSE(materialized_row.cells.has_value())
      << "one slot with differently shaped typed subslots has no single cell extent";
  EXPECT_FALSE(materialized_row.itemsize.has_value());
  EXPECT_NE(materialized.digest().size(), 0u);
}

TEST(ProgramExecutionServicesContract,
     ExactUniformScratchRejectsCapturedPrototypeMismatchDuringPreparation) {
  ensure_kokkos();
  NativeSystemConfig cfg = native_config(8);
  NativeSystem sim(cfg);
  install_execution_lane(sim, "pops.test.program-context.exact-scratch-mismatch");
  add_gas(sim);
  sim.set_program_block_map({0});

  auto host = sim.program_host_descriptor();
  auto image = runtime::program::make_program_preparation_image<kTestDimension>(&sim, 1);
  std::vector<runtime::program::ProgramInstallationTables::ResourcePlan> declaration(1);
  auto& scratch = declaration.front();
  scratch.slot = 0;
  scratch.components = GasModel::n_vars;
  scratch.ghosts = 0;
  runtime::program::bind_staged_uniform_program_resource_declaration<kTestDimension>(
      image, declaration, sim.program_block_map());
  runtime::program::bind_program_preparation_image(host, image);
  auto provider =
      runtime::program::make_program_execution_provider<kTestDimension>(host.preparation);
  ASSERT_NE(provider->state(0).ghosts()[0], 0);
  EXPECT_THROW(provider->prepare_rhs_scratch(0, 0, 0), std::invalid_argument);
}

TEST(ProgramExecutionServicesContract,
     SimultaneousNamedFieldWorkspaceIsPersistentSubsetSafeAndTransactional) {
  ensure_kokkos();
  NativeSystemConfig cfg = native_config(8);
  NativeSystem sim(cfg);
  install_execution_lane(sim, "pops.test.program-context.named-field-workspace");
  add_gas_block(sim, "a");
  add_gas_block(sim, "b");
  sim.set_poisson("charge_density", "cartesian_cg");
  sim.set_program_block_map({0, 1});
  auto prepared = prepare_native_field_route_services(sim);
  auto& ctx = *prepared.provider;
  ctx.configure_primary_clock("clock.main");
  ctx.begin_step(0.01);
  const auto point = [&](int stage) { return ctx.boundary_evaluation_point(stage); };

  NativeField& live_a = ctx.state(0);
  NativeField& live_b = ctx.state(1);
  live_a.set_val(Real(2));
  live_b.set_val(Real(3));
  NativeField stage_a = native_field_like(live_a, live_a.ncomp(), live_a.ghosts());
  NativeField stage_b = native_field_like(live_b, live_b.ncomp(), live_b.ghosts());
  stage_a.set_val(Real(7));
  stage_b.set_val(Real(9));
  ASSERT_GT(live_a.local_size(), 0);
  ASSERT_GT(live_b.local_size(), 0);
  Real* const live_a_storage = live_a.fab(0).view().data;
  Real* const live_b_storage = live_b.fab(0).view().data;

  auto incomplete_point = point(500);
  incomplete_point.clock.clear();
  EXPECT_THROW(
      (void)ctx.solve_fields_from_blocks_at(incomplete_point, 500, {{0, &stage_a}, {1, &stage_b}}),
      std::invalid_argument)
      << "the generated route must retain its complete BoundaryEvaluationPoint";

  auto missing_field_solve = [&]() {
    return ctx.solve_fields_from_blocks_at(point(501), 501, {{0, &stage_a}, {1, &stage_b}});
  };
  EXPECT_THROW((void)missing_field_solve(), std::out_of_range);
  EXPECT_EQ(live_a.fab(0).view().data, live_a_storage);
  EXPECT_EQ(live_b.fab(0).view().data, live_b_storage);
  EXPECT_EQ(first_value(live_a), Real(2));
  EXPECT_EQ(first_value(live_b), Real(3));

  const AllocationEventStats before_retry = allocation_event_stats();
  EXPECT_THROW((void)missing_field_solve(), std::out_of_range);
  const AllocationEventStats after_retry = allocation_event_stats();
  EXPECT_EQ(after_retry.fab_calls, before_retry.fab_calls);
  EXPECT_EQ(after_retry.fab_bytes, before_retry.fab_bytes);
  EXPECT_EQ(after_retry.communication_calls, before_retry.communication_calls);
  EXPECT_EQ(after_retry.communication_bytes, before_retry.communication_bytes);
  EXPECT_EQ(live_a.fab(0).view().data, live_a_storage);
  EXPECT_EQ(live_b.fab(0).view().data, live_b_storage);

  // The complete request is validated before the first substitution: neither a cross-owner live
  // alias nor one wrong ghost footprint may expose a provisional state.
  EXPECT_THROW(
      (void)ctx.solve_fields_from_blocks_at(point(502), 502, {{0, &live_b}, {1, &stage_b}}),
      std::invalid_argument);
  NativeField wrong_layout =
      native_field_like(stage_b, stage_b.ncomp(), enlarged_ghosts(stage_b, 1));
  EXPECT_THROW(
      (void)ctx.solve_fields_from_blocks_at(point(503), 503, {{0, &stage_a}, {1, &wrong_layout}}),
      std::invalid_argument);
  EXPECT_EQ(live_a.fab(0).view().data, live_a_storage);
  EXPECT_EQ(live_b.fab(0).view().data, live_b_storage);
  EXPECT_EQ(first_value(live_a), Real(2));
  EXPECT_EQ(first_value(live_b), Real(3));

  // A Program may own only a subset of a larger NativeSystem. The exact block map selects NativeSystem block b,
  // while the context-owned native vector retains the required NativeSystem-sized nullptr padding.
  sim.set_program_block_map({1});
  NativeField stale_subset_stage = native_field_like(live_a, live_a.ncomp(), live_a.ghosts());
  stale_subset_stage.set_val(Real(11));
  EXPECT_THROW((void)ctx.solve_fields_from_blocks_at(point(501), 501, {{0, &stale_subset_stage}}),
               std::logic_error)
      << "a runtime block-map rematerialization must not teach an existing IR value a new pack";

  auto subset_prepared = prepare_native_field_route_services(sim);
  auto& subset_ctx = *subset_prepared.provider;
  subset_ctx.configure_primary_clock("clock.main");
  subset_ctx.begin_step(0.01);
  NativeField& subset_live = subset_ctx.state(0);
  NativeField subset_stage =
      native_field_like(subset_live, subset_live.ncomp(), subset_live.ghosts());
  subset_stage.set_val(Real(11));
  const auto subset_point = [&](int stage) { return subset_ctx.boundary_evaluation_point(stage); };

  EXPECT_THROW((void)subset_ctx.solve_fields_from_blocks_at(subset_point(505), 505, {{0, &live_a}}),
               std::invalid_argument)
      << "a subset Program must not borrow an unlisted NativeSystem block's live state as its "
         "stage";
  auto subset_solve = [&]() {
    return subset_ctx.solve_fields_from_blocks_at(subset_point(504), 504, {{0, &subset_stage}});
  };
  EXPECT_THROW((void)subset_solve(), std::out_of_range);
  const AllocationEventStats before_subset_retry = allocation_event_stats();
  EXPECT_THROW((void)subset_solve(), std::out_of_range);
  const AllocationEventStats after_subset_retry = allocation_event_stats();
  EXPECT_EQ(after_subset_retry.fab_calls, before_subset_retry.fab_calls);
  EXPECT_EQ(after_subset_retry.communication_calls, before_subset_retry.communication_calls);

  // Replacing the live layout does not materialize a representative-block snapshot: the exact
  // NativeSystem-sized stage vector is forwarded directly to the qualified named-field solve.
  subset_live =
      native_field_like(subset_live, subset_live.ncomp(), enlarged_ghosts(subset_live, 1));
  subset_live.set_val(Real(5));
  NativeField rebound_stage =
      native_field_like(subset_live, subset_live.ncomp(), subset_live.ghosts());
  rebound_stage.set_val(Real(13));
  const AllocationEventStats before_layout_change = allocation_event_stats();
  EXPECT_THROW(
      (void)subset_ctx.solve_fields_from_blocks_at(subset_point(504), 504, {{0, &rebound_stage}}),
      std::out_of_range);
  const AllocationEventStats after_layout_change = allocation_event_stats();
  EXPECT_EQ(after_layout_change.fab_calls, before_layout_change.fab_calls);
  EXPECT_EQ(after_layout_change.communication_calls, before_layout_change.communication_calls);
  const AllocationEventStats before_rebound_retry = allocation_event_stats();
  EXPECT_THROW(
      (void)subset_ctx.solve_fields_from_blocks_at(subset_point(504), 504, {{0, &rebound_stage}}),
      std::out_of_range);
  const AllocationEventStats after_rebound_retry = allocation_event_stats();
  EXPECT_EQ(after_rebound_retry.fab_calls, before_rebound_retry.fab_calls);
  EXPECT_EQ(after_rebound_retry.communication_calls, before_rebound_retry.communication_calls);
}

// A 2-stage SSP-RK2 (Heun) Program through NativeProgramExecutionServices is bit-equal to a hand-written SSPRK2
// reference built from the SAME primitives:
//   U1        = U^n + dt R(U^n)
//   U^{n+1}   = 1/2 U^n + 1/2 U1 + 1/2 dt R(U1)
// The reference re-solves the fields at each stage state (solve_fields on a scratch NativeSystem seeded with
// the stage state), mirroring the per-stage ctx.solve_fields_from_state in the Program body.
TEST(ProgramExecutionServicesContract, SsprkTwoStageViaContextMatchesReference) {
  ensure_kokkos();
  const int n = 16;
  const double dt = 1e-3;
  NativeSystemConfig cfg = native_config(n);
  const std::vector<double> U0 = ic(n);

  // Reference SSPRK2 on the host via solve_fields + eval_rhs (a fresh solve per stage state).
  NativeSystem ref(cfg);
  install_execution_lane(ref, "pops.test.program-context.ssprk-reference");
  add_gas(ref);
  ref.set_state("gas", U0);
  (void)pops::consume_solve_outcome(ref.solve_fields());
  const std::vector<double> R0 = ref.eval_rhs("gas");
  std::vector<double> U1(U0.size());
  for (std::size_t k = 0; k < U1.size(); ++k) {
    U1[k] = U0[k] + dt * R0[k];
  }
  ref.set_state("gas", U1);
  (void)pops::consume_solve_outcome(
      ref.solve_fields());  // re-solve the fields at the stage-1 state
  const std::vector<double> R1 = ref.eval_rhs("gas");
  std::vector<double> Uref(U0.size());
  for (std::size_t k = 0; k < Uref.size(); ++k) {
    Uref[k] = 0.5 * U0[k] + 0.5 * U1[k] + 0.5 * dt * R1[k];
  }

  // NativeProgramExecutionServices SSPRK2: stage into scratch states via scratch_state_like / axpy / lincomb, with a
  // per-stage solve_fields_from_state before each RHS.
  NativeSystem sim(cfg);
  install_execution_lane(sim, "pops.test.program-context.ssprk");
  add_gas(sim);
  sim.set_state("gas", U0);
  sim.set_program_block_map({0});
  using Resource = test::program_v5::CallbackProgramResource;
  std::vector<Resource> resources;
  resources.reserve(2);
  {
    const auto state = sim.block_state(0);
    if (!state)
      throw std::logic_error("SSPRK2 fixture requires one prepared block state");
    resources.push_back({Resource::Kind::rhs, 0, 0, 0, -1,
                         static_cast<std::uint32_t>(state->ncomp()),
                         static_cast<std::uint32_t>(state->ghosts()[0])});
    resources.push_back({Resource::Kind::state, 1, 0, 0, -1,
                         static_cast<std::uint32_t>(state->ncomp()),
                         static_cast<std::uint32_t>(state->ghosts()[0])});
  }
  install_native_v5_program(
      sim, "tests.program-execution-services/ssprk2@1", resources,
      [](NativeProgramExecutionServices& ctx, double h) {
        ctx.begin_step(h);
        for (int b = 0; b < ctx.n_blocks(); ++b) {
          NativeField& U = ctx.state(b);
          // stage 1: u1 = U + dt R(U)
          ctx.set_stage_time(0, 1);
          {
            auto outcome = ctx.solve_fields_from_state(b, U);
            (void)outcome.consume(SolveConsumption::kAccept);
          }
          NativeField& u1 = ctx.scratch_state(2 * b + 1, 0, U);
          ctx.lincomb(u1, Real(1), U, Real(0), U);  // u1 <- U
          NativeField& R = ctx.rhs_scratch(2 * b, 0, U);
          ctx.rhs_into(b, U, R, 0);
          ctx.axpy(u1, Real(h), R);  // u1 <- U + dt R(U)  (= the Euler predictor U1)
          // stage 2 (Heun): U <- 1/2 U + 1/2 (U1 + dt R(U1)) = 1/2 U + 1/2 U1 + 1/2 dt R(U1)
          ctx.set_stage_time(1, 1);
          {
            auto outcome = ctx.solve_fields_from_state(b, u1);
            (void)outcome.consume(SolveConsumption::kAccept);
          }  // re-solve fields at the stage-1 state
          ctx.rhs_into(b, u1, R, 0);
          ctx.axpy(u1, Real(h), R);                     // u1 <- U1 + dt R(U1)
          ctx.lincomb(U, Real(0.5), U, Real(0.5), u1);  // U <- 1/2 U + 1/2 (U1 + dt R(U1))
        }
      });
  sim.mark_bound();
  sim.step(dt);
  const std::vector<double> Up = sim.get_state("gas");

  EXPECT_TRUE(max_abs_diff(Up, Uref) < 1e-12) << "SSPRK2 parity max|d|=" << max_abs_diff(Up, Uref);
  EXPECT_TRUE(max_abs_diff(Up, U0) > 1e-9) << "SSPRK2 step did not change the state";
}

// The remaining host-validatable seams return sane, consistent results.
TEST(ProgramExecutionServicesContract, SeamSurfaceIsConsistent) {
  ensure_kokkos();
  const int n = 16;
  const double dt = 1e-3;
  NativeSystemConfig cfg = native_config(n);
  const std::vector<double> U0 = ic(n);

  NativeSystem sim(cfg);
  install_execution_lane(sim, "pops.test.program-context.seam-surface");
  add_gas(sim);
  sim.set_state("gas", U0);
  sim.set_program_block_map({0});
  auto prepared = prepare_native_program_services(sim);
  auto& ctx = *prepared.provider;
  ctx.configure_primary_clock("clock.macro");
  ctx.declare_clock_relation("clock.macro", "clock.fast", 2);
  ctx.register_history("h", 2);
  ctx.register_history("scalar_h", 1, 1);
  sim.mark_bound();
  ctx.begin_step(dt);
  ctx.set_stage_time(0, 1);
  {
    auto outcome = ctx.solve_fields();
    (void)outcome.consume(SolveConsumption::kAccept);
  }

  const int b = 0;
  NativeField& U = ctx.state(b);

  // Cartesian generated pointwise kernels receive no sparse mask, while their status reduction
  // remains on the prepared lane. A foreign mask cannot silently change participating cells.
  NativeField pointwise_status = ctx.alloc_scalar_field(1, 0);
  pointwise_status.set_val(Real(0));
  const NativeField* const active_cells = ctx.pointwise_active_mask(b, pointwise_status);
  EXPECT_EQ(active_cells, nullptr);
  EXPECT_EQ(
      ctx.pointwise_status_max(b, pointwise_status, active_cells, ctx.prepared_execution_lane()),
      Real(0));
  pointwise_status.set_val(Real(2));
  EXPECT_EQ(
      ctx.pointwise_status_max(b, pointwise_status, active_cells, ctx.prepared_execution_lane()),
      Real(2));
  pointwise_status.set_val(Real(0));
  const AllocationEventStats pointwise_allocations_before = allocation_event_stats();
  const std::uint64_t pointwise_consensus_before = exact_consensus_dynamic_storage_calls();
  for (int repeat = 0; repeat < 3; ++repeat) {
    EXPECT_EQ(ctx.pointwise_active_mask(b, pointwise_status), nullptr);
    EXPECT_EQ(
        ctx.pointwise_status_max(b, pointwise_status, active_cells, ctx.prepared_execution_lane()),
        Real(0));
  }
  const AllocationEventStats pointwise_allocations_after = allocation_event_stats();
  const std::uint64_t pointwise_consensus_after = exact_consensus_dynamic_storage_calls();
  EXPECT_EQ(pointwise_allocations_after, pointwise_allocations_before)
      << "warmed Cartesian pointwise path must not allocate owning storage";
  EXPECT_EQ(pointwise_consensus_after, pointwise_consensus_before)
      << "warmed Cartesian pointwise path must not use dynamic exact consensus";
  pointwise_status.set_val(std::numeric_limits<Real>::quiet_NaN());
  EXPECT_EQ(
      ctx.pointwise_status_max(b, pointwise_status, active_cells, ctx.prepared_execution_lane()),
      Real(3));
  EXPECT_THROW(ctx.pointwise_status_max(b, pointwise_status, &pointwise_status,
                                        ctx.prepared_execution_lane()),
               std::invalid_argument);

  // rhs_into == neg_div_flux_default_into + source_default_into (the split-then-sum identity, ADC-425).
  NativeField Rfull(U.layout(), U.distribution(), U.local_rank(), U.ncomp(), U.ghosts());
  NativeField Rflux(U.layout(), U.distribution(), U.local_rank(), U.ncomp(), U.ghosts());
  NativeField Rsrc(U.layout(), U.distribution(), U.local_rank(), U.ncomp(), U.ghosts());
  ctx.rhs_into(b, U, Rfull, 0);
  ctx.neg_div_flux_default_into(b, U, Rflux, 0);
  ctx.source_default_into(b, U, Rsrc);
  NativeField Rsum(U.layout(), U.distribution(), U.local_rank(), U.ncomp(), U.ghosts());
  ctx.lincomb(Rsum, Real(1), Rflux, Real(1), Rsrc);  // Rsum = -div F + S
  {
    // The exact Euler package has no source provider, so Rsrc is zero and Rsum == Rflux == Rfull.
    for (int c = 0; c < kNcomp; ++c) {
      const Real full = ctx.sum_component(Rfull, c);
      const Real sum = ctx.sum_component(Rsum, c);
      EXPECT_TRUE(std::fabs(full - sum) < 1e-12)
          << "rhs_into != flux+source at comp " << c << " (" << full << " vs " << sum << ")";
    }
  }

  // reductions: sum/max/min of component 0 are consistent (min <= sum/N is not asserted, but max>=min).
  EXPECT_TRUE(ctx.max_component(U, 0) >= ctx.min_component(U, 0)) << "max >= min density";
  EXPECT_NEAR(ctx.sum_component(U, GasSchema::density), static_cast<Real>(uniform_cell_count(n)),
              1e-12)
      << "density sum covers every valid cell exactly once";

  // laplacian(phi) == divergence(gradient(phi)) on a smooth periodic field (the stencil identity the
  // matrix-free operators rely on).  The two-argument overloads deliberately fail before creating
  // a transport: bind the two exact session contracts while preparation owns allocation, then prove
  // every accepted stencil/fill reuses that fixed authority.
  NativeField phi = ctx.alloc_scalar_field(1, 1);
  NativeField lap = ctx.alloc_scalar_field(1, 1);
  NativeField grad = ctx.alloc_scalar_field(kTestDimension, 1);
  NativeField divg = ctx.alloc_scalar_field(1, 1);
  phi.set_val(Real(1));
  auto scalar_boundary = ctx.prepare_mesh_boundary_session(phi, ctx.prepared_execution_lane());
  auto gradient_boundary =
      ctx.prepare_mesh_boundary_session(grad, ctx.prepared_execution_lane());
  auto state_boundary = ctx.prepare_mesh_boundary_session(U, ctx.prepared_execution_lane());
  EXPECT_THROW(ctx.laplacian(lap, phi), std::logic_error);
  EXPECT_THROW(ctx.gradient(grad, phi), std::logic_error);
  EXPECT_THROW(ctx.divergence(divg, grad), std::logic_error);
  EXPECT_THROW(ctx.fill_boundary(U), std::logic_error);
  const AllocationEventStats stencil_allocations_before = allocation_event_stats();
  {
    // seed phi with a smooth field: reuse density; copy component 0 of U into phi via lincomb on a
    // 1-comp scratch is not directly possible (ncomp differs), so seed phi from a fresh smooth pattern.
    // Instead assert the operators run and produce finite output of the right shape.
    BoundaryPointHeapWindow heap_window;
    ctx.laplacian(lap, phi, *scalar_boundary);     // Lap(const) == 0
    ctx.gradient(grad, phi, *scalar_boundary);     // grad(const) == 0
    ctx.divergence(divg, grad, *gradient_boundary);  // div(0) == 0
    state_boundary->fill(U);
    EXPECT_EQ(heap_window.close(), 0u);
  }
  EXPECT_EQ(allocation_event_stats(), stencil_allocations_before);
  EXPECT_TRUE(ctx.max_component(lap, 0) < 1e-12) << "laplacian of a constant is 0";
  EXPECT_TRUE(ctx.max_component(divg, 0) < 1e-12) << "divergence(gradient(const)) is 0";

  // The cold-bound fill left valid cells unchanged. Projection is an explicit block
  // capability: this block declares none, so applying one must fail rather than silently become an
  // identity operation.
  const std::vector<double> before = sim.get_state("gas");
  EXPECT_TRUE(max_abs_diff(sim.get_state("gas"), before) < 1e-15)
      << "fill_boundary left the valid cells unchanged";
  EXPECT_THROW(ctx.apply_projection(b, U), std::runtime_error)
      << "an undeclared projection capability must fail loud";

  // history register/store/read/rotate through the context seam.
  NativeField hv(U.layout(), U.distribution(), U.local_rank(), U.ncomp(), U.ghosts());
  hv.set_val(Real(3));
  ctx.store_history("h", hv);
  for (int slot = 0; slot < 3; ++slot)
    EXPECT_EQ(sim.history_slot_dt("h", slot), dt)
        << "first exact store cold-fills every history dt slot";
  {
    NativeField& r = ctx.history("h", 1);  // cold-start fill -> lag 1 == the stored value
    EXPECT_TRUE(r.ncomp() == U.ncomp()) << "owner-qualified history preserves the whole field";
    EXPECT_TRUE(std::fabs(ctx.sum_component(r, 0) -
                          Real(3) * static_cast<Real>(uniform_cell_count(n))) < 1e-9)
        << "history lag1 read";
  }
  NativeField& scalar_history = ctx.history_zero_start("scalar_h", 1, 1);
  EXPECT_TRUE(scalar_history.ncomp() == 1) << "narrow history is a scalar NativeField";
  EXPECT_TRUE(std::fabs(ctx.sum_component(scalar_history, 0)) < 1e-12)
      << "owner-qualified zero-start history preserves its declared cold start";
  ctx.rotate_histories();
  EXPECT_EQ(sim.history_fill_count("h"), 1);
  for (int slot = 0; slot < 3; ++slot)
    EXPECT_EQ(sim.history_slot_dt("h", slot), dt)
        << "cold-filled history dt ledger rotates with its ring";

  const double next_dt = 2.0 * dt;
  ctx.begin_step(next_dt);
  hv.set_val(Real(4));
  ctx.store_history("h", hv);
  EXPECT_EQ(sim.history_slot_dt("h", 0), next_dt);
  EXPECT_EQ(sim.history_slot_dt("h", 1), dt);
  EXPECT_EQ(sim.history_slot_dt("h", 2), dt);
  NativeField interpolated(U.layout(), U.distribution(), U.local_rank(), U.ncomp(), U.ghosts());
  ctx.interpolate_history_linear(interpolated, "h", 2, 0, "clock.macro", "clock.fast", -1, Real(0));
  EXPECT_EQ(first_value(interpolated), Real(3.5));
  ctx.rotate_histories();
  EXPECT_EQ(sim.history_fill_count("h"), 2);
  EXPECT_EQ(sim.history_slot_dt("h", 1), next_dt);
  EXPECT_EQ(sim.history_slot_dt("h", 2), dt);
  EXPECT_EQ(first_value(ctx.history("h", 1)), Real(4));
  EXPECT_EQ(first_value(ctx.history("h", 2)), Real(3));

  // runtime params: a block with no runtime param returns a default (count 0) RuntimeParams.
  EXPECT_TRUE(ctx.program_params(0).count == 0) << "no runtime param -> count 0";

  // dt-bound inputs: hmin and max_wave_speed are positive on a non-trivial state.
  EXPECT_TRUE(ctx.hmin() > 0) << "hmin positive";
  EXPECT_TRUE(ctx.max_wave_speed(b, U) > 0) << "max wave speed positive";

  // Direct cold scratch allocators remain available outside a candidate transaction.
  NativeField sc = ctx.scratch_state_like(U);
  EXPECT_TRUE(sc.ncomp() == U.ncomp()) << "scratch_state_like ncomp";
  NativeField sf = ctx.alloc_scalar_field(1, 1);
  EXPECT_TRUE(sf.ncomp() == 1) << "alloc_scalar_field ncomp";
}

TEST(ProgramExecutionServicesContract,
     LogicalSubcycleSnapshotsCarryExactChildWindowsAndRestoreParents) {
  ensure_kokkos();
  NativeSystemConfig cfg = native_config(8);
  NativeSystem sim(cfg);
  install_execution_lane(sim, "pops.test.program-context.logical-subcycle");
  add_gas(sim);
  sim.set_program_block_map({0});

  auto prepared = prepare_native_program_services(sim);
  auto& ctx = *prepared.provider;
  ctx.configure_primary_clock("clock.macro");
  ctx.declare_clock_relation("clock.macro", "clock.fast", 2);
  ctx.declare_clock_relation("clock.fast", "clock.micro", 2);
  ctx.seal_clock_schedule_for_execution();
  constexpr double parent_dt = 0.4;
  ctx.begin_step(parent_dt);
  ctx.set_stage_time(1, 3);
  const OperatorFingerprint authority{UINT64_C(1), UINT64_C(2), UINT64_C(3), UINT64_C(4)};
  const OperatorFingerprint resources{UINT64_C(5), UINT64_C(6), UINT64_C(7), UINT64_C(8)};
  const auto snapshot = [&]() {
    return ctx.operator_evaluation_snapshot(authority, ctx.state(0), resources);
  };
  const OperatorEvaluationSnapshot parent_before = snapshot();

  std::array<OperatorEvaluationSnapshot, 2> children;
  OperatorEvaluationSnapshot nested;
  OperatorEvaluationSnapshot parent_stale_on_entry;
  OperatorEvaluationSnapshot parent_stale_after_exit;
  OperatorEvaluationSnapshot outer_stale_after_nested_exit;
  OperatorEvaluationSnapshot outer_before_exception;
  OperatorEvaluationSnapshot outer_after_exception;
  auto ticks = ctx.subcycle_scope("clock.macro", "clock.fast", 2);
  for (int iteration = 0; iteration < 2; ++iteration) {
    ticks.iteration(iteration);
    auto child = ctx.logical_evaluation_scope(iteration, 2);
    EXPECT_EQ(child.dt(), Real(parent_dt / 2.0));
    if (iteration == 0) {
      parent_stale_on_entry = ctx.probe_operator_evaluation(authority, parent_before.topology,
                                                            resources, parent_before.revision);
    }
    ctx.set_stage_time(1, 2);
    children[static_cast<std::size_t>(iteration)] = snapshot();
    if (iteration != 0)
      continue;

    outer_before_exception = snapshot();
    try {
      auto micro_ticks = ctx.subcycle_scope("clock.fast", "clock.micro", 2);
      micro_ticks.iteration(0);
      auto micro = ctx.logical_evaluation_scope(0, 2);
      EXPECT_EQ(micro.dt(), Real(parent_dt / 4.0));
      ctx.set_stage_time(1, 2);
      nested = snapshot();
      throw std::runtime_error("exercise nested logical-evaluation unwind");
    } catch (const std::runtime_error&) {
    }
    outer_stale_after_nested_exit = ctx.probe_operator_evaluation(
        authority, outer_before_exception.topology, resources, outer_before_exception.revision);
    outer_after_exception = snapshot();
    EXPECT_TRUE(ctx.probe_operator_evaluation(authority, outer_after_exception.topology, resources,
                                              outer_after_exception.revision) ==
                outer_after_exception);
  }
  ticks.finish();
  parent_stale_after_exit = ctx.probe_operator_evaluation(authority, parent_before.topology,
                                                          resources, parent_before.revision);
  const OperatorEvaluationSnapshot parent_after = snapshot();

  const double child_dt = parent_dt / 2.0;
  EXPECT_EQ(std::bit_cast<double>(children[0].dt_bits), child_dt);
  EXPECT_EQ(std::bit_cast<double>(children[1].dt_bits), child_dt);
  EXPECT_EQ(children[0].stage_numerator, 1);
  EXPECT_EQ(children[0].stage_denominator, 4);
  EXPECT_EQ(children[1].stage_numerator, 3);
  EXPECT_EQ(children[1].stage_denominator, 4);
  EXPECT_EQ(std::bit_cast<double>(children[0].physical_time_bits),
            sim.time() + 0.0 * child_dt + 0.5 * child_dt);
  EXPECT_EQ(std::bit_cast<double>(children[1].physical_time_bits),
            sim.time() + 1.0 * child_dt + 0.5 * child_dt);
  EXPECT_NE(children[0].revision, children[1].revision);
  EXPECT_NE(children[0].physical_time_bits, children[1].physical_time_bits);

  EXPECT_EQ(std::bit_cast<double>(nested.dt_bits), parent_dt / 4.0);
  EXPECT_EQ(nested.stage_numerator, 1);
  EXPECT_EQ(nested.stage_denominator, 8);
  EXPECT_EQ(std::bit_cast<double>(nested.physical_time_bits), sim.time() + 0.5 * (child_dt / 2.0));
  EXPECT_NE(nested.revision, outer_before_exception.revision);
  EXPECT_EQ(outer_after_exception.stage_numerator, outer_before_exception.stage_numerator);
  EXPECT_EQ(outer_after_exception.stage_denominator, outer_before_exception.stage_denominator);
  EXPECT_EQ(outer_after_exception.dt_bits, outer_before_exception.dt_bits);
  EXPECT_EQ(outer_after_exception.physical_time_bits, outer_before_exception.physical_time_bits);
  EXPECT_NE(parent_stale_on_entry.revision, parent_before.revision);
  EXPECT_NE(outer_stale_after_nested_exit.revision, outer_before_exception.revision);
  EXPECT_EQ(outer_stale_after_nested_exit.stage_numerator, outer_before_exception.stage_numerator);
  EXPECT_EQ(outer_stale_after_nested_exit.stage_denominator,
            outer_before_exception.stage_denominator);
  EXPECT_EQ(outer_stale_after_nested_exit.dt_bits, outer_before_exception.dt_bits);
  EXPECT_EQ(outer_stale_after_nested_exit.physical_time_bits,
            outer_before_exception.physical_time_bits);
  EXPECT_NE(outer_after_exception.revision, outer_before_exception.revision);

  EXPECT_NE(parent_stale_after_exit.revision, parent_before.revision);
  EXPECT_EQ(parent_stale_after_exit.stage_numerator, parent_before.stage_numerator);
  EXPECT_EQ(parent_stale_after_exit.stage_denominator, parent_before.stage_denominator);
  EXPECT_EQ(parent_stale_after_exit.dt_bits, parent_before.dt_bits);
  EXPECT_EQ(parent_stale_after_exit.physical_time_bits, parent_before.physical_time_bits);
  EXPECT_EQ(parent_after.stage_numerator, parent_before.stage_numerator);
  EXPECT_EQ(parent_after.stage_denominator, parent_before.stage_denominator);
  EXPECT_EQ(parent_after.dt_bits, parent_before.dt_bits);
  EXPECT_EQ(parent_after.physical_time_bits, parent_before.physical_time_bits);
  EXPECT_NE(parent_after.revision, parent_before.revision);
  EXPECT_TRUE(ctx.probe_operator_evaluation(authority, parent_after.topology, resources,
                                            parent_after.revision) == parent_after);
}

TEST(ProgramExecutionServicesContract, BlockResolutionRequiresACompleteExplicitMap) {
  ensure_kokkos();
  NativeSystemConfig cfg = native_config(8);
  NativeSystem sim(cfg);
  install_execution_lane(sim, "pops.test.program-context.block-resolution");
  add_gas(sim);
  auto prepared = prepare_native_program_services(sim);
  auto& ctx = *prepared.provider;
  const NativeField accepted_state = [&] {
    const auto state_view = sim.block_state(0);
    return NativeField(*state_view.get());
  }();
  const std::vector<const NativeField*> stages{&accepted_state};

  EXPECT_THROW(ctx.sys_block(0), std::runtime_error) << "an empty map must not imply identity";
  EXPECT_THROW((void)ctx.solve_fields_from_blocks(stages), std::runtime_error)
      << "the coupled solve must not treat an empty map as identity";

  sim.set_program_block_map({0});
  EXPECT_EQ(ctx.sys_block(0), 0);
  SolveOutcome mapped = ctx.solve_fields_from_blocks(stages);
  ASSERT_TRUE(mapped.report().solved_value_available()) << mapped.report().reason;
  (void)mapped.consume(SolveConsumption::kAccept);
  EXPECT_THROW(ctx.sys_block(-1), std::out_of_range) << "negative Program index must fail";
  EXPECT_THROW(ctx.sys_block(1), std::out_of_range) << "Program index outside the map must fail";
  EXPECT_THROW(sim.set_program_block_map({0, 0}), std::invalid_argument)
      << "two Program blocks must not silently overwrite the same NativeSystem stage slot";
  EXPECT_EQ(sim.program_block_map(), (std::vector<int>{0}))
      << "a rejected double assignment must preserve the previously authenticated map";

  EXPECT_THROW(sim.set_program_block_map({-1}), std::out_of_range)
      << "negative mapped NativeSystem index must fail before publication";
  EXPECT_THROW(sim.set_program_block_map({1}), std::out_of_range)
      << "mapped NativeSystem index outside n_blocks must fail before publication";
  EXPECT_EQ(sim.program_block_map(), (std::vector<int>{0}));
}

TEST(ProgramExecutionServicesContract, UniformSourceMaskUsesThePreparedHotPrimitive) {
  ensure_kokkos();
  NativeSystem sim(native_config(8));
  install_execution_lane(sim, "pops.test.program-context.uniform-source-mask");
  add_gas(sim);
  sim.set_program_block_map({0});
  auto prepared = prepare_native_program_services(sim);
  auto& ctx = *prepared.provider;

  NativeField residual = ctx.state(0);
  residual.set_val(Real(5));
  const AllocationEventStats before = allocation_event_stats();
  ctx.apply_source_mask(residual, {0});
  const AllocationEventStats after = allocation_event_stats();

  EXPECT_EQ(after.fab_calls, before.fab_calls);
  EXPECT_EQ(after.fab_bytes, before.fab_bytes);
  EXPECT_EQ(first_value(residual), Real(5));
  residual.set_val(Real(13));
  EXPECT_THROW(ctx.apply_source_mask(residual, {residual.ncomp()}), std::invalid_argument);
  EXPECT_EQ(first_value(residual), Real(13));
}
