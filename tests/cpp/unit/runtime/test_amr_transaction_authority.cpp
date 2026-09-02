/// @file
/// @brief AMR integration witnesses for the single Program transaction authority.

#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "explicit_amr_program.hpp"

#include <pops/core/foundation/allocator.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>

#include <bit>
#include <chrono>
#include <atomic>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <future>
#include <new>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

// Keep the standard-heap witness armed only around the prepared per-block diagnostic.  The
// aggregate compatibility API intentionally materializes a std::map, so it is never measured by
// this counter; AllocationEventStats below still covers Fab/communication storage for both APIs.
std::atomic<bool> g_heap_measurement_enabled{false};
std::atomic<std::uint64_t> g_measured_heap_allocations{0};

void note_measured_heap_allocation() noexcept {
  if (g_heap_measurement_enabled.load(std::memory_order_relaxed))
    g_measured_heap_allocations.fetch_add(1, std::memory_order_relaxed);
}

void* measured_allocate(std::size_t size) {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer == nullptr)
    throw std::bad_alloc();
  note_measured_heap_allocation();
  return pointer;
}

void* measured_allocate_nothrow(std::size_t size) noexcept {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer != nullptr)
    note_measured_heap_allocation();
  return pointer;
}

void* measured_aligned_allocate(std::size_t size, std::size_t alignment) {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) != 0)
    pointer = nullptr;
  if (pointer == nullptr)
    throw std::bad_alloc();
  note_measured_heap_allocation();
  return pointer;
}

void* measured_aligned_allocate_nothrow(std::size_t size, std::size_t alignment) noexcept {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) != 0)
    pointer = nullptr;
  if (pointer != nullptr)
    note_measured_heap_allocation();
  return pointer;
}

class HeapAllocationWindow final {
 public:
  HeapAllocationWindow() : before_(g_measured_heap_allocations.load(std::memory_order_relaxed)) {
    g_heap_measurement_enabled.store(true, std::memory_order_relaxed);
  }

  [[nodiscard]] std::uint64_t close() noexcept {
    g_heap_measurement_enabled.store(false, std::memory_order_relaxed);
    return g_measured_heap_allocations.load(std::memory_order_relaxed) - before_;
  }

 private:
  std::uint64_t before_ = 0;
};

}  // namespace

void* operator new(std::size_t size) {
  return measured_allocate(size);
}
void* operator new[](std::size_t size) {
  return measured_allocate(size);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  return measured_allocate_nothrow(size);
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  return measured_allocate_nothrow(size);
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
  return measured_aligned_allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return measured_aligned_allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return measured_aligned_allocate_nothrow(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return measured_aligned_allocate_nothrow(size, static_cast<std::size_t>(alignment));
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

namespace {

template <int Dim>
struct ScalarAdvectionModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;

  Law law{};

  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr-transaction-authority.scalar-advection", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(law.velocity()[axis]);
  }
  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"u"}, 1, {pops::VariableRole::Scalar}};
  }
  POPS_HD pops::nd::StateConversion<Primitive> recover(const State& state) const {
    return law.recover(state);
  }
  POPS_HD pops::nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return law.make_conservative(primitive);
  }
  POPS_HD pops::nd::StateConversionStatus admissibility(const State& state) const {
    return law.admissibility(state);
  }
  template <int Axis>
  POPS_HD State flux(const State& state) const {
    return law.template flux<Axis>(state);
  }
  template <int Axis>
  POPS_HD pops::Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, pops::Real& lower, pops::Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return {}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

struct ProgramAmrCouplingWitness final {
  // The callback contributes one authored operation.  Its prepared hierarchy dispatches that
  // operation over the finite level/subcycle groups; fixed storage keeps the witness itself out of
  // the hot allocator path.
  bool exercise_hierarchy_coupling = false;
  std::atomic<bool> reject_after_coupling{false};
  std::uint64_t operation_count = 0;
  std::uint64_t level_dispatches = 0;
  std::uint64_t coupling_evaluations = 0;
  std::array<int, 64> dispatched_levels{};

  void record_level(int level) {
    if (level_dispatches >= dispatched_levels.size())
      throw std::runtime_error("AMR Program coupling dispatch witness exceeded its cold bound");
    dispatched_levels[static_cast<std::size_t>(level_dispatches++)] = level;
  }
};

template <int Dim>
struct ConservativeProgramExchange final {
  pops::FieldView<pops::Real, Dim> donor{};
  pops::FieldView<pops::Real, Dim> receiver{};
  pops::Real dt = pops::Real(0);

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    const pops::Real amount = dt * pops::Real(0.25) * donor(cell, 0);
    donor(cell, 0) -= amount;
    receiver(cell, 0) += amount;
  }
};

template <int Dim, class Function>
void for_each_host_cell(const pops::Box<Dim>& box, Function&& function) {
  const pops::Extent<Dim> extent = box.extent();
  for (std::int64_t ordinal = 0; ordinal < box.numPts(); ++ordinal) {
    std::int64_t remainder = ordinal;
    pops::Index<Dim> cell = box.lo;
    for (int axis = 0; axis < Dim; ++axis) {
      cell[axis] += static_cast<int>(remainder % extent[axis]);
      remainder /= extent[axis];
    }
    function(cell);
  }
}

template <int Dim>
std::unique_ptr<pops::AmrSystem<Dim>> make_system(int program_stride = 1) {
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  auto owner = std::make_unique<pops::AmrSystem<Dim>>(config);
  auto& system = *owner;
  pops::test::install_amr_runtime_authority(system, "tests.amr-transaction-authority/runtime@1");
  system.install_block_state_route("tracer", "tests.amr-transaction-authority/state@1");
  pops::RealVector<Dim> velocity{};
  velocity[0] = pops::Real(1);
  pops::add_compiled_model<Dim>(
      system, "tracer",
      ScalarAdvectionModel<Dim>{pops::nd::ScalarAdvection<Dim>::prepare(velocity)}, "minmod",
      "rusanov", "conservative", "explicit", static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1,
      {}, {}, 0.0, static_cast<double>(pops::kWenoEpsilon), false,
      "tests.amr-transaction-authority/physical-flux@1");
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(config.shape[axis]);
  system.set_conservative_state("tracer", std::vector<double>(cells, 1.0));
  system.set_program_cadence(1, program_stride);
  using Resource = pops::test::program_v5::CallbackProgramResource;
  (void)system.n_levels();
  std::uint32_t state_components = 0;
  std::uint32_t state_ghosts = 0;
  {
    const auto state = system.prepared_amr_block_state(0, 0);
    if (!state)
      throw std::logic_error("AMR transaction authority fixture lost its prepared state");
    state_components = static_cast<std::uint32_t>(state->ncomp());
    state_ghosts = static_cast<std::uint32_t>(state->ghosts()[0]);
  }
  const std::vector<Resource> resources{
      {Resource::Kind::rhs, 0, 0, 0, 0, state_components, state_ghosts}};
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.amr-transaction-authority/forward-euler@1", "test.clock.macro", resources, {},
      [](auto& context, double macro_dt) {
        context.advance_hierarchy(macro_dt, [&context](double level_dt) {
          context.set_stage_time(0, 1);
          for (int block = 0; block < context.n_blocks(); ++block) {
            auto& state = context.state(block);
            auto& rhs = context.rhs_scratch(static_cast<pops::runtime::program::ProgramCacheSlot>(
                                                context.level() * context.n_blocks() + block),
                                            0, state);
            // This authority witness exercises transaction state and prepared scratch, not the
            // separate face-flux expression ledger.  Seed the already-primed RHS directly so the
            // test remains valid with the strict rule that an active flux basis must carry an
            // authenticated nonzero expression budget and static occurrence table.
            rhs.set_val(pops::Real(0));
            context.axpy(state, pops::Real(level_dt), rhs);
          }
        });
      });
  return owner;
}

template <int Dim>
std::unique_ptr<pops::AmrSystem<Dim>> make_lazy_runtime_system() {
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  auto owner = std::make_unique<pops::AmrSystem<Dim>>(config);
  auto& system = *owner;
  pops::test::install_amr_runtime_authority(system,
                                            "tests.amr-transaction-authority/lazy-runtime@1");
  system.install_block_state_route("tracer", "tests.amr-transaction-authority/lazy-state@1");
  pops::RealVector<Dim> velocity{};
  velocity[0] = pops::Real(1);
  pops::add_compiled_model<Dim>(
      system, "tracer",
      ScalarAdvectionModel<Dim>{pops::nd::ScalarAdvection<Dim>::prepare(velocity)}, "minmod",
      "rusanov", "conservative", "explicit", static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1,
      {}, {}, 0.0, static_cast<double>(pops::kWenoEpsilon), false,
      "tests.amr-transaction-authority/lazy-physical-flux@1");
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(config.shape[axis]);
  system.set_conservative_state("tracer", std::vector<double>(cells, 1.0));
  return owner;
}

template <int Dim>
std::unique_ptr<pops::AmrSystem<Dim>> make_all_cancel_static_flux_system(
    const std::shared_ptr<std::atomic<bool>>& throw_after_resident_write,
    bool outside_advance_hierarchy = false) {
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  auto owner = std::make_unique<pops::AmrSystem<Dim>>(config);
  auto& system = *owner;
  pops::test::install_amr_runtime_authority(system,
                                            "tests.amr-transaction-authority/static-flux@1");
  system.install_block_state_route("tracer", "tests.amr-transaction-authority/static-state@1");
  pops::RealVector<Dim> velocity{};
  velocity[0] = pops::Real(1);
  pops::add_compiled_model<Dim>(
      system, "tracer",
      ScalarAdvectionModel<Dim>{pops::nd::ScalarAdvection<Dim>::prepare(velocity)}, "minmod",
      "rusanov", "conservative", "explicit", static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1,
      {}, {}, 0.0, static_cast<double>(pops::kWenoEpsilon), false,
      "tests.amr-transaction-authority/static-physical-flux@1");
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(config.shape[axis]);
  system.set_conservative_state("tracer", std::vector<double>(cells, 1.0));
  system.set_program_cadence(1, 1);

  using Resource = pops::test::program_v5::CallbackProgramResource;
  system.refresh_prepared_amr_levels();
  std::uint32_t state_components = 0;
  std::uint32_t state_ghosts = 0;
  {
    const auto state = system.prepared_amr_block_state(0, 0);
    if (!state)
      throw std::logic_error("all-cancel static flux fixture lost its prepared state");
    state_components = static_cast<std::uint32_t>(state->ncomp());
    state_ghosts = static_cast<std::uint32_t>(state->ghosts()[0]);
  }
  Resource rhs_resource;
  rhs_resource.kind = Resource::Kind::rhs;
  rhs_resource.slot = 0;
  rhs_resource.subslot = 0;
  rhs_resource.program_block = 0;
  rhs_resource.level = 0;
  rhs_resource.components = state_components;
  rhs_resource.ghosts = state_ghosts;
  rhs_resource.value_id = 3000;
  rhs_resource.occurrence_path_id = 0x23000;
  rhs_resource.identity = "tests.amr-transaction-authority/static-rhs/3000@1";
  rhs_resource.occurrence_path = "root/static-rhs/3000";
  rhs_resource.owner = "tracer";
  rhs_resource.clock = "test.clock.macro";
  const std::vector<Resource> resources{std::move(rhs_resource)};
  const std::optional<std::vector<pops::runtime::program::ProgramFluxBudgetRecord>> budgets{
      std::vector<pops::runtime::program::ProgramFluxBudgetRecord>{{1, 0, 0, 0}}};
  const std::vector<pops::test::program_v5::CallbackProgramFluxBasisOccurrence> bases{{
      0,
      0,
      0,
      0,
      3000,
      0,
      0,
      1,
      "tests.amr-transaction-authority/static-basis@1",
      "root/static-basis/0",
      "tracer",
      "test.clock.macro",
  }};
  // No final term is the authenticated cancellation path.  The cold prime still binds an empty
  // resident ledger image, while hot RHS attachment writes only scalar resident basis metadata.
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.amr-transaction-authority/all-cancel-static-flux@1", "test.clock.macro",
      resources, {},
      [throw_after_resident_write, outside_advance_hierarchy](auto& context, double macro_dt) {
        if (outside_advance_hierarchy) {
          // This must fail before commit_many can capture a field snapshot.  An empty list keeps
          // the witness focused on the static-v5 transaction boundary rather than a field route.
          context.commit_many({});
          return;
        }
        context.advance_hierarchy(
            macro_dt, [&context, throw_after_resident_write](double level_dt) {
              context.set_stage_time(0, 1);
              auto& state = context.state(0);
              auto& rhs = context.rhs_scratch(
                  static_cast<pops::runtime::program::ProgramCacheSlot>(0), 0, state);
              context.rhs_into(0, state, rhs, 3000);
              // Exercise commit_many() while the v5 static table is bound.  The target is the
              // detached level candidate owned by advance_hierarchy(), so this must consume only
              // the prebound pointer/snapshot carriers and leave the sealed static metadata
              // untouched.  The injected failure below then proves its copy is rollback-safe.
              context.commit_many({{&state, &rhs}});
              if (throw_after_resident_write->exchange(false, std::memory_order_relaxed))
                throw std::runtime_error("injected static-flux failure after resident basis write");
              context.axpy(state, pops::Real(level_dt), rhs);
            });
      },
      {}, {}, budgets, {}, std::nullopt, bases, {});
  return owner;
}

template <int Dim>
pops::runtime::system::AuxiliaryComponentKey install_field_output(pops::AmrSystem<Dim>& system) {
  using namespace pops::runtime::system;
  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 1;
  AuxiliaryComponentKey key{"test.amr-transaction-authority", "field", "phi", "potential"};
  AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "amr-field", "scalar"};
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "test.amr-transaction-authority/field-output@1",
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      {{key, contract, shape}},
      {}});
  system.seal_auxiliary_providers();
  return key;
}

template <int Dim>
std::unique_ptr<pops::AmrSystem<Dim>> make_field_output_system() {
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  auto owner = std::make_unique<pops::AmrSystem<Dim>>(config);
  auto& system = *owner;
  pops::test::install_amr_runtime_authority(system,
                                            "tests.amr-transaction-authority/field-runtime@1");
  const pops::AmrFieldHierarchyPolicyAuthority hierarchy{
      "pops.field-hierarchy.level-local", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  system.set_field_solver_plan(
      "field/tracer", "test.amr-transaction-authority.field-plan@1",
      "test.amr-transaction-authority.field-provider@1", "test.amr-transaction-authority", "tracer",
      "phi", {{"test.amr-transaction-authority", "field", "phi", "potential"}}, 1, {"test.rhs"},
      {"tracer"}, {"charge"}, {1.0}, "geometric_mg", hierarchy,
      pops::geometric_mg_amr_field_solver_options(pops::GeometricMgOptions{},
                                                  pops::CompositeFacOptions{}));
  system.install_block_state_route("tracer", "tests.amr-transaction-authority/field-state@1");
  pops::RealVector<Dim> velocity{};
  velocity[0] = pops::Real(1);
  pops::add_compiled_model<Dim>(
      system, "tracer",
      ScalarAdvectionModel<Dim>{pops::nd::ScalarAdvection<Dim>::prepare(velocity)}, "minmod",
      "rusanov", "conservative", "explicit", static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1,
      {}, {}, 0.0, static_cast<double>(pops::kWenoEpsilon), false,
      "tests.amr-transaction-authority/field-physical-flux@1");
  const auto output_key = install_field_output(system);
  system.register_elliptic_field("tracer", "phi", {output_key}, 1);
  system.set_block_elliptic_field(
      "tracer", "phi", "test.amr-transaction-authority.field-rhs@1",
      [](const pops::MultiFab<Dim>&, pops::MultiFab<Dim>& rhs) { rhs.set_val(pops::Real(0)); });
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(config.shape[axis]);
  system.set_conservative_state("tracer", std::vector<double>(cells, 1.0));
  system.set_program_block_map({0});
  return owner;
}

template <int Dim>
std::unique_ptr<pops::AmrSystem<Dim>> make_regridding_coupling_system() {
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 2;
  // The refined layout is planned and materialized before the Program is bound.  This keeps
  // topology/bootstrap preparation outside the hot candidate witness while exercising the real
  // prepared AMR dispatch and coupling carriers below.
  config.regrid_every = 0;
  config.explicit_bootstrap = true;
  pops::Extent<Dim> transition_ratio{};
  pops::Extent<Dim> transition_buffer{};
  pops::Extent<Dim> transition_lookahead{};
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 8;
    transition_ratio[axis] = 2;
    transition_buffer[axis] = 0;
    transition_lookahead[axis] = 0;
  }
  config.transition_ratios = {transition_ratio};
  config.transition_buffers = {transition_buffer};
  config.transition_lookaheads = {transition_lookahead};

  auto owner = std::make_unique<pops::AmrSystem<Dim>>(config);
  auto& system = *owner;
  pops::test::install_amr_runtime_authority(
      system, "tests.amr-transaction-authority/regrid-coupling-runtime@1");
  system.install_block_state_route("donor", "tests.amr-transaction-authority/regrid-donor@1");
  system.install_block_state_route("receiver", "tests.amr-transaction-authority/regrid-receiver@1");
  pops::test::install_prepared_threshold_union(
      system,
      {{"donor", "u", 1.5, pops::test::PreparedThresholdRelation::Above,
        "tests.amr-transaction-authority/regrid-donor@1"}},
      "tests.amr-transaction-authority/regrid-coupling-tagger@1",
      "test.amr-transaction-authority.clock");
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  pops::RealVector<Dim> velocity{};
  velocity[0] = pops::Real(1);
  pops::add_compiled_model<Dim>(
      system, "donor", ScalarAdvectionModel<Dim>{pops::nd::ScalarAdvection<Dim>::prepare(velocity)},
      "minmod", "rusanov", "conservative", "explicit",
      static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
      static_cast<double>(pops::kWenoEpsilon), false,
      "tests.amr-transaction-authority/regrid-donor-flux@1");
  pops::add_compiled_model<Dim>(
      system, "receiver",
      ScalarAdvectionModel<Dim>{pops::nd::ScalarAdvection<Dim>::prepare(velocity)}, "minmod",
      "rusanov", "conservative", "explicit", static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1,
      {}, {}, 0.0, static_cast<double>(pops::kWenoEpsilon), false,
      "tests.amr-transaction-authority/regrid-receiver-flux@1");

  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(config.shape[axis]);
  system.set_conservative_state("donor", std::vector<double>(cells, 2.0));
  system.set_conservative_state("receiver", std::vector<double>(cells, 5.0));
  system.set_program_block_map({0, 1});

  pops::CouplingOperatorView view;
  view.label = "regrid-conservative-exchange";
  std::vector<pops::runtime::system::PreparedCouplingConservationGroup> conservation{
      {"regrid-donor-receiver.scalar", {{"donor", 0, 0, "scalar"}, {"receiver", 1, 0, "scalar"}}}};
  system.install_prepared_amr_coupling_operator(
      "tests.amr-transaction-authority/regrid-coupling-exchange@1", std::move(view),
      pops::runtime::system::PreparedCouplingOperator<Dim>(
          [](pops::Real dt, const std::vector<pops::MultiFab<Dim>*>& states) {
            if (states.size() != 2 || states[0] == nullptr || states[1] == nullptr)
              throw std::logic_error("regrid coupling lost its complete Program state pack");
            auto& donor = *states[0];
            auto& receiver = *states[1];
            if (donor.local_size() != receiver.local_size())
              throw std::logic_error("regrid coupling donor/receiver patch layout diverged");
            for (std::size_t local = 0; local < donor.local_size(); ++local) {
              const auto exchange = ConservativeProgramExchange<Dim>{
                  donor.fab(local).view(), receiver.fab(local).view(), dt};
              if constexpr (std::is_same_v<typename pops::MultiFab<Dim>::memory_space,
                                           Kokkos::HostSpace>) {
                for_each_host_cell(donor.box(local), exchange);
              } else {
                pops::for_each_cell(donor.box(local), exchange);
              }
            }
            pops::device_fence();
          },
          std::move(conservation)));

  pops::Index<Dim> fine_lower{};
  pops::Index<Dim> fine_upper{};
  pops::Extent<Dim> fine_shape{};
  for (int axis = 0; axis < Dim; ++axis) {
    fine_lower[axis] = config.shape[axis] / 2;
    fine_upper[axis] = 3 * config.shape[axis] / 2 - 1;
    fine_shape[axis] = 2 * config.shape[axis];
  }
  system.rebuild_hierarchy({pops::AmrPatch<Dim>{1, {fine_lower, fine_upper}}}, {0});
  // Rebuild materializes a detached zeroed successor for every block.  Reapply the exact coarse
  // source state after the accepted hierarchy swap, then seed the matching refined carriers.
  system.set_conservative_state("donor", std::vector<double>(cells, 2.0));
  system.set_conservative_state("receiver", std::vector<double>(cells, 5.0));
  std::size_t fine_cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    fine_cells *= static_cast<std::size_t>(fine_shape[axis]);
  system.set_block_level_state("donor", 1, std::vector<double>(fine_cells, 2.0));
  system.set_block_level_state("receiver", 1, std::vector<double>(fine_cells, 5.0));
  return owner;
}

template <int Dim>
void install_regridding_coupling_program(
    pops::AmrSystem<Dim>& system, const std::shared_ptr<ProgramAmrCouplingWitness>& witness) {
  using Resource = pops::test::program_v5::CallbackProgramResource;
  constexpr std::string_view kProgramIdentity =
      "tests.amr-transaction-authority/regrid-coupling-program@1";
  constexpr std::string_view kRateIdentity =
      "tests.amr-transaction-authority/regrid-coupling-rate@1";
  constexpr std::string_view kApplicationIdentity =
      "tests.amr-transaction-authority/regrid-coupling-application@1";
  using FluxBudget = pops::runtime::program::ProgramFluxBudgetRecord;
  const std::optional<std::vector<FluxBudget>> flux_budgets{std::vector<FluxBudget>{
      {0, 0, 1, kProgramIdentity.size() + kRateIdentity.size() + kApplicationIdentity.size()},
      {0, 0, 1, kProgramIdentity.size() + kRateIdentity.size() + kApplicationIdentity.size()}}};
  const auto names = system.block_names();
  if (names.size() != 2)
    throw std::logic_error("regrid coupling fixture requires donor and receiver blocks");
  system.refresh_prepared_amr_levels();
  std::vector<Resource> resources;
  resources.reserve(4);
  for (int block = 0; block < 2; ++block) {
    const auto state = system.prepared_amr_block_state(block, 0);
    if (!state)
      throw std::logic_error("regrid coupling Program resource has no prepared state");
    for (int scratch = 0; scratch < 2; ++scratch) {
      Resource resource{Resource::Kind::state,
                        static_cast<std::size_t>(block * 2 + scratch),
                        0,
                        block,
                        -1,
                        static_cast<std::uint32_t>(state->ncomp()),
                        static_cast<std::uint32_t>(state->ghosts()[0])};
      resource.value_id = static_cast<std::uint64_t>(100 + block * 2 + scratch);
      resource.identity =
          std::string(kProgramIdentity) + "/resource/" + std::to_string(resource.slot);
      resource.occurrence_path = "root/resource/" + std::to_string(resource.slot);
      resource.owner = names.at(static_cast<std::size_t>(block));
      resource.clock = "test.amr-transaction-authority.clock";
      resources.push_back(std::move(resource));
    }
  }
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, kProgramIdentity, "test.amr-transaction-authority.clock", names, resources, {},
      [witness, kProgramIdentity, kRateIdentity, kApplicationIdentity](auto& context,
                                                                       double macro_dt) {
        ++witness->operation_count;
        context.advance_hierarchy(macro_dt, [witness, &context, kProgramIdentity, kRateIdentity,
                                             kApplicationIdentity](double level_dt) {
          context.set_stage_time(0, 1);
          std::array<pops::MultiFab<Dim>*, 2> accepted{};
          std::array<pops::MultiFab<Dim>*, 2> candidates{};
          for (int block = 0; block < 2; ++block) {
            accepted[static_cast<std::size_t>(block)] = &context.state(block);
            const auto slot = static_cast<pops::runtime::program::ProgramCacheSlot>(block * 2 + 1);
            candidates[static_cast<std::size_t>(block)] =
                &context.scratch_state(slot, 0, *accepted[static_cast<std::size_t>(block)]);
            context.lincomb(*candidates[static_cast<std::size_t>(block)], pops::Real(1),
                            *accepted[static_cast<std::size_t>(block)], pops::Real(0),
                            *accepted[static_cast<std::size_t>(block)]);
          }
          witness->record_level(context.level());
          context.apply_coupling_operators(kProgramIdentity, kRateIdentity, kApplicationIdentity,
                                           pops::Real(level_dt),
                                           {{0, candidates[0]}, {1, candidates[1]}});
          ++witness->coupling_evaluations;
          context.commit_many({{accepted[0], candidates[0]}, {accepted[1], candidates[1]}});
          if (witness->reject_after_coupling.exchange(false, std::memory_order_relaxed))
            throw std::runtime_error("injected refined Program coupling rejection");
        });
      },
      {}, {}, flux_budgets);
}

template <int Dim>
std::unique_ptr<pops::AmrSystem<Dim>> make_regridding_field_output_system() {
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 2;
  // The first macro step is cursor 0 and stages the forward topology.  Cursor 1 performs the
  // post-seal field evaluation without another regrid allocation.
  config.regrid_every = 2;
  config.explicit_bootstrap = true;
  pops::Extent<Dim> transition_ratio{};
  pops::Extent<Dim> transition_buffer{};
  pops::Extent<Dim> transition_lookahead{};
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 8;
    transition_ratio[axis] = 2;
    transition_buffer[axis] = 0;
    transition_lookahead[axis] = 0;
  }
  config.transition_ratios = {transition_ratio};
  config.transition_buffers = {transition_buffer};
  config.transition_lookaheads = {transition_lookahead};
  auto owner = std::make_unique<pops::AmrSystem<Dim>>(config);
  auto& system = *owner;
  pops::test::install_amr_runtime_authority(
      system, "tests.amr-transaction-authority/regrid-field-runtime@1");
  // The initial state stays coarse-only.  The installed Program raises it above this exact
  // threshold before the first regrid, which yields a genuine forward topology image.
  pops::test::install_prepared_threshold_union(system, {{"tracer", "u", 1.5}},
                                               "tests.amr-transaction-authority/regrid-tagger@1",
                                               "test.amr-transaction-authority.clock");
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  const pops::AmrFieldHierarchyPolicyAuthority hierarchy{
      "pops.field-hierarchy.level-local", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  system.set_field_solver_plan(
      "field/tracer", "test.amr-transaction-authority.regrid-field-plan@1",
      "test.amr-transaction-authority.regrid-field-provider@1", "test.amr-transaction-authority",
      "tracer", "phi", {{"test.amr-transaction-authority", "field", "phi", "potential"}}, 1,
      {"test.rhs"}, {"tracer"}, {"charge"}, {1.0}, "geometric_mg", hierarchy,
      pops::geometric_mg_amr_field_solver_options(pops::GeometricMgOptions{},
                                                  pops::CompositeFacOptions{}));
  system.set_field_topology_authority("field/tracer", "builtin_rectangular_cell_graph_v1",
                                      "tests.amr-transaction-authority/regrid-field-transfer@1",
                                      "tests.amr-transaction-authority/regrid-field-topology@1");
  system.install_block_state_route("tracer",
                                   "tests.amr-transaction-authority/regrid-field-state@1");
  pops::RealVector<Dim> velocity{};
  velocity[0] = pops::Real(1);
  pops::add_compiled_model<Dim>(
      system, "tracer",
      ScalarAdvectionModel<Dim>{pops::nd::ScalarAdvection<Dim>::prepare(velocity)}, "minmod",
      "rusanov", "conservative", "explicit", static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1,
      {}, {}, 0.0, static_cast<double>(pops::kWenoEpsilon), false,
      "tests.amr-transaction-authority/regrid-field-physical-flux@1");
  const auto output_key = install_field_output(system);
  system.register_elliptic_field("tracer", "phi", {output_key}, 1);
  system.set_block_elliptic_field(
      "tracer", "phi", "test.amr-transaction-authority.regrid-field-rhs@1",
      [](const pops::MultiFab<Dim>&, pops::MultiFab<Dim>& rhs) { rhs.set_val(pops::Real(0)); });
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(config.shape[axis]);
  system.set_conservative_state("tracer", std::vector<double>(cells, 1.0));
  system.set_program_block_map({0});
  return owner;
}

template <int Dim>
pops::runtime::multiblock::BoundaryEvaluationPoint field_evaluation_point() {
  return {.clock = "test.amr-transaction-authority.clock",
          .tick = 0,
          .level = 0,
          .substep = 0,
          .stage = 1,
          .stage_fraction = {0, 1},
          .dt = 0.01,
          .physical_time = 0.0};
}

template <int Dim>
void install_throwing_field_output_program(pops::AmrSystem<Dim>& system,
                                           const std::shared_ptr<int>& effects) {
  using Resource = pops::test::program_v5::CallbackProgramResource;
  (void)system.n_levels();
  std::uint32_t state_components = 0;
  std::uint32_t state_ghosts = 0;
  {
    const auto state = system.prepared_amr_block_state(0, 0);
    if (!state)
      throw std::logic_error("AMR field-output Program fixture lost its prepared state");
    state_components = static_cast<std::uint32_t>(state->ncomp());
    state_ghosts = static_cast<std::uint32_t>(state->ghosts()[0]);
  }
  const std::vector<Resource> resources{{
      Resource::Kind::state,
      0,
      0,
      0,
      0,
      state_components,
      state_ghosts,
  }};
  const std::vector<pops::test::program_v5::CallbackProgramFieldRoute> field_routes{
      {0, "field/tracer", {0}}};
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.amr-transaction-authority/field-output-program@1",
      "test.amr-transaction-authority.clock", resources, field_routes,
      [effects](auto& context, double macro_dt) {
        context.begin_step(macro_dt);
        auto& stage = context.scratch_state(0, 0, context.state(0));
        stage.set_val(pops::Real(1));
        pops::SolveOutcome outcome = context.solve_fields_from_state_at(
            field_evaluation_point<Dim>(), "field/tracer", 0, stage);
        const pops::SolveReport accepted = pops::consume_solve_outcome(std::move(outcome));
        if (!accepted.solved())
          throw std::runtime_error("AMR field-output Program fixture solve was not accepted");
        ++*effects;
        throw std::runtime_error("injected post-solve Program effect failure");
      });
  // Installation prepares a detached Program authority; bind is the explicit cold boundary which
  // materializes its field plans and primes the resident accepted transaction image.
  system.mark_bound();
}

template <int Dim>
void install_field_candidate_savepoint_program(pops::AmrSystem<Dim>& system,
                                               const std::shared_ptr<int>& reentrancy_rejections) {
  using Resource = pops::test::program_v5::CallbackProgramResource;
  (void)system.n_levels();
  std::uint32_t state_components = 0;
  std::uint32_t state_ghosts = 0;
  {
    const auto state = system.prepared_amr_block_state(0, 0);
    if (!state)
      throw std::logic_error("AMR field-candidate Program fixture lost its prepared state");
    state_components = static_cast<std::uint32_t>(state->ncomp());
    state_ghosts = static_cast<std::uint32_t>(state->ghosts()[0]);
  }
  const std::vector<Resource> resources{{
      Resource::Kind::state,
      0,
      0,
      0,
      0,
      state_components,
      state_ghosts,
  }};
  const std::vector<pops::test::program_v5::CallbackProgramFieldRoute> field_routes{
      {0, "field/tracer", {0}}};
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.amr-transaction-authority/field-candidate-savepoint@1",
      "test.amr-transaction-authority.clock", resources, field_routes,
      [reentrancy_rejections](auto& context, double macro_dt) {
        context.begin_step(macro_dt);
        auto& accepted = context.state(0);
        auto& perturbed = context.scratch_state(0, 0, accepted);
        perturbed.set_val(pops::Real(2.75));
        context.evaluate_with_field_state_at(
            field_evaluation_point<Dim>(), "field/tracer", 0, perturbed, accepted,
            [&context, &perturbed, reentrancy_rejections] {
              // This is deliberately a non-trivial scientific mutation.  The nested savepoint
              // must restore it before the outer Program callback resumes.
              context.state(0).set_val(pops::Real(7.25));
              try {
                context.evaluate_with_field_state_at(
                    field_evaluation_point<Dim>(), "field/tracer", 0, perturbed, context.state(0),
                    [&context] { context.state(0).set_val(pops::Real(11.5)); });
              } catch (const std::logic_error&) {
                ++*reentrancy_rejections;
              }
            });
      });
  system.mark_bound();
}

template <int Dim>
void install_regridding_field_candidate_savepoint_program(
    pops::AmrSystem<Dim>& system, const std::shared_ptr<ProgramAmrCouplingWitness>& witness = {}) {
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const auto coupling_witness = witness ? witness : std::make_shared<ProgramAmrCouplingWitness>();
  (void)system.n_levels();
  std::uint32_t state_components = 0;
  std::uint32_t state_ghosts = 0;
  {
    const auto state = system.prepared_amr_block_state(0, 0);
    if (!state)
      throw std::logic_error("AMR regrid field-candidate fixture lost its prepared state");
    state_components = static_cast<std::uint32_t>(state->ncomp());
    state_ghosts = static_cast<std::uint32_t>(state->ghosts()[0]);
  }
  const std::vector<Resource> resources{
      {Resource::Kind::state, 0, 0, 0, 0, state_components, state_ghosts}};
  const std::vector<pops::test::program_v5::CallbackProgramFieldRoute> field_routes{
      {0, "field/tracer", {0}}};
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.amr-transaction-authority/regrid-field-candidate@1",
      "test.amr-transaction-authority.clock", resources, field_routes,
      [coupling_witness](auto& context, double macro_dt) {
        if (coupling_witness->exercise_hierarchy_coupling) {
          // The direct hierarchy driver owns begin_step() for this branch.  Calling it before
          // advance_hierarchy() would ask the accepted facade to refresh while the outer attempt is
          // active; the cold callbacks before the refined topology are intentionally measured only
          // after this operation has returned through the outer Program dispatcher.
          if (context.macro_step() == 0) {
            context.begin_step(macro_dt);
            context.state(0).set_val(pops::Real(2.75));
            return;
          }
          ++coupling_witness->operation_count;
          context.advance_hierarchy(macro_dt, [&context, coupling_witness](double level_dt) {
            (void)level_dt;
            coupling_witness->record_level(context.level());
            context.set_stage_time(0, 1);
            auto& accepted = context.state(0);
            auto& perturbed = context.scratch_state(0, 0, accepted);
            perturbed.set_val(pops::Real(5.25));
            const auto& point = context.prepared_boundary_evaluation_point(1);
            context.evaluate_with_field_state_at(point, "field/tracer", 0, perturbed, accepted,
                                                 [] {});
            ++coupling_witness->coupling_evaluations;
            if (coupling_witness->reject_after_coupling.exchange(false, std::memory_order_relaxed))
              throw std::runtime_error("injected AMR Program coupling rejection");
          });
          return;
        }
        context.begin_step(macro_dt);
        // The state mutation produces the first accepted fine level.  The nested field candidate
        // is deliberately deferred to the next macro-step, when it must use the forward-topology
        // savepoint adopted by the preceding seal.
        context.state(0).set_val(pops::Real(2.75));
        if (context.macro_step() == 0)
          return;
        auto& perturbed = context.scratch_state(0, 0, context.state(0));
        perturbed.set_val(pops::Real(5.25));
        context.evaluate_with_field_state_at(field_evaluation_point<Dim>(), "field/tracer", 0,
                                             perturbed, context.state(0), [] {});
      });
  system.mark_bound();
}

template <class Result>
struct ForeignReaderObservation final {
  std::future_status status_before_release = std::future_status::deferred;
  std::future_status status_after_commit = std::future_status::deferred;
  std::future_status status_after_release = std::future_status::deferred;
  std::optional<Result> value;
  std::exception_ptr failure;
};

struct StructuralReaderSnapshot final {
  int max_levels = 0;
  int n_vars = 0;
  std::vector<std::string> conservative_variable_names;
  int block_n_vars = 0;

  friend bool operator==(const StructuralReaderSnapshot&,
                         const StructuralReaderSnapshot&) = default;
};

template <int Dim>
StructuralReaderSnapshot structural_readers(pops::AmrSystem<Dim>& system) {
  return {system.max_levels(), system.n_vars(), system.variable_names("tracer"),
          system.block_n_vars("tracer")};
}

template <int Dim, class Result, class Reader>
ForeignReaderObservation<Result> rollback_after_foreign_reader(pops::AmrSystem<Dim>& system,
                                                               Reader reader) {
  ForeignReaderObservation<Result> observation;
  std::promise<void> reader_finished;
  auto reader_done = reader_finished.get_future();
  std::thread foreign_reader([&] {
    try {
      observation.value.emplace(reader());
    } catch (...) {
      observation.failure = std::current_exception();
    }
    reader_finished.set_value();
  });

  observation.status_before_release = reader_done.wait_for(std::chrono::milliseconds(20));
  system.rollback_step_transaction();
  observation.status_after_release = reader_done.wait_for(std::chrono::seconds(1));
  foreign_reader.join();
  return observation;
}

template <int Dim, class Result, class Reader>
ForeignReaderObservation<Result> seal_after_foreign_reader(pops::AmrSystem<Dim>& system,
                                                           Reader reader) {
  ForeignReaderObservation<Result> observation;
  std::promise<void> reader_finished;
  auto reader_done = reader_finished.get_future();
  std::thread foreign_reader([&] {
    try {
      observation.value.emplace(reader());
    } catch (...) {
      observation.failure = std::current_exception();
    }
    reader_finished.set_value();
  });

  observation.status_before_release = reader_done.wait_for(std::chrono::milliseconds(20));
  system.commit_step_transaction();
  observation.status_after_commit = reader_done.wait_for(std::chrono::milliseconds(20));
  system.finalize_step_transaction();
  observation.status_after_release = reader_done.wait_for(std::chrono::seconds(1));
  foreign_reader.join();
  return observation;
}

}  // namespace

TEST(AmrTransactionAuthority,
     FieldOutputDirtyIdentityRollsBackAfterAcceptedSolveWithoutHotAllocation) {
  constexpr int Dim = pops::kNativeDimension;
  auto system_owner = make_field_output_system<Dim>();
  auto& system = *system_owner;
  auto effects = std::make_shared<int>(0);
  install_throwing_field_output_program(system, effects);
  const auto accepted_dirty = system.dirty_auxiliary_provider_identities();
  ASSERT_TRUE(accepted_dirty.empty());

  const auto run_rejected_attempt = [&] {
    ASSERT_NO_THROW(system.begin_step_transaction());
    EXPECT_THROW(system.step(0.01), std::runtime_error);
    {
      auto provisional = system._provisional_read_scope();
      ASSERT_TRUE(provisional.valid());
      EXPECT_EQ(system.dirty_auxiliary_provider_identities(),
                (std::vector<std::string>{"test.amr-transaction-authority/field-output@1"}));
    }
    ASSERT_NO_THROW(system.rollback_step_transaction());
  };

  // The public ABI-v5 Program accepts the field SolveOutcome, which appends the field-output
  // identity, then throws from the following Program operation. Rollback must restore the exact
  // pre-solve dirty set rather than reject the changed vector shape.
  run_rejected_attempt();
  EXPECT_EQ(*effects, 1);
  EXPECT_EQ(system.dirty_auxiliary_provider_identities(), accepted_dirty)
      << "post-solve rejection must restore the captured dirty-provider cardinality and contents";

  // The retry follows the same accepted-solve/Program-effect/rollback path. Its allocator baseline
  // is intentionally taken after the cold DSO installation and warmup, so it witnesses only the
  // hot dirty-identity append and rollback carriers.
  const pops::AllocationEventStats allocation_before_retry = pops::allocation_event_stats();
  run_rejected_attempt();
  EXPECT_EQ(*effects, 2);
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_retry)
      << "accepted field-output dirty append and rollback must reuse sealed slots";
  EXPECT_EQ(system.dirty_auxiliary_provider_identities(), accepted_dirty)
      << "retry must restore the same accepted dirty-provider image bit-exactly";
}

TEST(AmrTransactionAuthority, FieldCandidateSavepointRestoresAcceptedImagesWithoutHotAllocation) {
  constexpr int Dim = pops::kNativeDimension;
  auto system_owner = make_field_output_system<Dim>();
  auto& system = *system_owner;
  auto reentrancy_rejections = std::make_shared<int>(0);
  install_field_candidate_savepoint_program(system, reentrancy_rejections);

  const pops::runtime::system::AuxiliaryComponentKey output_key{"test.amr-transaction-authority",
                                                                "field", "phi", "potential"};
  const auto accepted_state = system.block_level_state_global("tracer", 0);
  const auto accepted_output = system.auxiliary_component(output_key);
  const auto accepted_dirty = system.dirty_auxiliary_provider_identities();

  const auto require_nested_restore = [&] {
    auto provisional = system._provisional_read_scope();
    ASSERT_TRUE(provisional.valid());
    EXPECT_EQ(system.block_level_state_global("tracer", 0), accepted_state);
    EXPECT_EQ(system.auxiliary_component(output_key), accepted_output);
    EXPECT_EQ(system.dirty_auxiliary_provider_identities(), accepted_dirty);
  };

  // First attempt is the permitted cold warmup of the ABI-v5 callback and its resident savepoint.
  ASSERT_NO_THROW(system.begin_step_transaction());
  ASSERT_NO_THROW(system.step(0.01));
  require_nested_restore();
  ASSERT_NO_THROW(system.rollback_step_transaction());
  EXPECT_EQ(*reentrancy_rejections, 1)
      << "a nested field candidate must be refused before it can mutate the outer candidate";
  EXPECT_EQ(system.block_level_state_global("tracer", 0), accepted_state);
  EXPECT_EQ(system.auxiliary_component(output_key), accepted_output);
  EXPECT_EQ(system.dirty_auxiliary_provider_identities(), accepted_dirty);

  const pops::AllocationEventStats allocation_before = pops::allocation_event_stats();
  ASSERT_NO_THROW(system.begin_step_transaction());
  ASSERT_NO_THROW(system.step(0.01));
  require_nested_restore();
  ASSERT_NO_THROW(system.rollback_step_transaction());
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before)
      << "same-topology field candidate capture and restore must reuse the cold savepoint image";
  EXPECT_EQ(*reentrancy_rejections, 2);
  EXPECT_EQ(system.block_level_state_global("tracer", 0), accepted_state);
  EXPECT_EQ(system.auxiliary_component(output_key), accepted_output);
  EXPECT_EQ(system.dirty_auxiliary_provider_identities(), accepted_dirty);

  // A direct AmrSystem::step owns a local ProgramTransaction rather than the public external
  // envelope.  The same Candidate-phase authorization must admit this callback and seal exactly
  // once without allocating a second savepoint image.
  const pops::AllocationEventStats allocation_before_direct_step = pops::allocation_event_stats();
  ASSERT_NO_THROW(system.step(0.01));
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_direct_step);
  EXPECT_EQ(pops::allocation_event_stats().fab_calls, allocation_before_direct_step.fab_calls)
      << "the warmed direct step must consume the prepared AMR hierarchy witness";
  EXPECT_EQ(pops::allocation_event_stats().communication_calls,
            allocation_before_direct_step.communication_calls)
      << "the warmed direct step must not rebuild a collective point or batch contract";
  EXPECT_EQ(*reentrancy_rejections, 3);
  EXPECT_EQ(system.block_level_state_global("tracer", 0), accepted_state);
  EXPECT_EQ(system.auxiliary_component(output_key), accepted_output);
  EXPECT_EQ(system.dirty_auxiliary_provider_identities(), accepted_dirty);
}

TEST(AmrTransactionAuthority,
     AllCancelStaticFluxBasisIsHotAllocationFreeAndRetrySafeAfterResidentWriteFailure) {
  constexpr int Dim = pops::kNativeDimension;
  const auto throw_after_resident_write = std::make_shared<std::atomic<bool>>(false);
  auto system_owner = make_all_cancel_static_flux_system<Dim>(throw_after_resident_write);
  auto& system = *system_owner;
  constexpr double dt = 0.01;

  const auto run_success = [&](const char* witness) {
    const pops::AllocationEventStats events_before = pops::allocation_event_stats();
    const auto measure = [](auto&& operation) {
      HeapAllocationWindow heap_window;
      std::forward<decltype(operation)>(operation)();
      return heap_window.close();
    };
    const std::uint64_t begin_allocations = measure([&] { system.begin_step_transaction(); });
    const std::uint64_t step_allocations = measure([&] { system.step(dt); });
    const std::uint64_t commit_allocations = measure([&] { system.commit_step_transaction(); });
    const std::uint64_t finalize_allocations = measure([&] { system.finalize_step_transaction(); });
    EXPECT_EQ(begin_allocations, 0u) << witness << ": begin";
    EXPECT_EQ(step_allocations, 0u) << witness << ": step";
    EXPECT_EQ(commit_allocations, 0u) << witness << ": hidden publish";
    EXPECT_EQ(finalize_allocations, 0u) << witness << ": seal/finalize";
    EXPECT_EQ(pops::allocation_event_stats(), events_before) << witness;
  };

  // Cold install/bind/prime has completed in the fixture.  The first and repeated static RHS
  // attachments therefore execute after the exact helper preflight but before no dynamic route
  // construction, map insertion, or ledger-slot growth is permitted.
  run_success("first all-cancel static-flux execution must remain hot-allocation-free");
  run_success("repeated all-cancel static-flux execution must remain hot-allocation-free");

  std::optional<pops::MultiFab<Dim>> accepted_before_failure;
  {
    const auto accepted = system.prepared_amr_block_state(0, 0);
    ASSERT_TRUE(accepted);
    accepted_before_failure.emplace(*accepted);
  }
  throw_after_resident_write->store(true, std::memory_order_relaxed);
  ASSERT_NO_THROW(system.begin_step_transaction());
  // Exception construction is deliberately outside the allocation witness.  The callback throws
  // only after rhs_into() has attached the static basis and updated its resident scalar payload.
  EXPECT_THROW(system.step(dt), std::runtime_error);
  ASSERT_NO_THROW(system.rollback_step_transaction());
  {
    const auto restored = system.prepared_amr_block_state(0, 0);
    ASSERT_TRUE(restored);
    ASSERT_TRUE(accepted_before_failure.has_value());
    EXPECT_EQ(pops::difference_sum_sq_all(*restored, *accepted_before_failure), pops::Real(0));
  }

  run_success("static-flux retry after rollback must reuse the resident carrier and ledger image");
}

TEST(AmrTransactionAuthority,
     LongNamedHistoryStoreAndRotateReuseBindPrimedSlotsAcrossRetryAndRepeat) {
  constexpr int Dim = pops::kNativeDimension;
  constexpr double dt = 0.01;
  constexpr std::string_view kHistoryName =
      "tests.amr-transaction-authority.history-name-beyond-the-small-string-optimization";
  auto system_owner = make_system<Dim>();
  auto& system = *system_owner;
  const auto reject_after_history = std::make_shared<std::atomic<bool>>(false);
  const auto history_name = std::make_shared<const std::string>(kHistoryName);
  (void)system.n_levels();
  std::uint32_t state_components = 0;
  std::uint32_t state_ghosts = 0;
  {
    const auto state = system.prepared_amr_block_state(0, 0);
    ASSERT_TRUE(state);
    state_components = static_cast<std::uint32_t>(state->ncomp());
    state_ghosts = static_cast<std::uint32_t>(state->ghosts()[0]);
  }
  using Resource = pops::test::program_v5::CallbackProgramResource;
  const std::vector<Resource> resources{
      {Resource::Kind::rhs, 0, 0, 0, 0, state_components, state_ghosts}};
  const std::vector<pops::test::program_v5::CallbackProgramHistory> histories{{
      *history_name,
      1,
      static_cast<int>(state_components),
      0,
      "tests.amr-transaction-authority.history-state-identity-beyond-sso",
      "tests.amr-transaction-authority.history-space-identity-beyond-sso",
      "test.clock.macro",
      "tests.amr-transaction-authority.history-interpolation-identity-beyond-sso",
  }};
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.amr-transaction-authority/history-hot-slots@1", "test.clock.macro", resources,
      {},
      [reject_after_history, history_name](auto& context, double macro_dt) {
        context.advance_hierarchy(macro_dt, [&context, reject_after_history,
                                             history_name](double level_dt) {
          context.set_stage_time(0, 1);
          auto& state = context.state(0);
          auto& sample = context.rhs_scratch(
              static_cast<pops::runtime::program::ProgramCacheSlot>(context.level()), 0, state);
          sample.set_val(static_cast<pops::Real>(level_dt));
          context.store_history(*history_name, sample, 0);
          context.rotate_histories("test.clock.macro");
          if (reject_after_history->exchange(false, std::memory_order_relaxed))
            throw std::runtime_error("injected history-store rejection after resident rotation");
        });
      },
      histories);
  system.mark_bound();

  const auto run_success = [&](const char* witness) {
    const pops::AllocationEventStats events_before = pops::allocation_event_stats();
    const auto measure = [](auto&& operation) {
      HeapAllocationWindow heap_window;
      std::forward<decltype(operation)>(operation)();
      return heap_window.close();
    };
    EXPECT_EQ(measure([&] { system.begin_step_transaction(); }), 0u) << witness << ": begin";
    EXPECT_EQ(measure([&] { system.step(dt); }), 0u) << witness << ": store/rotate";
    EXPECT_EQ(measure([&] { system.commit_step_transaction(); }), 0u)
        << witness << ": hidden publish";
    EXPECT_EQ(measure([&] { system.finalize_step_transaction(); }), 0u) << witness << ": finalize";
    EXPECT_EQ(pops::allocation_event_stats(), events_before) << witness;
  };

  run_success("first long-name history store/rotate");
  reject_after_history->store(true, std::memory_order_relaxed);
  ASSERT_NO_THROW(system.begin_step_transaction());
  EXPECT_THROW(system.step(dt), std::runtime_error);
  ASSERT_NO_THROW(system.rollback_step_transaction());
  run_success("retry after rejected long-name history store/rotate");
  run_success("repeated long-name history store/rotate");
}

TEST(AmrTransactionAuthority,
     StaticFluxCommitOutsideAdvanceHierarchyRefusesBeforeSnapshotOrMutation) {
  constexpr int Dim = pops::kNativeDimension;
  const auto throw_after_resident_write = std::make_shared<std::atomic<bool>>(false);
  auto system_owner = make_all_cancel_static_flux_system<Dim>(throw_after_resident_write, true);
  auto& system = *system_owner;

  std::optional<pops::MultiFab<Dim>> accepted_before;
  {
    const auto accepted = system.prepared_amr_block_state(0, 0);
    ASSERT_TRUE(accepted);
    accepted_before.emplace(*accepted);
  }
  const auto accepted_manifest = system.program_accepted_state_manifest();
  const auto accepted_clock_manifest = system.program_clock_manifest();
  const auto accepted_epoch = system.checkpoint_topology_epoch();
  const auto accepted_generation = system.accepted_transaction_generation_();
  const auto events_before = pops::allocation_event_stats();

  ASSERT_NO_THROW(system.begin_step_transaction());
  EXPECT_THROW(system.step(0.01), std::runtime_error);
  EXPECT_EQ(pops::allocation_event_stats(), events_before)
      << "the static-v5 boundary must refuse before a commit snapshot can materialize";
  ASSERT_NO_THROW(system.rollback_step_transaction());

  EXPECT_EQ(system.program_accepted_state_manifest(), accepted_manifest);
  EXPECT_EQ(system.program_clock_manifest(), accepted_clock_manifest);
  EXPECT_EQ(system.checkpoint_topology_epoch(), accepted_epoch);
  EXPECT_EQ(system.accepted_transaction_generation_(), accepted_generation);
  const auto restored = system.prepared_amr_block_state(0, 0);
  ASSERT_TRUE(restored);
  ASSERT_TRUE(accepted_before.has_value());
  EXPECT_EQ(pops::difference_sum_sq_all(*restored, *accepted_before), pops::Real(0));
}

TEST(AmrTransactionAuthority,
     FieldCandidateSavepointAdoptsForwardTopologyWithoutPostSealAllocation) {
  constexpr int Dim = pops::kNativeDimension;
  auto system_owner = make_regridding_field_output_system<Dim>();
  auto& system = *system_owner;
  install_regridding_field_candidate_savepoint_program(system);

  ASSERT_EQ(system.n_levels(), 1);
  const std::uint64_t accepted_epoch_before = system.checkpoint_topology_epoch();

  // With a two-step regrid cadence, the first direct step warms the coarse field candidate while
  // retaining the accepted topology.  The second cold Candidate then stages the complete new
  // hierarchy plus both accepted/savepoint images.  Its seal is swap-only; the following direct
  // step proves that evaluation captures the new epoch rather than reusing the coarse image.
  ASSERT_NO_THROW(system.step(0.01));
  ASSERT_EQ(system.n_levels(), 1);
  EXPECT_EQ(system.checkpoint_topology_epoch(), accepted_epoch_before);

  ASSERT_NO_THROW(system.step(0.01));
  ASSERT_EQ(system.n_levels(), 2);
  const std::uint64_t accepted_epoch_after_regrid = system.checkpoint_topology_epoch();
  EXPECT_GT(accepted_epoch_after_regrid, accepted_epoch_before);

  const pops::AllocationEventStats allocation_before_post_regrid_evaluate =
      pops::allocation_event_stats();
  ASSERT_NO_THROW(system.step(0.01));
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_post_regrid_evaluate)
      << "the post-regrid nested field candidate must use the Candidate-primed forward savepoint";
  EXPECT_EQ(system.checkpoint_topology_epoch(), accepted_epoch_after_regrid);
  EXPECT_EQ(system.n_levels(), 2);

  // The prior direct step released the displaced topology bundle.  A new candidate must therefore
  // consume only ordinals rebound to the published hierarchy; rejecting it must not clobber the
  // accepted image rebuilt by the regrid publication.
  const auto accepted_manifest = system.program_accepted_state_manifest();
  const auto accepted_clock_manifest = system.program_clock_manifest();
  const auto accepted_state = system.block_level_state_global("tracer", 0);
  const auto accepted_generation = system.accepted_transaction_generation_();
  ASSERT_NO_THROW(system.begin_step_transaction());
  ASSERT_NO_THROW(system.step(0.01));
  ASSERT_NO_THROW(system.rollback_step_transaction());
  EXPECT_EQ(system.program_accepted_state_manifest(), accepted_manifest);
  EXPECT_EQ(system.program_clock_manifest(), accepted_clock_manifest);
  EXPECT_EQ(system.block_level_state_global("tracer", 0), accepted_state);
  EXPECT_EQ(system.checkpoint_topology_epoch(), accepted_epoch_after_regrid);
  EXPECT_EQ(system.accepted_transaction_generation_(), accepted_generation);
}

TEST(AmrTransactionAuthority,
     RefinedProgramCouplingIsHotAllocationFreeAndRollbackExactWithBoundedDispatch) {
  constexpr int Dim = pops::kNativeDimension;
  constexpr double dt = 0.01;
  auto witness = std::make_shared<ProgramAmrCouplingWitness>();
  auto system_owner = make_regridding_coupling_system<Dim>();
  auto& system = *system_owner;

  // DSO installation/bind and the planned refined hierarchy are deliberately cold.  Establish the
  // allocator baseline only after the accepted forward topology has two materialized levels.
  install_regridding_coupling_program(system, witness);
  ASSERT_EQ(system.n_levels(), 2);

  // Warm the external candidate envelope once after the forward topology is accepted.  The
  // envelope owns the first transaction-side std::vector slots on this path; it is deliberately
  // outside the hot witness, just like DSO installation and topology staging above.
  ASSERT_NO_THROW(system.begin_step_transaction());
  ASSERT_NO_THROW(system.step(dt));
  ASSERT_NO_THROW(system.rollback_step_transaction());
  witness->operation_count = 0;
  witness->level_dispatches = 0;
  witness->coupling_evaluations = 0;

  const auto capture_states = [&] {
    std::array<std::vector<pops::MultiFab<Dim>>, 2> states;
    for (int block = 0; block < 2; ++block) {
      auto& block_states = states[static_cast<std::size_t>(block)];
      block_states.reserve(static_cast<std::size_t>(system.n_levels()));
      for (int level = 0; level < system.n_levels(); ++level) {
        const auto view = system.prepared_amr_block_state(block, level);
        if (!view)
          throw std::logic_error("regrid coupling witness lost its accepted block state");
        block_states.emplace_back(*view);
      }
    }
    return states;
  };
  const auto accepted_states = capture_states();
  ASSERT_EQ(accepted_states[0].size(), 2u);
  ASSERT_EQ(accepted_states[1].size(), 2u);
  const std::uint64_t planned_levels = static_cast<std::uint64_t>(accepted_states[0].size());
  const auto level_totals = [&] {
    std::array<std::array<pops::Real, 2>, 2> totals{};
    for (int block = 0; block < 2; ++block)
      for (int level = 0; level < system.n_levels(); ++level) {
        const auto view = system.prepared_amr_block_state(block, level);
        if (!view)
          throw std::logic_error("regrid coupling witness lost its level total state");
        totals[static_cast<std::size_t>(block)][static_cast<std::size_t>(level)] =
            pops::reduce_sum(*view);
      }
    return totals;
  };
  const auto accepted_level_totals = level_totals();
  const auto accepted_manifest = system.program_accepted_state_manifest();
  const auto accepted_clock_manifest = system.program_clock_manifest();
  const auto accepted_epoch = system.checkpoint_topology_epoch();
  const auto accepted_generation = system.accepted_transaction_generation_();
  const pops::AllocationEventStats hot_events = pops::allocation_event_stats();

  const auto measure = [](const auto& operation) {
    HeapAllocationWindow window;
    operation();
    return window.close();
  };
  const auto assert_dispatch_bound = [&](std::uint64_t prior_operations,
                                         std::uint64_t prior_dispatches,
                                         std::uint64_t prior_couplings, const char* label) {
    const std::uint64_t operations = witness->operation_count - prior_operations;
    const std::uint64_t dispatches = witness->level_dispatches - prior_dispatches;
    const std::uint64_t couplings = witness->coupling_evaluations - prior_couplings;
    const std::uint64_t levels = static_cast<std::uint64_t>(system.n_levels());
    EXPECT_EQ(operations, 1u) << label;
    EXPECT_GT(dispatches, 0u) << label;
    EXPECT_EQ(couplings, dispatches) << label << ": one coupling per level dispatch";
    EXPECT_LE(dispatches, operations * levels * 2u)
        << label << ": Program dispatch must scale with operations*levels, never cells";
    bool saw_fine_level = false;
    for (std::uint64_t index = prior_dispatches;
         index < prior_dispatches + dispatches && index < witness->dispatched_levels.size();
         ++index) {
      EXPECT_GE(witness->dispatched_levels[static_cast<std::size_t>(index)], 0);
      EXPECT_LT(witness->dispatched_levels[static_cast<std::size_t>(index)], system.n_levels());
      saw_fine_level =
          saw_fine_level || witness->dispatched_levels[static_cast<std::size_t>(index)] == 1;
    }
    EXPECT_TRUE(saw_fine_level) << label << ": refined planned level was dispatched";
  };

  const auto run_successful_candidate = [&](const char* label) {
    const std::uint64_t prior_operations = witness->operation_count;
    const std::uint64_t prior_dispatches = witness->level_dispatches;
    const std::uint64_t prior_couplings = witness->coupling_evaluations;
    EXPECT_EQ(measure([&] { system.begin_step_transaction(); }), 0u) << label << ": begin";
    EXPECT_EQ(measure([&] { system.step(dt); }), 0u) << label << ": candidate";
    EXPECT_EQ(measure([&] { system.rollback_step_transaction(); }), 0u) << label << ": rollback";
    EXPECT_EQ(pops::allocation_event_stats(), hot_events) << label;
    assert_dispatch_bound(prior_operations, prior_dispatches, prior_couplings, label);
  };

  run_successful_candidate("first refined Program coupling candidate");
  run_successful_candidate("repeated refined Program coupling candidate");

  witness->reject_after_coupling.store(true, std::memory_order_relaxed);
  const std::uint64_t rejected_prior_operations = witness->operation_count;
  const std::uint64_t rejected_prior_dispatches = witness->level_dispatches;
  const std::uint64_t rejected_prior_couplings = witness->coupling_evaluations;
  ASSERT_EQ(measure([&] { system.begin_step_transaction(); }), 0u);
  bool rejected = false;
  // The native-loader boundary necessarily allocates its foreign-exception diagnostic while
  // converting the injected rejection.  Keep that diagnostic construction outside the standard
  // heap witness; the resident PoPS allocator and exact rollback image remain measured below.
  try {
    system.step(dt);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  EXPECT_TRUE(rejected);
  EXPECT_EQ(pops::allocation_event_stats(), hot_events);
  EXPECT_EQ(witness->operation_count - rejected_prior_operations, 1u);
  EXPECT_GT(witness->level_dispatches - rejected_prior_dispatches, 0u);
  EXPECT_EQ(witness->coupling_evaluations - rejected_prior_couplings,
            witness->level_dispatches - rejected_prior_dispatches);
  EXPECT_LE(witness->level_dispatches - rejected_prior_dispatches,
            (witness->operation_count - rejected_prior_operations) * planned_levels * 2u);
  EXPECT_EQ(measure([&] { system.rollback_step_transaction(); }), 0u);
  EXPECT_EQ(pops::allocation_event_stats(), hot_events);

  EXPECT_EQ(system.program_accepted_state_manifest(), accepted_manifest);
  EXPECT_EQ(system.program_clock_manifest(), accepted_clock_manifest);
  EXPECT_EQ(system.checkpoint_topology_epoch(), accepted_epoch);
  EXPECT_EQ(system.accepted_transaction_generation_(), accepted_generation);
  const auto restored_states = capture_states();
  for (int block = 0; block < 2; ++block)
    for (std::size_t level = 0; level < accepted_states[static_cast<std::size_t>(block)].size();
         ++level)
      EXPECT_EQ(
          pops::difference_sum_sq_all(restored_states[static_cast<std::size_t>(block)][level],
                                      accepted_states[static_cast<std::size_t>(block)][level]),
          pops::Real(0));
  EXPECT_EQ(level_totals(), accepted_level_totals);

  // The retry reuses the same prepared level/coupling carriers and is the only attempt below that
  // advances the accepted generation.  Its candidate and publication path remain allocation-free.
  witness->reject_after_coupling.store(false, std::memory_order_relaxed);
  const std::uint64_t retry_prior_operations = witness->operation_count;
  const std::uint64_t retry_prior_dispatches = witness->level_dispatches;
  const std::uint64_t retry_prior_couplings = witness->coupling_evaluations;
  EXPECT_EQ(measure([&] { system.begin_step_transaction(); }), 0u) << "retry: begin";
  EXPECT_EQ(measure([&] { system.step(dt); }), 0u) << "retry: candidate";
  EXPECT_EQ(measure([&] { system.commit_step_transaction(); }), 0u) << "retry: publish";
  EXPECT_EQ(measure([&] { system.finalize_step_transaction(); }), 0u) << "retry: finalize";
  EXPECT_EQ(pops::allocation_event_stats(), hot_events);
  assert_dispatch_bound(retry_prior_operations, retry_prior_dispatches, retry_prior_couplings,
                        "refined Program coupling retry");
  EXPECT_EQ(system.n_levels(), 2);
  const auto retry_level_totals = level_totals();
  for (int level = 0; level < system.n_levels(); ++level) {
    const auto index = static_cast<std::size_t>(level);
    const pops::Real accepted_level_total =
        accepted_level_totals[0][index] + accepted_level_totals[1][index];
    const pops::Real retry_level_total =
        retry_level_totals[0][index] + retry_level_totals[1][index];
    EXPECT_NEAR(retry_level_total, accepted_level_total, 1.0e-10)
        << "the Program coupling conservation ledger must preserve each AMR level total";
    EXPECT_LT(retry_level_totals[0][index], accepted_level_totals[0][index])
        << "the donor candidate must transfer a conservative amount";
    EXPECT_GT(retry_level_totals[1][index], accepted_level_totals[1][index])
        << "the receiver candidate must receive the same conservative amount";
  }
}

TEST(AmrTransactionAuthority, ExternalCandidateBlocksReadersAndSealsOnce) {
  constexpr int Dim = pops::kNativeDimension;
  auto system_owner = make_system<Dim>();
  auto& system = *system_owner;
  constexpr double dt = 0.125;

  EXPECT_EQ(system.accepted_transaction_generation_(), 0u);
  EXPECT_THROW((void)system.step_change_l2(), std::logic_error);
  ASSERT_NO_THROW(system.begin_step_transaction());
  ASSERT_NO_THROW(system.step(dt));
  // Candidate observations are permitted only through the explicit writer-owned provisional
  // lease acquired internally by step_change_l2(). A public accepted reader remains blocked.
  EXPECT_FALSE(system.step_change_l2().empty());
  EXPECT_EQ(system.accepted_transaction_generation_(), 0u);
  EXPECT_THROW((void)system.time(), std::logic_error);
  EXPECT_THROW((void)system.macro_step(), std::logic_error);

  std::promise<void> reader_finished;
  auto reader_done = reader_finished.get_future();
  std::exception_ptr reader_failure;
  std::optional<double> observed_time;
  std::optional<int> observed_levels;
  std::optional<std::uint64_t> observed_topology_epoch;
  std::optional<int> observed_components;
  std::thread reader([&] {
    try {
      observed_time = system.time();
      observed_levels = system.n_levels();
      observed_topology_epoch = system.checkpoint_topology_epoch();
      auto state = system.prepared_amr_block_state(0, 0);
      if (!state)
        throw std::logic_error("accepted AMR state reader returned an empty lease");
      observed_components = state->ncomp();
    } catch (...) {
      reader_failure = std::current_exception();
    }
    reader_finished.set_value();
  });
  EXPECT_EQ(reader_done.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  ASSERT_NO_THROW(system.rollback_step_transaction());
  EXPECT_EQ(reader_done.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  reader.join();
  EXPECT_EQ(system.accepted_transaction_generation_(), 0u);
  EXPECT_DOUBLE_EQ(system.time(), 0.0);
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_EQ(reader_failure, nullptr);
  ASSERT_TRUE(observed_time.has_value());
  ASSERT_TRUE(observed_levels.has_value());
  ASSERT_TRUE(observed_topology_epoch.has_value());
  ASSERT_TRUE(observed_components.has_value());
  EXPECT_DOUBLE_EQ(*observed_time, 0.0);
  EXPECT_EQ(*observed_levels, 1);
  EXPECT_EQ(*observed_components, 1);

  // The rejected external attempt above is the explicit warmup: it primes the resident accepted
  // image and exercises the reader lock without changing the accepted generation. The retry must
  // neither allocate a Fab nor acquire a new PoPS communication buffer.
  const pops::AllocationEventStats allocation_before = pops::allocation_event_stats();
  ASSERT_NO_THROW(system.begin_step_transaction());
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before)
      << "resident AMR capture must be fully primed before the candidate starts";
  ASSERT_NO_THROW(system.step(dt));
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before)
      << "AMR candidate execution must not materialize scratch, history or a step workspace";

  // The first prepared per-block query is part of the hot witness.  It must not allocate through
  // either the C++ heap or the PoPS Fab/communication allocators, and a repeated query must fold
  // the same resident slots in exactly the same order.
  const pops::AllocationEventStats first_block_events = pops::allocation_event_stats();
  HeapAllocationWindow first_block_heap;
  const double first_block_l2 = system.step_change_l2_for_block("tracer");
  const std::uint64_t first_block_heap_allocations = first_block_heap.close();
  EXPECT_EQ(first_block_heap_allocations, 0u)
      << "the first post-bind per-block diagnostic must not allocate standard heap storage";
  EXPECT_EQ(pops::allocation_event_stats(), first_block_events)
      << "the first post-bind per-block diagnostic must reuse resident Fab/communication storage";

  const pops::AllocationEventStats repeated_block_events = pops::allocation_event_stats();
  HeapAllocationWindow repeated_block_heap;
  const double repeated_block_l2 = system.step_change_l2_for_block("tracer");
  const std::uint64_t repeated_block_heap_allocations = repeated_block_heap.close();
  EXPECT_EQ(repeated_block_heap_allocations, 0u)
      << "a repeated per-block diagnostic must not allocate standard heap storage";
  EXPECT_EQ(pops::allocation_event_stats(), repeated_block_events)
      << "a repeated per-block diagnostic must reuse resident Fab/communication storage";
  EXPECT_EQ(std::bit_cast<std::uint64_t>(repeated_block_l2),
            std::bit_cast<std::uint64_t>(first_block_l2))
      << "prepared per-block reduction must be bit-identical across repeated calls";
  EXPECT_EQ(system._step_change_l2_last_dispatches(), 1u)
      << "one one-level, one-patch, one-component block must use one level-scale dispatch";

  // The named-map API is intentionally cold and its map allocation is outside the hot witness.
  // Its values must nevertheless be exactly the same as the prepared per-block result.
  const auto aggregate_l2 = system.step_change_l2();
  ASSERT_EQ(aggregate_l2.size(), 1u);
  const auto aggregate_value = aggregate_l2.find("tracer");
  ASSERT_NE(aggregate_value, aggregate_l2.end());
  EXPECT_EQ(std::bit_cast<std::uint64_t>(aggregate_value->second),
            std::bit_cast<std::uint64_t>(first_block_l2))
      << "the compatibility aggregate must delegate to the prepared per-block authority";
  EXPECT_EQ(system._step_change_l2_last_dispatches(), 1u)
      << "the aggregate's one block must retain the same level-scale dispatch bound";
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before)
      << "step_change_l2_for_block must reuse its resident composite workspace";
  ASSERT_NO_THROW(system.commit_step_transaction());
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before)
      << "hidden publication must exchange prebuilt authorities only";
  EXPECT_EQ(system.accepted_transaction_generation_(), 0u);
  ASSERT_NO_THROW(system.finalize_step_transaction());
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before)
      << "AMR seal/finalizer must discard preallocated rollback state without allocation";
  EXPECT_EQ(system.accepted_transaction_generation_(), 1u);
  EXPECT_DOUBLE_EQ(system.time(), dt);
  EXPECT_EQ(system.macro_step(), 1);

  // Refresh the same resident AMR context for a second accepted generation, then prove that a
  // third candidate can still reject and retry without rebuilding clocks/history/flux carriers.
  const pops::AllocationEventStats allocation_before_second_accept = pops::allocation_event_stats();
  ASSERT_NO_THROW(system.begin_step_transaction());
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_second_accept)
      << "the second capture must reuse the resident AMR image";
  ASSERT_NO_THROW(system.step(dt));
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_second_accept)
      << "the second candidate must not allocate after warmup";
  EXPECT_NO_THROW((void)system.step_change_l2_for_block("tracer"));
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_second_accept)
      << "the second block step-change measurement must reuse the resident workspace";
  ASSERT_NO_THROW(system.commit_step_transaction());
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_second_accept)
      << "the second hidden publication must only exchange prepared authorities";
  ASSERT_NO_THROW(system.finalize_step_transaction());
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_second_accept)
      << "the second finalizer must remain allocation-free";
  EXPECT_EQ(system.accepted_transaction_generation_(), 2u);
  EXPECT_DOUBLE_EQ(system.time(), 2.0 * dt);
  EXPECT_EQ(system.macro_step(), 2);

  std::optional<pops::MultiFab<Dim>> accepted_after_two;
  {
    auto accepted = system.prepared_amr_block_state(0, 0);
    ASSERT_TRUE(accepted);
    accepted_after_two.emplace(*accepted);
  }
  const std::uint64_t accepted_topology_epoch = system.checkpoint_topology_epoch();
  const int accepted_levels = system.n_levels();

  ASSERT_NO_THROW(system.begin_step_transaction());
  ASSERT_NO_THROW(system.step(dt));
  EXPECT_FALSE(system.step_change_l2().empty());
  const pops::AllocationEventStats allocation_after_candidate = pops::allocation_event_stats();
  EXPECT_EQ(allocation_after_candidate, allocation_before)
      << "the warmed AMR candidate must not allocate before an observer snapshots it";
  std::optional<pops::MultiFab<Dim>> rejected_candidate;
  {
    auto provisional = system._provisional_read_scope();
    auto candidate = system.prepared_amr_block_state(0, 0);
    ASSERT_TRUE(candidate);
    rejected_candidate.emplace(*candidate);
  }
  // The explicit test observation above deliberately copies a MultiFab.  Reset the allocation
  // baseline after that cold copy so this assertion measures only the runtime rollback path.
  const pops::AllocationEventStats allocation_before_rollback = pops::allocation_event_stats();
  ASSERT_NO_THROW(system.rollback_step_transaction());
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_rollback)
      << "rollback must only restore the resident AMR image and workspace";
  EXPECT_EQ(system.accepted_transaction_generation_(), 2u);
  EXPECT_DOUBLE_EQ(system.time(), 2.0 * dt);
  EXPECT_EQ(system.macro_step(), 2);
  EXPECT_EQ(system.checkpoint_topology_epoch(), accepted_topology_epoch);
  EXPECT_EQ(system.n_levels(), accepted_levels);
  {
    auto after_reject = system.prepared_amr_block_state(0, 0);
    ASSERT_TRUE(after_reject);
    ASSERT_TRUE(accepted_after_two.has_value());
    EXPECT_EQ(pops::difference_sum_sq_all(*after_reject, *accepted_after_two), pops::Real(0))
        << "RejectAttempt must restore the exact sealed AMR field image";
  }

  // The candidate observation is intentionally outside the measured runtime region.  The retry
  // below must be allocation-free through capture, candidate, hidden publish and finalization.
  const pops::AllocationEventStats allocation_before_retry = pops::allocation_event_stats();
  ASSERT_NO_THROW(system.begin_step_transaction());
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_retry);
  ASSERT_NO_THROW(system.step(dt));
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_retry);
  ASSERT_NO_THROW(system.commit_step_transaction());
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_retry);
  ASSERT_NO_THROW(system.finalize_step_transaction());
  EXPECT_EQ(pops::allocation_event_stats(), allocation_before_retry);
  EXPECT_EQ(system.accepted_transaction_generation_(), 3u);
  EXPECT_DOUBLE_EQ(system.time(), 3.0 * dt);
  EXPECT_EQ(system.macro_step(), 3);
  {
    auto after_retry = system.prepared_amr_block_state(0, 0);
    ASSERT_TRUE(after_retry);
    ASSERT_TRUE(rejected_candidate.has_value());
    EXPECT_EQ(pops::difference_sum_sq_all(*after_retry, *rejected_candidate), pops::Real(0))
        << "the retry must publish exactly the discarded candidate, once";
  }
}

TEST(AmrTransactionAuthority, AcceptedRuntimeLazyMaterializationUsesOneSealedWriterResult) {
  constexpr int Dim = pops::kNativeDimension;
  auto system_owner = make_lazy_runtime_system<Dim>();
  auto& system = *system_owner;
  EXPECT_FALSE(system.uses_runtime_engine());

  std::promise<void> start;
  const std::shared_future<void> ready = start.get_future().share();
  std::array<std::optional<std::pair<std::uint64_t, std::uint64_t>>, 2> observations;
  std::array<std::exception_ptr, 2> failures{};
  std::array<std::thread, 2> readers;
  for (std::size_t index = 0; index < readers.size(); ++index) {
    readers[index] = std::thread([&, index] {
      try {
        ready.wait();
        const auto accepted = system.accepted_amr_runtime();
        if (!accepted)
          throw std::logic_error("lazy accepted AMR runtime view is empty");
        observations[index] = {accepted->topology_epoch(), accepted->materialization_generation()};
      } catch (...) {
        failures[index] = std::current_exception();
      }
    });
  }
  start.set_value();
  for (auto& reader : readers)
    reader.join();

  EXPECT_EQ(failures[0], nullptr);
  EXPECT_EQ(failures[1], nullptr);
  ASSERT_TRUE(observations[0].has_value());
  ASSERT_TRUE(observations[1].has_value());
  EXPECT_EQ(*observations[0], *observations[1]);
  EXPECT_TRUE(system.uses_runtime_engine());
}

TEST(AmrTransactionAuthority, PublicReadersBlockUntilRollbackAndAcceptSealsExactState) {
  constexpr int Dim = pops::kNativeDimension;
  constexpr double dt = 0.125;
  auto system_owner = make_system<Dim>(2);
  auto& system = *system_owner;

  const double accepted_window_dt = system.program_cadence_window_dt();
  const int accepted_window_steps = system.program_cadence_window_steps();
  const double accepted_window_start = system.program_cadence_window_start_time();
  const double accepted_last_dt = system.program_last_dt();
  const int accepted_levels = system.n_levels();
  const auto accepted_patches = system.patch_boxes();
  const auto accepted_state = system.block_level_state_global("tracer", 0);
  const auto accepted_program_manifest = system.program_accepted_state_manifest();
  const auto accepted_temporal_relations = system.prepared_program_temporal_relations();
  const auto accepted_field_slots = system.field_provider_slots();
  const auto accepted_dirty_auxiliary = system.dirty_auxiliary_provider_identities();
  EXPECT_DOUBLE_EQ(accepted_window_dt, 0.0);
  EXPECT_EQ(accepted_window_steps, 0);
  EXPECT_DOUBLE_EQ(accepted_window_start, 0.0);

  const auto begin_held_candidate = [&] {
    system.begin_step_transaction();
    system.step(dt);
  };

  // A stride-2 first step holds a visible candidate cadence window.  The public same-thread
  // readers must refuse immediately; the explicit Program path itself has already used its private
  // seams successfully, and the provisional scope is the only deliberate candidate observation.
  ASSERT_NO_THROW(begin_held_candidate());
  EXPECT_THROW((void)system.program_cadence_window_dt(), std::logic_error);
  EXPECT_THROW((void)system.program_cadence_window_steps(), std::logic_error);
  EXPECT_THROW((void)system.program_cadence_window_start_time(), std::logic_error);
  EXPECT_THROW((void)system.program_last_dt(), std::logic_error);
  EXPECT_THROW((void)system.n_levels(), std::logic_error);
  EXPECT_THROW((void)system.patch_boxes(), std::logic_error);
  EXPECT_THROW((void)system.block_level_state_global("tracer", 0), std::logic_error);
  EXPECT_THROW((void)system.program_accepted_state(), std::logic_error);
  EXPECT_THROW((void)system.program_accepted_state_manifest(), std::logic_error);
  EXPECT_THROW((void)system.program_clock_manifest(), std::logic_error);
  EXPECT_THROW((void)system.prepared_program_temporal_relations(), std::logic_error);
  EXPECT_THROW((void)system.prepared_amr_interface_flux_ledger_budget(), std::logic_error);
  EXPECT_THROW((void)system.field_provider_slots(), std::logic_error);
  EXPECT_THROW((void)system.dirty_auxiliary_provider_identities(), std::logic_error);
  {
    auto provisional = system._provisional_read_scope();
    ASSERT_TRUE(provisional.valid());
    EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), dt);
    EXPECT_EQ(system.program_cadence_window_steps(), 1);
    EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), 0.0);
    EXPECT_EQ(system.n_levels(), accepted_levels);
    EXPECT_EQ(system.patch_boxes(), accepted_patches);
    EXPECT_EQ(system.block_level_state_global("tracer", 0), accepted_state);
    EXPECT_DOUBLE_EQ(system.program_last_dt(), accepted_last_dt);
    EXPECT_FALSE(system.program_accepted_state().empty());
    EXPECT_EQ(system.program_accepted_state_manifest(), accepted_program_manifest);
    EXPECT_FALSE(system.program_clock_manifest().empty());
    EXPECT_EQ(system.prepared_program_temporal_relations().size(),
              accepted_temporal_relations.size());
    (void)system.prepared_amr_interface_flux_ledger_budget();
    EXPECT_EQ(system.field_provider_slots(), accepted_field_slots);
    EXPECT_EQ(system.dirty_auxiliary_provider_identities(), accepted_dirty_auxiliary);
  }

  auto cadence_dt_reader = rollback_after_foreign_reader<Dim, double>(
      system, [&system] { return system.program_cadence_window_dt(); });
  EXPECT_EQ(cadence_dt_reader.status_before_release, std::future_status::timeout);
  EXPECT_EQ(cadence_dt_reader.status_after_release, std::future_status::ready);
  EXPECT_EQ(cadence_dt_reader.failure, nullptr);
  ASSERT_TRUE(cadence_dt_reader.value.has_value());
  EXPECT_DOUBLE_EQ(*cadence_dt_reader.value, accepted_window_dt);

  ASSERT_NO_THROW(begin_held_candidate());
  auto cadence_steps_reader = rollback_after_foreign_reader<Dim, int>(
      system, [&system] { return system.program_cadence_window_steps(); });
  EXPECT_EQ(cadence_steps_reader.status_before_release, std::future_status::timeout);
  EXPECT_EQ(cadence_steps_reader.status_after_release, std::future_status::ready);
  EXPECT_EQ(cadence_steps_reader.failure, nullptr);
  ASSERT_TRUE(cadence_steps_reader.value.has_value());
  EXPECT_EQ(*cadence_steps_reader.value, accepted_window_steps);

  ASSERT_NO_THROW(begin_held_candidate());
  auto cadence_start_reader = rollback_after_foreign_reader<Dim, double>(
      system, [&system] { return system.program_cadence_window_start_time(); });
  EXPECT_EQ(cadence_start_reader.status_before_release, std::future_status::timeout);
  EXPECT_EQ(cadence_start_reader.status_after_release, std::future_status::ready);
  EXPECT_EQ(cadence_start_reader.failure, nullptr);
  ASSERT_TRUE(cadence_start_reader.value.has_value());
  EXPECT_DOUBLE_EQ(*cadence_start_reader.value, accepted_window_start);

  ASSERT_NO_THROW(begin_held_candidate());
  auto level_reader =
      rollback_after_foreign_reader<Dim, int>(system, [&system] { return system.n_levels(); });
  EXPECT_EQ(level_reader.status_before_release, std::future_status::timeout);
  EXPECT_EQ(level_reader.status_after_release, std::future_status::ready);
  EXPECT_EQ(level_reader.failure, nullptr);
  ASSERT_TRUE(level_reader.value.has_value());
  EXPECT_EQ(*level_reader.value, accepted_levels);

  ASSERT_NO_THROW(begin_held_candidate());
  auto patch_reader = rollback_after_foreign_reader<Dim, std::vector<pops::AmrPatch<Dim>>>(
      system, [&system] { return system.patch_boxes(); });
  EXPECT_EQ(patch_reader.status_before_release, std::future_status::timeout);
  EXPECT_EQ(patch_reader.status_after_release, std::future_status::ready);
  EXPECT_EQ(patch_reader.failure, nullptr);
  ASSERT_TRUE(patch_reader.value.has_value());
  EXPECT_EQ(*patch_reader.value, accepted_patches);

  ASSERT_NO_THROW(begin_held_candidate());
  auto state_reader = rollback_after_foreign_reader<Dim, std::vector<double>>(
      system, [&system] { return system.block_level_state_global("tracer", 0); });
  EXPECT_EQ(state_reader.status_before_release, std::future_status::timeout);
  EXPECT_EQ(state_reader.status_after_release, std::future_status::ready);
  EXPECT_EQ(state_reader.failure, nullptr);
  ASSERT_TRUE(state_reader.value.has_value());
  EXPECT_EQ(*state_reader.value, accepted_state);

  EXPECT_EQ(system.accepted_transaction_generation_(), 0u);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), accepted_window_dt);
  EXPECT_EQ(system.program_cadence_window_steps(), accepted_window_steps);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), accepted_window_start);
  EXPECT_DOUBLE_EQ(system.program_last_dt(), accepted_last_dt);
  EXPECT_EQ(system.n_levels(), accepted_levels);
  EXPECT_EQ(system.patch_boxes(), accepted_patches);
  EXPECT_EQ(system.block_level_state_global("tracer", 0), accepted_state);
  EXPECT_FALSE(system.program_accepted_state().empty());
  EXPECT_EQ(system.program_accepted_state_manifest(), accepted_program_manifest);
  EXPECT_FALSE(system.program_clock_manifest().empty());
  EXPECT_EQ(system.prepared_program_temporal_relations().size(),
            accepted_temporal_relations.size());
  EXPECT_EQ(system.field_provider_slots(), accepted_field_slots);
  EXPECT_EQ(system.dirty_auxiliary_provider_identities(), accepted_dirty_auxiliary);

  // The second step closes the held window and invokes the installed v5 Program through its
  // private execution seams. Hidden publication keeps every public reader unavailable until seal.
  ASSERT_NO_THROW(system.begin_step_transaction());
  ASSERT_NO_THROW(system.step(dt));
  ASSERT_NO_THROW(system.step(dt));
  EXPECT_EQ(system.accepted_transaction_generation_(), 0u);

  double candidate_window_dt = 0.0;
  int candidate_window_steps = 0;
  double candidate_window_start = 0.0;
  int candidate_levels = 0;
  std::vector<pops::AmrPatch<Dim>> candidate_patches;
  std::vector<double> candidate_state;
  {
    auto provisional = system._provisional_read_scope();
    ASSERT_TRUE(provisional.valid());
    candidate_window_dt = system.program_cadence_window_dt();
    candidate_window_steps = system.program_cadence_window_steps();
    candidate_window_start = system.program_cadence_window_start_time();
    candidate_levels = system.n_levels();
    candidate_patches = system.patch_boxes();
    candidate_state = system.block_level_state_global("tracer", 0);
  }
  EXPECT_DOUBLE_EQ(candidate_window_dt, 0.0);
  EXPECT_EQ(candidate_window_steps, 0);
  EXPECT_DOUBLE_EQ(candidate_window_start, 0.0);

  auto sealed_level_reader =
      seal_after_foreign_reader<Dim, int>(system, [&system] { return system.n_levels(); });
  EXPECT_EQ(sealed_level_reader.status_before_release, std::future_status::timeout);
  EXPECT_EQ(sealed_level_reader.status_after_commit, std::future_status::timeout);
  EXPECT_EQ(sealed_level_reader.status_after_release, std::future_status::ready);
  EXPECT_EQ(sealed_level_reader.failure, nullptr);
  ASSERT_TRUE(sealed_level_reader.value.has_value());
  EXPECT_EQ(*sealed_level_reader.value, candidate_levels);

  EXPECT_EQ(system.accepted_transaction_generation_(), 1u);
  EXPECT_DOUBLE_EQ(system.time(), 2.0 * dt);
  EXPECT_EQ(system.macro_step(), 2);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_dt(), candidate_window_dt);
  EXPECT_EQ(system.program_cadence_window_steps(), candidate_window_steps);
  EXPECT_DOUBLE_EQ(system.program_cadence_window_start_time(), candidate_window_start);
  EXPECT_EQ(system.n_levels(), candidate_levels);
  EXPECT_EQ(system.patch_boxes(), candidate_patches);
  EXPECT_EQ(system.block_level_state_global("tracer", 0), candidate_state);
}

TEST(AmrTransactionAuthority, DistributionReplayAndDensityReadersRequireAnExplicitLease) {
  constexpr int Dim = pops::kNativeDimension;
  constexpr double dt = 0.125;
  auto system_owner = make_system<Dim>();
  auto& system = *system_owner;

  const auto accepted_owners = system.level_owner_ranks(0);
  const auto accepted_replay_steps = system.last_replay_regrid_steps();
  const auto accepted_density = system.density("tracer");
  const auto accepted_structural = structural_readers(system);

  const auto begin_candidate = [&] {
    system.begin_step_transaction();
    system.step(dt);
  };

  // Public readers on the writer thread cannot inspect a resident candidate without the explicit
  // provisional scope.  The scope is the only intentional candidate-read authority.
  ASSERT_NO_THROW(begin_candidate());
  EXPECT_THROW((void)system.level_owner_ranks(0), std::logic_error);
  EXPECT_THROW((void)system.last_replay_regrid_steps(), std::logic_error);
  EXPECT_THROW((void)system.density("tracer"), std::logic_error);
  EXPECT_THROW((void)system.max_levels(), std::logic_error);
  EXPECT_THROW((void)system.n_vars(), std::logic_error);
  EXPECT_THROW((void)system.variable_names("tracer"), std::logic_error);
  EXPECT_THROW((void)system.block_n_vars("tracer"), std::logic_error);
  {
    auto provisional = system._provisional_read_scope();
    ASSERT_TRUE(provisional.valid());
    EXPECT_EQ(system.level_owner_ranks(0), accepted_owners);
    EXPECT_EQ(system.last_replay_regrid_steps(), accepted_replay_steps);
    EXPECT_EQ(system.density("tracer"), accepted_density);
    EXPECT_EQ(structural_readers(system), accepted_structural);
  }

  // Each reader blocks on a foreign thread and resumes only after rollback has restored the
  // accepted image.
  auto owner_after_rollback = rollback_after_foreign_reader<Dim, std::vector<int>>(
      system, [&system] { return system.level_owner_ranks(0); });
  EXPECT_EQ(owner_after_rollback.status_before_release, std::future_status::timeout);
  EXPECT_EQ(owner_after_rollback.status_after_release, std::future_status::ready);
  EXPECT_EQ(owner_after_rollback.failure, nullptr);
  ASSERT_TRUE(owner_after_rollback.value.has_value());
  EXPECT_EQ(*owner_after_rollback.value, accepted_owners);

  ASSERT_NO_THROW(begin_candidate());
  auto replay_after_rollback = rollback_after_foreign_reader<Dim, std::vector<int>>(
      system, [&system] { return system.last_replay_regrid_steps(); });
  EXPECT_EQ(replay_after_rollback.status_before_release, std::future_status::timeout);
  EXPECT_EQ(replay_after_rollback.status_after_release, std::future_status::ready);
  EXPECT_EQ(replay_after_rollback.failure, nullptr);
  ASSERT_TRUE(replay_after_rollback.value.has_value());
  EXPECT_EQ(*replay_after_rollback.value, accepted_replay_steps);

  ASSERT_NO_THROW(begin_candidate());
  auto density_after_rollback = rollback_after_foreign_reader<Dim, std::vector<double>>(
      system, [&system] { return system.density("tracer"); });
  EXPECT_EQ(density_after_rollback.status_before_release, std::future_status::timeout);
  EXPECT_EQ(density_after_rollback.status_after_release, std::future_status::ready);
  EXPECT_EQ(density_after_rollback.failure, nullptr);
  ASSERT_TRUE(density_after_rollback.value.has_value());
  EXPECT_EQ(*density_after_rollback.value, accepted_density);

  // Grouped structural readers also share the accepted barrier: the first foreign read blocks
  // during a candidate, and after rollback the complete snapshot is the pre-candidate one.
  ASSERT_NO_THROW(begin_candidate());
  auto structural_after_rollback = rollback_after_foreign_reader<Dim, StructuralReaderSnapshot>(
      system, [&system] { return structural_readers(system); });
  EXPECT_EQ(structural_after_rollback.status_before_release, std::future_status::timeout);
  EXPECT_EQ(structural_after_rollback.status_after_release, std::future_status::ready);
  EXPECT_EQ(structural_after_rollback.failure, nullptr);
  ASSERT_TRUE(structural_after_rollback.value.has_value());
  EXPECT_EQ(*structural_after_rollback.value, accepted_structural);

  // A foreign density reader remains blocked through hidden publication and receives exactly the
  // candidate image only after the atomic seal/finalizer releases the accepted reader barrier.
  ASSERT_NO_THROW(begin_candidate());
  std::vector<double> candidate_density;
  {
    auto provisional = system._provisional_read_scope();
    ASSERT_TRUE(provisional.valid());
    candidate_density = system.density("tracer");
  }
  auto density_after_seal = seal_after_foreign_reader<Dim, std::vector<double>>(
      system, [&system] { return system.density("tracer"); });
  EXPECT_EQ(density_after_seal.status_before_release, std::future_status::timeout);
  EXPECT_EQ(density_after_seal.status_after_commit, std::future_status::timeout);
  EXPECT_EQ(density_after_seal.status_after_release, std::future_status::ready);
  EXPECT_EQ(density_after_seal.failure, nullptr);
  ASSERT_TRUE(density_after_seal.value.has_value());
  EXPECT_EQ(*density_after_seal.value, candidate_density);
}

// Regrid fault injection intentionally remains outside this witness: the public v5 fixture has
// DSO step failures only, while a regrid failure must be injected between forward staging and the
// hidden no-throw topology publication.  Adding a synthetic public hook here would create the
// very second authority this test is intended to forbid.  The regrid transaction integration
// fixture owns that phase-level fault matrix.
