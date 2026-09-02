/// @file
/// @brief Global-host-allocation witness for the prepared uniform step-change diagnostic.

#include <gtest/gtest.h>

#include "amr_tagging_test_authority.hpp"
#include "native_dso_compiler.hpp"
#include "explicit_amr_program.hpp"
#include "program_v5_fixture.hpp"
#include <pops/core/foundation/allocator.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/system.hpp>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::atomic<std::uint64_t> g_heap_allocations{0};
void* counted_allocate(std::size_t size) {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer == nullptr)
    throw std::bad_alloc();
  g_heap_allocations.fetch_add(1, std::memory_order_relaxed);
  return pointer;
}
void* counted_aligned_allocate(std::size_t size, std::size_t alignment) {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) != 0)
    pointer = nullptr;
  if (pointer == nullptr)
    throw std::bad_alloc();
  g_heap_allocations.fetch_add(1, std::memory_order_relaxed);
  return pointer;
}
void* counted_allocate_nothrow(std::size_t size) noexcept {
  void* pointer = std::malloc(size == 0 ? 1 : size);
  if (pointer != nullptr)
    g_heap_allocations.fetch_add(1, std::memory_order_relaxed);
  return pointer;
}
void* counted_aligned_allocate_nothrow(std::size_t size, std::size_t alignment) noexcept {
  void* pointer = nullptr;
  if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) != 0)
    pointer = nullptr;
  if (pointer != nullptr)
    g_heap_allocations.fetch_add(1, std::memory_order_relaxed);
  return pointer;
}
}  // namespace

void* operator new(std::size_t size) {
  return counted_allocate(size);
}
void* operator new[](std::size_t size) {
  return counted_allocate(size);
}
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  return counted_allocate_nothrow(size);
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  return counted_allocate_nothrow(size);
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
  return counted_aligned_allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
  return counted_aligned_allocate(size, static_cast<std::size_t>(alignment));
}
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return counted_aligned_allocate_nothrow(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  return counted_aligned_allocate_nothrow(size, static_cast<std::size_t>(alignment));
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

namespace pops {
template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}
}  // namespace pops

namespace {
template <int Dim>
pops::SystemConfig<Dim> probe_config(int cells_per_axis) {
  pops::SystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = cells_per_axis;
    config.lower[axis] = pops::Real(0);
    config.upper[axis] = pops::Real(1);
    config.periodicity[axis] = true;
  }
  return config;
}
template <int Dim>
void install_probe_blocks(pops::System<Dim>& system, const std::vector<std::string>& blocks) {
  for (const std::string& block : blocks)
    system.install_block_state_route(block, "test.step-change-l2/state/" + block + "@1");
  system.seal_auxiliary_providers();
  for (const std::string& block : blocks)
    pops::add_compiled_model(system, block, pops::nd::ScalarAdvection<Dim>{}, "none", "rusanov",
                             "conservative", "explicit");
}
template <int Dim>
void install_probe_program(pops::System<Dim>& system, const std::vector<std::string>& blocks) {
#if !defined(POPS_TEST_TMPDIR)
  throw std::runtime_error("step-change allocation probe requires POPS_TEST_TMPDIR");
#else
  static std::size_t fixture_index = 0;
  const std::string prefix =
      std::string(POPS_TEST_TMPDIR) + "/step_change_l2_probe_" + std::to_string(++fixture_index);
  const std::string source_path = prefix + ".cpp";
  const std::string library_path = prefix + ".so";
  std::ofstream source(source_path);
  if (!source)
    throw std::runtime_error("cannot create step-change allocation probe source");
  source << pops::test::program_v5::authority_program_source(
      "noop", "test.step-change-l2-allocation.v5", blocks);
  source.close();
  const auto compiled = pops::test::native_dso::compile_shared(source_path, library_path);
  if (!compiled.ok) {
    pops::test::native_dso::report_compile_failure("test_step_change_l2_uniform_allocation",
                                                   compiled);
    throw std::runtime_error("step-change allocation probe fixture compilation failed");
  }
  system.install_program(library_path);
#endif
}
template <int Dim>
void assert_prepared_calls_allocate_nothing(const std::vector<std::string>& blocks,
                                            int cells_per_axis,
                                            std::uint64_t& observed_dispatches) {
  pops::System<Dim> system(probe_config<Dim>(cells_per_axis));
  system.install_prepared_boundary_execution_lane(std::make_shared<pops::ExecutionLane>(
      pops::ExecutionLane::duplicate_world_collectively("pops.test.step-change-l2-allocation")));
  install_probe_blocks(system, blocks);
  install_probe_program(system, blocks);
  system.mark_bound();
  ASSERT_NO_THROW(system.begin_step_transaction());
  ASSERT_NO_THROW(system.step(0.125));
  auto scope = system._provisional_read_scope();
  ASSERT_TRUE(scope.valid());
  for (const std::string& block : blocks)
    ASSERT_NO_THROW((void)system.step_change_l2_for_block(block));
  observed_dispatches = system._step_change_l2_last_dispatches();
  const std::uint64_t before = g_heap_allocations.load(std::memory_order_relaxed);
  double accumulated = 0.0;
  for (int repeat = 0; repeat < 8; ++repeat)
    for (const std::string& block : blocks)
      accumulated += system.step_change_l2_for_block(block);
  const std::uint64_t after = g_heap_allocations.load(std::memory_order_relaxed);
  EXPECT_EQ(after, before);
  EXPECT_TRUE(std::isfinite(accumulated));
  ASSERT_NO_THROW(system.rollback_step_transaction());
}

template <int Dim>
struct ProbeAmrScalarModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  Law law{};
  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.step-change-l2.amr-scalar", 1};
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

template <int Dim>
void assert_prepared_amr_calls_allocate_nothing(const std::vector<std::string>& blocks,
                                                int cells_per_axis, int levels,
                                                std::uint64_t& observed_dispatches) {
  pops::AmrSystemConfig<Dim> config;
  config.level_count = levels;
  config.regrid_every = 0;
  config.distribute_coarse = true;
  if (levels == 1) {
    config.transition_ratios.clear();
    config.transition_buffers.clear();
    config.transition_lookaheads.clear();
  }
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = cells_per_axis;
    config.coarse_max_grid[axis] = cells_per_axis > 1 ? cells_per_axis / 2 : 1;
    config.periodicity[axis] = true;
  }
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "pops.test.step-change-l2-amr-allocation");
  if (levels > 1) {
    const auto transitions = static_cast<std::size_t>(levels - 1);
    system.set_temporal_relations(std::vector<std::int64_t>(transitions, 2),
                                  std::vector<std::int64_t>(transitions, 1),
                                  std::vector<std::string>(transitions, "integral_only"));
  }
  for (const std::string& block : blocks)
    system.install_block_state_route(block, "test.step-change-l2/amr-state/" + block + "@1");
  pops::RealVector<Dim> velocity{};
  velocity[0] = pops::Real(1);
  for (const std::string& block : blocks) {
    pops::add_compiled_model<Dim>(
        system, block, ProbeAmrScalarModel<Dim>{pops::nd::ScalarAdvection<Dim>::prepare(velocity)},
        "minmod", "rusanov", "conservative", "explicit",
        static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
        static_cast<double>(pops::kWenoEpsilon), false,
        "test.step-change-l2/physical-flux/" + block + "@1");
  }
  std::size_t cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    cells *= static_cast<std::size_t>(config.shape[axis]);
  std::vector<double> initial(cells, 1.0);
  for (std::size_t cell = 0; cell < cells; ++cell)
    initial[cell] += 0.01 * static_cast<double>(cell % 17U);
  for (const std::string& block : blocks)
    system.set_conservative_state(block, initial);
  if (levels > 1) {
    const std::string& tagged_block = blocks.front();
    pops::test::install_prepared_threshold_union(
        system,
        {{tagged_block, "u", 0.5, pops::test::PreparedThresholdRelation::Above,
          "test.step-change-l2/amr-state/" + tagged_block + "@1"}},
        "test.step-change-l2/amr-tagging@1");
  }
  ASSERT_EQ(system.max_levels(), levels);
  ASSERT_EQ(system.n_levels(), levels) << "the allocation probe requires its planned hierarchy";
  std::uint64_t expected_dispatches = 0;
  {
    const auto accepted = system.accepted_amr_runtime();
    ASSERT_TRUE(accepted);
    for (std::size_t level = 0; level < accepted->hierarchy().num_levels(); ++level)
      expected_dispatches += accepted->hierarchy().state(level).local_size() *
                             static_cast<std::size_t>(accepted->hierarchy().state(level).ncomp());
  }
  pops::test::install_forward_euler_program_execution_services(system, false);
  ASSERT_NO_THROW(system.begin_step_transaction());
  ASSERT_NO_THROW(system.step(0.01));
  auto scope = system._provisional_read_scope();
  ASSERT_TRUE(scope.valid());
  const auto composite_oracle = system.step_change_l2();
  for (const std::string& block : blocks) {
    double prepared_value = 0.0;
    ASSERT_NO_THROW(prepared_value = system.step_change_l2_for_block(block));
    const double expected_value = composite_oracle.at(block);
    const double scale = std::abs(expected_value) > 1.0 ? std::abs(expected_value) : 1.0;
    EXPECT_NEAR(prepared_value, expected_value,
                64.0 * std::numeric_limits<double>::epsilon() * scale);
    EXPECT_EQ(system._step_change_l2_last_dispatches(), expected_dispatches);
  }
  observed_dispatches = system._step_change_l2_last_dispatches();
  const auto fab_comm_before = pops::allocation_event_stats();
  const std::uint64_t heap_before = g_heap_allocations.load(std::memory_order_relaxed);
  double accumulated = 0.0;
  for (int repeat = 0; repeat < 8; ++repeat)
    for (const std::string& block : blocks)
      accumulated += system.step_change_l2_for_block(block);
  const std::uint64_t heap_after = g_heap_allocations.load(std::memory_order_relaxed);
  EXPECT_EQ(heap_after, heap_before);
  EXPECT_EQ(pops::allocation_event_stats(), fab_comm_before);
  EXPECT_TRUE(std::isfinite(accumulated));
  ASSERT_NO_THROW(system.rollback_step_transaction());
}
}  // namespace

TEST(StepChangeL2GlobalAllocationProbe, PreparedUniformSingleAndMultiBlockCallsAllocateNothing) {
  constexpr int dim = pops::kNativeDimension;
  const std::string long_name(192, 'x');
  std::uint64_t shape4 = 0;
  std::uint64_t shape8 = 0;
  std::uint64_t multi_block = 0;
  assert_prepared_calls_allocate_nothing<dim>({"gas"}, 4, shape4);
  assert_prepared_calls_allocate_nothing<dim>({"gas"}, 8, shape8);
  assert_prepared_calls_allocate_nothing<dim>({"gas", long_name}, 4, multi_block);
  EXPECT_EQ(shape4, shape8);
  EXPECT_EQ(shape4, multi_block);
}

TEST(StepChangeL2GlobalAllocationProbe,
     PreparedTwoLevelAmrSingleAndMultiBlockCallsAllocateNothing) {
  constexpr int dim = pops::kNativeDimension;
  const std::string long_name(192, 'a');
  std::uint64_t shape8 = 0;
  std::uint64_t shape16 = 0;
  std::uint64_t levels2 = 0;
  assert_prepared_amr_calls_allocate_nothing<dim>({"tracer"}, 8, 1, shape8);
  assert_prepared_amr_calls_allocate_nothing<dim>({"tracer"}, 16, 1, shape16);
  assert_prepared_amr_calls_allocate_nothing<dim>({"tracer", long_name}, 8, 2, levels2);
  EXPECT_EQ(shape8, shape16);
  EXPECT_GT(levels2, shape8);
}
