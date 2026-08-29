#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/time/integrators/implicit_stepper.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/program_execution_services.hpp>
#include <pops/runtime/program/step_transaction.hpp>

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kImplicitTransactionRetryReason = 0x49545259u;

template <int Dim>
pops::Extent<Dim> uniform_extent(int value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::MultiFab<Dim> one_patch_field(int width, int ncomp) {
  const pops::Box<Dim> box = pops::Box<Dim>::from_extents(uniform_extent<Dim>(width));
  const pops::mesh::BoxArray<Dim> layout(std::vector<pops::Box<Dim>>{box});
  const pops::mesh::RankSpace<Dim> ranks(pops::Index<Dim>{}, uniform_extent<Dim>(1));
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(layout, ranks);
  return pops::MultiFab<Dim>(layout, distribution, pops::Index<Dim>{}, ncomp, pops::Extent<Dim>{});
}

struct StiffLinearSource {
  using State = pops::StateVec<1>;
  static constexpr int n_vars = 1;

  POPS_HD State source(const State& state, const pops::ProviderValues<0>&) const {
    return State{-state[0]};
  }
  POPS_HD void source_jacobian(const State&, const pops::ProviderValues<0>&,
                               pops::Real (&jacobian)[1][1]) const {
    jacobian[0][0] = pops::Real(-1);
  }
};

template <int Dim>
struct RelaxingAdvectionModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;

  Law law{};
  pops::Real rate = pops::Real(0);
  pops::Real equilibrium = pops::Real(0);

  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.implicit-transaction.relaxing-advection", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(law.velocity()[axis]);
    contract.scalar(rate).scalar(equilibrium);
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
  POPS_HD State source(const State& state, const pops::ProviderValues<0>&) const {
    return State{rate * (equilibrium - state[0])};
  }
  POPS_HD void source_jacobian(const State&, const pops::ProviderValues<0>&,
                               pops::Real (&jacobian)[1][1]) const {
    jacobian[0][0] = -rate;
  }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

template <int Dim>
RelaxingAdvectionModel<Dim> relaxing_model(pops::Real rate, pops::Real equilibrium) {
  pops::RealVector<Dim> velocity{};
  return {pops::nd::ScalarAdvection<Dim>::prepare(velocity), rate, equilibrium};
}

template <int Dim>
std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
bool byte_exact_equal(const pops::MultiFab<Dim>& left, const pops::MultiFab<Dim>& right) {
  bool same = left.layout() == right.layout() && left.distribution() == right.distribution() &&
              left.local_rank() == right.local_rank() && left.local_size() == right.local_size() &&
              left.ncomp() == right.ncomp() && left.ghosts() == right.ghosts();
  for (std::size_t local = 0; same && local < left.local_size(); ++local) {
    const auto& left_fab = left.fab(local);
    const auto& right_fab = right.fab(local);
    auto left_host = left_fab.create_host_mirror();
    auto right_host = right_fab.create_host_mirror();
    left_fab.copy_to_host(left_host);
    right_fab.copy_to_host(right_host);
    if (left_host.size() != right_host.size())
      same = false;
    for (std::size_t entry = 0; same && entry < left_host.size(); ++entry)
      same = std::bit_cast<std::array<std::byte, sizeof(pops::Real)>>(left_host(entry)) ==
             std::bit_cast<std::array<std::byte, sizeof(pops::Real)>>(right_host(entry));
  }
  return pops::all_reduce_max(same ? 0L : 1L) == 0;
}

}  // namespace

TEST(test_amr_multiblock_implicit_transaction,
     RankedImplicitSolveRequiresAnExplicitLaneAndDefersPublication) {
  constexpr int Dim = pops::kNativeDimension;
  auto state = one_patch_field<Dim>(2, 1);
  state.set_val(pops::Real(2));
  const pops::ExecutionLane lane = pops::ExecutionLane::world();

  pops::SolveOutcome outcome = pops::backward_euler_source(
      StiffLinearSource{}, [](std::size_t) { return pops::ProviderStorageView<Dim, 0>{}; }, state,
      pops::Real(0.25), pops::NewtonOptions{}, lane);
  ASSERT_TRUE(outcome.report().solved_value_available());
  EXPECT_EQ(pops::reduce_min_local(state), pops::Real(2));
  const pops::SolveReport accepted = outcome.consume(pops::SolveConsumption::kAccept);
  EXPECT_TRUE(accepted.solved());
  EXPECT_NEAR(static_cast<double>(pops::reduce_min_local(state)), 1.6,
              64.0 * std::numeric_limits<double>::epsilon());
  EXPECT_NEAR(static_cast<double>(pops::reduce_max_local(state)), 1.6,
              64.0 * std::numeric_limits<double>::epsilon());
}

TEST(test_amr_multiblock_implicit_transaction,
     TwoBlockImplicitSourceTransactionRollsBackAndRetriesExactly) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;

  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system,
                                            "tests.implicit-transaction/multiblock-runtime@1");
  system.install_block_state_route("slow", "state/slow");
  system.install_block_state_route("fast", "state/fast");
  pops::add_compiled_model<Dim>(system, "slow", relaxing_model<Dim>(pops::Real(8), pops::Real(2)),
                                "minmod", "rusanov", "conservative", "explicit",
                                static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.implicit-transaction/slow/physical-flux");
  pops::add_compiled_model<Dim>(system, "fast", relaxing_model<Dim>(pops::Real(24), pops::Real(-1)),
                                "minmod", "rusanov", "conservative", "explicit",
                                static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.implicit-transaction/fast/physical-flux");
  system.set_conservative_state("slow", std::vector<double>(cell_count(config.shape), 0.25));
  system.set_conservative_state("fast", std::vector<double>(cell_count(config.shape), 3.0));
  using Resource = pops::test::program_v5::CallbackProgramResource;
  std::vector<Resource> resources;
  resources.reserve(2);
  for (int program_block = 0; program_block < 2; ++program_block) {
    const int runtime_block = program_block == 0 ? 1 : 0;
    const auto state = system.prepared_amr_block_state(runtime_block, 0);
    ASSERT_TRUE(state);
    resources.push_back({Resource::Kind::state, resources.size(), 0, program_block, 0,
                         static_cast<std::uint32_t>(state->ncomp()),
                         static_cast<std::uint32_t>(state->ghosts()[0])});
  }
  auto inject_retry = std::make_shared<bool>(true);
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, "tests.implicit-transaction/multiblock-program@1",
      "tests.implicit-transaction.multiblock-clock", std::vector<std::string>{"fast", "slow"},
      resources, {}, [inject_retry](auto& context, double macro_dt) {
        context.advance_hierarchy(macro_dt, [&context, inject_retry](double level_dt) {
          std::array<pops::MultiFab<Dim>*, 2> accepted{};
          std::array<pops::MultiFab<Dim>*, 2> candidates{};
          context.set_stage_time(0, 1);
          for (int program_block = 0; program_block < 2; ++program_block) {
            accepted[program_block] = &context.state(program_block);
            auto& candidate = context.scratch_state(
                static_cast<pops::runtime::program::ProgramCacheSlot>(program_block), 0,
                *accepted[program_block]);
            // This is intentionally a source-only, atomic two-block transaction.  Do not add a
            // numerically-zero transport stage: temporal composition belongs to authored Program
            // evidence, while this fixture proves the implicit solve/retry/commit contract.
            context.lincomb(candidate, pops::Real(1), *accepted[program_block], pops::Real(0),
                            *accepted[program_block]);
            pops::SolveOutcome implicit = context.solve_source_default(
                program_block, candidate, pops::Real(level_dt), pops::NewtonOptions{});
            const pops::SolveReport report = implicit.consume(pops::SolveConsumption::kAccept);
            if (!report.solved())
              throw std::runtime_error("two-block implicit source did not converge");
            candidates[program_block] = &candidate;
          }
          if (*inject_retry) {
            *inject_retry = false;
            throw pops::runtime::program::StepAttemptRejected(
                pops::SolveStatus::kIterationLimit,
                pops::runtime::program::StepAttemptDisposition::kRetry,
                kImplicitTransactionRetryReason, "implicit-source",
                "injected-implicit-transaction-retry");
          }
          context.commit_many({{accepted[0], candidates[0]}, {accepted[1], candidates[1]}});
        });
      });
  auto slow_before_view = system.prepared_amr_block_state(0, 0);
  auto fast_before_view = system.prepared_amr_block_state(1, 0);
  ASSERT_TRUE(slow_before_view);
  ASSERT_TRUE(fast_before_view);
  const pops::MultiFab<Dim> slow_before = *slow_before_view;
  const pops::MultiFab<Dim> fast_before = *fast_before_view;
  constexpr double dt = 0.05;
  EXPECT_EQ(system.accepted_transaction_generation_(), 0u);
  try {
    system.step(dt);
    FAIL() << "the full-pack retry was not surfaced";
  } catch (const pops::runtime::program::StepAttemptRejected& rejected) {
    EXPECT_EQ(rejected.disposition(), pops::runtime::program::StepAttemptDisposition::kRetry);
    EXPECT_EQ(rejected.reason_code(), kImplicitTransactionRetryReason);
    EXPECT_EQ(rejected.phase(), "implicit-source");
  }
  EXPECT_EQ(system.macro_step(), 0);
  EXPECT_DOUBLE_EQ(system.time(), 0.0);
  EXPECT_EQ(system.accepted_transaction_generation_(), 0u);
  auto slow_after_reject_view = system.prepared_amr_block_state(0, 0);
  auto fast_after_reject_view = system.prepared_amr_block_state(1, 0);
  ASSERT_TRUE(slow_after_reject_view);
  ASSERT_TRUE(fast_after_reject_view);
  EXPECT_TRUE(byte_exact_equal(slow_before, *slow_after_reject_view));
  EXPECT_TRUE(byte_exact_equal(fast_before, *fast_after_reject_view));

  ASSERT_NO_THROW(system.step(dt));
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.time(), dt);
  EXPECT_EQ(system.accepted_transaction_generation_(), 1u);
  const auto slow_after = system.block_level_state_global("slow", 0);
  const auto fast_after = system.block_level_state_global("fast", 0);
  ASSERT_FALSE(slow_after.empty());
  ASSERT_FALSE(fast_after.empty());
  EXPECT_NEAR(slow_after.front(), (0.25 + dt * 8.0 * 2.0) / (1.0 + dt * 8.0), 1.0e-12);
  EXPECT_NEAR(fast_after.front(), (3.0 + dt * 24.0 * -1.0) / (1.0 + dt * 24.0), 1.0e-12);
  EXPECT_NE(slow_after.front(), fast_after.front());

  // RuntimeInstance's outer transaction owns the candidate writer from begin through rollback or
  // finalize. A nested AMR step must not seal early or let another reader observe its provisional
  // clocks. First prove rollback; then prove commit/seal increments exactly once.
  system.begin_step_transaction();
  ASSERT_NO_THROW(system.step(dt));
  EXPECT_EQ(system.accepted_transaction_generation_(), 1u);
  EXPECT_THROW((void)system.time(), std::logic_error);
  EXPECT_THROW((void)system.macro_step(), std::logic_error);
  std::promise<void> reader_finished;
  auto reader_done = reader_finished.get_future();
  std::thread reader([&] {
    (void)system.time();
    reader_finished.set_value();
  });
  EXPECT_EQ(reader_done.wait_for(std::chrono::milliseconds(20)), std::future_status::timeout);
  system.rollback_step_transaction();
  EXPECT_EQ(system.accepted_transaction_generation_(), 1u);
  EXPECT_EQ(reader_done.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  reader.join();
  EXPECT_EQ(system.macro_step(), 1);
  EXPECT_DOUBLE_EQ(system.time(), dt);

  system.begin_step_transaction();
  ASSERT_NO_THROW(system.step(dt));
  ASSERT_NO_THROW(system.commit_step_transaction());
  EXPECT_EQ(system.accepted_transaction_generation_(), 1u);
  EXPECT_THROW((void)system.time(), std::logic_error);
  ASSERT_NO_THROW(system.finalize_step_transaction());
  EXPECT_EQ(system.accepted_transaction_generation_(), 2u);
  EXPECT_EQ(system.macro_step(), 2);
  EXPECT_DOUBLE_EQ(system.time(), 2.0 * dt);
}

TEST(test_amr_multiblock_implicit_transaction, MetadataNeverCreatesAnImplicitTemporalFallback) {
  constexpr int Dim = pops::kNativeDimension;
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = 8;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(
      system, "tests.implicit-transaction/no-native-dispatch-runtime@1");
  system.install_block_state_route("tracer", "state/tracer");
  pops::add_compiled_model<Dim>(system, "tracer", relaxing_model<Dim>(pops::Real(1), pops::Real(0)),
                                "minmod", "rusanov", "conservative", "imex",
                                static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                                static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.implicit-transaction.no-native-dispatch/physical-flux");
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  const std::vector<double> accepted = system.block_level_state_global("tracer", 0);

  try {
    system.step(0.1);
    FAIL() << "metadata must not synthesize an implicit Program";
  } catch (const std::logic_error& error) {
    EXPECT_STREQ(error.what(), "AmrSystem::step requires an installed whole-system Program");
  }
  EXPECT_EQ(system.block_level_state_global("tracer", 0), accepted);
}
