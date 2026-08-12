#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <int Dim>
struct ScalarModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;

  Law law{};

  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr.multiblock-coupled-source.scalar", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(law.velocity()[axis]);
  }
  static pops::VariableSet conservative_vars() {
    return {pops::VariableKind::Conservative, {"amount"}, 1, {pops::VariableRole::Scalar}};
  }
  static pops::VariableSet primitive_vars() {
    return {pops::VariableKind::Primitive, {"amount"}, 1, {pops::VariableRole::Scalar}};
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
  POPS_HD State project(const State& state, const pops::ProviderValues<0>&) const { return state; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

template <int Dim>
ScalarModel<Dim> scalar_model() {
  pops::RealVector<Dim> velocity{};
  for (int axis = 0; axis < Dim; ++axis)
    velocity[axis] = pops::Real(0.15) / pops::Real(axis + 1);
  return {pops::nd::ScalarAdvection<Dim>::prepare(velocity)};
}

template <int Dim>
std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
std::vector<double> smooth_state(const pops::Extent<Dim>& shape, double mean, double amplitude) {
  std::vector<double> result(cell_count(shape), mean);
  const double pi = std::acos(-1.0);
  for (std::size_t linear = 0; linear < result.size(); ++linear) {
    std::size_t quotient = linear;
    double wave = amplitude;
    for (int axis = 0; axis < Dim; ++axis) {
      const int coordinate = static_cast<int>(quotient % static_cast<std::size_t>(shape[axis]));
      quotient /= static_cast<std::size_t>(shape[axis]);
      const double x = (static_cast<double>(coordinate) + 0.5) / static_cast<double>(shape[axis]);
      wave *= std::sin(2.0 * pi * x);
    }
    result[linear] += wave;
  }
  return result;
}

template <int Dim>
struct ConservativeExchange {
  pops::FieldView<pops::Real, Dim> donor{};
  pops::FieldView<pops::Real, Dim> receiver{};
  pops::Real dt = pops::Real(0);

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    const pops::Real amount = dt * pops::Real(0.25) * donor(cell, 0);
    donor(cell, 0) -= amount;
    receiver(cell, 0) += amount;
  }
};

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

template <int Dim>
using BlockLevelSnapshot = std::array<std::vector<pops::MultiFab<Dim>>, 2>;

template <int Dim>
BlockLevelSnapshot<Dim> snapshot_block_levels(pops::AmrSystem<Dim>& system) {
  BlockLevelSnapshot<Dim> result;
  for (int runtime_block = 0; runtime_block < 2; ++runtime_block) {
    auto& levels = result[static_cast<std::size_t>(runtime_block)];
    levels.reserve(static_cast<std::size_t>(system.n_levels()));
    for (int level = 0; level < system.n_levels(); ++level)
      levels.emplace_back(system.prepared_amr_block_state(runtime_block, level));
  }
  return result;
}

template <int Dim>
bool byte_exact_block_levels(const BlockLevelSnapshot<Dim>& expected,
                             pops::AmrSystem<Dim>& actual) {
  bool same = true;
  for (int runtime_block = 0; runtime_block < 2; ++runtime_block)
    for (int level = 0; level < actual.n_levels(); ++level)
      same = same &&
             byte_exact_equal(
                 expected[static_cast<std::size_t>(runtime_block)][static_cast<std::size_t>(level)],
                 actual.prepared_amr_block_state(runtime_block, level));
  return same;
}

template <int Dim>
struct CoupledFixture {
  pops::AmrSystem<Dim> system;
  std::shared_ptr<pops::runtime::program::AmrProgramContext<Dim>> context;
  std::shared_ptr<bool> fail_on_rank_zero = std::make_shared<bool>(false);
  std::shared_ptr<bool> violate_conservation_on_rank_zero = std::make_shared<bool>(false);
  std::shared_ptr<bool> cross_owner_projection_scratch = std::make_shared<bool>(false);

  CoupledFixture() : system(config()) {
    system.install_block_state_route("donor", "state/donor");
    system.install_block_state_route("receiver", "state/receiver");
    pops::add_compiled_model<Dim>(system, "donor", scalar_model<Dim>(), "minmod", "rusanov",
                                  "conservative", "explicit",
                                  static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {},
                                  0.0, static_cast<double>(pops::kWenoEpsilon), false,
                                  "test.amr.multiblock-coupled-source/donor-flux");
    pops::add_compiled_model<Dim>(system, "receiver", scalar_model<Dim>(), "minmod", "rusanov",
                                  "conservative", "explicit",
                                  static_cast<double>(pops::kPhysicalDefaultGamma), 1, 1, {}, {},
                                  0.0, static_cast<double>(pops::kWenoEpsilon), false,
                                  "test.amr.multiblock-coupled-source/receiver-flux");
    system.set_temporal_relations({2}, {1}, {"integral_only"});
    system.set_conservative_state("donor", smooth_state(config().shape, 2.0, 0.2));
    system.set_conservative_state("receiver", smooth_state(config().shape, 5.0, 0.1));

    pops::CouplingOperatorView view;
    view.label = "owner-qualified-conservative-exchange";
    std::vector<pops::runtime::system::PreparedCouplingConservationGroup> conservation{
        {"donor-receiver.scalar", {{"donor", 0, 0, "scalar"}, {"receiver", 1, 0, "scalar"}}}};
    system.install_prepared_amr_coupling_operator(
        "test.amr.multiblock-coupled-source/exchange@1", std::move(view),
        pops::runtime::system::PreparedCouplingOperator<Dim>(
            [fail = fail_on_rank_zero, violate = violate_conservation_on_rank_zero](
                pops::Real dt, const std::vector<pops::MultiFab<Dim>*>& canonical_states) {
              if (canonical_states.size() != 2)
                throw std::runtime_error("coupled source lost its complete canonical state pack");
              auto& donor = *canonical_states[0];
              auto& receiver = *canonical_states[1];
              for (std::size_t local = 0; local < donor.local_size(); ++local)
                pops::for_each_cell(donor.box(local),
                                    ConservativeExchange<Dim>{donor.fab(local).view(),
                                                              receiver.fab(local).view(), dt});
              Kokkos::fence();
              if (*violate && pops::my_rank() == 0 && donor.local_size() != 0) {
                const auto values = donor.fab(0).view();
                const pops::Index<Dim> cell = donor.box(0).lo;
                pops::for_each_cell(pops::Box<Dim>{cell, cell},
                                    [=] POPS_HD(const pops::Index<Dim>& index) {
                                      values(index, 0) += pops::Real(1);
                                    });
                Kokkos::fence();
              }
              if (*fail && pops::my_rank() == 0)
                throw std::runtime_error("injected rank-local coupled-source failure");
            },
            std::move(conservation)));

    pops::Index<Dim> fine_lower{};
    pops::Index<Dim> fine_upper{};
    pops::Extent<Dim> fine_shape{};
    for (int axis = 0; axis < Dim; ++axis) {
      fine_lower[axis] = config().shape[axis] / 2;
      fine_upper[axis] = 3 * config().shape[axis] / 2 - 1;
      fine_shape[axis] = 2 * config().shape[axis];
    }
    system.rebuild_hierarchy({pops::AmrPatch<Dim>{1, {fine_lower, fine_upper}}}, {0});
    system.set_block_level_state("donor", 1, smooth_state(fine_shape, 2.0, 0.2));
    system.set_block_level_state("receiver", 1, smooth_state(fine_shape, 5.0, 0.1));

    context = pops::runtime::program::make_program_execution_provider(&system);
    context->configure_primary_clock("test.clock.macro");
    context->install(
        [context = context,
         cross_owner_projection_scratch = cross_owner_projection_scratch](double macro_dt) {
          context->advance_hierarchy(macro_dt, [context,
                                                cross_owner_projection_scratch](double level_dt) {
            std::array<pops::MultiFab<Dim>*, 2> accepted{};
            std::array<pops::MultiFab<Dim>*, 2> candidates{};
            for (int program_block = 0; program_block < 2; ++program_block) {
              accepted[static_cast<std::size_t>(program_block)] = &context->state(program_block);
              auto& stage =
                  context->scratch_state(1000 + program_block, 0, *accepted[program_block]);
              auto& rate_zero =
                  context->rhs_scratch(2000 + program_block, 0, *accepted[program_block]);
              auto& rate_one =
                  context->rhs_scratch(3000 + program_block, 0, *accepted[program_block]);
              auto& candidate =
                  context->scratch_state(4000 + program_block, 0, *accepted[program_block]);

              context->set_stage_time(0, 1);
              context->neg_div_flux_default_into(program_block, *accepted[program_block], rate_zero,
                                                 5000 + 2 * program_block);
              context->lincomb(stage, pops::Real(1), *accepted[program_block], pops::Real(0),
                               *accepted[program_block], pops::Real(level_dt), {{0, 1, 1}},
                               {{0, 0, 1}});
              context->axpy(stage, pops::Real(level_dt), rate_zero, pops::Real(level_dt),
                            {{1, 1, 1}});

              context->set_stage_time(1, 1);
              context->neg_div_flux_default_into(program_block, stage, rate_one,
                                                 5001 + 2 * program_block);
              context->lincomb(candidate, pops::Real(0.5), *accepted[program_block],
                               pops::Real(0.5), stage, pops::Real(level_dt), {{0, 1, 2}},
                               {{0, 1, 2}});
              context->axpy(candidate, pops::Real(0.5 * level_dt), rate_one, pops::Real(level_dt),
                            {{1, 1, 2}});

              candidates[static_cast<std::size_t>(program_block)] = &candidate;
            }
            if (*cross_owner_projection_scratch)
              context->apply_projection(1, *candidates[0]);
            for (int program_block = 0; program_block < 2; ++program_block)
              context->apply_projection(program_block,
                                        *candidates[static_cast<std::size_t>(program_block)]);
            context->apply_coupling_operators(
                "test.amr.multiblock-coupled-source/program-ssprk2-imex-v1",
                "test.amr.multiblock-coupled-source/rate-final",
                "test.amr.multiblock-coupled-source/application-final", pops::Real(level_dt),
                {{0, candidates[0]}, {1, candidates[1]}});
            context->commit_many({{accepted[0], candidates[0]}, {accepted[1], candidates[1]}});
          });
        },
        context);
    // Program order is intentionally the reverse of canonical runtime ownership.  The prepared
    // ProgramBlockMap must route candidates by block identity; the provider never sees this order.
    system.set_program_block_map({1, 0});
    using FluxBudget = typename pops::AmrSystem<Dim>::PreparedAmrProgramFluxExpressionBlockBudget;
    constexpr std::string_view graph_identity =
        "test.amr.multiblock-coupled-source/program-ssprk2-imex-v1";
    constexpr std::string_view rate_identity = "test.amr.multiblock-coupled-source/rate-final";
    constexpr std::string_view application_identity =
        "test.amr.multiblock-coupled-source/application-final";
    system.install_prepared_amr_program_flux_expression_budget(
        std::string(graph_identity), std::vector<FluxBudget>{{2, 1}, {2, 1}}, 1,
        graph_identity.size() + rate_identity.size() + application_identity.size());
  }

  static pops::AmrSystemConfig<Dim> config() {
    pops::AmrSystemConfig<Dim> result;
    result.level_count = 2;
    result.regrid_every = 0;
    result.distribute_coarse = true;
    result.transition_ratios.resize(1);
    result.transition_buffers.resize(1);
    result.transition_lookaheads.resize(1);
    for (int axis = 0; axis < Dim; ++axis) {
      result.shape[axis] = 8;
      result.lower[axis] = pops::Real(0);
      result.upper[axis] = pops::Real(1);
      result.periodicity[axis] = true;
      result.coarse_max_grid[axis] = 4;
      result.transition_ratios[0][axis] = 2;
      result.transition_buffers[0][axis] = 1;
      result.transition_lookaheads[0][axis] = 1;
    }
    return result;
  }
};

template <int Dim>
void prove_conservative_execution_rollback_and_retry() {
  CoupledFixture<Dim> fixture;
  ASSERT_EQ(fixture.system.n_levels(), 2);
  const BlockLevelSnapshot<Dim> initial = snapshot_block_levels(fixture.system);
  const double total_before = fixture.system.mass("donor") + fixture.system.mass("receiver");

  // The rank-zero request intentionally offers the other block's accepted carrier.  The facade
  // must converge that rank-local preflight failure before any rank enters the projection closure;
  // both accepted blocks remain byte-exact and a clean detached retry is still executable.
  const pops::MultiFab<Dim>& selected_live = fixture.system.prepared_amr_block_state(0, 0);
  pops::MultiFab<Dim> detached(selected_live.layout(), selected_live.distribution(),
                               selected_live.local_rank(), selected_live.ncomp(),
                               selected_live.ghosts());
  detached.set_val(pops::Real(0));
  const auto cross_owner_live_projection = [&] {
    if (pops::my_rank() == 0)
      fixture.system.project_prepared_amr_block_level_state(
          0, 0, 0, fixture.system.prepared_amr_block_state(1, 0));
    else
      fixture.system.project_prepared_amr_block_level_state(0, 0, 0, detached);
  };
  if (pops::n_ranks() == 1)
    EXPECT_THROW(cross_owner_live_projection(), std::invalid_argument);
  else
    EXPECT_THROW(cross_owner_live_projection(), std::runtime_error);
  EXPECT_TRUE(byte_exact_block_levels(initial, fixture.system));
  EXPECT_NO_THROW(fixture.system.project_prepared_amr_block_level_state(0, 0, 0, detached));
  EXPECT_TRUE(byte_exact_block_levels(initial, fixture.system));

  fixture.system.step(0.4);
  for (int level = 0; level < fixture.system.n_levels(); ++level) {
    EXPECT_LT(pops::reduce_max(fixture.system.prepared_amr_block_state(0, level)),
              pops::reduce_max(initial[0][static_cast<std::size_t>(level)]));
    EXPECT_GT(pops::reduce_min(fixture.system.prepared_amr_block_state(1, level)),
              pops::reduce_min(initial[1][static_cast<std::size_t>(level)]));
  }
  EXPECT_NEAR(fixture.system.mass("donor") + fixture.system.mass("receiver"), total_before,
              2.0e-11);

  const BlockLevelSnapshot<Dim> rollback = snapshot_block_levels(fixture.system);
  *fixture.cross_owner_projection_scratch = true;
  if (pops::n_ranks() == 1)
    EXPECT_THROW(fixture.system.step(0.2), std::invalid_argument);
  else
    EXPECT_THROW(fixture.system.step(0.2), std::runtime_error);
  EXPECT_TRUE(byte_exact_block_levels(rollback, fixture.system));

  *fixture.cross_owner_projection_scratch = false;
  *fixture.violate_conservation_on_rank_zero = true;
  EXPECT_THROW(fixture.system.step(0.2), std::runtime_error);
  EXPECT_TRUE(byte_exact_block_levels(rollback, fixture.system));

  *fixture.violate_conservation_on_rank_zero = false;
  *fixture.fail_on_rank_zero = true;
  EXPECT_THROW(fixture.system.step(0.2), std::runtime_error);
  EXPECT_TRUE(byte_exact_block_levels(rollback, fixture.system));

  *fixture.fail_on_rank_zero = false;
  EXPECT_NO_THROW(fixture.system.step(0.2));
  EXPECT_FALSE(byte_exact_block_levels(rollback, fixture.system));
  EXPECT_NEAR(fixture.system.mass("donor") + fixture.system.mass("receiver"), total_before,
              2.0e-11);
}

}  // namespace

TEST(test_amr_multiblock_coupled_source, ConservativeExecutionRollbackAndRetry) {
#if defined(POPS_HAS_KOKKOS)
  std::unique_ptr<Kokkos::ScopeGuard> kokkos;
  if (!Kokkos::is_initialized()) {
    int argc = 0;
    char** argv = nullptr;
    kokkos = std::make_unique<Kokkos::ScopeGuard>(argc, argv);
  }
#endif
  prove_conservative_execution_rollback_and_retry<pops::kNativeDimension>();
}
