#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/program_execution_services.hpp>

#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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
  pops::Real strength = pops::Real(0);

  POPS_HD void operator()(const pops::Index<Dim>& cell) const {
    const pops::Real amount = dt * strength * donor(cell, 0);
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
  const long local_levels = system.n_levels();
  const long maximum_levels = pops::all_reduce_max(local_levels);
  const long minimum_levels = -pops::all_reduce_max(-local_levels);
  if (minimum_levels != maximum_levels)
    throw std::runtime_error("AMR accepted block level count diverged across ranks");
  for (int runtime_block = 0; runtime_block < 2; ++runtime_block) {
    auto& levels = result[static_cast<std::size_t>(runtime_block)];
    levels.reserve(static_cast<std::size_t>(maximum_levels));
    for (int level = 0; level < maximum_levels; ++level) {
      auto view = system.prepared_amr_block_state(runtime_block, level);
      if (pops::all_reduce_max(view ? 0L : 1L) != 0)
        throw std::runtime_error("AMR accepted block view is invalid collectively");
      levels.emplace_back(*view);
    }
  }
  return result;
}

template <int Dim>
bool byte_exact_block_levels(const BlockLevelSnapshot<Dim>& expected,
                             pops::AmrSystem<Dim>& actual) {
  bool same = true;
  const long local_levels = actual.n_levels();
  const long maximum_levels = pops::all_reduce_max(local_levels);
  const long minimum_levels = -pops::all_reduce_max(-local_levels);
  const bool local_shape_matches =
      minimum_levels == maximum_levels &&
      std::all_of(expected.begin(), expected.end(), [maximum_levels](const auto& levels) {
        return levels.size() == static_cast<std::size_t>(maximum_levels);
      });
  if (pops::all_reduce_max(local_shape_matches ? 0L : 1L) != 0)
    return false;
  for (int runtime_block = 0; runtime_block < 2; ++runtime_block)
    for (int level = 0; level < maximum_levels; ++level) {
      auto view = actual.prepared_amr_block_state(runtime_block, level);
      if (pops::all_reduce_max(view ? 0L : 1L) != 0) {
        same = false;
        continue;
      }
      same = same &&
             byte_exact_equal(
                 expected[static_cast<std::size_t>(runtime_block)][static_cast<std::size_t>(level)],
                 *view);
    }
  return same;
}

template <int Dim>
struct CoupledFixture {
  static constexpr pops::Real kExchangeStrength = pops::Real(0.25);

  pops::AmrSystem<Dim> system;
  std::shared_ptr<bool> fail_on_rank_zero = std::make_shared<bool>(false);
  std::shared_ptr<bool> violate_conservation_on_rank_zero = std::make_shared<bool>(false);
  std::shared_ptr<bool> cross_owner_projection_scratch = std::make_shared<bool>(false);

  explicit CoupledFixture(pops::Real exchange_strength) : system(config()) {
    pops::test::install_amr_runtime_authority(system,
                                              "test.amr.multiblock-coupled-source/runtime@1");
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
            [exchange_strength, fail = fail_on_rank_zero,
             violate = violate_conservation_on_rank_zero](
                pops::Real dt, const std::vector<pops::MultiFab<Dim>*>& canonical_states) {
              if (canonical_states.size() != 2)
                throw std::runtime_error("coupled source lost its complete canonical state pack");
              auto& donor = *canonical_states[0];
              auto& receiver = *canonical_states[1];
              for (std::size_t local = 0; local < donor.local_size(); ++local)
                pops::for_each_cell(
                    donor.box(local),
                    ConservativeExchange<Dim>{donor.fab(local).view(), receiver.fab(local).view(),
                                              dt, exchange_strength});
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

    // Program order is intentionally the reverse of canonical runtime ownership.  The detached
    // v5 plan routes every dense resource by block identity; no accepted facade escapes dispatch.
    using Resource = pops::test::program_v5::CallbackProgramResource;
    using FluxBasis = pops::test::program_v5::CallbackProgramFluxBasisOccurrence;
    using FluxStage = pops::test::program_v5::CallbackProgramFaceFluxStage;
    using FluxBudget = pops::runtime::program::ProgramFluxBudgetRecord;
    constexpr std::string_view kProgramIdentity =
        "test.amr.multiblock-coupled-source/program-ssprk2-imex-v1";
    constexpr std::string_view kRateIdentity = "test.amr.multiblock-coupled-source/rate-final";
    constexpr std::string_view kApplicationIdentity =
        "test.amr.multiblock-coupled-source/application-final";
    std::vector<Resource> resources;
    resources.reserve(8);
    for (int program_block = 0; program_block < 2; ++program_block) {
      const int runtime_block = program_block == 0 ? 1 : 0;
      const auto state = system.prepared_amr_block_state(runtime_block, 0);
      if (!state)
        throw std::logic_error("coupled-source Program resource has no accepted block state");
      const std::string_view owner = program_block == 0 ? "receiver" : "donor";
      const auto append_resource = [&](Resource::Kind kind, std::uint64_t value_id) {
        Resource resource{kind,
                          resources.size(),
                          0,
                          program_block,
                          -1,
                          static_cast<std::uint32_t>(state->ncomp()),
                          static_cast<std::uint32_t>(state->ghosts()[0])};
        resource.value_id = value_id;
        resource.identity =
            std::string(kProgramIdentity) + "/resource/" + std::to_string(resource.slot);
        resource.occurrence_path = "root/resource/" + std::to_string(resource.slot);
        resource.owner = owner;
        resource.clock = "test.clock.macro";
        resources.push_back(std::move(resource));
      };
      append_resource(Resource::Kind::state, 0);
      append_resource(Resource::Kind::rhs, static_cast<std::uint64_t>(5000 + 2 * program_block));
      append_resource(Resource::Kind::rhs, static_cast<std::uint64_t>(5001 + 2 * program_block));
      append_resource(Resource::Kind::state, 0);
    }
    const std::optional<std::vector<FluxBudget>> flux_budgets{std::vector<FluxBudget>{
        {2, 2, 1, kProgramIdentity.size() + kRateIdentity.size() + kApplicationIdentity.size()},
        {2, 2, 1, kProgramIdentity.size() + kRateIdentity.size() + kApplicationIdentity.size()}}};
    const auto basis = [](std::uint32_t basis_slot, std::uint32_t expression_slot,
                          int program_block, int rhs_identity, std::int64_t stage_numerator,
                          std::string_view owner) {
      return FluxBasis{basis_slot,
                       expression_slot,
                       program_block,
                       -1,
                       rhs_identity,
                       pops::test::program_v5::kPreparedDefaultFluxProvider,
                       stage_numerator,
                       1,
                       "test.amr.multiblock-coupled-source/basis/" + std::to_string(basis_slot),
                       "root/basis/" + std::to_string(basis_slot),
                       std::string(owner),
                       "test.clock.macro"};
    };
    const std::vector<FluxBasis> flux_bases{
        basis(0, 1, 0, 5000, 0, "receiver"), basis(1, 2, 0, 5001, 1, "receiver"),
        basis(2, 5, 1, 5002, 0, "donor"), basis(3, 6, 1, 5003, 1, "donor")};
    const auto final_stage = [](std::uint32_t slot, std::uint32_t basis_slot,
                                std::uint32_t expression_slot, std::string_view owner) {
      return FluxStage{slot,
                       basis_slot,
                       expression_slot,
                       1,
                       1,
                       2,
                       "test.amr.multiblock-coupled-source/final-flux/" + std::to_string(slot),
                       "root/final-flux/" + std::to_string(slot),
                       std::string(owner),
                       "test.clock.macro"};
    };
    const std::vector<FluxStage> flux_stages{
        final_stage(0, 0, 1, "receiver"), final_stage(1, 1, 2, "receiver"),
        final_stage(2, 2, 5, "donor"), final_stage(3, 3, 6, "donor")};
    const auto projection_scratch = cross_owner_projection_scratch;
    pops::test::install_explicit_amr_callback_program<Dim>(
        system, kProgramIdentity, "test.clock.macro", std::vector<std::string>{"receiver", "donor"},
        resources, {},
        [projection_scratch, kProgramIdentity, kRateIdentity, kApplicationIdentity](
            auto& context, double macro_dt) {
          context.advance_hierarchy(
              macro_dt, [&context, projection_scratch, kProgramIdentity, kRateIdentity,
                         kApplicationIdentity](double level_dt) {
                using ExactCoefficientTerm = pops::runtime::program::ExactCoefficientTerm;
                std::array<pops::MultiFab<Dim>*, 2> accepted{};
                std::array<pops::MultiFab<Dim>*, 2> candidates{};
                for (int program_block = 0; program_block < 2; ++program_block) {
                  accepted[static_cast<std::size_t>(program_block)] = &context.state(program_block);
                  const auto base =
                      static_cast<pops::runtime::program::ProgramCacheSlot>(program_block * 4);
                  auto& stage = context.scratch_state(base, 0, *accepted[program_block]);
                  auto& rate_zero = context.rhs_scratch(base + 1, 0, *accepted[program_block]);
                  auto& rate_one = context.rhs_scratch(base + 2, 0, *accepted[program_block]);
                  auto& candidate = context.scratch_state(base + 3, 0, *accepted[program_block]);

                  context.set_stage_time(0, 1);
                  context.neg_div_flux_default_into(program_block, *accepted[program_block],
                                                    rate_zero, 5000 + 2 * program_block);
                  context.lincomb(stage, pops::Real(1), *accepted[program_block], pops::Real(0),
                                  *accepted[program_block], pops::Real(level_dt),
                                  std::initializer_list<ExactCoefficientTerm>{{0, 1, 1}},
                                  std::initializer_list<ExactCoefficientTerm>{{0, 0, 1}});
                  context.axpy(stage, pops::Real(level_dt), rate_zero, pops::Real(level_dt),
                               {{1, 1, 1}});

                  context.set_stage_time(1, 1);
                  context.neg_div_flux_default_into(program_block, stage, rate_one,
                                                    5001 + 2 * program_block);
                  context.lincomb(candidate, pops::Real(0.5), *accepted[program_block],
                                  pops::Real(0.5), stage, pops::Real(level_dt),
                                  std::initializer_list<ExactCoefficientTerm>{{0, 1, 2}},
                                  std::initializer_list<ExactCoefficientTerm>{{0, 1, 2}});
                  context.axpy(candidate, pops::Real(0.5 * level_dt), rate_one,
                               pops::Real(level_dt), {{1, 1, 2}});
                  candidates[static_cast<std::size_t>(program_block)] = &candidate;
                }
                if (*projection_scratch)
                  context.apply_projection(1, *candidates[0]);
                for (int program_block = 0; program_block < 2; ++program_block)
                  context.apply_projection(program_block,
                                           *candidates[static_cast<std::size_t>(program_block)]);
                context.apply_coupling_operators(kProgramIdentity, kRateIdentity,
                                                 kApplicationIdentity, pops::Real(level_dt),
                                                 {{0, candidates[0]}, {1, candidates[1]}});
                context.commit_many({{accepted[0], candidates[0]}, {accepted[1], candidates[1]}});
              });
        },
        {}, {}, flux_budgets, {}, std::nullopt, flux_bases, flux_stages);
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
  CoupledFixture<Dim> fixture(CoupledFixture<Dim>::kExchangeStrength);
  CoupledFixture<Dim> control(pops::Real(0));
  ASSERT_EQ(fixture.system.n_levels(), 2);
  const BlockLevelSnapshot<Dim> initial = snapshot_block_levels(fixture.system);
  const double total_before = fixture.system.mass("donor") + fixture.system.mass("receiver");

  // The rank-zero request claims the other block as the owner of a detached candidate.  The facade
  // must converge that rank-local preflight failure before any rank enters the projection closure;
  // both accepted blocks remain byte-exact and a clean detached retry is still executable.
  pops::MultiFab<Dim> detached = [&fixture] {
    const auto selected_live_view = fixture.system.prepared_amr_block_state(0, 0);
    if (!selected_live_view)
      throw std::logic_error("coupled-source test has no accepted donor state");
    const pops::MultiFab<Dim>& selected_live = *selected_live_view;
    pops::MultiFab<Dim> candidate(selected_live.layout(), selected_live.distribution(),
                                  selected_live.local_rank(), selected_live.ncomp(),
                                  selected_live.ghosts());
    candidate.set_val(pops::Real(0));
    return candidate;
  }();
  const auto cross_owner_live_projection = [&] {
    if (pops::my_rank() == 0) {
      fixture.system.project_prepared_amr_block_level_state(0, 0, 1, detached);
    } else {
      fixture.system.project_prepared_amr_block_level_state(0, 0, 0, detached);
    }
  };
  if (pops::n_ranks() == 1)
    EXPECT_THROW(cross_owner_live_projection(), std::invalid_argument);
  else
    EXPECT_THROW(cross_owner_live_projection(), std::runtime_error);
  EXPECT_TRUE(byte_exact_block_levels(initial, fixture.system));
  EXPECT_NO_THROW(fixture.system.project_prepared_amr_block_level_state(0, 0, 0, detached));
  EXPECT_TRUE(byte_exact_block_levels(initial, fixture.system));

  fixture.system.step(0.4);
  control.system.step(0.4);
  EXPECT_LT(fixture.system.mass("donor"), control.system.mass("donor"));
  EXPECT_GT(fixture.system.mass("receiver"), control.system.mass("receiver"));
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
