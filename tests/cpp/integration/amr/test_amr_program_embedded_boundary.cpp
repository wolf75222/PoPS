// End-to-end AMR Program embedded-boundary ownership and publication regression.
//
// The facade is native-ranked, so this suite intentionally instantiates only
// kNativeDimension.  Each mode materializes a sparse two-level hierarchy and exercises the
// authenticated ProgramContext rather than a ModelSpec or a synthetic communicator.

#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace {

template <int Dim>
struct ZeroAdvectionModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;

  Law law{};

  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"tests.amr.program-eb.zero-advection", 1};
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
  POPS_HD pops::nd::StateConversion<State> make_conservative(const Primitive& value) const {
    return law.make_conservative(value);
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
pops::AmrSystemConfig<Dim> two_level_config() {
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 2;
  config.regrid_every = 0;
  config.explicit_bootstrap = true;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 8;
    config.lower[axis] = pops::Real(0);
    config.upper[axis] = pops::Real(1);
    config.periodicity[axis] = true;
  }
  return config;
}

template <int Dim>
std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

template <int Dim>
pops::amr::tagging::ClusterResult<Dim> centered_cluster(
    const pops::amr::hierarchy::LevelLayout<Dim>& parent) {
  pops::Index<Dim> lower{};
  pops::Index<Dim> upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = parent.domain().lo[axis] + 2;
    upper[axis] = parent.domain().hi[axis] - 2;
  }
  const pops::mesh::BoxArray<Dim> boxes(std::vector<pops::Box<Dim>>{pops::Box<Dim>{lower, upper}});
  pops::amr::tagging::ClusterOptions<Dim> options;
  options.min_efficiency = 0.7;
  for (int axis = 0; axis < Dim; ++axis) {
    options.min_box_size[static_cast<std::size_t>(axis)] = 1;
    options.max_box_size[static_cast<std::size_t>(axis)] = 16;
  }
  options.budget = {16, 256, 8192, 64, 1U << 20};
  pops::amr::tagging::ClusterResultIdentity<Dim> identity{
      "tests.amr.program-eb.centered-cluster", parent.exact_identity(), options, {}, boxes.boxes()};
  return {boxes, std::move(identity)};
}

template <int Dim>
void materialize_centered_fine_level(pops::AmrSystem<Dim>& system) {
  auto* engine = system.engine();
  ASSERT_NE(engine, nullptr);
  std::array<int, Dim> ratio_components{};
  ratio_components.fill(2);
  const pops::amr::RefinementRatio<Dim> ratio(ratio_components);
  const pops::amr::regridding::RegridPreparationBudget budget{
      .clustered_parent_layout = {16, 120},
      .fine_layout = {16, 120},
      .load_balance = {16, 16, std::numeric_limits<std::int64_t>::max()},
  };
  auto prepared =
      engine->prepare_regrid(0, ratio, centered_cluster(engine->hierarchy().layout(0)), budget);
  ASSERT_FALSE(prepared.removes_fine_level());
  ASSERT_TRUE(prepared.fine_layout().has_value());
  pops::MultiFab<Dim> child(
      prepared.fine_layout()->patches(), prepared.fine_layout()->distribution(),
      engine->hierarchy().state(0).local_rank(), engine->hierarchy().state(0).ncomp(),
      engine->hierarchy().state(0).ghosts());
  child.set_val(pops::Real(1));
  engine->publish_regrid(0, std::move(prepared), std::move(child));
  system.refresh_prepared_amr_levels();
}

template <int Dim>
POPS_HD bool contains(const pops::Box<Dim>& box, const pops::Index<Dim>& cell) {
  for (int axis = 0; axis < Dim; ++axis)
    if (cell[axis] < box.lo[axis] || cell[axis] > box.hi[axis])
      return false;
  return true;
}

template <int Dim>
void seed_inactive_and_ghosts(pops::MultiFab<Dim>& field, const pops::MultiFab<Dim>& active,
                              pops::Real inactive_value, pops::Real ghost_value) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const pops::Box<Dim> valid = field.fab(local).box();
    const auto output = field.fab(local).view();
    const auto mask = std::as_const(active.fab(local)).view();
    pops::for_each_cell(field.fab(local).grown_box(), [=] POPS_HD(const pops::Index<Dim>& cell) {
      if (!contains(valid, cell))
        output(cell, 0) = ghost_value;
      else if (mask(cell, 0) < pops::Real(0.5))
        output(cell, 0) = inactive_value;
    });
  }
  pops::device_fence();
}

template <int Dim>
void set_active_valid(pops::MultiFab<Dim>& field, const pops::MultiFab<Dim>& active,
                      pops::Real value) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto output = field.fab(local).view();
    const auto mask = std::as_const(active.fab(local)).view();
    pops::for_each_cell(field.fab(local).box(), [=] POPS_HD(const pops::Index<Dim>& cell) {
      if (mask(cell, 0) >= pops::Real(0.5))
        output(cell, 0) = value;
    });
  }
  pops::device_fence();
}

template <int Dim>
void poison_inactive_and_ghosts(pops::MultiFab<Dim>& field, const pops::MultiFab<Dim>& active) {
  const pops::Real nan = std::numeric_limits<pops::Real>::quiet_NaN();
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const pops::Box<Dim> valid = field.fab(local).box();
    const auto output = field.fab(local).view();
    const auto mask = std::as_const(active.fab(local)).view();
    pops::for_each_cell(field.fab(local).grown_box(), [=] POPS_HD(const pops::Index<Dim>& cell) {
      if (!contains(valid, cell) || mask(cell, 0) < pops::Real(0.5))
        output(cell, 0) = nan;
    });
  }
  pops::device_fence();
}

template <int Dim>
void write_status(pops::MultiFab<Dim>& status, const pops::MultiFab<Dim>& active,
                  pops::Real active_value, pops::Real inactive_value) {
  for (std::size_t local = 0; local < status.local_size(); ++local) {
    const auto output = status.fab(local).view();
    const auto mask = std::as_const(active.fab(local)).view();
    pops::for_each_cell(status.fab(local).box(), [=] POPS_HD(const pops::Index<Dim>& cell) {
      output(cell, 0) = mask(cell, 0) >= pops::Real(0.5) ? active_value : inactive_value;
    });
  }
  pops::device_fence();
}

template <int Dim>
bool inactive_valid_and_ghosts_equal(const pops::MultiFab<Dim>& current,
                                     const pops::MultiFab<Dim>& before,
                                     const pops::MultiFab<Dim>& active) {
  pops::Real mismatches = pops::Real(0);
  for (std::size_t local = 0; local < current.local_size(); ++local) {
    const pops::Box<Dim> valid = current.fab(local).box();
    const auto now = std::as_const(current.fab(local)).view();
    const auto prior = std::as_const(before.fab(local)).view();
    const auto mask = std::as_const(active.fab(local)).view();
    mismatches += pops::for_each_cell_reduce_sum(
        current.fab(local).grown_box(), [=] POPS_HD(const pops::Index<Dim>& cell) -> pops::Real {
          if (contains(valid, cell) && mask(cell, 0) >= pops::Real(0.5))
            return pops::Real(0);
          return now(cell, 0) == prior(cell, 0) ? pops::Real(0) : pops::Real(1);
        });
  }
  return pops::all_reduce_sum(mismatches) == pops::Real(0);
}

template <int Dim>
void prove_program_embedded_boundary_contract(const std::string& mode) {
  const auto config = two_level_config<Dim>();
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system, "tests.amr.program-eb/" + mode + "/runtime@1");
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  system.install_block_state_route("tracer", "tests.amr.program-eb/state/tracer");
  pops::RealVector<Dim> velocity{};
  pops::add_compiled_model<Dim>(
      system, "tracer", ZeroAdvectionModel<Dim>{pops::nd::ScalarAdvection<Dim>::prepare(velocity)},
      "minmod", "rusanov", "conservative", "explicit", 1.4, 1, 1, {}, {}, 0.0,
      static_cast<double>(pops::kWenoEpsilon), false,
      "tests.amr.program-eb/" + mode + "/physical-flux");
  system.set_analytic_level_set({"x", "constant", "sub"}, {0.0, 0.5, 0.0}, mode);
  system.set_conservative_state("tracer", std::vector<double>(cell_count(config.shape), 1.0));
  materialize_centered_fine_level(system);
  ASSERT_EQ(system.n_levels(), 2) << mode;

  const auto context = pops::test::install_forward_euler_program_context(system, false);
  std::vector<const pops::MultiFab<Dim>*> active(2);
  for (int level = 0; level < 2; ++level) {
    active[static_cast<std::size_t>(level)] = system.prepared_amr_block_level_active_mask(0, level);
    ASSERT_NE(active[static_cast<std::size_t>(level)], nullptr) << mode << " level=" << level;
    const pops::Real active_cells =
        pops::all_reduce_sum(pops::reduce_sum_local(*active[static_cast<std::size_t>(level)], 0));
    pops::Real valid_cells = pops::Real(0);
    const auto& state = system.prepared_amr_block_state(0, level);
    for (std::size_t local = 0; local < state.local_size(); ++local)
      valid_cells += static_cast<pops::Real>(state.fab(local).box().numPts());
    valid_cells = pops::all_reduce_sum(valid_cells);
    EXPECT_GT(active_cells, pops::Real(0)) << mode << " level=" << level;
    EXPECT_LT(active_cells, valid_cells) << mode << " level=" << level;
  }

  // The state seed is an authenticated Program owner.  A NaN in every inactive valid cell must
  // not poison any raw all-level Program reduction.
  pops::Real active_count = pops::Real(0);
  for (int level = 0; level < 2; ++level) {
    auto& state = system.prepared_amr_block_state(0, level);
    state.set_val(pops::Real(1));
    poison_inactive_and_ghosts(state, *active[static_cast<std::size_t>(level)]);
    active_count +=
        pops::all_reduce_sum(pops::reduce_sum_local(*active[static_cast<std::size_t>(level)], 0));
  }
  context->with_program_resource_level(0, [&] {
    auto& state = context->state(0);
    EXPECT_EQ(context->sum_component(0, state, 0), active_count) << mode;
    EXPECT_EQ(context->abs_sum_component(0, state, 0), active_count) << mode;
    EXPECT_EQ(context->min_component(0, state, 0), pops::Real(1)) << mode;
    EXPECT_EQ(context->max_component(0, state, 0), pops::Real(1)) << mode;
    EXPECT_EQ(context->norm2(0, state), std::sqrt(active_count)) << mode;
  });

  // Scratch ownership is also level-aware.  Build one status carrier per live level, then reduce
  // through its level-zero owner so an inactive NaN on either level is explicitly ignored.
  std::vector<pops::MultiFab<Dim>*> statuses(2);
  std::vector<const pops::MultiFab<Dim>*> status_masks(2);
  context->for_each_program_resource_level([&](int level) {
    auto& state = context->state(0);
    auto& status = context->scalar_scratch(920, 0, state, 1, 0);
    const auto* mask = context->pointwise_active_mask(0, status);
    ASSERT_NE(mask, nullptr) << mode << " level=" << level;
    statuses[static_cast<std::size_t>(level)] = &status;
    status_masks[static_cast<std::size_t>(level)] = mask;
    write_status(status, *mask, pops::Real(0), std::numeric_limits<pops::Real>::quiet_NaN());
  });
  EXPECT_EQ(context->pointwise_status_max(0, *statuses[0], status_masks[0],
                                          context->prepared_execution_lane()),
            pops::Real(0))
      << mode;
  write_status(*statuses[1], *status_masks[1], std::numeric_limits<pops::Real>::quiet_NaN(),
               pops::Real(0));
  EXPECT_EQ(context->pointwise_status_max(0, *statuses[0], status_masks[0],
                                          context->prepared_execution_lane()),
            pops::Real(3))
      << mode;

  // Restore finite accepted state, then give inactive valid cells and all allocated ghosts unique
  // sentinels.  A terminal direct facade publication may change only active valid cells.
  std::vector<pops::MultiFab<Dim>> accepted_before;
  accepted_before.reserve(2);
  for (int level = 0; level < 2; ++level) {
    auto& accepted = system.prepared_amr_block_state(0, level);
    accepted.set_val(pops::Real(1));
    seed_inactive_and_ghosts(accepted, *active[static_cast<std::size_t>(level)],
                             pops::Real(31 + level), pops::Real(71 + level));
    accepted_before.emplace_back(accepted);

    pops::MultiFab<Dim> candidate(accepted);
    pops::MultiFab<Dim> expected(accepted);
    set_active_valid(candidate, *active[static_cast<std::size_t>(level)], pops::Real(5 + level));
    poison_inactive_and_ghosts(candidate, *active[static_cast<std::size_t>(level)]);
    set_active_valid(expected, *active[static_cast<std::size_t>(level)], pops::Real(5 + level));
    std::vector<pops::MultiFab<Dim>*> pack{&candidate};
    system.publish_prepared_amr_program_candidates(level, pack);
    EXPECT_EQ(pops::difference_sum_sq_all(system.prepared_amr_block_state(0, level), expected),
              pops::Real(0))
        << mode << " level=" << level;
  }

  // One bad active value in one live level rejects the whole publication attempt.  No accepted
  // level is changed, which is the fail-closed boundary before a subcycle can commit it.
  std::vector<pops::MultiFab<Dim>> before_rejection;
  before_rejection.emplace_back(system.prepared_amr_block_state(0, 0));
  before_rejection.emplace_back(system.prepared_amr_block_state(0, 1));
  pops::MultiFab<Dim> rejected_candidate(system.prepared_amr_block_state(0, 1));
  set_active_valid(rejected_candidate, *active[1], std::numeric_limits<pops::Real>::quiet_NaN());
  std::vector<pops::MultiFab<Dim>*> rejected_pack{&rejected_candidate};
  EXPECT_ANY_THROW(system.publish_prepared_amr_program_candidates(1, rejected_pack)) << mode;
  for (int level = 0; level < 2; ++level)
    EXPECT_EQ(pops::difference_sum_sq_all(system.prepared_amr_block_state(0, level),
                                          before_rejection[static_cast<std::size_t>(level)]),
              pops::Real(0))
        << mode << " level=" << level;

  // advance_hierarchy() routes the exact same terminal publication through the subcycling engine.
  // It must retain the inactive sentinels and ghosts seeded above, not merely the direct facade.
  system.step(1.0e-4);
  for (int level = 0; level < 2; ++level)
    EXPECT_TRUE(inactive_valid_and_ghosts_equal(system.prepared_amr_block_state(0, level),
                                                before_rejection[static_cast<std::size_t>(level)],
                                                *active[static_cast<std::size_t>(level)]))
        << mode << " level=" << level;
}

TEST(test_amr_program_embedded_boundary,
     NativeRankedStaircaseAndCutCellPreserveInactiveCellsAcrossDirectAndSubcycledPublication) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  constexpr int Dim = pops::kNativeDimension;
  prove_program_embedded_boundary_contract<Dim>("staircase");
  prove_program_embedded_boundary_contract<Dim>("cutcell");
}

}  // namespace
