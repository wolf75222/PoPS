// End-to-end AMR Program embedded-boundary ownership and publication regression.
//
// The facade is native-ranked, so this suite intentionally instantiates only
// kNativeDimension.  Each mode materializes a sparse two-level hierarchy and exercises the
// authenticated ProgramExecutionServices rather than a ModelSpec or a synthetic communicator.

#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/program_execution_services.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
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
  auto* engine = pops::test::AmrSystemTestAccess<Dim>::engine(system);
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
  const int level_count = system.n_levels();

  using Context = pops::runtime::program::ProgramExecutionServices<Dim>;
  using Field = pops::MultiFab<Dim>;
  using Resource = pops::test::program_v5::CallbackProgramResource;
  enum class Operation : std::uint8_t { seed, status, publish, reject, advance };
  struct Command final {
    Operation operation = Operation::seed;
  };
  struct Result final {
    bool invoked = false;
    bool threw = false;
    pops::Real active_count = pops::Real(0);
    pops::Real state_sum = pops::Real(0);
    pops::Real state_abs_sum = pops::Real(0);
    pops::Real state_min = pops::Real(0);
    pops::Real state_max = pops::Real(0);
    pops::Real state_norm2 = pops::Real(0);
    pops::Real status_max_before = pops::Real(0);
    pops::Real status_max_after = pops::Real(0);
  };

  // Copy masks before installing/stepping the MODULE.  Holding an AcceptedReadLease across a
  // candidate step would block the writer that publishes the detached image.
  std::vector<pops::MultiFab<Dim>> active_storage;
  active_storage.reserve(2);
  std::vector<const pops::MultiFab<Dim>*> active(2);
  for (int level = 0; level < 2; ++level) {
    auto active_view = system.prepared_amr_block_level_active_mask(0, level);
    ASSERT_TRUE(active_view) << mode << " level=" << level;
    active_storage.emplace_back(*active_view);
    active[static_cast<std::size_t>(level)] = &active_storage.back();
    ASSERT_NE(active[static_cast<std::size_t>(level)], nullptr) << mode << " level=" << level;
    const pops::Real active_cells =
        pops::all_reduce_sum(pops::reduce_sum_local(*active[static_cast<std::size_t>(level)], 0));
    pops::Real valid_cells = pops::Real(0);
    auto state_view = system.prepared_amr_block_state(0, level);
    ASSERT_TRUE(state_view);
    const auto& state = *state_view;
    for (std::size_t local = 0; local < state.local_size(); ++local)
      valid_cells += static_cast<pops::Real>(state.fab(local).box().numPts());
    valid_cells = pops::all_reduce_sum(valid_cells);
    EXPECT_GT(active_cells, pops::Real(0)) << mode << " level=" << level;
    EXPECT_LT(active_cells, valid_cells) << mode << " level=" << level;
  }

  const int block_count = system.n_blocks();
  if (block_count < 1)
    throw std::logic_error("embedded-boundary callback requires at least one Program block");
  const std::string program_identity = "tests.amr.program-eb/" + mode + "/command@1";
  const std::string program_clock = "test.clock.macro";
  const std::vector<std::string> program_blocks = system.block_names();
  if (program_blocks.size() != static_cast<std::size_t>(block_count))
    throw std::logic_error("embedded-boundary callback Program block authority is incomplete");
  std::vector<Resource> resources;
  resources.reserve(static_cast<std::size_t>(level_count * block_count + 2));
  std::vector<pops::test::program_v5::CallbackProgramFluxBasisOccurrence> flux_basis_occurrences;
  std::vector<pops::test::program_v5::CallbackProgramFaceFluxStage> face_flux_stages;
  flux_basis_occurrences.reserve(static_cast<std::size_t>(level_count * block_count));
  face_flux_stages.reserve(static_cast<std::size_t>(level_count * block_count));
  for (int level = 0; level < level_count; ++level) {
    for (int block = 0; block < block_count; ++block) {
      const auto state_view = system.prepared_amr_block_state(block, level);
      ASSERT_TRUE(state_view) << mode << " level=" << level << " block=" << block;
      const std::size_t slot = resources.size();
      const int rhs_identity = 3000 + block;
      const std::string resource_identity = "tests.amr.program-eb/" + mode + "/rhs/" +
                                            std::to_string(block) + "/level/" +
                                            std::to_string(level);
      Resource resource{Resource::Kind::rhs,
                        slot,
                        0,
                        block,
                        level,
                        static_cast<std::uint32_t>(state_view->ncomp()),
                        static_cast<std::uint32_t>(state_view->ghosts()[0])};
      resource.value_id = static_cast<std::uint64_t>(rhs_identity);
      resource.identity = resource_identity;
      resource.occurrence_path = resource_identity + "/occurrence";
      resource.owner = program_blocks.at(static_cast<std::size_t>(block));
      resource.clock = program_clock;
      resources.push_back(std::move(resource));

      const auto dense_slot = static_cast<std::uint32_t>(slot);
      flux_basis_occurrences.push_back(
          {dense_slot, dense_slot, block, level, rhs_identity, 0, 0, 1,
           resource_identity + "/flux-basis", resource_identity + "/flux-basis/occurrence",
           program_blocks.at(static_cast<std::size_t>(block)), program_clock});
      face_flux_stages.push_back(
          {dense_slot, dense_slot, dense_slot, 1, 1, 1, resource_identity + "/face-flux",
           resource_identity + "/face-flux/occurrence",
           program_blocks.at(static_cast<std::size_t>(block)), program_clock});
    }
  }
  const auto status_slot = resources.size();
  resources.push_back({Resource::Kind::scalar, status_slot, 0, 0, 0, 1, 0});
  resources.push_back({Resource::Kind::scalar, status_slot + 1, 0, 0, 1, 1, 0});
  const std::optional<std::vector<pops::runtime::program::ProgramFluxBudgetRecord>> flux_budgets{
      std::vector<pops::runtime::program::ProgramFluxBudgetRecord>(
          static_cast<std::size_t>(block_count), {1, 1, 0, 0})};

  Command command;
  Result result;
  pops::test::install_explicit_amr_callback_program<Dim>(
      system, program_identity, program_clock, program_blocks, resources, {},
      [&command, &result, &active, status_slot, block_count](Context& context, double macro_dt) {
        // Static ABI-v5 flux authority is complete for every active level group, including the
        // seed/status/publication probes below. This fixture's advection speed is exactly zero,
        // so those probes evaluate their authenticated RHS without changing state; only `advance`
        // applies it to the candidate.
        const auto evaluate_static_rhs = [&context, block_count](double level_dt,
                                                                 bool advance_state) {
          context.set_stage_time(0, 1);
          for (int block = 0; block < block_count; ++block) {
            auto& state = context.state(block);
            const auto slot = static_cast<pops::runtime::program::ProgramCacheSlot>(
                context.level() * block_count + block);
            auto& residual = context.rhs_scratch(slot, 0, state);
            context.rhs_into(block, state, residual, 3000 + block);
            if (advance_state)
              context.axpy(state, pops::Real(level_dt), residual);
          }
        };
        result = {};
        result.invoked = true;
        switch (command.operation) {
          case Operation::seed: {
            std::array<bool, 2> observed{};
            context.advance_hierarchy(macro_dt, [&context, &result, &active, &observed,
                                                 &evaluate_static_rhs](double level_dt) {
              evaluate_static_rhs(level_dt, false);
              const int level = context.level();
              auto& state = context.state(0);
              const auto& mask = *active[static_cast<std::size_t>(level)];
              state.set_val(pops::Real(1));
              seed_inactive_and_ghosts(state, mask, pops::Real(31 + level), pops::Real(71 + level));
              if (!observed[static_cast<std::size_t>(level)]) {
                if (level == 0) {
                  result.active_count = pops::all_reduce_sum(pops::reduce_sum_local(mask, 0));
                  result.state_sum =
                      pops::all_reduce_sum(pops::reduce_active_sum_local(state, 0, &mask));
                  result.state_abs_sum =
                      pops::all_reduce_sum(pops::reduce_active_abs_sum_local(state, 0, &mask));
                  result.state_min =
                      pops::all_reduce_min(pops::reduce_active_min_local(state, 0, &mask));
                  result.state_max =
                      pops::all_reduce_max(pops::reduce_active_max_local(state, 0, &mask));
                  result.state_norm2 = std::sqrt(
                      pops::all_reduce_sum(pops::dot_active_local(state, state, 0, &mask)));
                }
                observed[static_cast<std::size_t>(level)] = true;
              }
            });
            break;
          }
          case Operation::status: {
            std::array<bool, 2> observed{};
            context.advance_hierarchy(macro_dt, [&context, &result, &active, &observed, status_slot,
                                                 &evaluate_static_rhs](double level_dt) {
              evaluate_static_rhs(level_dt, false);
              const int level = context.level();
              auto& state = context.state(0);
              const auto& mask = *active[static_cast<std::size_t>(level)];
              auto& status = context.scalar_scratch(status_slot + level, 0, state, 1, 0);
              if (level == 0)
                write_status(status, mask, pops::Real(0),
                             std::numeric_limits<pops::Real>::quiet_NaN());
              else
                write_status(status, mask, std::numeric_limits<pops::Real>::quiet_NaN(),
                             pops::Real(0));
              if (!observed[static_cast<std::size_t>(level)]) {
                const auto local = pops::reduce_masked_max_local(status, 0, &mask);
                const long has_active = pops::all_reduce_max(local.has_active ? 1L : 0L);
                const long has_invalid = pops::all_reduce_max(local.has_invalid ? 1L : 0L);
                const pops::Real maximum = pops::all_reduce_max(local.maximum);
                const pops::Real value = has_invalid != 0 ? pops::Real(3) : maximum;
                if (level == 0)
                  result.status_max_before = has_active != 0 ? value : pops::Real(0);
                else
                  result.status_max_after = has_invalid != 0 ? pops::Real(3) : value;
                observed[static_cast<std::size_t>(level)] = true;
              }
            });
            break;
          }
          case Operation::publish:
            context.advance_hierarchy(
                macro_dt, [&context, &active, &evaluate_static_rhs](double level_dt) {
                  evaluate_static_rhs(level_dt, false);
                  const int level = context.level();
                  auto& state = context.state(0);
                  set_active_valid(state, *active[static_cast<std::size_t>(level)],
                                   pops::Real(5 + level));
                });
            break;
          case Operation::reject:
            context.advance_hierarchy(
                macro_dt, [&context, &active, &evaluate_static_rhs](double level_dt) {
                  evaluate_static_rhs(level_dt, false);
                  if (context.level() != 1)
                    return;
                  auto& state = context.state(0);
                  set_active_valid(state, *active[1], std::numeric_limits<pops::Real>::quiet_NaN());
                  throw std::runtime_error("injected embedded-boundary candidate rejection");
                });
            break;
          case Operation::advance:
            context.advance_hierarchy(macro_dt, [&evaluate_static_rhs](double level_dt) {
              evaluate_static_rhs(level_dt, true);
            });
            break;
        }
      },
      {}, {}, flux_budgets, {}, std::nullopt, flux_basis_occurrences, face_flux_stages);
  const auto run_command = [&](Operation operation) {
    command.operation = operation;
    result = {};
    try {
      system.step(1.0e-4);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "embedded-boundary %s command failed: %s\n", mode.c_str(), error.what());
      result.threw = true;
    } catch (...) {
      std::fprintf(stderr, "embedded-boundary %s command failed with unknown exception\n",
                   mode.c_str());
      result.threw = true;
    }
    return result;
  };

  // The state seed is an authenticated Program owner.  A NaN in every inactive valid cell must
  // not poison any raw all-level Program reduction.
  const Result seed_result = run_command(Operation::seed);
  EXPECT_TRUE(seed_result.invoked && !seed_result.threw) << mode;
  EXPECT_EQ(seed_result.state_sum, seed_result.active_count) << mode;
  EXPECT_EQ(seed_result.state_abs_sum, seed_result.active_count) << mode;
  EXPECT_EQ(seed_result.state_min, pops::Real(1)) << mode;
  EXPECT_EQ(seed_result.state_max, pops::Real(1)) << mode;
  EXPECT_EQ(seed_result.state_norm2, std::sqrt(seed_result.active_count)) << mode;

  // Scratch ownership is also level-aware.  Keep all scratch and reductions inside the MODULE
  // callback; the host observes only the scalar result channel after the candidate seals.
  const Result status_result = run_command(Operation::status);
  EXPECT_TRUE(status_result.invoked && !status_result.threw) << mode;
  EXPECT_EQ(status_result.status_max_before, pops::Real(0)) << mode;
  EXPECT_EQ(status_result.status_max_after, pops::Real(3)) << mode;

  // The seed callback leaves finite accepted state with unique inactive/ghost sentinels.  Capture
  // it through the public read API before asking the MODULE to publish a candidate.
  std::vector<pops::MultiFab<Dim>> accepted_before;
  accepted_before.reserve(2);
  for (int level = 0; level < 2; ++level) {
    const auto state_view = system.prepared_amr_block_state(0, level);
    ASSERT_TRUE(state_view) << mode << " level=" << level;
    accepted_before.emplace_back(*state_view);
  }

  // The callback mutates only its detached candidate.  System publishes it after the aggregate
  // validation/seal, then the host observes the committed generation.
  const Result publish_result = run_command(Operation::publish);
  EXPECT_TRUE(publish_result.invoked && !publish_result.threw) << mode;
  for (int level = 0; level < 2; ++level) {
    const auto state_view = system.prepared_amr_block_state(0, level);
    ASSERT_TRUE(state_view) << mode << " level=" << level;
    EXPECT_TRUE(inactive_valid_and_ghosts_equal(*state_view,
                                                accepted_before[static_cast<std::size_t>(level)],
                                                *active[static_cast<std::size_t>(level)]))
        << mode << " level=" << level;
    const auto& mask = *active[static_cast<std::size_t>(level)];
    EXPECT_EQ(pops::all_reduce_min(pops::reduce_active_min_local(*state_view, 0, &mask)),
              pops::Real(5 + level))
        << mode << " level=" << level;
    // Fine active cells are six. The terminal AMR publication average-downs covered fine cells
    // into level zero, so six is also the coarse active maximum while uncovered coarse cells stay
    // five. This is the accepted hierarchy authority, not a per-level independent copy oracle.
    EXPECT_EQ(pops::all_reduce_max(pops::reduce_active_max_local(*state_view, 0, &mask)),
              pops::Real(6))
        << mode << " level=" << level;
  }

  // One bad active value in one live level rejects the whole publication attempt.  No accepted
  // level is changed, which is the fail-closed boundary before a subcycle can commit it.
  std::vector<pops::MultiFab<Dim>> before_rejection;
  {
    const auto before_rejection_0_view = system.prepared_amr_block_state(0, 0);
    const auto before_rejection_1_view = system.prepared_amr_block_state(0, 1);
    ASSERT_TRUE(before_rejection_0_view);
    ASSERT_TRUE(before_rejection_1_view);
    before_rejection.emplace_back(*before_rejection_0_view);
    before_rejection.emplace_back(*before_rejection_1_view);
  }
  const Result reject_result = run_command(Operation::reject);
  EXPECT_TRUE(reject_result.invoked && reject_result.threw) << mode;
  for (int level = 0; level < 2; ++level) {
    auto state_view = system.prepared_amr_block_state(0, level);
    ASSERT_TRUE(state_view);
    EXPECT_EQ(
        pops::difference_sum_sq_all(*state_view, before_rejection[static_cast<std::size_t>(level)]),
        pops::Real(0))
        << mode << " level=" << level;
  }

  // advance_hierarchy() routes the exact same terminal publication through the subcycling engine.
  // It must retain the inactive sentinels and ghosts seeded above, not merely the direct facade.
  const Result advance_result = run_command(Operation::advance);
  EXPECT_TRUE(advance_result.invoked && !advance_result.threw) << mode;
  for (int level = 0; level < 2; ++level) {
    auto state_view = system.prepared_amr_block_state(0, level);
    ASSERT_TRUE(state_view);
    EXPECT_TRUE(inactive_valid_and_ghosts_equal(*state_view,
                                                before_rejection[static_cast<std::size_t>(level)],
                                                *active[static_cast<std::size_t>(level)]))
        << mode << " level=" << level;
  }
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
