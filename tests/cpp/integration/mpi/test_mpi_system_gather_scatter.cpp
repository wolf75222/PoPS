// Exact-ranked collective System regression.  A single global patch is intentionally owned by
// rank zero so every write/gather path has empty peers; no test-side local replica may mask a
// missing collective payload transfer.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/spatial/primitives/wave_speed.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/system.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif
#ifdef POPS_HAS_MPI
#include <mpi.h>
#endif

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

template <int Dim>
struct ScalarAdvectionModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;

  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_providers = 0;
  Law law{};

  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.mpi.system-gather-scatter.scalar", 1};
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
ScalarAdvectionModel<Dim> scalar_advection_model() {
  pops::RealVector<Dim> velocity{};
  for (int axis = 0; axis < Dim; ++axis)
    velocity[axis] = axis == 0 ? pops::Real(0.25) : pops::Real(0);
  return {pops::nd::ScalarAdvection<Dim>::prepare(velocity)};
}

template <int Dim>
pops::SystemConfig<Dim> one_patch_config(int cells_per_axis) {
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
std::size_t cell_count(const pops::Extent<Dim>& extent) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(extent[axis]);
  return result;
}

template <int Dim>
std::vector<double> gather_distributed_scalar_state(pops::System<Dim>& system, const char* name,
                                                     const pops::SystemConfig<Dim>& config) {
  const pops::Box<Dim> domain = config.index_domain();
  const std::vector<double> local = system.get_state(name);
  std::vector<double> global(cell_count(config.shape), 0.0);
  std::size_t local_offset = 0;
  for (const pops::Box<Dim>& local_box : system.local_boxes(name))
    pops::runtime::system::marshaling::for_each_host_index(
        local_box, [&](const pops::Index<Dim>& index, std::size_t) {
          if (local_offset >= local.size())
            throw std::runtime_error("local System state does not match its ranked patch geometry");
          global[pops::runtime::system::marshaling::domain_ordinal(domain, index)] =
              local[local_offset++];
        });
  if (local_offset != local.size())
    throw std::runtime_error("local System state has trailing values after ranked marshaling");
  pops::all_reduce_sum_inplace(global.data(), global.size());
  return global;
}

template <int Dim>
struct DirectDtProbe {
  using State = pops::StateVec<1>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = 1;
  pops::Real value;

  POPS_HD pops::Real stability_dt(const State&) const { return value; }
};

template <int Dim>
void install_forward_euler_program(pops::System<Dim>& system) {
  std::vector<int> block_map(static_cast<std::size_t>(system.n_blocks()));
  std::iota(block_map.begin(), block_map.end(), 0);
  system.set_program_block_map(block_map);

  pops::runtime::program::ProgramContext<Dim> context(&system);
  context.configure_primary_clock("test.clock.macro");
  context.install([context](double dt) {
    context.begin_step(dt);
    context.set_stage_time(0, 1);
    (void)pops::consume_solve_outcome(context.solve_fields());

    std::vector<pops::MultiFab<Dim>*> states;
    std::vector<pops::MultiFab<Dim>*> next_states;
    states.reserve(static_cast<std::size_t>(context.n_blocks()));
    next_states.reserve(static_cast<std::size_t>(context.n_blocks()));
    for (int block = 0; block < context.n_blocks(); ++block) {
      pops::MultiFab<Dim>& state = context.state(block);
      pops::MultiFab<Dim>& residual = context.rhs_scratch(1000 + block, 0, state);
      pops::MultiFab<Dim>& next = context.scratch_state(2000 + block, 0, state);
      context.rhs_into(block, state, residual, 3000 + block);
      context.lincomb(next, pops::Real(1), state, pops::Real(dt), residual);
      states.push_back(&state);
      next_states.push_back(&next);
    }
    for (std::size_t block = 0; block < states.size(); ++block)
      context.lincomb(*states[block], pops::Real(0), *states[block], pops::Real(1),
                      *next_states[block]);
  });
  system.set_program_block_map(block_map);
}

int pops_run_test_mpi_system_gather_scatter(int argc, char** argv) {
  constexpr int Dim = pops::kNativeDimension;
  pops::comm_init(&argc, &argv);
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard(argc, argv);
#endif
  const int rank = pops::my_rank();
  const int ranks = pops::n_ranks();
  long failures = 0;
  const auto check = [&](bool condition, const char* what) {
    if (!condition) {
      std::printf("[rank %d/%d] FAIL %s\n", rank, ranks, what);
      ++failures;
    }
  };

  const int cells_per_axis = 16;
  const double density = 1.5;
  const double dt = 0.01;
  const int steps = 5;
  const pops::SystemConfig<Dim> config = one_patch_config<Dim>(cells_per_axis);
  const std::size_t cells = cell_count(config.shape);

  pops::System<Dim> system(config);
  system.install_block_state_route("u", "test.mpi.system-gather-scatter/u/state@1");
  pops::add_compiled_model<Dim>(system, "u", scalar_advection_model<Dim>(), "none", "rusanov",
                                "conservative", "explicit");
  system.set_poisson("composite", "cartesian_cg");
  install_forward_euler_program(system);

  // set_state authenticates and scatters one global component-major image.  Every rank enters the
  // operation even though only rank zero owns the sole patch.
  system.set_state("u", std::vector<double>(cells, density));
  const std::vector<double> scattered = gather_distributed_scalar_state(system, "u", config);
  bool scatter_preserved = scattered.size() == cells;
  for (double value : scattered)
    scatter_preserved = scatter_preserved && value == density;
  check(scatter_preserved, "global scatter and gather preserve the initialized image");

  for (int step = 0; step < steps; ++step)
    system.step(dt);

  const double dt_cfl = system.step_cfl(0.5);
  check(std::isfinite(dt_cfl) && dt_cfl > 0, "cfl step is collective and finite");

  // eval_rhs performs its finite-value preflight collectively; empty ranks must participate and
  // return their natural empty local image rather than taking a local-only route.
  const std::vector<double> residual = system.eval_rhs("u");
  if (rank == 0) {
    bool residual_finite = residual.size() == cells;
    for (double value : residual)
      residual_finite = residual_finite && std::isfinite(value);
    check(residual_finite, "owner residual is finite");
  }

  const std::vector<double> advanced = gather_distributed_scalar_state(system, "u", config);
  bool state_finite = advanced.size() == cells;
  for (double value : advanced)
    state_finite = state_finite && std::isfinite(value) && std::fabs(value - density) < 1e-10;
  check(state_finite, "global gathered state remains uniform after collective steps");

  const double total_mass = system.mass("u");
  check(std::isfinite(total_mass), "mass reduction is finite");
  check(std::fabs(total_mass - density * static_cast<double>(cells)) < 1e-9,
        "mass reduction preserves the globally scattered state");

  pops::Extent<Dim> reduction_extent{};
  pops::Extent<Dim> process_extent{};
  for (int axis = 0; axis < Dim; ++axis) {
    reduction_extent[axis] = 2;
    process_extent[axis] = 1;
  }
  process_extent[0] = ranks;
  const pops::Box<Dim> reduction_box = pops::Box<Dim>::from_extents(reduction_extent);
  const pops::mesh::BoxArray<Dim> reduction_boxes(std::vector<pops::Box<Dim>>{reduction_box});
  const pops::mesh::RankSpace<Dim> rank_space(pops::Index<Dim>{}, process_extent);
  const pops::mesh::Distribution<Dim> reduction_owners = pops::mesh::Distribution<Dim>::partitioned(
      reduction_boxes, rank_space, {rank_space.coordinate(0)});
  pops::MultiFab<Dim> reduction_state(reduction_boxes, reduction_owners,
                                      rank_space.coordinate(static_cast<std::size_t>(rank)), 1,
                                      pops::Extent<Dim>{});
  reduction_state.set_val(pops::Real(1));

  bool rejected_invalid_dt = false;
  try {
    (void)pops::nd::min_stability_dt_mf(
        DirectDtProbe<Dim>{rank == 0 ? pops::Real(0) : pops::Real(1)}, reduction_state);
  } catch (const std::domain_error&) {
    rejected_invalid_dt = true;
  }
  check(rejected_invalid_dt, "invalid stability bound is rejected collectively");

  const pops::Real direct_dt =
      pops::nd::min_stability_dt_mf(DirectDtProbe<Dim>{pops::Real(0.25)}, reduction_state);
  check(direct_dt == pops::Real(0.25), "valid stability bound is reduced to every empty peer");

#ifdef POPS_HAS_MPI
  if (ranks > 1) {
    long global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    failures = global_failures;
  }
#endif
  if (rank == 0 && failures == 0)
    std::printf("OK test_mpi_system_gather_scatter (np=%d dim=%d)\n", ranks, Dim);
  pops::comm_finalize();
  return failures == 0 ? 0 : 1;
}

}  // namespace

TEST(test_mpi_system_gather_scatter, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&pops_run_test_mpi_system_gather_scatter,
                                    "test_mpi_system_gather_scatter"),
            0);
}
