// Default generated System flux must fill WENO ghosts through HaloExchange whenever the
// HaloSchedule has remote jobs, matching GhostTransport::materialize. Local fill stays
// when every rank has only local jobs. A remote-only skip (2-arg fill_boundary) is the
// CP-02 MPI first-step failure and is refused here.

#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/numerics/spatial/primitives/wave_speed.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>
#include <pops/runtime/system.hpp>

#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

using namespace pops;

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

constexpr int Dim = kNativeDimension;
constexpr int kCells = 16;
constexpr int kWenoGhosts = 3;

template <int RankDim>
struct ScalarModel {
  using Law = nd::ScalarAdvection<RankDim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = RankDim;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_providers = 0;
  Law law{};

  [[nodiscard]] static constexpr PreparedProviderIdentity provider_identity() noexcept {
    return {"test.mpi-generated-weno-halo.scalar", 1};
  }
  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    for (int axis = 0; axis < RankDim; ++axis)
      contract.scalar(law.velocity()[axis]);
  }
  POPS_HD nd::StateConversion<Primitive> recover(const State& state) const {
    return law.recover(state);
  }
  POPS_HD nd::StateConversion<State> make_conservative(const Primitive& primitive) const {
    return law.make_conservative(primitive);
  }
  POPS_HD nd::StateConversionStatus admissibility(const State& state) const {
    return law.admissibility(state);
  }
  template <int Axis>
  POPS_HD State flux(const State& state) const {
    return law.template flux<Axis>(state);
  }
  template <int Axis>
  POPS_HD Real max_wave_speed(const State& state) const {
    return law.template max_wave_speed<Axis>(state);
  }
  template <int Axis>
  POPS_HD void wave_speeds(const State& state, Real& lower, Real& upper) const {
    law.template wave_speeds<Axis>(state, lower, upper);
  }
  POPS_HD State source(const State&, const ProviderValues<0>&) const { return {}; }
  POPS_HD Real elliptic_rhs(const State&) const { return Real(0); }

  static VariableSet conservative_vars() {
    return {VariableKind::Conservative, {"u"}, 1, {VariableRole::Scalar}};
  }
  static VariableSet primitive_vars() {
    return {VariableKind::Primitive, {"u"}, 1, {VariableRole::Scalar}};
  }
};

int wrap_index(int index, int extent) {
  int wrapped = index % extent;
  if (wrapped < 0)
    wrapped += extent;
  return wrapped;
}

std::size_t storage_ordinal(const Box<Dim>& storage, const Index<Dim>& index) {
  std::size_t ordinal = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    ordinal += static_cast<std::size_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(storage.length(axis));
  }
  return ordinal;
}

SystemConfig<Dim> weno_config() {
  const int ranks = n_ranks();
  if (ranks < 1 || kCells % ranks != 0)
    throw std::invalid_argument("WENO halo test extent must divide the communicator size");
  SystemConfig<Dim> config;
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = kCells;
    config.lower[axis] = Real(0);
    config.upper[axis] = Real(1);
    config.periodicity[static_cast<std::size_t>(axis)] = true;
  }
  config.boxes = {Box<Dim>::from_extents(config.shape)};
  return config;
}

HaloSchedule<Dim> schedule_for(const MultiFab<Dim>& state, const SystemConfig<Dim>& config) {
  Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    ghosts[axis] = kWenoGhosts;
  const Geometry<Dim> geometry =
      Geometry<Dim>::from_bounds(config.index_domain(), config.lower, config.upper);
  const BoundaryTopology<Dim> topology = BoundaryTopology<Dim>::axis_periodic(config.periodicity);
  return HaloSchedule<Dim>(
      state.layout(), state.distribution(), state.local_rank(), geometry.domain(), ghosts, topology,
      state.ncomp(),
      generated_system_detail::halo_budget(state, geometry.domain(), topology, ghosts));
}

bool ghosts_match_periodic_last_axis(const MultiFab<Dim>& state) {
  if (state.local_size() == 0)
    return true;
  for (std::size_t local = 0; local < state.local_size(); ++local) {
    const auto& fab = state.fab(local);
    if (fab.ghosts()[Dim - 1] != kWenoGhosts)
      return false;
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim> grown = fab.grown_box();
    const Box<Dim> valid = fab.box();
    Index<Dim> cursor = grown.lo;
    const auto step = [&](auto&& self, int axis) -> bool {
      if (axis == Dim) {
        if (valid.contains(cursor))
          return true;
        const Real observed = host(storage_ordinal(grown, cursor));
        const Real expected = static_cast<Real>(wrap_index(cursor[Dim - 1], kCells));
        return std::fabs(observed - expected) < Real(1e-12);
      }
      for (cursor[axis] = grown.lo[axis]; cursor[axis] <= grown.hi[axis]; ++cursor[axis]) {
        if (!self(self, axis + 1))
          return false;
      }
      return true;
    };
    if (!step(step, 0))
      return false;
  }
  return true;
}

bool verify_generated_weno_halo() {
  const SystemConfig<Dim> config = weno_config();
  System<Dim> system(config);
  system.install_prepared_boundary_execution_lane(
      std::make_shared<ExecutionLane>(ExecutionLane::duplicate_world_collectively(
          "test.mpi-generated-weno-halo/runtime@1")));
  system.install_block_state_route("u", "test.mpi-generated-weno-halo/state@1");
  add_compiled_model(system, "u", ScalarModel<Dim>{}, "weno5", "rusanov", "conservative",
                     "explicit");
  system.set_poisson("composite", "cartesian_cg");

  const std::size_t cells = static_cast<std::size_t>(std::pow(kCells, Dim));
  std::vector<double> values(cells, 0.0);
  for (std::size_t cell = 0; cell < cells; ++cell) {
    std::size_t remaining = cell;
    int last = 0;
    for (int axis = 0; axis < Dim; ++axis) {
      last = static_cast<int>(remaining % static_cast<std::size_t>(kCells));
      remaining /= static_cast<std::size_t>(kCells);
    }
    values[cell] = static_cast<double>(last);
  }
  system.set_state("u", values);

  const MultiFab<Dim>& state = system.block_state(0);
  if (state.ghosts()[Dim - 1] != kWenoGhosts)
    return false;
  const HaloSchedule<Dim> schedule = schedule_for(state, config);
  const bool remote = schedule.has_remote_jobs();
  bool local_fill_threw = false;
  try {
    schedule.require_local_execution();
  } catch (const std::logic_error& error) {
    local_fill_threw = std::string(error.what()).find("HaloExchange") != std::string::npos;
  }
  if (n_ranks() > 1) {
    if (!remote || !local_fill_threw)
      return false;
    const ExecutionLane fail_lane = ExecutionLane::duplicate_world_collectively(
        "test.mpi-generated-weno-halo/collective-failure@1");
    HaloExchangeContext failure{};
    failure.context_generation = 1;
    failure.schedule_generation = 1;
    failure.fail_allocation_rank = 0;
    bool allocation_threw = false;
    try {
      HaloExchange<Dim> exchange(schedule, fail_lane, failure);
      (void)exchange;
    } catch (const std::exception&) {
      allocation_threw = true;
    }
    if (all_reduce_max(allocation_threw ? 0L : 1L, fail_lane) != 0L)
      return false;
  } else if (remote || local_fill_threw) {
    return false;
  }

  bool rhs_ok = false;
  try {
    const std::vector<double> residual = system.eval_rhs("u");
    rhs_ok = !residual.empty();
    for (double value : residual)
      rhs_ok = rhs_ok && std::isfinite(value);
  } catch (const std::exception&) {
    return false;
  }
  return rhs_ok && ghosts_match_periodic_last_axis(system.block_state(0));
}

int run_generated_weno_halo(int argc, char** argv) {
  comm_init(&argc, &argv);
  int result = 1;
  {
#if defined(POPS_HAS_KOKKOS)
    Kokkos::ScopeGuard guard(argc, argv);
#endif
    const long local = verify_generated_weno_halo() ? 0 : 1;
    const long failures = all_reduce_sum(local);
    if (failures == 0 && my_rank() == 0)
      std::printf("OK test_mpi_generated_weno_halo remote WENO ghosts (np=%d)\n", n_ranks());
    result = failures == 0 ? 0 : 1;
  }
  comm_finalize();
  return result;
}

}  // namespace

TEST(test_mpi_generated_weno_halo, default_flux_fills_remote_weno_ghosts) {
  EXPECT_EQ(pops::test::RunTestBody(&run_generated_weno_halo, "test_mpi_generated_weno_halo"), 0);
}
