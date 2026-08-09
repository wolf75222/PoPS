#include <gtest/gtest.h>

#include "gtest_compat.hpp"
#include "test_harness.hpp"

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

template <int Dim>
struct AdvectionModel {
  using Law = pops::nd::ScalarAdvection<Dim>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  using Aux = pops::AuxState<Dim>;
  static constexpr int dimension = Dim;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_aux = pops::aux_comps_for<Law, Dim>();

  Law law{};

  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.mpi-amr-package-parity.scalar-advection", 1};
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
  POPS_HD State source(const State&, const Aux&) const { return {}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

template <int Dim>
AdvectionModel<Dim> advection_model() {
  pops::RealVector<Dim> velocity{};
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
std::vector<double> gaussian(const pops::Extent<Dim>& shape) {
  std::vector<double> values(cell_count(shape));
  for (std::size_t linear = 0; linear < values.size(); ++linear) {
    std::size_t remainder = linear;
    double radius_squared = 0.0;
    for (int axis = 0; axis < Dim; ++axis) {
      const auto width = static_cast<std::size_t>(shape[axis]);
      const int coordinate = static_cast<int>(remainder % width);
      remainder /= width;
      const double offset = (coordinate + 0.5) / static_cast<double>(shape[axis]) - 0.5;
      radius_squared += offset * offset;
    }
    values[linear] = 1.0 + 0.4 * std::exp(-radius_squared / 0.02);
  }
  return values;
}

struct RunResult {
  std::vector<double> state;
  double mass = 0.0;
  bool second_package_refused = false;
};

template <int Dim>
RunResult run_mode(bool distribute_coarse) {
  pops::AmrSystemConfig<Dim> config;
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  for (int axis = 0; axis < Dim; ++axis) {
    config.shape[axis] = 32;
    config.coarse_max_grid[axis] = distribute_coarse ? 16 : 0;
  }
  config.distribute_coarse = distribute_coarse;
  pops::AmrSystem<Dim> system(config);
  system.install_block_state_route("first", "state/first");
  pops::add_compiled_model<Dim>(system, "first", advection_model<Dim>());
  system.set_conservative_state("first", gaussian(config.shape));
  RunResult result;
  result.state = system.block_level_state_global("first", 0);
  result.mass = system.mass("first");

  try {
    system.install_block_state_route("second", "state/second");
  } catch (const std::logic_error&) {
    result.second_package_refused = true;
  }
  if (system.n_blocks() != 1 || system.block_level_state_global("first", 0) != result.state)
    throw std::runtime_error("second-package refusal changed the accepted first package");
  return result;
}

double checksum(const std::vector<double>& values) {
  double result = 0.0;
  for (const double value : values)
    result += value * value;
  return result;
}

int run_collective_parity(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  int failure = 0;
  {
    Kokkos::ScopeGuard guard(argc, argv);
    try {
      const RunResult replicated = run_mode<pops::kNativeDimension>(false);
      const RunResult distributed = run_mode<pops::kNativeDimension>(true);
      const double replicated_checksum = checksum(replicated.state);
      const double distributed_checksum = checksum(distributed.state);
      const auto spread = [](double value) {
        return pops::all_reduce_max(value) - (-pops::all_reduce_max(-value));
      };
      EXPECT_TRUE(replicated.second_package_refused);
      EXPECT_TRUE(distributed.second_package_refused);
      EXPECT_EQ(replicated.state, distributed.state);
      EXPECT_NEAR(replicated.mass, distributed.mass, 1.0e-13);
      EXPECT_EQ(spread(replicated_checksum), 0.0);
      EXPECT_EQ(spread(distributed_checksum), 0.0);
      EXPECT_EQ(spread(replicated.mass), 0.0);
      EXPECT_EQ(spread(distributed.mass), 0.0);
    } catch (const std::exception& error) {
      std::fprintf(stderr, "rank %d exact package parity failed: %s\n", pops::my_rank(),
                   error.what());
      failure = 1;
    }
    failure = static_cast<int>(
        pops::all_reduce_max(static_cast<long>(failure || ::testing::Test::HasFailure())));
    if (pops::my_rank() == 0 && failure == 0)
      std::printf("OK test_mpi_amr_twoblock_parity np=%d dim=%d fail-closed-package-parity\n",
                  pops::n_ranks(), pops::kNativeDimension);
  }
  pops::comm_finalize();
  return failure;
}

}  // namespace

TEST(test_mpi_amr_twoblock_parity, Runs) {
  EXPECT_EQ(pops::test::RunTestBody(&run_collective_parity, "test_mpi_amr_twoblock_parity"), 0);
}
