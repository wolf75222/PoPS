/// @file
/// @brief Exact-rank spatial parity between the generated System and AmrSystem facades.

#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/builders/compiled/dsl_block.hpp>
#include <pops/runtime/builders/compiled/generated_system_block.hpp>

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(POPS_HAS_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

namespace pops {

template <int Dim, class Model>
PreparedSystemBlock<Dim> prepare_exact_system_block(
    CompiledSystemBlockPreparation<Dim, Model> request) {
  return prepare_generated_system_block(std::move(request));
}

}  // namespace pops

namespace {

constexpr int Dim = pops::kNativeDimension;
constexpr pops::Real kGamma = pops::Real(1.4);
constexpr int kCellsPerAxis = 16;

template <int Rank>
struct EulerModel : pops::nd::IdealGasEuler<Rank> {
  using Law = pops::nd::IdealGasEuler<Rank>;
  using State = typename Law::State;

  EulerModel() : Law(Law::prepare(kGamma)) {}

  [[nodiscard]] static constexpr pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr-spatial-parity.euler", 1};
  }

  void serialize_exact_parameters(pops::ExactContractBuilder& contract) const {
    Law::serialize_exact_parameters(contract);
  }

  POPS_HD State source(const State&, const pops::ProviderValues<0>&) const { return {}; }
  POPS_HD pops::Real elliptic_rhs(const State&) const { return pops::Real(0); }
};

template <class Ranked, class Value>
Ranked filled(Value value) {
  Ranked result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

std::size_t cell_count(const pops::Extent<Dim>& shape) {
  std::size_t result = 1;
  for (int axis = 0; axis < Dim; ++axis)
    result *= static_cast<std::size_t>(shape[axis]);
  return result;
}

pops::Index<Dim> index_from_ordinal(std::size_t ordinal, const pops::Box<Dim>& box) {
  pops::Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const auto length = static_cast<std::size_t>(box.length(axis));
    result[axis] = box.lo[axis] + static_cast<int>(ordinal % length);
    ordinal /= length;
  }
  return result;
}

std::size_t storage_ordinal(const pops::Box<Dim>& box, const pops::Index<Dim>& index) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result += static_cast<std::size_t>(index[axis] - box.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(box.length(axis));
  }
  return result;
}

std::vector<double> smooth_positive_state(const pops::Extent<Dim>& shape) {
  using Schema = typename EulerModel<Dim>::Schema;
  const pops::Box<Dim> domain = pops::Box<Dim>::from_extents(shape);
  const std::size_t cells = cell_count(shape);
  std::vector<double> state(static_cast<std::size_t>(EulerModel<Dim>::n_vars) * cells, 0.0);
  constexpr double two_pi = 6.283185307179586476925286766559;
  for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
    const pops::Index<Dim> cell = index_from_ordinal(ordinal, domain);
    double radius_squared = 0.0;
    double phase = 0.0;
    double velocity[Dim]{};
    for (int axis = 0; axis < Dim; ++axis) {
      const double coordinate =
          (static_cast<double>(cell[axis]) + 0.5) / static_cast<double>(shape[axis]);
      const double centered = coordinate - 0.5;
      radius_squared += centered * centered;
      phase += static_cast<double>(axis + 1) * two_pi * coordinate;
      velocity[axis] =
          0.08 * static_cast<double>(axis + 1) * std::sin(two_pi * coordinate + 0.3 * axis);
    }
    const double density = 1.0 + 0.35 * std::exp(-radius_squared / 0.025);
    const double pressure = 1.0 + 0.08 * std::sin(phase);
    double kinetic = 0.0;
    state[static_cast<std::size_t>(Schema::density) * cells + ordinal] = density;
    for (int axis = 0; axis < Dim; ++axis) {
      const double momentum = density * velocity[axis];
      state[static_cast<std::size_t>(axis + 1) * cells + ordinal] = momentum;
      kinetic += 0.5 * density * velocity[axis] * velocity[axis];
    }
    state[static_cast<std::size_t>(Schema::energy) * cells + ordinal] =
        pressure / (static_cast<double>(kGamma) - 1.0) + kinetic;
  }
  return state;
}

pops::SystemConfig<Dim> system_config() {
  pops::SystemConfig<Dim> config;
  config.shape = filled<pops::Extent<Dim>>(kCellsPerAxis);
  config.lower = filled<pops::RealVector<Dim>>(pops::Real(0));
  config.upper = filled<pops::RealVector<Dim>>(pops::Real(1));
  config.periodicity.fill(true);
  config.boxes = {config.index_domain()};
  return config;
}

pops::AmrSystemConfig<Dim> amr_config() {
  pops::AmrSystemConfig<Dim> config;
  config.shape = filled<pops::Extent<Dim>>(kCellsPerAxis);
  config.lower = filled<pops::RealVector<Dim>>(pops::Real(0));
  config.upper = filled<pops::RealVector<Dim>>(pops::Real(1));
  config.periodicity.fill(true);
  config.level_count = 1;
  config.transition_ratios.clear();
  config.transition_buffers.clear();
  config.transition_lookaheads.clear();
  config.regrid_every = 0;
  return config;
}

std::vector<double> flatten(const pops::MultiFab<Dim>& field) {
  EXPECT_EQ(field.local_size(), 1U);
  if (field.local_size() != 1U)
    return {};
  const auto& fab = field.fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  const std::size_t cells = static_cast<std::size_t>(fab.box().numPts());
  std::vector<double> result(static_cast<std::size_t>(fab.ncomp()) * cells);
  for (std::size_t ordinal = 0; ordinal < cells; ++ordinal) {
    const pops::Index<Dim> cell = index_from_ordinal(ordinal, fab.box());
    for (int component = 0; component < fab.ncomp(); ++component) {
      const std::size_t source =
          static_cast<std::size_t>(component) * static_cast<std::size_t>(fab.grown_box().numPts()) +
          storage_ordinal(fab.grown_box(), cell);
      result[static_cast<std::size_t>(component) * cells + ordinal] = host(source);
    }
  }
  return result;
}

double max_difference(const std::vector<double>& left, const std::vector<double>& right) {
  if (left.size() != right.size())
    return HUGE_VAL;
  double result = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index)
    result = std::fmax(result, std::fabs(left[index] - right[index]));
  return result;
}

double max_magnitude(const std::vector<double>& values) {
  double result = 0.0;
  for (const double value : values)
    result = std::fmax(result, std::fabs(value));
  return result;
}

struct SpatialPair {
  std::vector<double> uniform;
  std::vector<double> adaptive;
};

SpatialPair evaluate_pair(const std::string& riemann, const std::string& reconstruction,
                          const std::vector<double>& state) {
  pops::System<Dim> uniform(system_config());
  uniform.install_prepared_boundary_execution_lane(
      std::make_shared<pops::ExecutionLane>(pops::ExecutionLane::duplicate_world_collectively(
          "tests.amr-spatial-parity/uniform-runtime@1")));
  uniform.install_block_state_route("gas", "tests.amr-spatial-parity/system/gas/state@1");
  uniform.seal_auxiliary_providers();
  pops::add_compiled_model<Dim>(uniform, "gas", EulerModel<Dim>{}, "minmod", riemann,
                                reconstruction, "explicit", static_cast<double>(kGamma));
  uniform.set_state("gas", state);

  pops::AmrSystem<Dim> adaptive(amr_config());
  pops::test::install_amr_runtime_authority(adaptive, "tests.amr-spatial-parity/amr-runtime@1");
  adaptive.install_block_state_route("gas", "tests.amr-spatial-parity/amr/gas/state@1");
  pops::add_compiled_model<Dim>(adaptive, "gas", EulerModel<Dim>{}, "minmod", riemann,
                                reconstruction, "explicit", static_cast<double>(kGamma), 1, 1, {},
                                {}, 0.0, static_cast<double>(pops::kWenoEpsilon), false,
                                "tests.amr-spatial-parity/physical_flux");
  adaptive.set_conservative_state("gas", state);
  const auto& evaluation = adaptive.evaluate_prepared_amr_level(
      {.clock = "test.amr-spatial-parity", .level = 0, .stage_fraction = {0, 1}});
  return {uniform.eval_rhs("gas"), flatten(evaluation.residual)};
}

}  // namespace

TEST(test_amr_spatial_parity, GeneratedFacadesShareExactRankedHllcAndRoeSpatialAuthority) {
#if defined(POPS_HAS_KOKKOS)
  Kokkos::ScopeGuard guard;
#endif
  const std::vector<double> state = smooth_positive_state(system_config().shape);

  const SpatialPair hllc_primitive = evaluate_pair("hllc", "primitive", state);
  const SpatialPair hllc_conservative = evaluate_pair("hllc", "conservative", state);
  const SpatialPair roe_primitive = evaluate_pair("roe", "primitive", state);
  const SpatialPair roe_conservative = evaluate_pair("roe", "conservative", state);

  for (const auto* pair :
       {&hllc_primitive, &hllc_conservative, &roe_primitive, &roe_conservative}) {
    EXPECT_LT(max_difference(pair->uniform, pair->adaptive), 1e-13);
    EXPECT_GT(max_magnitude(pair->uniform), 1e-6);
  }
  EXPECT_GT(max_difference(hllc_primitive.uniform, hllc_conservative.uniform), 1e-10);
  EXPECT_GT(max_difference(roe_primitive.uniform, roe_conservative.uniform), 1e-10);
  EXPECT_GT(max_difference(hllc_primitive.uniform, roe_primitive.uniform), 1e-10)
      << "HLLC and Roe must remain distinct primitive-state Riemann authorities";
  EXPECT_GT(max_difference(hllc_conservative.uniform, roe_conservative.uniform), 1e-10)
      << "HLLC and Roe must remain distinct conservative-state Riemann authorities";
}
