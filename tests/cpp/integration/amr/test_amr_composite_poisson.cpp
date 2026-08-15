#include <gtest/gtest.h>

#include "explicit_amr_program.hpp"
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/numerics/spatial/nd/conservation_laws.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/runtime/program/amr_program_context.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace pops {
template <int Dim, class Model>
void add_test_compiled_model(AmrSystem<Dim>& system, const std::string& name, Model model) {
  add_compiled_model<Dim>(system, name, std::move(model), "minmod", "rusanov", "conservative",
                          "explicit", static_cast<double>(kPhysicalDefaultGamma), 1, 1, {}, {}, 0.0,
                          static_cast<double>(kWenoEpsilon), false,
                          "tests.composite-poisson/physical_flux");
}
}  // namespace pops

namespace {

constexpr int Dim = pops::kNativeDimension;
constexpr int kCells = 24;

template <class Ranked, class Value>
Ranked filled(Value value) {
  Ranked result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Rank>
struct AdvectionModel {
  using Law = pops::nd::ScalarAdvection<Rank>;
  using Schema = typename Law::Schema;
  using State = typename Law::State;
  using Primitive = typename Law::Primitive;
  static constexpr int dimension = Rank;
  static constexpr int n_vars = Law::n_vars;
  static constexpr int n_providers = 0;
  Law law{};

  static pops::PreparedProviderIdentity provider_identity() noexcept {
    return {"test.amr-composite-poisson.scalar-advection", 1};
  }
  void serialize_exact_parameters(pops::ExactContractBuilder& exact) const {
    for (int axis = 0; axis < Rank; ++axis)
      exact.scalar(law.velocity()[axis]);
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

AdvectionModel<Dim> advection_model() {
  pops::RealVector<Dim> velocity{};
  velocity[0] = pops::Real(1);
  return {pops::nd::ScalarAdvection<Dim>::prepare(velocity)};
}

pops::Index<Dim> index_from_ordinal(const pops::Box<Dim>& box, std::size_t ordinal) {
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

pops::Real exact(const pops::Geometry<Dim>& geometry, const pops::Index<Dim>& index) {
  const pops::Real pi = std::acos(pops::Real(-1));
  pops::Real result = pops::Real(1);
  for (int axis = 0; axis < Dim; ++axis)
    result *= std::sin(pi * geometry.cell_coordinate(axis, index[axis]));
  return result;
}

void fill_rhs(pops::MultiFab<Dim>& rhs, const pops::Geometry<Dim>& geometry) {
  const pops::Real pi = std::acos(pops::Real(-1));
  const pops::Real eigenvalue = static_cast<pops::Real>(Dim) * pi * pi;
  for (std::size_t local = 0; local < rhs.local_size(); ++local) {
    auto& fab = rhs.fab(local);
    auto host = fab.create_host_mirror();
    for (std::size_t n = 0; n < static_cast<std::size_t>(fab.box().numPts()); ++n) {
      const auto index = index_from_ordinal(fab.box(), n);
      host(storage_ordinal(fab.grown_box(), index)) = eigenvalue * exact(geometry, index);
    }
    fab.copy_from_host(host);
  }
}

pops::EllipticBuildRequest<Dim> coarse_request(const pops::Geometry<Dim>& geometry) {
  const pops::mesh::BoxArray<Dim> layout(std::vector<pops::Box<Dim>>{geometry.domain()});
  const pops::mesh::RankSpace<Dim> ranks{pops::Index<Dim>{},
                                         filled<pops::Extent<Dim>>(std::int64_t{1})};
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(layout, ranks);
  std::array<pops::PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  faces.fill({pops::PhysicalBoundaryKind::dirichlet, pops::Real(0)});
  pops::RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  return {geometry,
          layout,
          distribution,
          pops::Index<Dim>{},
          {pops::BoundaryTopology<Dim>::physical(), faces, spacing},
          pops::Extent<Dim>{},
          filled<pops::Extent<Dim>>(std::int64_t{1}),
          {1, 0}};
}

double multifab_error(const pops::MultiFab<Dim>& field, const pops::Geometry<Dim>& geometry,
                      const pops::Box<Dim>& region) {
  double result = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto overlap = field.box(local).intersect(region);
    if (overlap.empty())
      continue;
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t n = 0; n < static_cast<std::size_t>(overlap.numPts()); ++n) {
      const auto index = index_from_ordinal(overlap, n);
      result = std::max(result,
                        std::abs(static_cast<double>(host(storage_ordinal(fab.grown_box(), index)) -
                                                     exact(geometry, index))));
    }
  }
  return pops::all_reduce_max(result);
}

double flattened_error(const std::vector<double>& field, const pops::Geometry<Dim>& geometry,
                       const pops::Box<Dim>& region) {
  double result = 0;
  for (std::size_t n = 0; n < static_cast<std::size_t>(region.numPts()); ++n) {
    const auto index = index_from_ordinal(region, n);
    result = std::max(result, std::abs(field[storage_ordinal(geometry.domain(), index)] -
                                       static_cast<double>(exact(geometry, index))));
  }
  return result;
}

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
    options.max_box_size[static_cast<std::size_t>(axis)] = 64;
  }
  options.budget = {16, 256, 8192, 64, 1U << 20};
  pops::amr::tagging::ClusterResultIdentity<Dim> identity{
      "test.amr-composite-poisson.cluster", parent.exact_identity(), options, {}, boxes.boxes()};
  return {boxes, std::move(identity)};
}

void publish_centered_fine_level(pops::AmrSystem<Dim>& system) {
  auto* engine = system.engine();
  ASSERT_NE(engine, nullptr);
  const pops::amr::RefinementRatio<Dim> ratio(filled<std::array<int, Dim>>(2));
  const pops::amr::regridding::RegridPreparationBudget budget{
      .clustered_parent_layout = {16, 120},
      .fine_layout = {16, 120},
      .load_balance = {16, 16, std::numeric_limits<std::int64_t>::max()},
  };
  auto prepared =
      engine->prepare_regrid(0, ratio, centered_cluster(engine->hierarchy().layout(0)), budget);
  ASSERT_TRUE(prepared.fine_layout().has_value());
  pops::MultiFab<Dim> child(
      prepared.fine_layout()->patches(), prepared.fine_layout()->distribution(),
      engine->hierarchy().state(0).local_rank(), engine->hierarchy().state(0).ncomp(),
      engine->hierarchy().state(0).ghosts());
  child.set_val(pops::Real(1));
  engine->publish_regrid(0, std::move(prepared), std::move(child));
}

pops::runtime::system::AuxiliaryComponentKey install_field_output(pops::AmrSystem<Dim>& system) {
  using namespace pops::runtime::system;
  AuxiliaryStorageShape<Dim> shape;
  for (int axis = 0; axis < Dim; ++axis)
    shape.halo[axis] = 1;
  const AuxiliaryComponentKey key{"test.amr-composite-poisson", "field", "phi", "potential"};
  const AuxiliaryComponentContract contract{"cell-average", "cell", "unitless", "amr-field",
                                            "scalar"};
  system.install_prepared_auxiliary_provider(PreparedAuxiliaryProvider<Dim>{
      "test.amr-composite-poisson.field-output",
      AuxiliaryProviderKind::field_output,
      {AuxiliaryEvaluationEvent::before_field_solve, AuxiliaryFreshness::evaluation},
      {{key, contract, shape}},
      {}});
  system.seal_auxiliary_providers();
  return key;
}

pops::runtime::multiblock::BoundaryEvaluationPoint evaluation_point() {
  return {.clock = "test-clock",
          .tick = 0,
          .level = 0,
          .substep = 0,
          .stage = 0,
          .stage_fraction = {0, 1},
          .dt = 0.01,
          .physical_time = 0.0};
}

}  // namespace

TEST(test_amr_composite_poisson,
     exact_provider_composite_route_publishes_a_more_accurate_refined_field) {
  pops::comm_init();
  const pops::Box<Dim> coarse_domain{pops::Index<Dim>{}, filled<pops::Index<Dim>>(kCells - 1)};
  const auto coarse_geometry = pops::Geometry<Dim>::from_bounds(
      coarse_domain, pops::RealVector<Dim>{}, filled<pops::RealVector<Dim>>(pops::Real(1)));

  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.amr.composite-reference");
  pops::elliptic::mg::GeometricMultigridOptions coarse_controls;
  coarse_controls.relative_tolerance = pops::Real(1e-10);
  coarse_controls.maximum_cycles = 100;
  pops::elliptic::mg::GeometricMG<Dim> coarse_solver(coarse_request(coarse_geometry), lane,
                                                     coarse_controls);
  coarse_solver.install_nullspace(pops::FieldNullspacePlan<Dim>{},
                                  pops::PreparedVectorDistribution<Dim>::replicated());
  fill_rhs(coarse_solver.rhs(), coarse_geometry);
  const pops::SolveReport coarse_report = coarse_solver.solve();
  ASSERT_TRUE(coarse_report.solved()) << coarse_report.reason;

  pops::AmrSystemConfig<Dim> config;
  config.level_count = 2;
  config.regrid_every = 0;
  config.periodicity.fill(false);
  for (int axis = 0; axis < Dim; ++axis)
    config.shape[axis] = kCells;
  pops::AmrSystem<Dim> system(config);
  pops::test::install_amr_runtime_authority(system,
                                            "tests.amr.composite-poisson/exact-provider-runtime");
  const pops::AmrFieldHierarchyPolicyAuthority hierarchy{
      "pops.field-hierarchy.composite", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  pops::CompositeFacOptions fac_controls;
  fac_controls.max_iters = 80;
  fac_controls.fine_sweeps = 80;
  fac_controls.rel_tol = pops::Real(1e-9);
  const std::string slot = "field/manufactured";
  system.set_field_solver_plan(
      slot, "test.amr-composite-poisson.plan", "test.amr-composite-poisson.provider",
      "test.amr-composite-poisson", "tracer", "phi",
      {{"test.amr-composite-poisson", "field", "phi", "potential"}}, 1, {"test.manufactured-rhs"},
      {"tracer"}, {"manufactured"}, {1.0}, "geometric_mg", hierarchy,
      pops::geometric_mg_amr_field_solver_options(pops::GeometricMgOptions{}, fac_controls));
  constexpr const char* state_route = "state/tracer";
  system.install_block_state_route("tracer", state_route);
  std::vector<std::string> face_types(static_cast<std::size_t>(2 * Dim), "dirichlet");
  std::vector<std::string> face_identities;
  face_identities.reserve(static_cast<std::size_t>(2 * Dim));
  for (int face = 0; face < 2 * Dim; ++face)
    face_identities.push_back("tests.amr.composite-poisson/exact-provider-face-" +
                              std::to_string(face));
  system.install_hyperbolic_boundary(
      "tracer", "tests.amr.composite-poisson/exact-provider-boundary@1", 1, face_types,
      std::vector<double>(static_cast<std::size_t>(2 * Dim), 0.0), face_identities, {"Scalar"},
      state_route);
  pops::add_test_compiled_model(system, "tracer", advection_model());
  const auto output_key = install_field_output(system);
  system.register_elliptic_field("tracer", "phi", {output_key}, 1);
  system.set_block_elliptic_field(
      "tracer", "phi", "test.amr-composite-poisson.rhs.manufactured@1",
      [coarse_geometry](const pops::MultiFab<Dim>&, pops::MultiFab<Dim>& rhs) {
        const bool fine_level = rhs.box(0).hi[0] >= kCells;
        const auto geometry =
            fine_level ? coarse_geometry.refine(filled<pops::Extent<Dim>>(std::int64_t{2}))
                       : coarse_geometry;
        fill_rhs(rhs, geometry);
      });
  std::size_t coarse_cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    coarse_cells *= static_cast<std::size_t>(kCells);
  system.set_conservative_state("tracer", std::vector<double>(coarse_cells, 1.0));
  publish_centered_fine_level(system);
  system.refresh_prepared_amr_levels();
  system.set_program_block_map({0});

  auto context = pops::runtime::program::make_program_execution_provider(&system);
  context->configure_primary_clock("test-clock");
  context->begin_step(0.01);
  pops::MultiFab<Dim> stage = context->scratch_state_like(context->state(0));
  stage.set_val(pops::Real(1));
  pops::SolveOutcome outcome =
      context->solve_fields_from_state_at(evaluation_point(), slot, 0, stage);
  const pops::SolveReport accepted =
      outcome.consume(outcome.report().solved_value_available()
                          ? pops::SolveConsumption::kAccept
                          : (outcome.report().action == pops::SolveAction::kRejectAttempt
                                 ? pops::SolveConsumption::kRejectAttempt
                                 : pops::SolveConsumption::kFailRun));
  ASSERT_TRUE(accepted.solved()) << accepted.reason;

  EXPECT_EQ(system.field_provider_slots(), std::vector<std::string>{slot});
  EXPECT_EQ(system.field_provider_levels(slot), 2);
  const auto fine_geometry = system.prepared_amr_level_geometry(1);
  const auto refined = system.field_potential_level_global(slot, 1);
  EXPECT_EQ(refined, system.auxiliary_component(output_key, 1))
      << "acceptance must publish the refined candidate through the named field output";
  std::size_t fine_cells = 1;
  for (int axis = 0; axis < Dim; ++axis)
    fine_cells *= static_cast<std::size_t>(2 * kCells);
  ASSERT_EQ(refined.size(), fine_cells);

  pops::Index<Dim> coarse_lo{};
  pops::Index<Dim> coarse_hi{};
  pops::Index<Dim> fine_lo{};
  pops::Index<Dim> fine_hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    coarse_lo[axis] = 5;
    coarse_hi[axis] = 18;
    fine_lo[axis] = 10;
    fine_hi[axis] = 37;
  }
  const double coarse_error =
      multifab_error(coarse_solver.phi(), coarse_geometry, pops::Box<Dim>{coarse_lo, coarse_hi});
  const double refined_error =
      flattened_error(refined, fine_geometry, pops::Box<Dim>{fine_lo, fine_hi});
  EXPECT_GT(coarse_error, 0.0);
  EXPECT_LT(refined_error, coarse_error)
      << "the current ExactAmrFieldSolverProvider composite route must use the refined hierarchy";
  pops::comm_finalize();
}
