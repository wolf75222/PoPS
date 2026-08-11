#include <gtest/gtest.h>

#include <pops/core/foundation/native_dimension.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr int kDim = pops::kNativeDimension;
using Field = pops::MultiFab<kDim>;
using Solver = pops::elliptic::mg::GeometricMG<kDim>;

class CommEnvironment final : public ::testing::Environment {
 public:
  void SetUp() override { pops::comm_init(); }
  void TearDown() override { pops::comm_finalize(); }
};

[[maybe_unused]] const ::testing::Environment* const kCommEnvironment =
    ::testing::AddGlobalTestEnvironment(new CommEnvironment);

template <class Ranked, class Value>
Ranked filled(Value value) {
  Ranked result{};
  for (int axis = 0; axis < kDim; ++axis)
    result[axis] = value;
  return result;
}

pops::Index<kDim> index_from_ordinal(const pops::Box<kDim>& box, std::size_t ordinal) {
  pops::Index<kDim> result{};
  for (int axis = 0; axis < kDim; ++axis) {
    const std::size_t length = static_cast<std::size_t>(box.length(axis));
    result[axis] = box.lo[axis] + static_cast<int>(ordinal % length);
    ordinal /= length;
  }
  return result;
}

std::size_t storage_ordinal(const pops::Box<kDim>& box, const pops::Index<kDim>& index) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < kDim; ++axis) {
    result += static_cast<std::size_t>(index[axis] - box.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(box.length(axis));
  }
  return result;
}

pops::EllipticBuildRequest<kDim> request(int cells, bool periodic = false,
                                         pops::Real boundary_value = pops::Real(0)) {
  const pops::Box<kDim> domain{pops::Index<kDim>{}, filled<pops::Index<kDim>>(cells - 1)};
  const auto geometry = pops::Geometry<kDim>::from_bounds(
      domain, pops::RealVector<kDim>{}, filled<pops::RealVector<kDim>>(pops::Real(1)));
  const pops::mesh::BoxArray<kDim> layout(std::vector<pops::Box<kDim>>{domain});
  pops::Extent<kDim> rank_extent = filled<pops::Extent<kDim>>(std::int64_t{1});
  rank_extent[0] = pops::n_ranks();
  pops::Index<kDim> local_rank{};
  local_rank[0] = pops::my_rank();
  const pops::mesh::RankSpace<kDim> ranks{pops::Index<kDim>{}, rank_extent};
  const auto distribution = pops::mesh::Distribution<kDim>::replicated(layout, ranks);
  std::array<pops::PhysicalBoundaryFace, 2 * kDim> faces{};
  if (!periodic)
    faces.fill(pops::PhysicalBoundaryFace{pops::PhysicalBoundaryKind::dirichlet, boundary_value});
  std::array<bool, kDim> periodic_axes{};
  periodic_axes.fill(periodic);
  pops::RealVector<kDim> spacing{};
  for (int axis = 0; axis < kDim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  const auto topology = periodic ? pops::BoundaryTopology<kDim>::axis_periodic(periodic_axes)
                                 : pops::BoundaryTopology<kDim>::physical();
  return {geometry,
          layout,
          distribution,
          local_rank,
          pops::PhysicalBoundaryConditions<kDim>{topology, faces, spacing},
          pops::Extent<kDim>{},
          filled<pops::Extent<kDim>>(std::int64_t{1}),
          {layout.size(), 0}};
}

void install_nullspace(Solver& solver, bool periodic) {
  if (!periodic) {
    solver.install_nullspace(pops::FieldNullspacePlan<kDim>{},
                             pops::PreparedVectorDistribution<kDim>::replicated());
    return;
  }
  pops::Real measure = pops::Real(1);
  for (int axis = 0; axis < kDim; ++axis)
    measure *= solver.geom().spacing(axis);
  solver.install_nullspace(
      pops::constant_mean_zero_nullspace<kDim>("periodic-geometric-mg", "unit-test", measure),
      pops::PreparedVectorDistribution<kDim>::replicated());
}

pops::Real exact_mode(const pops::Geometry<kDim>& geometry, const pops::Index<kDim>& index) {
  const pops::Real pi = std::acos(pops::Real(-1));
  pops::Real result = pops::Real(1);
  for (int axis = 0; axis < kDim; ++axis)
    result *= std::sin(pi * geometry.cell_coordinate(axis, index[axis]));
  return result;
}

void fill_dirichlet_mode(Solver& solver) {
  const pops::Real pi = std::acos(pops::Real(-1));
  const pops::Real eigenvalue = static_cast<pops::Real>(kDim) * pi * pi;
  for (std::size_t local = 0; local < solver.rhs().local_size(); ++local) {
    auto& fab = solver.rhs().fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
         ++ordinal) {
      const auto index = index_from_ordinal(fab.box(), ordinal);
      host(storage_ordinal(fab.grown_box(), index)) = eigenvalue * exact_mode(solver.geom(), index);
    }
    fab.copy_from_host(host);
  }
}

double maximum_mode_error(const Solver& solver) {
  double result = 0;
  for (std::size_t local = 0; local < solver.phi().local_size(); ++local) {
    const auto& fab = solver.phi().fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
         ++ordinal) {
      const auto index = index_from_ordinal(fab.box(), ordinal);
      result = std::max(result,
                        std::abs(static_cast<double>(host(storage_ordinal(fab.grown_box(), index)) -
                                                     exact_mode(solver.geom(), index))));
    }
  }
  return pops::all_reduce_max(result);
}

double maximum_difference_from(const Field& field, pops::Real expected) {
  double result = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
         ++ordinal) {
      const auto index = index_from_ordinal(fab.box(), ordinal);
      result = std::max(
          result,
          std::abs(static_cast<double>(host(storage_ordinal(fab.grown_box(), index)) - expected)));
    }
  }
  return pops::all_reduce_max(result);
}

std::string canonical_valid_bytes(const Field& field) {
  std::string bytes;
  for (std::size_t global = 0; global < field.layout().size(); ++global) {
    const std::size_t local = field.local_index_of(global);
    if (local == Field::not_local)
      return {};
    const auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    for (int component = 0; component < field.ncomp(); ++component) {
      const std::size_t component_offset =
          static_cast<std::size_t>(component) * static_cast<std::size_t>(fab.grown_box().numPts());
      for (std::size_t ordinal = 0; ordinal < static_cast<std::size_t>(fab.box().numPts());
           ++ordinal) {
        const auto index = index_from_ordinal(fab.box(), ordinal);
        const pops::Real value = host(component_offset + storage_ordinal(fab.grown_box(), index));
        bytes.append(reinterpret_cast<const char*>(&value), sizeof(value));
      }
    }
  }
  return bytes;
}

bool replicas_agree_exactly(const Field& field) {
  const std::string bytes = canonical_valid_bytes(field);
  return !bytes.empty() && pops::all_ranks_agree_exact_ordered_byte_pairs(
                               {{std::string_view("geometric-mg-phi"), bytes}});
}

}  // namespace

TEST(GeometricMgCollectiveContract, MpiRouteInitializesRequestedCommunicator) {
  const char* expected_ranks = std::getenv("POPS_TEST_EXPECT_RANKS");
  if (expected_ranks != nullptr)
    EXPECT_EQ(pops::n_ranks(), std::atoi(expected_ranks));
  else if (pops::n_ranks() == 1)
    GTEST_SKIP() << "the serial registration has no remote rank";
}

TEST(GeometricMgTest, prepared_options_fail_closed_before_the_hierarchy_is_published) {
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.geometric-mg.options");
  pops::elliptic::mg::GeometricMultigridOptions options;
  options.maximum_cycles = 0;
  EXPECT_THROW((void)Solver(request(8), lane, options), std::exception);
  options = {};
  options.relative_tolerance = std::numeric_limits<pops::Real>::quiet_NaN();
  EXPECT_THROW((void)Solver(request(8), lane, options), std::exception);
  options = {};
  options.absolute_tolerance = pops::Real(-1);
  EXPECT_THROW((void)Solver(request(8), lane, options), std::exception);
}

TEST(GeometricMgCollectiveContract,
     manufactured_dirichlet_mode_converges_and_publishes_exact_replicas) {
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.geometric-mg.manufactured");
  pops::elliptic::mg::GeometricMultigridOptions options;
  options.relative_tolerance = pops::Real(1e-10);
  options.absolute_tolerance = pops::Real(1e-12);
  options.maximum_cycles = 100;
  options.bottom_sweeps = 60;
  auto build = request(16);
  const auto expected_contract = Solver::expected_operator_contract(build, options);
  Solver solver(std::move(build), lane, options);
  install_nullspace(solver, false);
  EXPECT_EQ(solver.prepared_operator_contract().exact_fingerprint(),
            expected_contract.exact_fingerprint());
  fill_dirichlet_mode(solver);
  solver.phi().set_val(pops::Real(0));

  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_GT(report.iters, 0);
  EXPECT_LT(report.rel_residual, pops::Real(1e-10));
  EXPECT_LT(maximum_mode_error(solver), 0.01);
  EXPECT_TRUE(replicas_agree_exactly(solver.phi()));
}

TEST(GeometricMgTest, zero_problem_exits_without_mutating_the_exact_zero_state) {
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.geometric-mg.zero");
  Solver solver(request(8), lane);
  install_nullspace(solver, false);
  solver.rhs().set_val(pops::Real(0));
  solver.phi().set_val(pops::Real(0));
  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_EQ(report.iters, 0);
  EXPECT_EQ(report.reason, "geometric_mg_initial_residual");
  EXPECT_EQ(maximum_difference_from(solver.phi(), pops::Real(0)), 0.0);
}

TEST(GeometricMgTest, inhomogeneous_dirichlet_faces_recover_one_constant_solution) {
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.geometric-mg.boundary");
  pops::elliptic::mg::GeometricMultigridOptions options;
  options.relative_tolerance = pops::Real(1e-9);
  options.absolute_tolerance = pops::Real(1e-11);
  options.maximum_cycles = 100;
  Solver solver(request(16, false, pops::Real(1)), lane, options);
  install_nullspace(solver, false);
  solver.rhs().set_val(pops::Real(0));
  solver.phi().set_val(pops::Real(0));
  const pops::SolveReport report = solver.solve();
  ASSERT_TRUE(report.solved()) << report.reason;
  EXPECT_LT(maximum_difference_from(solver.phi(), pops::Real(1)), 2e-7);
}

TEST(GeometricMgTest, periodic_nullspace_rejects_incompatible_rhs_without_mutation) {
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.geometric-mg.periodic");
  Solver solver(request(8, true), lane);
  install_nullspace(solver, true);
  solver.rhs().set_val(pops::Real(1));
  solver.phi().set_val(pops::Real(3));
  const pops::SolveReport report = solver.solve();
  EXPECT_EQ(report.status, pops::SolveStatus::kIncompatibleRhs);
  EXPECT_EQ(report.action, pops::SolveAction::kFailRun);
  EXPECT_EQ(maximum_difference_from(solver.phi(), pops::Real(3)), 0.0);
}

TEST(GeometricMgTest, nonfinite_rhs_is_a_structured_fail_run) {
  const pops::ExecutionLane lane = pops::ExecutionLane::world("tests.geometric-mg.nonfinite");
  for (const pops::Real invalid : {std::numeric_limits<pops::Real>::quiet_NaN(),
                                   std::numeric_limits<pops::Real>::infinity()}) {
    Solver solver(request(8), lane);
    install_nullspace(solver, false);
    solver.rhs().set_val(invalid);
    solver.phi().set_val(pops::Real(0));
    const pops::SolveReport report = solver.solve();
    EXPECT_EQ(report.status, pops::SolveStatus::kInvalidEvaluation);
    EXPECT_EQ(report.action, pops::SolveAction::kFailRun);
    EXPECT_TRUE(report.valid());
  }
}
