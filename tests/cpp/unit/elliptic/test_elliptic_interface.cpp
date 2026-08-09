#include <gtest/gtest.h>

#include <pops/numerics/elliptic/interface/elliptic_interface.hpp>
#include <pops/numerics/elliptic/mg/geometric_mg.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

using namespace pops;

namespace {

template <int Dim>
class ExactTestElliptic {
 public:
  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim>;
  using request_type = EllipticBuildRequest<Dim>;

  explicit ExactTestElliptic(request_type request)
      : geometry_(request.geometry),
        boundary_(request.boundary),
        rhs_(request.boxes, request.distribution, request.local_rank, 1, request.rhs_ghosts),
        phi_(request.boxes, request.distribution, request.local_rank, 1, request.phi_ghosts),
        contract_(expected_operator_contract(request)) {}

  ExactTestElliptic(const ExactTestElliptic&) = delete;
  ExactTestElliptic& operator=(const ExactTestElliptic&) = delete;
  ExactTestElliptic(ExactTestElliptic&&) noexcept = default;
  ExactTestElliptic& operator=(ExactTestElliptic&&) noexcept = default;
  ~ExactTestElliptic() noexcept = default;

  static constexpr EllipticOperatorIdentity operator_identity() noexcept {
    return {"pops.test.exact-ranked-elliptic", 1};
  }
  static EllipticOperatorContract expected_operator_contract(const request_type& request) {
    return make_expected_elliptic_operator_contract(operator_identity(), request,
                                                    "test-options-v1");
  }

  field_type& rhs() { return rhs_; }
  field_type& phi() { return phi_; }
  SolveReport solve() {
    residual_ = Real(0);
    report_.mark_solved("exact_test_elliptic");
    return report_;
  }
  Real residual() const { return residual_; }
  int maximum_iterations() const noexcept { return 4; }
  const Geometry<Dim>& geom() const noexcept { return geometry_; }
  const PhysicalBoundaryConditions<Dim>& boundary() const noexcept { return boundary_; }
  const SolveReport& last_solve_report() const noexcept { return report_; }
  const EllipticOperatorContract& prepared_operator_contract() const noexcept { return contract_; }

 private:
  Geometry<Dim> geometry_;
  PhysicalBoundaryConditions<Dim> boundary_;
  field_type rhs_;
  field_type phi_;
  EllipticOperatorContract contract_;
  Real residual_ = Real(1);
  SolveReport report_{};
};

static_assert(EllipticSolver<ExactTestElliptic<1>>);
static_assert(EllipticSolver<ExactTestElliptic<2>>);
static_assert(EllipticSolver<ExactTestElliptic<3>>);
static_assert(EllipticOperator<ExactTestElliptic<1>>);
static_assert(EllipticOperator<ExactTestElliptic<2>>);
static_assert(EllipticOperator<ExactTestElliptic<3>>);
static_assert(LinearSolver<ExactTestElliptic<1>>);
static_assert(LinearSolver<ExactTestElliptic<2>>);
static_assert(LinearSolver<ExactTestElliptic<3>>);
static_assert(FieldPostProcessor<CenteredFieldPostProcessor<1>>);
static_assert(FieldPostProcessor<CenteredFieldPostProcessor<2>>);
static_assert(FieldPostProcessor<CenteredFieldPostProcessor<3>>);
static_assert(EllipticOperator<pops::elliptic::mg::GeometricMG<1>>);
static_assert(EllipticOperator<pops::elliptic::mg::GeometricMG<2>>);
static_assert(EllipticOperator<pops::elliptic::mg::GeometricMG<3>>);
static_assert(LinearSolver<pops::elliptic::mg::GeometricMG<1>>);
static_assert(LinearSolver<pops::elliptic::mg::GeometricMG<2>>);
static_assert(LinearSolver<pops::elliptic::mg::GeometricMG<3>>);
static_assert(EllipticFactory<DefaultEllipticFactory<ExactTestElliptic<3>>, ExactTestElliptic<3>>);

template <int Dim>
EllipticBuildRequest<Dim> request_for_lane(const ExecutionLane& lane) {
  Index<Dim> lower{};
  Index<Dim> upper{};
  Extent<Dim> rank_extent{};
  Index<Dim> local_rank{};
  RealVector<Dim> physical_lower{};
  RealVector<Dim> physical_upper{};
  RealVector<Dim> spacing{};
  Extent<Dim> maximum{};
  Extent<Dim> phi_ghosts{};
  for (int axis = 0; axis < Dim; ++axis) {
    lower[axis] = -2;
    upper[axis] = 3;
    rank_extent[axis] = axis == 0 ? lane.size() : 1;
    local_rank[axis] = axis == 0 ? lane.rank() : 0;
    physical_lower[axis] = Real(-axis);
    physical_upper[axis] = Real(axis + 2);
    spacing[axis] = (physical_upper[axis] - physical_lower[axis]) / Real(6);
    maximum[axis] = 6;
    phi_ghosts[axis] = 1;
  }
  const Box<Dim> domain{lower, upper};
  const mesh::BoxArray<Dim> boxes = mesh::BoxArray<Dim>::from_domain(domain, maximum);
  const mesh::RankSpace<Dim> rank_space(Index<Dim>{}, rank_extent);
  const mesh::Distribution<Dim> distribution =
      mesh::Distribution<Dim>::replicated(boxes, rank_space);
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  faces.fill(PhysicalBoundaryFace{PhysicalBoundaryKind::dirichlet, Real(0)});
  return EllipticBuildRequest<Dim>{
      Geometry<Dim>::from_bounds(domain, physical_lower, physical_upper),
      boxes,
      distribution,
      local_rank,
      PhysicalBoundaryConditions<Dim>{BoundaryTopology<Dim>::physical(), faces, spacing},
      Extent<Dim>{},
      phi_ghosts,
      mesh::BoxArrayValidationBudget{boxes.size(), boxes.size() * (boxes.size() - 1) / 2}};
}

template <int Dim>
void expect_factory_builds_exact_rank() {
  const ExecutionLane lane = ExecutionLane::world();
  auto request = request_for_lane<Dim>(lane);
  const auto expected_geometry = request.geometry;
  const auto expected_layout = request.boxes;
  const auto expected_distribution = request.distribution;
  const auto expected_local_rank = request.local_rank;
  ExactTestElliptic<Dim> solver = make_elliptic_solver<ExactTestElliptic<Dim>>(
      std::move(request), DefaultEllipticFactory<ExactTestElliptic<Dim>>{}, lane);

  EXPECT_EQ(solver.geom(), expected_geometry);
  EXPECT_EQ(solver.rhs().layout(), expected_layout);
  EXPECT_EQ(solver.rhs().distribution(), expected_distribution);
  EXPECT_EQ(solver.rhs().local_rank(), expected_local_rank);
  EXPECT_FALSE(solver.rhs().shares_storage_with(solver.phi()));
  const SolveReport report = solver.solve();
  EXPECT_TRUE(report.solved());
  EXPECT_TRUE(solver.last_solve_report().solved());
  EXPECT_EQ(solver.residual(), Real(0));
}

template <int Dim>
struct WrongGhostFactory {
  using solver_type = ExactTestElliptic<Dim>;
  using request_type = EllipticBuildRequest<Dim>;

  std::string_view collective_contract() const noexcept { return "wrong-ghost-factory@1"; }
  EllipticOperatorContract expected_operator_contract(const request_type& request) const {
    return solver_type::expected_operator_contract(request);
  }
  bool supports(const request_type&) const noexcept { return true; }
  EllipticFactoryBuildResult<solver_type> build(request_type request) const noexcept {
    ++request.phi_ghosts[0];
    return capture_local_elliptic_factory_build<solver_type>(
        [request = std::move(request)]() mutable { return solver_type(std::move(request)); });
  }
};

static_assert(EllipticFactory<WrongGhostFactory<2>, ExactTestElliptic<2>>);

}  // namespace

TEST(test_elliptic_interface, default_factory_materializes_1d_2d_and_3d_exactly) {
  expect_factory_builds_exact_rank<1>();
  expect_factory_builds_exact_rank<2>();
  expect_factory_builds_exact_rank<3>();
}

TEST(test_elliptic_interface, backend_cannot_silently_change_allocated_ghost_contract) {
  const ExecutionLane lane = ExecutionLane::world();
  EXPECT_THROW((void)make_elliptic_solver<ExactTestElliptic<2>>(request_for_lane<2>(lane),
                                                                WrongGhostFactory<2>{}, lane),
               std::invalid_argument);
}

TEST(test_elliptic_interface, invalid_rank_coordinate_and_insufficient_budget_fail_collectively) {
  const ExecutionLane lane = ExecutionLane::world();
  auto wrong_rank = request_for_lane<3>(lane);
  wrong_rank.local_rank[0] = lane.size();
  EXPECT_THROW((void)make_elliptic_solver<ExactTestElliptic<3>>(
                   std::move(wrong_rank), DefaultEllipticFactory<ExactTestElliptic<3>>{}, lane),
               std::invalid_argument);

  auto insufficient_budget = request_for_lane<1>(lane);
  insufficient_budget.layout_budget = {};
  EXPECT_THROW(
      (void)make_elliptic_solver<ExactTestElliptic<1>>(
          std::move(insufficient_budget), DefaultEllipticFactory<ExactTestElliptic<1>>{}, lane),
      std::invalid_argument);
}
