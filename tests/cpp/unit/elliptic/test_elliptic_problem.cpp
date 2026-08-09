#include <gtest/gtest.h>

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/rank_space.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using pops::BoundarySide;
using pops::BoundaryTopology;
using pops::Box;
using pops::CellIndex;
using pops::EllipticBuildRequest;
using pops::EllipticOperatorContract;
using pops::EllipticOperatorIdentity;
using pops::EllipticProblem;
using pops::ExecutionLane;
using pops::Extent;
using pops::Face;
using pops::FieldPostProcess;
using pops::FieldView;
using pops::Geometry;
using pops::Index;
using pops::MultiFab;
using pops::PhysicalBoundaryConditions;
using pops::PhysicalBoundaryFace;
using pops::PhysicalBoundaryKind;
using pops::Real;
using pops::RealVector;
using pops::mesh::BoxArray;
using pops::mesh::Distribution;
using pops::mesh::RankSpace;

template <int Dim>
Index<Dim> filled_index(int value) {
  Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
Extent<Dim> filled_extent(std::int64_t value) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
RealVector<Dim> filled_real(Real value) {
  RealVector<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
EllipticBuildRequest<Dim> build_request(const ExecutionLane& lane) {
  const Box<Dim> domain{filled_index<Dim>(0), filled_index<Dim>(5)};
  const BoxArray<Dim> boxes(std::vector<Box<Dim>>{domain});
  Extent<Dim> rank_extents = filled_extent<Dim>(1);
  rank_extents[0] = lane.size();
  Index<Dim> local_rank{};
  local_rank[0] = lane.rank();
  const Distribution<Dim> distribution =
      Distribution<Dim>::replicated(boxes, RankSpace<Dim>{Index<Dim>{}, rank_extents});
  std::array<PhysicalBoundaryFace, PhysicalBoundaryConditions<Dim>::face_count> faces{};
  faces.fill(PhysicalBoundaryFace{PhysicalBoundaryKind::dirichlet, Real(3)});
  const Geometry<Dim> geometry =
      Geometry<Dim>::from_bounds(domain, filled_real<Dim>(Real(-1)), filled_real<Dim>(Real(2)));
  RealVector<Dim> spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    spacing[axis] = geometry.spacing(axis);
  return {geometry,
          boxes,
          distribution,
          local_rank,
          PhysicalBoundaryConditions<Dim>{BoundaryTopology<Dim>::physical(), faces, spacing},
          Extent<Dim>{},
          filled_extent<Dim>(1),
          {1, 0}};
}

template <int Dim>
class ExactProblemSolver {
 public:
  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim>;
  using request_type = EllipticBuildRequest<Dim>;

  explicit ExactProblemSolver(request_type request)
      : geometry_(request.geometry),
        boundary_(request.boundary),
        rhs_(request.boxes, request.distribution, request.local_rank, 1, request.rhs_ghosts),
        phi_(request.boxes, request.distribution, request.local_rank, 1, request.phi_ghosts),
        contract_(expected_operator_contract(request)) {}

  ExactProblemSolver(const ExactProblemSolver&) = delete;
  ExactProblemSolver& operator=(const ExactProblemSolver&) = delete;
  ExactProblemSolver(ExactProblemSolver&&) noexcept = default;
  ExactProblemSolver& operator=(ExactProblemSolver&&) noexcept = default;
  ~ExactProblemSolver() noexcept = default;

  static EllipticOperatorContract expected_operator_contract(const request_type& request) {
    return pops::make_expected_elliptic_operator_contract(
        EllipticOperatorIdentity{"pops.test.elliptic-problem.exact", 1}, request);
  }

  field_type& rhs() noexcept { return rhs_; }
  field_type& phi() noexcept { return phi_; }
  void solve() noexcept { residual_ = Real(0); }
  Real residual() const noexcept { return residual_; }
  const Geometry<Dim>& geom() const noexcept { return geometry_; }
  const PhysicalBoundaryConditions<Dim>& boundary() const noexcept { return boundary_; }
  const EllipticOperatorContract& prepared_operator_contract() const noexcept { return contract_; }

 private:
  Geometry<Dim> geometry_;
  PhysicalBoundaryConditions<Dim> boundary_;
  field_type rhs_;
  field_type phi_;
  EllipticOperatorContract contract_;
  Real residual_ = Real(1);
};

static_assert(pops::EllipticSolver<ExactProblemSolver<1>>);
static_assert(pops::EllipticSolver<ExactProblemSolver<2>>);
static_assert(pops::EllipticSolver<ExactProblemSolver<3>>);

template <int Dim>
void expect_exact_problem_factory() {
  const ExecutionLane lane = ExecutionLane::world();
  auto request = build_request<Dim>(lane);
  const Geometry<Dim> geometry = request.geometry;
  const auto boundary = request.boundary;
  ExactProblemSolver<Dim> solver = pops::make_elliptic_solver<ExactProblemSolver<Dim>>(
      EllipticProblem<Dim>{std::move(request), Real(1)}, lane);
  EXPECT_EQ(solver.geom(), geometry);
  EXPECT_EQ(solver.boundary(), boundary);
  EXPECT_EQ(solver.phi().ghosts(), filled_extent<Dim>(1));
}

template <int Dim>
void expect_homogeneous_boundary() {
  const ExecutionLane lane = ExecutionLane::world();
  EllipticProblem<Dim> problem{build_request<Dim>(lane), Real(1)};
  const auto homogeneous = pops::homogeneous_bc(problem);
  EXPECT_EQ(homogeneous.topology(), problem.build.boundary.topology());
  EXPECT_EQ(homogeneous.spacing(), problem.build.boundary.spacing());
  for (int axis = 0; axis < Dim; ++axis) {
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      EXPECT_EQ(homogeneous.at(face).kind, problem.build.boundary.at(face).kind);
      EXPECT_EQ(homogeneous.at(face).value, Real(0));
      EXPECT_EQ(homogeneous.at(face).alpha, problem.build.boundary.at(face).alpha);
      EXPECT_EQ(homogeneous.at(face).beta, problem.build.boundary.at(face).beta);
    }
  }
}

template <int Dim>
struct PolynomialPotential {
  FieldView<Real, Dim> value{};

  POPS_HD void operator()(const CellIndex<Dim>& cell) const {
    Real result = Real(7);
    for (int axis = 0; axis < Dim; ++axis)
      result += Real(axis + 1) * Real(cell[axis] * cell[axis]);
    value(cell, 0) = result;
  }
};

template <int Dim>
struct PostprocessError {
  FieldView<const Real, Dim> result{};
  Geometry<Dim> geometry;
  Real sign = Real(1);
  int gradient_component = 0;
  bool stores_potential = false;

  POPS_HD Real operator()(const CellIndex<Dim>& cell) const {
    Real error = Real(0);
    if (stores_potential) {
      Real expected = Real(7);
      for (int axis = 0; axis < Dim; ++axis)
        expected += Real(axis + 1) * Real(cell[axis] * cell[axis]);
      const Real difference = result(cell, 0) - expected;
      error = difference < Real(0) ? -difference : difference;
    }
    for (int axis = 0; axis < Dim; ++axis) {
      const Real expected = sign * Real(2 * (axis + 1) * cell[axis]) / geometry.spacing(axis);
      const Real difference = result(cell, gradient_component + axis) - expected;
      const Real magnitude = difference < Real(0) ? -difference : difference;
      error = error < magnitude ? magnitude : error;
    }
    return error;
  }
};

template <int Dim>
void expect_postprocess() {
  const ExecutionLane lane = ExecutionLane::world();
  const auto request = build_request<Dim>(lane);
  MultiFab<Dim> potential(request.boxes, request.distribution, request.local_rank, 1,
                          filled_extent<Dim>(1));
  for (std::size_t local = 0; local < potential.local_size(); ++local)
    pops::for_each_cell(potential.fab(local).grown_box(),
                        PolynomialPotential<Dim>{potential.fab(local).view()});

  MultiFab<Dim> plus(request.boxes, request.distribution, request.local_rank, Dim + 1,
                     Extent<Dim>{});
  pops::field_postprocess(request.geometry, potential, plus,
                          FieldPostProcess{FieldPostProcess::GradSign::Plus, true});
  Real plus_error = Real(0);
  for (std::size_t local = 0; local < plus.local_size(); ++local)
    plus_error =
        std::max(plus_error,
                 pops::for_each_cell_reduce_max(
                     plus.box(local), PostprocessError<Dim>{std::as_const(plus.fab(local)).view(),
                                                            request.geometry, Real(1), 1, true}));
  EXPECT_EQ(plus_error, Real(0));

  MultiFab<Dim> minus(request.boxes, request.distribution, request.local_rank, Dim, Extent<Dim>{});
  pops::field_postprocess(request.geometry, potential, minus,
                          FieldPostProcess{FieldPostProcess::GradSign::Minus, false});
  Real minus_error = Real(0);
  for (std::size_t local = 0; local < minus.local_size(); ++local)
    minus_error = std::max(
        minus_error,
        pops::for_each_cell_reduce_max(
            minus.box(local), PostprocessError<Dim>{std::as_const(minus.fab(local)).view(),
                                                    request.geometry, Real(-1), 0, false}));
  EXPECT_EQ(minus_error, Real(0));
}

}  // namespace

TEST(test_elliptic_problem, exact_factory_materializes_one_two_and_three_dimensions) {
  expect_exact_problem_factory<1>();
  expect_exact_problem_factory<2>();
  expect_exact_problem_factory<3>();
}

TEST(test_elliptic_problem, unsupported_coefficient_fails_closed) {
  const ExecutionLane lane = ExecutionLane::world();
  EXPECT_THROW((void)pops::make_elliptic_solver<ExactProblemSolver<3>>(
                   EllipticProblem<3>{build_request<3>(lane), Real(2)}, lane),
               std::invalid_argument);
}

TEST(test_elliptic_problem, homogeneous_boundary_retains_every_exact_ranked_law) {
  expect_homogeneous_boundary<1>();
  expect_homogeneous_boundary<2>();
  expect_homogeneous_boundary<3>();
}

TEST(test_elliptic_problem, one_axis_generic_postprocess_executes_in_one_two_and_three_dimensions) {
  expect_postprocess<1>();
  expect_postprocess<2>();
  expect_postprocess<3>();
}

TEST(test_elliptic_problem, postprocess_rejects_missing_axis_ghosts) {
  const ExecutionLane lane = ExecutionLane::world();
  const auto request = build_request<2>(lane);
  MultiFab<2> potential(request.boxes, request.distribution, request.local_rank, 1, Extent<2>{});
  MultiFab<2> output(request.boxes, request.distribution, request.local_rank, 3, Extent<2>{});
  EXPECT_THROW(pops::field_postprocess(request.geometry, potential, output, FieldPostProcess{}),
               std::invalid_argument);
}
