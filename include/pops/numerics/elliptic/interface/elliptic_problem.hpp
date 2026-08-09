/// @file
/// @brief Exact-ranked elliptic problem description and potential post-processing.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/index/real_vector.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace pops {

/// Complete constant-coefficient scalar elliptic problem in one immutable spatial rank.
///
/// The exact materialization request owns geometry, layout, distribution and boundary conditions;
/// they are never repeated as dimension-erased side data.  Nullspaces and gauges deliberately do
/// not appear as a boolean here: a backend that is singular must receive an exact
/// FieldNullspacePlan through its prepared provider before solve.
template <int Dim>
struct EllipticProblem {
  static_assert(Dim >= 1 && Dim <= 3, "EllipticProblem only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;

  EllipticBuildRequest<Dim> build;
  Real eps = Real(1);
};

namespace elliptic_problem_detail {

template <int Dim>
PhysicalBoundaryConditions<Dim> homogeneous_boundary(
    const PhysicalBoundaryConditions<Dim>& boundary) {
  std::array<PhysicalBoundaryFace, PhysicalBoundaryConditions<Dim>::face_count> faces{};
  for (int axis = 0; axis < Dim; ++axis) {
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      PhysicalBoundaryFace law = boundary.at(face);
      law.value = Real(0);
      faces[static_cast<std::size_t>(face.ordinal())] = law;
    }
  }
  return PhysicalBoundaryConditions<Dim>{boundary.topology(), std::move(faces), boundary.spacing()};
}

template <int Dim>
struct FieldPostprocessKernel {
  FieldView<const Real, Dim> potential{};
  FieldView<Real, Dim> output{};
  RealVector<Dim> centered_factors{};
  Real sign = Real(1);
  int gradient_component = 0;
  bool store_potential = false;

  POPS_HD void operator()(const CellIndex<Dim>& cell) const {
    if (store_potential)
      output(cell, 0) = potential(cell, 0);
    for (int axis = 0; axis < Dim; ++axis) {
      CellIndex<Dim> lower = cell;
      CellIndex<Dim> upper = cell;
      --lower[axis];
      ++upper[axis];
      output(cell, gradient_component + axis) =
          sign * (potential(upper, 0) - potential(lower, 0)) * centered_factors[axis];
    }
  }
};

template <int Dim, class MemorySpace>
void require_postprocess_contract(const Geometry<Dim>& geometry,
                                  const MultiFab<Dim, MemorySpace>& potential,
                                  const MultiFab<Dim, MemorySpace>& output, bool store_potential) {
  if (potential.ncomp() != 1)
    throw std::invalid_argument("field_postprocess requires a scalar potential");
  const int expected_components = Dim + (store_potential ? 1 : 0);
  if (output.ncomp() != expected_components)
    throw std::invalid_argument(
        "field_postprocess output component count must equal Dim plus the optional potential");
  if (potential.layout() != output.layout() || potential.distribution() != output.distribution() ||
      potential.local_rank() != output.local_rank())
    throw std::invalid_argument(
        "field_postprocess requires exactly co-distributed potential and output fields");
  for (int axis = 0; axis < Dim; ++axis)
    if (potential.ghosts()[axis] < 1)
      throw std::invalid_argument(
          "field_postprocess requires one potential ghost cell on every axis");
  for (std::size_t local = 0; local < output.local_size(); ++local)
    if (!geometry.domain().contains(output.box(local)))
      throw std::invalid_argument(
          "field_postprocess output patches must lie inside the exact geometry domain");
}

}  // namespace elliptic_problem_detail

/// Homogeneous correction boundary for the exact-ranked problem.  Topology, law kind, Robin
/// coefficients and spacing remain unchanged; only the affine right-hand side is zeroed.
template <int Dim>
PhysicalBoundaryConditions<Dim> homogeneous_bc(const EllipticProblem<Dim>& problem) {
  return elliptic_problem_detail::homogeneous_boundary(problem.build.boundary);
}

/// Build one exact-ranked backend through its authenticated factory declaration.
///
/// This overload adds the PDE-level coefficient guard, then delegates to the canonical factory in
/// elliptic_solver.hpp.  It never selects a backend by a runtime dimension and never constructs a
/// legacy fallback.  Singular backends still require their explicit nullspace provider after build.
template <EllipticSolver Solver, class Factory>
  requires EllipticFactory<std::remove_cvref_t<Factory>, Solver>
Solver make_elliptic_solver(EllipticProblem<Solver::dimension> problem, Factory&& factory,
                            const ExecutionLane& lane) {
  const long unsupported_coefficient = all_reduce_max(problem.eps == Real(1) ? 0L : 1L, lane);
  if (unsupported_coefficient != 0)
    throw std::invalid_argument(
        "EllipticProblem::eps != 1 is unsupported by the constant-coefficient operator");
  return make_elliptic_solver<Solver>(std::move(problem.build), std::forward<Factory>(factory),
                                      lane);
}

/// Convenience route for an exact backend whose constructor is its public factory provider.
template <EllipticSolver Solver>
Solver make_elliptic_solver(EllipticProblem<Solver::dimension> problem, const ExecutionLane& lane) {
  return make_elliptic_solver<Solver>(std::move(problem), DefaultEllipticFactory<Solver>{}, lane);
}

/// Convention used to publish the potential and its centered Cartesian gradient.
struct FieldPostProcess {
  enum class GradSign { Plus, Minus };

  GradSign sign = GradSign::Plus;
  bool store_phi = true;
};

/// Publish ``phi`` (optionally) and ``+/- grad(phi)`` through one axis-generic kernel.
///
/// The geometry supplies each centered factor ``1 / (2 dx_axis)``.  Output component zero stores
/// the potential when requested; the following ``Dim`` components are ordered by spatial axis.
template <int Dim, class MemorySpace>
void field_postprocess(const Geometry<Dim>& geometry, const MultiFab<Dim, MemorySpace>& potential,
                       MultiFab<Dim, MemorySpace>& output, FieldPostProcess spec) {
  elliptic_problem_detail::require_postprocess_contract(geometry, potential, output,
                                                        spec.store_phi);

  RealVector<Dim> centered_factors{};
  for (int axis = 0; axis < Dim; ++axis)
    centered_factors[axis] = Real(1) / (Real(2) * geometry.spacing(axis));
  const Real sign = spec.sign == FieldPostProcess::GradSign::Plus ? Real(1) : Real(-1);
  const int gradient_component = spec.store_phi ? 1 : 0;

  for (std::size_t local = 0; local < output.local_size(); ++local)
    for_each_cell(output.box(local),
                  elliptic_problem_detail::FieldPostprocessKernel<Dim>{
                      potential.fab(local).view(), output.fab(local).view(), centered_factors, sign,
                      gradient_component, spec.store_phi});
}

}  // namespace pops
