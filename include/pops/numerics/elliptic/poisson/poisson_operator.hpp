/// @file
/// @brief Allocation-free compile-time-ranked Cartesian Poisson operator kernels.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace pops::elliptic::mg {

namespace detail {

template <int Dim>
struct CopyScalarKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> source{};

  POPS_HD void operator()(const Index<Dim>& index) const {
    destination(index, 0) = source(index, 0);
  }
};

template <int Dim>
struct AddScalarKernel {
  FieldView<Real, Dim> values{};
  Real increment = Real(0);

  POPS_HD void operator()(const Index<Dim>& index) const { values(index, 0) += increment; }
};

template <int Dim>
struct NegativeLaplacianKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> source{};
  Real inverse_spacing_squared[Dim]{};
  Real reaction = Real(0);

  POPS_HD void operator()(const Index<Dim>& index) const {
    Real value = reaction * source(index, 0);
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> lower = index;
      Index<Dim> upper = index;
      --lower[axis];
      ++upper[axis];
      value += (Real(2) * source(index, 0) - source(lower, 0) - source(upper, 0)) *
               inverse_spacing_squared[axis];
    }
    destination(index, 0) = value;
  }
};

template <int Dim>
struct ResidualKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> iterate{};
  FieldView<const Real, Dim> right_hand_side{};
  Real inverse_spacing_squared[Dim]{};
  Real reaction = Real(0);

  POPS_HD void operator()(const Index<Dim>& index) const {
    Real image = reaction * iterate(index, 0);
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> lower = index;
      Index<Dim> upper = index;
      --lower[axis];
      ++upper[axis];
      image += (Real(2) * iterate(index, 0) - iterate(lower, 0) - iterate(upper, 0)) *
               inverse_spacing_squared[axis];
    }
    destination(index, 0) = right_hand_side(index, 0) - image;
  }
};

template <int Dim>
struct DampedJacobiKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> iterate{};
  FieldView<const Real, Dim> right_hand_side{};
  Real inverse_spacing_squared[Dim]{};
  Real inverse_diagonal = Real(0);
  Real relaxation = Real(2) / Real(3);
  Real reaction = Real(0);

  POPS_HD void operator()(const Index<Dim>& index) const {
    Real image = reaction * iterate(index, 0);
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> lower = index;
      Index<Dim> upper = index;
      --lower[axis];
      ++upper[axis];
      image += (Real(2) * iterate(index, 0) - iterate(lower, 0) - iterate(upper, 0)) *
               inverse_spacing_squared[axis];
    }
    destination(index, 0) =
        iterate(index, 0) + relaxation * inverse_diagonal * (right_hand_side(index, 0) - image);
  }
};

template <int Dim, class LeftSpace, class RightSpace>
void require_same_scalar_layout(const MultiFab<Dim, LeftSpace>& left,
                                const MultiFab<Dim, RightSpace>& right,
                                const char* operation) {
  if (left.layout() != right.layout() || left.distribution() != right.distribution() ||
      left.local_rank() != right.local_rank() || left.ncomp() != 1 || right.ncomp() != 1)
    throw std::invalid_argument(std::string(operation) +
                                " requires one exact scalar ND field layout");
}

template <int Dim, class MemorySpace>
void require_unit_halo(const MultiFab<Dim, MemorySpace>& field, const char* operation) {
  for (int axis = 0; axis < Dim; ++axis)
    if (field.ghosts()[axis] < 1)
      throw std::invalid_argument(std::string(operation) +
                                  " requires at least one ghost cell on every axis");
}

template <int Dim>
void inverse_spacing_squared(const Geometry<Dim>& geometry, Real (&result)[Dim]) {
  for (int axis = 0; axis < Dim; ++axis) {
    const Real spacing = geometry.spacing(axis);
    if (!std::isfinite(static_cast<double>(spacing)) || !(spacing > Real(0)))
      throw std::invalid_argument("Poisson operator requires finite positive spacing");
    const Real inverse = Real(1) / spacing;
    result[axis] = inverse * inverse;
  }
}

}  // namespace detail

/// Copy valid scalar cells between fields retaining the same exact ranked layout.
template <int Dim, class SourceSpace, class DestinationSpace>
void copy_scalar_valid(const MultiFab<Dim, SourceSpace>& source,
                       MultiFab<Dim, DestinationSpace>& destination) {
  detail::require_same_scalar_layout(source, destination, "copy_scalar_valid");
  for (std::size_t local = 0; local < source.local_size(); ++local)
    for_each_cell(source.box(local), detail::CopyScalarKernel<Dim>{
                                         destination.fab(local).view(), source.fab(local).view()});
  Kokkos::fence();
}

/// Add one constant to every valid scalar cell.
template <int Dim, class MemorySpace>
void add_scalar_valid(MultiFab<Dim, MemorySpace>& values, Real increment) {
  if (values.ncomp() != 1)
    throw std::invalid_argument("add_scalar_valid requires a scalar field");
  for (std::size_t local = 0; local < values.local_size(); ++local)
    for_each_cell(values.box(local),
                  detail::AddScalarKernel<Dim>{values.fab(local).view(), increment});
  Kokkos::fence();
}

/// Apply ``A u = -laplacian(u) + reaction*u`` after the caller has filled input ghosts.
template <int Dim, class InputSpace, class OutputSpace>
void apply_poisson_operator_valid(const MultiFab<Dim, InputSpace>& input,
                                  const Geometry<Dim>& geometry,
                                  MultiFab<Dim, OutputSpace>& output,
                                  Real reaction = Real(0)) {
  detail::require_same_scalar_layout(input, output, "apply_poisson_operator_valid");
  detail::require_unit_halo(input, "apply_poisson_operator_valid");
  if (!std::isfinite(static_cast<double>(reaction)) || reaction < Real(0))
    throw std::invalid_argument("Poisson reaction must be finite and non-negative");
  Real inverse_spacing_squared[Dim]{};
  detail::inverse_spacing_squared(geometry, inverse_spacing_squared);
  for (std::size_t local = 0; local < output.local_size(); ++local) {
    detail::NegativeLaplacianKernel<Dim> kernel{output.fab(local).view(),
                                                input.fab(local).view(), {}, reaction};
    for (int axis = 0; axis < Dim; ++axis)
      kernel.inverse_spacing_squared[axis] = inverse_spacing_squared[axis];
    for_each_cell(output.box(local), kernel);
  }
  Kokkos::fence();
}

/// Compute ``rhs - A u`` after the caller has filled iterate ghosts.
template <int Dim, class IterateSpace, class RhsSpace, class ResidualSpace>
void poisson_residual_valid(const MultiFab<Dim, IterateSpace>& iterate,
                            const MultiFab<Dim, RhsSpace>& right_hand_side,
                            const Geometry<Dim>& geometry,
                            MultiFab<Dim, ResidualSpace>& residual,
                            Real reaction = Real(0)) {
  detail::require_same_scalar_layout(iterate, right_hand_side, "poisson_residual_valid");
  detail::require_same_scalar_layout(iterate, residual, "poisson_residual_valid");
  detail::require_unit_halo(iterate, "poisson_residual_valid");
  if (!std::isfinite(static_cast<double>(reaction)) || reaction < Real(0))
    throw std::invalid_argument("Poisson reaction must be finite and non-negative");
  Real inverse_spacing_squared[Dim]{};
  detail::inverse_spacing_squared(geometry, inverse_spacing_squared);
  for (std::size_t local = 0; local < residual.local_size(); ++local) {
    detail::ResidualKernel<Dim> kernel{residual.fab(local).view(),
                                       iterate.fab(local).view(),
                                       right_hand_side.fab(local).view(), {}, reaction};
    for (int axis = 0; axis < Dim; ++axis)
      kernel.inverse_spacing_squared[axis] = inverse_spacing_squared[axis];
    for_each_cell(residual.box(local), kernel);
  }
  Kokkos::fence();
}

/// Submit one allocation-free damped-Jacobi update after iterate ghosts are valid.
template <int Dim, class IterateSpace, class RhsSpace, class DestinationSpace>
void damped_jacobi_update_valid(const MultiFab<Dim, IterateSpace>& iterate,
                                const MultiFab<Dim, RhsSpace>& right_hand_side,
                                const Geometry<Dim>& geometry,
                                MultiFab<Dim, DestinationSpace>& destination,
                                Real relaxation = Real(2) / Real(3),
                                Real reaction = Real(0)) {
  detail::require_same_scalar_layout(iterate, right_hand_side, "damped_jacobi_update_valid");
  detail::require_same_scalar_layout(iterate, destination, "damped_jacobi_update_valid");
  detail::require_unit_halo(iterate, "damped_jacobi_update_valid");
  if (!std::isfinite(static_cast<double>(relaxation)) || !(relaxation > Real(0)) ||
      !(relaxation < Real(2)) || !std::isfinite(static_cast<double>(reaction)) ||
      reaction < Real(0))
    throw std::invalid_argument("damped Jacobi controls are invalid");
  Real inverse_spacing_squared[Dim]{};
  detail::inverse_spacing_squared(geometry, inverse_spacing_squared);
  Real diagonal = reaction;
  for (int axis = 0; axis < Dim; ++axis)
    diagonal += Real(2) * inverse_spacing_squared[axis];
  const Real inverse_diagonal = Real(1) / diagonal;
  for (std::size_t local = 0; local < destination.local_size(); ++local) {
    detail::DampedJacobiKernel<Dim> kernel{
        destination.fab(local).view(), iterate.fab(local).view(),
        right_hand_side.fab(local).view(), {}, inverse_diagonal, relaxation, reaction};
    for (int axis = 0; axis < Dim; ++axis)
      kernel.inverse_spacing_squared[axis] = inverse_spacing_squared[axis];
    for_each_cell(destination.box(local), kernel);
  }
  Kokkos::fence();
}

}  // namespace pops::elliptic::mg
