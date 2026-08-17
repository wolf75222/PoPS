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

POPS_HD inline Real harmonic_face_coefficient(Real left, Real right) {
  const Real denominator = left + right;
  return denominator != Real(0) ? Real(2) * left * right / denominator : Real(0);
}

template <int Dim>
POPS_HD Real view_or_one(const FieldView<const Real, Dim>& field, const Index<Dim>& index,
                         int component) {
  return field.data == nullptr ? Real(1) : field(index, component);
}

/// Cell-centered conductivity with harmonic faces, optional cut-cell measure and apertures.
/// Absent optional views are treated as the constant-coefficient Cartesian operator (k=1, a=1,
/// inv_vol=1). Face coefficients are formed from cell values; this never recenters a face field
/// onto cells.
template <int Dim>
struct WeightedPoissonKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> iterate{};
  FieldView<const Real, Dim> right_hand_side{};
  FieldView<const Real, Dim> coefficient{};
  FieldView<const Real, Dim> inverse_volume{};
  FieldView<const Real, Dim> aperture_lower{};
  FieldView<const Real, Dim> aperture_upper{};
  FieldView<const Real, Dim> active{};
  FieldView<const Real, Dim> covered{};
  Real inverse_spacing_squared[Dim]{};
  Real reaction = Real(0);
  Real relaxation = Real(2) / Real(3);
  bool write_residual = false;
  bool write_jacobi = false;

  POPS_HD void operator()(const Index<Dim>& index) const {
    if (covered.data != nullptr && covered(index, 0) >= Real(0.5)) {
      if (write_jacobi)
        destination(index, 0) = iterate(index, 0);
      else
        destination(index, 0) = Real(0);
      return;
    }
    if (active.data != nullptr && active(index, 0) < Real(0.5)) {
      if (write_jacobi)
        destination(index, 0) = iterate(index, 0);
      else
        destination(index, 0) = Real(0);
      return;
    }

    const Real center = iterate(index, 0);
    const Real inv_vol = view_or_one(inverse_volume, index, 0);
    const Real center_k = view_or_one(coefficient, index, 0);
    Real image = reaction * center;
    Real diagonal = reaction;
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> lower = index;
      Index<Dim> upper = index;
      --lower[axis];
      ++upper[axis];
      const Real lower_k = view_or_one(coefficient, lower, 0);
      const Real upper_k = view_or_one(coefficient, upper, 0);
      const Real lower_face = harmonic_face_coefficient(lower_k, center_k);
      const Real upper_face = harmonic_face_coefficient(center_k, upper_k);
      const Real lower_aperture = view_or_one(aperture_lower, index, axis);
      const Real upper_aperture = view_or_one(aperture_upper, index, axis);
      const Real scale = inv_vol * inverse_spacing_squared[axis];
      image += scale * (lower_aperture * lower_face * (center - iterate(lower, 0)) +
                        upper_aperture * upper_face * (center - iterate(upper, 0)));
      diagonal += scale * (lower_aperture * lower_face + upper_aperture * upper_face);
    }
    if (write_jacobi) {
      const Real inverse_diagonal = diagonal != Real(0) ? Real(1) / diagonal : Real(0);
      destination(index, 0) =
          center + relaxation * inverse_diagonal * (right_hand_side(index, 0) - image);
      return;
    }
    destination(index, 0) = write_residual ? right_hand_side(index, 0) - image : image;
  }
};

}  // namespace detail

/// Optional cell-centered conductivity and cut-cell metric for the weighted Poisson stencil.
/// Coefficient ghosts are required. Inverse volume and apertures are valid-cell-only and must not
/// advertise ghosts. Every present field must share the iterate layout; a missing EB authority
/// with any other EB field present is refused.
template <int Dim, class MemorySpace>
struct WeightedPoissonFields {
  const MultiFab<Dim, MemorySpace>* coefficient = nullptr;
  const MultiFab<Dim, MemorySpace>* inverse_volume = nullptr;
  const MultiFab<Dim, MemorySpace>* aperture_lower = nullptr;
  const MultiFab<Dim, MemorySpace>* aperture_upper = nullptr;
  const MultiFab<Dim, MemorySpace>* active = nullptr;
  const MultiFab<Dim, MemorySpace>* covered = nullptr;
};

template <int Dim, class MemorySpace>
void validate_weighted_poisson_fields(const MultiFab<Dim, MemorySpace>& iterate,
                                      const WeightedPoissonFields<Dim, MemorySpace>& fields,
                                      const char* operation) {
  const auto require_layout = [&](const MultiFab<Dim, MemorySpace>* field, int components,
                                  bool require_ghosts, bool forbid_ghosts, const char* name) {
    if (field == nullptr)
      return;
    if (field->layout() != iterate.layout() || field->distribution() != iterate.distribution() ||
        field->local_rank() != iterate.local_rank() || field->ncomp() != components)
      throw std::invalid_argument(std::string(operation) + " " + name +
                                  " must share the iterate layout");
    for (int axis = 0; axis < Dim; ++axis) {
      if (require_ghosts && field->ghosts()[axis] < 1)
        throw std::invalid_argument(std::string(operation) + " " + name +
                                    " requires at least one ghost on every axis");
      if (forbid_ghosts && field->ghosts()[axis] != 0)
        throw std::invalid_argument(std::string(operation) + " " + name +
                                    " is valid-cell-only and must not advertise ghosts");
    }
  };
  require_layout(fields.coefficient, 1, true, false, "coefficient");
  require_layout(fields.inverse_volume, 1, false, true, "inverse_volume");
  require_layout(fields.aperture_lower, Dim, false, true, "aperture_lower");
  require_layout(fields.aperture_upper, Dim, false, true, "aperture_upper");
  require_layout(fields.active, 1, false, false, "active");
  require_layout(fields.covered, 1, false, true, "covered");
  const bool any_eb = fields.inverse_volume != nullptr || fields.aperture_lower != nullptr ||
                      fields.aperture_upper != nullptr;
  if (any_eb && (fields.inverse_volume == nullptr || fields.aperture_lower == nullptr ||
                 fields.aperture_upper == nullptr || fields.active == nullptr))
    throw std::invalid_argument(std::string(operation) +
                                " embedded-boundary Poisson requires inverse volume, both face "
                                "apertures, and an active mask");
}

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

template <int Dim, class MemorySpace>
void apply_weighted_poisson_operator_valid(
    const MultiFab<Dim, MemorySpace>& input, const Geometry<Dim>& geometry,
    MultiFab<Dim, MemorySpace>& output, Real reaction,
    const WeightedPoissonFields<Dim, MemorySpace>& fields) {
  detail::require_same_scalar_layout(input, output, "apply_weighted_poisson_operator_valid");
  detail::require_unit_halo(input, "apply_weighted_poisson_operator_valid");
  validate_weighted_poisson_fields(input, fields, "apply_weighted_poisson_operator_valid");
  if (!std::isfinite(static_cast<double>(reaction)) || reaction < Real(0))
    throw std::invalid_argument("Poisson reaction must be finite and non-negative");
  Real inverse_spacing_squared[Dim]{};
  detail::inverse_spacing_squared(geometry, inverse_spacing_squared);
  for (std::size_t local = 0; local < output.local_size(); ++local) {
    detail::WeightedPoissonKernel<Dim> kernel{};
    kernel.destination = output.fab(local).view();
    kernel.iterate = input.fab(local).view();
    if (fields.coefficient != nullptr)
      kernel.coefficient = fields.coefficient->fab(local).view();
    if (fields.inverse_volume != nullptr)
      kernel.inverse_volume = fields.inverse_volume->fab(local).view();
    if (fields.aperture_lower != nullptr)
      kernel.aperture_lower = fields.aperture_lower->fab(local).view();
    if (fields.aperture_upper != nullptr)
      kernel.aperture_upper = fields.aperture_upper->fab(local).view();
    if (fields.active != nullptr)
      kernel.active = fields.active->fab(local).view();
    if (fields.covered != nullptr)
      kernel.covered = fields.covered->fab(local).view();
    kernel.reaction = reaction;
    for (int axis = 0; axis < Dim; ++axis)
      kernel.inverse_spacing_squared[axis] = inverse_spacing_squared[axis];
    for_each_cell(output.box(local), kernel);
  }
  Kokkos::fence();
}

template <int Dim, class MemorySpace>
void weighted_poisson_residual_valid(const MultiFab<Dim, MemorySpace>& iterate,
                                     const MultiFab<Dim, MemorySpace>& right_hand_side,
                                     const Geometry<Dim>& geometry,
                                     MultiFab<Dim, MemorySpace>& residual, Real reaction,
                                     const WeightedPoissonFields<Dim, MemorySpace>& fields) {
  detail::require_same_scalar_layout(iterate, right_hand_side, "weighted_poisson_residual_valid");
  detail::require_same_scalar_layout(iterate, residual, "weighted_poisson_residual_valid");
  detail::require_unit_halo(iterate, "weighted_poisson_residual_valid");
  validate_weighted_poisson_fields(iterate, fields, "weighted_poisson_residual_valid");
  if (!std::isfinite(static_cast<double>(reaction)) || reaction < Real(0))
    throw std::invalid_argument("Poisson reaction must be finite and non-negative");
  Real inverse_spacing_squared[Dim]{};
  detail::inverse_spacing_squared(geometry, inverse_spacing_squared);
  for (std::size_t local = 0; local < residual.local_size(); ++local) {
    detail::WeightedPoissonKernel<Dim> kernel{};
    kernel.destination = residual.fab(local).view();
    kernel.iterate = iterate.fab(local).view();
    kernel.right_hand_side = right_hand_side.fab(local).view();
    if (fields.coefficient != nullptr)
      kernel.coefficient = fields.coefficient->fab(local).view();
    if (fields.inverse_volume != nullptr)
      kernel.inverse_volume = fields.inverse_volume->fab(local).view();
    if (fields.aperture_lower != nullptr)
      kernel.aperture_lower = fields.aperture_lower->fab(local).view();
    if (fields.aperture_upper != nullptr)
      kernel.aperture_upper = fields.aperture_upper->fab(local).view();
    if (fields.active != nullptr)
      kernel.active = fields.active->fab(local).view();
    if (fields.covered != nullptr)
      kernel.covered = fields.covered->fab(local).view();
    kernel.reaction = reaction;
    kernel.write_residual = true;
    for (int axis = 0; axis < Dim; ++axis)
      kernel.inverse_spacing_squared[axis] = inverse_spacing_squared[axis];
    for_each_cell(residual.box(local), kernel);
  }
  Kokkos::fence();
}

template <int Dim, class MemorySpace>
void damped_jacobi_weighted_update_valid(const MultiFab<Dim, MemorySpace>& iterate,
                                         const MultiFab<Dim, MemorySpace>& right_hand_side,
                                         const Geometry<Dim>& geometry,
                                         MultiFab<Dim, MemorySpace>& destination,
                                         Real relaxation, Real reaction,
                                         const WeightedPoissonFields<Dim, MemorySpace>& fields) {
  detail::require_same_scalar_layout(iterate, right_hand_side, "damped_jacobi_weighted_update_valid");
  detail::require_same_scalar_layout(iterate, destination, "damped_jacobi_weighted_update_valid");
  detail::require_unit_halo(iterate, "damped_jacobi_weighted_update_valid");
  validate_weighted_poisson_fields(iterate, fields, "damped_jacobi_weighted_update_valid");
  if (!std::isfinite(static_cast<double>(relaxation)) || !(relaxation > Real(0)) ||
      !(relaxation < Real(2)) || !std::isfinite(static_cast<double>(reaction)) ||
      reaction < Real(0))
    throw std::invalid_argument("damped Jacobi controls are invalid");
  Real inverse_spacing_squared[Dim]{};
  detail::inverse_spacing_squared(geometry, inverse_spacing_squared);
  for (std::size_t local = 0; local < destination.local_size(); ++local) {
    detail::WeightedPoissonKernel<Dim> kernel{};
    kernel.destination = destination.fab(local).view();
    kernel.iterate = iterate.fab(local).view();
    kernel.right_hand_side = right_hand_side.fab(local).view();
    if (fields.coefficient != nullptr)
      kernel.coefficient = fields.coefficient->fab(local).view();
    if (fields.inverse_volume != nullptr)
      kernel.inverse_volume = fields.inverse_volume->fab(local).view();
    if (fields.aperture_lower != nullptr)
      kernel.aperture_lower = fields.aperture_lower->fab(local).view();
    if (fields.aperture_upper != nullptr)
      kernel.aperture_upper = fields.aperture_upper->fab(local).view();
    if (fields.active != nullptr)
      kernel.active = fields.active->fab(local).view();
    if (fields.covered != nullptr)
      kernel.covered = fields.covered->fab(local).view();
    kernel.reaction = reaction;
    kernel.relaxation = relaxation;
    kernel.write_jacobi = true;
    for (int axis = 0; axis < Dim; ++axis)
      kernel.inverse_spacing_squared[axis] = inverse_spacing_squared[axis];
    for_each_cell(destination.box(local), kernel);
  }
  Kokkos::fence();
}

}  // namespace pops::elliptic::mg
