/// @file
/// @brief Allocation-free exact-ranked Cartesian full-tensor elliptic point operator.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/field_view.hpp>

#include <array>
#include <cstddef>

namespace pops::elliptic::nd {

/// Select the authored sign of the conservative tensor divergence.
enum class CartesianTensorDivergenceSign : unsigned char {
  positive_divergence,
  negative_divergence,
};

/// Optional homogeneous FV face laws. A zero-flux face suppresses the complete conormal
/// flux, including cross derivatives. The masks use the lower/upper face ordinal 2*axis+side.
/// Arithmetic averaging is useful for smooth mapped metrics (for example K_rr=r on a disk).
struct CartesianTensorStencilOptions {
  unsigned zero_flux_faces = 0;
  unsigned dirichlet_faces = 0;
  bool arithmetic_diagonal = false;
};

/// Non-owning adapter for one row-major ``Dim*Dim`` coefficient field.
template <int Dim>
struct PackedCartesianTensorCoefficients {
  static_assert(Dim >= 1 && Dim <= 3,
                "PackedCartesianTensorCoefficients supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;

  FieldView<const Real, Dim> field{};

  POPS_HD Real at(const Index<Dim>& cell, int row, int column) const {
    return field(cell, row * Dim + column);
  }
};

/// Non-owning adapter for ``Dim*Dim`` scalar coefficient fields in row-major order.
template <int Dim>
struct SplitCartesianTensorCoefficients {
  static_assert(Dim >= 1 && Dim <= 3,
                "SplitCartesianTensorCoefficients supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;

  std::array<FieldView<const Real, Dim>, static_cast<std::size_t>(Dim* Dim)> fields{};

  POPS_HD Real at(const Index<Dim>& cell, int row, int column) const {
    return fields[static_cast<std::size_t>(row * Dim + column)](cell, 0);
  }
};

POPS_HD inline Real harmonic_tensor_face_average(Real left, Real right) {
  const Real denominator = left + right;
  return denominator != Real(0) ? Real(2) * left * right / denominator : Real(0);
}

/// Exact conservative Cartesian ``+/- div(A grad(phi))`` stencil at one cell.
///
/// Diagonal coefficients use harmonic face averaging. Off-diagonal coefficients use arithmetic
/// face averaging and centered tangential derivatives formed from both cells adjacent to the
/// normal face. ``Coefficients`` is a non-owning packed or split adapter; constructing and applying
/// this object never allocates or repacks storage.
template <int Dim, CartesianTensorDivergenceSign Sign, class Coefficients>
struct CartesianTensorOperator {
  static_assert(Dim >= 1 && Dim <= 3, "CartesianTensorOperator supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;

  FieldView<const Real, Dim> phi{};
  Coefficients coefficients{};
  RealVector<Dim> inverse_spacing{};
  Box<Dim> domain{};
  CartesianTensorStencilOptions options{};

  POPS_HD Real diagonal_face(Real left, Real right) const {
    return options.arithmetic_diagonal ? Real(0.5) * (left + right)
                                       : harmonic_tensor_face_average(left, right);
  }

  POPS_HD bool on_face(const Index<Dim>& cell, int axis, int side, unsigned mask) const {
    return (mask & (1u << (2 * axis + side))) != 0 &&
           cell[axis] == (side == 0 ? domain.lo[axis] : domain.hi[axis]);
  }

  // Both faces use positive coordinate-axis orientation. Keep their arithmetic shared with
  // the level image so coarse/fine replacement retains every tensor cross derivative and
  // the exact physical-face coefficient and zero-conormal laws.
  POPS_HD std::array<Real, 2> face_fluxes(const Index<Dim>& cell, int row) const {
    Index<Dim> lower = cell;
    Index<Dim> upper = cell;
    --lower[row];
    ++upper[row];
    Real lower_flux = Real(0);
    Real upper_flux = Real(0);
    for (int column = 0; column < Dim; ++column) {
      const Real center_coefficient = coefficients.at(cell, row, column);
      const Real lower_coefficient = coefficients.at(lower, row, column);
      const Real upper_coefficient = coefficients.at(upper, row, column);
      Real lower_face =
          row == column ? diagonal_face(lower_coefficient, center_coefficient)
                        : Real(0.5) * (lower_coefficient + center_coefficient);
      Real upper_face =
          row == column ? diagonal_face(center_coefficient, upper_coefficient)
                        : Real(0.5) * (center_coefficient + upper_coefficient);
      // Smooth coordinate metrics are sampled at physical faces by linear one-sided
      // extrapolation; coefficient ghosts carry an extrapolation law, not a face location.
      if (options.arithmetic_diagonal) {
        const unsigned physical = options.zero_flux_faces | options.dirichlet_faces;
        if (on_face(cell, row, 0, physical))
          lower_face = Real(1.5) * center_coefficient - Real(0.5) * upper_coefficient;
        if (on_face(cell, row, 1, physical))
          upper_face = Real(1.5) * center_coefficient - Real(0.5) * lower_coefficient;
      }
      if (row == column) {
        lower_flux += lower_face * (phi(cell, 0) - phi(lower, 0)) * inverse_spacing[column];
        upper_flux += upper_face * (phi(upper, 0) - phi(cell, 0)) * inverse_spacing[column];
        continue;
      }

      Index<Dim> cell_lower = cell;
      Index<Dim> cell_upper = cell;
      Index<Dim> lower_lower = lower;
      Index<Dim> lower_upper = lower;
      Index<Dim> upper_lower = upper;
      Index<Dim> upper_upper = upper;
      --cell_lower[column];
      ++cell_upper[column];
      --lower_lower[column];
      ++lower_upper[column];
      --upper_lower[column];
      ++upper_upper[column];
      const Real tangent_scale = Real(0.25) * inverse_spacing[column];
      const Real lower_tangent =
          (phi(cell_upper, 0) - phi(cell_lower, 0) + phi(lower_upper, 0) - phi(lower_lower, 0)) *
          tangent_scale;
      const Real upper_tangent =
          (phi(cell_upper, 0) - phi(cell_lower, 0) + phi(upper_upper, 0) - phi(upper_lower, 0)) *
          tangent_scale;
      lower_flux += lower_face * lower_tangent;
      upper_flux += upper_face * upper_tangent;
    }
    if (on_face(cell, row, 0, options.zero_flux_faces))
      lower_flux = Real(0);
    if (on_face(cell, row, 1, options.zero_flux_faces))
      upper_flux = Real(0);
    return {lower_flux, upper_flux};
  }

  POPS_HD Real face_flux(const Index<Dim>& cell, int row, int side) const {
    return face_fluxes(cell, row)[static_cast<std::size_t>(side)];
  }

  POPS_HD Real image(const Index<Dim>& cell) const {
    Real divergence = Real(0);
    for (int row = 0; row < Dim; ++row) {
      const auto flux = face_fluxes(cell, row);
      divergence += (flux[1] - flux[0]) * inverse_spacing[row];
    }
    if constexpr (Sign == CartesianTensorDivergenceSign::negative_divergence)
      return -divergence;
    return divergence;
  }

  /// Return the exact center coefficient of the selected signed operator.
  POPS_HD Real diagonal(const Index<Dim>& cell) const {
    Real negative_divergence_diagonal = Real(0);
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> lower = cell;
      Index<Dim> upper = cell;
      --lower[axis];
      ++upper[axis];
      const Real center = coefficients.at(cell, axis, axis);
      Real lower_factor = on_face(cell, axis, 0, options.zero_flux_faces) ? Real(0) : Real(1);
      Real upper_factor = on_face(cell, axis, 1, options.zero_flux_faces) ? Real(0) : Real(1);
      if (on_face(cell, axis, 0, options.dirichlet_faces))
        lower_factor = Real(2);
      if (on_face(cell, axis, 1, options.dirichlet_faces))
        upper_factor = Real(2);
      Real lower_face = diagonal_face(coefficients.at(lower, axis, axis), center);
      Real upper_face = diagonal_face(center, coefficients.at(upper, axis, axis));
      if (options.arithmetic_diagonal) {
        const unsigned physical = options.zero_flux_faces | options.dirichlet_faces;
        if (on_face(cell, axis, 0, physical))
          lower_face = Real(1.5) * center - Real(0.5) * coefficients.at(upper, axis, axis);
        if (on_face(cell, axis, 1, physical))
          upper_face = Real(1.5) * center - Real(0.5) * coefficients.at(lower, axis, axis);
      }
      negative_divergence_diagonal +=
          (lower_factor * lower_face + upper_factor * upper_face) *
          inverse_spacing[axis] * inverse_spacing[axis];
    }
    if constexpr (Sign == CartesianTensorDivergenceSign::positive_divergence)
      return -negative_divergence_diagonal;
    return negative_divergence_diagonal;
  }
};

template <int Dim>
POPS_HD PackedCartesianTensorCoefficients<Dim> packed_cartesian_tensor_coefficients(
    FieldView<const Real, Dim> field) {
  return {field};
}

template <int Dim>
POPS_HD SplitCartesianTensorCoefficients<Dim> split_cartesian_tensor_coefficients(
    std::array<FieldView<const Real, Dim>, static_cast<std::size_t>(Dim* Dim)> fields) {
  return {fields};
}

template <CartesianTensorDivergenceSign Sign, int Dim, class Coefficients>
POPS_HD CartesianTensorOperator<Dim, Sign, Coefficients> make_cartesian_tensor_operator(
    FieldView<const Real, Dim> phi, Coefficients coefficients, const Geometry<Dim>& geometry,
    CartesianTensorStencilOptions options = {}) {
  RealVector<Dim> inverse_spacing{};
  for (int axis = 0; axis < Dim; ++axis)
    inverse_spacing[axis] = Real(1) / geometry.spacing(axis);
  return {phi, coefficients, inverse_spacing, geometry.domain(), options};
}

}  // namespace pops::elliptic::nd
