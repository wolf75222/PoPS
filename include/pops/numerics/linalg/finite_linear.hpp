#pragma once
#include <pops/numerics/linalg/block_inverse.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <limits>

namespace pops::detail {
// Explicit finite DOFs only: these arrays never denote a mesh decomposition.
template<int Rows, int Columns>
POPS_HD inline Kokkos::Array<Real, Rows> finite_linear_apply(
    const Kokkos::Array<Real, Rows*Columns>& matrix,
    const Kokkos::Array<Real, Columns>& input) {
  static_assert(Rows > 0 && Columns > 0);
  Kokkos::Array<Real, Rows> result{};
  bool valid=true;
  for (int i=0; i<Rows; ++i) {
    Real sum=0;
    for (int j=0; j<Columns; ++j) {
      const Real product=matrix[i*Columns+j]*input[j];
      sum += product;
      if (!Kokkos::isfinite(product) || !Kokkos::isfinite(sum)) {
        valid=false;
        sum=std::numeric_limits<Real>::quiet_NaN();
        break;
      }
    }
    result[i]=sum;
  }
  if (!valid)
    for (int i=0; i<Rows; ++i) result[i]=std::numeric_limits<Real>::quiet_NaN();
  return result;
}

template<int Rows, int Columns>
POPS_HD inline Kokkos::Array<Real, Columns> finite_linear_solve(
    const Kokkos::Array<Real, Rows*Columns>& matrix,
    const Kokkos::Array<Real, Rows>& input) {
  static_assert(Rows > 0 && Rows == Columns, "finite solve requires a square map");
  Real a[Rows][Rows], rhs[Rows], candidate[Rows];
  for (int i=0; i<Rows; ++i) {
    rhs[i]=input[i];
    for (int j=0; j<Rows; ++j) a[i][j]=matrix[i*Rows+j];
  }
  const bool accepted=block_apply_inverse<Rows>(a, rhs, candidate);
  Kokkos::Array<Real, Columns> result{};
  for (int i=0; i<Rows; ++i)
    result[i]=accepted ? candidate[i] : std::numeric_limits<Real>::quiet_NaN();
  return result;
}
} // namespace pops::detail
