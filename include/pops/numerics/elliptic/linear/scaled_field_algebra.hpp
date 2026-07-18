#pragma once

/// @file
/// @brief Device-safe fused field updates driven by binary-scaled coefficients.
///
/// The ordinary `Real` algebra remains the right fast path for representable recurrence
/// coefficients.  These helpers are deliberately narrow: they preserve the affine update shape
/// while evaluating its coefficient products and signed cancellation as one `ScaledScalar`
/// expression per cell.  They allocate no scratch and never materialize a coefficient merely to
/// multiply it by a field.

#include <pops/numerics/elliptic/linear/pure_field_algebra.hpp>
#include <pops/numerics/elliptic/linear/scaled_scalar.hpp>

#include <limits>

namespace pops {
namespace detail {

struct ScaledAxpyKernel {
  Array4 destination;
  ConstArray4 source;
  ScaledScalar coefficient;
  int component;

  POPS_HD void operator()(int i, int j) const {
    Real value = std::numeric_limits<Real>::quiet_NaN();
    (void)ScaledScalar::try_sum_products(ScaledScalar::from(Real(1)), destination(i, j, component),
                                         coefficient, source(i, j, component), value);
    destination(i, j, component) = value;
  }
};

struct ScaledLincombKernel {
  Array4 destination;
  ConstArray4 left;
  ConstArray4 right;
  ScaledScalar left_coefficient;
  ScaledScalar right_coefficient;
  int component;

  POPS_HD void operator()(int i, int j) const {
    Real value = std::numeric_limits<Real>::quiet_NaN();
    (void)ScaledScalar::try_sum_products(left_coefficient, left(i, j, component), right_coefficient,
                                         right(i, j, component), value);
    destination(i, j, component) = value;
  }
};

struct ScaledTrilincombKernel {
  Array4 destination;
  ConstArray4 first;
  ConstArray4 second;
  ConstArray4 third;
  ScaledScalar first_coefficient;
  ScaledScalar second_coefficient;
  ScaledScalar third_coefficient;
  int component;

  POPS_HD void operator()(int i, int j) const {
    Real value = std::numeric_limits<Real>::quiet_NaN();
    (void)ScaledScalar::try_sum_products(first_coefficient, first(i, j, component),
                                         second_coefficient, second(i, j, component),
                                         third_coefficient, third(i, j, component), value);
    destination(i, j, component) = value;
  }
};

/// Unchecked scaled algebra for an already authenticated prepared solve.  Each source may alias
/// `destination`; every kernel reads all cell values before writing the final fused result.
struct ScaledFieldAlgebra {
  static void axpy(MultiFab& destination, const ScaledScalar& coefficient, const MultiFab& source) {
    // Preserve the established arithmetic path bit-for-bit when the coefficient is ordinary. The
    // scaled kernel is an overflow escape hatch, not a different rounding policy for every Krylov
    // iteration.
    Real materialized = Real(0);
    if (coefficient.try_materialize(materialized)) {
      pops::saxpy(destination, materialized, source);
      return;
    }
    for (int local = 0; local < destination.local_size(); ++local) {
      const Array4 output = destination.fab(local).array();
      const ConstArray4 input = source.fab(local).const_array();
      const Box2D valid = destination.box(local);
      for (int component = 0; component < destination.ncomp(); ++component)
        for_each_cell(valid, detail::ScaledAxpyKernel{output, input, coefficient, component});
    }
  }

  static void lincomb(MultiFab& destination, const ScaledScalar& left_coefficient,
                      const MultiFab& left, const ScaledScalar& right_coefficient,
                      const MultiFab& right) {
    Real left_materialized = Real(0);
    Real right_materialized = Real(0);
    if (left_coefficient.try_materialize(left_materialized) &&
        right_coefficient.try_materialize(right_materialized)) {
      pops::lincomb(destination, left_materialized, left, right_materialized, right);
      return;
    }
    for (int local = 0; local < destination.local_size(); ++local) {
      const Array4 output = destination.fab(local).array();
      const ConstArray4 left_values = left.fab(local).const_array();
      const ConstArray4 right_values = right.fab(local).const_array();
      const Box2D valid = destination.box(local);
      for (int component = 0; component < destination.ncomp(); ++component)
        for_each_cell(
            valid, detail::ScaledLincombKernel{output, left_values, right_values, left_coefficient,
                                               right_coefficient, component});
    }
  }

  static void trilincomb(MultiFab& destination, const ScaledScalar& first_coefficient,
                         const MultiFab& first, const ScaledScalar& second_coefficient,
                         const MultiFab& second, const ScaledScalar& third_coefficient,
                         const MultiFab& third) {
    Real first_materialized = Real(0);
    Real second_materialized = Real(0);
    Real third_materialized = Real(0);
    if (first_coefficient.try_materialize(first_materialized) &&
        second_coefficient.try_materialize(second_materialized) &&
        third_coefficient.try_materialize(third_materialized) && first_materialized == Real(1) &&
        destination.shares_storage_with(first)) {
      // This is the BiCGStab legacy sequence x += alpha*p; x += omega*s. Retaining it on the
      // representable route avoids a harmless-looking reassociation from changing a long solve's
      // convergence decision at the final tolerance.
      pops::saxpy(destination, second_materialized, second);
      pops::saxpy(destination, third_materialized, third);
      return;
    }
    for (int local = 0; local < destination.local_size(); ++local) {
      const Array4 output = destination.fab(local).array();
      const ConstArray4 first_values = first.fab(local).const_array();
      const ConstArray4 second_values = second.fab(local).const_array();
      const ConstArray4 third_values = third.fab(local).const_array();
      const Box2D valid = destination.box(local);
      for (int component = 0; component < destination.ncomp(); ++component)
        for_each_cell(
            valid, detail::ScaledTrilincombKernel{output, first_values, second_values, third_values,
                                                  first_coefficient, second_coefficient,
                                                  third_coefficient, component});
    }
  }
};

}  // namespace detail
}  // namespace pops
