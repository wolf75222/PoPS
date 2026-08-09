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

template <int Dim>
struct ScaledAxpyKernel {
  FieldView<Real, Dim> destination;
  FieldView<const Real, Dim> source;
  ScaledScalar coefficient;
  int component;

  POPS_HD void operator()(const Index<Dim>& index) const {
    Real value = std::numeric_limits<Real>::quiet_NaN();
    (void)ScaledScalar::try_sum_products(ScaledScalar::from(Real(1)), destination(index, component),
                                         coefficient, source(index, component), value);
    destination(index, component) = value;
  }
};

template <int Dim>
struct ScaledLincombKernel {
  FieldView<Real, Dim> destination;
  FieldView<const Real, Dim> left;
  FieldView<const Real, Dim> right;
  ScaledScalar left_coefficient;
  ScaledScalar right_coefficient;
  int component;

  POPS_HD void operator()(const Index<Dim>& index) const {
    Real value = std::numeric_limits<Real>::quiet_NaN();
    (void)ScaledScalar::try_sum_products(left_coefficient, left(index, component),
                                         right_coefficient, right(index, component), value);
    destination(index, component) = value;
  }
};

template <int Dim>
struct ScaledTrilincombKernel {
  FieldView<Real, Dim> destination;
  FieldView<const Real, Dim> first;
  FieldView<const Real, Dim> second;
  FieldView<const Real, Dim> third;
  ScaledScalar first_coefficient;
  ScaledScalar second_coefficient;
  ScaledScalar third_coefficient;
  int component;

  POPS_HD void operator()(const Index<Dim>& index) const {
    Real value = std::numeric_limits<Real>::quiet_NaN();
    (void)ScaledScalar::try_sum_products(first_coefficient, first(index, component),
                                         second_coefficient, second(index, component),
                                         third_coefficient, third(index, component), value);
    destination(index, component) = value;
  }
};

/// Unchecked scaled algebra for an already authenticated prepared solve.  Each source may alias
/// `destination`; every kernel reads all cell values before writing the final fused result.
struct ScaledFieldAlgebra {
  template <int Dim>
  static void axpy(MultiFab<Dim>& destination, const ScaledScalar& coefficient,
                   const MultiFab<Dim>& source) {
    // Preserve the established arithmetic path bit-for-bit when the coefficient is ordinary. The
    // scaled kernel is an overflow escape hatch, not a different rounding policy for every Krylov
    // iteration.
    Real materialized = Real(0);
    if (coefficient.try_materialize(materialized)) {
      pops::saxpy(destination, materialized, source);
      return;
    }
    for (int local = 0; local < destination.local_size(); ++local) {
      const FieldView<Real, Dim> output = destination.fab(local).view();
      const FieldView<const Real, Dim> input = std::as_const(source.fab(local)).view();
      const Box<Dim> valid = destination.box(local);
      for (int component = 0; component < destination.ncomp(); ++component)
        for_each_cell(valid, detail::ScaledAxpyKernel<Dim>{output, input, coefficient, component});
    }
  }

  template <int Dim>
  static void lincomb(MultiFab<Dim>& destination, const ScaledScalar& left_coefficient,
                      const MultiFab<Dim>& left, const ScaledScalar& right_coefficient,
                      const MultiFab<Dim>& right) {
    Real left_materialized = Real(0);
    Real right_materialized = Real(0);
    if (left_coefficient.try_materialize(left_materialized) &&
        right_coefficient.try_materialize(right_materialized)) {
      pops::lincomb(destination, left_materialized, left, right_materialized, right);
      return;
    }
    for (int local = 0; local < destination.local_size(); ++local) {
      const FieldView<Real, Dim> output = destination.fab(local).view();
      const FieldView<const Real, Dim> left_values = std::as_const(left.fab(local)).view();
      const FieldView<const Real, Dim> right_values = std::as_const(right.fab(local)).view();
      const Box<Dim> valid = destination.box(local);
      for (int component = 0; component < destination.ncomp(); ++component)
        for_each_cell(valid, detail::ScaledLincombKernel<Dim>{output, left_values, right_values,
                                                              left_coefficient, right_coefficient,
                                                              component});
    }
  }

  template <int Dim>
  static void trilincomb(MultiFab<Dim>& destination, const ScaledScalar& first_coefficient,
                         const MultiFab<Dim>& first, const ScaledScalar& second_coefficient,
                         const MultiFab<Dim>& second, const ScaledScalar& third_coefficient,
                         const MultiFab<Dim>& third) {
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
      const FieldView<Real, Dim> output = destination.fab(local).view();
      const FieldView<const Real, Dim> first_values = std::as_const(first.fab(local)).view();
      const FieldView<const Real, Dim> second_values = std::as_const(second.fab(local)).view();
      const FieldView<const Real, Dim> third_values = std::as_const(third.fab(local)).view();
      const Box<Dim> valid = destination.box(local);
      for (int component = 0; component < destination.ncomp(); ++component)
        for_each_cell(valid,
                      detail::ScaledTrilincombKernel<Dim>{
                          output, first_values, second_values, third_values, first_coefficient,
                          second_coefficient, third_coefficient, component});
    }
  }
};

}  // namespace detail
}  // namespace pops
