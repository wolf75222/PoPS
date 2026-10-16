#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <limits>

namespace pops::runtime::program::detail {

// The generated Integrand contains only authored algebra and immutable views.
// This wrapper owns physical measure and finest-cell selection, independently
// of the model, observable, root solver or time integrator.
template <int Dim, class Integrand>
struct CompositeIntegralKernel {
  Integrand integrand;
  FieldView<const Real, Dim> active{}, coverage{}, fraction{};
  Real cell_volume = Real{1};
  bool has_active = false;
  bool has_fraction = false;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real a = has_active ? active(index, 0) : Real{1};
    const Real c = coverage(index, 0);
    const Real kappa = has_fraction ? fraction(index, 0) : Real{1};
    if (!Kokkos::isfinite(a) || !Kokkos::isfinite(c) || !Kokkos::isfinite(kappa) ||
        kappa < Real{0} || kappa > Real{1})
      return std::numeric_limits<Real>::infinity();
    if (a < Real{0.5} || c < Real{0.5} || kappa == Real{0})
      return Real{0};
    const Real value = integrand(index);
    const Real measured = value * (cell_volume * kappa);
    return Kokkos::isfinite(value) && Kokkos::isfinite(measured)
               ? measured
               : std::numeric_limits<Real>::infinity();
  }
};

template <int Dim>
struct CompositePairIntegrand {
  FieldView<const Real, Dim> left{}, right{};
  int component = 0;
  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real a = left(index, component), b = right(index, component);
    const Real value = a * b;
    return Kokkos::isfinite(a) && Kokkos::isfinite(b) && Kokkos::isfinite(value)
               ? value
               : std::numeric_limits<Real>::infinity();
  }
};

template <int Dim>
struct FixedMeasureDifferenceKernel {
  FieldView<const Real, Dim> before{}, after{};
  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real a = before(index, 0), b = after(index, 0);
    return Kokkos::isfinite(a) && Kokkos::isfinite(b) && a == b ? Real{0} : Real{1};
  }
};

}  // namespace pops::runtime::program::detail
