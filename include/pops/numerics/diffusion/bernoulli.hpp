#pragma once
#include <pops/core/foundation/types.hpp>
#include <Kokkos_Core.hpp>

namespace pops::runtime::program {
/// Stable B(z)=z/(exp(z)-1), including the removable singularity and large drift limits.
POPS_HD inline Real scharfetter_gummel_bernoulli(Real z) {
  const Real absolute=Kokkos::abs(z);
  if(absolute<Real(1e-4)) {
    const Real square=z*z;
    return Real(1)-z/Real(2)+square*(Real(1)/Real(12)+square*(-Real(1)/Real(720)+square/Real(30240)));
  }
  if(z>Real(50)) {
    const Real inverse=Kokkos::exp(-z);
    return z*inverse/(Real(1)-inverse);
  }
  if(z<Real(-50)) return -z/(Real(1)-Kokkos::exp(z));
  return z/Kokkos::expm1(z);
}
}
