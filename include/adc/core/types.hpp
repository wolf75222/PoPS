#pragma once

// Types scalaires de base. Volontairement local et minimal pour garder le
// premier socle sans dependance externe. La bascule vers pde_core::Real
// (partage avec advection_cpp / euler_cpp / poisson_cpp) se fera quand le
// maillage distribue arrivera, pas avant.

namespace adc {

using Real = double;

}  // namespace adc
