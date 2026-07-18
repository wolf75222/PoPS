#pragma once

namespace pops::runtime::system {

/// Common optional observations published by a prepared backend.  A direct solver honestly
/// returns zeros; no synthetic capability is required merely to expose the common report shape.
struct EllipticBackendMetrics {
  int multigrid_cycles = 0;
  int krylov_iterations = 0;
  int multigrid_levels = 0;
  double bottom_seconds = 0.0;
};

}  // namespace pops::runtime::system
