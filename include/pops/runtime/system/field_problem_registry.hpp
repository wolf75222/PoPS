#pragma once

#include <pops/runtime/context/aux_layout.hpp>       // AuxLayout, AuxChannel, default_poisson_layout

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

/// @file
/// @brief The single native descriptor of every field problem a System / AmrSystem solves: the
///        default shared Poisson AND each named elliptic field, unified. LIGHT host-only header.
///
/// Motivation (ADC-596): today the default Poisson (the `ell_` / `mg_` solve) and the named
/// elliptic fields (`NamedField` in system_field_solver.hpp and amr_runtime.hpp) are two parallel
/// bookkeeping paths, and "which solver / which output layout / which boundary" is validated late
/// (at solve time, per backend). FieldProblemRegistry records each field problem ONCE as a typed
/// entry (id, equation kind, output AuxLayout, boundary, elliptic solver kind, Uniform/AMR
/// support) and validates the combination BEFORE bind. It is a DESCRIPTOR registry: it does not
/// own solvers and does not change any numerics -- the existing lazy solver build
/// (ensure_named_elliptic) and RHS assembly (assemble_poisson_rhs) stay the numerical truth; this
/// only names and validates what they realize, so Uniform and AMR share one field abstraction.

namespace pops {

/// The elliptic operator a field problem solves. Mirrors the historical bricks: a plain Poisson
/// (-Laplacian phi = rhs), a screened Poisson (Helmholtz, -Laplacian phi + kappa phi = rhs), or an
/// anisotropic Poisson (-div(eps grad phi) = rhs). Enumerated so a report / validation talks about
/// the operator by kind, not by a free string.
enum class EquationKind { Poisson, ScreenedPoisson, AnisotropicPoisson };

/// The potential boundary a field problem imposes. Mirrors the mesh-layer `pops::BCType`
/// (Periodic / Foextrap / Dirichlet) but is DECOUPLED from it so this descriptor header stays
/// light (no Kokkos-heavy mesh include): the Uniform / AMR register sites map their `BCType` onto
/// this enum. Values are held only for validation and reports, never for the numerics.
enum class FieldBoundaryKind { Periodic, Foextrap, Dirichlet };

/// The elliptic SOLVER backend a field problem requires. Named to match the solver variant the
/// native NamedField already carries (GeometricMG / PoissonFFTSolver / RemappedFFTSolver): the
/// registry stores the typed kind so a solver x layout mismatch is refused early (e.g. FFT on AMR),
/// instead of a free string checked deep in the backend.
enum class EllipticSolverKind { GeometricMG, FFT, RemappedFFT };

/// Where a field problem must be solvable. A registry entry declares whether it supports the
/// Uniform (single-mesh System) route, the AMR (AmrSystem hierarchy) route, or both. Used by
/// validate() to refuse an entry on a route it does not support.
enum class LayoutRoute { Uniform, Amr };

inline const char* to_string(EquationKind k) {
  switch (k) {
    case EquationKind::Poisson: return "Poisson";
    case EquationKind::ScreenedPoisson: return "ScreenedPoisson";
    case EquationKind::AnisotropicPoisson: return "AnisotropicPoisson";
  }
  return "?";
}

inline const char* to_string(EllipticSolverKind k) {
  switch (k) {
    case EllipticSolverKind::GeometricMG: return "GeometricMG";
    case EllipticSolverKind::FFT: return "FFT";
    case EllipticSolverKind::RemappedFFT: return "RemappedFFT";
  }
  return "?";
}

inline const char* to_string(LayoutRoute r) {
  return r == LayoutRoute::Uniform ? "Uniform" : "AMR";
}

/// One field problem, fully described. The default shared Poisson is the entry with id "phi";
/// every named elliptic field is its own entry. `layout` is the output manifest (ADC-588); the
/// low-level phi/grad component fields the native NamedField keeps are DERIVED from it.
struct FieldProblemEntry {
  std::string id;                                      ///< "phi" (default) or a named field
  EquationKind equation = EquationKind::Poisson;
  AuxLayout layout;                                    ///< outputs -> aux components (the manifest)
  FieldBoundaryKind boundary = FieldBoundaryKind::Periodic;                  ///< potential boundary kind
  EllipticSolverKind solver = EllipticSolverKind::GeometricMG;
  bool supports_uniform = true;
  bool supports_amr = true;
};

/// The registry: one entry per field problem, shared by Uniform and AMR. Insertion-ordered; the
/// integer id returned by register_problem is a stable index a FieldContext / Program IR can carry.
class FieldProblemRegistry {
 public:
  /// Register (or replace, by id) a field problem. Returns its stable integer id (its index).
  /// Re-registering an id overwrites the entry in place and keeps the same id, so a re-declared
  /// named field does not leak a duplicate.
  int register_problem(FieldProblemEntry entry) {
    if (entry.id.empty())
      throw std::invalid_argument("FieldProblemRegistry: a field problem must have a non-empty id");
    const int existing = find(entry.id);
    if (existing >= 0) {
      entries_[static_cast<std::size_t>(existing)] = std::move(entry);
      return existing;
    }
    entries_.push_back(std::move(entry));
    return static_cast<int>(entries_.size()) - 1;
  }

  int size() const { return static_cast<int>(entries_.size()); }
  const FieldProblemEntry& at(int id) const {
    if (id < 0 || id >= size())
      throw std::out_of_range("FieldProblemRegistry: field problem id " + std::to_string(id) +
                              " out of range [0, " + std::to_string(size()) + ")");
    return entries_[static_cast<std::size_t>(id)];
  }

  /// Find a problem by id string; returns its integer id, or -1 on miss.
  int find(std::string_view id) const {
    for (int i = 0; i < size(); ++i)
      if (entries_[static_cast<std::size_t>(i)].id == id)
        return i;
    return -1;
  }

  const std::vector<FieldProblemEntry>& entries() const { return entries_; }

  /// Pre-bind validation of ONE entry for a given route: refuse a solver x layout x output
  /// combination that the runtime could only reject late (or worse, silently). Throws a message
  /// naming the field problem, the offending axis and the route. This is the ADC-596 "early
  /// solver/layout/boundary/output validation" contract; it changes no numerics.
  void validate(int id, LayoutRoute route) const {
    const FieldProblemEntry& e = at(id);
    const bool route_supported =
        (route == LayoutRoute::Uniform) ? e.supports_uniform : e.supports_amr;
    if (!route_supported) {
      throw std::invalid_argument("field problem '" + e.id + "' is not available on the " +
                                  to_string(route) + " route");
    }
    // FFT solvers need a single uniform periodic mesh; they are not defined on an AMR hierarchy.
    if (route == LayoutRoute::Amr &&
        (e.solver == EllipticSolverKind::FFT || e.solver == EllipticSolverKind::RemappedFFT)) {
      throw std::invalid_argument(
          "field problem '" + e.id + "': solver " + to_string(e.solver) +
          " requires a single uniform mesh, not the AMR route; use GeometricMG on AMR");
    }
    // An FFT solve assumes a fully periodic potential; a Dirichlet boundary is a GeometricMG job.
    if ((e.solver == EllipticSolverKind::FFT || e.solver == EllipticSolverKind::RemappedFFT) &&
        e.boundary == FieldBoundaryKind::Dirichlet) {
      throw std::invalid_argument(
          "field problem '" + e.id + "': solver " + to_string(e.solver) +
          " requires periodic boundaries; a Dirichlet potential needs GeometricMG");
    }
    // The output manifest must at least carry the potential (component 0); an empty layout means the
    // solve produces nothing readable.
    if (e.layout.find("phi") == nullptr && e.layout.channels().empty()) {
      throw std::invalid_argument("field problem '" + e.id +
                                  "': output layout declares no field (expected at least 'phi')");
    }
  }

  /// Validate every registered entry for a route in one call (used at bind).
  void validate_all(LayoutRoute route) const {
    for (int i = 0; i < size(); ++i)
      validate(i, route);
  }

 private:
  std::vector<FieldProblemEntry> entries_;
};

/// Build the default shared-Poisson entry ("phi") with the historical base contract layout and a
/// GeometricMG solver. Convenience for the System/AmrSystem default registration so the single
/// field case is described the same way as a named one.
inline FieldProblemEntry default_poisson_entry(FieldBoundaryKind boundary = FieldBoundaryKind::Periodic,
                                               EllipticSolverKind solver =
                                                   EllipticSolverKind::GeometricMG) {
  FieldProblemEntry e;
  e.id = "phi";
  e.equation = EquationKind::Poisson;
  e.layout = default_poisson_layout();
  e.boundary = boundary;
  e.solver = solver;
  e.supports_uniform = true;
  e.supports_amr = true;
  return e;
}

/// Build a named-elliptic-field entry from the aux output components the native NamedField records
/// (phi at @p phi_comp, optional centered gradient at @p gx_comp / @p gy_comp when >= 0). This is
/// the adapter the Uniform/AMR register_named_field paths use so a named field becomes a registry
/// entry WITHOUT re-encoding its layout: the components are the low-level truth, the AuxLayout is
/// the named view of them.
inline FieldProblemEntry named_field_entry(const std::string& id, int phi_comp, int gx_comp,
                                           int gy_comp,
                                           EllipticSolverKind solver =
                                               EllipticSolverKind::GeometricMG,
                                           FieldBoundaryKind boundary = FieldBoundaryKind::Periodic) {
  FieldProblemEntry e;
  e.id = id;
  e.equation = EquationKind::Poisson;
  if (phi_comp >= 0)
    e.layout.add_channel(id, phi_comp, FieldChannelRole::kNamed);
  if (gx_comp >= 0)
    e.layout.add_channel(id + "_grad_x", gx_comp, FieldChannelRole::kGradient);
  if (gy_comp >= 0)
    e.layout.add_channel(id + "_grad_y", gy_comp, FieldChannelRole::kGradient);
  e.boundary = boundary;
  e.solver = solver;
  e.supports_uniform = true;
  e.supports_amr = true;
  return e;
}

}  // namespace pops
