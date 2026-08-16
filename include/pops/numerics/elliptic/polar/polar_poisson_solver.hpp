/// @file
/// @brief Exact-rank direct Poisson solver on a two-dimensional annulus.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_1d_internal.hpp>
#include <pops/numerics/elliptic/polar/polar_geometry.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

/// The concrete algorithm performs a complete azimuthal FFT followed by one radial Thomas solve
/// per Fourier mode.  Its data decomposition is therefore exactly rank two and one-box/one-rank.
template <int Dim>
struct PolarPoissonCapabilities {
  static_assert(Dim >= 1 && Dim <= 3,
                "PolarPoissonCapabilities only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  static constexpr bool available = Dim == 2;
  static constexpr bool direct = available;
  static constexpr bool distributed = false;
  static constexpr std::string_view unavailable_reason =
      available ? std::string_view{}
                : std::string_view{"polar FFT/Thomas Poisson has exactly the axes (r, theta)"};
};

static_assert(!PolarPoissonCapabilities<1>::available);
static_assert(PolarPoissonCapabilities<2>::available);
static_assert(!PolarPoissonCapabilities<3>::available);

template <class Solver>
concept PolarEllipticSolver = requires(Solver solver, const Solver constant_solver) {
  { Solver::dimension } -> std::convertible_to<int>;
  typename Solver::field_type;
  requires std::same_as<typename Solver::field_type::box_type, Box<Solver::dimension>>;
  { solver.rhs() } -> std::same_as<typename Solver::field_type&>;
  { solver.phi() } -> std::same_as<typename Solver::field_type&>;
  { solver.solve() } -> std::same_as<SolveReport>;
  { constant_solver.residual() } -> std::convertible_to<Real>;
  { constant_solver.geom() } ->
      std::same_as<const PolarGeometry<Solver::dimension>&>;
};

namespace polar_poisson_detail {

inline std::size_t host_offset(const Box<2>& storage, int i, int j) {
  return static_cast<std::size_t>(i - storage.lo[0]) +
         static_cast<std::size_t>(j - storage.lo[1]) *
             static_cast<std::size_t>(storage.length(0));
}

inline bool radial_boundary_supported(const PhysicalBoundaryFace& face) noexcept {
  if (face.kind == PhysicalBoundaryKind::dirichlet)
    return true;
  if (face.kind == PhysicalBoundaryKind::constant_extrapolation)
    return true;
  return face.kind == PhysicalBoundaryKind::neumann && face.value == Real(0);
}

inline bool is_dirichlet(const PhysicalBoundaryFace& face) noexcept {
  return face.kind == PhysicalBoundaryKind::dirichlet;
}

}  // namespace polar_poisson_detail

/// Direct conservative radial / spectral azimuthal inversion on one exact annulus.
template <int Dim>
  requires(PolarPoissonCapabilities<Dim>::available)
class PolarPoissonSolver {
 public:
  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim>;
  using request_type = PolarEllipticBuildRequest<Dim>;

  explicit PolarPoissonSolver(request_type request)
      : geometry_(request.geometry),
        boundary_(request.boundary),
        rhs_(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}),
        phi_(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}),
        trial_(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}) {
    validate_request_(request);
  }

  PolarPoissonSolver(const PolarPoissonSolver&) = delete;
  PolarPoissonSolver& operator=(const PolarPoissonSolver&) = delete;
  PolarPoissonSolver(PolarPoissonSolver&&) noexcept = default;
  PolarPoissonSolver& operator=(PolarPoissonSolver&&) noexcept = default;
  ~PolarPoissonSolver() noexcept = default;

  static constexpr PolarPoissonCapabilities<Dim> capabilities() noexcept { return {}; }

  field_type& rhs() noexcept { return rhs_; }
  const field_type& rhs() const noexcept { return rhs_; }
  field_type& phi() noexcept { return phi_; }
  const field_type& phi() const noexcept { return phi_; }
  const PolarGeometry<Dim>& geom() const noexcept { return geometry_; }
  const PhysicalBoundaryConditions<Dim>& boundary() const noexcept { return boundary_; }
  const SolveReport& last_solve_report() const noexcept { return last_report_; }

  SolveReport solve() {
    SolveReport report;
    try {
      invert_into_trial_();
      report.evaluations = 1;
      report.reference_residual_norm = rhs_norm_inf_();
      report.residual_norm = residual_of_(trial_);
      report.rel_residual = report.reference_residual_norm > Real(0)
                                ? report.residual_norm / report.reference_residual_norm
                                : report.residual_norm;
      if (!std::isfinite(static_cast<double>(report.residual_norm)) ||
          !std::isfinite(static_cast<double>(report.rel_residual))) {
        report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                           "polar_poisson_non_finite_residual");
        last_report_ = report;
        return last_report_;
      }
      const Real envelope = Real(1024) * std::numeric_limits<Real>::epsilon() *
                            std::sqrt(static_cast<Real>(geometry_.domain().numPts())) *
                            std::max(Real(1), report.reference_residual_norm);
      if (report.residual_norm > envelope) {
        report.mark_failed(SolveStatus::kInadmissibleCandidate, SolveAction::kFailRun,
                           "polar_poisson_residual_exceeds_roundoff_envelope");
        last_report_ = report;
        return last_report_;
      }
      std::swap(phi_, trial_);
      report.mark_solved("polar_poisson_fft_thomas_direct");
    } catch (const std::exception& error) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         std::string("polar_poisson_solve_failed: ") + error.what());
    }
    last_report_ = report;
    return last_report_;
  }

  /// Residual of the currently published field under the exact operator inverted by solve().
  Real residual() const { return residual_of_(phi_); }

 private:
  using complex_type = std::complex<Real>;

  static void validate_request_(const request_type& request) {
    if (n_ranks() != 1)
      throw std::invalid_argument(
          "PolarPoissonSolver<2> requires one rank; distributed transpose is unavailable");
    if (request.geometry.domain().empty() || request.boxes.size() != 1 ||
        request.boxes[0] != request.geometry.domain() ||
        !request.boxes.tiles_exactly(request.geometry.domain(), request.layout_budget) ||
        !request.distribution.matches_layout(request.boxes) ||
        request.distribution.rank_space().size() != 1 ||
        !request.distribution.rank_space().contains(request.local_rank) ||
        request.distribution.rank_space().linear_rank(request.local_rank) != 0)
      throw std::invalid_argument(
          "PolarPoissonSolver<2> requires one exact full-annulus local patch");

    const auto& bc = request.boundary;
    if (bc.spacing()[0] != request.geometry.dr() ||
        bc.spacing()[1] != request.geometry.dtheta())
      throw std::invalid_argument("polar Poisson boundary spacing differs from its geometry");
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> radial{0, side};
      const Face<Dim> azimuthal{1, side};
      if (bc.topology().is_periodic(radial) ||
          !polar_poisson_detail::radial_boundary_supported(bc.at(radial)))
        throw std::invalid_argument(
            "polar Poisson radial boundary must be Dirichlet or homogeneous Neumann");
      if (!bc.topology().is_periodic(azimuthal) ||
          bc.at(azimuthal).kind != PhysicalBoundaryKind::external)
        throw std::invalid_argument("polar Poisson azimuthal boundary must be periodic");
    }
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t cells = request.geometry.domain().length(axis);
      if (cells <= 0 || cells > std::numeric_limits<int>::max())
        throw std::invalid_argument("polar Poisson extent exceeds the concrete index range");
    }
  }

  static std::vector<std::vector<complex_type>> transform_rows_(const field_type& field) {
    const auto& fab = field.fab(0);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& storage = fab.grown_box();
    const int nr = static_cast<int>(valid.length(0));
    const int nth = static_cast<int>(valid.length(1));
    std::vector<std::vector<complex_type>> transformed(static_cast<std::size_t>(nr));
    for (int radial = 0; radial < nr; ++radial) {
      auto& row = transformed[static_cast<std::size_t>(radial)];
      row.resize(static_cast<std::size_t>(nth));
      for (int azimuthal = 0; azimuthal < nth; ++azimuthal) {
        const int i = valid.lo[0] + radial;
        const int j = valid.lo[1] + azimuthal;
        row[static_cast<std::size_t>(azimuthal)] =
            complex_type(host(polar_poisson_detail::host_offset(storage, i, j)), Real(0));
      }
      elliptic::poisson::internal::host_fft1d(row, false);
    }
    return transformed;
  }

  void invert_into_trial_() {
    const Box<Dim>& valid = rhs_.box(0);
    const int nr = static_cast<int>(valid.length(0));
    const int nth = static_cast<int>(valid.length(1));
    const Real dr = geometry_.dr();
    auto rhs_hat = transform_rows_(rhs_);

    std::vector<Real> lower(static_cast<std::size_t>(nr));
    std::vector<Real> upper(static_cast<std::size_t>(nr));
    std::vector<Real> radial_diagonal(static_cast<std::size_t>(nr));
    std::vector<Real> inverse_radius_squared(static_cast<std::size_t>(nr));
    for (int radial = 0; radial < nr; ++radial) {
      const int i = valid.lo[0] + radial;
      const Real radius = geometry_.r_cell(i);
      const Real inverse = Real(1) / (radius * dr * dr);
      lower[static_cast<std::size_t>(radial)] = geometry_.r_face(i) * inverse;
      upper[static_cast<std::size_t>(radial)] = geometry_.r_face(i + 1) * inverse;
      radial_diagonal[static_cast<std::size_t>(radial)] =
          -(lower[static_cast<std::size_t>(radial)] +
            upper[static_cast<std::size_t>(radial)]);
      inverse_radius_squared[static_cast<std::size_t>(radial)] = Real(1) / (radius * radius);
    }

    const auto& low = boundary_.at(Face<Dim>{0, BoundarySide::lower});
    const auto& high = boundary_.at(Face<Dim>{0, BoundarySide::upper});
    const bool low_dirichlet = polar_poisson_detail::is_dirichlet(low);
    const bool high_dirichlet = polar_poisson_detail::is_dirichlet(high);
    const bool pin_constant = !low_dirichlet && !high_dirichlet;

    std::vector<std::vector<complex_type>> phi_hat(static_cast<std::size_t>(nr));
    for (auto& row : phi_hat)
      row.resize(static_cast<std::size_t>(nth));
    std::vector<Real> diagonal(static_cast<std::size_t>(nr));
    std::vector<complex_type> mode_rhs(static_cast<std::size_t>(nr));
    std::vector<complex_type> mode_solution(static_cast<std::size_t>(nr));
    for (int mode = 0; mode < nth; ++mode) {
      const int signed_mode = mode <= nth / 2 ? mode : mode - nth;
      const Real eigenvalue =
          -static_cast<Real>(signed_mode) * static_cast<Real>(signed_mode);
      for (int radial = 0; radial < nr; ++radial) {
        const std::size_t index = static_cast<std::size_t>(radial);
        diagonal[index] = radial_diagonal[index] +
                          eigenvalue * inverse_radius_squared[index];
        mode_rhs[index] = rhs_hat[index][static_cast<std::size_t>(mode)];
      }
      if (low_dirichlet) {
        diagonal[0] -= lower[0];
        if (mode == 0)
          mode_rhs[0] -= Real(2) * lower[0] * low.value * static_cast<Real>(nth);
      } else {
        diagonal[0] += lower[0];
      }
      const std::size_t last = static_cast<std::size_t>(nr - 1);
      if (high_dirichlet) {
        diagonal[last] -= upper[last];
        if (mode == 0)
          mode_rhs[last] -=
              Real(2) * upper[last] * high.value * static_cast<Real>(nth);
      } else {
        diagonal[last] += upper[last];
      }

      thomas_(lower, diagonal, upper, mode_rhs, mode_solution,
              pin_constant && mode == 0);
      for (int radial = 0; radial < nr; ++radial)
        phi_hat[static_cast<std::size_t>(radial)][static_cast<std::size_t>(mode)] =
            mode_solution[static_cast<std::size_t>(radial)];
    }

    auto& output = trial_.fab(0);
    auto host = output.create_host_mirror();
    output.copy_to_host(host);
    const Box<Dim>& storage = output.grown_box();
    for (int radial = 0; radial < nr; ++radial) {
      auto& row = phi_hat[static_cast<std::size_t>(radial)];
      elliptic::poisson::internal::host_fft1d(row, true);
      for (int azimuthal = 0; azimuthal < nth; ++azimuthal) {
        const int i = valid.lo[0] + radial;
        const int j = valid.lo[1] + azimuthal;
        host(polar_poisson_detail::host_offset(storage, i, j)) =
            row[static_cast<std::size_t>(azimuthal)].real();
      }
    }
    output.copy_from_host(host);
  }

  void thomas_(const std::vector<Real>& lower, const std::vector<Real>& diagonal,
               const std::vector<Real>& upper, const std::vector<complex_type>& rhs,
               std::vector<complex_type>& solution, bool pin_first) {
    const std::size_t size = diagonal.size();
    working_diagonal_ = diagonal;
    working_upper_ = upper;
    working_rhs_ = rhs;
    if (pin_first) {
      working_diagonal_[0] = Real(1);
      working_upper_[0] = Real(0);
      working_rhs_[0] = complex_type{};
    }
    if (working_diagonal_[0] == Real(0))
      throw std::runtime_error("polar Poisson Thomas factorization found a null first pivot");
    solution.resize(size);
    solution[0] = working_rhs_[0] / working_diagonal_[0];
    for (std::size_t row = 1; row < size; ++row) {
      const Real multiplier = lower[row] / working_diagonal_[row - 1];
      working_diagonal_[row] -= multiplier * working_upper_[row - 1];
      working_rhs_[row] -= multiplier * working_rhs_[row - 1];
      if (working_diagonal_[row] == Real(0))
        throw std::runtime_error("polar Poisson Thomas factorization found a null pivot");
      solution[row] = working_rhs_[row] / working_diagonal_[row];
    }
    for (std::size_t row = size - 1; row-- > 0;)
      solution[row] =
          (working_rhs_[row] - working_upper_[row] * solution[row + 1]) /
          working_diagonal_[row];
  }

  Real residual_of_(const field_type& candidate) const {
    const Box<Dim>& valid = rhs_.box(0);
    const int nr = static_cast<int>(valid.length(0));
    const int nth = static_cast<int>(valid.length(1));
    const Real dr = geometry_.dr();
    auto phi_hat = transform_rows_(candidate);
    auto rhs_hat = transform_rows_(rhs_);

    const auto& low = boundary_.at(Face<Dim>{0, BoundarySide::lower});
    const auto& high = boundary_.at(Face<Dim>{0, BoundarySide::upper});
    const bool low_dirichlet = polar_poisson_detail::is_dirichlet(low);
    const bool high_dirichlet = polar_poisson_detail::is_dirichlet(high);
    const bool pin_constant = !low_dirichlet && !high_dirichlet;
    Real maximum = 0;
    for (int mode = 0; mode < nth; ++mode) {
      const int signed_mode = mode <= nth / 2 ? mode : mode - nth;
      const Real eigenvalue =
          -static_cast<Real>(signed_mode) * static_cast<Real>(signed_mode);
      for (int radial = 0; radial < nr; ++radial) {
        if (pin_constant && mode == 0 && radial == 0)
          continue;
        const int i = valid.lo[0] + radial;
        const Real radius = geometry_.r_cell(i);
        const Real inverse = Real(1) / (radius * dr * dr);
        const Real lower = geometry_.r_face(i) * inverse;
        const Real upper = geometry_.r_face(i + 1) * inverse;
        Real diagonal = -(lower + upper) + eigenvalue / (radius * radius);
        complex_type applied{};
        complex_type expected =
            rhs_hat[static_cast<std::size_t>(radial)][static_cast<std::size_t>(mode)];
        if (radial == 0) {
          diagonal += low_dirichlet ? -lower : lower;
          if (low_dirichlet && mode == 0)
            expected -= Real(2) * lower * low.value * static_cast<Real>(nth);
        } else {
          applied += lower *
                     phi_hat[static_cast<std::size_t>(radial - 1)]
                            [static_cast<std::size_t>(mode)];
        }
        if (radial == nr - 1) {
          diagonal += high_dirichlet ? -upper : upper;
          if (high_dirichlet && mode == 0)
            expected -= Real(2) * upper * high.value * static_cast<Real>(nth);
        } else {
          applied += upper *
                     phi_hat[static_cast<std::size_t>(radial + 1)]
                            [static_cast<std::size_t>(mode)];
        }
        applied += diagonal *
                   phi_hat[static_cast<std::size_t>(radial)][static_cast<std::size_t>(mode)];
        maximum = std::max(maximum,
                           static_cast<Real>(std::abs(applied - expected)) /
                               static_cast<Real>(nth));
      }
    }
    return maximum;
  }

  Real rhs_norm_inf_() const {
    const auto& fab = rhs_.fab(0);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& storage = fab.grown_box();
    Real maximum = 0;
    for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
      for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
        maximum = std::max(maximum,
                           std::abs(host(polar_poisson_detail::host_offset(storage, i, j))));
    return maximum;
  }

  PolarGeometry<Dim> geometry_;
  PhysicalBoundaryConditions<Dim> boundary_;
  field_type rhs_;
  field_type phi_;
  field_type trial_;
  std::vector<Real> working_diagonal_;
  std::vector<Real> working_upper_;
  std::vector<complex_type> working_rhs_;
  SolveReport last_report_{};
};

static_assert(PolarEllipticSolver<PolarPoissonSolver<2>>);

/// Dimension-diagnostic provider.  The build operation does not exist for rank 1 or 3.
template <int Dim>
struct PolarPoissonProvider {
  static_assert(Dim >= 1 && Dim <= 3,
                "PolarPoissonProvider only supports dimensions 1, 2, and 3");

  static constexpr bool available = PolarPoissonCapabilities<Dim>::available;
  static constexpr std::string_view rejection_reason() noexcept {
    return PolarPoissonCapabilities<Dim>::unavailable_reason;
  }

  template <int ExactDim = Dim>
    requires(PolarPoissonCapabilities<ExactDim>::available)
  static PolarPoissonSolver<ExactDim> build(PolarEllipticBuildRequest<ExactDim> request) {
    return PolarPoissonSolver<ExactDim>{std::move(request)};
  }
};

static_assert(!PolarPoissonProvider<1>::available);
static_assert(PolarPoissonProvider<2>::available);
static_assert(!PolarPoissonProvider<3>::available);

}  // namespace pops
