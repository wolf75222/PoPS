/// @file
/// @brief Exact-rank matrix-free tensor elliptic operator on a two-dimensional annulus.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/halo_exchange.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/numerics/elliptic/polar/polar_geometry.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

template <int Dim>
struct PolarTensorCapabilities {
  static_assert(Dim >= 1 && Dim <= 3,
                "PolarTensorCapabilities only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  static constexpr bool available = Dim == 2;
  static constexpr bool tensor_cross_terms = available;
  static constexpr bool distributed_halos = available;
  static constexpr std::string_view unavailable_reason =
      available ? std::string_view{}
                : std::string_view{"polar tensor elliptic operator has exactly the axes (r, theta)"};
};

static_assert(!PolarTensorCapabilities<1>::available);
static_assert(PolarTensorCapabilities<2>::available);
static_assert(!PolarTensorCapabilities<3>::available);

enum class PolarPreconditioner : unsigned char { jacobi, radial_line };

struct PolarTensorOptions {
  Real relative_tolerance = Real(1e-10);
  Real absolute_tolerance = Real(0);
  int maximum_iterations = 400;
  PolarPreconditioner preconditioner = PolarPreconditioner::radial_line;
};

namespace polar_tensor_detail {

template <int Dim>
struct ApplyKernel {
  FieldView<const Real, Dim> phi{};
  FieldView<Real, Dim> output{};
  FieldView<const Real, Dim> radial_radial{};
  FieldView<const Real, Dim> azimuthal_azimuthal{};
  FieldView<const Real, Dim> radial_azimuthal{};
  FieldView<const Real, Dim> azimuthal_radial{};
  bool has_radial_azimuthal = false;
  bool has_azimuthal_radial = false;
  int radial_origin = 0;
  Real radial_lower = 0;
  Real dr = 0;
  Real inverse_dr = 0;
  Real inverse_dtheta = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    const int i = index[0];
    const int j = index[1];
    const Index<Dim> center{i, j};
    const Index<Dim> radial_low{i - 1, j};
    const Index<Dim> radial_high{i + 1, j};
    const Index<Dim> azimuthal_low{i, j - 1};
    const Index<Dim> azimuthal_high{i, j + 1};
    const Real radial_offset = static_cast<Real>(i - radial_origin);
    const Real radius = radial_lower + (radial_offset + Real(0.5)) * dr;
    const Real radius_low = radial_lower + radial_offset * dr;
    const Real radius_high = radial_lower + (radial_offset + Real(1)) * dr;

    const Real rr_high =
        Real(0.5) * (radial_radial(center) + radial_radial(radial_high));
    const Real rr_low = Real(0.5) * (radial_radial(center) + radial_radial(radial_low));
    Real value =
        (radius_high * rr_high * (phi(radial_high) - phi(center)) -
         radius_low * rr_low * (phi(center) - phi(radial_low))) *
        (inverse_dr * inverse_dr / radius);

    const Real tt_high = Real(0.5) *
                         (azimuthal_azimuthal(center) +
                          azimuthal_azimuthal(azimuthal_high));
    const Real tt_low =
        Real(0.5) * (azimuthal_azimuthal(center) +
                     azimuthal_azimuthal(azimuthal_low));
    value += (tt_high * (phi(azimuthal_high) - phi(center)) -
              tt_low * (phi(center) - phi(azimuthal_low))) *
             (inverse_dtheta * inverse_dtheta / (radius * radius));

    if (has_radial_azimuthal) {
      const Real rt_high =
          Real(0.5) * (radial_azimuthal(center) + radial_azimuthal(radial_high));
      const Real rt_low =
          Real(0.5) * (radial_azimuthal(center) + radial_azimuthal(radial_low));
      const Real theta_gradient_high =
          (phi(Index<Dim>{i, j + 1}) + phi(Index<Dim>{i + 1, j + 1}) -
           phi(Index<Dim>{i, j - 1}) - phi(Index<Dim>{i + 1, j - 1})) *
          (Real(0.25) * inverse_dtheta);
      const Real theta_gradient_low =
          (phi(Index<Dim>{i - 1, j + 1}) + phi(Index<Dim>{i, j + 1}) -
           phi(Index<Dim>{i - 1, j - 1}) - phi(Index<Dim>{i, j - 1})) *
          (Real(0.25) * inverse_dtheta);
      value += (rt_high * theta_gradient_high - rt_low * theta_gradient_low) *
               (inverse_dr / radius);
    }

    if (has_azimuthal_radial) {
      const Real tr_high =
          Real(0.5) * (azimuthal_radial(center) + azimuthal_radial(azimuthal_high));
      const Real tr_low =
          Real(0.5) * (azimuthal_radial(center) + azimuthal_radial(azimuthal_low));
      const Real radial_gradient_high =
          (phi(Index<Dim>{i + 1, j}) + phi(Index<Dim>{i + 1, j + 1}) -
           phi(Index<Dim>{i - 1, j}) - phi(Index<Dim>{i - 1, j + 1})) *
          (Real(0.25) * inverse_dr);
      const Real radial_gradient_low =
          (phi(Index<Dim>{i + 1, j - 1}) + phi(Index<Dim>{i + 1, j}) -
           phi(Index<Dim>{i - 1, j - 1}) - phi(Index<Dim>{i - 1, j})) *
          (Real(0.25) * inverse_dr);
      value += (tr_high * radial_gradient_high - tr_low * radial_gradient_low) *
               (inverse_dtheta / radius);
    }
    output(center) = value;
  }
};

template <int Dim>
struct InverseDiagonalKernel {
  FieldView<const Real, Dim> radial_radial{};
  FieldView<const Real, Dim> azimuthal_azimuthal{};
  FieldView<Real, Dim> inverse_diagonal{};
  int radial_origin = 0;
  int radial_last = 0;
  Real radial_lower = 0;
  Real dr = 0;
  Real inverse_dr = 0;
  Real inverse_dtheta = 0;
  Real lower_ghost_scale = Real(1);
  Real upper_ghost_scale = Real(1);

  POPS_HD void operator()(const Index<Dim>& index) const {
    const int i = index[0];
    const int j = index[1];
    const Index<Dim> center{i, j};
    const Index<Dim> radial_low{i - 1, j};
    const Index<Dim> radial_high{i + 1, j};
    const Index<Dim> azimuthal_low{i, j - 1};
    const Index<Dim> azimuthal_high{i, j + 1};
    const Real radial_offset = static_cast<Real>(i - radial_origin);
    const Real radius = radial_lower + (radial_offset + Real(0.5)) * dr;
    const Real radius_low = radial_lower + radial_offset * dr;
    const Real radius_high = radial_lower + (radial_offset + Real(1)) * dr;
    const Real rr_high =
        Real(0.5) * (radial_radial(center) + radial_radial(radial_high));
    const Real rr_low = Real(0.5) * (radial_radial(center) + radial_radial(radial_low));
    const Real tt_high = Real(0.5) *
                         (azimuthal_azimuthal(center) +
                          azimuthal_azimuthal(azimuthal_high));
    const Real tt_low =
        Real(0.5) * (azimuthal_azimuthal(center) +
                     azimuthal_azimuthal(azimuthal_low));
    const Real radial_low_coefficient =
        radius_low * rr_low * (inverse_dr * inverse_dr / radius);
    const Real radial_high_coefficient =
        radius_high * rr_high * (inverse_dr * inverse_dr / radius);
    Real diagonal = -(radial_low_coefficient + radial_high_coefficient) -
                    (tt_low + tt_high) *
                        (inverse_dtheta * inverse_dtheta / (radius * radius));
    if (i == radial_origin)
      diagonal += radial_low_coefficient * lower_ghost_scale;
    if (i == radial_last)
      diagonal += radial_high_coefficient * upper_ghost_scale;
    inverse_diagonal(center) = diagonal != Real(0) ? Real(1) / diagonal : Real(0);
  }
};

template <int Dim>
struct JacobiKernel {
  FieldView<const Real, Dim> input{};
  FieldView<const Real, Dim> inverse_diagonal{};
  FieldView<Real, Dim> output{};
  POPS_HD void operator()(const Index<Dim>& index) const {
    output(index) = input(index) * inverse_diagonal(index);
  }
};

inline std::size_t checked_multiply(std::size_t left, std::size_t right,
                                    const char* operation) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(operation);
  return left * right;
}

template <int Dim>
HaloScheduleBudget exact_halo_budget(const mesh::BoxArray<Dim>& layout,
                                     const Box<Dim>& domain) {
  const std::size_t boxes = layout.size();
  const std::size_t pairs = checked_multiply(boxes, boxes, "polar halo pair budget overflow");
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis)
    images = checked_multiply(images, 3, "polar halo image budget overflow");
  const std::size_t work =
      checked_multiply(pairs, images, "polar halo work budget overflow");
  const std::size_t jobs =
      checked_multiply(work, static_cast<std::size_t>(2 * Dim),
                       "polar halo job budget overflow");
  const std::size_t cells = static_cast<std::size_t>(domain.numPts());
  const std::size_t elements =
      checked_multiply(jobs, cells, "polar halo element budget overflow");
  return {mesh::BoxArrayValidationBudget{boxes, pairs},
          work,
          jobs,
          images,
          checked_multiply(boxes, std::size_t{2}, "polar halo peer budget overflow"),
          elements,
          elements,
          elements};
}

inline Real homogeneous_ghost_scale(const PhysicalBoundaryFace& face, Real spacing) {
  if (face.kind == PhysicalBoundaryKind::dirichlet)
    return Real(-1);
  if (face.kind == PhysicalBoundaryKind::constant_extrapolation ||
      face.kind == PhysicalBoundaryKind::neumann)
    return Real(1);
  if (face.kind == PhysicalBoundaryKind::robin) {
    const Real denominator = face.alpha / Real(2) + face.beta / spacing;
    if (!std::isfinite(static_cast<double>(denominator)) || denominator == Real(0))
      throw std::invalid_argument("polar tensor Robin boundary has a singular ghost transform");
    return -(face.alpha / Real(2) - face.beta / spacing) / denominator;
  }
  throw std::invalid_argument("polar tensor radial boundary is externally owned");
}

inline bool fixes_gauge(const PhysicalBoundaryFace& face) noexcept {
  return face.kind == PhysicalBoundaryKind::dirichlet ||
         (face.kind == PhysicalBoundaryKind::robin && face.alpha != Real(0));
}

template <int Dim>
PhysicalBoundaryConditions<Dim> homogeneous_boundary(
    const PhysicalBoundaryConditions<Dim>& source) {
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  for (int axis = 0; axis < Dim; ++axis) {
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      PhysicalBoundaryFace law = source.at(face);
      law.value = Real(0);
      faces[static_cast<std::size_t>(face.ordinal())] = law;
    }
  }
  return {source.topology(), faces, source.spacing()};
}

template <int Dim>
PhysicalBoundaryConditions<Dim> coefficient_boundary(
    const PhysicalBoundaryConditions<Dim>& source) {
  std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
  for (int axis = 0; axis < Dim; ++axis) {
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      faces[static_cast<std::size_t>(face.ordinal())] =
          source.topology().is_periodic(face)
              ? PhysicalBoundaryFace{}
              : PhysicalBoundaryFace{PhysicalBoundaryKind::constant_extrapolation};
    }
  }
  return {source.topology(), faces, source.spacing()};
}

inline std::size_t host_offset(const Box<2>& storage, int i, int j) {
  return static_cast<std::size_t>(i - storage.lo[0]) +
         static_cast<std::size_t>(j - storage.lo[1]) *
             static_cast<std::size_t>(storage.length(0));
}

template <int Dim>
void require_same_scalar_layout(const MultiFab<Dim>& reference, const MultiFab<Dim>& field,
                                const char* role) {
  if (reference.layout() != field.layout() ||
      reference.distribution() != field.distribution() ||
      reference.local_rank() != field.local_rank() || field.ncomp() != 1)
    throw std::invalid_argument(std::string("polar tensor ") + role +
                                " differs from the exact solver layout");
  for (int axis = 0; axis < Dim; ++axis)
    if (field.ghosts()[axis] < 1)
      throw std::invalid_argument(std::string("polar tensor ") + role +
                                  " requires one ghost on every axis");
}

}  // namespace polar_tensor_detail

/// Apply div(A grad(phi)) over valid cells.  All operands retain the same exact compile-time rank;
/// the caller owns same-level, periodic, and physical ghost preparation.
template <int Dim>
  requires(PolarTensorCapabilities<Dim>::available)
void apply_polar_tensor(const MultiFab<Dim>& phi, const PolarGeometry<Dim>& geometry,
                        MultiFab<Dim>& output, const MultiFab<Dim>& radial_radial,
                        const MultiFab<Dim>& azimuthal_azimuthal,
                        const MultiFab<Dim>* radial_azimuthal = nullptr,
                        const MultiFab<Dim>* azimuthal_radial = nullptr) {
  polar_tensor_detail::require_same_scalar_layout(phi, radial_radial, "a_rr");
  polar_tensor_detail::require_same_scalar_layout(phi, azimuthal_azimuthal, "a_tt");
  if (radial_azimuthal)
    polar_tensor_detail::require_same_scalar_layout(phi, *radial_azimuthal, "a_rt");
  if (azimuthal_radial)
    polar_tensor_detail::require_same_scalar_layout(phi, *azimuthal_radial, "a_tr");
  if (output.layout() != phi.layout() || output.distribution() != phi.distribution() ||
      output.local_rank() != phi.local_rank() || output.ncomp() != 1)
    throw std::invalid_argument("polar tensor output differs from the exact input layout");

  const Real inverse_dr = Real(1) / geometry.dr();
  const Real inverse_dtheta = Real(1) / geometry.dtheta();
  for (std::size_t local = 0; local < phi.local_size(); ++local) {
    const FieldView<const Real, Dim> empty{};
    for_each_cell(
        output.box(local),
        polar_tensor_detail::ApplyKernel<Dim>{
            phi.fab(local).view(), output.fab(local).view(), radial_radial.fab(local).view(),
            azimuthal_azimuthal.fab(local).view(),
            radial_azimuthal ? radial_azimuthal->fab(local).view() : empty,
            azimuthal_radial ? azimuthal_radial->fab(local).view() : empty,
            radial_azimuthal != nullptr, azimuthal_radial != nullptr,
            geometry.domain().lo[0], geometry.radial_lower(), geometry.dr(), inverse_dr,
            inverse_dtheta});
  }
}

template <class Solver>
concept PolarLinearSolver = requires(Solver solver, const Solver constant_solver) {
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

template <int Dim>
  requires(PolarTensorCapabilities<Dim>::available)
class PolarTensorKrylovSolver {
 public:
  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim>;
  using request_type = PolarEllipticBuildRequest<Dim>;

  explicit PolarTensorKrylovSolver(request_type request, PolarTensorOptions options = {})
      : geometry_(request.geometry),
        boundary_(request.boundary),
        homogeneous_boundary_(polar_tensor_detail::homogeneous_boundary(request.boundary)),
        coefficient_boundary_(polar_tensor_detail::coefficient_boundary(request.boundary)),
        options_(options),
        rhs_(make_field_(request, Extent<Dim>{})),
        phi_(make_field_(request, unit_ghosts_())),
        trial_(make_field_(request, unit_ghosts_())),
        residual_(make_field_(request, unit_ghosts_())),
        shadow_residual_(make_field_(request, unit_ghosts_())),
        direction_(make_field_(request, unit_ghosts_())),
        operator_direction_(make_field_(request, unit_ghosts_())),
        intermediate_(make_field_(request, unit_ghosts_())),
        operator_intermediate_(make_field_(request, unit_ghosts_())),
        preconditioned_direction_(make_field_(request, unit_ghosts_())),
        preconditioned_intermediate_(make_field_(request, unit_ghosts_())),
        inverse_diagonal_(make_field_(request, unit_ghosts_())),
        radial_radial_store_(make_field_(request, unit_ghosts_())),
        azimuthal_azimuthal_store_(make_field_(request, unit_ghosts_())) {
    validate_request_(request);
    validate_options_();
    radial_radial_store_.set_val(Real(1));
    azimuthal_azimuthal_store_.set_val(Real(1));
    radial_radial_ = &radial_radial_store_;
    azimuthal_azimuthal_ = &azimuthal_azimuthal_store_;
    prepare_boundaries_and_halo_();
  }

  PolarTensorKrylovSolver(const PolarTensorKrylovSolver&) = delete;
  PolarTensorKrylovSolver& operator=(const PolarTensorKrylovSolver&) = delete;
  PolarTensorKrylovSolver(PolarTensorKrylovSolver&&) noexcept = default;
  PolarTensorKrylovSolver& operator=(PolarTensorKrylovSolver&&) noexcept = default;
  ~PolarTensorKrylovSolver() noexcept = default;

  static constexpr PolarTensorCapabilities<Dim> capabilities() noexcept { return {}; }

  field_type& rhs() noexcept { return rhs_; }
  const field_type& rhs() const noexcept { return rhs_; }
  field_type& phi() noexcept { return phi_; }
  const field_type& phi() const noexcept { return phi_; }
  const PolarGeometry<Dim>& geom() const noexcept { return geometry_; }
  const PhysicalBoundaryConditions<Dim>& boundary() const noexcept { return boundary_; }
  const SolveReport& last_solve_report() const noexcept { return last_report_; }
  Real residual() const noexcept { return last_report_.residual_norm; }

  void set_coefficients(field_type& radial_radial, field_type& azimuthal_azimuthal,
                        field_type* radial_azimuthal = nullptr,
                        field_type* azimuthal_radial = nullptr) {
    polar_tensor_detail::require_same_scalar_layout(phi_, radial_radial, "a_rr");
    polar_tensor_detail::require_same_scalar_layout(phi_, azimuthal_azimuthal, "a_tt");
    if (radial_azimuthal)
      polar_tensor_detail::require_same_scalar_layout(phi_, *radial_azimuthal, "a_rt");
    if (azimuthal_radial)
      polar_tensor_detail::require_same_scalar_layout(phi_, *azimuthal_radial, "a_tr");
    radial_radial_ = &radial_radial;
    azimuthal_azimuthal_ = &azimuthal_azimuthal;
    radial_azimuthal_ = radial_azimuthal;
    azimuthal_radial_ = azimuthal_radial;
    coefficients_ready_ = false;
  }

  SolveReport solve() {
    SolveReport report;
    try {
      prepare_coefficients_();
      lincomb(trial_, Real(1), phi_, Real(0), phi_);
      apply_full_(trial_, operator_direction_);
      lincomb(residual_, Real(1), rhs_, Real(-1), operator_direction_);
      if (pin_gauge_)
        project_mean_(residual_);
      lincomb(shadow_residual_, Real(1), residual_, Real(0), residual_);
      direction_.set_val(Real(0));
      operator_direction_.set_val(Real(0));

      report.reference_residual_norm = reduce_norm_inf(residual_);
      const Real tolerance = std::max(options_.absolute_tolerance,
                                      options_.relative_tolerance *
                                          report.reference_residual_norm);
      if (report.reference_residual_norm <= tolerance) {
        report.evaluations = 1;
        report.residual_norm = report.reference_residual_norm;
        report.rel_residual = report.reference_residual_norm > Real(0) ? Real(1) : Real(0);
        publish_(report, "polar_tensor_initial_candidate");
        return last_report_;
      }

      Real rho_previous = Real(1);
      Real alpha = Real(1);
      Real omega = Real(1);
      for (int iteration = 1; iteration <= options_.maximum_iterations; ++iteration) {
        const Real rho = dot(shadow_residual_, residual_);
        if (!finite_nonzero_(rho))
          return fail_(report, SolveStatus::kBreakdown, iteration,
                       "polar_tensor_bicgstab_rho_breakdown");
        if (iteration == 1) {
          lincomb(direction_, Real(1), residual_, Real(0), residual_);
        } else {
          if (!finite_nonzero_(omega))
            return fail_(report, SolveStatus::kBreakdown, iteration,
                         "polar_tensor_bicgstab_omega_breakdown");
          const Real beta = (rho / rho_previous) * (alpha / omega);
          lincomb(direction_, Real(1), direction_, -omega, operator_direction_);
          lincomb(direction_, beta, direction_, Real(1), residual_);
        }

        apply_preconditioner_(direction_, preconditioned_direction_);
        if (pin_gauge_)
          project_mean_(preconditioned_direction_);
        apply_linear_(preconditioned_direction_, operator_direction_);
        const Real denominator = dot(shadow_residual_, operator_direction_);
        if (!finite_nonzero_(denominator))
          return fail_(report, SolveStatus::kBreakdown, iteration,
                       "polar_tensor_bicgstab_alpha_breakdown");
        alpha = rho / denominator;
        lincomb(intermediate_, Real(1), residual_, -alpha, operator_direction_);

        if (reduce_norm_inf(intermediate_) <= tolerance) {
          saxpy(trial_, alpha, preconditioned_direction_);
          report.iters = iteration;
          return authenticate_and_publish_(report, "polar_tensor_bicgstab");
        }

        apply_preconditioner_(intermediate_, preconditioned_intermediate_);
        if (pin_gauge_)
          project_mean_(preconditioned_intermediate_);
        apply_linear_(preconditioned_intermediate_, operator_intermediate_);
        const Real tt = dot(operator_intermediate_, operator_intermediate_);
        if (!finite_nonzero_(tt))
          return fail_(report, SolveStatus::kBreakdown, iteration,
                       "polar_tensor_bicgstab_t_norm_breakdown");
        omega = dot(operator_intermediate_, intermediate_) / tt;
        if (!finite_nonzero_(omega))
          return fail_(report, SolveStatus::kBreakdown, iteration,
                       "polar_tensor_bicgstab_omega_breakdown");
        saxpy(trial_, alpha, preconditioned_direction_);
        saxpy(trial_, omega, preconditioned_intermediate_);
        lincomb(residual_, Real(1), intermediate_, -omega, operator_intermediate_);
        if (pin_gauge_)
          project_mean_(residual_);
        report.iters = iteration;
        if (reduce_norm_inf(residual_) <= tolerance)
          return authenticate_and_publish_(report, "polar_tensor_bicgstab");
        rho_previous = rho;
      }
      return fail_(report, SolveStatus::kIterationLimit, options_.maximum_iterations,
                   "polar_tensor_iteration_limit");
    } catch (const std::exception& error) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         std::string("polar_tensor_solve_failed: ") + error.what());
      last_report_ = report;
      return last_report_;
    }
  }

 private:
  static Extent<Dim> unit_ghosts_() {
    Extent<Dim> ghosts{};
    for (int axis = 0; axis < Dim; ++axis)
      ghosts[axis] = 1;
    return ghosts;
  }

  static field_type make_field_(const request_type& request, Extent<Dim> ghosts) {
    return field_type(request.boxes, request.distribution, request.local_rank, 1, ghosts);
  }

  static bool finite_nonzero_(Real value) noexcept {
    return std::isfinite(static_cast<double>(value)) && value != Real(0);
  }

  void validate_options_() const {
    if (!std::isfinite(static_cast<double>(options_.relative_tolerance)) ||
        !std::isfinite(static_cast<double>(options_.absolute_tolerance)) ||
        options_.relative_tolerance < Real(0) || options_.absolute_tolerance < Real(0) ||
        options_.maximum_iterations < 1)
      throw std::invalid_argument("polar tensor options are invalid");
  }

  static void validate_request_(const request_type& request) {
    if (request.geometry.domain().empty() || request.boxes.empty() ||
        !request.boxes.tiles_exactly(request.geometry.domain(), request.layout_budget) ||
        !request.distribution.matches_layout(request.boxes) ||
        !request.distribution.rank_space().contains(request.local_rank) ||
        request.distribution.rank_space().size() != static_cast<std::size_t>(n_ranks()) ||
        request.distribution.rank_space().linear_rank(request.local_rank) !=
            static_cast<std::size_t>(my_rank()))
      throw std::invalid_argument("polar tensor received an invalid exact-ranked layout request");
    if (request.boundary.spacing()[0] != request.geometry.dr() ||
        request.boundary.spacing()[1] != request.geometry.dtheta())
      throw std::invalid_argument("polar tensor boundary spacing differs from its geometry");
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> radial{0, side};
      const Face<Dim> azimuthal{1, side};
      if (request.boundary.topology().is_periodic(radial) ||
          request.boundary.at(radial).kind == PhysicalBoundaryKind::external)
        throw std::invalid_argument("polar tensor radial boundary must be physically owned");
      (void)polar_tensor_detail::homogeneous_ghost_scale(request.boundary.at(radial),
                                                         request.geometry.dr());
      if (!request.boundary.topology().is_periodic(azimuthal) ||
          request.boundary.at(azimuthal).kind != PhysicalBoundaryKind::external)
        throw std::invalid_argument("polar tensor azimuthal boundary must be periodic");
    }
  }

  void prepare_boundaries_and_halo_() {
    const Extent<Dim> ghosts = unit_ghosts_();
    full_physical_ = std::make_unique<PreparedPhysicalBoundary<Dim>>(
        prepare_physical_boundary(geometry_.domain(), ghosts, boundary_,
                                  BoundaryScheduleBudget{8}));
    homogeneous_physical_ = std::make_unique<PreparedPhysicalBoundary<Dim>>(
        prepare_physical_boundary(geometry_.domain(), ghosts, homogeneous_boundary_,
                                  BoundaryScheduleBudget{8}));
    coefficient_physical_ = std::make_unique<PreparedPhysicalBoundary<Dim>>(
        prepare_physical_boundary(geometry_.domain(), ghosts, coefficient_boundary_,
                                  BoundaryScheduleBudget{8}));
    halo_schedule_ = std::make_unique<HaloSchedule<Dim>>(prepare_halo_schedule(
        trial_, geometry_.domain(), boundary_.topology(),
        polar_tensor_detail::exact_halo_budget(trial_.layout(), geometry_.domain())));
    const bool remote = all_reduce_max(halo_schedule_->has_remote_jobs() ? 1L : 0L) != 0;
    if (remote) {
      halo_lane_ = std::make_unique<ExecutionLane>(ExecutionLane::duplicate_world_collectively(
          "pops.polar-tensor.exact-rank2/halo"));
      HaloExchangeContext context{};
      context.context_generation = 1;
      context.schedule_generation = 1;
      halo_exchange_ =
          std::make_unique<HaloExchange<Dim>>(*halo_schedule_, *halo_lane_, context);
    }
  }

  void fill_(field_type& field, const PreparedPhysicalBoundary<Dim>& physical) {
    if (halo_exchange_)
      halo_exchange_->execute(field, *halo_lane_);
    else
      fill_boundary(field, *halo_schedule_);
    fill_physical_boundary(field, physical);
  }

  void apply_full_(field_type& input, field_type& output) {
    fill_(input, *full_physical_);
    apply_polar_tensor(input, geometry_, output, *radial_radial_,
                       *azimuthal_azimuthal_, radial_azimuthal_, azimuthal_radial_);
  }

  void apply_linear_(field_type& input, field_type& output) {
    fill_(input, *homogeneous_physical_);
    apply_polar_tensor(input, geometry_, output, *radial_radial_,
                       *azimuthal_azimuthal_, radial_azimuthal_, azimuthal_radial_);
  }

  void prepare_coefficients_() {
    if (coefficients_ready_)
      return;
    fill_(*radial_radial_, *coefficient_physical_);
    fill_(*azimuthal_azimuthal_, *coefficient_physical_);
    if (radial_azimuthal_)
      fill_(*radial_azimuthal_, *coefficient_physical_);
    if (azimuthal_radial_)
      fill_(*azimuthal_radial_, *coefficient_physical_);

    const Real lower_scale = polar_tensor_detail::homogeneous_ghost_scale(
        boundary_.at(Face<Dim>{0, BoundarySide::lower}), geometry_.dr());
    const Real upper_scale = polar_tensor_detail::homogeneous_ghost_scale(
        boundary_.at(Face<Dim>{0, BoundarySide::upper}), geometry_.dr());
    const Real inverse_dr = Real(1) / geometry_.dr();
    const Real inverse_dtheta = Real(1) / geometry_.dtheta();
    for (std::size_t local = 0; local < inverse_diagonal_.local_size(); ++local)
      for_each_cell(
          inverse_diagonal_.box(local),
          polar_tensor_detail::InverseDiagonalKernel<Dim>{
              std::as_const(*radial_radial_).fab(local).view(),
              std::as_const(*azimuthal_azimuthal_).fab(local).view(),
              inverse_diagonal_.fab(local).view(), geometry_.domain().lo[0],
              geometry_.domain().hi[0], geometry_.radial_lower(), geometry_.dr(), inverse_dr,
              inverse_dtheta, lower_scale, upper_scale});
    if (options_.preconditioner == PolarPreconditioner::radial_line)
      build_radial_lines_(lower_scale, upper_scale);
    pin_gauge_ = !polar_tensor_detail::fixes_gauge(
                     boundary_.at(Face<Dim>{0, BoundarySide::lower})) &&
                 !polar_tensor_detail::fixes_gauge(
                     boundary_.at(Face<Dim>{0, BoundarySide::upper}));
    coefficients_ready_ = true;
  }

  void build_radial_lines_(Real lower_scale, Real upper_scale) {
    for (const Box<Dim>& box : radial_radial_->layout().boxes())
      if (box.lo[0] != geometry_.domain().lo[0] ||
          box.hi[0] != geometry_.domain().hi[0])
        throw std::invalid_argument(
            "polar radial-line preconditioner requires every patch to span the radial domain");

    const int radial_cells = static_cast<int>(geometry_.domain().length(0));
    line_diagonal_.resize(radial_radial_->local_size());
    line_lower_.resize(radial_radial_->local_size());
    line_upper_.resize(radial_radial_->local_size());
    for (std::size_t local = 0; local < radial_radial_->local_size(); ++local) {
      const auto& rr_fab = radial_radial_->fab(local);
      const auto& tt_fab = azimuthal_azimuthal_->fab(local);
      auto rr = rr_fab.create_host_mirror();
      auto tt = tt_fab.create_host_mirror();
      rr_fab.copy_to_host(rr);
      tt_fab.copy_to_host(tt);
      const Box<Dim>& valid = rr_fab.box();
      const Box<Dim>& rr_storage = rr_fab.grown_box();
      const Box<Dim>& tt_storage = tt_fab.grown_box();
      const int local_theta = static_cast<int>(valid.length(1));
      const std::size_t entries =
          static_cast<std::size_t>(radial_cells) * static_cast<std::size_t>(local_theta);
      auto& diagonal = line_diagonal_[local];
      auto& lower = line_lower_[local];
      auto& upper = line_upper_[local];
      diagonal.assign(entries, Real(0));
      lower.assign(entries, Real(0));
      upper.assign(entries, Real(0));
      for (int theta = 0; theta < local_theta; ++theta) {
        const int j = valid.lo[1] + theta;
        for (int radial = 0; radial < radial_cells; ++radial) {
          const int i = geometry_.domain().lo[0] + radial;
          const Real radius = geometry_.r_cell(i);
          const Real low_radius = geometry_.r_face(i);
          const Real high_radius = geometry_.r_face(i + 1);
          const Real rr_high = Real(0.5) *
                               (rr(polar_tensor_detail::host_offset(rr_storage, i, j)) +
                                rr(polar_tensor_detail::host_offset(rr_storage, i + 1, j)));
          const Real rr_low = Real(0.5) *
                              (rr(polar_tensor_detail::host_offset(rr_storage, i, j)) +
                               rr(polar_tensor_detail::host_offset(rr_storage, i - 1, j)));
          const Real tt_high = Real(0.5) *
                               (tt(polar_tensor_detail::host_offset(tt_storage, i, j)) +
                                tt(polar_tensor_detail::host_offset(tt_storage, i, j + 1)));
          const Real tt_low = Real(0.5) *
                              (tt(polar_tensor_detail::host_offset(tt_storage, i, j)) +
                               tt(polar_tensor_detail::host_offset(tt_storage, i, j - 1)));
          const Real lower_coefficient =
              low_radius * rr_low / (radius * geometry_.dr() * geometry_.dr());
          const Real upper_coefficient =
              high_radius * rr_high / (radius * geometry_.dr() * geometry_.dr());
          const Real azimuthal_diagonal =
              (tt_low + tt_high) /
              (radius * radius * geometry_.dtheta() * geometry_.dtheta());
          const std::size_t entry = static_cast<std::size_t>(theta) *
                                        static_cast<std::size_t>(radial_cells) +
                                    static_cast<std::size_t>(radial);
          lower[entry] = radial == 0 ? Real(0) : lower_coefficient;
          upper[entry] = radial == radial_cells - 1 ? Real(0) : upper_coefficient;
          diagonal[entry] = -(lower_coefficient + upper_coefficient) - azimuthal_diagonal;
          if (radial == 0)
            diagonal[entry] += lower_coefficient * lower_scale;
          if (radial == radial_cells - 1)
            diagonal[entry] += upper_coefficient * upper_scale;
        }
      }
    }
  }

  void apply_preconditioner_(const field_type& input, field_type& output) {
    if (options_.preconditioner == PolarPreconditioner::jacobi) {
      for (std::size_t local = 0; local < output.local_size(); ++local)
        for_each_cell(output.box(local), polar_tensor_detail::JacobiKernel<Dim>{
                                             input.fab(local).view(),
                                             std::as_const(inverse_diagonal_).fab(local).view(),
                                             output.fab(local).view()});
      return;
    }
    apply_radial_lines_(input, output);
  }

  void apply_radial_lines_(const field_type& input, field_type& output) {
    const int radial_cells = static_cast<int>(geometry_.domain().length(0));
    std::vector<Real> modified_upper(static_cast<std::size_t>(radial_cells));
    std::vector<Real> solution(static_cast<std::size_t>(radial_cells));
    for (std::size_t local = 0; local < input.local_size(); ++local) {
      const auto& input_fab = input.fab(local);
      auto& output_fab = output.fab(local);
      auto input_host = input_fab.create_host_mirror();
      auto output_host = output_fab.create_host_mirror();
      input_fab.copy_to_host(input_host);
      output_fab.copy_to_host(output_host);
      const Box<Dim>& valid = input_fab.box();
      const Box<Dim>& input_storage = input_fab.grown_box();
      const Box<Dim>& output_storage = output_fab.grown_box();
      const int local_theta = static_cast<int>(valid.length(1));
      for (int theta = 0; theta < local_theta; ++theta) {
        const int j = valid.lo[1] + theta;
        const std::size_t base = static_cast<std::size_t>(theta) *
                                 static_cast<std::size_t>(radial_cells);
        Real pivot = line_diagonal_[local][base];
        if (!finite_nonzero_(pivot))
          throw std::runtime_error("polar radial-line preconditioner found a null pivot");
        modified_upper[0] = line_upper_[local][base] / pivot;
        solution[0] = input_host(polar_tensor_detail::host_offset(
                          input_storage, geometry_.domain().lo[0], j)) /
                      pivot;
        for (int radial = 1; radial < radial_cells; ++radial) {
          const std::size_t entry = base + static_cast<std::size_t>(radial);
          pivot = line_diagonal_[local][entry] -
                  line_lower_[local][entry] *
                      modified_upper[static_cast<std::size_t>(radial - 1)];
          if (!finite_nonzero_(pivot))
            throw std::runtime_error("polar radial-line preconditioner found a null pivot");
          modified_upper[static_cast<std::size_t>(radial)] =
              radial + 1 < radial_cells ? line_upper_[local][entry] / pivot : Real(0);
          const int i = geometry_.domain().lo[0] + radial;
          solution[static_cast<std::size_t>(radial)] =
              (input_host(polar_tensor_detail::host_offset(input_storage, i, j)) -
               line_lower_[local][entry] *
                   solution[static_cast<std::size_t>(radial - 1)]) /
              pivot;
        }
        for (int radial = radial_cells - 2; radial >= 0; --radial)
          solution[static_cast<std::size_t>(radial)] -=
              modified_upper[static_cast<std::size_t>(radial)] *
              solution[static_cast<std::size_t>(radial + 1)];
        for (int radial = 0; radial < radial_cells; ++radial) {
          const int i = geometry_.domain().lo[0] + radial;
          output_host(polar_tensor_detail::host_offset(output_storage, i, j)) =
              solution[static_cast<std::size_t>(radial)];
        }
      }
      output_fab.copy_from_host(output_host);
    }
  }

  void project_mean_(field_type& field) {
    Real local_sum = 0;
    Real local_measure = 0;
    for (std::size_t local = 0; local < field.local_size(); ++local) {
      auto& fab = field.fab(local);
      auto host = fab.create_host_mirror();
      fab.copy_to_host(host);
      const Box<Dim>& valid = fab.box();
      const Box<Dim>& storage = fab.grown_box();
      for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
          const Real weight = geometry_.r_cell(i) * geometry_.dr() * geometry_.dtheta();
          local_sum += host(polar_tensor_detail::host_offset(storage, i, j)) * weight;
          local_measure += weight;
        }
    }
    const Real global_sum = static_cast<Real>(all_reduce_sum(static_cast<double>(local_sum)));
    const Real global_measure =
        static_cast<Real>(all_reduce_sum(static_cast<double>(local_measure)));
    const Real mean = global_measure > Real(0) ? global_sum / global_measure : Real(0);
    for (std::size_t local = 0; local < field.local_size(); ++local) {
      auto& fab = field.fab(local);
      auto host = fab.create_host_mirror();
      fab.copy_to_host(host);
      const Box<Dim>& valid = fab.box();
      const Box<Dim>& storage = fab.grown_box();
      for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
          host(polar_tensor_detail::host_offset(storage, i, j)) -= mean;
      fab.copy_from_host(host);
    }
  }

  SolveReport authenticate_and_publish_(SolveReport report, std::string reason) {
    apply_full_(trial_, operator_direction_);
    lincomb(residual_, Real(1), rhs_, Real(-1), operator_direction_);
    report.evaluations += 1;
    report.residual_norm = reduce_norm_inf(residual_);
    report.rel_residual = report.reference_residual_norm > Real(0)
                              ? report.residual_norm / report.reference_residual_norm
                              : report.residual_norm;
    const Real tolerance = std::max(options_.absolute_tolerance,
                                    options_.relative_tolerance *
                                        report.reference_residual_norm);
    if (!std::isfinite(static_cast<double>(report.residual_norm)) ||
        report.residual_norm > tolerance)
      return fail_(report, SolveStatus::kInadmissibleCandidate, report.iters,
                   "polar_tensor_final_residual_rejected");
    publish_(report, std::move(reason));
    return last_report_;
  }

  void publish_(SolveReport& report, std::string reason) {
    if (pin_gauge_)
      project_mean_(trial_);
    std::swap(phi_, trial_);
    report.mark_solved(std::move(reason));
    last_report_ = report;
  }

  SolveReport fail_(SolveReport report, SolveStatus status, int iterations,
                    std::string reason) {
    report.iters = iterations;
    report.residual_norm = reduce_norm_inf(residual_);
    report.rel_residual = report.reference_residual_norm > Real(0)
                              ? report.residual_norm / report.reference_residual_norm
                              : report.residual_norm;
    report.mark_failed(status, SolveAction::kFailRun, std::move(reason));
    last_report_ = report;
    return last_report_;
  }

  PolarGeometry<Dim> geometry_;
  PhysicalBoundaryConditions<Dim> boundary_;
  PhysicalBoundaryConditions<Dim> homogeneous_boundary_;
  PhysicalBoundaryConditions<Dim> coefficient_boundary_;
  PolarTensorOptions options_;
  field_type rhs_;
  field_type phi_;
  field_type trial_;
  field_type residual_;
  field_type shadow_residual_;
  field_type direction_;
  field_type operator_direction_;
  field_type intermediate_;
  field_type operator_intermediate_;
  field_type preconditioned_direction_;
  field_type preconditioned_intermediate_;
  field_type inverse_diagonal_;
  field_type radial_radial_store_;
  field_type azimuthal_azimuthal_store_;
  field_type* radial_radial_ = nullptr;
  field_type* azimuthal_azimuthal_ = nullptr;
  field_type* radial_azimuthal_ = nullptr;
  field_type* azimuthal_radial_ = nullptr;
  std::unique_ptr<PreparedPhysicalBoundary<Dim>> full_physical_;
  std::unique_ptr<PreparedPhysicalBoundary<Dim>> homogeneous_physical_;
  std::unique_ptr<PreparedPhysicalBoundary<Dim>> coefficient_physical_;
  std::unique_ptr<HaloSchedule<Dim>> halo_schedule_;
  std::unique_ptr<ExecutionLane> halo_lane_;
  std::unique_ptr<HaloExchange<Dim>> halo_exchange_;
  std::vector<std::vector<Real>> line_diagonal_;
  std::vector<std::vector<Real>> line_lower_;
  std::vector<std::vector<Real>> line_upper_;
  bool coefficients_ready_ = false;
  bool pin_gauge_ = false;
  SolveReport last_report_{};
};

static_assert(PolarLinearSolver<PolarTensorKrylovSolver<2>>);

template <int Dim>
struct PolarTensorProvider {
  static_assert(Dim >= 1 && Dim <= 3,
                "PolarTensorProvider only supports dimensions 1, 2, and 3");

  static constexpr bool available = PolarTensorCapabilities<Dim>::available;
  static constexpr std::string_view rejection_reason() noexcept {
    return PolarTensorCapabilities<Dim>::unavailable_reason;
  }

  template <int ExactDim = Dim>
    requires(PolarTensorCapabilities<ExactDim>::available)
  static PolarTensorKrylovSolver<ExactDim> build(
      PolarEllipticBuildRequest<ExactDim> request, PolarTensorOptions options = {}) {
    return PolarTensorKrylovSolver<ExactDim>{std::move(request), options};
  }
};

static_assert(!PolarTensorProvider<1>::available);
static_assert(PolarTensorProvider<2>::available);
static_assert(!PolarTensorProvider<3>::available);

}  // namespace pops
