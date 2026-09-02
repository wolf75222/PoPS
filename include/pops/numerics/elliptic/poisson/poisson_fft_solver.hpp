/// @file
/// @brief Exact-ranked wrapper for the device-resident Cartesian FFT engine.

#pragma once

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/halo_exchange.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_workspace.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

enum class PoissonFFTSymbol : unsigned char {
  discrete_cartesian,
  continuous_spectral,
};

/// Compile-time capability declaration for the one Cartesian ND FFT implementation.
template <int Dim>
struct PoissonFFTCapabilities {
  static_assert(Dim >= 1 && Dim <= 3,
                "PoissonFFTCapabilities only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  static constexpr bool available = true;
  static constexpr bool periodic = available;
  static constexpr bool distributed_slabs = available;
  static constexpr bool discrete_symbol = available;
  // The low-level engine can invert this symbol, but does not expose the matching apply operation.
  // Publishing it as an EllipticSolver would therefore require a fabricated residual norm.
  static constexpr bool continuous_spectral_symbol = false;
  static constexpr std::string_view unavailable_reason{};

  static constexpr bool supports(PoissonFFTSymbol symbol) noexcept {
    return symbol == PoissonFFTSymbol::discrete_cartesian && discrete_symbol;
  }

  static constexpr std::string_view rejection_reason(PoissonFFTSymbol symbol) noexcept {
    if (!available)
      return unavailable_reason;
    return supports(symbol)
               ? std::string_view{}
               : std::string_view{"continuous spectral FFT has no exact apply/residual provider"};
  }
};

static_assert(PoissonFFTCapabilities<1>::available);
static_assert(PoissonFFTCapabilities<2>::available);
static_assert(PoissonFFTCapabilities<3>::available);

namespace fft_solver_detail {

inline constexpr Real kDirectResidualSafetyFactor = Real(512);

inline std::string options_contract() {
  ExactContractBuilder contract;
  contract.text("pops.elliptic.poisson-fft.options")
      .scalar(std::uint32_t{3})
      .text("discrete-cartesian")
      .scalar(kDirectResidualSafetyFactor);
  return std::move(contract).release();
}

inline std::size_t checked_multiply(std::size_t left, std::size_t right, const char* operation) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(operation);
  return left * right;
}

template <int Dim>
HaloScheduleBudget exact_halo_budget(const mesh::BoxArray<Dim>& layout, const Box<Dim>& domain) {
  const std::size_t boxes = layout.size();
  const std::size_t pairs = checked_multiply(boxes, boxes, "FFT halo pair budget overflow");
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis)
    images = checked_multiply(images, 3, "FFT halo image budget overflow");
  const std::size_t work = checked_multiply(pairs, images, "FFT halo work budget overflow");
  const std::size_t jobs =
      checked_multiply(work, static_cast<std::size_t>(2 * Dim), "FFT halo job budget overflow");
  const std::int64_t signed_cells = domain.numPts();
  if (signed_cells <= 0)
    throw std::invalid_argument("FFT halo domain must be non-empty");
  const std::size_t cells = static_cast<std::size_t>(signed_cells);
  const std::size_t elements = checked_multiply(jobs, cells, "FFT halo element budget overflow");
  return HaloScheduleBudget{
      mesh::BoxArrayValidationBudget{boxes, pairs},
      work,
      jobs,
      images,
      checked_multiply(boxes, std::size_t{2}, "FFT halo peer budget overflow"),
      elements,
      elements,
      elements};
}

template <int Dim>
void validate_periodic_boundary(const Geometry<Dim>& geometry,
                                const PhysicalBoundaryConditions<Dim>& boundary) {
  for (int axis = 0; axis < Dim; ++axis) {
    if (boundary.spacing()[axis] != geometry.spacing(axis))
      throw std::invalid_argument("FFT boundary spacing differs from the exact geometry");
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      if (!boundary.topology().is_periodic(face) ||
          boundary.at(face).kind != PhysicalBoundaryKind::external)
        throw std::invalid_argument("Poisson FFT requires periodic topology on every face");
    }
  }
}

template <int Dim>
POPS_HD std::size_t local_slab_ordinal(const Box<Dim>& valid, const CellIndex<Dim>& cell) {
  std::size_t ordinal = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    ordinal += static_cast<std::size_t>(cell[axis] - valid.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(valid.length(axis));
  }
  return ordinal;
}

template <int Dim>
std::array<int, Dim> fft_cells(const Geometry<Dim>& geometry) {
  std::array<int, Dim> cells{};
  for (int axis = 0; axis < Dim; ++axis)
    cells[axis] = static_cast<int>(geometry.domain().length(axis));
  return cells;
}

template <int Dim>
std::array<double, Dim> fft_lengths(const Geometry<Dim>& geometry) {
  std::array<double, Dim> lengths{};
  for (int axis = 0; axis < Dim; ++axis)
    lengths[axis] = static_cast<double>(geometry.upper()[axis] - geometry.lower()[axis]);
  return lengths;
}

}  // namespace fft_solver_detail

/// One exact Cartesian slab-distributed wrapper around the PoPS device FFT engine.
/// A serial run is the one-rank instance of the same execution trace.
template <int Dim>
class PoissonFFTSolver {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "PoissonFFTSolver supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim>;
  using request_type = EllipticBuildRequest<Dim>;

 private:
  struct PreparedTag {};

  struct PreparedStorage {
    Geometry<Dim> geometry;
    PhysicalBoundaryConditions<Dim> boundary;
    field_type rhs;
    field_type phi;
    field_type trial;
    field_type residual_field;
    typename PoissonFFT<Dim>::device_view fft_rhs;
    typename PoissonFFT<Dim>::device_view fft_phi;
    std::unique_ptr<HaloSchedule<Dim>> halo_schedule;
    EllipticOperatorContract operator_contract;
  };

 public:
  PoissonFFTSolver(request_type request, const ExecutionLane& lane)
      : PoissonFFTSolver(
            prepare_storage_collectively_(prepare_collectively_(std::move(request), lane), lane),
            lane, PreparedTag{}) {}

  PoissonFFTSolver(const PoissonFFTSolver&) = delete;
  PoissonFFTSolver& operator=(const PoissonFFTSolver&) = delete;
  PoissonFFTSolver(PoissonFFTSolver&&) noexcept = default;
  PoissonFFTSolver& operator=(PoissonFFTSolver&&) noexcept = default;
  ~PoissonFFTSolver() noexcept = default;

  static constexpr PoissonFFTCapabilities<Dim> capabilities() noexcept { return {}; }
  static constexpr EllipticOperatorIdentity operator_identity() noexcept {
    return {"pops.elliptic.poisson-fft.discrete-cartesian", 3};
  }
  static EllipticOperatorContract expected_operator_contract(const request_type& request) {
    return make_expected_elliptic_operator_contract(operator_identity(), request,
                                                    fft_solver_detail::options_contract());
  }
  static bool supports(const request_type& request, const ExecutionLane& lane) noexcept {
    try {
      validate_local_(request, lane);
      return true;
    } catch (...) {
      return false;
    }
  }

  field_type& rhs() noexcept { return rhs_; }
  const field_type& rhs() const noexcept { return rhs_; }
  field_type& phi() noexcept { return phi_; }
  const field_type& phi() const noexcept { return phi_; }
  const Geometry<Dim>& geom() const noexcept { return geometry_; }
  const PhysicalBoundaryConditions<Dim>& boundary() const noexcept { return boundary_; }
  static constexpr PoissonFFTSymbol symbol() noexcept {
    return PoissonFFTSymbol::discrete_cartesian;
  }
  Real residual() const noexcept { return last_report_.residual_norm; }
  const SolveReport& last_solve_report() const noexcept { return last_report_; }
  const EllipticOperatorContract& prepared_operator_contract() const noexcept {
    return prepared_operator_contract_;
  }
  bool borrows_execution_lane(const ExecutionLane& lane) const noexcept { return lane_ == &lane; }

  void install_nullspace(FieldNullspacePlan<Dim> plan,
                         PreparedVectorDistribution<Dim> distribution) {
    if (nullspace_workspace_)
      throw std::logic_error("Poisson FFT nullspace authority is already installed");
    if (plan.empty())
      throw std::invalid_argument(
          "periodic Poisson FFT requires an explicit non-empty nullspace plan");
    std::vector<const field_type*> layouts{&rhs_};
    std::vector<PreparedVectorDistribution<Dim>> distributions{std::move(distribution)};
    nullspace_workspace_ = std::make_unique<FieldNullspaceWorkspace<Dim>>(
        std::move(plan), std::move(layouts), std::move(distributions), *lane_);
  }

  SolveReport solve() {
    if (!nullspace_workspace_)
      throw std::logic_error("Poisson FFT solve has no prepared nullspace authority");

    SolveReport report;
    try {
      nullspace_workspace_->require_compatible(rhs_);
    } catch (const FieldNullspaceIncompatibleRhs& error) {
      report.mark_failed(SolveStatus::kIncompatibleRhs, SolveAction::kFailRun, error.what());
      last_report_ = report;
      return last_report_;
    } catch (const FieldNullspaceInvalidEvaluation& error) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun, error.what());
      last_report_ = report;
      return last_report_;
    }

    // Before touching fab(0), make every rank agree that its exact local slab and the staged
    // device views still match the prepared FFT contract.  A rank with no local fab must never
    // strand its peers in the transform exchange.
    std::exception_ptr layout_error;
    Box<Dim> valid{};
    try {
      if (!fft_ || rhs_.local_size() != 1 || phi_.local_size() != 1 || trial_.local_size() != 1 ||
          residual_field_.local_size() != 1)
        throw std::invalid_argument("Poisson FFT solve requires exactly one prepared local slab");
      valid = rhs_.fab(0).box();
      if (phi_.fab(0).box() != valid || trial_.fab(0).box() != valid ||
          residual_field_.fab(0).box() != valid ||
          fft_rhs_.extent(0) != static_cast<std::size_t>(valid.numPts()) ||
          fft_phi_.extent(0) != static_cast<std::size_t>(valid.numPts()))
        throw std::invalid_argument(
            "Poisson FFT solve local field/view layout differs from its slab");
    } catch (...) {
      layout_error = std::current_exception();
    }
    if (all_reduce_max(layout_error ? 1L : 0L, *lane_) != 0) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "poisson_fft_local_slab_layout_mismatch");
      last_report_ = report;
      return last_report_;
    }

    // The brick inverts +laplacian while the public elliptic contract owns A=-laplacian.
    // This sign conversion and both field transfers stay on the selected Kokkos device.
    const auto rhs_view = rhs_.fab(0).view();
    const auto trial_view = trial_.fab(0).view();
    const auto fft_rhs = fft_rhs_;
    const auto fft_phi = fft_phi_;
    if (!execute_solve_stage_collectively_(
            [&] {
              for_each_cell(valid, [=](const CellIndex<Dim>& cell) {
                const std::size_t ordinal = fft_solver_detail::local_slab_ordinal(valid, cell);
                fft_rhs[ordinal] =
                    typename PoissonFFT<Dim>::complex_type(-rhs_view(cell, 0), Real(0));
              });
            },
            report, "poisson_fft_rhs_pack_failed_collectively"))
      return last_report_;
    if (!execute_solve_stage_collectively_([&] { fft_->solve(fft_rhs_, fft_phi_); }, report,
                                           "poisson_fft_transform_failed_collectively"))
      return last_report_;
    if (!execute_solve_stage_collectively_(
            [&] {
              for_each_cell(valid, [=](const CellIndex<Dim>& cell) {
                const std::size_t ordinal = fft_solver_detail::local_slab_ordinal(valid, cell);
                trial_view(cell, 0) = fft_phi[ordinal].real();
              });
            },
            report, "poisson_fft_solution_unpack_failed_collectively"))
      return last_report_;

    try {
      nullspace_workspace_->apply_gauge(trial_);
    } catch (const FieldNullspaceInvalidEvaluation& error) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun, error.what());
      last_report_ = report;
      return last_report_;
    }
    fill_periodic_(trial_);

    report.evaluations = 1;
    Real local_reference_residual = 0;
    Real local_residual = 0;
    if (!execute_solve_stage_collectively_(
            [&] {
              local_reference_residual = norm_inf(rhs_);
              compute_discrete_residual_();
              local_residual = norm_inf(residual_field_);
            },
            report, "poisson_fft_residual_evaluation_failed_collectively"))
      return last_report_;
    report.reference_residual_norm =
        static_cast<Real>(all_reduce_max(static_cast<double>(local_reference_residual), *lane_));
    report.residual_norm =
        static_cast<Real>(all_reduce_max(static_cast<double>(local_residual), *lane_));
    report.rel_residual = report.reference_residual_norm > Real(0)
                              ? report.residual_norm / report.reference_residual_norm
                              : report.residual_norm;
    if (!std::isfinite(static_cast<double>(report.residual_norm)) ||
        !std::isfinite(static_cast<double>(report.rel_residual))) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "poisson_fft_non_finite_residual");
      last_report_ = report;
      return last_report_;
    }
    const Real residual_envelope = fft_solver_detail::kDirectResidualSafetyFactor *
                                   std::numeric_limits<Real>::epsilon() *
                                   std::sqrt(static_cast<Real>(geometry_.domain().numPts())) *
                                   std::max(Real(1), report.reference_residual_norm);
    if (report.residual_norm > residual_envelope) {
      report.mark_failed(SolveStatus::kInadmissibleCandidate, SolveAction::kFailRun,
                         "poisson_fft_discrete_residual_exceeds_roundoff_envelope");
      last_report_ = report;
      return last_report_;
    }

    // The complete valid+halo candidate has passed the exact residual gate.  Swapping owned
    // storage publishes it without a second fallible halo exchange and leaves the previous live
    // field as the next transactional scratch buffer.
    std::swap(phi_, trial_);
    report.mark_solved("poisson_fft_discrete_direct");
    last_report_ = report;
    return last_report_;
  }

 private:
  template <class Operation>
  bool execute_solve_stage_collectively_(Operation&& operation, SolveReport& report,
                                         const char* failure_reason) {
    std::exception_ptr local_error;
    try {
      std::forward<Operation>(operation)();
      ::pops::device_fence();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, *lane_) == 0)
      return true;
    report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun, failure_reason);
    last_report_ = report;
    return false;
  }

  PoissonFFTSolver(PreparedStorage prepared, const ExecutionLane& lane, PreparedTag)
      : geometry_(std::move(prepared.geometry)),
        boundary_(std::move(prepared.boundary)),
        rhs_(std::move(prepared.rhs)),
        phi_(std::move(prepared.phi)),
        trial_(std::move(prepared.trial)),
        residual_field_(std::move(prepared.residual_field)),
        fft_rhs_(std::move(prepared.fft_rhs)),
        fft_phi_(std::move(prepared.fft_phi)),
        halo_schedule_(std::move(prepared.halo_schedule)),
        lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        prepared_operator_contract_(std::move(prepared.operator_contract)) {
    fft_.emplace(fft_solver_detail::fft_cells(geometry_), fft_solver_detail::fft_lengths(geometry_),
                 lane, "pops.poisson-fft/cartesian");
    const bool remote = all_reduce_max(halo_schedule_->has_remote_jobs() ? 1L : 0L, lane) != 0;
    if (remote) {
      std::exception_ptr exchange_error;
      try {
        HaloExchangeContext context{};
        context.context_generation = 1;
        context.schedule_generation = 1;
        halo_exchange_ = std::make_unique<HaloExchange<Dim>>(*halo_schedule_, lane, context);
      } catch (...) {
        exchange_error = std::current_exception();
      }
      if (all_reduce_max(exchange_error ? 1L : 0L, lane) != 0) {
        throw std::runtime_error("Poisson FFT halo-exchange preparation failed collectively");
      }
    }
  }

  static PreparedStorage prepare_storage_collectively_(const request_type& request,
                                                       const ExecutionLane& lane) {
    std::optional<PreparedStorage> prepared;
    std::exception_ptr allocation_error;
    try {
      field_type rhs(request.boxes, request.distribution, request.local_rank, 1,
                     request.rhs_ghosts);
      field_type phi(request.boxes, request.distribution, request.local_rank, 1,
                     request.phi_ghosts);
      field_type trial(request.boxes, request.distribution, request.local_rank, 1,
                       request.phi_ghosts);
      field_type residual_field(request.boxes, request.distribution, request.local_rank, 1,
                                Extent<Dim>{});
      typename PoissonFFT<Dim>::device_view fft_rhs("poisson_fft_rhs",
                                                    local_fft_cell_count_(request, lane));
      typename PoissonFFT<Dim>::device_view fft_phi("poisson_fft_phi",
                                                    local_fft_cell_count_(request, lane));
      auto halo_schedule = std::make_unique<HaloSchedule<Dim>>(prepare_halo_schedule(
          trial, request.geometry.domain(), request.boundary.topology(),
          fft_solver_detail::exact_halo_budget(trial.layout(), request.geometry.domain())));
      prepared.emplace(PreparedStorage{
          request.geometry, request.boundary, std::move(rhs), std::move(phi), std::move(trial),
          std::move(residual_field), std::move(fft_rhs), std::move(fft_phi),
          std::move(halo_schedule), expected_operator_contract(request)});
    } catch (...) {
      allocation_error = std::current_exception();
    }
    if (all_reduce_max(allocation_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && allocation_error)
        std::rethrow_exception(allocation_error);
      throw std::runtime_error("Poisson FFT wrapper allocation failed collectively");
    }
    return std::move(*prepared);
  }

  static std::size_t local_fft_cell_count_(const request_type& request, const ExecutionLane& lane) {
    std::size_t local_cells = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t extent = request.geometry.domain().length(axis);
      const std::size_t local_extent =
          static_cast<std::size_t>(axis == Dim - 1 ? extent / lane.size() : extent);
      local_cells = fft_solver_detail::checked_multiply(
          local_cells, local_extent, "Poisson FFT local slab element count overflow");
    }
    return local_cells;
  }

  static void validate_local_(const request_type& request, const ExecutionLane& lane) {
    const int ranks = lane.size();
    if (ranks < 1 || request.geometry.domain().empty() || request.boxes.empty() ||
        !request.boxes.tiles_exactly(request.geometry.domain(), request.layout_budget) ||
        !request.distribution.matches_layout(request.boxes) ||
        !request.distribution.rank_space().contains(request.local_rank) ||
        request.distribution.rank_space().size() != static_cast<std::size_t>(ranks) ||
        request.distribution.rank_space().linear_rank(request.local_rank) !=
            static_cast<std::size_t>(lane.rank()))
      throw std::invalid_argument("Poisson FFT received an invalid exact-ranked layout request");
    if (request.boxes.size() != static_cast<std::size_t>(ranks))
      throw std::invalid_argument("Poisson FFT requires exactly one slab per communicator rank");
    if (ranks > 1 && request.distribution.replicated())
      throw std::invalid_argument(
          "distributed Poisson FFT slabs require unique owners (no replicated final axis)");
    // Boxes must be unique last-axis slabs. The process rank space may be any
    // 1-D embedding of the same communicator (System uses the first axis).
    int process_axes = 0;
    for (int axis = 0; axis < Dim; ++axis)
      if (request.distribution.rank_space().extent()[axis] != 1)
        ++process_axes;
    if (ranks > 1 && process_axes != 1)
      throw std::invalid_argument(
          "Poisson FFT rank space must be a one-dimensional process grid");

    fft_solver_detail::validate_periodic_boundary(request.geometry, request.boundary);
    for (int axis = 0; axis < Dim; ++axis)
      if (request.rhs_ghosts[axis] != 0 || request.phi_ghosts[axis] != 1)
        throw std::invalid_argument(
            "Poisson FFT requires a ghost-free RHS and exactly one solution ghost");

    std::size_t local_elements = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t extent = request.geometry.domain().length(axis);
      if (extent <= 0 || extent > std::numeric_limits<int>::max())
        throw std::invalid_argument("Poisson FFT extent exceeds Cartesian index range");
      if (axis == Dim - 1 && extent % ranks != 0)
        throw std::invalid_argument(
            "Poisson FFT communicator size must divide the final Cartesian axis "
            "(unique slabs, no replicated Z)");
      const std::size_t local_extent =
          static_cast<std::size_t>(axis == Dim - 1 ? extent / ranks : extent);
      local_elements = fft_solver_detail::checked_multiply(
          local_elements, local_extent, "Poisson FFT local slab element count overflow");
    }
    if (local_elements > static_cast<std::size_t>(std::numeric_limits<int>::max()) /
                             sizeof(typename PoissonFFT<Dim>::complex_type))
      throw std::invalid_argument("Poisson FFT local slab exceeds MPI byte count range");

    const Box<Dim>& domain = request.geometry.domain();
    for (int rank = 0; rank < ranks; ++rank) {
      Box<Dim> expected = domain;
      const int axis = Dim - 1;
      const int local_last = domain.length(axis) / ranks;
      expected.lo[axis] = domain.lo[axis] + rank * local_last;
      expected.hi[axis] = expected.lo[axis] + local_last - 1;
      if (request.boxes[static_cast<std::size_t>(rank)] != expected)
        throw std::invalid_argument(
            "Poisson FFT layout is not the canonical ordered unique-slab layout");
      if (!request.distribution.replicated() &&
          request.distribution.owner(static_cast<std::size_t>(rank)) !=
              request.distribution.rank_space().coordinate(static_cast<std::size_t>(rank)))
        throw std::invalid_argument("Poisson FFT slab owner differs from its ordered rank");
    }
  }

  static request_type prepare_collectively_(request_type request, const ExecutionLane& lane) {
    std::exception_ptr error;
    std::string exact_request;
    try {
      validate_local_(request, lane);
      exact_request.assign(expected_operator_contract(request).exact_fingerprint());
    } catch (...) {
      error = std::current_exception();
    }
    if (all_reduce_max(error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && error)
        std::rethrow_exception(error);
      throw std::runtime_error("Poisson FFT preparation failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"pops.poisson-fft-solver.exact-request@3", exact_request}}, lane))
      throw std::invalid_argument("Poisson FFT solver request differs across communicator ranks");
    return request;
  }

  void fill_periodic_(field_type& field) {
    if (halo_exchange_)
      halo_exchange_->execute(field, *lane_);
    else
      fill_boundary(field, *halo_schedule_);
  }

  void compute_discrete_residual_() {
    elliptic::mg::poisson_residual_valid(trial_, rhs_, geometry_, residual_field_);
  }

  Geometry<Dim> geometry_;
  PhysicalBoundaryConditions<Dim> boundary_;
  field_type rhs_;
  field_type phi_;
  field_type trial_;
  field_type residual_field_;
  typename PoissonFFT<Dim>::device_view fft_rhs_{};
  typename PoissonFFT<Dim>::device_view fft_phi_{};
  std::unique_ptr<HaloSchedule<Dim>> halo_schedule_;
  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  std::unique_ptr<HaloExchange<Dim>> halo_exchange_{};
  std::optional<PoissonFFT<Dim>> fft_{};
  std::unique_ptr<FieldNullspaceWorkspace<Dim>> nullspace_workspace_{};
  EllipticOperatorContract prepared_operator_contract_{};
  SolveReport last_report_{};
};

/// Exact provider declaration for discrete or continuous-symbol FFT construction.
template <int Dim>
class PoissonFFTFactory {
 public:
  using solver_type = PoissonFFTSolver<Dim>;
  using request_type = EllipticBuildRequest<Dim>;

  explicit PoissonFFTFactory(const ExecutionLane& lane,
                             PoissonFFTSymbol symbol = PoissonFFTSymbol::discrete_cartesian)
      : lane_(&lane) {
    if (!PoissonFFTCapabilities<Dim>::supports(symbol))
      throw std::invalid_argument(
          std::string(PoissonFFTCapabilities<Dim>::rejection_reason(symbol)));
  }

  std::string_view collective_contract() const noexcept {
    return "pops.poisson-fft-factory.discrete-cartesian@3";
  }
  EllipticOperatorContract expected_operator_contract(const request_type& request) const {
    return solver_type::expected_operator_contract(request);
  }
  bool supports(const request_type& request) const noexcept {
    return solver_type::supports(request, *lane_);
  }
  EllipticFactoryBuildResult<solver_type> build(request_type request) const noexcept {
    return capture_local_elliptic_factory_build<solver_type>(
        [request = std::move(request), lane = lane_]() mutable {
          return solver_type(std::move(request), *lane);
        });
  }

 private:
  const ExecutionLane* lane_ = nullptr;
};

static_assert(EllipticSolver<PoissonFFTSolver<1>>);
static_assert(EllipticSolver<PoissonFFTSolver<2>>);
static_assert(EllipticSolver<PoissonFFTSolver<3>>);
static_assert(EllipticFactory<PoissonFFTFactory<1>, PoissonFFTSolver<1>>);
static_assert(EllipticFactory<PoissonFFTFactory<2>, PoissonFFTSolver<2>>);
static_assert(EllipticFactory<PoissonFFTFactory<3>, PoissonFFTSolver<3>>);

}  // namespace pops
