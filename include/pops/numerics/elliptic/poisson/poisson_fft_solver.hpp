/// @file
/// @brief Exact-ranked wrapper for the concrete two-dimensional PoPS FFT engine.

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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

enum class PoissonFFTSymbol : unsigned char {
  discrete_five_point,
  continuous_spectral,
};

/// Compile-time capability declaration for the concrete FFT implementation shipped by PoPS.
/// The underlying engine performs FFT-x, a y/x slab transpose, then FFT-y; it is therefore an
/// intrinsically two-dimensional implementation even though periodic Poisson is mathematically ND.
template <int Dim>
struct PoissonFFTCapabilities {
  static_assert(Dim >= 1 && Dim <= 3,
                "PoissonFFTCapabilities only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  static constexpr bool available = Dim == 2;
  static constexpr bool periodic = available;
  static constexpr bool distributed_slabs = available;
  static constexpr bool discrete_symbol = available;
  // The low-level engine can invert this symbol, but does not expose the matching apply operation.
  // Publishing it as an EllipticSolver would therefore require a fabricated residual norm.
  static constexpr bool continuous_spectral_symbol = false;
  static constexpr std::string_view unavailable_reason =
      available ? std::string_view{} : std::string_view{"PoPS concrete FFT engine has rank two"};

  static constexpr bool supports(PoissonFFTSymbol symbol) noexcept {
    return symbol == PoissonFFTSymbol::discrete_five_point && discrete_symbol;
  }

  static constexpr std::string_view rejection_reason(PoissonFFTSymbol symbol) noexcept {
    if (!available)
      return unavailable_reason;
    return supports(symbol)
               ? std::string_view{}
               : std::string_view{"continuous spectral FFT has no exact apply/residual provider"};
  }
};

static_assert(!PoissonFFTCapabilities<1>::available);
static_assert(PoissonFFTCapabilities<2>::available);
static_assert(!PoissonFFTCapabilities<3>::available);

namespace fft_solver_detail {

inline constexpr Real kDirectResidualSafetyFactor = Real(512);

inline std::string options_contract() {
  ExactContractBuilder contract;
  contract.text("pops.elliptic.poisson-fft.options")
      .scalar(std::uint32_t{3})
      .text("discrete-five-point")
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

inline std::size_t host_offset(const Box<2>& grown, int i, int j) {
  const std::size_t width = static_cast<std::size_t>(grown.length(0));
  return static_cast<std::size_t>(j - grown.lo[1]) * width +
         static_cast<std::size_t>(i - grown.lo[0]);
}

inline std::vector<double> pack_local_scalar(const MultiFab<2>& field, Real factor) {
  if (field.local_size() != 1)
    throw std::logic_error("Poisson FFT requires exactly one local slab per rank");
  const auto& fab = field.fab(0);
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  const Box<2>& valid = fab.box();
  const Box<2>& grown = fab.grown_box();
  std::vector<double> packed(static_cast<std::size_t>(valid.numPts()));
  std::size_t cursor = 0;
  for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
    for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
      packed[cursor++] = static_cast<double>(factor * host(host_offset(grown, i, j)));
  return packed;
}

inline void unpack_local_scalar(const std::vector<double>& packed, MultiFab<2>& field) {
  if (field.local_size() != 1)
    throw std::logic_error("Poisson FFT requires exactly one local slab per rank");
  auto& fab = field.fab(0);
  const Box<2>& valid = fab.box();
  if (packed.size() != static_cast<std::size_t>(valid.numPts()))
    throw std::invalid_argument("Poisson FFT returned a slab with the wrong element count");
  auto host = fab.create_host_mirror();
  fab.copy_to_host(host);
  const Box<2>& grown = fab.grown_box();
  std::size_t cursor = 0;
  for (int j = valid.lo[1]; j <= valid.hi[1]; ++j)
    for (int i = valid.lo[0]; i <= valid.hi[0]; ++i)
      host(host_offset(grown, i, j)) = packed[cursor++];
  fab.copy_from_host(host);
}

}  // namespace fft_solver_detail

/// One exact slab-distributed wrapper around the concrete PoPS two-dimensional FFT engine.
/// A serial run is the one-rank instance of the same layout; there is no separate serial,
/// distributed, or remapped compatibility class.
template <int Dim>
  requires(PoissonFFTCapabilities<Dim>::available)
class PoissonFFTSolver {
 public:
  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim>;
  using request_type = EllipticBuildRequest<Dim>;

  explicit PoissonFFTSolver(request_type request)
      : PoissonFFTSolver(prepare_collectively_(std::move(request)), PreparedTag{}) {}

  PoissonFFTSolver(const PoissonFFTSolver&) = delete;
  PoissonFFTSolver& operator=(const PoissonFFTSolver&) = delete;
  PoissonFFTSolver(PoissonFFTSolver&&) noexcept = default;
  PoissonFFTSolver& operator=(PoissonFFTSolver&&) noexcept = default;
  ~PoissonFFTSolver() noexcept = default;

  static constexpr PoissonFFTCapabilities<Dim> capabilities() noexcept { return {}; }
  static constexpr EllipticOperatorIdentity operator_identity() noexcept {
    return {"pops.elliptic.poisson-fft.discrete-rank2", 1};
  }
  static EllipticOperatorContract expected_operator_contract(const request_type& request) {
    return make_expected_elliptic_operator_contract(operator_identity(), request,
                                                    fft_solver_detail::options_contract());
  }
  static bool supports(const request_type& request) noexcept {
    try {
      validate_local_(request);
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
    return PoissonFFTSymbol::discrete_five_point;
  }
  Real residual() const noexcept { return last_report_.residual_norm; }
  const SolveReport& last_solve_report() const noexcept { return last_report_; }
  const EllipticOperatorContract& prepared_operator_contract() const noexcept {
    return prepared_operator_contract_;
  }

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
        std::move(plan), std::move(layouts), std::move(distributions));
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

    // The concrete brick inverts +laplacian.  The public elliptic contract owns
    // A=-laplacian, so the exact adapter is a sign on the scalar payload, not a second operator.
    const std::vector<double> local_rhs = fft_solver_detail::pack_local_scalar(rhs_, Real(-1));
    std::vector<double> local_solution;
    fft_.solve(local_rhs, local_solution);
    fft_solver_detail::unpack_local_scalar(local_solution, trial_);

    try {
      nullspace_workspace_->apply_gauge(trial_);
    } catch (const FieldNullspaceInvalidEvaluation& error) {
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun, error.what());
      last_report_ = report;
      return last_report_;
    }
    fill_periodic_(trial_);

    report.evaluations = 1;
    report.reference_residual_norm =
        static_cast<Real>(all_reduce_max(static_cast<double>(norm_inf(rhs_))));
    compute_discrete_residual_();
    report.residual_norm =
        static_cast<Real>(all_reduce_max(static_cast<double>(norm_inf(residual_field_))));
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
  struct PreparedTag {};

  PoissonFFTSolver(request_type request, PreparedTag)
      : geometry_(request.geometry),
        boundary_(request.boundary),
        rhs_(request.boxes, request.distribution, request.local_rank, 1, request.rhs_ghosts),
        phi_(request.boxes, request.distribution, request.local_rank, 1, request.phi_ghosts),
        trial_(request.boxes, request.distribution, request.local_rank, 1, request.phi_ghosts),
        residual_field_(request.boxes, request.distribution, request.local_rank, 1, Extent<Dim>{}),
        fft_(static_cast<int>(geometry_.domain().length(0)),
             static_cast<int>(geometry_.domain().length(1)),
             geometry_.upper()[0] - geometry_.lower()[0],
             geometry_.upper()[1] - geometry_.lower()[1], false),
        halo_schedule_(std::make_unique<HaloSchedule<Dim>>(prepare_halo_schedule(
            trial_, geometry_.domain(), boundary_.topology(),
            fft_solver_detail::exact_halo_budget(trial_.layout(), geometry_.domain())))) {
    const bool remote = all_reduce_max(halo_schedule_->has_remote_jobs() ? 1L : 0L) != 0;
    if (remote) {
      halo_lane_ = std::make_unique<ExecutionLane>(ExecutionLane::duplicate_world_collectively(
          "pops.poisson-fft.exact-rank2/periodic-halo"));
      HaloExchangeContext context{};
      context.context_generation = 1;
      context.schedule_generation = 1;
      halo_exchange_ = std::make_unique<HaloExchange<Dim>>(*halo_schedule_, *halo_lane_, context);
    }
    prepared_operator_contract_ = expected_operator_contract(request);
  }

  static void validate_local_(const request_type& request) {
    const int ranks = n_ranks();
    if (ranks < 1 || request.geometry.domain().empty() || request.boxes.empty() ||
        !request.boxes.tiles_exactly(request.geometry.domain(), request.layout_budget) ||
        !request.distribution.matches_layout(request.boxes) ||
        !request.distribution.rank_space().contains(request.local_rank) ||
        request.distribution.rank_space().size() != static_cast<std::size_t>(ranks) ||
        request.distribution.rank_space().linear_rank(request.local_rank) !=
            static_cast<std::size_t>(my_rank()))
      throw std::invalid_argument("Poisson FFT received an invalid exact-ranked layout request");
    if (request.boxes.size() != static_cast<std::size_t>(ranks))
      throw std::invalid_argument("Poisson FFT requires exactly one slab per communicator rank");
    if (ranks > 1 && request.distribution.replicated())
      throw std::invalid_argument("distributed Poisson FFT slabs require unique owners");

    fft_solver_detail::validate_periodic_boundary(request.geometry, request.boundary);
    for (int axis = 0; axis < Dim; ++axis)
      if (request.rhs_ghosts[axis] != 0 || request.phi_ghosts[axis] != 1)
        throw std::invalid_argument(
            "Poisson FFT requires a ghost-free RHS and exactly one solution ghost");

    const std::int64_t nx64 = request.geometry.domain().length(0);
    const std::int64_t ny64 = request.geometry.domain().length(1);
    if (nx64 <= 0 || ny64 <= 0 || nx64 > std::numeric_limits<int>::max() ||
        ny64 > std::numeric_limits<int>::max())
      throw std::invalid_argument("Poisson FFT extents exceed the concrete engine index range");
    const int nx = static_cast<int>(nx64);
    const int ny = static_cast<int>(ny64);
    if (nx % ranks != 0 || ny % ranks != 0)
      throw std::invalid_argument(
          "Poisson FFT x and y extents must both be divisible by communicator size");
    const int local_y = ny / ranks;
    const std::size_t local_elements = fft_solver_detail::checked_multiply(
        static_cast<std::size_t>(nx), static_cast<std::size_t>(local_y),
        "Poisson FFT local slab element count overflow");
    if (local_elements > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("Poisson FFT local slab exceeds MPI count range");

    const Box<Dim>& domain = request.geometry.domain();
    for (int rank = 0; rank < ranks; ++rank) {
      Box<Dim> expected = domain;
      expected.lo[1] = domain.lo[1] + rank * local_y;
      expected.hi[1] = expected.lo[1] + local_y - 1;
      if (request.boxes[static_cast<std::size_t>(rank)] != expected)
        throw std::invalid_argument("Poisson FFT layout is not the canonical ordered slab layout");
      if (!request.distribution.replicated() &&
          request.distribution.owner(static_cast<std::size_t>(rank)) !=
              request.distribution.rank_space().coordinate(static_cast<std::size_t>(rank)))
        throw std::invalid_argument("Poisson FFT slab owner differs from its ordered rank");
    }
  }

  static request_type prepare_collectively_(request_type request) {
    std::exception_ptr error;
    try {
      validate_local_(request);
    } catch (...) {
      error = std::current_exception();
    }
    if (all_reduce_max(error ? 1L : 0L) != 0) {
      if (n_ranks() == 1 && error)
        std::rethrow_exception(error);
      throw std::runtime_error("Poisson FFT preparation failed collectively");
    }
    return request;
  }

  void fill_periodic_(field_type& field) {
    if (halo_exchange_)
      halo_exchange_->execute(field, *halo_lane_);
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
  PoissonFFT fft_;
  std::unique_ptr<HaloSchedule<Dim>> halo_schedule_;
  std::unique_ptr<ExecutionLane> halo_lane_{};
  std::unique_ptr<HaloExchange<Dim>> halo_exchange_{};
  std::unique_ptr<FieldNullspaceWorkspace<Dim>> nullspace_workspace_{};
  EllipticOperatorContract prepared_operator_contract_{};
  SolveReport last_report_{};
};

/// Exact provider declaration for discrete or continuous-symbol FFT construction.
template <int Dim>
  requires(PoissonFFTCapabilities<Dim>::available)
class PoissonFFTFactory {
 public:
  using solver_type = PoissonFFTSolver<Dim>;
  using request_type = EllipticBuildRequest<Dim>;

  explicit PoissonFFTFactory(PoissonFFTSymbol symbol = PoissonFFTSymbol::discrete_five_point) {
    if (!PoissonFFTCapabilities<Dim>::supports(symbol))
      throw std::invalid_argument(
          std::string(PoissonFFTCapabilities<Dim>::rejection_reason(symbol)));
  }

  std::string_view collective_contract() const noexcept {
    return "pops.poisson-fft-factory.discrete-rank2@1";
  }
  EllipticOperatorContract expected_operator_contract(const request_type& request) const {
    return solver_type::expected_operator_contract(request);
  }
  bool supports(const request_type& request) const noexcept {
    return solver_type::supports(request);
  }
  EllipticFactoryBuildResult<solver_type> build(request_type request) const noexcept {
    return capture_local_elliptic_factory_build<solver_type>(
        [request = std::move(request)]() mutable { return solver_type(std::move(request)); });
  }
};

static_assert(EllipticSolver<PoissonFFTSolver<2>>);
static_assert(EllipticFactory<PoissonFFTFactory<2>, PoissonFFTSolver<2>>);

}  // namespace pops
