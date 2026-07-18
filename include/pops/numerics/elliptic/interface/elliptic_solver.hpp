#pragma once

/// @file
/// @brief EllipticSolver concept: common contract for elliptic solvers at the MultiFab level
///        (solve D phi = f), so couplers depend on the concept and not on a concrete class.
///
/// Layer: `include/pops/numerics/elliptic/interface`.
/// Role: express the "elliptic solver" dependency through a C++20 concept rather than a
/// hard-coded GeometricMG, which prepares swapping MG for another backend (FFT wrapper, PETSc,
/// Hypre) without touching the coupling logic.
/// Contract: an EllipticSolver exposes rhs() -> MultiFab& (right-hand side f, written before solve),
/// phi() -> MultiFab& (solution read after solve, kept between calls for the warm start),
/// solve() (solves phi from rhs in place), residual() -> Real (residual norm ||D phi - f||),
/// geom() -> const Geometry& (geometry of the solved level).
///
/// Invariants:
/// - the contract is at the MultiFab level: poisson_fft.hpp (slabs + raw vectors) does NOT model it
///   directly; PoissonFFTSolver/DistributedFFTSolver are what wrap it;
/// - phi() is kept between calls (warm start): do NOT assume an implicit reset to zero.

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution_mapping.hpp>
#include <pops/mesh/layout/field_distribution.hpp>
#include <pops/mesh/storage/field_replica_consensus.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/interface/spatial_provider.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

template <class S>
concept EllipticSolver = requires(S s) {
  { s.rhs() } -> std::same_as<MultiFab&>;
  { s.phi() } -> std::same_as<MultiFab&>;
  s.solve();
  { s.residual() } -> std::convertible_to<Real>;
  { s.geom() } -> std::convertible_to<const Geometry&>;
};

/// Backend-neutral construction request. Representation semantics and exact box ownership are
/// properties of the solved field, while solver-specific options stay in the injected factory. A
/// backend may therefore use a constructor, PETSc/Hypre builder, plugin registry or another creation
/// mechanism without changing a coupler.
struct EllipticBuildRequest {
  Geometry geometry;
  BoxArray boxes;
  DistributionMapping mapping;
  BCRec boundary;
  ActiveRegionProvider2D active;
  FieldDistribution distribution;
  int rhs_ghosts{0};
  int phi_ghosts{1};
};

/// Stable semantic identity of one materialized elliptic operator implementation.  The identity
/// selects the numerical operator family; every resolved parameter and physical input is carried by
/// EllipticOperatorContract rather than being hidden in this name.
struct EllipticOperatorIdentity {
  std::string_view name;
  std::uint64_t version = 0;
};

/// Collision-free exact fingerprint of a prepared elliptic operator.
///
/// The fingerprint is a canonical framed byte sequence, not a lossy hash.  It authenticates the
/// backend identity, the complete physical/materialization state and the backend options that can
/// change the prepared map.  Factories produce the expected contract from EllipticBuildRequest;
/// solvers independently produce the actual contract from their stored state and allocated fields.
/// Comparing both therefore detects a factory that accepts, but silently ignores, a boundary,
/// active-region provider, field distribution, layout, geometry, ghost contract or backend option.
class EllipticOperatorContract {
 public:
  EllipticOperatorContract() = default;

  [[nodiscard]] static EllipticOperatorContract make(EllipticOperatorIdentity identity,
                                                     std::string exact_materialization,
                                                     std::string exact_backend_options) {
    if (identity.name.empty() || identity.version == 0)
      throw std::invalid_argument(
          "elliptic operator identity requires a non-empty name and positive version");
    ExactContractBuilder contract;
    contract.text("pops.elliptic.materialized-operator")
        .scalar(std::uint32_t{1})
        .text(identity.name)
        .scalar(identity.version)
        .bytes(exact_materialization)
        .bytes(exact_backend_options);
    EllipticOperatorContract result;
    result.exact_fingerprint_ = std::move(contract).release();
    return result;
  }

  [[nodiscard]] bool valid() const noexcept { return !exact_fingerprint_.empty(); }
  [[nodiscard]] std::string_view exact_fingerprint() const noexcept { return exact_fingerprint_; }

 private:
  std::string exact_fingerprint_;
};

template <class Solver>
  requires std::is_nothrow_move_constructible_v<Solver> && std::is_nothrow_destructible_v<Solver>
struct EllipticFactoryBuildResult {
  std::optional<Solver> solver;
  std::exception_ptr error;
};

/// Capture every local factory failure without letting one rank skip the common failure reduction.
/// A backend whose construction itself uses MPI must perform its own internal failure agreement and
/// return from `build()` on every rank; no exception may escape the factory protocol.
template <class Solver, class Builder>
  requires std::is_nothrow_move_constructible_v<Solver> && std::is_nothrow_destructible_v<Solver> &&
           std::invocable<Builder> && std::same_as<std::invoke_result_t<Builder>, Solver>
EllipticFactoryBuildResult<Solver> capture_local_elliptic_factory_build(
    Builder&& builder) noexcept {
  EllipticFactoryBuildResult<Solver> result;
  try {
    result.solver.emplace(std::invoke(std::forward<Builder>(builder)));
  } catch (...) {
    result.error = std::current_exception();
  }
  return result;
}

namespace detail {

inline bool elliptic_build_request_is_valid(const EllipticBuildRequest& request, int rank,
                                            int ranks) noexcept {
  if (!field_distribution_is_valid(request.distribution) || request.boxes.size() <= 0 ||
      request.mapping.size() != request.boxes.size() || rank < 0 || ranks < 1 || rank >= ranks ||
      request.rhs_ghosts != 0 || request.phi_ghosts < 0)
    return false;
  for (const int owner : request.mapping.ranks()) {
    if (request.distribution == FieldDistribution::Replicated) {
      if (owner != rank)
        return false;
    } else if (owner < 0 || owner >= ranks) {
      return false;
    }
  }
  if (request.geometry.domain.empty() || !std::isfinite(request.geometry.xlo) ||
      !std::isfinite(request.geometry.xhi) || !std::isfinite(request.geometry.ylo) ||
      !std::isfinite(request.geometry.yhi) || request.geometry.xhi <= request.geometry.xlo ||
      request.geometry.yhi <= request.geometry.ylo)
    return false;
  if (!request.boxes.tiles_exactly(request.geometry.domain))
    return false;
  const auto valid_boundary_type = [](BCType type) {
    switch (type) {
      case BCType::Periodic:
      case BCType::Foextrap:
      case BCType::Dirichlet:
      case BCType::Robin:
      case BCType::External:
        return true;
    }
    return false;
  };
  const BCRec& boundary = request.boundary;
  if (!valid_boundary_type(boundary.xlo) || !valid_boundary_type(boundary.xhi) ||
      !valid_boundary_type(boundary.ylo) || !valid_boundary_type(boundary.yhi))
    return false;
  if ((boundary.xlo == BCType::Periodic) != (boundary.xhi == BCType::Periodic) ||
      (boundary.ylo == BCType::Periodic) != (boundary.yhi == BCType::Periodic))
    return false;
  const Real values[] = {boundary.xlo_val,   boundary.xhi_val,   boundary.ylo_val,
                         boundary.yhi_val,   boundary.xlo_alpha, boundary.xlo_beta,
                         boundary.xhi_alpha, boundary.xhi_beta,  boundary.ylo_alpha,
                         boundary.ylo_beta,  boundary.yhi_alpha, boundary.yhi_beta};
  for (const Real value : values)
    if (!std::isfinite(value))
      return false;
  const auto valid_robin = [](BCType type, Real alpha, Real beta) {
    return type != BCType::Robin || alpha != Real(0) || beta != Real(0);
  };
  if (!valid_robin(boundary.xlo, boundary.xlo_alpha, boundary.xlo_beta) ||
      !valid_robin(boundary.xhi, boundary.xhi_alpha, boundary.xhi_beta) ||
      !valid_robin(boundary.ylo, boundary.ylo_alpha, boundary.ylo_beta) ||
      !valid_robin(boundary.yhi, boundary.yhi_alpha, boundary.yhi_beta))
    return false;
  return true;
}

inline void append_elliptic_geometry_contract(ExactContractBuilder& contract,
                                              const Geometry& geometry) {
  contract.text("geometry")
      .scalar(geometry.domain.lo[0])
      .scalar(geometry.domain.lo[1])
      .scalar(geometry.domain.hi[0])
      .scalar(geometry.domain.hi[1])
      .scalar(geometry.xlo)
      .scalar(geometry.xhi)
      .scalar(geometry.ylo)
      .scalar(geometry.yhi);
}

inline void append_elliptic_boundary_contract(ExactContractBuilder& contract,
                                              const BCRec& boundary) {
  contract.text("boundary")
      .scalar(boundary.xlo)
      .scalar(boundary.xhi)
      .scalar(boundary.ylo)
      .scalar(boundary.yhi)
      .scalar(boundary.xlo_val)
      .scalar(boundary.xhi_val)
      .scalar(boundary.ylo_val)
      .scalar(boundary.yhi_val)
      .scalar(boundary.xlo_alpha)
      .scalar(boundary.xlo_beta)
      .scalar(boundary.xhi_alpha)
      .scalar(boundary.xhi_beta)
      .scalar(boundary.ylo_alpha)
      .scalar(boundary.ylo_beta)
      .scalar(boundary.yhi_alpha)
      .scalar(boundary.yhi_beta)
      .scalar(boundary.dx)
      .scalar(boundary.dy);
}

inline void append_elliptic_field_layout_contract(ExactContractBuilder& contract,
                                                  std::string_view role, const BoxArray& boxes,
                                                  const DistributionMapping& mapping,
                                                  int components, int ghosts,
                                                  FieldDistribution distribution) {
  std::vector<int> canonical_owners = mapping.ranks();
  if (distribution == FieldDistribution::Replicated)
    std::fill(canonical_owners.begin(), canonical_owners.end(), 0);
  contract.text(role)
      .scalar(components)
      .scalar(ghosts)
      .sequence(boxes.boxes(),
                [](ExactContractBuilder& element, const Box2D& box) {
                  element.scalar(box.lo[0]).scalar(box.lo[1]).scalar(box.hi[0]).scalar(box.hi[1]);
                })
      .sequence(canonical_owners);
}

inline std::string elliptic_materialization_contract(
    const Geometry& geometry, const BCRec& boundary, const ActiveRegionProvider2D& active,
    FieldDistribution distribution, const BoxArray& rhs_boxes,
    const DistributionMapping& rhs_mapping, int rhs_components, int rhs_ghosts,
    const BoxArray& phi_boxes, const DistributionMapping& phi_mapping, int phi_components,
    int phi_ghosts) {
  ExactContractBuilder contract;
  contract.text("pops.elliptic.materialization").scalar(std::uint32_t{1});
  append_elliptic_geometry_contract(contract, geometry);
  append_elliptic_boundary_contract(contract, boundary);
  contract.text("active-region").optional_collective_contract(active);
  contract.text("field-distribution").scalar(distribution);
  append_elliptic_field_layout_contract(contract, "rhs-layout", rhs_boxes, rhs_mapping,
                                        rhs_components, rhs_ghosts, distribution);
  append_elliptic_field_layout_contract(contract, "phi-layout", phi_boxes, phi_mapping,
                                        phi_components, phi_ghosts, distribution);
  return std::move(contract).release();
}

/// Exact communicator-wide identity of every construction input. Replicated owners are normalized
/// because every rank intentionally names itself as owner of every global box.
inline std::string elliptic_build_request_contract(const EllipticBuildRequest& request) {
  return elliptic_materialization_contract(request.geometry, request.boundary, request.active,
                                           request.distribution, request.boxes, request.mapping, 1,
                                           request.rhs_ghosts, request.boxes, request.mapping, 1,
                                           request.phi_ghosts);
}

inline bool elliptic_geometry_exactly_matches(const Geometry& left, const Geometry& right) {
  ExactContractBuilder left_contract;
  ExactContractBuilder right_contract;
  append_elliptic_geometry_contract(left_contract, left);
  append_elliptic_geometry_contract(right_contract, right);
  return left_contract.view() == right_contract.view();
}

}  // namespace detail

/// Generic factory-side construction of the expected exact operator contract.
inline EllipticOperatorContract make_expected_elliptic_operator_contract(
    EllipticOperatorIdentity identity, const EllipticBuildRequest& request,
    std::string exact_backend_options = {}) {
  return EllipticOperatorContract::make(identity, detail::elliptic_build_request_contract(request),
                                        std::move(exact_backend_options));
}

/// Generic solver-side construction of the actual exact operator contract.  Every argument is read
/// from state owned by the materialized solver; passing the original request here would defeat the
/// post-build check and is deliberately unnecessary.
inline EllipticOperatorContract make_materialized_elliptic_operator_contract(
    EllipticOperatorIdentity identity, const Geometry& geometry, const BCRec& boundary,
    const ActiveRegionProvider2D& active, FieldDistribution distribution, const MultiFab& rhs,
    const MultiFab& phi, std::string exact_backend_options = {}) {
  return EllipticOperatorContract::make(
      identity,
      detail::elliptic_materialization_contract(
          geometry, boundary, active, distribution, rhs.box_array(), rhs.dmap(), rhs.ncomp(),
          rhs.n_grow(), phi.box_array(), phi.dmap(), phi.ncomp(), phi.n_grow()),
      std::move(exact_backend_options));
}

template <class Solver>
struct DefaultEllipticFactory {
  std::string contract{"pops.elliptic-factory.default-constructor@1"};

  [[nodiscard]] std::string_view collective_contract() const noexcept { return contract; }

  [[nodiscard]] EllipticOperatorContract expected_operator_contract(
      const EllipticBuildRequest& request) const
    requires requires {
      { Solver::expected_operator_contract(request) } -> std::same_as<EllipticOperatorContract>;
    }
  {
    return Solver::expected_operator_contract(request);
  }

  [[nodiscard]] FieldDistribution materialized_distribution(
      const EllipticBuildRequest& request) const noexcept {
    return request.distribution;
  }

  [[nodiscard]] bool supports(const EllipticBuildRequest&) const noexcept { return true; }

  EllipticFactoryBuildResult<Solver> build(EllipticBuildRequest request) const noexcept
    requires std::constructible_from<Solver, const Geometry&, const BoxArray&,
                                     const DistributionMapping&, const BCRec&,
                                     ActiveRegionProvider2D, FieldDistribution>
  {
    return capture_local_elliptic_factory_build<Solver>([request = std::move(request)]() mutable {
      return Solver(request.geometry, request.boxes, request.mapping, request.boundary,
                    std::move(request.active), request.distribution);
    });
  }
};

template <class Factory, class Solver>
concept EllipticFactory =
    EllipticSolver<Solver> && std::is_nothrow_move_constructible_v<Solver> &&
    std::is_nothrow_destructible_v<Solver> && requires(const Solver& solver) {
      { solver.field_distribution() } noexcept -> std::same_as<FieldDistribution>;
      {
        solver.prepared_operator_contract()
      } noexcept -> std::same_as<const EllipticOperatorContract&>;
    } && requires(const Factory& declaration, Factory& factory, EllipticBuildRequest request) {
      { declaration.collective_contract() } noexcept -> std::same_as<std::string_view>;
      { declaration.expected_operator_contract(request) } -> std::same_as<EllipticOperatorContract>;
      {
        declaration.materialized_distribution(request)
      } noexcept -> std::same_as<FieldDistribution>;
      { declaration.supports(request) } noexcept -> std::same_as<bool>;
      {
        factory.build(std::move(request))
      } noexcept -> std::same_as<EllipticFactoryBuildResult<Solver>>;
    };

template <EllipticSolver Solver, class Factory>
  requires EllipticFactory<std::remove_cvref_t<Factory>, Solver>
Solver make_elliptic_solver(EllipticBuildRequest request, Factory&& factory,
                            const ExecutionLane& lane) {
  const int rank = lane.rank();
  const int ranks = lane.size();
  const long raw_distribution = static_cast<long>(request.distribution);
  const long minimum_distribution = all_reduce_min(raw_distribution, lane);
  const long maximum_distribution = all_reduce_max(raw_distribution, lane);
  const long invalid_request =
      all_reduce_max(detail::elliptic_build_request_is_valid(request, rank, ranks) ? 0L : 1L, lane);
  if (minimum_distribution != maximum_distribution)
    throw std::invalid_argument(
        "elliptic solver field distribution differs between communicator ranks");
  if (invalid_request != 0)
    throw std::invalid_argument("elliptic solver received an invalid construction request");

  // BC metric is derived, not authored. Every backend receives and authenticates the same canonical
  // spacing, so semantically identical requests cannot disagree because one caller left defaults.
  request.boundary.dx = request.geometry.dx();
  request.boundary.dy = request.geometry.dy();
  std::string request_contract;
  long request_contract_failure_local = 0;
  try {
    request_contract = detail::elliptic_build_request_contract(request);
  } catch (...) {
    request_contract_failure_local = 1;
  }
  if (all_reduce_max(request_contract_failure_local, lane) != 0)
    throw std::runtime_error(
        "elliptic construction-request contract failed on at least one communicator rank");
  const bool request_agrees = all_ranks_agree_exact_ordered_byte_pairs(
      {{std::string_view("elliptic-build-request"), std::string_view(request_contract)}}, lane);
  if (!request_agrees)
    throw std::invalid_argument(
        "elliptic solver construction request differs between communicator ranks");
  const FieldDistribution requested_distribution = request.distribution;
  std::vector<Box2D> requested_boxes;
  std::vector<int> requested_owners;
  Geometry requested_geometry{};
  int requested_rhs_ghosts = 0;
  int requested_phi_ghosts = 0;
  long request_capture_failure_local = 0;
  try {
    requested_boxes = request.boxes.boxes();
    requested_owners = request.mapping.ranks();
    requested_geometry = request.geometry;
    requested_rhs_ghosts = request.rhs_ghosts;
    requested_phi_ghosts = request.phi_ghosts;
  } catch (...) {
    request_capture_failure_local = 1;
  }
  if (all_reduce_max(request_capture_failure_local, lane) != 0)
    throw std::runtime_error(
        "elliptic construction-request capture failed on at least one communicator rank");
  bool local_declaration_failed = false;
  std::string factory_contract;
  EllipticOperatorContract expected_operator_contract;
  FieldDistribution materialized_distribution = FieldDistribution::Distributed;
  bool factory_supported = false;
  try {
    factory_contract = std::as_const(factory).collective_contract();
    expected_operator_contract = std::as_const(factory).expected_operator_contract(request);
    materialized_distribution = std::as_const(factory).materialized_distribution(request);
    factory_supported = std::as_const(factory).supports(request);
  } catch (...) {
    local_declaration_failed = true;
  }
  if (all_reduce_max(local_declaration_failed ? 1L : 0L, lane) != 0)
    throw std::runtime_error("elliptic factory declaration failed on at least one rank");
  const bool factory_contract_valid =
      !factory_contract.empty() && expected_operator_contract.valid();
  const bool factory_agrees = all_ranks_agree_exact_ordered_byte_pairs(
      {{std::string_view("elliptic-factory"), std::string_view(factory_contract)},
       {std::string_view("elliptic-expected-operator"),
        expected_operator_contract.exact_fingerprint()}},
      lane);
  const bool semantic_mismatch = !factory_supported || !factory_contract_valid ||
                                 !field_distribution_is_valid(materialized_distribution) ||
                                 materialized_distribution != requested_distribution;
  if (all_reduce_max(semantic_mismatch ? 1L : 0L, lane) != 0)
    throw std::invalid_argument(
        "elliptic factory cannot materialize the exact construction request");
  if (!factory_agrees)
    throw std::invalid_argument(
        "elliptic factory implementation or options differ between communicator ranks");
  EllipticFactoryBuildResult<Solver> build = factory.build(std::move(request));
  const bool local_build_failed = build.error != nullptr || !build.solver.has_value();
  if (all_reduce_max(local_build_failed ? 1L : 0L, lane) != 0)
    throw std::runtime_error("elliptic factory construction failed on at least one rank");
  Solver& solver = *build.solver;

  // Inspect the complete local result before entering another collective. A third-party backend is
  // allowed to throw from ordinary accessors, but one rank must never escape while its peers enter
  // the post-build agreement.
  bool local_inspection_failed = false;
  bool local_materialization_mismatch = false;
  std::string actual_operator_contract;
  std::string rhs_layout_contract;
  std::string phi_layout_contract;
  try {
    MultiFab& rhs = solver.rhs();
    MultiFab& phi = solver.phi();
    const FieldDistribution actual_distribution = std::as_const(solver).field_distribution();
    actual_operator_contract =
        std::as_const(solver).prepared_operator_contract().exact_fingerprint();
    const auto field_mismatch = [&](const MultiFab& field, int expected_ghosts) {
      const bool distribution_layout_matches =
          requested_distribution == FieldDistribution::Distributed ||
          (requested_distribution == FieldDistribution::Replicated &&
           std::all_of(field.dmap().ranks().begin(), field.dmap().ranks().end(),
                       [rank](int owner) { return owner == rank; }));
      return field.box_array().boxes() != requested_boxes ||
             field.dmap().ranks() != requested_owners || field.ncomp() != 1 ||
             field.n_grow() != expected_ghosts || !distribution_layout_matches;
    };
    local_materialization_mismatch =
        actual_distribution != requested_distribution || actual_operator_contract.empty() ||
        actual_operator_contract != expected_operator_contract.exact_fingerprint() ||
        field_mismatch(rhs, requested_rhs_ghosts) || field_mismatch(phi, requested_phi_ghosts) ||
        rhs.shares_storage_with(phi) ||
        !detail::elliptic_geometry_exactly_matches(solver.geom(), requested_geometry);
    rhs_layout_contract = detail::field_distribution_layout_contract(rhs, requested_distribution);
    phi_layout_contract = detail::field_distribution_layout_contract(phi, requested_distribution);
  } catch (...) {
    local_inspection_failed = true;
  }
  if (all_reduce_max(local_inspection_failed ? 1L : 0L, lane) != 0)
    throw std::runtime_error("elliptic backend inspection failed on at least one rank");
  const bool backend_agrees = all_ranks_agree_exact_ordered_byte_pairs(
      {{std::string_view("elliptic-actual-operator"), std::string_view(actual_operator_contract)},
       {std::string_view("elliptic-rhs-layout"), std::string_view(rhs_layout_contract)},
       {std::string_view("elliptic-phi-layout"), std::string_view(phi_layout_contract)}},
      lane);
  if (all_reduce_max(local_materialization_mismatch ? 1L : 0L, lane) != 0)
    throw std::invalid_argument(
        "elliptic backend did not materialize the requested operator and field contract");
  if (!backend_agrees)
    throw std::invalid_argument(
        "elliptic backend operator or field contract differs between communicator ranks");
  return std::move(solver);
}

/// Sequential/control-path compatibility wrapper. Prepared runtime sessions pass their private
/// ExecutionLane explicitly so independently ordered solver construction never shares WORLD.
template <EllipticSolver Solver, class Factory>
  requires EllipticFactory<std::remove_cvref_t<Factory>, Solver>
Solver make_elliptic_solver(EllipticBuildRequest request, Factory&& factory) {
  const ExecutionLane lane = ExecutionLane::world();
  return make_elliptic_solver<Solver>(std::move(request), std::forward<Factory>(factory), lane);
}

}  // namespace pops
