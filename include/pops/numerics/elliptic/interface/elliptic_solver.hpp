/// @file
/// @brief Exact compile-time-ranked public contract for owning elliptic solver backends.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace pops {

/// An owning elliptic backend publishes one immutable spatial specialization and distinct scalar
/// RHS/solution fields.  The rank is a type property; the concept has no dynamic dimension route.
template <class Solver>
concept EllipticSolver = requires(Solver solver, const Solver constant_solver) {
  { Solver::dimension } -> std::convertible_to<int>;
  requires(Solver::dimension >= 1 && Solver::dimension <= 3);
  typename Solver::field_type;
  requires std::same_as<typename Solver::field_type::box_type, Box<Solver::dimension>>;
  { solver.rhs() } -> std::same_as<typename Solver::field_type&>;
  { solver.phi() } -> std::same_as<typename Solver::field_type&>;
  solver.solve();
  { constant_solver.residual() } -> std::convertible_to<Real>;
  { constant_solver.geom() } -> std::same_as<const Geometry<Solver::dimension>&>;
};

/// Complete backend-neutral construction request.  Python selects Dim once; every downstream
/// layout, ownership coordinate, ghost extent, boundary law, and field allocation retains it.
template <int Dim>
struct EllipticBuildRequest {
  static_assert(Dim >= 1 && Dim <= 3, "EllipticBuildRequest only supports dimensions 1, 2, and 3");

  Geometry<Dim> geometry;
  mesh::BoxArray<Dim> boxes;
  mesh::Distribution<Dim> distribution;
  Index<Dim> local_rank{};
  PhysicalBoundaryConditions<Dim> boundary;
  Extent<Dim> rhs_ghosts{};
  Extent<Dim> phi_ghosts{};
  mesh::BoxArrayValidationBudget layout_budget{};
};

struct EllipticOperatorIdentity {
  std::string_view name;
  std::uint64_t version = 0;
};

/// Collision-free canonical identity of a materialized elliptic operator.
class EllipticOperatorContract {
 public:
  EllipticOperatorContract() = default;

  static EllipticOperatorContract make(EllipticOperatorIdentity identity,
                                       std::string exact_materialization,
                                       std::string exact_backend_options) {
    if (identity.name.empty() || identity.version == 0)
      throw std::invalid_argument(
          "elliptic operator identity requires a non-empty name and positive version");
    ExactContractBuilder contract;
    contract.text("pops.elliptic.materialized-operator")
        .scalar(std::uint32_t{2})
        .text(identity.name)
        .scalar(identity.version)
        .bytes(exact_materialization)
        .bytes(exact_backend_options);
    EllipticOperatorContract result;
    result.exact_fingerprint_ = std::move(contract).release();
    return result;
  }

  bool valid() const noexcept { return !exact_fingerprint_.empty(); }
  std::string_view exact_fingerprint() const noexcept { return exact_fingerprint_; }

 private:
  std::string exact_fingerprint_;
};

template <class Solver>
  requires std::is_nothrow_move_constructible_v<Solver> && std::is_nothrow_destructible_v<Solver>
struct EllipticFactoryBuildResult {
  std::optional<Solver> solver;
  std::exception_ptr error;
};

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

namespace elliptic_contract_detail {

template <int Dim>
void append_index(ExactContractBuilder& contract, const Index<Dim>& index) {
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(index[axis]);
}

template <int Dim>
void append_extent(ExactContractBuilder& contract, const Extent<Dim>& extent) {
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(extent[axis]);
}

template <int Dim>
void append_box(ExactContractBuilder& contract, const Box<Dim>& box) {
  append_index(contract, box.lo);
  append_index(contract, box.hi);
}

template <int Dim>
void append_geometry(ExactContractBuilder& contract, const Geometry<Dim>& geometry) {
  contract.text("geometry");
  append_box(contract, geometry.domain());
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(geometry.lower()[axis]).scalar(geometry.upper()[axis]);
}

template <int Dim>
void append_boundary(ExactContractBuilder& contract,
                     const PhysicalBoundaryConditions<Dim>& boundary) {
  contract.text("boundary");
  for (int axis = 0; axis < Dim; ++axis) {
    contract.scalar(boundary.spacing()[axis]);
    for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
      const Face<Dim> face{axis, side};
      const auto& topology = boundary.topology().at(face);
      const auto& law = boundary.at(face);
      contract.scalar(topology.kind)
          .scalar(topology.partner.axis)
          .scalar(topology.partner.side)
          .scalar(law.kind)
          .scalar(law.value)
          .scalar(law.alpha)
          .scalar(law.beta);
    }
  }
}

template <int Dim>
void append_layout(ExactContractBuilder& contract, const mesh::BoxArray<Dim>& boxes) {
  contract.sequence(boxes.boxes(), [](ExactContractBuilder& element, const Box<Dim>& box) {
    append_box(element, box);
  });
}

template <int Dim>
void append_distribution(ExactContractBuilder& contract,
                         const mesh::Distribution<Dim>& distribution) {
  contract.scalar(distribution.mode());
  append_index(contract, distribution.rank_space().origin());
  append_extent(contract, distribution.rank_space().extent());
  contract.sequence(
      distribution.owners(),
      [](ExactContractBuilder& element, const Index<Dim>& owner) { append_index(element, owner); });
}

template <int Dim>
void append_field_layout(ExactContractBuilder& contract, std::string_view role,
                         const mesh::BoxArray<Dim>& boxes,
                         const mesh::Distribution<Dim>& distribution, int components,
                         const Extent<Dim>& ghosts) {
  contract.text(role).scalar(components);
  append_extent(contract, ghosts);
  append_layout(contract, boxes);
  append_distribution(contract, distribution);
}

template <int Dim>
std::string materialization_contract(const Geometry<Dim>& geometry,
                                     const PhysicalBoundaryConditions<Dim>& boundary,
                                     const mesh::BoxArray<Dim>& rhs_boxes,
                                     const mesh::Distribution<Dim>& rhs_distribution,
                                     int rhs_components, const Extent<Dim>& rhs_ghosts,
                                     const mesh::BoxArray<Dim>& phi_boxes,
                                     const mesh::Distribution<Dim>& phi_distribution,
                                     int phi_components, const Extent<Dim>& phi_ghosts) {
  ExactContractBuilder contract;
  contract.text("pops.elliptic.materialization").scalar(std::uint32_t{2});
  append_geometry(contract, geometry);
  append_boundary(contract, boundary);
  append_field_layout(contract, "rhs-layout", rhs_boxes, rhs_distribution, rhs_components,
                      rhs_ghosts);
  append_field_layout(contract, "phi-layout", phi_boxes, phi_distribution, phi_components,
                      phi_ghosts);
  return std::move(contract).release();
}

template <int Dim>
std::string build_request_contract(const EllipticBuildRequest<Dim>& request) {
  ExactContractBuilder contract;
  contract
      .bytes(materialization_contract(request.geometry, request.boundary, request.boxes,
                                      request.distribution, 1, request.rhs_ghosts, request.boxes,
                                      request.distribution, 1, request.phi_ghosts))
      .text("layout-validation-budget")
      .scalar(request.layout_budget.boxes)
      .scalar(request.layout_budget.overlap_pairs);
  return std::move(contract).release();
}

template <int Dim>
bool request_is_valid(const EllipticBuildRequest<Dim>& request, const ExecutionLane& lane) {
  try {
    if (request.geometry.domain().empty() || request.boxes.empty() ||
        !request.boxes.tiles_exactly(request.geometry.domain(), request.layout_budget) ||
        !request.distribution.matches_layout(request.boxes) ||
        !request.distribution.rank_space().contains(request.local_rank) ||
        request.distribution.rank_space().size() != static_cast<std::size_t>(lane.size()) ||
        request.distribution.rank_space().linear_rank(request.local_rank) !=
            static_cast<std::size_t>(lane.rank()))
      return false;
    for (int axis = 0; axis < Dim; ++axis)
      if (request.rhs_ghosts[axis] < 0 || request.phi_ghosts[axis] < 0)
        return false;
    return true;
  } catch (...) {
    return false;
  }
}

template <int Dim, class MemorySpace>
std::string field_layout_contract(const MultiFab<Dim, MemorySpace>& field) {
  ExactContractBuilder contract;
  append_field_layout(contract, "field-layout", field.layout(), field.distribution(), field.ncomp(),
                      field.ghosts());
  return std::move(contract).release();
}

}  // namespace elliptic_contract_detail

template <int Dim>
EllipticOperatorContract make_expected_elliptic_operator_contract(
    EllipticOperatorIdentity identity, const EllipticBuildRequest<Dim>& request,
    std::string exact_backend_options = {}) {
  return EllipticOperatorContract::make(identity,
                                        elliptic_contract_detail::build_request_contract(request),
                                        std::move(exact_backend_options));
}

template <int Dim, class MemorySpace>
EllipticOperatorContract make_materialized_elliptic_operator_contract(
    EllipticOperatorIdentity identity, const Geometry<Dim>& geometry,
    const PhysicalBoundaryConditions<Dim>& boundary, const MultiFab<Dim, MemorySpace>& rhs,
    const MultiFab<Dim, MemorySpace>& phi, std::string exact_backend_options = {}) {
  return EllipticOperatorContract::make(
      identity,
      elliptic_contract_detail::materialization_contract(
          geometry, boundary, rhs.layout(), rhs.distribution(), rhs.ncomp(), rhs.ghosts(),
          phi.layout(), phi.distribution(), phi.ncomp(), phi.ghosts()),
      std::move(exact_backend_options));
}

template <class Solver>
  requires EllipticSolver<Solver>
struct DefaultEllipticFactory {
  static constexpr int dimension = Solver::dimension;
  using request_type = EllipticBuildRequest<dimension>;

  std::string contract{"pops.elliptic-factory.exact-ranked-constructor@2"};

  std::string_view collective_contract() const noexcept { return contract; }

  EllipticOperatorContract expected_operator_contract(const request_type& request) const
    requires requires {
      { Solver::expected_operator_contract(request) } -> std::same_as<EllipticOperatorContract>;
    }
  {
    return Solver::expected_operator_contract(request);
  }

  bool supports(const request_type&) const noexcept {
    return std::constructible_from<Solver, request_type>;
  }

  EllipticFactoryBuildResult<Solver> build(request_type request) const noexcept
    requires std::constructible_from<Solver, request_type>
  {
    return capture_local_elliptic_factory_build<Solver>(
        [request = std::move(request)]() mutable { return Solver(std::move(request)); });
  }
};

template <class Factory, class Solver>
concept EllipticFactory = EllipticSolver<Solver> && std::is_nothrow_move_constructible_v<Solver> &&
                          std::is_nothrow_destructible_v<Solver> &&
                          requires(const Solver& solver) {
                            {
                              solver.prepared_operator_contract()
                            } noexcept -> std::same_as<const EllipticOperatorContract&>;
                          } &&
                          requires(const Factory& declaration, Factory& factory,
                                   EllipticBuildRequest<Solver::dimension> request) {
                            {
                              declaration.collective_contract()
                            } noexcept -> std::same_as<std::string_view>;
                            {
                              declaration.expected_operator_contract(request)
                            } -> std::same_as<EllipticOperatorContract>;
                            { declaration.supports(request) } noexcept -> std::same_as<bool>;
                            {
                              factory.build(std::move(request))
                            } noexcept -> std::same_as<EllipticFactoryBuildResult<Solver>>;
                          };

template <EllipticSolver Solver, class Factory>
  requires EllipticFactory<std::remove_cvref_t<Factory>, Solver>
Solver make_elliptic_solver(EllipticBuildRequest<Solver::dimension> request, Factory&& factory,
                            const ExecutionLane& lane) {
  const long invalid_request =
      all_reduce_max(elliptic_contract_detail::request_is_valid(request, lane) ? 0L : 1L, lane);
  if (invalid_request != 0)
    throw std::invalid_argument("elliptic solver received an invalid exact-ranked request");

  std::string request_contract;
  long request_contract_failure = 0;
  try {
    request_contract = elliptic_contract_detail::build_request_contract(request);
  } catch (...) {
    request_contract_failure = 1;
  }
  if (all_reduce_max(request_contract_failure, lane) != 0)
    throw std::runtime_error(
        "elliptic construction-request contract failed on at least one execution-lane rank");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("elliptic-build-request"), std::string_view(request_contract)}}, lane))
    throw std::invalid_argument(
        "elliptic solver construction request differs between execution-lane ranks");

  const auto requested_geometry = request.geometry;
  const auto requested_boxes = request.boxes;
  const auto requested_distribution = request.distribution;
  const auto requested_local_rank = request.local_rank;
  const auto requested_rhs_ghosts = request.rhs_ghosts;
  const auto requested_phi_ghosts = request.phi_ghosts;

  bool declaration_failed = false;
  bool supported = false;
  std::string factory_contract;
  EllipticOperatorContract expected;
  try {
    factory_contract = std::as_const(factory).collective_contract();
    expected = std::as_const(factory).expected_operator_contract(request);
    supported = std::as_const(factory).supports(request);
  } catch (...) {
    declaration_failed = true;
  }
  if (all_reduce_max(declaration_failed ? 1L : 0L, lane) != 0)
    throw std::runtime_error("elliptic factory declaration failed on at least one rank");
  if (all_reduce_max(!supported || factory_contract.empty() || !expected.valid() ? 1L : 0L, lane) !=
      0)
    throw std::invalid_argument("elliptic factory cannot materialize the exact-ranked request");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("elliptic-factory"), std::string_view(factory_contract)},
           {std::string_view("elliptic-expected-operator"), expected.exact_fingerprint()}},
          lane))
    throw std::invalid_argument("elliptic factory differs between execution-lane ranks");

  EllipticFactoryBuildResult<Solver> build = factory.build(std::move(request));
  if (all_reduce_max(build.error != nullptr || !build.solver.has_value() ? 1L : 0L, lane) != 0)
    throw std::runtime_error("elliptic factory construction failed on at least one rank");
  Solver& solver = *build.solver;

  bool inspection_failed = false;
  bool mismatch = false;
  std::string actual_contract;
  std::string rhs_contract;
  std::string phi_contract;
  try {
    auto& rhs = solver.rhs();
    auto& phi = solver.phi();
    actual_contract = solver.prepared_operator_contract().exact_fingerprint();
    const auto field_mismatch = [&](const auto& field, const Extent<Solver::dimension>& ghosts) {
      return field.layout() != requested_boxes || field.distribution() != requested_distribution ||
             field.local_rank() != requested_local_rank || field.ncomp() != 1 ||
             field.ghosts() != ghosts;
    };
    mismatch = actual_contract != expected.exact_fingerprint() ||
               field_mismatch(rhs, requested_rhs_ghosts) ||
               field_mismatch(phi, requested_phi_ghosts) || rhs.shares_storage_with(phi) ||
               solver.geom() != requested_geometry;
    rhs_contract = elliptic_contract_detail::field_layout_contract(rhs);
    phi_contract = elliptic_contract_detail::field_layout_contract(phi);
  } catch (...) {
    inspection_failed = true;
  }
  if (all_reduce_max(inspection_failed ? 1L : 0L, lane) != 0)
    throw std::runtime_error("elliptic backend inspection failed on at least one rank");
  if (all_reduce_max(mismatch ? 1L : 0L, lane) != 0)
    throw std::invalid_argument(
        "elliptic backend did not materialize the requested operator and field contract");
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("elliptic-actual-operator"), std::string_view(actual_contract)},
           {std::string_view("elliptic-rhs-layout"), std::string_view(rhs_contract)},
           {std::string_view("elliptic-phi-layout"), std::string_view(phi_contract)}},
          lane))
    throw std::invalid_argument("elliptic backend differs between execution-lane ranks");
  return std::move(solver);
}

template <EllipticSolver Solver, class Factory>
  requires EllipticFactory<std::remove_cvref_t<Factory>, Solver>
Solver make_elliptic_solver(EllipticBuildRequest<Solver::dimension> request, Factory&& factory) {
  const ExecutionLane lane = ExecutionLane::world();
  return make_elliptic_solver<Solver>(std::move(request), std::forward<Factory>(factory), lane);
}

}  // namespace pops
