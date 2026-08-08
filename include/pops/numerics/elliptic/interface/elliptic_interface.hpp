/// @file
/// @brief Compile-time-ranked concepts and providers for the Cartesian elliptic stage.

#pragma once

#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/interface/elliptic_problem.hpp>
#include <pops/numerics/elliptic/interface/elliptic_solver.hpp>
#include <pops/numerics/elliptic/linear/solve_report.hpp>

#include <concepts>
#include <type_traits>

namespace pops {

namespace elliptic_interface_detail {

template <class Owner>
using OwnerField = typename Owner::field_type;

template <class Owner>
concept ExactRankedFieldOwner = requires {
  { Owner::dimension } -> std::convertible_to<int>;
  requires(Owner::dimension >= 1 && Owner::dimension <= 3);
  typename Owner::field_type;
  requires(OwnerField<Owner>::dimension == Owner::dimension);
  requires std::same_as<typename OwnerField<Owner>::box_type, Box<Owner::dimension>>;
};

}  // namespace elliptic_interface_detail

/// Prepared Cartesian operator whose rank, materialization request and published contract agree at
/// compile time.  Coefficient families are advertised separately by backend capability providers;
/// this concept deliberately does not revive the former nullable 2D coefficient-pointer protocol.
template <class Operator>
concept EllipticOperator =
    elliptic_interface_detail::ExactRankedFieldOwner<Operator> &&
    requires(const Operator& operation, const EllipticBuildRequest<Operator::dimension>& request) {
      typename Operator::request_type;
      requires std::same_as<typename Operator::request_type,
                            EllipticBuildRequest<Operator::dimension>>;
      { Operator::operator_identity() } noexcept -> std::same_as<EllipticOperatorIdentity>;
      { Operator::expected_operator_contract(request) } -> std::same_as<EllipticOperatorContract>;
      { operation.geom() } noexcept -> std::same_as<const Geometry<Operator::dimension>&>;
      {
        operation.boundary()
      } noexcept -> std::same_as<const PhysicalBoundaryConditions<Operator::dimension>&>;
      {
        operation.prepared_operator_contract()
      } noexcept -> std::same_as<const EllipticOperatorContract&>;
    };

/// Iterative exact-ranked backend with a prepared stopping policy and one fallible SolveReport.
/// An exact direct provider may still model EllipticSolver, but a backend that returns `void`
/// intentionally does not model this stronger contract.
template <class Solver>
concept LinearSolver = EllipticSolver<Solver> && EllipticOperator<Solver> &&
                       requires(Solver& solver, const Solver& constant_solver) {
                         {
                           constant_solver.maximum_iterations()
                         } noexcept -> std::convertible_to<int>;
                         { solver.solve() } -> std::same_as<SolveReport>;
                         {
                           constant_solver.last_solve_report()
                         } noexcept -> std::same_as<const SolveReport&>;
                       };

/// Exact-ranked provider for the canonical centered potential/gradient publication.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct CenteredFieldPostProcessor {
  static_assert(Dim >= 1 && Dim <= 3,
                "CenteredFieldPostProcessor only supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim, MemorySpace>;

  void operator()(const Geometry<Dim>& geometry, const field_type& potential, field_type& output,
                  FieldPostProcess spec) const {
    field_postprocess(geometry, potential, output, spec);
  }
};

/// Field publication provider whose geometry and field rank are one immutable type property.
template <class Processor>
concept FieldPostProcessor =
    elliptic_interface_detail::ExactRankedFieldOwner<Processor> &&
    requires(const Processor& processor, const Geometry<Processor::dimension>& geometry,
             const typename Processor::field_type& potential,
             typename Processor::field_type& output, FieldPostProcess spec) {
      { processor(geometry, potential, output, spec) } -> std::same_as<void>;
    };

static_assert(FieldPostProcessor<CenteredFieldPostProcessor<1>>);
static_assert(FieldPostProcessor<CenteredFieldPostProcessor<2>>);
static_assert(FieldPostProcessor<CenteredFieldPostProcessor<3>>);

}  // namespace pops
