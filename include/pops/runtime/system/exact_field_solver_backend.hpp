/// @file
/// @brief Exact-ranked elliptic backend authority shared by built-in and component providers.

#pragma once

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/halo_exchange.hpp>
#include <pops/numerics/elliptic/nd/cartesian_poisson.hpp>
#include <pops/runtime/system/prepared_field_solver_component.hpp>

#include <array>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::system {

template <int Dim>
class ExactFieldSolverBackend {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "ExactFieldSolverBackend supports dimensions 1, 2, and 3");
  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim>;

  virtual ~ExactFieldSolverBackend() = default;
  virtual field_type& rhs() noexcept = 0;
  virtual field_type& candidate() noexcept = 0;
  virtual const field_type& candidate() const noexcept = 0;
  virtual SolveReport solve(const field_type& warm_start) = 0;
  virtual int maximum_iterations() const noexcept = 0;
  virtual std::string_view provider_identity() const noexcept = 0;
  virtual std::vector<runtime::field::FieldTopologyReportRow> topology_report() const = 0;
};

template <int Dim>
class CartesianCgFieldSolverBackend final : public ExactFieldSolverBackend<Dim> {
 public:
  static constexpr int dimension = Dim;
  using field_type = typename ExactFieldSolverBackend<Dim>::field_type;

  CartesianCgFieldSolverBackend(const Geometry<Dim>& geometry, const mesh::BoxArray<Dim>& layout,
                                const mesh::Distribution<Dim>& distribution, Index<Dim> local_rank,
                                BoundaryTopology<Dim> topology,
                                elliptic::nd::CartesianPoissonOptions<Dim> options,
                                std::string identity = "pops.field-solver.cartesian-cg@1")
      : identity_(std::move(identity)),
        solver_(geometry, layout, distribution, local_rank, std::move(topology),
                std::move(options)) {
    if (identity_.empty())
      throw std::invalid_argument("Cartesian field solver backend identity must be non-empty");
  }

  field_type& rhs() noexcept override { return solver_.rhs(); }
  field_type& candidate() noexcept override { return solver_.candidate(); }
  const field_type& candidate() const noexcept override { return solver_.candidate(); }
  SolveReport solve(const field_type& warm_start) override { return solver_.solve(warm_start); }
  int maximum_iterations() const noexcept override { return solver_.maximum_iterations(); }
  std::string_view provider_identity() const noexcept override { return identity_; }
  std::vector<runtime::field::FieldTopologyReportRow> topology_report() const override {
    return {};
  }

 private:
  std::string identity_;
  elliptic::nd::CartesianPoissonSolver<Dim> solver_;
};

template <int Dim>
class ComponentFieldSolverBackend final : public ExactFieldSolverBackend<Dim> {
 public:
  static constexpr int dimension = Dim;
  using field_type = typename ExactFieldSolverBackend<Dim>::field_type;
  using component_type = runtime::field::PreparedFieldSolverComponent<Dim>;

  ComponentFieldSolverBackend(std::string identity, const Geometry<Dim>& geometry,
                              const mesh::BoxArray<Dim>& layout,
                              const mesh::Distribution<Dim>& distribution, Index<Dim> local_rank,
                              BoundaryTopology<Dim> topology, std::array<bool, Dim> periodicity,
                              std::shared_ptr<component_type> component)
      : identity_(std::move(identity)),
        geometry_(geometry),
        topology_(std::move(topology)),
        periodicity_(periodicity),
        rhs_(layout, distribution, local_rank, 1, Extent<Dim>{}),
        candidate_(layout, distribution, local_rank, 1, unit_ghosts_()),
        halo_schedule_(prepare_halo_schedule(
            candidate_, geometry.domain(), topology_,
            elliptic::nd::detail::exact_halo_budget(layout, geometry.domain(), 1))),
        component_(std::move(component)) {
    std::exception_ptr validation_error;
    try {
      if (identity_.empty() || !component_)
        throw std::invalid_argument("component field backend requires exact provider authorities");
      for (int axis = 0; axis < Dim; ++axis) {
        const Face<Dim> lower{axis, BoundarySide::lower};
        const Face<Dim> upper{axis, BoundarySide::upper};
        if (!periodicity_[static_cast<std::size_t>(axis)] || !topology_.is_periodic(lower) ||
            !topology_.is_periodic(upper))
          throw std::invalid_argument(
              "component field backend currently requires an exact fully-periodic topology");
      }
    } catch (...) {
      validation_error = std::current_exception();
    }
    if (all_reduce_max(validation_error ? 1L : 0L) != 0) {
      if (n_ranks() == 1 && validation_error)
        std::rethrow_exception(validation_error);
      throw std::runtime_error("component field backend preparation failed collectively");
    }
    const bool distributed_halo = all_reduce_max(halo_schedule_.has_remote_jobs() ? 1L : 0L) != 0;
    if (distributed_halo) {
      lane_ = std::make_unique<ExecutionLane>(
          ExecutionLane::duplicate_world_collectively(identity_ + "/solution-halo"));
      HaloExchangeContext context{};
      context.context_generation = 1;
      context.schedule_generation = 1;
      exchange_ = std::make_unique<HaloExchange<Dim>>(halo_schedule_, *lane_, context);
    }
  }

  field_type& rhs() noexcept override { return rhs_; }
  field_type& candidate() noexcept override { return candidate_; }
  const field_type& candidate() const noexcept override { return candidate_; }

  SolveReport solve(const field_type& warm_start) override {
    elliptic::nd::detail::copy_component(warm_start, 0, candidate_, 0);
    Kokkos::fence();
    SolveReport report = component_->solve(rhs_, candidate_, geometry_, periodicity_);
    if (!report.solved_value_available())
      return report;
    if (exchange_)
      exchange_->execute(candidate_, *lane_);
    else
      fill_boundary(candidate_, halo_schedule_);
    return report;
  }

  int maximum_iterations() const noexcept override { return component_->maximum_iterations(); }
  std::string_view provider_identity() const noexcept override { return identity_; }
  std::vector<runtime::field::FieldTopologyReportRow> topology_report() const override {
    return component_->topology_report();
  }

 private:
  static Extent<Dim> unit_ghosts_() {
    Extent<Dim> result{};
    for (int axis = 0; axis < Dim; ++axis)
      result[axis] = 1;
    return result;
  }

  std::string identity_;
  Geometry<Dim> geometry_;
  BoundaryTopology<Dim> topology_;
  std::array<bool, Dim> periodicity_{};
  field_type rhs_;
  field_type candidate_;
  HaloSchedule<Dim> halo_schedule_;
  std::shared_ptr<component_type> component_;
  std::unique_ptr<ExecutionLane> lane_;
  std::unique_ptr<HaloExchange<Dim>> exchange_;
};

}  // namespace pops::runtime::system
