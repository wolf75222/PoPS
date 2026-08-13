/// @file
/// @brief Exact-ranked elliptic backend authority shared by built-in and component providers.

#pragma once

#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/boundary/halo_exchange.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/numerics/elliptic/nd/cartesian_poisson.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>
#include <pops/runtime/system/prepared_field_solver_component.hpp>

#include <array>
#include <cstdint>
#include <exception>
#include <memory>
#include <new>
#include <optional>
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
  virtual void install_boundary_kernel(CompiledFieldBoundaryKernel<Dim> kernel) = 0;
  virtual void validate_boundary_kernel_replacement(
      const std::optional<CompiledFieldBoundaryKernel<Dim>>& kernel) const = 0;
  virtual void replace_boundary_kernel(
      std::optional<CompiledFieldBoundaryKernel<Dim>> kernel) noexcept = 0;
  virtual void set_boundary_contexts(
      std::shared_ptr<const PreparedFieldBoundaryContextSet<Dim>> contexts) = 0;
  virtual void install_newton(FieldNewtonOptions options) = 0;
  virtual void install_nullspace(FieldNullspacePlan<Dim> plan,
                                 PreparedVectorDistribution<Dim> distribution) = 0;
  virtual SolveReport solve(const field_type& warm_start, const ExecutionLane& lane) = 0;
  virtual int maximum_iterations() const noexcept = 0;
  virtual std::string_view provider_identity() const noexcept = 0;
  virtual std::vector<runtime::field::FieldTopologyReportRow> topology_report() const = 0;
};

template <int Dim>
class CartesianCgFieldSolverBackend final : public ExactFieldSolverBackend<Dim> {
  struct DeferCollectivePreparation {};

 public:
  static constexpr int dimension = Dim;
  using field_type = typename ExactFieldSolverBackend<Dim>::field_type;

  CartesianCgFieldSolverBackend(const Geometry<Dim>& geometry, const mesh::BoxArray<Dim>& layout,
                                const mesh::Distribution<Dim>& distribution, Index<Dim> local_rank,
                                BoundaryTopology<Dim> topology,
                                elliptic::nd::CartesianPoissonOptions<Dim> options,
                                const ExecutionLane& lane,
                                std::string identity = "pops.field-solver.cartesian-cg@1")
      : CartesianCgFieldSolverBackend(geometry, layout, distribution, local_rank,
                                      std::move(topology), std::move(options), lane,
                                      std::move(identity), DeferCollectivePreparation{}) {
    solver_.finish_preparation_collectively(lane);
  }

  static std::unique_ptr<ExactFieldSolverBackend<Dim>> prepare_collectively(
      const Geometry<Dim>& geometry, const mesh::BoxArray<Dim>& layout,
      const mesh::Distribution<Dim>& distribution, Index<Dim> local_rank,
      BoundaryTopology<Dim> topology, elliptic::nd::CartesianPoissonOptions<Dim> options,
      const ExecutionLane& lane, std::string_view identity = "pops.field-solver.cartesian-cg@1") {
    std::unique_ptr<CartesianCgFieldSolverBackend> prepared;
    std::exception_ptr local_error;
    try {
      prepared.reset(new CartesianCgFieldSolverBackend(
          geometry, layout, distribution, local_rank, std::move(topology), std::move(options), lane,
          std::string(identity), DeferCollectivePreparation{}));
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("Cartesian field solver local preparation failed collectively");
    }
    std::exception_ptr collective_error;
    try {
      prepared->solver_.finish_preparation_collectively(lane);
    } catch (...) {
      collective_error = std::current_exception();
    }
    if (all_reduce_max(collective_error ? 1L : 0L, lane) != 0) {
      prepared.reset();
      if (lane.size() == 1 && collective_error)
        std::rethrow_exception(collective_error);
      throw std::runtime_error("Cartesian field solver preparation failed collectively");
    }
    return prepared;
  }

 private:
  CartesianCgFieldSolverBackend(const Geometry<Dim>& geometry, const mesh::BoxArray<Dim>& layout,
                                const mesh::Distribution<Dim>& distribution, Index<Dim> local_rank,
                                BoundaryTopology<Dim> topology,
                                elliptic::nd::CartesianPoissonOptions<Dim> options,
                                const ExecutionLane& lane, std::string identity,
                                DeferCollectivePreparation)
      : identity_(std::move(identity)),
        solver_(elliptic::nd::CartesianPoissonSolver<Dim>::prepare_local(
            geometry, layout, distribution, local_rank, std::move(topology), std::move(options),
            lane)) {
    if (identity_.empty())
      throw std::invalid_argument("Cartesian field solver backend identity must be non-empty");
  }

 public:
  field_type& rhs() noexcept override { return solver_.rhs(); }
  field_type& candidate() noexcept override { return solver_.candidate(); }
  const field_type& candidate() const noexcept override { return solver_.candidate(); }
  void install_boundary_kernel(CompiledFieldBoundaryKernel<Dim> kernel) override {
    solver_.install_boundary_kernel(std::move(kernel));
  }
  void validate_boundary_kernel_replacement(
      const std::optional<CompiledFieldBoundaryKernel<Dim>>& kernel) const override {
    if (kernel)
      kernel->validate();
  }
  void replace_boundary_kernel(
      std::optional<CompiledFieldBoundaryKernel<Dim>> kernel) noexcept override {
    solver_.replace_boundary_kernel(std::move(kernel));
  }
  void set_boundary_contexts(
      std::shared_ptr<const PreparedFieldBoundaryContextSet<Dim>> contexts) override {
    solver_.set_boundary_contexts(std::move(contexts));
  }
  void install_newton(FieldNewtonOptions options) override { solver_.install_newton(options); }
  void install_nullspace(FieldNullspacePlan<Dim> plan,
                         PreparedVectorDistribution<Dim> distribution) override {
    solver_.install_nullspace(std::move(plan), std::move(distribution));
  }
  SolveReport solve(const field_type& warm_start, const ExecutionLane& lane) override {
    return solver_.solve(warm_start, lane);
  }
  int maximum_iterations() const noexcept override { return solver_.maximum_iterations(); }
  std::string_view provider_identity() const noexcept override { return identity_; }
  std::vector<runtime::field::FieldTopologyReportRow> topology_report() const override {
    return {};
  }

 private:
  std::string identity_;
  elliptic::nd::CartesianPoissonSolver<Dim> solver_;
};

/// Exact direct Cartesian periodic Poisson backend.  This adapter deliberately exposes only the
/// common System field surface; dynamic boundary and nonlinear contracts cannot be represented by
/// the constant-coefficient FFT operator and therefore fail closed.
template <int Dim>
class PoissonFftFieldSolverBackend final : public ExactFieldSolverBackend<Dim> {
 public:
  static constexpr int dimension = Dim;
  using field_type = typename ExactFieldSolverBackend<Dim>::field_type;
  using request_type = EllipticBuildRequest<Dim>;

  PoissonFftFieldSolverBackend(request_type request, const ExecutionLane& lane,
                               std::string identity = "pops.field-solver.fft@1")
      : identity_(std::move(identity)),
        solver_(make_elliptic_solver<PoissonFFTSolver<Dim>>(std::move(request),
                                                            PoissonFFTFactory<Dim>{lane}, lane)) {
    if (identity_.empty())
      throw std::invalid_argument("FFT field solver backend identity must be non-empty");
  }

  /// Allocate the stable polymorphic object before entering the solver constructor. A rank-local
  /// allocator failure is published on the already prepared execution lane before construction.
  static std::unique_ptr<ExactFieldSolverBackend<Dim>> prepare_collectively(
      request_type request, std::string_view identity, const ExecutionLane& lane) {
    static_assert(std::is_nothrow_move_constructible_v<request_type>);
    void* storage = nullptr;
    std::optional<std::string> prepared_identity;
    std::exception_ptr allocation_error;
    try {
      if (identity.empty())
        throw std::invalid_argument("FFT field solver backend identity must be non-empty");
      prepared_identity.emplace(identity);
      storage = ::operator new(sizeof(PoissonFftFieldSolverBackend));
    } catch (...) {
      allocation_error = std::current_exception();
    }
    if (all_reduce_max(allocation_error ? 1L : 0L, lane) != 0) {
      ::operator delete(storage);
      if (lane.size() == 1 && allocation_error)
        std::rethrow_exception(allocation_error);
      throw std::runtime_error("FFT field solver backend allocation failed collectively");
    }

    PoissonFftFieldSolverBackend* result = nullptr;
    std::exception_ptr construction_error;
    try {
      result = ::new (storage)
          PoissonFftFieldSolverBackend(std::move(request), lane, std::move(*prepared_identity));
    } catch (...) {
      ::operator delete(storage);
      construction_error = std::current_exception();
    }
    if (all_reduce_max(construction_error ? 1L : 0L, lane) != 0) {
      delete result;
      if (lane.size() == 1 && construction_error)
        std::rethrow_exception(construction_error);
      throw std::runtime_error("FFT field solver construction failed collectively");
    }
    return std::unique_ptr<ExactFieldSolverBackend<Dim>>(result);
  }

  field_type& rhs() noexcept override { return solver_.rhs(); }
  field_type& candidate() noexcept override { return solver_.phi(); }
  const field_type& candidate() const noexcept override { return solver_.phi(); }

  void install_boundary_kernel(CompiledFieldBoundaryKernel<Dim>) override {
    throw std::logic_error(
        "FFT field solver has a fixed periodic boundary contract and rejects dynamic kernels");
  }
  void validate_boundary_kernel_replacement(
      const std::optional<CompiledFieldBoundaryKernel<Dim>>& kernel) const override {
    if (kernel)
      throw std::logic_error(
          "FFT field solver has a fixed periodic boundary contract and rejects dynamic kernels");
  }
  void replace_boundary_kernel(
      std::optional<CompiledFieldBoundaryKernel<Dim>> kernel) noexcept override {
    if (kernel)
      std::terminate();
  }
  void set_boundary_contexts(std::shared_ptr<const PreparedFieldBoundaryContextSet<Dim>>) override {
    throw std::logic_error("FFT field solver does not consume a dynamic boundary context");
  }
  void install_newton(FieldNewtonOptions) override {
    throw std::logic_error("FFT field solver is a direct linear backend and rejects Newton setup");
  }
  void install_nullspace(FieldNullspacePlan<Dim> plan,
                         PreparedVectorDistribution<Dim> distribution) override {
    solver_.install_nullspace(std::move(plan), std::move(distribution));
  }
  SolveReport solve(const field_type& warm_start, const ExecutionLane& lane) override {
    if (!solver_.borrows_execution_lane(lane))
      throw std::logic_error(
          "FFT field solver execution lane differs from its prepared lane authority");
    std::exception_ptr copy_error;
    try {
      elliptic::nd::detail::copy_component(warm_start, 0, solver_.phi(), 0);
      Kokkos::fence();
    } catch (...) {
      copy_error = std::current_exception();
    }
    if (all_reduce_max(copy_error ? 1L : 0L, lane) != 0) {
      SolveReport report;
      report.mark_failed(SolveStatus::kInvalidEvaluation, SolveAction::kFailRun,
                         "poisson_fft_warm_start_copy_failed_collectively");
      return report;
    }
    return solver_.solve();
  }
  int maximum_iterations() const noexcept override { return 1; }
  std::string_view provider_identity() const noexcept override { return identity_; }
  std::vector<runtime::field::FieldTopologyReportRow> topology_report() const override {
    return {};
  }

 private:
  std::string identity_;
  PoissonFFTSolver<Dim> solver_;
};

template <int Dim>
class ComponentFieldSolverBackend final : public ExactFieldSolverBackend<Dim> {
  struct DeferCollectivePreparation {};

 public:
  static constexpr int dimension = Dim;
  using field_type = typename ExactFieldSolverBackend<Dim>::field_type;
  using component_type = runtime::field::PreparedFieldSolverComponent<Dim>;

  ComponentFieldSolverBackend(std::string identity, const Geometry<Dim>& geometry,
                              const mesh::BoxArray<Dim>& layout,
                              const mesh::Distribution<Dim>& distribution, Index<Dim> local_rank,
                              BoundaryTopology<Dim> topology, std::array<bool, Dim> periodicity,
                              std::shared_ptr<component_type> component, const ExecutionLane& lane)
      : ComponentFieldSolverBackend(std::move(identity), geometry, layout, distribution, local_rank,
                                    std::move(topology), periodicity, std::move(component), lane,
                                    DeferCollectivePreparation{}) {
    finish_preparation_collectively(lane);
  }

  static std::unique_ptr<ExactFieldSolverBackend<Dim>> prepare_collectively(
      std::string_view identity, const Geometry<Dim>& geometry, const mesh::BoxArray<Dim>& layout,
      const mesh::Distribution<Dim>& distribution, Index<Dim> local_rank,
      BoundaryTopology<Dim> topology, std::array<bool, Dim> periodicity,
      std::shared_ptr<component_type> component, const ExecutionLane& lane) {
    std::unique_ptr<ComponentFieldSolverBackend> prepared;
    std::exception_ptr local_error;
    try {
      prepared.reset(new ComponentFieldSolverBackend(
          std::string(identity), geometry, layout, distribution, local_rank, std::move(topology),
          periodicity, std::move(component), lane, DeferCollectivePreparation{}));
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane) != 0) {
      if (lane.size() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("component field backend local preparation failed collectively");
    }
    std::exception_ptr collective_error;
    try {
      prepared->finish_preparation_collectively(lane);
    } catch (...) {
      collective_error = std::current_exception();
    }
    if (all_reduce_max(collective_error ? 1L : 0L, lane) != 0) {
      prepared.reset();
      if (lane.size() == 1 && collective_error)
        std::rethrow_exception(collective_error);
      throw std::runtime_error("component field backend preparation failed collectively");
    }
    return prepared;
  }

 private:
  ComponentFieldSolverBackend(std::string identity, const Geometry<Dim>& geometry,
                              const mesh::BoxArray<Dim>& layout,
                              const mesh::Distribution<Dim>& distribution, Index<Dim> local_rank,
                              BoundaryTopology<Dim> topology, std::array<bool, Dim> periodicity,
                              std::shared_ptr<component_type> component, const ExecutionLane& lane,
                              DeferCollectivePreparation)
      : identity_(std::move(identity)),
        geometry_(geometry),
        topology_(std::move(topology)),
        periodicity_(periodicity),
        rhs_(layout, distribution, local_rank, 1, Extent<Dim>{}),
        candidate_(layout, distribution, local_rank, 1, unit_ghosts_()),
        halo_schedule_(prepare_halo_schedule(
            candidate_, geometry.domain(), topology_,
            elliptic::nd::detail::exact_halo_budget(layout, geometry.domain(), 1))),
        component_(std::move(component)),
        lane_(&lane),
        lane_borrow_(lane.borrow_immutably()) {
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
  }

  void finish_preparation_collectively(const ExecutionLane& lane) {
    if (&lane != lane_)
      throw std::invalid_argument("component field backend preparation lane changed");
    const bool distributed_halo =
        all_reduce_max(halo_schedule_.has_remote_jobs() ? 1L : 0L, lane) != 0;
    if (!distributed_halo)
      return;
    void* storage = nullptr;
    std::exception_ptr allocation_error;
    try {
      storage = ::operator new(sizeof(HaloExchange<Dim>));
    } catch (...) {
      allocation_error = std::current_exception();
    }
    if (all_reduce_max(allocation_error ? 1L : 0L, lane) != 0) {
      ::operator delete(storage);
      if (lane.size() == 1 && allocation_error)
        std::rethrow_exception(allocation_error);
      throw std::runtime_error("component field halo allocation failed collectively");
    }
    HaloExchangeContext context{};
    context.context_generation = 1;
    context.schedule_generation = 1;
    HaloExchange<Dim>* prepared_exchange = nullptr;
    std::exception_ptr construction_error;
    try {
      prepared_exchange = ::new (storage) HaloExchange<Dim>(halo_schedule_, lane, context);
    } catch (...) {
      ::operator delete(storage);
      construction_error = std::current_exception();
    }
    if (all_reduce_max(construction_error ? 1L : 0L, lane) != 0) {
      delete prepared_exchange;
      if (lane.size() == 1 && construction_error)
        std::rethrow_exception(construction_error);
      throw std::runtime_error("component field halo preparation failed collectively");
    }
    exchange_.reset(prepared_exchange);
  }

 public:
  field_type& rhs() noexcept override { return rhs_; }
  field_type& candidate() noexcept override { return candidate_; }
  const field_type& candidate() const noexcept override { return candidate_; }

  void install_boundary_kernel(CompiledFieldBoundaryKernel<Dim>) override {
    throw std::logic_error(
        "external exact field components must own their boundary closure in the component ABI");
  }

  void validate_boundary_kernel_replacement(
      const std::optional<CompiledFieldBoundaryKernel<Dim>>& kernel) const override {
    if (kernel)
      throw std::logic_error(
          "external exact field components must own their boundary closure in the component ABI");
  }

  void replace_boundary_kernel(
      std::optional<CompiledFieldBoundaryKernel<Dim>> kernel) noexcept override {
    if (kernel)
      std::terminate();
  }

  void set_boundary_contexts(std::shared_ptr<const PreparedFieldBoundaryContextSet<Dim>>) override {
    throw std::logic_error(
        "external exact field components do not consume the generated Cartesian boundary ABI");
  }

  void install_newton(FieldNewtonOptions) override {
    throw std::logic_error(
        "external exact field components must own nonlinear iteration in their component ABI");
  }

  void install_nullspace(FieldNullspacePlan<Dim> plan,
                         PreparedVectorDistribution<Dim> distribution) override {
    if (nullspace_installed_)
      throw std::logic_error("component field nullspace is already installed");
    const std::array<PreparedVectorDistribution<Dim>, 1> distributions{distribution};
    validate_field_nullspace_basis<Dim>({&rhs_}, plan,
                                        std::span<const PreparedVectorDistribution<Dim>>(
                                            distributions.data(), distributions.size()),
                                        *lane_);
    nullspace_plan_ = std::move(plan);
    nullspace_distribution_ = std::move(distribution);
    nullspace_installed_ = true;
  }

  SolveReport solve(const field_type& warm_start, const ExecutionLane& lane) override {
    if (all_reduce_max(&lane == lane_ ? 0L : 1L, *lane_) != 0)
      throw std::invalid_argument("component field solve requires its prepared execution lane");
    if (!nullspace_installed_)
      throw std::logic_error("component field solve has no prepared nullspace authority");
    require_field_nullspace_compatible(rhs_, nullspace_plan_, nullspace_distribution_, lane);
    elliptic::nd::detail::copy_component(warm_start, 0, candidate_, 0);
    apply_field_gauge(candidate_, nullspace_plan_, nullspace_distribution_, lane);
    Kokkos::fence();
    SolveReport report = component_->solve(rhs_, candidate_, geometry_, periodicity_);
    if (!report.solved_value_available())
      return report;
    apply_field_gauge(candidate_, nullspace_plan_, nullspace_distribution_, lane);
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
  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  std::unique_ptr<HaloExchange<Dim>> exchange_;
  FieldNullspacePlan<Dim> nullspace_plan_;
  PreparedVectorDistribution<Dim> nullspace_distribution_ =
      PreparedVectorDistribution<Dim>::distributed();
  bool nullspace_installed_ = false;
};

}  // namespace pops::runtime::system
