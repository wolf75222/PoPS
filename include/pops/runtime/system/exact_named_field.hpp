/// @file
/// @brief Transactional exact-ranked named elliptic field provider for System.

#pragma once

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_provider.hpp>
#include <pops/runtime/named_field_output.hpp>
#include <pops/runtime/named_field_publication.hpp>
#include <pops/runtime/system/derived_aux_provider.hpp>
#include <pops/runtime/system/exact_field_solver_backend.hpp>

#include <cmath>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::system {

template <int Dim>
class ExactNamedField final {
 public:
  using field_type = MultiFab<Dim>;
  using rhs_type = std::function<void(const field_type&, field_type&)>;
  using solver_type = ExactFieldSolverBackend<Dim>;
  using output_key_type = AuxiliaryComponentKey;

  struct AcceptedState {
    field_type potential;
    field_type outputs;
  };

  struct PreparedRhs {
    rhs_type evaluate;
    Real coefficient = Real(1);
  };

  ExactNamedField(std::string identity, std::string output_block,
                  runtime::field::NamedFieldOutput<Dim> output, const Geometry<Dim>& geometry,
                  const mesh::BoxArray<Dim>& layout, const mesh::Distribution<Dim>& distribution,
                  Index<Dim> local_rank, BoundaryTopology<Dim> topology,
                  elliptic::nd::CartesianPoissonOptions<Dim> options, std::size_t block_count,
                  const ExecutionLane& lane, std::vector<output_key_type> output_keys = {})
      : lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        identity_(std::move(identity)),
        output_block_(std::move(output_block)),
        output_(output),
        output_keys_(std::move(output_keys)),
        geometry_(geometry),
        accepted_(layout, distribution, local_rank, 1, unit_ghosts_()),
        accepted_outputs_(layout, distribution, local_rank,
                          static_cast<int>(output.component_count()), Extent<Dim>{}),
        candidate_outputs_(layout, distribution, local_rank,
                           static_cast<int>(output.component_count()), Extent<Dim>{}),
        solver_(std::make_unique<CartesianCgFieldSolverBackend<Dim>>(
            geometry, layout, distribution, local_rank, std::move(topology), std::move(options),
            lane)),
        rhs_by_block_(block_count) {
    validate_();
  }

  ExactNamedField(std::string identity, std::string output_block,
                  runtime::field::NamedFieldOutput<Dim> output, const Geometry<Dim>& geometry,
                  const mesh::BoxArray<Dim>& layout, const mesh::Distribution<Dim>& distribution,
                  Index<Dim> local_rank, std::unique_ptr<solver_type> solver,
                  std::size_t block_count, const ExecutionLane& lane,
                  std::vector<output_key_type> output_keys = {})
      : lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        identity_(std::move(identity)),
        output_block_(std::move(output_block)),
        output_(output),
        output_keys_(std::move(output_keys)),
        geometry_(geometry),
        accepted_(layout, distribution, local_rank, 1, unit_ghosts_()),
        accepted_outputs_(layout, distribution, local_rank,
                          static_cast<int>(output.component_count()), Extent<Dim>{}),
        candidate_outputs_(layout, distribution, local_rank,
                           static_cast<int>(output.component_count()), Extent<Dim>{}),
        solver_(std::move(solver)),
        rhs_by_block_(block_count) {
    validate_();
  }

  const std::string& identity() const noexcept { return identity_; }
  const std::string& output_block() const noexcept { return output_block_; }
  const runtime::field::NamedFieldOutput<Dim>& output() const noexcept { return output_; }
  const std::vector<output_key_type>& output_keys() const noexcept { return output_keys_; }
  const field_type& accepted_potential() const noexcept { return accepted_; }
  const field_type& accepted_outputs() const noexcept { return accepted_outputs_; }
  const field_type& candidate_outputs() const {
    if (!active_ || !candidate_ready_)
      throw std::logic_error("named elliptic field has no prepared output candidate");
    return candidate_outputs_;
  }
  const field_type& dependency_potential() const noexcept {
    return active_ && candidate_ready_ ? solver_->candidate() : accepted_;
  }
  AcceptedState accepted_state() const { return {accepted_, accepted_outputs_}; }
  void restore_accepted_state(const AcceptedState& state) {
    if (active_)
      throw std::logic_error("cannot restore a named field while a solve is active");
    authenticate_layout_(state.potential, "restored potential", false);
    authenticate_output_layout_(state.outputs, "restored outputs");
    accepted_ = state.potential;
    accepted_outputs_ = state.outputs;
  }
  field_type& accepted_potential_for_restore() { return accepted_; }
  /// Restore authority for the accepted output image. The owning System transaction has already
  /// authenticated the exact field layout; exposing this resident buffer lets it deep-copy into
  /// preallocated storage without invoking value assignment (which would allocate a fresh Fab).
  field_type& accepted_outputs_for_restore() { return accepted_outputs_; }
  int maximum_iterations() const noexcept { return solver_->maximum_iterations(); }
  std::string_view solver_provider_identity() const noexcept {
    return solver_->provider_identity();
  }
  std::vector<runtime::field::FieldTopologyReportRow> topology_report() const {
    return accepted_topology_report_;
  }

  void install_boundary_kernel(CompiledFieldBoundaryKernel<Dim> kernel) {
    if (active_)
      throw std::logic_error("cannot install a named-field boundary while a solve is active");
    if (has_boundary_kernel_)
      throw std::logic_error("named-field boundary kernel is already installed");
    solver_->install_boundary_kernel(std::move(kernel));
    has_boundary_kernel_ = true;
  }

  void validate_boundary_kernel_replacement(
      const std::optional<CompiledFieldBoundaryKernel<Dim>>& kernel) const {
    if (active_)
      throw std::logic_error("cannot replace a named-field boundary while a solve is active");
    solver_->validate_boundary_kernel_replacement(kernel);
  }

  void replace_boundary_kernel(std::optional<CompiledFieldBoundaryKernel<Dim>> kernel) noexcept {
    const bool installed = kernel.has_value();
    solver_->replace_boundary_kernel(std::move(kernel));
    has_boundary_kernel_ = installed;
  }

  void install_newton(FieldNewtonOptions options) {
    if (active_)
      throw std::logic_error("cannot install named-field Newton while a solve is active");
    if (has_newton_)
      throw std::logic_error("named-field Newton authority is already installed");
    solver_->install_newton(options);
    has_newton_ = true;
  }

  void install_nullspace(PreparedFieldNullspace<Dim> prepared,
                         PreparedVectorDistribution<Dim> distribution) {
    if (active_)
      throw std::logic_error("cannot install a named-field nullspace while a solve is active");
    if (prepared_nullspace_)
      throw std::logic_error("named-field nullspace is already installed");
    if (prepared.provider_identity.empty() || prepared.provider_version == 0 ||
        prepared.exact_prepared_contract.empty())
      throw std::invalid_argument("named-field nullspace lacks an exact prepared authority");
    solver_->install_nullspace(std::move(prepared.plan), std::move(distribution));
    prepared_nullspace_ = std::move(prepared);
  }

  using rhs_image_type = std::vector<std::vector<PreparedRhs>>;

  static void append_rhs(rhs_image_type& image, std::size_t block, rhs_type rhs, Real coefficient) {
    if (block >= image.size())
      throw std::out_of_range("named-field RHS block index is outside the prepared registry");
    if (!rhs || !std::isfinite(static_cast<double>(coefficient)))
      throw std::invalid_argument("named-field RHS provider or coefficient is invalid");
    image[block].push_back({std::move(rhs), coefficient});
  }

  void add_rhs(std::size_t block, rhs_type rhs, Real coefficient) {
    if (active_)
      throw std::logic_error("cannot add a named-field RHS while a solve is active");
    append_rhs(rhs_by_block_, block, std::move(rhs), coefficient);
  }

  [[nodiscard]] rhs_image_type rhs_image() const {
    if (active_)
      throw std::logic_error("cannot snapshot a named-field RHS while a solve is active");
    return rhs_by_block_;
  }

  void publish_rhs_image(rhs_image_type& candidate) noexcept {
    static_assert(std::is_nothrow_swappable_v<rhs_image_type>);
    if (active_ || candidate.size() != rhs_by_block_.size())
      std::terminate();
    rhs_by_block_.swap(candidate);
  }

  SolveReport solve_candidate(
      const std::vector<const field_type*>& states,
      std::shared_ptr<const PreparedFieldBoundaryContextSet<Dim>> boundary_contexts,
      const ExecutionLane& lane) {
    if (all_reduce_max(&lane == lane_ ? 0L : 1L, *lane_) != 0)
      throw std::invalid_argument(
          "named elliptic field solve requires its prepared execution lane");
    std::exception_ptr preparation_error;
    try {
      if (active_)
        throw std::logic_error("named elliptic field already owns an unconsumed solve candidate");
      if (states.size() != rhs_by_block_.size())
        throw std::invalid_argument("named-field state vector does not cover the block registry");
      authenticate_output_layout_(candidate_outputs_, "candidate outputs");

      bool has_rhs = false;
      for (std::size_t block = 0; block < states.size(); ++block) {
        if (rhs_by_block_[block].empty())
          continue;
        if (states[block] == nullptr)
          throw std::invalid_argument("named-field contributing block has no state");
        authenticate_layout_(*states[block], "contributing state", false);
        has_rhs = true;
      }
      if (!has_rhs)
        throw std::runtime_error("named elliptic field has no prepared RHS provider");
      if (has_boundary_kernel_) {
        if (!boundary_contexts || boundary_contexts->size() != 1)
          throw std::logic_error(
              "named elliptic field dynamic boundary has no prepared execution context");
        solver_->set_boundary_contexts(boundary_contexts);
      } else if (boundary_contexts) {
        throw std::logic_error(
            "named elliptic field received a boundary context without a compiled kernel");
      }
    } catch (...) {
      preparation_error = std::current_exception();
    }
    collective_rethrow_(preparation_error, "named-field preparation failed collectively", lane);

    active_ = true;
    candidate_ready_ = false;

    std::exception_ptr rhs_error;
    try {
      solver_->rhs().set_val(Real(0));
      if (!contribution_scratch_)
        contribution_scratch_.emplace(solver_->rhs().layout(), solver_->rhs().distribution(),
                                      solver_->rhs().local_rank(), 1, Extent<Dim>{});
      for (std::size_t block = 0; block < states.size(); ++block)
        for (const PreparedRhs& provider : rhs_by_block_[block]) {
          contribution_scratch_->set_val(Real(0));
          provider.evaluate(*states[block], *contribution_scratch_);
          saxpy(solver_->rhs(), provider.coefficient, *contribution_scratch_);
        }
      ::pops::device_fence();
    } catch (...) {
      rhs_error = std::current_exception();
    }
    if (all_reduce_max(rhs_error ? 1L : 0L, lane) != 0) {
      clear_candidate_();
      if (lane.size() == 1 && rhs_error)
        std::rethrow_exception(rhs_error);
      throw std::runtime_error("named-field RHS assembly failed collectively");
    }

    try {
      SolveReport report = solver_->solve(accepted_, lane);
      if (!report.solved_value_available())
        return report;

      runtime::field::publish_named_field(solver_->candidate(), candidate_outputs_, geometry_,
                                          output_);
      candidate_topology_report_ = solver_->topology_report();
      candidate_ready_ = true;
      return report;
    } catch (...) {
      active_ = false;
      candidate_ready_ = false;
      throw;
    }
  }

  void validate_candidate() const {
    if (!active_ || !candidate_ready_)
      throw std::logic_error("named elliptic field has no publication candidate");
    authenticate_output_layout_(candidate_outputs_, "candidate outputs");
  }

  void accept_candidate() noexcept {
    try {
      validate_candidate();
      copy_all_cells_(solver_->candidate(), accepted_);
      copy_all_cells_(candidate_outputs_, accepted_outputs_);
      accepted_topology_report_ = std::move(candidate_topology_report_);
      ::pops::device_fence();
      clear_candidate_();
    } catch (...) {
      std::terminate();
    }
  }

  void reject_candidate() noexcept { clear_candidate_(); }

 private:
  void validate_() {
    if (identity_.empty() || output_block_.empty() || !solver_)
      throw std::invalid_argument("exact named field identities must be non-empty");
    if (!output_keys_.empty() && output_keys_.size() != output_.component_count())
      throw std::invalid_argument(
          "exact named field output keys do not cover its compact publication carrier");
    for (const auto& key : output_keys_)
      key.validate();
    accepted_.set_val(Real(0));
    accepted_outputs_.set_val(Real(0));
    candidate_outputs_.set_val(Real(0));
  }

  static void collective_rethrow_(const std::exception_ptr& error, const char* message,
                                  const ExecutionLane& lane) {
    if (all_reduce_max(error ? 1L : 0L, lane) == 0)
      return;
    if (lane.size() == 1 && error)
      std::rethrow_exception(error);
    throw std::runtime_error(message);
  }

  static Extent<Dim> unit_ghosts_() {
    Extent<Dim> result{};
    for (int axis = 0; axis < Dim; ++axis)
      result[axis] = 1;
    return result;
  }

  void authenticate_layout_(const field_type& field, const char* role,
                            bool require_components = true) const {
    if (field.layout() != accepted_.layout() || field.distribution() != accepted_.distribution() ||
        field.local_rank() != accepted_.local_rank())
      throw std::invalid_argument(std::string("exact named field ") + role +
                                  " does not share its prepared ND layout");
    if (require_components && field.ncomp() < 1)
      throw std::invalid_argument("exact named field input has no components");
  }

  void authenticate_output_layout_(const field_type& field, const char* role) const {
    if (field.layout() != accepted_outputs_.layout() ||
        field.distribution() != accepted_outputs_.distribution() ||
        field.local_rank() != accepted_outputs_.local_rank() ||
        field.ghosts() != accepted_outputs_.ghosts() || field.ncomp() != accepted_outputs_.ncomp())
      throw std::invalid_argument(std::string("exact named field ") + role +
                                  " does not share its compact output layout");
  }

  static void copy_all_cells_(const field_type& source, field_type& destination) {
    if (source.layout() != destination.layout() ||
        source.distribution() != destination.distribution() ||
        source.local_rank() != destination.local_rank() || source.ncomp() != destination.ncomp() ||
        source.ghosts() != destination.ghosts())
      throw std::invalid_argument("exact named field full copy layout differs");
    for (std::size_t local = 0; local < source.local_size(); ++local)
      for (int component = 0; component < source.ncomp(); ++component)
        for_each_cell(
            source.fab(local).grown_box(),
            elliptic::nd::detail::CopyComponentKernel<Dim>{
                destination.fab(local).view(), source.fab(local).view(), component, component});
  }

  void clear_candidate_() noexcept {
    active_ = false;
    candidate_ready_ = false;
    candidate_topology_report_.clear();
  }

  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  std::string identity_;
  std::string output_block_;
  runtime::field::NamedFieldOutput<Dim> output_;
  std::vector<output_key_type> output_keys_;
  Geometry<Dim> geometry_;
  field_type accepted_;
  field_type accepted_outputs_;
  field_type candidate_outputs_;
  std::unique_ptr<solver_type> solver_;
  std::vector<std::vector<PreparedRhs>> rhs_by_block_;
  bool has_boundary_kernel_ = false;
  bool has_newton_ = false;
  std::optional<PreparedFieldNullspace<Dim>> prepared_nullspace_;
  std::optional<field_type> contribution_scratch_;
  std::vector<runtime::field::FieldTopologyReportRow> accepted_topology_report_;
  std::vector<runtime::field::FieldTopologyReportRow> candidate_topology_report_;
  bool active_ = false;
  bool candidate_ready_ = false;
};

}  // namespace pops::runtime::system
