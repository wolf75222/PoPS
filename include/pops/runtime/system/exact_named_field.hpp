/// @file
/// @brief Transactional exact-ranked named elliptic field provider for System.

#pragma once

#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/runtime/named_field_output.hpp>
#include <pops/runtime/named_field_publication.hpp>
#include <pops/runtime/system/exact_field_solver_backend.hpp>

#include <cmath>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::runtime::system {

template <int Dim>
class ExactNamedField final {
 public:
  using field_type = MultiFab<Dim>;
  using rhs_type = std::function<void(const field_type&, field_type&)>;
  using solver_type = ExactFieldSolverBackend<Dim>;

  struct PreparedRhs {
    rhs_type evaluate;
    Real coefficient = Real(1);
  };

  ExactNamedField(std::string identity, std::string output_block,
                  runtime::field::NamedFieldOutput<Dim> output, const Geometry<Dim>& geometry,
                  const mesh::BoxArray<Dim>& layout, const mesh::Distribution<Dim>& distribution,
                  Index<Dim> local_rank, BoundaryTopology<Dim> topology,
                  elliptic::nd::CartesianPoissonOptions<Dim> options, std::size_t block_count)
      : identity_(std::move(identity)),
        output_block_(std::move(output_block)),
        output_(output),
        geometry_(geometry),
        accepted_(layout, distribution, local_rank, 1, unit_ghosts_()),
        solver_(std::make_unique<CartesianCgFieldSolverBackend<Dim>>(
            geometry, layout, distribution, local_rank, std::move(topology), std::move(options))),
        rhs_by_block_(block_count) {
    validate_();
  }

  ExactNamedField(std::string identity, std::string output_block,
                  runtime::field::NamedFieldOutput<Dim> output, const Geometry<Dim>& geometry,
                  const mesh::BoxArray<Dim>& layout, const mesh::Distribution<Dim>& distribution,
                  Index<Dim> local_rank, std::unique_ptr<solver_type> solver,
                  std::size_t block_count)
      : identity_(std::move(identity)),
        output_block_(std::move(output_block)),
        output_(output),
        geometry_(geometry),
        accepted_(layout, distribution, local_rank, 1, unit_ghosts_()),
        solver_(std::move(solver)),
        rhs_by_block_(block_count) {
    validate_();
  }

  const std::string& identity() const noexcept { return identity_; }
  const std::string& output_block() const noexcept { return output_block_; }
  const runtime::field::NamedFieldOutput<Dim>& output() const noexcept { return output_; }
  const field_type& accepted_potential() const noexcept { return accepted_; }
  field_type& accepted_potential_for_restore() {
    if (active_)
      throw std::logic_error("cannot restore a named field while a solve is active");
    return accepted_;
  }
  int maximum_iterations() const noexcept { return solver_->maximum_iterations(); }
  std::string_view solver_provider_identity() const noexcept {
    return solver_->provider_identity();
  }
  std::vector<runtime::field::FieldTopologyReportRow> topology_report() const {
    return accepted_topology_report_;
  }

  void add_rhs(std::size_t block, rhs_type rhs, Real coefficient) {
    if (active_)
      throw std::logic_error("cannot add a named-field RHS while a solve is active");
    if (block >= rhs_by_block_.size())
      throw std::out_of_range("named-field RHS block index is outside the prepared registry");
    if (!rhs || !std::isfinite(static_cast<double>(coefficient)))
      throw std::invalid_argument("named-field RHS provider or coefficient is invalid");
    rhs_by_block_[block].push_back({std::move(rhs), coefficient});
  }

  SolveReport solve_candidate(const std::vector<const field_type*>& states, field_type& live_aux) {
    std::optional<field_type> prepared_aux;
    std::exception_ptr preparation_error;
    try {
      if (active_)
        throw std::logic_error("named elliptic field already owns an unconsumed solve candidate");
      if (states.size() != rhs_by_block_.size())
        throw std::invalid_argument("named-field state vector does not cover the block registry");
      output_.validate_width(live_aux.ncomp(), "exact named field");
      authenticate_layout_(live_aux, "live auxiliary");

      if (!candidate_aux_ || candidate_aux_->ncomp() != live_aux.ncomp() ||
          candidate_aux_->layout() != live_aux.layout() ||
          candidate_aux_->distribution() != live_aux.distribution() ||
          candidate_aux_->local_rank() != live_aux.local_rank())
        prepared_aux.emplace(live_aux.layout(), live_aux.distribution(), live_aux.local_rank(),
                             live_aux.ncomp(), live_aux.ghosts());

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
    } catch (...) {
      preparation_error = std::current_exception();
    }
    collective_rethrow_(preparation_error, "named-field preparation failed collectively");

    if (prepared_aux)
      candidate_aux_ = std::move(prepared_aux);
    live_aux_ = &live_aux;
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
      Kokkos::fence();
    } catch (...) {
      rhs_error = std::current_exception();
    }
    if (all_reduce_max(rhs_error ? 1L : 0L) != 0) {
      clear_candidate_();
      if (n_ranks() == 1 && rhs_error)
        std::rethrow_exception(rhs_error);
      throw std::runtime_error("named-field RHS assembly failed collectively");
    }

    try {
      SolveReport report = solver_->solve(accepted_);
      if (!report.solved_value_available())
        return report;

      copy_all_valid_(live_aux, *candidate_aux_);
      runtime::field::publish_named_field(solver_->candidate(), *candidate_aux_, geometry_,
                                          output_);
      candidate_topology_report_ = solver_->topology_report();
      candidate_ready_ = true;
      return report;
    } catch (...) {
      active_ = false;
      candidate_ready_ = false;
      live_aux_ = nullptr;
      throw;
    }
  }

  void validate_candidate() const {
    if (!active_ || !candidate_ready_ || live_aux_ == nullptr || !candidate_aux_)
      throw std::logic_error("named elliptic field has no publication candidate");
    authenticate_layout_(*live_aux_, "publication destination");
    if (candidate_aux_->ncomp() != live_aux_->ncomp() ||
        candidate_aux_->ghosts() != live_aux_->ghosts())
      throw std::runtime_error("named elliptic field publication layout changed after solve");
  }

  void accept_candidate() noexcept {
    try {
      validate_candidate();
      copy_all_cells_(solver_->candidate(), accepted_);
      copy_output_components_(*candidate_aux_, *live_aux_);
      accepted_topology_report_ = std::move(candidate_topology_report_);
      Kokkos::fence();
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
    accepted_.set_val(Real(0));
  }

  static void collective_rethrow_(const std::exception_ptr& error, const char* message) {
    if (all_reduce_max(error ? 1L : 0L) == 0)
      return;
    if (n_ranks() == 1 && error)
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
                            bool require_aux_width = true) const {
    if (field.layout() != accepted_.layout() || field.distribution() != accepted_.distribution() ||
        field.local_rank() != accepted_.local_rank())
      throw std::invalid_argument(std::string("exact named field ") + role +
                                  " does not share its prepared ND layout");
    if (require_aux_width && field.ncomp() < 1)
      throw std::invalid_argument("exact named field auxiliary carrier has no components");
  }

  static void copy_all_valid_(const field_type& source, field_type& destination) {
    if (source.ncomp() != destination.ncomp())
      throw std::invalid_argument("exact named field copy width differs");
    for (int component = 0; component < source.ncomp(); ++component)
      elliptic::nd::detail::copy_component(source, component, destination, component);
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

  void copy_output_components_(const field_type& source, field_type& destination) const {
    elliptic::nd::detail::copy_component(source, output_.potential_component(), destination,
                                         output_.potential_component());
    if (output_.has_gradients())
      for (int axis = 0; axis < Dim; ++axis)
        elliptic::nd::detail::copy_component(source, output_.gradient_component(axis), destination,
                                             output_.gradient_component(axis));
  }

  void clear_candidate_() noexcept {
    active_ = false;
    candidate_ready_ = false;
    live_aux_ = nullptr;
    candidate_topology_report_.clear();
  }

  std::string identity_;
  std::string output_block_;
  runtime::field::NamedFieldOutput<Dim> output_;
  Geometry<Dim> geometry_;
  field_type accepted_;
  std::unique_ptr<solver_type> solver_;
  std::vector<std::vector<PreparedRhs>> rhs_by_block_;
  std::optional<field_type> contribution_scratch_;
  std::optional<field_type> candidate_aux_;
  std::vector<runtime::field::FieldTopologyReportRow> accepted_topology_report_;
  std::vector<runtime::field::FieldTopologyReportRow> candidate_topology_report_;
  field_type* live_aux_ = nullptr;
  bool active_ = false;
  bool candidate_ready_ = false;
};

}  // namespace pops::runtime::system
