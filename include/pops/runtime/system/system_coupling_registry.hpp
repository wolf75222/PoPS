#pragma once

#include <pops/core/foundation/types.hpp>              // Real
#include <pops/coupling/source/coupling_operator.hpp>  // CouplingOperatorView (inspect metadata)
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/system/exact_field_marshaling.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

/// @file
/// @brief The inter-species COUPLING registry of a System (ADC-578).
///
/// Extracted from the inline coupling members of `System::Impl`: the splitting-source operators, the
/// GLOBAL host dt bounds, the constant / per-cell coupled-source frequency bounds, and the typed
/// coupling-operator inspect views. Grouping them names one subsystem: "the couplings and the step
/// bounds they impose".
///
/// STEPPER VISIBILITY: `dt_bounds`, `coupled_freqs` and `coupled_freq_exprs` are read by
/// `System<Dim>::step_cfl`; `operators` are consumed only by explicit Program lowering.
/// `coupled_operators` is metadata only and is inspected directly from the registry.
///
/// OWNERSHIP CONTRACT: every field is FROZEN AT BIND (populated only by the structural setters
/// add_coupled_source / add_coupling_operator / add_dt_bound, refused once bound) and READ during run
/// by the stepper. Nothing here is checkpointed (re-declared by replaying the composition).

namespace pops {

namespace runtime {
namespace system {

/// GLOBAL time-step bound (System::add_dt_bound): evaluated ONCE per `step_cfl` (host). Hook for
/// non-cell-local constraints (multi-block coupling, Schur/Poisson, scheduler). Empty means no
/// additional Program macro-step constraint.
struct GlobalDtBound {
  std::string label;
  std::function<double()> fn;
};

/// DECLARED constant frequency of a coupled source (CoupledSource.frequency). The couplings apply
/// ONCE per MACRO-step, so the bound is on the macro-dt: dt <= cfl / mu, WITHOUT a substeps/stride
/// factor. Empty (default) -> no bound.
struct CoupledFreq {
  std::string label;
  double mu;
};

/// Prepared exact-ranked reduction of a state-dependent coupling frequency. The provider captures
/// its authenticated fields and returns the collective maximum; the generic registry never stores
/// one provider's bytecode or array view.
struct PreparedCoupledFrequency {
  std::string label;
  std::function<Real()> maximum_frequency;
};

/// One exact owner-qualified conservative state component.  `owner` and `state_role` are semantic
/// identities; canonical block/component ordinals are the already-resolved storage address.  The
/// registry never parses axis strings or looks through an auxiliary slab to recover this mapping.
struct PreparedCouplingStateRole {
  std::string owner;
  std::size_t canonical_block = std::numeric_limits<std::size_t>::max();
  int component = -1;
  std::string state_role;

  friend bool operator==(const PreparedCouplingStateRole&,
                         const PreparedCouplingStateRole&) = default;
};

/// A declared pointwise invariant across owner-qualified state components.  Coupled sources are
/// cell-local, so validating every local cell is stronger than comparing one cancellation-prone
/// global sum and lets the prepared hierarchy consensus a rank-local rejection before publication.
struct PreparedCouplingConservationGroup {
  std::string identity;
  std::vector<PreparedCouplingStateRole> members;
  Real absolute_tolerance = Real(64) * std::numeric_limits<Real>::epsilon();
  Real relative_tolerance = Real(64) * std::numeric_limits<Real>::epsilon();
};

/// One executable coupling plus its structured conservation ledger.  The executable receives the
/// complete canonical state pack selected by the Program and may mutate only detached candidates.
template <int Dim>
class PreparedCouplingOperator {
 public:
  using operation_type = std::function<void(Real, const std::vector<MultiFab<Dim>*>&)>;

  PreparedCouplingOperator() = default;
  PreparedCouplingOperator(operation_type operation,
                           std::vector<PreparedCouplingConservationGroup> conservation = {})
      : operation_(std::move(operation)), conservation_(std::move(conservation)) {
    validate_conservation_contract_();
  }

  template <class Operation>
    requires(!std::is_same_v<std::remove_cvref_t<Operation>, PreparedCouplingOperator> &&
             std::is_invocable_r_v<void, Operation&, Real, const std::vector<MultiFab<Dim>*>&>)
  PreparedCouplingOperator(Operation&& operation)
      : PreparedCouplingOperator(operation_type(std::forward<Operation>(operation))) {}

  explicit operator bool() const noexcept { return static_cast<bool>(operation_); }
  void operator()(Real dt, const std::vector<MultiFab<Dim>*>& states) const {
    operation_(dt, states);
  }
  const std::vector<PreparedCouplingConservationGroup>& conservation_groups() const noexcept {
    return conservation_;
  }

 private:
  void validate_conservation_contract_() const {
    for (const auto& group : conservation_) {
      if (group.identity.empty() || group.members.size() < 2 ||
          !std::isfinite(static_cast<double>(group.absolute_tolerance)) ||
          !std::isfinite(static_cast<double>(group.relative_tolerance)) ||
          group.absolute_tolerance < Real(0) || group.relative_tolerance < Real(0))
        throw std::invalid_argument("prepared coupling conservation group is incomplete");
      for (std::size_t member = 0; member < group.members.size(); ++member) {
        const auto& role = group.members[member];
        if (role.owner.empty() || role.state_role.empty() || role.component < 0 ||
            std::find(group.members.begin(), group.members.begin() + member, role) !=
                group.members.begin() + member)
          throw std::invalid_argument(
              "prepared coupling conservation group has an invalid owner-qualified state role");
      }
    }
  }

  operation_type operation_;
  std::vector<PreparedCouplingConservationGroup> conservation_;
};

/// Prepared registry of the couplings and the step bounds they impose.
template <int Dim>
struct SystemCouplingRegistry {
  static_assert(Dim >= 1 && Dim <= 3,
                "SystemCouplingRegistry only supports dimensions 1, 2, and 3");

  /// Inter-species coupled sources applied by an explicit Program node after transport.  Each
  /// operator consumes the exact simultaneous candidate-state pack supplied by that Program.
  std::vector<PreparedCouplingOperator<Dim>> operators;
  /// Exact ordered provider contracts corresponding one-to-one with `operators`.  The contract
  /// authenticates executable identity/version, inspect metadata, block map and frequency route;
  /// application compares the whole sequence across ranks rather than trusting only its length.
  std::vector<std::string> operator_contracts;
  /// GLOBAL host dt bounds (add_dt_bound). Read by the stepper.
  std::vector<GlobalDtBound> dt_bounds;
  /// constant coupled-source frequency bounds. Read by the stepper.
  std::vector<CoupledFreq> coupled_freqs;
  /// per-cell coupled-source frequency bounds. Read by the stepper.
  std::vector<PreparedCoupledFrequency> coupled_frequencies;
  /// TYPED coupling-operator inspect views (label + declared conservation / frequency contracts), in
  /// registration order. METADATA ONLY: never read by the stepper.
  std::vector<CouplingOperatorView> coupled_operators;

  std::size_t apply(Real dt, const std::vector<MultiFab<Dim>*>& states) const {
    for (std::size_t index = 0; index < operators.size(); ++index) {
      const auto& op = operators[index];
      std::vector<MultiFab<Dim>> before;
      if (!op.conservation_groups().empty()) {
        before.reserve(states.size());
        for (const MultiFab<Dim>* state : states) {
          if (state == nullptr)
            throw std::invalid_argument("prepared coupling candidate pack contains a null state");
          before.emplace_back(*state);
        }
      }
      op(dt, states);
      require_finite_candidates_(states, index);
      require_conservative_candidates_(before, states, op.conservation_groups(), index);
    }
    return operators.size();
  }

 private:
  static void require_finite_candidates_(const std::vector<MultiFab<Dim>*>& states,
                                         std::size_t operator_index) {
    for (std::size_t block = 0; block < states.size(); ++block) {
      const MultiFab<Dim>* field = states[block];
      if (field == nullptr)
        throw std::invalid_argument("prepared coupling candidate pack contains a null state");
      for (std::size_t local = 0; local < field->local_size(); ++local) {
        const Fab<Dim>& fab = field->fab(local);
        auto host = fab.create_host_mirror();
        fab.copy_to_host(host);
        runtime::system::marshaling::for_each_host_index(
            field->box(local), [&](const Index<Dim>& cell, std::size_t) {
              for (int component = 0; component < field->ncomp(); ++component) {
                const auto ordinal =
                    runtime::system::marshaling::storage_ordinal(fab, cell, component);
                if (!std::isfinite(static_cast<double>(host(ordinal))))
                  throw std::runtime_error("prepared coupling operator " +
                                           std::to_string(operator_index) +
                                           " produced a non-finite candidate for runtime block " +
                                           std::to_string(block));
              }
            });
      }
    }
  }

  static void require_conservative_candidates_(
      const std::vector<MultiFab<Dim>>& before, const std::vector<MultiFab<Dim>*>& after,
      const std::vector<PreparedCouplingConservationGroup>& groups, std::size_t operator_index) {
    if (groups.empty())
      return;
    if (before.size() != after.size())
      throw std::logic_error("prepared coupling conservation snapshot is incomplete");
    for (const auto& group : groups) {
      const auto& prototype_role = group.members.front();
      if (prototype_role.canonical_block >= before.size())
        throw std::out_of_range("prepared coupling conservation owner is out of range");
      const MultiFab<Dim>& prototype = before[prototype_role.canonical_block];
      for (const auto& role : group.members) {
        if (role.canonical_block >= before.size())
          throw std::out_of_range("prepared coupling conservation state role is out of range");
        const MultiFab<Dim>& field = before[role.canonical_block];
        if (role.component >= field.ncomp() || field.layout() != prototype.layout() ||
            field.distribution() != prototype.distribution() ||
            field.local_rank() != prototype.local_rank())
          throw std::invalid_argument(
              "prepared coupling conservation roles do not share one exact cell layout");
      }
      for (std::size_t local = 0; local < prototype.local_size(); ++local) {
        std::vector<typename Fab<Dim>::host_mirror_type> before_hosts;
        std::vector<typename Fab<Dim>::host_mirror_type> after_hosts;
        before_hosts.reserve(group.members.size());
        after_hosts.reserve(group.members.size());
        for (const auto& role : group.members) {
          const Fab<Dim>& before_fab = before[role.canonical_block].fab(local);
          const Fab<Dim>& after_fab = after[role.canonical_block]->fab(local);
          before_hosts.push_back(before_fab.create_host_mirror());
          after_hosts.push_back(after_fab.create_host_mirror());
          before_fab.copy_to_host(before_hosts.back());
          after_fab.copy_to_host(after_hosts.back());
        }
        runtime::system::marshaling::for_each_host_index(
            prototype.box(local), [&](const Index<Dim>& cell, std::size_t) {
              Real prior = Real(0);
              Real candidate = Real(0);
              for (std::size_t member = 0; member < group.members.size(); ++member) {
                const auto& role = group.members[member];
                const Fab<Dim>& before_fab = before[role.canonical_block].fab(local);
                const Fab<Dim>& after_fab = after[role.canonical_block]->fab(local);
                prior += before_hosts[member](
                    runtime::system::marshaling::storage_ordinal(before_fab, cell, role.component));
                candidate += after_hosts[member](
                    runtime::system::marshaling::storage_ordinal(after_fab, cell, role.component));
              }
              const Real tolerance =
                  group.absolute_tolerance +
                  group.relative_tolerance * std::max(std::abs(prior), std::abs(candidate));
              if (!std::isfinite(static_cast<double>(prior)) ||
                  !std::isfinite(static_cast<double>(candidate)) ||
                  std::abs(candidate - prior) > tolerance)
                throw std::runtime_error("prepared coupling operator " +
                                         std::to_string(operator_index) +
                                         " violated conservation group '" + group.identity + "'");
            });
      }
    }
  }
};

}  // namespace system
}  // namespace runtime
}  // namespace pops
