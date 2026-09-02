#pragma once

/// @file
/// @brief Exact-rank, physics-neutral prepared auxiliary-component providers.
///
/// This header deliberately names only storage and scheduling contracts.  It does not reserve
/// component numbers, interpret a component as a particular physical quantity, or loop over
/// cells.  A generated/native component owns its Kokkos launch; the host-side provider below
/// merely authenticates that launch and gives it an immutable, exact evaluation point.

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/mesh/index/index.hpp>
#include <pops/mesh/storage/multifab.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::system {

/// Producer class.  Input data is supplied by an external native uploader, derived data by a
/// prepared native launcher, and field output by a separately prepared field solver.  None of
/// these values implies a physical formula.
enum class AuxiliaryProviderKind : std::uint8_t {
  input = 0,
  derived = 1,
  field_output = 2,
};

struct AuxiliaryBoundaryPolicy {
  enum class Kind : std::uint8_t {
    inherit_topology = 0,
    first_order_extrapolation = 1,
    dirichlet = 2,
  };

  Kind kind = Kind::inherit_topology;
  std::optional<Real> value;

  void validate() const {
    if (kind != Kind::inherit_topology && kind != Kind::first_order_extrapolation &&
        kind != Kind::dirichlet)
      throw std::invalid_argument("auxiliary boundary policy kind is invalid");
    if (kind == Kind::dirichlet) {
      if (!value || !std::isfinite(static_cast<double>(*value)))
        throw std::invalid_argument("Dirichlet auxiliary boundary requires a finite value");
    } else if (value) {
      throw std::invalid_argument("only Dirichlet auxiliary boundary carries a value");
    }
  }
  void serialize_exact(ExactContractBuilder& exact) const {
    validate();
    exact.text("pops.auxiliary-boundary-policy").scalar(std::uint32_t{1}).scalar(kind);
    exact.presence(value.has_value());
    if (value)
      exact.scalar(*value);
  }
};

/// Scheduler event at which a provider is permitted to publish a candidate.
enum class AuxiliaryEvaluationEvent : std::uint8_t {
  initialization = 0,
  before_residual = 1,
  before_field_solve = 2,
  nonlinear_iteration = 3,
  after_regrid = 4,
  output = 5,
};

/// Freshness rule for an auxiliary component.  The rule is structural: it compares exact integer
/// identities only and never infers a stage from a rounded floating physical time.
enum class AuxiliaryFreshness : std::uint8_t {
  once = 0,
  accepted_step = 1,
  evaluation = 2,
  layout_generation = 3,
};

/// Stable address of one component.  The owner prevents otherwise-valid packages from silently
/// aliasing each other; (space_kind, space_name, component) identifies the carrier within it.
struct AuxiliaryComponentKey {
  std::string owner_qid;
  std::string space_kind;
  std::string space_name;
  std::string component;

  friend bool operator==(const AuxiliaryComponentKey&, const AuxiliaryComponentKey&) = default;

  void validate() const {
    if (owner_qid.empty() || space_kind.empty() || space_name.empty() || component.empty())
      throw std::invalid_argument(
          "auxiliary component key requires non-empty owner/space kind/space name/component");
  }

  [[nodiscard]] std::string exact_key() const {
    validate();
    ExactContractBuilder contract;
    contract.text("pops.auxiliary-component-key")
        .scalar(std::uint32_t{1})
        .text(owner_qid)
        .text(space_kind)
        .text(space_name)
        .text(component);
    return std::move(contract).release();
  }

  void serialize_exact(ExactContractBuilder& contract) const { contract.bytes(exact_key()); }
};

/// Python-isomorphic semantic component contract.  Text fields remain package-defined, so the
/// registry compares their exact bytes without maintaining a closed vocabulary. Optional
/// unit/value_kind keep an explicit presence bit instead of conflating absent with empty.
struct AuxiliaryComponentContract {
  std::string representation;
  std::string centering;
  std::optional<std::string> unit;
  std::string layout;
  std::optional<std::string> value_kind;

  friend bool operator==(const AuxiliaryComponentContract&,
                         const AuxiliaryComponentContract&) = default;

  void validate() const {
    if (representation.empty() || centering.empty() || layout.empty())
      throw std::invalid_argument(
          "auxiliary component contract requires representation, centering, and layout");
    if ((unit && unit->empty()) || (value_kind && value_kind->empty()))
      throw std::invalid_argument(
          "auxiliary component optional unit/value kind must be non-empty when provided");
  }

  void serialize_exact(ExactContractBuilder& contract) const {
    validate();
    contract.text("pops.provider-pack.component-contract")
        .scalar(std::uint32_t{1})
        .text(representation)
        .text(centering)
        .presence(unit.has_value());
    if (unit)
      contract.text(*unit);
    contract.text(layout).presence(value_kind.has_value());
    if (value_kind)
      contract.text(*value_kind);
  }
};

/// Native allocation shape. This deliberately stays separate from the Python ComponentContract:
/// semantic lowerings compare the five Python fields exactly, while native rank/halo requirements
/// remain validated before allocation and publication.
template <int Dim>
struct AuxiliaryStorageShape {
  static_assert(Dim >= 1 && Dim <= 3, "AuxiliaryStorageShape supports dimensions 1, 2, and 3");

  int spatial_rank = Dim;
  int value_components = 1;
  Index<Dim> halo{};

  friend bool operator==(const AuxiliaryStorageShape&, const AuxiliaryStorageShape&) = default;

  void validate() const {
    if (spatial_rank != Dim)
      throw std::invalid_argument("auxiliary storage shape rank differs from native dimension");
    if (value_components != 1)
      throw std::invalid_argument(
          "one AuxiliaryComponentKey denotes exactly one scalar provider value");
    for (int axis = 0; axis < Dim; ++axis)
      if (halo[axis] < 0)
        throw std::invalid_argument("auxiliary storage shape halo extent must be non-negative");
  }

  void serialize_exact(ExactContractBuilder& contract) const {
    validate();
    contract.text("pops.auxiliary-storage-shape")
        .scalar(std::uint32_t{1})
        .scalar(spatial_rank)
        .scalar(value_components);
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(halo[axis]);
  }
};

/// One provider-local output declaration.  It deliberately has no global storage index: multiple
/// independently compiled packages may each declare their first output.  The sealed registry
/// assigns globally compact storage components from the canonical ComponentKey order.
template <int Dim>
struct AuxiliaryOutput {
  AuxiliaryComponentKey key;
  AuxiliaryComponentContract contract;
  AuxiliaryStorageShape<Dim> shape;
  AuxiliaryBoundaryPolicy boundary{};

  void validate() const {
    key.validate();
    contract.validate();
    shape.validate();
    boundary.validate();
  }

  void serialize_exact(ExactContractBuilder& exact) const {
    validate();
    exact.bytes(key.exact_key());
    contract.serialize_exact(exact);
    shape.serialize_exact(exact);
    boundary.serialize_exact(exact);
  }
};

/// A consumer requirement.  It carries the complete expected contract, so a same-named component
/// with a different representation, layout, unit, centering, rank, or halo is refused before any
/// candidate storage is touched.
template <int Dim>
struct AuxiliaryDependency {
  AuxiliaryComponentKey key;
  AuxiliaryComponentContract contract;
  AuxiliaryStorageShape<Dim> shape;

  void validate() const {
    key.validate();
    contract.validate();
    shape.validate();
  }

  void serialize_exact(ExactContractBuilder& exact) const {
    validate();
    exact.bytes(key.exact_key());
    contract.serialize_exact(exact);
    shape.serialize_exact(exact);
  }
};

/// One value consumed by a native block/operator.  ``consumer_slot`` is local to that consumer and
/// deliberately independent from the globally compact carrier slot: two consumers can pack the
/// same producer value at different device-friendly positions without aliasing global storage.
template <int Dim>
struct AuxiliaryConsumerValue {
  AuxiliaryDependency<Dim> dependency;
  std::size_t consumer_slot = 0;

  void validate() const { dependency.validate(); }

  void serialize_exact(ExactContractBuilder& exact) const {
    dependency.serialize_exact(exact);
    exact.scalar(static_cast<std::uint64_t>(consumer_slot));
  }
};

/// Canonical provider-value image consumed by one native component.  It contains no model formula
/// and no storage pointer.  The registry resolves each dependency to a global storage component at
/// seal; the generated kernel then gathers the listed values into ``ProviderValues<N>``.
template <int Dim>
struct AuxiliaryConsumerProviderPlan {
  std::string consumer_qid;
  std::vector<AuxiliaryConsumerValue<Dim>> values;

  void validate() const {
    if (consumer_qid.empty())
      throw std::invalid_argument("auxiliary consumer plan requires a non-empty consumer identity");
    for (const auto& value : values)
      value.validate();
  }

  void serialize_exact(ExactContractBuilder& exact) const {
    validate();
    exact.text("pops.auxiliary-consumer-provider-plan")
        .scalar(std::uint32_t{1})
        .text(consumer_qid);
    exact.sequence(values, [](ExactContractBuilder& item,
                              const AuxiliaryConsumerValue<Dim>& value) {
      value.serialize_exact(item);
    });
  }
};

/// Integer-identical point supplied by Program/AMR.  Physical time is intentionally absent: the
/// time authority owns it, while auxiliary freshness needs an unambiguous accepted-step/stage
/// identity that survives checkpoint/restart and MPI rank ordering.
struct AuxiliaryEvaluationPoint {
  std::string clock;
  std::uint64_t accepted_step = 0;
  std::uint64_t layout_generation = 0;
  int level = 0;
  int substep = 0;
  int stage = 0;
  int nonlinear_iteration = 0;
  AuxiliaryEvaluationEvent event = AuxiliaryEvaluationEvent::initialization;

  friend bool operator==(const AuxiliaryEvaluationPoint&,
                         const AuxiliaryEvaluationPoint&) = default;

  void validate() const {
    if (clock.empty() || level < 0 || substep < 0 || stage < 0 || nonlinear_iteration < 0)
      throw std::invalid_argument(
          "auxiliary evaluation point requires a clock and non-negative level/substep/stage/iteration");
  }

  void serialize_exact(ExactContractBuilder& exact) const {
    validate();
    exact.text("pops.auxiliary-evaluation-point")
        .scalar(std::uint32_t{2})
        .text(clock)
        .scalar(accepted_step)
        .scalar(layout_generation)
        .scalar(level)
        .scalar(substep)
        .scalar(stage)
        .scalar(nonlinear_iteration)
        .scalar(event);
  }
};

/// Exact event and freshness selection for one provider.
struct AuxiliaryEvaluationPolicy {
  std::vector<AuxiliaryEvaluationEvent> allowed_events{AuxiliaryEvaluationEvent::initialization};
  AuxiliaryFreshness freshness = AuxiliaryFreshness::once;

  AuxiliaryEvaluationPolicy() = default;
  AuxiliaryEvaluationPolicy(AuxiliaryEvaluationEvent event, AuxiliaryFreshness freshness_value)
      : allowed_events{event}, freshness(freshness_value) {}
  AuxiliaryEvaluationPolicy(std::vector<AuxiliaryEvaluationEvent> events,
                            AuxiliaryFreshness freshness_value)
      : allowed_events(std::move(events)), freshness(freshness_value) {
    validate();
  }

  void validate() const {
    if (allowed_events.empty())
      throw std::invalid_argument("auxiliary evaluation policy requires an allowed event");
    for (const auto event : allowed_events)
      if (event != AuxiliaryEvaluationEvent::initialization &&
          event != AuxiliaryEvaluationEvent::before_residual &&
          event != AuxiliaryEvaluationEvent::before_field_solve &&
          event != AuxiliaryEvaluationEvent::nonlinear_iteration &&
          event != AuxiliaryEvaluationEvent::after_regrid && event != AuxiliaryEvaluationEvent::output)
        throw std::invalid_argument("auxiliary evaluation policy has an invalid event");
  }

  [[nodiscard]] bool allows(AuxiliaryEvaluationEvent event) const {
    validate();
    return std::find(allowed_events.begin(), allowed_events.end(), event) != allowed_events.end();
  }

  [[nodiscard]] bool requires_evaluation(const std::optional<AuxiliaryEvaluationPoint>& accepted,
                                         const AuxiliaryEvaluationPoint& requested) const {
    requested.validate();
    if (!allows(requested.event))
      return false;
    if (!accepted)
      return true;
    switch (freshness) {
      case AuxiliaryFreshness::once:
        return false;
      case AuxiliaryFreshness::accepted_step:
        return accepted->accepted_step != requested.accepted_step;
      case AuxiliaryFreshness::evaluation:
        return *accepted != requested;
      case AuxiliaryFreshness::layout_generation:
        return accepted->layout_generation != requested.layout_generation;
    }
    throw std::invalid_argument("auxiliary freshness policy is invalid");
  }

  void serialize_exact(ExactContractBuilder& exact) const {
    validate();
    std::vector<AuxiliaryEvaluationEvent> canonical = allowed_events;
    std::sort(canonical.begin(), canonical.end(), [](const auto left, const auto right) {
      return static_cast<std::uint8_t>(left) < static_cast<std::uint8_t>(right);
    });
    if (std::adjacent_find(canonical.begin(), canonical.end()) != canonical.end())
      throw std::invalid_argument("auxiliary evaluation policy duplicates an allowed event");
    exact.text("pops.auxiliary-evaluation-policy").scalar(std::uint32_t{2});
    exact.sequence(canonical);
    exact.scalar(freshness);
  }
};

/// Exact storage class shared only by components with a compatible physical layout.  A registry
/// assigns dense component indices *inside* this group; no process-wide scalar component exists.
template <int Dim>
struct AuxiliaryStorageGroupKey {
  std::string representation;
  std::string centering;
  std::string layout;
  AuxiliaryStorageShape<Dim> shape;

  void validate() const {
    if (representation.empty() || centering.empty() || layout.empty())
      throw std::invalid_argument("auxiliary storage group requires storage representation/centering/layout");
    shape.validate();
  }
  [[nodiscard]] std::string exact_key() const {
    ExactContractBuilder exact;
    exact.text("pops.auxiliary-storage-group").scalar(std::uint32_t{1});
    exact.text(representation).text(centering).text(layout);
    shape.serialize_exact(exact);
    return std::move(exact).release();
  }
};

template <int Dim>
struct AuxiliaryStorageAddress {
  std::string group;
  std::size_t component = 0;
};

template <int Dim>
struct ResolvedAuxiliaryStorageGroup {
  std::string identity;
  AuxiliaryComponentContract contract;
  AuxiliaryStorageShape<Dim> shape;
  std::size_t component_count = 0;
};

/// Compact resolved address passed to native launchers.  The address has already been checked for
/// ownership, representation, centering, layout, rank and halo by ExactAuxiliaryRegistry.
template <int Dim>
struct AuxiliaryResolvedSlot {
  AuxiliaryStorageAddress<Dim> address;
  AuxiliaryComponentKey key;
  AuxiliaryComponentContract contract;
  AuxiliaryStorageShape<Dim> shape;
};

/// Registry-resolved row supplied to a prepared consumer.  ``storage_group/component`` locates the
/// provider value without assuming a common centering; ``consumer_slot`` is a dense local index.
template <int Dim>
struct ResolvedAuxiliaryConsumerValue {
  AuxiliaryComponentKey key;
  AuxiliaryComponentContract contract;
  AuxiliaryStorageShape<Dim> shape;
  AuxiliaryStorageAddress<Dim> address;
  std::size_t consumer_slot = 0;
};

template <int Dim>
struct ResolvedAuxiliaryConsumerPlan {
  std::string consumer_qid;
  std::vector<ResolvedAuxiliaryConsumerValue<Dim>> values;

  [[nodiscard]] std::size_t value_count() const noexcept { return values.size(); }
};

/// Runtime-owned storage exposed to a prepared native provider for one publication transaction.
/// ``accepted`` is the last globally accepted carrier; ``candidate`` starts as an exact copy of it
/// and is the only mutable destination.  A provider may read already-staged dependencies from the
/// candidate, including dependencies produced earlier in the same topological transaction.  The
/// registry never owns either allocation and a provider never receives a Python callback or a raw
/// cell pointer.
template <int Dim>
struct AuxiliaryStorageGroups {
  std::map<std::string, MultiFab<Dim>, std::less<>> groups;

  [[nodiscard]] const MultiFab<Dim>* find(std::string_view group) const {
    const auto found = groups.find(group);
    return found == groups.end() ? nullptr : &found->second;
  }
  [[nodiscard]] MultiFab<Dim>* find(std::string_view group) {
    const auto found = groups.find(group);
    return found == groups.end() ? nullptr : &found->second;
  }
};

template <int Dim>
struct AuxiliaryCarrierStorage {
  const AuxiliaryStorageGroups<Dim>* accepted = nullptr;
  AuxiliaryStorageGroups<Dim>* candidate = nullptr;

  void validate() const {
    if (accepted == nullptr || candidate == nullptr)
      throw std::invalid_argument(
          "auxiliary native launch requires accepted and candidate carrier storage");
    if (accepted->groups.size() != candidate->groups.size())
      throw std::invalid_argument(
          "auxiliary accepted and candidate carriers must have identical storage groups");
    for (const auto& [identity, accepted_group] : accepted->groups) {
      const auto candidate_group = candidate->groups.find(identity);
      if (candidate_group == candidate->groups.end() ||
          accepted_group.layout() != candidate_group->second.layout() ||
          accepted_group.distribution() != candidate_group->second.distribution() ||
          accepted_group.local_rank() != candidate_group->second.local_rank() ||
          accepted_group.local_size() != candidate_group->second.local_size() ||
          accepted_group.ncomp() != candidate_group->second.ncomp() ||
          accepted_group.ghosts() != candidate_group->second.ghosts())
        throw std::invalid_argument(
            "auxiliary accepted and candidate storage groups differ in exact ranked layout");
    }
  }
};

/// Host launch description for a generated/native provider.  It deliberately carries no Python
/// callback and no cell accessor: the implementation launches its own prepared Kokkos kernel over
/// the supplied candidate storage owned by the integrating runtime.
template <int Dim>
struct AuxiliaryKernelLaunchContext {
  const AuxiliaryEvaluationPoint& point;
  std::uint64_t candidate_generation = 0;
  const std::vector<AuxiliaryResolvedSlot<Dim>>& outputs;
  const std::vector<AuxiliaryResolvedSlot<Dim>>& dependencies;
  AuxiliaryCarrierStorage<Dim> storage{};
};

/// Immutable provider package.  A derived route must own a typed PreparedProvider launcher;
/// input and field-output routes are externally staged by their corresponding native subsystems.
/// The generic registry accepts no untyped lambda and never dispatches on a physics name.
template <int Dim>
class PreparedAuxiliaryProvider final {
 public:
  using launcher_type = PreparedProvider<void(const AuxiliaryKernelLaunchContext<Dim>&)>;

  PreparedAuxiliaryProvider(std::string identity, AuxiliaryProviderKind kind,
                            AuxiliaryEvaluationPolicy policy,
                            std::vector<AuxiliaryOutput<Dim>> outputs,
                            std::vector<AuxiliaryDependency<Dim>> dependencies,
                            std::optional<launcher_type> launcher = std::nullopt)
      : identity_(std::move(identity)),
        kind_(kind),
        policy_(policy),
        outputs_(std::move(outputs)),
        dependencies_(std::move(dependencies)),
        launcher_(std::move(launcher)) {
    validate_();
    ExactContractBuilder exact;
    exact.text("pops.prepared-auxiliary-provider")
        .scalar(std::uint32_t{1})
        .text(identity_)
        .scalar(kind_);
    policy_.serialize_exact(exact);
    exact.sequence(outputs_, [](ExactContractBuilder& item, const AuxiliaryOutput<Dim>& output) {
      output.serialize_exact(item);
    });
    exact.sequence(dependencies_,
                   [](ExactContractBuilder& item, const AuxiliaryDependency<Dim>& dependency) {
                     dependency.serialize_exact(item);
                   });
    exact.presence(launcher_.has_value());
    if (launcher_)
      exact.bytes(launcher_->collective_contract());
    collective_contract_ = std::move(exact).release();
  }

  [[nodiscard]] const std::string& identity() const noexcept { return identity_; }
  [[nodiscard]] AuxiliaryProviderKind kind() const noexcept { return kind_; }
  [[nodiscard]] const AuxiliaryEvaluationPolicy& policy() const noexcept { return policy_; }
  [[nodiscard]] const std::vector<AuxiliaryOutput<Dim>>& outputs() const noexcept {
    return outputs_;
  }
  [[nodiscard]] const std::vector<AuxiliaryDependency<Dim>>& dependencies() const noexcept {
    return dependencies_;
  }
  [[nodiscard]] bool has_launcher() const noexcept { return launcher_.has_value(); }
  [[nodiscard]] std::string_view collective_contract() const noexcept {
    return collective_contract_;
  }

  void launch(const AuxiliaryKernelLaunchContext<Dim>& context) const {
    if (!launcher_)
      throw std::logic_error("auxiliary provider has no native launcher");
    (*launcher_)(context);
  }

 private:
  void validate_() const {
    if (identity_.empty())
      throw std::invalid_argument("auxiliary provider identity must not be empty");
    if (outputs_.empty())
      throw std::invalid_argument("auxiliary provider requires at least one output");
    for (const auto& output : outputs_)
      output.validate();
    for (const auto& dependency : dependencies_)
      dependency.validate();
    if (kind_ == AuxiliaryProviderKind::derived && !launcher_)
      throw std::invalid_argument("derived auxiliary provider requires a native launcher");
    if (kind_ != AuxiliaryProviderKind::derived && launcher_)
      throw std::invalid_argument(
          "input and field-output auxiliary providers are externally staged, not host-launched");
  }

  std::string identity_;
  AuxiliaryProviderKind kind_ = AuxiliaryProviderKind::input;
  AuxiliaryEvaluationPolicy policy_;
  std::vector<AuxiliaryOutput<Dim>> outputs_;
  std::vector<AuxiliaryDependency<Dim>> dependencies_;
  std::optional<launcher_type> launcher_;
  std::string collective_contract_;
};

}  // namespace pops::runtime::system
