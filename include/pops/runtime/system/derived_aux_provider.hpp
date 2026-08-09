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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::system {

/// Storage namespace containing one component.  The namespace is structural, not a catalogue of
/// physics: packages may use their own owners and spaces without extending this enum.
enum class AuxiliarySpaceKind : std::uint8_t {
  state = 0,
  auxiliary = 1,
  field = 2,
  external = 3,
};

/// Location of a component on an exact-rank mesh.
enum class AuxiliaryCentering : std::uint8_t {
  cell = 0,
  face = 1,
  edge = 2,
  node = 3,
};

/// Algebraic shape of one published component group.  The component key still addresses one
/// compact carrier slot; @c value_components records the provider-visible group width.
enum class AuxiliaryValueKind : std::uint8_t {
  scalar = 0,
  vector = 1,
  tensor = 2,
  opaque = 3,
};

/// Producer class.  Input data is supplied by an external native uploader, derived data by a
/// prepared native launcher, and field output by a separately prepared field solver.  None of
/// these values implies a physical formula.
enum class AuxiliaryProviderKind : std::uint8_t {
  input = 0,
  derived = 1,
  field_output = 2,
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

[[nodiscard]] inline std::string_view auxiliary_space_kind_name(AuxiliarySpaceKind value) {
  switch (value) {
    case AuxiliarySpaceKind::state:
      return "state";
    case AuxiliarySpaceKind::auxiliary:
      return "auxiliary";
    case AuxiliarySpaceKind::field:
      return "field";
    case AuxiliarySpaceKind::external:
      return "external";
  }
  throw std::invalid_argument("auxiliary space kind is invalid");
}

/// Stable address of one component.  The owner prevents otherwise-valid packages from silently
/// aliasing each other; (space_kind, space_name, component) identifies the carrier within it.
struct AuxiliaryComponentKey {
  std::string owner;
  AuxiliarySpaceKind space_kind = AuxiliarySpaceKind::auxiliary;
  std::string space_name;
  int component = -1;

  friend bool operator==(const AuxiliaryComponentKey&, const AuxiliaryComponentKey&) = default;

  void validate() const {
    if (owner.empty() || space_name.empty() || component < 0)
      throw std::invalid_argument(
          "auxiliary component key requires non-empty owner/space name and non-negative component");
  }

  [[nodiscard]] std::string exact_key() const {
    validate();
    ExactContractBuilder contract;
    contract.text("pops.auxiliary-component-key")
        .scalar(std::uint32_t{1})
        .text(owner)
        .scalar(space_kind)
        .text(space_name)
        .scalar(component);
    return std::move(contract).release();
  }

  void serialize_exact(ExactContractBuilder& contract) const { contract.bytes(exact_key()); }
};

/// Full carrier contract.  Text fields are deliberately package-defined; the registry compares
/// their exact bytes and therefore cannot accidentally reinterpret units, layouts, or a model's
/// representation.  @c spatial_rank must equal the compile-time native rank.
template <int Dim>
struct AuxiliaryComponentContract {
  static_assert(Dim >= 1 && Dim <= 3, "AuxiliaryComponentContract supports dimensions 1, 2, and 3");

  std::string representation;
  AuxiliaryCentering centering = AuxiliaryCentering::cell;
  std::string unit;
  std::string layout;
  AuxiliaryValueKind value_kind = AuxiliaryValueKind::scalar;
  int value_components = 1;
  int spatial_rank = Dim;
  Index<Dim> halo{};

  friend bool operator==(const AuxiliaryComponentContract&,
                         const AuxiliaryComponentContract&) = default;

  void validate() const {
    if (representation.empty() || unit.empty() || layout.empty())
      throw std::invalid_argument(
          "auxiliary component contract requires representation, unit, and layout");
    if (value_components < 1)
      throw std::invalid_argument("auxiliary component contract requires positive component width");
    if (spatial_rank != Dim)
      throw std::invalid_argument(
          "auxiliary component contract rank differs from native dimension");
    for (int axis = 0; axis < Dim; ++axis)
      if (halo[axis] < 0)
        throw std::invalid_argument("auxiliary component halo extent must be non-negative");
  }

  void serialize_exact(ExactContractBuilder& contract) const {
    validate();
    contract.text("pops.auxiliary-component-contract")
        .scalar(std::uint32_t{1})
        .text(representation)
        .scalar(centering)
        .text(unit)
        .text(layout)
        .scalar(value_kind)
        .scalar(value_components)
        .scalar(spatial_rank);
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(halo[axis]);
  }
};

/// One explicitly requested compact output slot.  Slots are validated collectively by the
/// registry: every slot in [0, N) must occur exactly once, with no inferred/reserved offsets.
template <int Dim>
struct AuxiliaryOutput {
  AuxiliaryComponentKey key;
  AuxiliaryComponentContract<Dim> contract;
  std::size_t slot = 0;

  void validate() const {
    key.validate();
    contract.validate();
  }

  void serialize_exact(ExactContractBuilder& exact) const {
    validate();
    exact.bytes(key.exact_key());
    contract.serialize_exact(exact);
    exact.scalar(static_cast<std::uint64_t>(slot));
  }
};

/// A consumer requirement.  It carries the complete expected contract, so a same-named component
/// with a different representation, layout, unit, centering, rank, or halo is refused before any
/// candidate storage is touched.
template <int Dim>
struct AuxiliaryDependency {
  AuxiliaryComponentKey key;
  AuxiliaryComponentContract<Dim> contract;

  void validate() const {
    key.validate();
    contract.validate();
  }

  void serialize_exact(ExactContractBuilder& exact) const {
    validate();
    exact.bytes(key.exact_key());
    contract.serialize_exact(exact);
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
  AuxiliaryEvaluationEvent event = AuxiliaryEvaluationEvent::initialization;

  friend bool operator==(const AuxiliaryEvaluationPoint&,
                         const AuxiliaryEvaluationPoint&) = default;

  void validate() const {
    if (clock.empty() || level < 0 || substep < 0 || stage < 0)
      throw std::invalid_argument(
          "auxiliary evaluation point requires a clock and non-negative level/substep/stage");
  }

  void serialize_exact(ExactContractBuilder& exact) const {
    validate();
    exact.text("pops.auxiliary-evaluation-point")
        .scalar(std::uint32_t{1})
        .text(clock)
        .scalar(accepted_step)
        .scalar(layout_generation)
        .scalar(level)
        .scalar(substep)
        .scalar(stage)
        .scalar(event);
  }
};

/// Exact event and freshness selection for one provider.
struct AuxiliaryEvaluationPolicy {
  AuxiliaryEvaluationEvent event = AuxiliaryEvaluationEvent::initialization;
  AuxiliaryFreshness freshness = AuxiliaryFreshness::once;

  [[nodiscard]] bool requires_evaluation(const std::optional<AuxiliaryEvaluationPoint>& accepted,
                                         const AuxiliaryEvaluationPoint& requested) const {
    requested.validate();
    if (requested.event != event)
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
    exact.text("pops.auxiliary-evaluation-policy")
        .scalar(std::uint32_t{1})
        .scalar(event)
        .scalar(freshness);
  }
};

/// Compact resolved slot passed to native launchers.  The slot has already been checked for
/// density, uniqueness, rank, contract, halo and dependency ownership by ExactAuxiliaryRegistry.
template <int Dim>
struct AuxiliaryResolvedSlot {
  std::size_t slot = 0;
  AuxiliaryComponentKey key;
  AuxiliaryComponentContract<Dim> contract;
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
