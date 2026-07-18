#pragma once

/// @file
/// @brief Provider-neutral preparation protocol for elliptic nullspaces and gauges.

#include <pops/core/identity/prepared_provider.hpp>
#include <pops/core/identity/prepared_provider_options.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {

/// What one resolved boundary entity proves about the spatially constant mode. This is an operator
/// fact, not a boundary-condition family: adapters may derive it from builtin BCs or obtain it from
/// an authenticated external boundary provider.
enum class FieldBoundaryNullspaceBehavior : std::uint8_t {
  Opaque = 0,
  PreservesConstantMode = 1,
  ConstrainsConstantMode = 2,
};

struct FieldBoundaryNullspaceFact {
  std::string boundary_id;
  FieldBoundaryNullspaceBehavior behavior = FieldBoundaryNullspaceBehavior::Opaque;
};

inline bool field_boundary_nullspace_behavior_is_valid(
    FieldBoundaryNullspaceBehavior behavior) noexcept {
  switch (behavior) {
    case FieldBoundaryNullspaceBehavior::Opaque:
    case FieldBoundaryNullspaceBehavior::PreservesConstantMode:
    case FieldBoundaryNullspaceBehavior::ConstrainsConstantMode:
      return true;
  }
  return false;
}

inline bool field_boundary_id_bytewise_less(std::string_view left,
                                            std::string_view right) noexcept {
  return std::lexicographical_compare(
      left.begin(), left.end(), right.begin(), right.end(), [](char lhs, char rhs) noexcept {
        return static_cast<unsigned char>(lhs) < static_cast<unsigned char>(rhs);
      });
}

struct FieldNullspaceOperatorFacts {
  bool has_reaction = false;
  bool has_coercive_internal_constraint = false;
  /// Opaque identity of the complete resolved boundary set.  A non-empty identity distinguishes a
  /// deliberately boundaryless topology from a default-constructed, unauthenticated request.
  std::string boundary_set_identity;
  /// Canonical bytewise order by boundary_id.  The provider core interprets only behavior; IDs and
  /// cardinality remain owned by the boundary topology provider.
  std::vector<FieldBoundaryNullspaceFact> boundaries;

  [[nodiscard]] bool is_canonical() const noexcept {
    if (boundary_set_identity.empty())
      return false;
    for (std::size_t index = 0; index < boundaries.size(); ++index) {
      const FieldBoundaryNullspaceFact& current = boundaries[index];
      if (current.boundary_id.empty() ||
          !field_boundary_nullspace_behavior_is_valid(current.behavior))
        return false;
      if (index != 0 &&
          !field_boundary_id_bytewise_less(boundaries[index - 1].boundary_id,
                                           current.boundary_id))
        return false;
    }
    return true;
  }

  [[nodiscard]] std::string exact_contract() const {
    if (!is_canonical())
      throw std::invalid_argument(
          "field-nullspace operator facts require a canonical identified boundary sequence");
    ExactContractBuilder contract;
    contract.text("pops.field-nullspace.operator-facts")
        .scalar(std::uint32_t{2})
        .scalar(has_reaction)
        .scalar(has_coercive_internal_constraint)
        .text(boundary_set_identity)
        .sequence(boundaries,
                  [](ExactContractBuilder& item, const FieldBoundaryNullspaceFact& boundary) {
                    item.text(boundary.boundary_id).scalar(boundary.behavior);
                  });
    return std::move(contract).release();
  }
};

/// Canonicalize boundary-provider facts once at preparation/topology-change time.  Solver and
/// workspace hot paths retain the resulting immutable facts and perform no sorting or allocation.
inline FieldNullspaceOperatorFacts make_field_nullspace_operator_facts(
    std::string boundary_set_identity, std::vector<FieldBoundaryNullspaceFact> boundaries,
    bool has_reaction, bool internal_constraint = false) {
  std::sort(boundaries.begin(), boundaries.end(),
            [](const FieldBoundaryNullspaceFact& left,
               const FieldBoundaryNullspaceFact& right) {
              return field_boundary_id_bytewise_less(left.boundary_id, right.boundary_id);
            });
  FieldNullspaceOperatorFacts facts{has_reaction, internal_constraint,
                                    std::move(boundary_set_identity), std::move(boundaries)};
  if (!facts.is_canonical())
    throw std::invalid_argument(
        "field-nullspace operator facts require non-empty unique boundary identities and valid "
        "behaviors");
  return facts;
}

/// Materialized topology facts for exactly one solver vector space. Borrowed layouts remain owned
/// by the caller; label and coverage fields are shared immutable prepared resources.
struct FieldNullspaceTopologyFacts {
  std::string identity;
  std::string exact_layout_contract;
  int first_level = 0;
  int field_component = 0;
  /// Exact proof that the active topology has one connected component. Empty means no such proof;
  /// providers that need a constant connected-mode basis must then consume an explicit component
  /// partition instead. This opaque contract avoids a closed connectivity enum.
  std::string connected_component_contract;
  std::vector<const MultiFab*> layouts;
  std::vector<std::shared_ptr<const MultiFab>> component_labels;
  std::vector<std::string> component_label_contracts;
  std::vector<FieldConnectedComponent> connected_components;
  std::vector<std::shared_ptr<const MultiFab>> coverage;
  std::vector<std::string> coverage_contracts;
  std::vector<Real> cell_measure;
  /// One vector-distribution provider per active layout, in the same order. Storage families are
  /// not a closed capability vocabulary; each provider authenticates layouts and reductions
  /// through its own exact contract.
  std::vector<PreparedVectorDistribution> level_distributions;
};

struct FieldNullspaceProviderRequest {
  std::string plan_identity;
  FieldNullspaceOperatorFacts operator_facts;
  FieldNullspaceTopologyFacts topology;
  PreparedProviderOptions options;
};

struct FieldNullspaceProviderSelection {
  std::string provider_identity;
  PreparedProviderOptions options;

  [[nodiscard]] std::string exact_contract() const {
    if (provider_identity.empty())
      throw std::invalid_argument("field-nullspace selection requires a provider identity");
    ExactContractBuilder contract;
    contract.text("pops.field-nullspace.provider-selection")
        .scalar(std::uint32_t{1})
        .text(provider_identity)
        .bytes(options.exact_contract());
    return std::move(contract).release();
  }
};

struct PreparedFieldNullspace {
  std::string provider_identity;
  std::uint64_t provider_version = 0;
  std::string exact_prepared_contract;
  FieldNullspacePlan plan;
};

class FieldNullspaceProvider {
 public:
  virtual ~FieldNullspaceProvider() = default;
  [[nodiscard]] virtual std::string_view identity() const noexcept = 0;
  [[nodiscard]] virtual std::uint64_t interface_version() const noexcept = 0;
  [[nodiscard]] virtual std::string_view collective_contract() const noexcept = 0;
  [[nodiscard]] virtual PreparedProviderOptions default_options() const = 0;
  [[nodiscard]] virtual bool accepts_options(
      const PreparedProviderOptions& options) const noexcept = 0;
  [[nodiscard]] virtual PreparedProviderSupport supports(
      const FieldNullspaceProviderRequest& request) const noexcept = 0;
  [[nodiscard]] virtual std::string expected_prepared_contract(
      const FieldNullspaceProviderRequest& request) const = 0;
  [[nodiscard]] virtual PreparedFieldNullspace prepare(
      const FieldNullspaceProviderRequest& request) const = 0;
};

class FieldNullspaceProviderRegistry {
 public:
  void add(std::shared_ptr<const FieldNullspaceProvider> provider) {
    if (!provider || provider->identity().empty() || provider->interface_version() == 0 ||
        provider->collective_contract().empty())
      throw std::invalid_argument("field-nullspace provider requires exact identities");
    const std::string identity(provider->identity());
    if (!providers_.emplace(identity, std::move(provider)).second)
      throw std::invalid_argument("duplicate field-nullspace provider identity '" + identity +
                                  "'");
  }

  [[nodiscard]] std::shared_ptr<const FieldNullspaceProvider> resolve(
      std::string_view identity) const {
    const auto found = providers_.find(std::string(identity));
    if (found == providers_.end())
      throw std::invalid_argument("unknown field-nullspace provider '" + std::string(identity) +
                                  "'");
    return found->second;
  }

 private:
  std::map<std::string, std::shared_ptr<const FieldNullspaceProvider>> providers_;
};

}  // namespace pops
