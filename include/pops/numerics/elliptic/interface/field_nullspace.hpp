#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/numerics/elliptic/linear/vector_distribution.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

/// One basis vector of an elliptic nullspace.  An empty @c masks vector denotes the constant
/// function on the selected field component.  Otherwise masks[level] is the already-resolved,
/// co-distributed basis field for that hierarchy level.  The solver owns no registry lookup: the
/// complete basis (including readable provenance) is attached to its install plan before a solve.
struct FieldNullspaceBasis {
  std::string identity;
  std::string provenance;
  std::string recipe_identity;
  int field_component = 0;
  std::vector<std::shared_ptr<const MultiFab>> masks;
  /// Optional quadrature coverage per absolute hierarchy level (1 on active cells, 0 on cells
  /// excluded from this basis' physical support). Empty means full coverage. Entries outside the
  /// prepared level range may be null placeholders; an external provider is otherwise free to
  /// define arbitrary, basis-specific overlapping or coupled hierarchy supports.
  std::vector<std::shared_ptr<const MultiFab>> coverage;
  /// Physical cell measure per level.  It is part of the resolved topology/layout recipe and is
  /// applied to every compatibility/gauge moment.
  std::vector<Real> cell_measure;

  const MultiFab* mask(int level) const {
    if (masks.empty())
      return nullptr;
    if (level < 0)
      throw std::runtime_error("field nullspace basis is missing a resolved hierarchy mask");
    const std::size_t index = static_cast<std::size_t>(level);
    if (index >= masks.size() || !masks[index])
      throw std::runtime_error("field nullspace basis is missing a resolved hierarchy mask");
    return masks[index].get();
  }

  const MultiFab* coverage_mask(int level) const {
    if (coverage.empty())
      return nullptr;
    if (level < 0)
      throw std::runtime_error("field nullspace basis has an invalid coverage level");
    const std::size_t index = static_cast<std::size_t>(level);
    if (index >= coverage.size() || !coverage[index])
      return nullptr;
    return coverage[index].get();
  }

  Real measure(int level) const {
    if (level < 0)
      throw std::runtime_error("field nullspace basis is missing a positive cell measure");
    const std::size_t index = static_cast<std::size_t>(level);
    if (index >= cell_measure.size() || !(cell_measure[index] > Real(0)))
      throw std::runtime_error("field nullspace basis is missing a positive cell measure");
    return cell_measure[index];
  }
};

struct FieldGaugeConstraint {
  std::string basis_identity;
  Real value = Real(0);
};

/// A nullspace and a representative-selection policy are deliberately separate.  Compatibility
/// never projects the RHS.  A gauge may constrain every basis independently (for example one mean
/// per disconnected component); identities, rather than vector positions, authenticate the pairing.
struct FieldNullspacePlan {
  std::string identity;
  std::string layout_identity;
  std::vector<FieldNullspaceBasis> bases;
  std::vector<FieldGaugeConstraint> gauges;

  bool empty() const { return bases.empty(); }

};

/// Scientific incompatibility is distinct from a malformed nullspace contract and from a
/// non-finite evaluation. Prepared solvers catch only this type when publishing
/// SolveStatus::kIncompatibleRhs; ordinary plan/layout errors remain programming errors.
class FieldNullspaceIncompatibleRhs : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// The compatibility reduction completed but produced a non-finite scientific witness.
class FieldNullspaceInvalidEvaluation : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/// Stable identity for one connected material component supplied by a prepared topology provider.
/// Labels are positive integers in the co-distributed label fields passed to
/// labelled_mean_zero_nullspace(); zero denotes an inactive cell.  The topology provider, not the
/// field solver, owns how those labels were derived (structured grid, embedded boundary, external
/// mesh, ...).  This keeps the nullspace layer independent of concrete geometry classes.
struct FieldConnectedComponent {
  int label = 0;
  std::string identity;
  std::string provenance;
};

namespace detail {

inline constexpr std::size_t kFieldNullspaceCollectiveCapacity =
    static_cast<std::size_t>(std::numeric_limits<int>::max());

/// Return an exact contiguous reduction length without ever evaluating an overflowing product.
/// Field-nullspace reductions use the native int-count collective API, so its count bound is
/// stricter than size_t on supported 64-bit hosts and is also the allocation bound for these
/// contiguous buffers.
inline std::size_t checked_field_nullspace_collective_product(std::size_t left, std::size_t right,
                                                              const char* quantity) {
  if (left != 0 && right != 0 && left > kFieldNullspaceCollectiveCapacity / right)
    throw std::overflow_error(std::string(quantity) +
                              " exceeds the native MPI collective count capacity");
  return left * right;
}

inline std::size_t checked_field_nullspace_collective_sum(std::size_t value, std::size_t increment,
                                                          const char* quantity) {
  if (increment > kFieldNullspaceCollectiveCapacity ||
      value > kFieldNullspaceCollectiveCapacity - increment)
    throw std::overflow_error(std::string(quantity) +
                              " exceeds the native MPI collective count capacity");
  return value + increment;
}

inline int checked_field_nullspace_collective_count(std::size_t count, const char* quantity) {
  if (count > kFieldNullspaceCollectiveCapacity)
    throw std::overflow_error(std::string(quantity) +
                              " exceeds the native MPI collective count capacity");
  return static_cast<int>(count);
}

/// Prove that every `first_level + offset` represented by a hierarchy vector fits in int before
/// entering a loop or converting an offset.  INT_MAX itself is a valid resolved level.
inline void validate_field_nullspace_level_capacity(std::size_t level_count, int first_level,
                                                    const char* quantity) {
  if (first_level < 0)
    throw std::invalid_argument(std::string(quantity) +
                                " requires a non-negative first hierarchy level");
  const std::size_t available =
      kFieldNullspaceCollectiveCapacity - static_cast<std::size_t>(first_level) + 1;
  if (level_count > available)
    throw std::overflow_error(std::string(quantity) +
                              " exceeds the native hierarchy-level index capacity");
}

inline bool field_nullspace_level_capacity_is_valid(std::size_t level_count,
                                                    int first_level) noexcept {
  if (first_level < 0)
    return false;
  const std::size_t available =
      kFieldNullspaceCollectiveCapacity - static_cast<std::size_t>(first_level) + 1;
  return level_count <= available;
}

enum class FieldNullspaceCollectiveBoundary { BasisValidation, Compatibility, Gauge, Preparation };

inline const char* field_nullspace_boundary_name(FieldNullspaceCollectiveBoundary boundary) {
  switch (boundary) {
    case FieldNullspaceCollectiveBoundary::BasisValidation:
      return "basis validation";
    case FieldNullspaceCollectiveBoundary::Compatibility:
      return "compatibility";
    case FieldNullspaceCollectiveBoundary::Gauge:
      return "gauge";
    case FieldNullspaceCollectiveBoundary::Preparation:
      return "preparation";
  }
  return "unknown boundary";
}

/// Canonical native payload for one field-nullspace collective boundary.  It stores complete
/// metadata rather than a digest.  Ownership is typed by the resolved plan: distributed mappings
/// must agree exactly, while a replicated mapping must own every global box on the local rank and is
/// encoded rank-independently. Distributed field values never participate.
class FieldNullspacePreflightPayload {
 public:
  template <class Value>
  void append_scalar(const Value& value) {
    static_assert(std::is_trivially_copyable_v<Value>);
    bytes_.append(reinterpret_cast<const char*>(&value), sizeof(Value));
  }

  void append_size(std::size_t value) { append_scalar(static_cast<std::uint64_t>(value)); }

  void append_text(std::string_view value) {
    append_size(value.size());
    bytes_.append(value.data(), value.size());
  }

  /// Prepared single-vector path: the installed distribution provider, not the legacy two-value
  /// storage descriptor, owns every mask/coverage layout contract.
  template <class Distribution>
  void append_plan(const FieldNullspacePlan& plan, const Distribution& distribution) {
    append_plan_with_layout_(plan, [this, &distribution](const MultiFab* field, std::size_t) {
      append_layout(field, distribution);
    });
  }

  template <class Distribution>
  void append_layout(const MultiFab* field, const Distribution& distribution) {
    const std::uint8_t present = field == nullptr ? 0u : 1u;
    append_scalar(present);
    if (field == nullptr)
      return;
    append_text(distribution.collective_contract());
    bool matches = false;
    try {
      matches = distribution.layout_matches(*field);
      append_text(distribution.layout_contract(*field));
    } catch (...) {
      matches = false;
      append_text({});
    }
    require(matches);
  }

  void append_absent_layout() { append_scalar(std::uint8_t{0}); }

  void append_plan(const FieldNullspacePlan& plan, int first_level,
                   std::span<const PreparedVectorDistribution> distributions) {
    append_plan_with_layout_(
        plan, [this, first_level, distributions](const MultiFab* field, std::size_t level) {
          if (field == nullptr) {
            append_absent_layout();
            return;
          }
          const bool resolved = first_level >= 0 && level >= static_cast<std::size_t>(first_level) &&
                                level - static_cast<std::size_t>(first_level) <
                                    distributions.size();
          require(resolved);
          if (!resolved) {
            append_scalar(std::uint8_t{1});
            append_text({});
            return;
          }
          append_layout(field,
                        distributions[level - static_cast<std::size_t>(first_level)]);
        });
  }

  void require(bool condition) noexcept { valid_ = valid_ && condition; }
  bool valid() const noexcept { return valid_; }
  const std::string& bytes() const noexcept { return bytes_; }

 private:
  template <class AppendLayout>
  void append_plan_with_layout_(const FieldNullspacePlan& plan, AppendLayout&& append_layout_fn) {
    append_text(plan.identity);
    append_text(plan.layout_identity);
    append_size(plan.bases.size());
    for (const FieldNullspaceBasis& basis : plan.bases) {
      append_text(basis.identity);
      append_text(basis.provenance);
      append_text(basis.recipe_identity);
      append_scalar(basis.field_component);
      append_size(basis.masks.size());
      for (std::size_t level = 0; level < basis.masks.size(); ++level)
        append_layout_fn(basis.masks[level].get(), level);
      append_size(basis.coverage.size());
      for (std::size_t level = 0; level < basis.coverage.size(); ++level)
        append_layout_fn(basis.coverage[level].get(), level);
      append_size(basis.cell_measure.size());
      for (const Real measure : basis.cell_measure)
        append_scalar(measure);
    }
    append_size(plan.gauges.size());
    for (const FieldGaugeConstraint& gauge : plan.gauges) {
      append_text(gauge.basis_identity);
      append_scalar(gauge.value);
    }
  }

  std::string bytes_;
  bool valid_ = true;
};

inline bool field_nullspace_layout_is_materialized(const MultiFab& field) noexcept {
  return field.ncomp() > 0 && field.n_grow() >= 0 && !field.box_array().boxes().empty() &&
         field.box_array().boxes().size() == field.dmap().ranks().size();
}

inline bool field_nullspace_layouts_match(const MultiFab& left, const MultiFab& right) noexcept {
  return left.box_array().boxes() == right.box_array().boxes() &&
         left.dmap().ranks() == right.dmap().ranks();
}

inline std::size_t basis_index(const FieldNullspacePlan& plan, const std::string& identity) {
  for (std::size_t i = 0; i < plan.bases.size(); ++i)
    if (plan.bases[i].identity == identity)
      return i;
  return plan.bases.size();
}

inline void validate_field_nullspace_plan_locally(FieldNullspacePreflightPayload& payload,
                                                  const FieldNullspacePlan& plan) {
  payload.require(plan.bases.empty() ? plan.gauges.empty()
                                     : (!plan.identity.empty() && !plan.layout_identity.empty()));
  for (std::size_t index = 0; index < plan.bases.size(); ++index) {
    const FieldNullspaceBasis& basis = plan.bases[index];
    payload.require(!basis.identity.empty() && !basis.provenance.empty() &&
                    !basis.recipe_identity.empty() && basis.field_component >= 0);
    for (std::size_t previous = 0; previous < index; ++previous)
      payload.require(plan.bases[previous].identity != basis.identity);
    for (const auto& mask : basis.masks) {
      if (mask != nullptr)
        payload.require(field_nullspace_layout_is_materialized(*mask) && mask->ncomp() == 1);
    }
    for (const auto& coverage : basis.coverage) {
      if (coverage != nullptr)
        payload.require(field_nullspace_layout_is_materialized(*coverage) &&
                        coverage->ncomp() == 1);
    }
    // Providers address hierarchy levels by absolute index and may carry zero placeholders outside
    // the prepared range. The active range is required to be strictly positive below, once
    // first_level is known.
    for (const Real measure : basis.cell_measure) {
      payload.require(std::isfinite(static_cast<double>(measure)) && measure >= Real(0));
    }
  }
  for (std::size_t index = 0; index < plan.gauges.size(); ++index) {
    const FieldGaugeConstraint& gauge = plan.gauges[index];
    payload.require(!gauge.basis_identity.empty() &&
                    std::isfinite(static_cast<double>(gauge.value)) &&
                    basis_index(plan, gauge.basis_identity) != plan.bases.size());
    for (std::size_t previous = 0; previous < index; ++previous)
      payload.require(plan.gauges[previous].basis_identity != gauge.basis_identity);
  }
}

inline void finish_field_nullspace_preflight(FieldNullspacePreflightPayload& payload,
                                             FieldNullspaceCollectiveBoundary boundary) {
  const std::uint8_t valid = payload.valid() ? 1u : 0u;
  payload.append_scalar(valid);
  const std::string_view boundary_name = field_nullspace_boundary_name(boundary);
  const std::vector<std::pair<std::string_view, std::string_view>> collective_identity{
      {boundary_name, payload.bytes()}};
  const bool agreed = all_ranks_agree_exact_ordered_byte_pairs(collective_identity);
  if (!agreed || !payload.valid())
    throw std::runtime_error(std::string("field nullspace ") + std::string(boundary_name) +
                             " collective preflight rejected malformed local structure or "
                             "rank-divergent metadata");
}

template <class FieldVector>
inline void preflight_field_nullspace_fields(const FieldVector& fields,
                                             const FieldNullspacePlan& plan,
                                             std::span<const PreparedVectorDistribution> distributions,
                                             int first_level,
                                             FieldNullspaceCollectiveBoundary boundary) {
  FieldNullspacePreflightPayload payload;
  payload.append_scalar(first_level);
  payload.append_plan(plan, first_level, distributions);
  validate_field_nullspace_plan_locally(payload, plan);

  const bool active = boundary == FieldNullspaceCollectiveBoundary::Gauge ? !plan.gauges.empty()
                                                                          : !plan.bases.empty();
  // An empty nullspace performs no scientific field reduction, so its field layouts are outside
  // this collective contract. This also avoids forcing an irrelevant ownership declaration on an
  // invertible operator.
  payload.append_size(active ? fields.size() : 0u);
  const bool level_range_valid =
      field_nullspace_level_capacity_is_valid(fields.size(), first_level);
  if (active) {
    payload.require(distributions.size() == fields.size());
    for (std::size_t level = 0; level < fields.size(); ++level) {
      const bool resolved = level_range_valid && level < distributions.size();
      payload.require(resolved);
      if (resolved)
        payload.append_layout(fields[level], distributions[level]);
      else
        payload.append_absent_layout();
    }
  }
  if (active) {
    payload.require(!fields.empty());
    payload.require(field_nullspace_level_capacity_is_valid(fields.size(), first_level));
    if (boundary == FieldNullspaceCollectiveBoundary::Gauge)
      payload.require(plan.gauges.size() == plan.bases.size());
    try {
      if (boundary == FieldNullspaceCollectiveBoundary::BasisValidation)
        (void)checked_field_nullspace_collective_product(plan.bases.size(), plan.bases.size(),
                                                         "field nullspace Gram matrix");
      else
        (void)checked_field_nullspace_collective_product(plan.bases.size(), std::size_t{2},
                                                         "field nullspace moments");
    } catch (const std::exception&) {
      payload.require(false);
    }

    for (std::size_t level = 0; level < fields.size(); ++level) {
      const MultiFab* field = fields[level];
      payload.require(field != nullptr);
      if (field == nullptr)
        continue;
      payload.require(field_nullspace_layout_is_materialized(*field));
      if (!level_range_valid)
        continue;
      const std::size_t resolved_level = static_cast<std::size_t>(first_level) + level;
      for (const FieldNullspaceBasis& basis : plan.bases) {
        payload.require(basis.field_component >= 0 && basis.field_component < field->ncomp());
        if (!basis.masks.empty()) {
          payload.require(resolved_level < basis.masks.size() &&
                          basis.masks[resolved_level] != nullptr);
          if (resolved_level < basis.masks.size() && basis.masks[resolved_level] != nullptr)
            payload.require(field_nullspace_layouts_match(*field, *basis.masks[resolved_level]));
        }
        if (resolved_level < basis.coverage.size() && basis.coverage[resolved_level] != nullptr)
          payload.require(field_nullspace_layouts_match(*field, *basis.coverage[resolved_level]));
        payload.require(resolved_level < basis.cell_measure.size());
        if (resolved_level < basis.cell_measure.size()) {
          const Real measure = basis.cell_measure[resolved_level];
          payload.require(std::isfinite(static_cast<double>(measure)) && measure > Real(0));
        }
      }
      if (boundary == FieldNullspaceCollectiveBoundary::BasisValidation) {
        for (std::size_t left = 0; left < plan.bases.size(); ++left) {
          for (std::size_t right = left + 1; right < plan.bases.size(); ++right) {
            if (plan.bases[left].field_component != plan.bases[right].field_component ||
                resolved_level >= plan.bases[left].cell_measure.size() ||
                resolved_level >= plan.bases[right].cell_measure.size())
              continue;
            payload.require(plan.bases[left].cell_measure[resolved_level] ==
                            plan.bases[right].cell_measure[resolved_level]);
          }
        }
      }
    }
  }
  finish_field_nullspace_preflight(payload, boundary);

  if (!active)
    return;
  for (std::size_t level = 0; level < fields.size(); ++level) {
    const int resolved_level = first_level + static_cast<int>(level);
    const PreparedVectorDistribution& distribution = distributions[level];
    std::vector<char, comm_allocator<char>> validation_storage(
        distribution.validation_scratch_byte_count(), char{0});
    if (boundary == FieldNullspaceCollectiveBoundary::Compatibility ||
        boundary == FieldNullspaceCollectiveBoundary::Gauge)
      distribution.require_exact_values(*fields[level], validation_storage,
                                        "field nullspace solved field");
    for (const FieldNullspaceBasis& basis : plan.bases) {
      if (const MultiFab* mask = basis.mask(resolved_level); mask != nullptr)
        distribution.require_exact_values(*mask, validation_storage,
                                          "field nullspace basis mask");
      if (const MultiFab* coverage = basis.coverage_mask(resolved_level);
          coverage != nullptr)
        distribution.require_exact_values(*coverage, validation_storage,
                                          "field nullspace coverage mask");
    }
  }
}

inline void preflight_labelled_field_nullspace(
    std::string_view identity, std::string_view layout_identity,
    const std::vector<std::shared_ptr<const MultiFab>>& labels,
    const std::vector<FieldConnectedComponent>& components,
    const std::vector<std::shared_ptr<const MultiFab>>& coverage,
    const std::vector<Real>& cell_measure,
    std::span<const PreparedVectorDistribution> distributions, int field_component,
    int first_level) {
  FieldNullspacePreflightPayload payload;
  payload.append_text(identity);
  payload.append_text(layout_identity);
  payload.append_scalar(field_component);
  payload.append_scalar(first_level);
  payload.append_size(distributions.size());
  for (const PreparedVectorDistribution& distribution : distributions)
    payload.append_text(distribution.collective_contract());
  payload.append_size(labels.size());
  for (std::size_t level = 0; level < labels.size(); ++level) {
    const bool level_resolved = field_nullspace_level_capacity_is_valid(labels.size(), first_level);
    const bool distribution_resolved = level_resolved && level < distributions.size();
    payload.require(distribution_resolved);
    if (distribution_resolved)
      payload.append_layout(labels[level].get(), distributions[level]);
    else
      payload.append_absent_layout();
  }
  payload.append_size(components.size());
  for (const FieldConnectedComponent& component : components) {
    payload.append_scalar(component.label);
    payload.append_text(component.identity);
    payload.append_text(component.provenance);
  }
  payload.append_size(coverage.size());
  for (std::size_t level = 0; level < coverage.size(); ++level) {
    const bool level_resolved = field_nullspace_level_capacity_is_valid(labels.size(), first_level);
    const bool distribution_resolved = level_resolved && level < distributions.size();
    payload.require(distribution_resolved);
    if (distribution_resolved)
      payload.append_layout(coverage[level].get(), distributions[level]);
    else
      payload.append_absent_layout();
  }
  payload.append_size(cell_measure.size());
  for (const Real measure : cell_measure)
    payload.append_scalar(measure);

  payload.require(!identity.empty() && !layout_identity.empty());
  payload.require(!labels.empty() && !components.empty() && field_component >= 0);
  payload.require(field_nullspace_level_capacity_is_valid(labels.size(), first_level));
  payload.require(cell_measure.size() == labels.size());
  payload.require(distributions.size() == labels.size());
  payload.require(coverage.empty() || coverage.size() == labels.size());
  try {
    (void)checked_field_nullspace_collective_sum(components.size(), std::size_t{1},
                                                 "connected-component label counts");
  } catch (const std::exception&) {
    payload.require(false);
  }
  for (std::size_t level = 0; level < labels.size(); ++level) {
    payload.require(labels[level] != nullptr);
    if (labels[level] != nullptr)
      payload.require(field_nullspace_layout_is_materialized(*labels[level]) &&
                      labels[level]->ncomp() == 1);
    if (level < cell_measure.size())
      payload.require(std::isfinite(static_cast<double>(cell_measure[level])) &&
                      cell_measure[level] > Real(0));
    if (!coverage.empty() && level < coverage.size()) {
      payload.require(coverage[level] != nullptr);
      if (labels[level] != nullptr && coverage[level] != nullptr)
        payload.require(field_nullspace_layout_is_materialized(*coverage[level]) &&
                        coverage[level]->ncomp() == 1 &&
                        field_nullspace_layouts_match(*labels[level], *coverage[level]));
    }
  }
  for (std::size_t index = 0; index < components.size(); ++index) {
    const FieldConnectedComponent& component = components[index];
    payload.require(component.label > 0 && !component.identity.empty() &&
                    !component.provenance.empty());
    for (std::size_t previous = 0; previous < index; ++previous)
      payload.require(components[previous].label != component.label &&
                      components[previous].identity != component.identity);
  }
  finish_field_nullspace_preflight(payload, FieldNullspaceCollectiveBoundary::Preparation);
  for (std::size_t level = 0; level < labels.size(); ++level) {
    const PreparedVectorDistribution& distribution = distributions[level];
    std::vector<char, comm_allocator<char>> validation_storage(
        distribution.validation_scratch_byte_count(), char{0});
    distribution.require_exact_values(*labels[level], validation_storage,
                                      "connected-component label field");
    if (!coverage.empty())
      distribution.require_exact_values(*coverage[level], validation_storage,
                                        "connected-component coverage field");
  }
}

struct FieldBasisMomentKernel {
  ConstArray4 value;
  ConstArray4 basis, coverage;
  int component;
  bool masked, covered;
  Real measure;
  POPS_HD void operator()(int i, int j, Real& sum) const {
    const Real b = masked ? basis(i, j, 0) : Real(1);
    const Real active = covered ? coverage(i, j, 0) : Real(1);
    sum += value(i, j, component) * b * active * measure;
  }
};

struct FieldBasisAbsMomentKernel {
  ConstArray4 value;
  ConstArray4 basis, coverage;
  int component;
  bool masked, covered;
  Real measure;
  POPS_HD void operator()(int i, int j, Real& sum) const {
    const Real b = masked ? basis(i, j, 0) : Real(1);
    const Real active = covered ? coverage(i, j, 0) : Real(1);
    const Real weighted = value(i, j, component) * b * active * measure;
    sum += weighted < Real(0) ? -weighted : weighted;
  }
};

struct FieldBasisGramKernel {
  ConstArray4 left, right, left_coverage, right_coverage;
  bool left_masked, right_masked, left_covered, right_covered;
  Real measure;
  POPS_HD void operator()(int i, int j, Real& sum) const {
    const Real a = left_masked ? left(i, j, 0) : Real(1);
    const Real b = right_masked ? right(i, j, 0) : Real(1);
    const Real wa = left_covered ? left_coverage(i, j, 0) : Real(1);
    const Real wb = right_covered ? right_coverage(i, j, 0) : Real(1);
    sum += a * wa * b * wb * measure;
  }
};

struct ShiftFieldBasisKernel {
  Array4 value;
  ConstArray4 mask, coverage;
  int component;
  bool masked, covered;
  Real coefficient;
  POPS_HD void operator()(int i, int j) const {
    const Real basis = masked ? mask(i, j, 0) : Real(1);
    value(i, j, component) -=
        coefficient * basis * (covered ? coverage(i, j, 0) : Real(1));
  }
};

inline void validate_basis_layout(const MultiFab& value, const MultiFab* mask,
                                  const FieldNullspaceBasis& basis) {
  if (basis.identity.empty() || basis.provenance.empty() || basis.recipe_identity.empty())
    throw std::runtime_error(
        "field nullspace basis requires identity, provenance and deterministic recipe identity");
  if (basis.field_component < 0 || basis.field_component >= value.ncomp())
    throw std::runtime_error("field nullspace basis component is outside the solved field");
  if (mask != nullptr &&
      (mask->ncomp() != 1 || mask->box_array().boxes() != value.box_array().boxes() ||
       mask->dmap().ranks() != value.dmap().ranks()))
    throw std::runtime_error("field nullspace mask is not co-distributed with the solved field");
}

inline void validate_mask_layout(const MultiFab& value, const MultiFab* mask, const char* kind) {
  if (mask != nullptr &&
      (mask->ncomp() != 1 || mask->box_array().boxes() != value.box_array().boxes() ||
       mask->dmap().ranks() != value.dmap().ranks()))
    throw std::runtime_error(std::string("field nullspace ") + kind +
                             " mask is not co-distributed with the solved field");
}

inline std::size_t gauge_index(const FieldNullspacePlan& plan, const std::string& identity) {
  for (std::size_t i = 0; i < plan.gauges.size(); ++i)
    if (plan.gauges[i].basis_identity == identity)
      return i;
  return plan.gauges.size();
}

inline void reduce_field_nullspace_values_inplace(
    std::vector<double>& values, const PreparedVectorDistribution& distribution,
    const char* quantity) {
  std::vector<double> scratch(distribution.reduction_scratch_value_count(
                                  std::max(values.size(), std::size_t{1})),
                              0.0);
  distribution.reduce_sum_values(values, scratch, quantity);
}

inline std::vector<double> reduce_field_nullspace_level_values(
    std::vector<std::vector<double>>& level_values,
    std::span<const PreparedVectorDistribution> distributions, const char* quantity) {
  if (level_values.size() != distributions.size())
    throw std::logic_error("field nullspace level-distribution contract is incoherent");
  if (level_values.empty())
    return {};
  std::vector<double> result(level_values.front().size(), 0.0);
  for (std::size_t level = 0; level < level_values.size(); ++level) {
    if (level_values[level].size() != result.size())
      throw std::logic_error("field nullspace level reduction widths are incoherent");
    reduce_field_nullspace_values_inplace(level_values[level], distributions[level], quantity);
    for (std::size_t value = 0; value < result.size(); ++value)
      result[value] += level_values[level][value];
  }
  return result;
}

/// Factor a dense symmetric Gram matrix in place. External providers may supply overlapping basis
/// functions; only linear dependence is rejected. The lower triangle stores a Cholesky factor.
inline void factor_field_nullspace_gram(std::span<double> gram, std::size_t count,
                                        const char* where) {
  if (gram.size() != checked_field_nullspace_collective_product(count, count, where))
    throw std::logic_error(std::string(where) + " storage is incoherent");
  double diagonal_scale = 0.0;
  for (std::size_t index = 0; index < count; ++index) {
    const double diagonal = gram[index * count + index];
    if (!std::isfinite(diagonal) || !(diagonal > 0.0))
      throw std::runtime_error(std::string(where) +
                               " has a non-positive or non-finite basis norm");
    diagonal_scale = std::max(diagonal_scale, diagonal);
  }
  const double pivot_tolerance =
      128.0 * std::numeric_limits<Real>::epsilon() * diagonal_scale *
      static_cast<double>(std::max(count, std::size_t{1}));
  for (std::size_t row = 0; row < count; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      double value = gram[row * count + column];
      if (!std::isfinite(value))
        throw std::runtime_error(std::string(where) + " contains a non-finite overlap");
      for (std::size_t inner = 0; inner < column; ++inner)
        value -= gram[row * count + inner] * gram[column * count + inner];
      if (row == column) {
        if (!(value > pivot_tolerance))
          throw std::runtime_error(std::string(where) +
                                   " is singular or numerically linearly dependent");
        gram[row * count + column] = std::sqrt(value);
      } else {
        gram[row * count + column] = value / gram[column * count + column];
      }
    }
    for (std::size_t column = row + 1; column < count; ++column)
      gram[row * count + column] = 0.0;
  }
}

inline void solve_field_nullspace_gram(std::span<const double> factor, std::size_t count,
                                       std::span<double> values) {
  if (factor.size() != count * count || values.size() != count)
    throw std::logic_error("field nullspace dense gauge storage is incoherent");
  for (std::size_t row = 0; row < count; ++row) {
    double value = values[row];
    for (std::size_t column = 0; column < row; ++column)
      value -= factor[row * count + column] * values[column];
    values[row] = value / factor[row * count + row];
  }
  for (std::size_t reverse = count; reverse != 0; --reverse) {
    const std::size_t row = reverse - 1;
    double value = values[row];
    for (std::size_t column = row + 1; column < count; ++column)
      value -= factor[column * count + row] * values[column];
    values[row] = value / factor[row * count + row];
  }
}

}  // namespace detail

/// Validate the resolved topology basis once, outside every solve. The dense Gram matrix is
/// assembled with one batched reduction per level distribution and factorized exactly once.
/// Overlapping provider bases are valid; only a singular or numerically dependent basis is rejected.
inline void validate_field_nullspace_basis(const std::vector<const MultiFab*>& level_layouts,
                                           const FieldNullspacePlan& plan,
                                           std::span<const PreparedVectorDistribution> distributions,
                                           int first_level = 0) {
  detail::preflight_field_nullspace_fields(
      level_layouts, plan, distributions, first_level,
      detail::FieldNullspaceCollectiveBoundary::BasisValidation);
  if (plan.empty())
    return;
  if (level_layouts.empty())
    throw std::runtime_error("field nullspace validation requires a materialized layout");
  detail::validate_field_nullspace_level_capacity(level_layouts.size(), first_level,
                                                  "field nullspace basis validation");
  const std::size_t count = plan.bases.size();
  const std::size_t gram_size = detail::checked_field_nullspace_collective_product(
      count, count, "field nullspace Gram matrix");
  std::vector<std::vector<double>> level_grams(
      level_layouts.size(), std::vector<double>(gram_size, 0.0));
  for (std::size_t a = 0; a < count; ++a) {
    for (std::size_t b = a; b < count; ++b) {
      if (plan.bases[a].field_component != plan.bases[b].field_component)
        continue;
      for (std::size_t level = 0; level < level_layouts.size(); ++level) {
        const int resolved_level = first_level + static_cast<int>(level);
        std::vector<double>& contribution = level_grams[level];
        const MultiFab& layout = *level_layouts[level];
        const MultiFab* left = plan.bases[a].mask(resolved_level);
        const MultiFab* right = plan.bases[b].mask(resolved_level);
        const MultiFab* left_coverage = plan.bases[a].coverage_mask(resolved_level);
        const MultiFab* right_coverage = plan.bases[b].coverage_mask(resolved_level);
        if (plan.bases[a].measure(resolved_level) != plan.bases[b].measure(resolved_level))
          throw std::runtime_error("field nullspace modes disagree on hierarchy cell measure");
        detail::validate_basis_layout(layout, left, plan.bases[a]);
        detail::validate_basis_layout(layout, right, plan.bases[b]);
        detail::validate_mask_layout(layout, left_coverage, "left coverage");
        detail::validate_mask_layout(layout, right_coverage, "right coverage");
        for (int li = 0; li < layout.local_size(); ++li) {
          const ConstArray4 left_array =
              left == nullptr ? ConstArray4{} : left->fab(li).const_array();
          const ConstArray4 right_array =
              right == nullptr ? ConstArray4{} : right->fab(li).const_array();
          const ConstArray4 left_coverage_array =
              left_coverage == nullptr ? ConstArray4{} : left_coverage->fab(li).const_array();
          const ConstArray4 right_coverage_array =
              right_coverage == nullptr ? ConstArray4{} : right_coverage->fab(li).const_array();
          contribution[a * count + b] += static_cast<double>(reduce_sum_cell(
              layout.box(li),
              detail::FieldBasisGramKernel{left_array, right_array, left_coverage_array,
                                           right_coverage_array, left != nullptr,
                                           right != nullptr, left_coverage != nullptr,
                                           right_coverage != nullptr,
                                           plan.bases[a].measure(resolved_level)}));
        }
      }
      for (std::vector<double>& contribution : level_grams)
        contribution[b * count + a] = contribution[a * count + b];
    }
  }
  std::vector<double> gram = detail::reduce_field_nullspace_level_values(
      level_grams, distributions, "field nullspace Gram matrix");
  detail::factor_field_nullspace_gram(gram, count, "field nullspace Gram matrix");
  std::vector<std::string> constrained;
  for (const auto& gauge : plan.gauges) {
    if (detail::basis_index(plan, gauge.basis_identity) == plan.bases.size())
      throw std::runtime_error("field gauge references an unknown nullspace basis");
    if (std::find(constrained.begin(), constrained.end(), gauge.basis_identity) !=
        constrained.end())
      throw std::runtime_error("field gauge constrains one nullspace basis more than once");
    constrained.push_back(gauge.basis_identity);
  }
}

/// Check every basis compatibility moment with one batched reduction per ownership class. The returned witness is
/// [dot(rhs,b_0), abs(rhs*b_0), ...]; keeping it contiguous also makes checkpoint/diagnostic capture
/// deterministic.  No RHS projection is performed.
inline std::vector<double> require_field_nullspace_compatible(
    const std::vector<const MultiFab*>& rhs_levels, const FieldNullspacePlan& plan,
    std::span<const PreparedVectorDistribution> distributions,
    int first_level = 0) {
  detail::preflight_field_nullspace_fields(rhs_levels, plan, distributions, first_level,
                                           detail::FieldNullspaceCollectiveBoundary::Compatibility);
  if (plan.empty())
    return {};
  if (plan.identity.empty() || plan.layout_identity.empty())
    throw std::runtime_error("field nullspace plan requires exact plan and layout identities");
  if (rhs_levels.empty())
    throw std::runtime_error("field nullspace compatibility requires at least one hierarchy level");
  detail::validate_field_nullspace_level_capacity(rhs_levels.size(), first_level,
                                                  "field nullspace compatibility");
  const std::size_t moment_count = detail::checked_field_nullspace_collective_product(
      plan.bases.size(), std::size_t{2}, "field nullspace compatibility moments");
  std::vector<std::vector<double>> level_moments(
      rhs_levels.size(), std::vector<double>(moment_count, 0.0));
  for (std::size_t b = 0; b < plan.bases.size(); ++b) {
    const auto& basis = plan.bases[b];
    for (std::size_t level = 0; level < rhs_levels.size(); ++level) {
      const MultiFab& rhs = *rhs_levels[level];
      const int resolved_level = first_level + static_cast<int>(level);
      std::vector<double>& contribution = level_moments[level];
      const MultiFab* mask = basis.mask(resolved_level);
      const MultiFab* coverage = basis.coverage_mask(resolved_level);
      const Real measure = basis.measure(resolved_level);
      detail::validate_basis_layout(rhs, mask, basis);
      detail::validate_mask_layout(rhs, coverage, "coverage");
      for (int li = 0; li < rhs.local_size(); ++li) {
        const ConstArray4 value = rhs.fab(li).const_array();
        const ConstArray4 mask_array =
            mask == nullptr ? ConstArray4{} : mask->fab(li).const_array();
        const ConstArray4 coverage_array =
            coverage == nullptr ? ConstArray4{} : coverage->fab(li).const_array();
        const Box2D valid = rhs.box(li);
        contribution[2 * b] += static_cast<double>(reduce_sum_cell(
            valid,
            detail::FieldBasisMomentKernel{value, mask_array, coverage_array, basis.field_component,
                                           mask != nullptr, coverage != nullptr, measure}));
        contribution[2 * b + 1] += static_cast<double>(reduce_sum_cell(
            valid, detail::FieldBasisAbsMomentKernel{value, mask_array, coverage_array,
                                                     basis.field_component, mask != nullptr,
                                                     coverage != nullptr, measure}));
      }
    }
  }
  const std::vector<double> moments = detail::reduce_field_nullspace_level_values(
      level_moments, distributions, "field nullspace compatibility moments");
  for (std::size_t b = 0; b < plan.bases.size(); ++b) {
    if (!std::isfinite(moments[2 * b]) || !std::isfinite(moments[2 * b + 1]))
      throw FieldNullspaceInvalidEvaluation(
          "field RHS has a non-finite compatibility moment for nullspace basis '" +
          plan.bases[b].identity + "' (" + plan.bases[b].provenance +
          "); silent projection is forbidden");
    const double scale = moments[2 * b + 1] > 1.0 ? moments[2 * b + 1] : 1.0;
    const double tolerance = 128.0 * std::numeric_limits<Real>::epsilon() * scale;
    if (std::abs(moments[2 * b]) > tolerance) {
      std::ostringstream witness;
      witness << std::setprecision(17) << moments[2 * b];
      std::ostringstream allowed;
      allowed << std::setprecision(17) << tolerance;
      throw FieldNullspaceIncompatibleRhs(
          "field RHS is incompatible with nullspace basis '" + plan.bases[b].identity + "' (" +
          plan.bases[b].provenance + "): moment=" + witness.str() + " tolerance=" + allowed.str() +
          "; silent projection is forbidden");
    }
  }
  return moments;
}

inline std::vector<double> require_field_nullspace_compatible(const MultiFab& rhs,
                                                              const FieldNullspacePlan& plan,
                                                              const PreparedVectorDistribution&
                                                                  distribution =
                                                                      PreparedVectorDistribution::Distributed) {
  const std::array<PreparedVectorDistribution, 1> distributions{distribution};
  return require_field_nullspace_compatible(std::vector<const MultiFab*>{&rhs}, plan,
                                            distributions, 0);
}

/// Apply every declared gauge with one batched reduction per level distribution. The same dense
/// Gram system used at preparation makes overlapping provider bases order-independent.
inline void apply_field_gauge(const std::vector<MultiFab*>& phi_levels,
                              const FieldNullspacePlan& plan,
                              std::span<const PreparedVectorDistribution> distributions,
                              int first_level = 0) {
  detail::preflight_field_nullspace_fields(phi_levels, plan, distributions, first_level,
                                           detail::FieldNullspaceCollectiveBoundary::Gauge);
  if (plan.gauges.empty())
    return;
  if (plan.identity.empty() || plan.layout_identity.empty())
    throw std::runtime_error("field nullspace plan requires exact plan and layout identities");
  if (plan.gauges.size() != plan.bases.size())
    throw std::runtime_error(
        "field gauge must constrain every declared nullspace basis exactly once");
  detail::validate_field_nullspace_level_capacity(phi_levels.size(), first_level,
                                                  "field gauge application");
  const std::size_t basis_count = plan.bases.size();
  const std::size_t gram_count = detail::checked_field_nullspace_collective_product(
      basis_count, basis_count, "field gauge Gram matrix");
  const std::size_t reduction_count = detail::checked_field_nullspace_collective_sum(
      basis_count, gram_count, "field gauge reduction");
  std::vector<std::vector<double>> level_values(
      phi_levels.size(), std::vector<double>(reduction_count, 0.0));
  for (std::size_t b = 0; b < plan.bases.size(); ++b) {
    const auto& basis = plan.bases[b];
    if (detail::gauge_index(plan, basis.identity) == plan.gauges.size())
      throw std::runtime_error("field gauge references do not cover nullspace basis '" +
                               basis.identity + "'");
    for (std::size_t level = 0; level < phi_levels.size(); ++level) {
      MultiFab& phi = *phi_levels[level];
      const int resolved_level = first_level + static_cast<int>(level);
      std::vector<double>& contribution = level_values[level];
      const MultiFab* mask = basis.mask(resolved_level);
      const MultiFab* coverage = basis.coverage_mask(resolved_level);
      const Real measure = basis.measure(resolved_level);
      detail::validate_basis_layout(phi, mask, basis);
      detail::validate_mask_layout(phi, coverage, "coverage");
      for (int li = 0; li < phi.local_size(); ++li) {
        const ConstArray4 value = phi.fab(li).const_array();
        const ConstArray4 mask_array =
            mask == nullptr ? ConstArray4{} : mask->fab(li).const_array();
        const ConstArray4 coverage_array =
            coverage == nullptr ? ConstArray4{} : coverage->fab(li).const_array();
        const Box2D valid = phi.box(li);
        contribution[b] += static_cast<double>(reduce_sum_cell(
            valid,
            detail::FieldBasisMomentKernel{value, mask_array, coverage_array, basis.field_component,
                                           mask != nullptr, coverage != nullptr, measure}));
      }
    }
  }
  for (std::size_t left = 0; left < basis_count; ++left) {
    for (std::size_t right = left; right < basis_count; ++right) {
      if (plan.bases[left].field_component != plan.bases[right].field_component)
        continue;
      for (std::size_t level = 0; level < phi_levels.size(); ++level) {
        const int resolved_level = first_level + static_cast<int>(level);
        MultiFab& phi = *phi_levels[level];
        const MultiFab* left_mask = plan.bases[left].mask(resolved_level);
        const MultiFab* right_mask = plan.bases[right].mask(resolved_level);
        const MultiFab* left_coverage = plan.bases[left].coverage_mask(resolved_level);
        const MultiFab* right_coverage = plan.bases[right].coverage_mask(resolved_level);
        if (plan.bases[left].measure(resolved_level) !=
            plan.bases[right].measure(resolved_level))
          throw std::runtime_error("field nullspace modes disagree on hierarchy cell measure");
        for (int local = 0; local < phi.local_size(); ++local) {
          const ConstArray4 left_values =
              left_mask == nullptr ? ConstArray4{} : left_mask->fab(local).const_array();
          const ConstArray4 right_values =
              right_mask == nullptr ? ConstArray4{} : right_mask->fab(local).const_array();
          const ConstArray4 left_coverage_values = left_coverage == nullptr
                                                       ? ConstArray4{}
                                                       : left_coverage->fab(local).const_array();
          const ConstArray4 right_coverage_values = right_coverage == nullptr
                                                        ? ConstArray4{}
                                                        : right_coverage->fab(local).const_array();
          level_values[level][basis_count + left * basis_count + right] +=
              static_cast<double>(reduce_sum_cell(
                  phi.box(local),
                  detail::FieldBasisGramKernel{
                      left_values, right_values, left_coverage_values, right_coverage_values,
                      left_mask != nullptr, right_mask != nullptr, left_coverage != nullptr,
                      right_coverage != nullptr,
                      plan.bases[left].measure(resolved_level)}));
        }
      }
      for (std::vector<double>& values : level_values)
        values[basis_count + right * basis_count + left] =
            values[basis_count + left * basis_count + right];
    }
  }
  const std::vector<double> reduced = detail::reduce_field_nullspace_level_values(
      level_values, distributions, "field dense gauge system");
  std::vector<double> coefficients(reduced.begin(), reduced.begin() + basis_count);
  std::vector<double> gram(reduced.begin() + basis_count, reduced.end());
  detail::factor_field_nullspace_gram(gram, basis_count, "field gauge Gram matrix");
  detail::solve_field_nullspace_gram(gram, basis_count, coefficients);
  for (std::size_t b = 0; b < basis_count; ++b) {
    const auto& basis = plan.bases[b];
    const std::size_t g = detail::gauge_index(plan, basis.identity);
    if (!std::isfinite(coefficients[b]))
      throw FieldNullspaceInvalidEvaluation(
          "field gauge produced a non-finite dense coefficient");
    const Real coefficient = static_cast<Real>(coefficients[b]) - plan.gauges[g].value;
    for (std::size_t level = 0; level < phi_levels.size(); ++level) {
      MultiFab& phi = *phi_levels[level];
      const MultiFab* mask = basis.mask(first_level + static_cast<int>(level));
      const MultiFab* coverage = basis.coverage_mask(first_level + static_cast<int>(level));
      for (int li = 0; li < phi.local_size(); ++li) {
        Array4 value = phi.fab(li).array();
        const ConstArray4 mask_array =
            mask == nullptr ? ConstArray4{} : mask->fab(li).const_array();
        const ConstArray4 coverage_array =
            coverage == nullptr ? ConstArray4{} : coverage->fab(li).const_array();
        for_each_cell(phi.box(li),
                      detail::ShiftFieldBasisKernel{
                          value, mask_array, coverage_array, basis.field_component,
                          mask != nullptr, coverage != nullptr, coefficient});
      }
    }
  }
}

inline void apply_field_gauge(
    MultiFab& phi, const FieldNullspacePlan& plan,
    const PreparedVectorDistribution& distribution = PreparedVectorDistribution::Distributed) {
  std::vector<MultiFab*> levels{&phi};
  const std::array<PreparedVectorDistribution, 1> distributions{distribution};
  apply_field_gauge(levels, plan, distributions);
}

inline FieldNullspacePlan constant_mean_zero_nullspace(
    std::string identity, std::string provenance, Real cell_measure = Real(1)) {
  FieldNullspacePlan result;
  result.identity = std::move(identity);
  result.layout_identity = result.identity + ":layout";
  result.bases.push_back(FieldNullspaceBasis{result.identity + ":constant",
                                             std::move(provenance),
                                             result.identity + ":recipe",
                                             0,
                                             {},
                                             {},
                                             {cell_measure}});
  result.gauges.push_back(FieldGaugeConstraint{result.bases[0].identity, Real(0)});
  return result;
}

/// Materialise one constant nullspace basis per connected-component label.
///
/// This is an installation-time operation.  Component discovery may be implemented by any topology
/// provider, but the provider must hand the solver exact, co-distributed integer label fields.  The
/// function validates the complete label vocabulary collectively, builds immutable basis masks, and
/// creates one independent mean-zero gauge per component.  No registry or string lookup remains in
/// the solve hot path, and RHS compatibility is still checked by require_field_nullspace_compatible()
/// without projection.
inline FieldNullspacePlan labelled_mean_zero_nullspace(
    std::string identity, std::string layout_identity,
    const std::vector<std::shared_ptr<const MultiFab>>& labels,
    std::vector<FieldConnectedComponent> components,
    std::vector<std::shared_ptr<const MultiFab>> coverage, std::vector<Real> cell_measure,
    int field_component,
    std::span<const PreparedVectorDistribution> distributions,
    int first_level = 0) {
  std::sort(components.begin(), components.end(),
            [](const FieldConnectedComponent& left, const FieldConnectedComponent& right) {
              return left.label < right.label;
            });
  detail::preflight_labelled_field_nullspace(identity, layout_identity, labels, components,
                                             coverage, cell_measure, distributions,
                                             field_component, first_level);
  if (identity.empty() || layout_identity.empty())
    throw std::invalid_argument(
        "labelled field nullspace requires non-empty plan and layout identities");
  if (labels.empty())
    throw std::invalid_argument("labelled field nullspace requires at least one label field");
  if (components.empty())
    throw std::invalid_argument(
        "labelled field nullspace requires at least one connected component");
  if (cell_measure.size() != labels.size())
    throw std::invalid_argument(
        "labelled field nullspace requires one cell measure per hierarchy level");
  if (!coverage.empty() && coverage.size() != labels.size())
    throw std::invalid_argument(
        "labelled field nullspace requires either no coverage or one coverage mask per level");
  if (distributions.size() != labels.size())
    throw std::invalid_argument(
        "labelled field nullspace requires one vector-distribution provider per resolved level");
  detail::validate_field_nullspace_level_capacity(labels.size(), first_level,
                                                  "labelled field nullspace");
  const std::size_t count_size = detail::checked_field_nullspace_collective_sum(
      components.size(), std::size_t{1}, "connected-component label counts");

  for (std::size_t index = 0; index < components.size(); ++index) {
    const auto& component = components[index];
    if (component.label <= 0 || component.identity.empty() || component.provenance.empty())
      throw std::invalid_argument(
          "connected field components require a positive label, identity and provenance");
    if (index != 0 && components[index - 1].label == component.label)
      throw std::invalid_argument("connected field component labels must be unique");
    for (std::size_t previous = 0; previous < index; ++previous)
      if (components[previous].identity == component.identity)
        throw std::invalid_argument("connected field component identities must be unique");
  }

  std::vector<std::vector<std::shared_ptr<MultiFab>>> masks(
      components.size(),
      std::vector<std::shared_ptr<MultiFab>>(static_cast<std::size_t>(first_level)));
  std::vector<std::vector<double>> counts_by_level(
      labels.size(), std::vector<double>(count_size, 0.0));
  for (std::size_t level = 0; level < labels.size(); ++level) {
    const std::size_t resolved_level = static_cast<std::size_t>(first_level) + level;
    std::vector<double>& level_counts = counts_by_level[level];
    if (!labels[level] || labels[level]->ncomp() != 1)
      throw std::invalid_argument(
          "connected-component label fields must be materialized one-component MultiFabs");
    if (!(cell_measure[level] > Real(0)))
      throw std::invalid_argument("field nullspace cell measures must be positive");
    if (!coverage.empty()) {
      if (!coverage[level] || coverage[level]->ncomp() != 1 ||
          coverage[level]->box_array().boxes() != labels[level]->box_array().boxes() ||
          coverage[level]->dmap().ranks() != labels[level]->dmap().ranks())
        throw std::invalid_argument(
          "field nullspace coverage must be co-distributed with component labels");
    }
    for (auto& per_component : masks)
      per_component.push_back(
          std::make_shared<MultiFab>(labels[level]->box_array(), labels[level]->dmap(), 1, 0));

    for (int li = 0; li < labels[level]->local_size(); ++li) {
      const ConstArray4 source = labels[level]->fab(li).const_array();
      std::vector<Array4> outputs;
      outputs.reserve(components.size());
      for (auto& per_component : masks)
        outputs.push_back(per_component.at(resolved_level)->fab(li).array());
      const Box2D valid = labels[level]->box(li);
      for (int j = valid.lo[1]; j <= valid.hi[1]; ++j) {
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
          const Real raw = source(i, j, 0);
          const Real integral = std::nearbyint(raw);
          std::size_t selected = components.size();
          if (!std::isfinite(raw) || raw < Real(0) || raw != integral ||
              integral > static_cast<Real>(std::numeric_limits<int>::max())) {
            level_counts.back() += 1.0;
          } else if (integral > Real(0)) {
            const int label = static_cast<int>(integral);
            const auto found = std::lower_bound(components.begin(), components.end(), label,
                                                [](const FieldConnectedComponent& component,
                                                   int value) { return component.label < value; });
            if (found == components.end() || found->label != label) {
              level_counts.back() += 1.0;
            } else {
              selected = static_cast<std::size_t>(std::distance(components.begin(), found));
              level_counts[selected] += 1.0;
            }
          }
          for (std::size_t component = 0; component < outputs.size(); ++component)
            outputs[component](i, j, 0) = component == selected ? Real(1) : Real(0);
        }
      }
    }
  }
  const std::vector<double> counts = detail::reduce_field_nullspace_level_values(
      counts_by_level, distributions, "connected-component label counts");
  if (counts.back() != 0.0)
    throw std::runtime_error(
        "connected-component label fields contain invalid or undeclared positive labels");
  for (std::size_t component = 0; component < components.size(); ++component)
    if (!(counts[component] > 0.0))
      throw std::runtime_error("connected field component '" + components[component].identity +
                               "' has no material cells");

  FieldNullspacePlan result;
  result.identity = std::move(identity);
  result.layout_identity = std::move(layout_identity);
  for (std::size_t component = 0; component < components.size(); ++component) {
    FieldNullspaceBasis basis;
    basis.identity = components[component].identity;
    basis.provenance = components[component].provenance;
    basis.recipe_identity =
        result.identity + ":component-label:" + std::to_string(components[component].label);
    basis.field_component = field_component;
    basis.masks.reserve(masks[component].size());
    for (auto& mask : masks[component])
      basis.masks.push_back(std::move(mask));
    basis.coverage = coverage;
    basis.cell_measure.assign(static_cast<std::size_t>(first_level), Real(0));
    basis.cell_measure.insert(basis.cell_measure.end(), cell_measure.begin(), cell_measure.end());
    result.gauges.push_back(FieldGaugeConstraint{basis.identity, Real(0)});
    result.bases.push_back(std::move(basis));
  }
  std::vector<const MultiFab*> layouts;
  layouts.reserve(labels.size());
  for (const auto& label : labels)
    layouts.push_back(label.get());
  // Label fields are one-component topology data, not the solved field.  The Gram validation only
  // needs their layout and the materialized masks; validate those masks on component zero while
  // preserving the authored target component for the later resolved-field boundary.
  FieldNullspacePlan mask_validation = result;
  for (FieldNullspaceBasis& basis : mask_validation.bases)
    basis.field_component = 0;
  validate_field_nullspace_basis(layouts, mask_validation, distributions, first_level);
  return result;
}

}  // namespace pops
