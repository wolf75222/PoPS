#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops {

enum class FieldNullspaceScope { Uniform, LevelLocal, Composite };

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
  /// Composite-valid coverage (1 on active cells, 0 on coarse cells covered by finer levels).
  /// Empty is valid only outside Composite scope.
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

  const MultiFab* coverage_mask(int level, FieldNullspaceScope scope) const {
    if (scope != FieldNullspaceScope::Composite)
      return nullptr;
    if (level < 0)
      throw std::runtime_error(
          "composite field nullspace basis is missing its valid-cell coverage mask");
    const std::size_t index = static_cast<std::size_t>(level);
    if (index >= coverage.size() || !coverage[index])
      throw std::runtime_error(
          "composite field nullspace basis is missing its valid-cell coverage mask");
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
  FieldNullspaceScope scope = FieldNullspaceScope::Uniform;
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
inline std::size_t checked_field_nullspace_collective_product(std::size_t left,
                                                              std::size_t right,
                                                              const char* quantity) {
  if (left != 0 && right != 0 && left > kFieldNullspaceCollectiveCapacity / right)
    throw std::overflow_error(std::string(quantity) +
                              " exceeds the native MPI collective count capacity");
  return left * right;
}

inline std::size_t checked_field_nullspace_collective_sum(std::size_t value,
                                                          std::size_t increment,
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

enum class FieldNullspaceCollectiveBoundary { BasisValidation, Compatibility, Gauge, Prepare };

inline const char* field_nullspace_boundary_name(FieldNullspaceCollectiveBoundary boundary) {
  switch (boundary) {
    case FieldNullspaceCollectiveBoundary::BasisValidation:
      return "basis validation";
    case FieldNullspaceCollectiveBoundary::Compatibility:
      return "compatibility";
    case FieldNullspaceCollectiveBoundary::Gauge:
      return "gauge";
    case FieldNullspaceCollectiveBoundary::Prepare:
      return "labelled preparation";
  }
  return "unknown boundary";
}

/// Canonical native payload for one field-nullspace collective boundary.  It deliberately stores
/// complete replicated metadata rather than a digest: the communicator helper first agrees lengths,
/// then compares every byte with chunked collectives.  Distributed field values are not replicated
/// and therefore never participate; their complete BoxArray/DistributionMapping metadata does.
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

  void append_layout(const MultiFab* field) {
    const std::uint8_t present = field == nullptr ? 0u : 1u;
    append_scalar(present);
    if (field == nullptr)
      return;
    append_scalar(field->ncomp());
    append_scalar(field->n_grow());
    const auto& boxes = field->box_array().boxes();
    const auto& ranks = field->dmap().ranks();
    append_size(boxes.size());
    for (const Box2D& box : boxes) {
      append_scalar(box.lo[0]);
      append_scalar(box.lo[1]);
      append_scalar(box.hi[0]);
      append_scalar(box.hi[1]);
    }
    append_size(ranks.size());
    for (const int rank : ranks)
      append_scalar(rank);
  }

  void append_plan(const FieldNullspacePlan& plan) {
    append_text(plan.identity);
    append_text(plan.layout_identity);
    append_scalar(static_cast<int>(plan.scope));
    append_size(plan.bases.size());
    for (const FieldNullspaceBasis& basis : plan.bases) {
      append_text(basis.identity);
      append_text(basis.provenance);
      append_text(basis.recipe_identity);
      append_scalar(basis.field_component);
      append_size(basis.masks.size());
      for (const auto& mask : basis.masks)
        append_layout(mask.get());
      append_size(basis.coverage.size());
      for (const auto& mask : basis.coverage)
        append_layout(mask.get());
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

  void require(bool condition) noexcept { valid_ = valid_ && condition; }
  bool valid() const noexcept { return valid_; }
  const std::string& bytes() const noexcept { return bytes_; }

 private:
  std::string bytes_;
  bool valid_ = true;
};

inline bool field_nullspace_scope_is_valid(FieldNullspaceScope scope) noexcept {
  return scope == FieldNullspaceScope::Uniform || scope == FieldNullspaceScope::LevelLocal ||
         scope == FieldNullspaceScope::Composite;
}

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
  payload.require(field_nullspace_scope_is_valid(plan.scope));
  payload.require(plan.bases.empty() ? plan.gauges.empty()
                                     : (!plan.identity.empty() && !plan.layout_identity.empty()));
  for (std::size_t index = 0; index < plan.bases.size(); ++index) {
    const FieldNullspaceBasis& basis = plan.bases[index];
    payload.require(!basis.identity.empty() && !basis.provenance.empty() &&
                    !basis.recipe_identity.empty() && basis.field_component >= 0);
    for (std::size_t previous = 0; previous < index; ++previous)
      payload.require(plan.bases[previous].identity != basis.identity);
    for (const auto& mask : basis.masks) {
      payload.require(mask != nullptr);
      if (mask != nullptr)
        payload.require(field_nullspace_layout_is_materialized(*mask) && mask->ncomp() == 1);
    }
    if (plan.scope != FieldNullspaceScope::Composite)
      payload.require(basis.coverage.empty());
    for (const auto& coverage : basis.coverage) {
      payload.require(coverage != nullptr);
      if (coverage != nullptr)
        payload.require(field_nullspace_layout_is_materialized(*coverage) &&
                        coverage->ncomp() == 1);
    }
    for (const Real measure : basis.cell_measure)
      payload.require(std::isfinite(static_cast<double>(measure)) && measure > Real(0));
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
  const std::vector<std::pair<std::string_view, std::string_view>> identity{
      {boundary_name, payload.bytes()}};
  const bool agreed = all_ranks_agree_exact_ordered_byte_pairs(identity);
  if (!agreed || !payload.valid())
    throw std::runtime_error(std::string("field nullspace ") + std::string(boundary_name) +
                             " collective preflight rejected malformed local structure or "
                             "rank-divergent metadata");
}

template <class FieldVector>
inline void preflight_field_nullspace_fields(const FieldVector& fields,
                                             const FieldNullspacePlan& plan, int first_level,
                                             FieldNullspaceCollectiveBoundary boundary) {
  FieldNullspacePreflightPayload payload;
  payload.append_scalar(first_level);
  payload.append_plan(plan);
  payload.append_size(fields.size());
  for (const auto* field : fields)
    payload.append_layout(field);
  validate_field_nullspace_plan_locally(payload, plan);

  const bool active = boundary == FieldNullspaceCollectiveBoundary::Gauge
                          ? !plan.gauges.empty()
                          : !plan.bases.empty();
  if (active) {
    payload.require(!fields.empty());
    payload.require(field_nullspace_level_capacity_is_valid(fields.size(), first_level));
    payload.require(plan.scope != FieldNullspaceScope::LevelLocal || fields.size() == 1);
    if (boundary == FieldNullspaceCollectiveBoundary::Gauge)
      payload.require(plan.gauges.size() == plan.bases.size());
    try {
      if (boundary == FieldNullspaceCollectiveBoundary::BasisValidation)
        (void)checked_field_nullspace_collective_product(
            plan.bases.size(), plan.bases.size(), "field nullspace Gram matrix");
      else
        (void)checked_field_nullspace_collective_product(
            plan.bases.size(), std::size_t{2}, "field nullspace moments");
    } catch (const std::exception&) {
      payload.require(false);
    }

    const bool level_range_valid =
        field_nullspace_level_capacity_is_valid(fields.size(), first_level);
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
          payload.require(resolved_level < basis.masks.size());
          if (resolved_level < basis.masks.size() && basis.masks[resolved_level] != nullptr)
            payload.require(field_nullspace_layouts_match(*field, *basis.masks[resolved_level]));
        }
        if (plan.scope == FieldNullspaceScope::Composite) {
          payload.require(resolved_level < basis.coverage.size());
          if (resolved_level < basis.coverage.size() && basis.coverage[resolved_level] != nullptr)
            payload.require(field_nullspace_layouts_match(*field, *basis.coverage[resolved_level]));
        }
        payload.require(resolved_level < basis.cell_measure.size());
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
}

inline void preflight_labelled_field_nullspace(
    std::string_view identity, std::string_view layout_identity, FieldNullspaceScope scope,
    const std::vector<std::shared_ptr<const MultiFab>>& labels,
    const std::vector<FieldConnectedComponent>& components,
    const std::vector<std::shared_ptr<const MultiFab>>& coverage,
    const std::vector<Real>& cell_measure, int field_component) {
  FieldNullspacePreflightPayload payload;
  payload.append_text(identity);
  payload.append_text(layout_identity);
  payload.append_scalar(static_cast<int>(scope));
  payload.append_scalar(field_component);
  payload.append_size(labels.size());
  for (const auto& label : labels)
    payload.append_layout(label.get());
  payload.append_size(components.size());
  for (const FieldConnectedComponent& component : components) {
    payload.append_scalar(component.label);
    payload.append_text(component.identity);
    payload.append_text(component.provenance);
  }
  payload.append_size(coverage.size());
  for (const auto& mask : coverage)
    payload.append_layout(mask.get());
  payload.append_size(cell_measure.size());
  for (const Real measure : cell_measure)
    payload.append_scalar(measure);

  payload.require(!identity.empty() && !layout_identity.empty() &&
                  field_nullspace_scope_is_valid(scope));
  payload.require(!labels.empty() && !components.empty() && field_component >= 0);
  payload.require(field_nullspace_level_capacity_is_valid(labels.size(), 0));
  payload.require(cell_measure.size() == labels.size());
  payload.require(scope == FieldNullspaceScope::Composite ? coverage.size() == labels.size()
                                                          : coverage.empty());
  try {
    (void)checked_field_nullspace_collective_sum(
        components.size(), std::size_t{1}, "connected-component label counts");
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
    if (scope == FieldNullspaceScope::Composite && level < coverage.size()) {
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
  finish_field_nullspace_preflight(payload, FieldNullspaceCollectiveBoundary::Prepare);
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

struct FieldBasisNormKernel {
  ConstArray4 basis, coverage;
  bool masked, covered;
  Real measure;
  POPS_HD void operator()(int i, int j, Real& sum) const {
    const Real b = masked ? basis(i, j, 0) : Real(1);
    const Real active = covered ? coverage(i, j, 0) : Real(1);
    sum += b * b * active * measure;
  }
};

struct FieldBasisGramKernel {
  ConstArray4 left, right, coverage;
  bool left_masked, right_masked, covered;
  Real measure;
  POPS_HD void operator()(int i, int j, Real& sum) const {
    const Real a = left_masked ? left(i, j, 0) : Real(1);
    const Real b = right_masked ? right(i, j, 0) : Real(1);
    const Real active = covered ? coverage(i, j, 0) : Real(1);
    sum += a * b * active * measure;
  }
};

struct ShiftFieldBasisKernel {
  Array4 value;
  ConstArray4 mask;
  int component;
  bool masked;
  Real coefficient;
  POPS_HD void operator()(int i, int j) const {
    value(i, j, component) -= coefficient * (masked ? mask(i, j, 0) : Real(1));
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

}  // namespace detail

/// Validate the resolved topology basis once, outside every solve.  The Gram matrix is assembled
/// with one collective; modes on different solved-field components are orthogonal by construction,
/// while same-component modes must be disjoint/orthogonal to roundoff.  This prevents the gauge
/// application below from acquiring a hidden order dependence.
inline void validate_field_nullspace_basis(const std::vector<const MultiFab*>& level_layouts,
                                           const FieldNullspacePlan& plan, int first_level = 0) {
  detail::preflight_field_nullspace_fields(
      level_layouts, plan, first_level, detail::FieldNullspaceCollectiveBoundary::BasisValidation);
  if (plan.empty())
    return;
  if (level_layouts.empty())
    throw std::runtime_error("field nullspace validation requires a materialized layout");
  if (plan.scope == FieldNullspaceScope::LevelLocal && level_layouts.size() != 1)
    throw std::runtime_error("level-local nullspace basis must be validated one level at a time");
  detail::validate_field_nullspace_level_capacity(level_layouts.size(), first_level,
                                                  "field nullspace basis validation");
  const std::size_t count = plan.bases.size();
  const std::size_t gram_size = detail::checked_field_nullspace_collective_product(
      count, count, "field nullspace Gram matrix");
  std::vector<double> gram(gram_size, 0.0);
  for (std::size_t a = 0; a < count; ++a) {
    for (std::size_t b = a; b < count; ++b) {
      if (plan.bases[a].field_component != plan.bases[b].field_component)
        continue;
      for (std::size_t level = 0; level < level_layouts.size(); ++level) {
        const int resolved_level = first_level + static_cast<int>(level);
        const MultiFab& layout = *level_layouts[level];
        const MultiFab* left = plan.bases[a].mask(resolved_level);
        const MultiFab* right = plan.bases[b].mask(resolved_level);
        const MultiFab* coverage = plan.bases[a].coverage_mask(resolved_level, plan.scope);
        if (plan.bases[a].measure(resolved_level) != plan.bases[b].measure(resolved_level))
          throw std::runtime_error("field nullspace modes disagree on hierarchy cell measure");
        detail::validate_basis_layout(layout, left, plan.bases[a]);
        detail::validate_basis_layout(layout, right, plan.bases[b]);
        detail::validate_mask_layout(layout, coverage, "coverage");
        for (int li = 0; li < layout.local_size(); ++li) {
          const ConstArray4 left_array =
              left == nullptr ? ConstArray4{} : left->fab(li).const_array();
          const ConstArray4 right_array =
              right == nullptr ? ConstArray4{} : right->fab(li).const_array();
          const ConstArray4 coverage_array =
              coverage == nullptr ? ConstArray4{} : coverage->fab(li).const_array();
          gram[a * count + b] += static_cast<double>(reduce_sum_cell(
              layout.box(li),
              detail::FieldBasisGramKernel{left_array, right_array, coverage_array, left != nullptr,
                                           right != nullptr, coverage != nullptr,
                                           plan.bases[a].measure(resolved_level)}));
        }
      }
      gram[b * count + a] = gram[a * count + b];
    }
  }
  all_reduce_sum_inplace(
      gram.data(),
      detail::checked_field_nullspace_collective_count(gram.size(),
                                                       "field nullspace Gram matrix"));
  for (std::size_t a = 0; a < count; ++a) {
    if (!std::isfinite(gram[a * count + a]))
      throw std::runtime_error("field nullspace basis '" + plan.bases[a].identity +
                               "' has non-finite composite measure");
    if (!(gram[a * count + a] > 0.0))
      throw std::runtime_error("field nullspace basis '" + plan.bases[a].identity +
                               "' has zero composite measure");
    for (std::size_t b = a + 1; b < count; ++b) {
      if (plan.bases[a].field_component != plan.bases[b].field_component)
        continue;
      if (!std::isfinite(gram[a * count + b]))
        throw std::runtime_error("field nullspace bases '" + plan.bases[a].identity + "' and '" +
                                 plan.bases[b].identity +
                                 "' have a non-finite overlap measure");
      const double tolerance = 128.0 * std::numeric_limits<Real>::epsilon() *
                               std::sqrt(gram[a * count + a] * gram[b * count + b]);
      if (std::abs(gram[a * count + b]) > tolerance)
        throw std::runtime_error("field nullspace bases '" + plan.bases[a].identity + "' and '" +
                                 plan.bases[b].identity + "' are not orthogonal/disjoint");
    }
  }
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

/// Check every basis compatibility moment with exactly ONE collective.  The returned witness is
/// [dot(rhs,b_0), abs(rhs*b_0), ...]; keeping it contiguous also makes checkpoint/diagnostic capture
/// deterministic.  No RHS projection is performed.
inline std::vector<double> require_field_nullspace_compatible(
    const std::vector<const MultiFab*>& rhs_levels, const FieldNullspacePlan& plan,
    int first_level = 0) {
  detail::preflight_field_nullspace_fields(
      rhs_levels, plan, first_level, detail::FieldNullspaceCollectiveBoundary::Compatibility);
  if (plan.empty())
    return {};
  if (plan.identity.empty() || plan.layout_identity.empty())
    throw std::runtime_error("field nullspace plan requires exact plan and layout identities");
  if (rhs_levels.empty())
    throw std::runtime_error("field nullspace compatibility requires at least one hierarchy level");
  if (plan.scope == FieldNullspaceScope::LevelLocal && rhs_levels.size() != 1)
    throw std::runtime_error(
        "level-local field nullspace compatibility must be evaluated independently per level");
  detail::validate_field_nullspace_level_capacity(rhs_levels.size(), first_level,
                                                  "field nullspace compatibility");
  const std::size_t moment_count = detail::checked_field_nullspace_collective_product(
      plan.bases.size(), std::size_t{2}, "field nullspace compatibility moments");
  std::vector<double> moments(moment_count, 0.0);
  for (std::size_t b = 0; b < plan.bases.size(); ++b) {
    const auto& basis = plan.bases[b];
    for (std::size_t level = 0; level < rhs_levels.size(); ++level) {
      const MultiFab& rhs = *rhs_levels[level];
      const int resolved_level = first_level + static_cast<int>(level);
      const MultiFab* mask = basis.mask(resolved_level);
      const MultiFab* coverage = basis.coverage_mask(resolved_level, plan.scope);
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
        moments[2 * b] += static_cast<double>(reduce_sum_cell(
            valid,
            detail::FieldBasisMomentKernel{value, mask_array, coverage_array, basis.field_component,
                                           mask != nullptr, coverage != nullptr, measure}));
        moments[2 * b + 1] += static_cast<double>(reduce_sum_cell(
            valid, detail::FieldBasisAbsMomentKernel{value, mask_array, coverage_array,
                                                     basis.field_component, mask != nullptr,
                                                     coverage != nullptr, measure}));
      }
    }
  }
  all_reduce_sum_inplace(
      moments.data(), detail::checked_field_nullspace_collective_count(
                          moments.size(), "field nullspace compatibility moments"));
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
          plan.bases[b].provenance + "): moment=" + witness.str() +
          " tolerance=" + allowed.str() + "; silent projection is forbidden");
    }
  }
  return moments;
}

inline std::vector<double> require_field_nullspace_compatible(const MultiFab& rhs,
                                                              const FieldNullspacePlan& plan) {
  return require_field_nullspace_compatible(std::vector<const MultiFab*>{&rhs}, plan, 0);
}

/// Apply every declared gauge with one collective for all dot products and basis norms.  Masks must
/// be disjoint/orthogonal; non-orthogonal gauges require an explicit dense gauge solver and are
/// rejected at plan construction rather than being silently order-dependent.
inline void apply_field_gauge(const std::vector<MultiFab*>& phi_levels,
                              const FieldNullspacePlan& plan, int first_level = 0) {
  detail::preflight_field_nullspace_fields(
      phi_levels, plan, first_level, detail::FieldNullspaceCollectiveBoundary::Gauge);
  if (plan.gauges.empty())
    return;
  if (plan.identity.empty() || plan.layout_identity.empty())
    throw std::runtime_error("field nullspace plan requires exact plan and layout identities");
  if (plan.scope == FieldNullspaceScope::LevelLocal && phi_levels.size() != 1)
    throw std::runtime_error("level-local field gauges must be applied independently per level");
  if (plan.gauges.size() != plan.bases.size())
    throw std::runtime_error(
        "field gauge must constrain every declared nullspace basis exactly once");
  detail::validate_field_nullspace_level_capacity(phi_levels.size(), first_level,
                                                  "field gauge application");
  const std::size_t moment_count = detail::checked_field_nullspace_collective_product(
      plan.bases.size(), std::size_t{2}, "field gauge moments");
  std::vector<double> moments(moment_count, 0.0);
  for (std::size_t b = 0; b < plan.bases.size(); ++b) {
    const auto& basis = plan.bases[b];
    if (detail::gauge_index(plan, basis.identity) == plan.gauges.size())
      throw std::runtime_error("field gauge references do not cover nullspace basis '" +
                               basis.identity + "'");
    for (std::size_t level = 0; level < phi_levels.size(); ++level) {
      MultiFab& phi = *phi_levels[level];
      const int resolved_level = first_level + static_cast<int>(level);
      const MultiFab* mask = basis.mask(resolved_level);
      const MultiFab* coverage = basis.coverage_mask(resolved_level, plan.scope);
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
        moments[2 * b] += static_cast<double>(reduce_sum_cell(
            valid,
            detail::FieldBasisMomentKernel{value, mask_array, coverage_array, basis.field_component,
                                           mask != nullptr, coverage != nullptr, measure}));
        moments[2 * b + 1] += static_cast<double>(reduce_sum_cell(
            valid, detail::FieldBasisNormKernel{mask_array, coverage_array, mask != nullptr,
                                                coverage != nullptr, measure}));
      }
    }
  }
  all_reduce_sum_inplace(moments.data(), detail::checked_field_nullspace_collective_count(
                                             moments.size(), "field gauge moments"));
  for (std::size_t b = 0; b < plan.bases.size(); ++b) {
    const auto& basis = plan.bases[b];
    const std::size_t g = detail::gauge_index(plan, basis.identity);
    const Real norm = static_cast<Real>(moments[2 * b + 1]);
    if (!(norm > Real(0)))
      throw std::runtime_error("field nullspace basis has zero norm");
    const Real coefficient =
        static_cast<Real>(moments[2 * b]) / norm - plan.gauges[g].value;
    for (std::size_t level = 0; level < phi_levels.size(); ++level) {
      MultiFab& phi = *phi_levels[level];
      const MultiFab* mask = basis.mask(first_level + static_cast<int>(level));
      for (int li = 0; li < phi.local_size(); ++li) {
        Array4 value = phi.fab(li).array();
        const ConstArray4 mask_array =
            mask == nullptr ? ConstArray4{} : mask->fab(li).const_array();
        for_each_cell(phi.box(li),
                      detail::ShiftFieldBasisKernel{value, mask_array, basis.field_component,
                                                    mask != nullptr, coefficient});
      }
    }
  }
}

inline void apply_field_gauge(MultiFab& phi, const FieldNullspacePlan& plan) {
  std::vector<MultiFab*> levels{&phi};
  apply_field_gauge(levels, plan);
}

namespace detail {

/// Prepared single-level compatibility kernel. The owning prepared problem has already validated
/// the immutable plan/layout contract collectively and supplies persistent moment storage. This
/// route therefore performs only the one scientific sum and allocates nothing in a hot solve.
inline void require_field_nullspace_compatible_prevalidated(
    const MultiFab& rhs, const FieldNullspacePlan& plan, int resolved_level, double* moments,
    std::size_t moment_count) {
  const std::size_t required = plan.bases.size() * 2u;
  if (moment_count != required || (required != 0 && moments == nullptr))
    throw std::logic_error("prepared field nullspace compatibility storage is incoherent");
  std::fill_n(moments, moment_count, 0.0);
  for (std::size_t b = 0; b < plan.bases.size(); ++b) {
    const FieldNullspaceBasis& basis = plan.bases[b];
    const MultiFab* mask = basis.mask(resolved_level);
    const MultiFab* coverage = basis.coverage_mask(resolved_level, plan.scope);
    const Real measure = basis.measure(resolved_level);
    for (int li = 0; li < rhs.local_size(); ++li) {
      const ConstArray4 value = rhs.fab(li).const_array();
      const ConstArray4 mask_array =
          mask == nullptr ? ConstArray4{} : mask->fab(li).const_array();
      const ConstArray4 coverage_array =
          coverage == nullptr ? ConstArray4{} : coverage->fab(li).const_array();
      const Box2D valid = rhs.box(li);
      moments[2 * b] += static_cast<double>(reduce_sum_cell(
          valid, FieldBasisMomentKernel{value, mask_array, coverage_array, basis.field_component,
                                        mask != nullptr, coverage != nullptr, measure}));
      moments[2 * b + 1] += static_cast<double>(reduce_sum_cell(
          valid, FieldBasisAbsMomentKernel{value, mask_array, coverage_array, basis.field_component,
                                           mask != nullptr, coverage != nullptr, measure}));
    }
  }
  all_reduce_sum_inplace(
      moments, checked_field_nullspace_collective_count(
                   moment_count, "prepared field nullspace compatibility moments"));
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
          plan.bases[b].provenance + "): moment=" + witness.str() +
          " tolerance=" + allowed.str() + "; silent projection is forbidden");
    }
  }
}

/// Prepared single-level gauge kernel, paired with the compatibility helper above. Contract and
/// layout checks stay on the public defensive functions and on PreparedNullspacePolicy::prepare().
inline void apply_field_gauge_prevalidated(MultiFab& phi, const FieldNullspacePlan& plan,
                                           int resolved_level, double* moments,
                                           std::size_t moment_count) {
  const std::size_t required = plan.bases.size() * 2u;
  if (moment_count != required || (required != 0 && moments == nullptr))
    throw std::logic_error("prepared field nullspace gauge storage is incoherent");
  std::fill_n(moments, moment_count, 0.0);
  for (std::size_t b = 0; b < plan.bases.size(); ++b) {
    const FieldNullspaceBasis& basis = plan.bases[b];
    const MultiFab* mask = basis.mask(resolved_level);
    const MultiFab* coverage = basis.coverage_mask(resolved_level, plan.scope);
    const Real measure = basis.measure(resolved_level);
    for (int li = 0; li < phi.local_size(); ++li) {
      const ConstArray4 value = phi.fab(li).const_array();
      const ConstArray4 mask_array =
          mask == nullptr ? ConstArray4{} : mask->fab(li).const_array();
      const ConstArray4 coverage_array =
          coverage == nullptr ? ConstArray4{} : coverage->fab(li).const_array();
      const Box2D valid = phi.box(li);
      moments[2 * b] += static_cast<double>(reduce_sum_cell(
          valid, FieldBasisMomentKernel{value, mask_array, coverage_array, basis.field_component,
                                        mask != nullptr, coverage != nullptr, measure}));
      moments[2 * b + 1] += static_cast<double>(reduce_sum_cell(
          valid, FieldBasisNormKernel{mask_array, coverage_array, mask != nullptr,
                                      coverage != nullptr, measure}));
    }
  }
  all_reduce_sum_inplace(
      moments, checked_field_nullspace_collective_count(moment_count,
                                                        "prepared field nullspace gauge moments"));
  for (std::size_t b = 0; b < plan.bases.size(); ++b) {
    const FieldNullspaceBasis& basis = plan.bases[b];
    const std::size_t gauge = gauge_index(plan, basis.identity);
    const Real norm = static_cast<Real>(moments[2 * b + 1]);
    if (!(norm > Real(0)))
      throw std::runtime_error("field nullspace basis has zero norm");
    const Real coefficient =
        static_cast<Real>(moments[2 * b]) / norm - plan.gauges[gauge].value;
    const MultiFab* mask = basis.mask(resolved_level);
    for (int li = 0; li < phi.local_size(); ++li) {
      Array4 value = phi.fab(li).array();
      const ConstArray4 mask_array =
          mask == nullptr ? ConstArray4{} : mask->fab(li).const_array();
      for_each_cell(phi.box(li), ShiftFieldBasisKernel{value, mask_array, basis.field_component,
                                                       mask != nullptr, coefficient});
    }
  }
}

}  // namespace detail

inline FieldNullspacePlan constant_mean_zero_nullspace(std::string identity, std::string provenance,
                                                       Real cell_measure = Real(1)) {
  FieldNullspacePlan result;
  result.identity = std::move(identity);
  result.layout_identity = result.identity + ":layout";
  result.scope = FieldNullspaceScope::Uniform;
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
    std::string identity, std::string layout_identity, FieldNullspaceScope scope,
    const std::vector<std::shared_ptr<const MultiFab>>& labels,
    std::vector<FieldConnectedComponent> components,
    std::vector<std::shared_ptr<const MultiFab>> coverage, std::vector<Real> cell_measure,
    int field_component = 0) {
  std::sort(components.begin(), components.end(),
            [](const FieldConnectedComponent& left, const FieldConnectedComponent& right) {
              return left.label < right.label;
            });
  detail::preflight_labelled_field_nullspace(identity, layout_identity, scope, labels, components,
                                             coverage, cell_measure, field_component);
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
  if (scope == FieldNullspaceScope::Composite && coverage.size() != labels.size())
    throw std::invalid_argument(
        "composite labelled field nullspace requires one coverage mask per hierarchy level");
  if (scope != FieldNullspaceScope::Composite && !coverage.empty())
    throw std::invalid_argument(
        "uniform/level-local labelled field nullspace must not carry composite coverage");
  detail::validate_field_nullspace_level_capacity(labels.size(), 0,
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

  std::vector<std::vector<std::shared_ptr<MultiFab>>> masks(components.size());
  std::vector<double> counts(count_size, 0.0);
  for (std::size_t level = 0; level < labels.size(); ++level) {
    if (!labels[level] || labels[level]->ncomp() != 1)
      throw std::invalid_argument(
          "connected-component label fields must be materialized one-component MultiFabs");
    if (!(cell_measure[level] > Real(0)))
      throw std::invalid_argument("field nullspace cell measures must be positive");
    if (scope == FieldNullspaceScope::Composite) {
      if (!coverage[level] || coverage[level]->ncomp() != 1 ||
          coverage[level]->box_array().boxes() != labels[level]->box_array().boxes() ||
          coverage[level]->dmap().ranks() != labels[level]->dmap().ranks())
        throw std::invalid_argument(
            "composite field nullspace coverage must be co-distributed with component labels");
    }
    for (auto& per_component : masks)
      per_component.push_back(
          std::make_shared<MultiFab>(labels[level]->box_array(), labels[level]->dmap(), 1, 0));

    for (int li = 0; li < labels[level]->local_size(); ++li) {
      const ConstArray4 source = labels[level]->fab(li).const_array();
      std::vector<Array4> outputs;
      outputs.reserve(components.size());
      for (auto& per_component : masks)
        outputs.push_back(per_component[level]->fab(li).array());
      const Box2D valid = labels[level]->box(li);
      for (int j = valid.lo[1]; j <= valid.hi[1]; ++j) {
        for (int i = valid.lo[0]; i <= valid.hi[0]; ++i) {
          const Real raw = source(i, j, 0);
          const Real integral = std::nearbyint(raw);
          std::size_t selected = components.size();
          if (!std::isfinite(raw) || raw < Real(0) || raw != integral ||
              integral > static_cast<Real>(std::numeric_limits<int>::max())) {
            counts.back() += 1.0;
          } else if (integral > Real(0)) {
            const int label = static_cast<int>(integral);
            const auto found = std::lower_bound(components.begin(), components.end(), label,
                                                [](const FieldConnectedComponent& component,
                                                   int value) { return component.label < value; });
            if (found == components.end() || found->label != label) {
              counts.back() += 1.0;
            } else {
              selected = static_cast<std::size_t>(std::distance(components.begin(), found));
              counts[selected] += 1.0;
            }
          }
          for (std::size_t component = 0; component < outputs.size(); ++component)
            outputs[component](i, j, 0) =
                component == selected ? Real(1) : Real(0);
        }
      }
    }
  }
  all_reduce_sum_inplace(
      counts.data(), detail::checked_field_nullspace_collective_count(
                         counts.size(), "connected-component label counts"));
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
  result.scope = scope;
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
    basis.cell_measure = cell_measure;
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
  validate_field_nullspace_basis(layouts, mask_validation);
  return result;
}

}  // namespace pops
