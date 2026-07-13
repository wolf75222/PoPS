#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
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
    if (level < 0 || level >= static_cast<int>(masks.size()) || !masks[level])
      throw std::runtime_error("field nullspace basis is missing a resolved hierarchy mask");
    return masks[static_cast<std::size_t>(level)].get();
  }

  const MultiFab* coverage_mask(int level, FieldNullspaceScope scope) const {
    if (scope != FieldNullspaceScope::Composite)
      return nullptr;
    if (level < 0 || level >= static_cast<int>(coverage.size()) || !coverage[level])
      throw std::runtime_error(
          "composite field nullspace basis is missing its valid-cell coverage mask");
    return coverage[static_cast<std::size_t>(level)].get();
  }

  Real measure(int level) const {
    if (level < 0 || level >= static_cast<int>(cell_measure.size()) ||
        !(cell_measure[static_cast<std::size_t>(level)] > Real(0)))
      throw std::runtime_error("field nullspace basis is missing a positive cell measure");
    return cell_measure[static_cast<std::size_t>(level)];
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

namespace detail {

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
  if (mask != nullptr && (mask->ncomp() != 1 ||
                          mask->box_array().boxes() != value.box_array().boxes() ||
                          mask->dmap().ranks() != value.dmap().ranks()))
    throw std::runtime_error("field nullspace mask is not co-distributed with the solved field");
}

inline void validate_mask_layout(const MultiFab& value, const MultiFab* mask,
                                 const char* kind) {
  if (mask != nullptr && (mask->ncomp() != 1 ||
                          mask->box_array().boxes() != value.box_array().boxes() ||
                          mask->dmap().ranks() != value.dmap().ranks()))
    throw std::runtime_error(std::string("field nullspace ") + kind +
                             " mask is not co-distributed with the solved field");
}

inline int gauge_index(const FieldNullspacePlan& plan, const std::string& identity) {
  for (int i = 0; i < static_cast<int>(plan.gauges.size()); ++i)
    if (plan.gauges[static_cast<std::size_t>(i)].basis_identity == identity)
      return i;
  return -1;
}

}  // namespace detail

/// Validate the resolved topology basis once, outside every solve.  The Gram matrix is assembled
/// with one collective; modes on different solved-field components are orthogonal by construction,
/// while same-component modes must be disjoint/orthogonal to roundoff.  This prevents the gauge
/// application below from acquiring a hidden order dependence.
inline void validate_field_nullspace_basis(const std::vector<const MultiFab*>& level_layouts,
                                           const FieldNullspacePlan& plan,
                                           int first_level = 0) {
  if (plan.empty())
    return;
  if (level_layouts.empty())
    throw std::runtime_error("field nullspace validation requires a materialized layout");
  if (plan.scope == FieldNullspaceScope::LevelLocal && level_layouts.size() != 1)
    throw std::runtime_error("level-local nullspace basis must be validated one level at a time");
  const std::size_t count = plan.bases.size();
  std::vector<double> gram(count * count, 0.0);
  for (std::size_t a = 0; a < count; ++a) {
    for (std::size_t b = a; b < count; ++b) {
      if (plan.bases[a].field_component != plan.bases[b].field_component)
        continue;
      for (int level = 0; level < static_cast<int>(level_layouts.size()); ++level) {
        const int resolved_level = first_level + level;
        const MultiFab& layout = *level_layouts[static_cast<std::size_t>(level)];
        const MultiFab* left = plan.bases[a].mask(resolved_level);
        const MultiFab* right = plan.bases[b].mask(resolved_level);
        const MultiFab* coverage = plan.bases[a].coverage_mask(resolved_level, plan.scope);
        if (plan.bases[a].measure(resolved_level) != plan.bases[b].measure(resolved_level))
          throw std::runtime_error("field nullspace modes disagree on hierarchy cell measure");
        detail::validate_basis_layout(layout, left, plan.bases[a]);
        detail::validate_basis_layout(layout, right, plan.bases[b]);
        detail::validate_mask_layout(layout, coverage, "coverage");
        for (int li = 0; li < layout.local_size(); ++li) {
          const ConstArray4 left_array = left == nullptr ? ConstArray4{} :
              left->fab(li).const_array();
          const ConstArray4 right_array = right == nullptr ? ConstArray4{} :
              right->fab(li).const_array();
          const ConstArray4 coverage_array = coverage == nullptr ? ConstArray4{} :
              coverage->fab(li).const_array();
          gram[a * count + b] += static_cast<double>(reduce_sum_cell(
              layout.box(li), detail::FieldBasisGramKernel{
                  left_array, right_array, coverage_array, left != nullptr, right != nullptr,
                  coverage != nullptr, plan.bases[a].measure(resolved_level)}));
        }
      }
      gram[b * count + a] = gram[a * count + b];
    }
  }
  all_reduce_sum_inplace(gram.data(), static_cast<int>(gram.size()));
  for (std::size_t a = 0; a < count; ++a) {
    if (!(gram[a * count + a] > 0.0))
      throw std::runtime_error("field nullspace basis '" + plan.bases[a].identity +
                               "' has zero composite measure");
    for (std::size_t b = a + 1; b < count; ++b) {
      if (plan.bases[a].field_component != plan.bases[b].field_component)
        continue;
      const double tolerance = 128.0 * std::numeric_limits<Real>::epsilon() *
          std::sqrt(gram[a * count + a] * gram[b * count + b]);
      if (std::abs(gram[a * count + b]) > tolerance)
        throw std::runtime_error("field nullspace bases '" + plan.bases[a].identity + "' and '" +
                                 plan.bases[b].identity + "' are not orthogonal/disjoint");
    }
  }
  std::vector<std::string> constrained;
  for (const auto& gauge : plan.gauges) {
    if (detail::gauge_index(plan, gauge.basis_identity) < 0)
      throw std::runtime_error("field gauge references an unknown nullspace basis");
    if (std::find(constrained.begin(), constrained.end(), gauge.basis_identity) != constrained.end())
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
  if (plan.empty())
    return {};
  if (plan.identity.empty() || plan.layout_identity.empty())
    throw std::runtime_error("field nullspace plan requires exact plan and layout identities");
  if (rhs_levels.empty())
    throw std::runtime_error("field nullspace compatibility requires at least one hierarchy level");
  if (plan.scope == FieldNullspaceScope::LevelLocal && rhs_levels.size() != 1)
    throw std::runtime_error(
        "level-local field nullspace compatibility must be evaluated independently per level");
  std::vector<double> moments(plan.bases.size() * 2, 0.0);
  for (std::size_t b = 0; b < plan.bases.size(); ++b) {
    const auto& basis = plan.bases[b];
    for (int level = 0; level < static_cast<int>(rhs_levels.size()); ++level) {
      const MultiFab& rhs = *rhs_levels[static_cast<std::size_t>(level)];
      const int resolved_level = first_level + level;
      const MultiFab* mask = basis.mask(resolved_level);
      const MultiFab* coverage = basis.coverage_mask(resolved_level, plan.scope);
      const Real measure = basis.measure(resolved_level);
      detail::validate_basis_layout(rhs, mask, basis);
      detail::validate_mask_layout(rhs, coverage, "coverage");
      for (int li = 0; li < rhs.local_size(); ++li) {
        const ConstArray4 value = rhs.fab(li).const_array();
        const ConstArray4 mask_array = mask == nullptr ? ConstArray4{} :
            mask->fab(li).const_array();
        const ConstArray4 coverage_array = coverage == nullptr ? ConstArray4{} :
            coverage->fab(li).const_array();
        const Box2D valid = rhs.box(li);
        moments[2 * b] += static_cast<double>(reduce_sum_cell(
            valid, detail::FieldBasisMomentKernel{
                       value, mask_array, coverage_array, basis.field_component,
                       mask != nullptr, coverage != nullptr, measure}));
        moments[2 * b + 1] += static_cast<double>(reduce_sum_cell(
            valid, detail::FieldBasisAbsMomentKernel{
                       value, mask_array, coverage_array, basis.field_component,
                       mask != nullptr, coverage != nullptr, measure}));
      }
    }
  }
  all_reduce_sum_inplace(moments.data(), static_cast<int>(moments.size()));
  for (std::size_t b = 0; b < plan.bases.size(); ++b) {
    const double scale = moments[2 * b + 1] > 1.0 ? moments[2 * b + 1] : 1.0;
    const double tolerance = 128.0 * std::numeric_limits<Real>::epsilon() * scale;
    if (std::abs(moments[2 * b]) > tolerance)
      throw std::runtime_error(
          "field RHS is incompatible with nullspace basis '" + plan.bases[b].identity +
          "' (" + plan.bases[b].provenance + "); silent projection is forbidden");
  }
  return moments;
}

inline std::vector<double> require_field_nullspace_compatible(
    const MultiFab& rhs, const FieldNullspacePlan& plan) {
  return require_field_nullspace_compatible(std::vector<const MultiFab*>{&rhs}, plan, 0);
}

/// Apply every declared gauge with one collective for all dot products and basis norms.  Masks must
/// be disjoint/orthogonal; non-orthogonal gauges require an explicit dense gauge solver and are
/// rejected at plan construction rather than being silently order-dependent.
inline void apply_field_gauge(const std::vector<MultiFab*>& phi_levels,
                              const FieldNullspacePlan& plan, int first_level = 0) {
  if (plan.gauges.empty())
    return;
  if (plan.identity.empty() || plan.layout_identity.empty())
    throw std::runtime_error("field nullspace plan requires exact plan and layout identities");
  if (plan.scope == FieldNullspaceScope::LevelLocal && phi_levels.size() != 1)
    throw std::runtime_error("level-local field gauges must be applied independently per level");
  if (plan.gauges.size() != plan.bases.size())
    throw std::runtime_error("field gauge must constrain every declared nullspace basis exactly once");
  std::vector<double> moments(plan.bases.size() * 2, 0.0);
  for (std::size_t b = 0; b < plan.bases.size(); ++b) {
    const auto& basis = plan.bases[b];
    if (detail::gauge_index(plan, basis.identity) < 0)
      throw std::runtime_error("field gauge references do not cover nullspace basis '" +
                               basis.identity + "'");
    for (int level = 0; level < static_cast<int>(phi_levels.size()); ++level) {
      MultiFab& phi = *phi_levels[static_cast<std::size_t>(level)];
      const int resolved_level = first_level + level;
      const MultiFab* mask = basis.mask(resolved_level);
      const MultiFab* coverage = basis.coverage_mask(resolved_level, plan.scope);
      const Real measure = basis.measure(resolved_level);
      detail::validate_basis_layout(phi, mask, basis);
      detail::validate_mask_layout(phi, coverage, "coverage");
      for (int li = 0; li < phi.local_size(); ++li) {
        const ConstArray4 value = phi.fab(li).const_array();
        const ConstArray4 mask_array = mask == nullptr ? ConstArray4{} :
            mask->fab(li).const_array();
        const ConstArray4 coverage_array = coverage == nullptr ? ConstArray4{} :
            coverage->fab(li).const_array();
        const Box2D valid = phi.box(li);
        moments[2 * b] += static_cast<double>(reduce_sum_cell(
            valid, detail::FieldBasisMomentKernel{
                       value, mask_array, coverage_array, basis.field_component,
                       mask != nullptr, coverage != nullptr, measure}));
        moments[2 * b + 1] += static_cast<double>(reduce_sum_cell(
            valid, detail::FieldBasisNormKernel{
                       mask_array, coverage_array, mask != nullptr, coverage != nullptr, measure}));
      }
    }
  }
  all_reduce_sum_inplace(moments.data(), static_cast<int>(moments.size()));
  for (std::size_t b = 0; b < plan.bases.size(); ++b) {
    const auto& basis = plan.bases[b];
    const int g = detail::gauge_index(plan, basis.identity);
    const Real norm = static_cast<Real>(moments[2 * b + 1]);
    if (!(norm > Real(0)))
      throw std::runtime_error("field nullspace basis has zero norm");
    const Real coefficient = static_cast<Real>(moments[2 * b]) / norm -
                             plan.gauges[static_cast<std::size_t>(g)].value;
    for (int level = 0; level < static_cast<int>(phi_levels.size()); ++level) {
      MultiFab& phi = *phi_levels[static_cast<std::size_t>(level)];
      const MultiFab* mask = basis.mask(first_level + level);
      for (int li = 0; li < phi.local_size(); ++li) {
        Array4 value = phi.fab(li).array();
        const ConstArray4 mask_array = mask == nullptr ? ConstArray4{} :
            mask->fab(li).const_array();
        for_each_cell(phi.box(li), detail::ShiftFieldBasisKernel{
            value, mask_array, basis.field_component, mask != nullptr, coefficient});
      }
    }
  }
}

inline void apply_field_gauge(MultiFab& phi, const FieldNullspacePlan& plan) {
  std::vector<MultiFab*> levels{&phi};
  apply_field_gauge(levels, plan);
}

inline FieldNullspacePlan constant_mean_zero_nullspace(std::string identity,
                                                       std::string provenance,
                                                       Real cell_measure = Real(1)) {
  FieldNullspacePlan result;
  result.identity = std::move(identity);
  result.layout_identity = result.identity + ":layout";
  result.scope = FieldNullspaceScope::Uniform;
  result.bases.push_back(FieldNullspaceBasis{
      result.identity + ":constant", std::move(provenance), result.identity + ":recipe",
      0, {}, {}, {cell_measure}});
  result.gauges.push_back(FieldGaugeConstraint{result.bases[0].identity, Real(0)});
  return result;
}

}  // namespace pops
