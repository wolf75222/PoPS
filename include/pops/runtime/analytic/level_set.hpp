/// @file
/// @brief Device-safe level-set adapter and transactional materialization for analytic programs.

#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/fab2d.hpp>
#include <pops/runtime/analytic/expression.hpp>

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace pops::analytic {

/// Lightweight callable used by embedded-boundary and cut-cell kernels.  The owning
/// AnalyticProgram must outlive every kernel that captures this view.
struct AnalyticLevelSet {
  AnalyticProgramView expression{};

  POPS_HD Real level_set(Real x, Real y) const { return expression.eval(x, y); }
  POPS_HD Real operator()(Real x, Real y) const { return level_set(x, y); }
  POPS_HD bool cell_active(Real x, Real y) const { return level_set(x, y) < Real(0); }
};

static_assert(std::is_trivially_copyable_v<AnalyticLevelSet>);

/// Validate the static level-set contract before a view can reach a kernel.
inline AnalyticLevelSet make_analytic_level_set(const AnalyticProgram& program) {
  if (program.empty())
    throw std::invalid_argument("analytic level set: program must not be empty");
  if (program.result_type() != AnalyticValueType::Scalar)
    throw std::invalid_argument("analytic level set: expression must have scalar result type");
  return AnalyticLevelSet{program.view()};
}

/// Materialized signed values and their staircase active mask over the same valid box and ghosts.
/// This object is published only after every sampled value has passed the finite-value preflight.
struct AnalyticLevelSetMaterialization {
  Fab2D values;
  Fab2D active_mask;

  AnalyticLevelSetMaterialization() = default;
  AnalyticLevelSetMaterialization(const Box2D& valid, int n_ghost)
      : values(valid, 1, n_ghost), active_mask(valid, 1, n_ghost) {}

  [[nodiscard]] const Box2D& box() const noexcept { return values.box(); }
  [[nodiscard]] const Box2D& grown_box() const noexcept { return values.grown_box(); }
  [[nodiscard]] int n_ghost() const noexcept { return values.n_ghost(); }
};

static_assert(std::is_nothrow_move_assignable_v<AnalyticLevelSetMaterialization>,
              "transactional publication requires a no-throw materialization move");

namespace detail {

struct MaterializeAnalyticLevelSetKernel {
  AnalyticLevelSet level_set;
  Geometry geometry;
  Array4 values;
  Array4 active_mask;

  POPS_HD void operator()(int i, int j) const {
    const Real value = level_set(geometry.x_cell(i), geometry.y_cell(j));
    values(i, j) = value;
    active_mask(i, j) = value < Real(0) ? Real(1) : Real(0);
  }
};

struct NonFiniteLevelSetIndicator {
  ConstArray4 values;

  POPS_HD Real operator()(int i, int j) const {
    return Kokkos::isfinite(values(i, j)) ? Real(0) : Real(1);
  }
};

inline void validate_materialization_request(const AnalyticProgram& program,
                                             const Geometry& geometry, const Box2D& valid,
                                             int n_ghost) {
  (void)make_analytic_level_set(program);
  if (geometry.domain.empty())
    throw std::invalid_argument("analytic level set: geometry domain must not be empty");
  if (valid.empty())
    throw std::invalid_argument("analytic level set: materialization box must not be empty");
  if (!geometry.domain.contains(valid))
    throw std::invalid_argument(
        "analytic level set: materialization box must be contained in the geometry domain");
  if (n_ghost < 0)
    throw std::invalid_argument("analytic level set: ghost width must be non-negative");
}

}  // namespace detail

/// Evaluate one scalar analytic program at every cell center of valid.grow(n_ghost).
///
/// The values and mask live in temporary storage until a second device pass proves all values
/// finite.  On failure, the temporary is discarded and no caller-owned state has been modified.
/// The active convention is strict and shared with the EB core: phi < 0 is active; phi == 0 is not.
inline AnalyticLevelSetMaterialization materialize_analytic_level_set(
    const AnalyticProgram& program, const Geometry& geometry, const Box2D& valid, int n_ghost) {
  detail::validate_materialization_request(program, geometry, valid, n_ghost);
  AnalyticLevelSetMaterialization staged(valid, n_ghost);
  const Box2D sampled = staged.grown_box();

  for_each_cell(sampled,
                detail::MaterializeAnalyticLevelSetKernel{
                    make_analytic_level_set(program), geometry, staged.values.array(),
                    staged.active_mask.array()});
  const Real has_non_finite = for_each_cell_reduce_max(
      sampled, detail::NonFiniteLevelSetIndicator{staged.values.const_array()});
  if (has_non_finite != Real(0))
    throw std::domain_error(
        "analytic level set: expression produced a non-finite value on the sampled box");
  return staged;
}

/// Strong transactional replacement for runtime owners that already hold a materialization.
inline void replace_analytic_level_set_materialization(
    AnalyticLevelSetMaterialization& destination, const AnalyticProgram& program,
    const Geometry& geometry, const Box2D& valid, int n_ghost) {
  AnalyticLevelSetMaterialization staged =
      materialize_analytic_level_set(program, geometry, valid, n_ghost);
  destination = std::move(staged);
}

}  // namespace pops::analytic
