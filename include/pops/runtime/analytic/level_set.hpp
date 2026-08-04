/// @file
/// @brief Device-safe ranked level-set adapter and transactional materialization.

#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/runtime/analytic/expression.hpp>

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace pops::analytic {

/// Lightweight callable used by embedded-boundary and cut-cell kernels. The owning program must
/// outlive every kernel that captures this view.
template <int Dim>
struct AnalyticLevelSet {
  AnalyticProgramView expression{};

  POPS_HD Real level_set(const RealVector<Dim>& point) const { return expression.eval(point); }
  POPS_HD Real operator()(const RealVector<Dim>& point) const { return level_set(point); }
  POPS_HD bool cell_active(const RealVector<Dim>& point) const {
    return level_set(point) < Real(0);
  }
};

static_assert(std::is_trivially_copyable_v<AnalyticLevelSet<1>>);
static_assert(std::is_trivially_copyable_v<AnalyticLevelSet<2>>);
static_assert(std::is_trivially_copyable_v<AnalyticLevelSet<3>>);

/// Validate the static level-set contract before a view can reach a ranked kernel.
template <int Dim>
AnalyticLevelSet<Dim> make_analytic_level_set(const AnalyticProgram& program) {
  if (program.empty())
    throw std::invalid_argument("analytic level set: program must not be empty");
  if (program.result_type() != AnalyticValueType::Scalar)
    throw std::invalid_argument("analytic level set: expression must have scalar result type");
  if (program.required_dimension() > Dim)
    throw std::invalid_argument("analytic level set: expression requires a higher spatial rank");
  return AnalyticLevelSet<Dim>{program.view()};
}

namespace detail {

template <int Dim>
Extent<Dim> uniform_ghost_extent(int width) {
  Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = width;
  return result;
}

template <int Dim>
struct MaterializeAnalyticLevelSetKernel {
  AnalyticLevelSet<Dim> level_set;
  Geometry<Dim> geometry;
  FieldView<Real, Dim> values;
  FieldView<Real, Dim> active_mask;

  POPS_HD void operator()(const Index<Dim>& index) const {
    const Real value = level_set(geometry.cell_center(index));
    values(index) = value;
    active_mask(index) = value < Real(0) ? Real(1) : Real(0);
  }
};

template <int Dim>
struct NonFiniteLevelSetIndicator {
  FieldView<const Real, Dim> values;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    return Kokkos::isfinite(values(index)) ? Real(0) : Real(1);
  }
};

template <int Dim>
void validate_materialization_request(const AnalyticProgram& program,
                                      const Geometry<Dim>& geometry, const Box<Dim>& valid,
                                      int n_ghost) {
  (void)make_analytic_level_set<Dim>(program);
  if (geometry.domain().empty())
    throw std::invalid_argument("analytic level set: geometry domain must not be empty");
  if (valid.empty())
    throw std::invalid_argument("analytic level set: materialization box must not be empty");
  if (!geometry.domain().contains(valid))
    throw std::invalid_argument(
        "analytic level set: materialization box must be contained in the geometry domain");
  if (n_ghost < 0)
    throw std::invalid_argument("analytic level set: ghost width must be non-negative");
}

}  // namespace detail

/// Materialized signed values and staircase active mask over the same ranked valid box and ghosts.
/// The object is published only after every sampled value passes the finite-value preflight.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct AnalyticLevelSetMaterialization {
  Fab<Dim, MemorySpace> values;
  Fab<Dim, MemorySpace> active_mask;
  int ghost_width = 0;

  AnalyticLevelSetMaterialization() = default;
  AnalyticLevelSetMaterialization(const Box<Dim>& valid, int n_ghost)
      : values(valid, 1, detail::uniform_ghost_extent<Dim>(n_ghost)),
        active_mask(valid, 1, detail::uniform_ghost_extent<Dim>(n_ghost)),
        ghost_width(n_ghost) {}

  [[nodiscard]] const Box<Dim>& box() const noexcept { return values.box(); }
  [[nodiscard]] const Box<Dim>& grown_box() const noexcept { return values.grown_box(); }
  [[nodiscard]] int n_ghost() const noexcept { return ghost_width; }
};

static_assert(std::is_nothrow_move_assignable_v<AnalyticLevelSetMaterialization<1>>);
static_assert(std::is_nothrow_move_assignable_v<AnalyticLevelSetMaterialization<2>>);
static_assert(std::is_nothrow_move_assignable_v<AnalyticLevelSetMaterialization<3>>);

/// Evaluate one scalar analytic program at every cell center of valid.grow(n_ghost).
///
/// Values and mask live in temporary storage until a second device pass proves all values finite.
/// On failure the temporary is discarded. The active convention is strict: phi < 0 is active.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
AnalyticLevelSetMaterialization<Dim, MemorySpace> materialize_analytic_level_set(
    const AnalyticProgram& program, const Geometry<Dim>& geometry, const Box<Dim>& valid,
    int n_ghost) {
  detail::validate_materialization_request(program, geometry, valid, n_ghost);
  AnalyticLevelSetMaterialization<Dim, MemorySpace> staged(valid, n_ghost);
  const Box<Dim> sampled = staged.grown_box();

  for_each_cell(sampled, detail::MaterializeAnalyticLevelSetKernel<Dim>{
                             make_analytic_level_set<Dim>(program), geometry, staged.values.view(),
                             staged.active_mask.view()});
  const Real has_non_finite = for_each_cell_reduce_max(
      sampled, detail::NonFiniteLevelSetIndicator<Dim>{
                   static_cast<const Fab<Dim, MemorySpace>&>(staged.values).view()});
  if (has_non_finite != Real(0))
    throw std::domain_error(
        "analytic level set: expression produced a non-finite value on the sampled box");
  return staged;
}

/// Strong transactional replacement for runtime owners that already hold a materialization.
template <int Dim, class MemorySpace>
void replace_analytic_level_set_materialization(
    AnalyticLevelSetMaterialization<Dim, MemorySpace>& destination,
    const AnalyticProgram& program, const Geometry<Dim>& geometry, const Box<Dim>& valid,
    int n_ghost) {
  auto staged = materialize_analytic_level_set<Dim, MemorySpace>(program, geometry, valid, n_ghost);
  destination = std::move(staged);
}

}  // namespace pops::analytic
