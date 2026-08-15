/// @file
/// @brief Compile-time-ranked MultiFab arithmetic over valid cells.

#pragma once

#include <pops/core/foundation/types.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/field_view.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace pops {

/// Optional active-cell and inverse-volume-fraction metrics on the exact field layout.
template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
struct RelativeCellMeasure {
  const MultiFab<Dim, MemorySpace>* active_cells = nullptr;
  const MultiFab<Dim, MemorySpace>* inverse_volume_fraction = nullptr;
};

namespace mf_arith_detail {

template <int Dim>
struct SaxpyKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> source{};
  Real factor = 0;
  int component = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    destination(index, component) += factor * source(index, component);
  }
};

template <int Dim>
struct ActiveSaxpyKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> source{};
  FieldView<const Real, Dim> active{};
  Real factor = 0;
  int component = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    if (active(index, 0) >= Real{0.5})
      destination(index, component) += factor * source(index, component);
  }
};

template <int Dim>
struct ScaleKernel {
  FieldView<Real, Dim> values{};
  Real factor = 0;
  int component = 0;

  POPS_HD void operator()(const Index<Dim>& index) const { values(index, component) *= factor; }
};

template <int Dim>
struct LincombKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> left{};
  FieldView<const Real, Dim> right{};
  Real left_factor = 0;
  Real right_factor = 0;
  int component = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    destination(index, component) =
        left_factor * left(index, component) + right_factor * right(index, component);
  }
};

template <int Dim>
struct ActiveLincombKernel {
  FieldView<Real, Dim> destination{};
  FieldView<const Real, Dim> left{};
  FieldView<const Real, Dim> right{};
  FieldView<const Real, Dim> active{};
  Real left_factor = 0;
  Real right_factor = 0;
  int component = 0;

  POPS_HD void operator()(const Index<Dim>& index) const {
    if (active(index, 0) >= Real{0.5})
      destination(index, component) =
          left_factor * left(index, component) + right_factor * right(index, component);
  }
};

template <int Dim>
struct SumKernel {
  FieldView<const Real, Dim> values{};
  int component = 0;
  POPS_HD Real operator()(const Index<Dim>& index) const { return values(index, component); }
};

template <int Dim>
struct AbsSumKernel {
  FieldView<const Real, Dim> values{};
  int component = 0;
  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real value = values(index, component);
    return value < 0 ? -value : value;
  }
};

template <int Dim>
struct MaxKernel {
  FieldView<const Real, Dim> values{};
  int component = 0;
  POPS_HD Real operator()(const Index<Dim>& index) const { return values(index, component); }
};

template <int Dim>
struct NegatedKernel {
  FieldView<const Real, Dim> values{};
  int component = 0;
  POPS_HD Real operator()(const Index<Dim>& index) const { return -values(index, component); }
};

template <int Dim>
struct NormInfKernel {
  FieldView<const Real, Dim> values{};
  int component = 0;
  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real value = values(index, component);
    const Real magnitude = value < 0 ? -value : value;
    return magnitude <= std::numeric_limits<Real>::max() ? magnitude
                                                         : std::numeric_limits<Real>::infinity();
  }
};

template <int Dim>
struct DotKernel {
  FieldView<const Real, Dim> left{};
  FieldView<const Real, Dim> right{};
  int component = 0;
  POPS_HD Real operator()(const Index<Dim>& index) const {
    return left(index, component) * right(index, component);
  }
};

template <int Dim>
struct DifferenceSqKernel {
  FieldView<const Real, Dim> current{};
  FieldView<const Real, Dim> previous{};
  int component = 0;
  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real difference = current(index, component) - previous(index, component);
    return difference * difference;
  }
};

template <int Dim>
struct MeasuredValueKernel {
  FieldView<const Real, Dim> values{};
  FieldView<const Real, Dim> active{};
  FieldView<const Real, Dim> inverse{};
  int component = 0;
  bool absolute = false;
  bool has_inverse = false;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    if (active(index, 0) < Real{0.5})
      return Real{0};
    Real value = values(index, component);
    if (absolute && value < 0)
      value = -value;
    return has_inverse ? value * inverse(index, 0) : value;
  }
};

template <int Dim>
struct MeasuredDotKernel {
  FieldView<const Real, Dim> left{};
  FieldView<const Real, Dim> right{};
  FieldView<const Real, Dim> active{};
  FieldView<const Real, Dim> inverse{};
  int component = 0;
  bool has_inverse = false;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    if (active(index, 0) < Real{0.5})
      return Real{0};
    Real value = left(index, component) * right(index, component);
    return has_inverse ? value * inverse(index, 0) : value;
  }
};

template <int Dim>
struct MeasuredDifferenceSqKernel {
  FieldView<const Real, Dim> current{};
  FieldView<const Real, Dim> previous{};
  FieldView<const Real, Dim> active{};
  FieldView<const Real, Dim> inverse{};
  int component = 0;
  bool has_inverse = false;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    if (active(index, 0) < Real{0.5})
      return Real{0};
    const Real difference = current(index, component) - previous(index, component);
    const Real value = difference * difference;
    return has_inverse ? value * inverse(index, 0) : value;
  }
};

template <int Dim>
struct ActiveMaxKernel {
  FieldView<const Real, Dim> values{};
  FieldView<const Real, Dim> active{};
  int component = 0;
  bool negate = false;
  bool magnitude = false;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    if (active(index, 0) < Real{0.5})
      return std::numeric_limits<Real>::lowest();
    Real value = values(index, component);
    if (magnitude) {
      value = value < 0 ? -value : value;
      if (!(value <= std::numeric_limits<Real>::max()))
        value = std::numeric_limits<Real>::infinity();
    }
    return negate ? -value : value;
  }
};

template <int Dim, class LeftSpace, class RightSpace>
void require_same_layout(const MultiFab<Dim, LeftSpace>& left,
                         const MultiFab<Dim, RightSpace>& right, const char* operation,
                         bool require_components = true) {
  if (left.layout() != right.layout() || left.distribution() != right.distribution() ||
      left.local_rank() != right.local_rank() ||
      (require_components && left.ncomp() != right.ncomp()))
    throw std::invalid_argument(std::string(operation) +
                                ": fields must have the same exact ND layout");
}

template <int Dim, class MemorySpace>
void require_component(const MultiFab<Dim, MemorySpace>& field, int component,
                       const char* operation) {
  if (component < 0 || component >= field.ncomp())
    throw std::out_of_range(std::string(operation) + ": component is outside the field");
}

template <int Dim, class MemorySpace>
void require_collective_identity(const MultiFab<Dim, MemorySpace>& field, const char* operation) {
  const std::size_t communicator_size = static_cast<std::size_t>(n_ranks());
  if (field.rank_space().size() != communicator_size ||
      field.rank_space().linear_rank(field.local_rank()) != static_cast<std::size_t>(my_rank()))
    throw std::logic_error(std::string(operation) +
                           ": ND rank space does not match the active communicator");
  if (field.distribution().replicated() && communicator_size != 1)
    throw std::logic_error(std::string(operation) +
                           ": collective reduction of replicated ND storage would overcount");
}

template <int Dim, class MemorySpace>
void validate_measure(const MultiFab<Dim, MemorySpace>& field,
                      const RelativeCellMeasure<Dim, MemorySpace>& measure, const char* operation) {
  if (measure.active_cells == nullptr) {
    if (measure.inverse_volume_fraction != nullptr)
      throw std::invalid_argument(std::string(operation) +
                                  ": inverse volume fraction requires an active-cell mask");
    return;
  }
  require_same_layout(field, *measure.active_cells, operation, false);
  if (measure.active_cells->ncomp() != 1)
    throw std::invalid_argument(std::string(operation) +
                                ": active-cell mask must have one component");
  if (measure.inverse_volume_fraction != nullptr) {
    require_same_layout(field, *measure.inverse_volume_fraction, operation, false);
    if (measure.inverse_volume_fraction->ncomp() != 1)
      throw std::invalid_argument(std::string(operation) +
                                  ": inverse volume fraction must have one component");
  }
}

}  // namespace mf_arith_detail

template <int Dim, class MemorySpace>
void saxpy(MultiFab<Dim, MemorySpace>& destination, Real factor,
           const MultiFab<Dim, MemorySpace>& source) {
  mf_arith_detail::require_same_layout(destination, source, "pops::saxpy");
  for (std::size_t local = 0; local < destination.local_size(); ++local)
    for (int component = 0; component < destination.ncomp(); ++component)
      for_each_cell(destination.box(local),
                    mf_arith_detail::SaxpyKernel<Dim>{destination.fab(local).view(),
                                                      source.fab(local).view(), factor, component});
}

template <int Dim, class MemorySpace>
void saxpy_active(MultiFab<Dim, MemorySpace>& destination, Real factor,
                  const MultiFab<Dim, MemorySpace>& source,
                  const MultiFab<Dim, MemorySpace>& active_cells) {
  mf_arith_detail::require_same_layout(destination, source, "pops::saxpy_active");
  mf_arith_detail::require_same_layout(destination, active_cells, "pops::saxpy_active", false);
  if (active_cells.ncomp() != 1)
    throw std::invalid_argument("pops::saxpy_active mask must have one component");
  for (std::size_t local = 0; local < destination.local_size(); ++local)
    for (int component = 0; component < destination.ncomp(); ++component)
      for_each_cell(destination.box(local),
                    mf_arith_detail::ActiveSaxpyKernel<Dim>{
                        destination.fab(local).view(), source.fab(local).view(),
                        active_cells.fab(local).view(), factor, component});
}

template <int Dim, class MemorySpace>
void scale(MultiFab<Dim, MemorySpace>& field, Real factor) {
  for (std::size_t local = 0; local < field.local_size(); ++local)
    for (int component = 0; component < field.ncomp(); ++component)
      for_each_cell(field.box(local),
                    mf_arith_detail::ScaleKernel<Dim>{field.fab(local).view(), factor, component});
}

template <int Dim, class MemorySpace>
void lincomb(MultiFab<Dim, MemorySpace>& destination, Real left_factor,
             const MultiFab<Dim, MemorySpace>& left, Real right_factor,
             const MultiFab<Dim, MemorySpace>& right) {
  mf_arith_detail::require_same_layout(destination, left, "pops::lincomb");
  mf_arith_detail::require_same_layout(destination, right, "pops::lincomb");
  for (std::size_t local = 0; local < destination.local_size(); ++local)
    for (int component = 0; component < destination.ncomp(); ++component)
      for_each_cell(destination.box(local),
                    mf_arith_detail::LincombKernel<Dim>{
                        destination.fab(local).view(), left.fab(local).view(),
                        right.fab(local).view(), left_factor, right_factor, component});
}

template <int Dim, class MemorySpace>
void lincomb_active(MultiFab<Dim, MemorySpace>& destination, Real left_factor,
                    const MultiFab<Dim, MemorySpace>& left, Real right_factor,
                    const MultiFab<Dim, MemorySpace>& right,
                    const MultiFab<Dim, MemorySpace>& active_cells) {
  mf_arith_detail::require_same_layout(destination, left, "pops::lincomb_active");
  mf_arith_detail::require_same_layout(destination, right, "pops::lincomb_active");
  mf_arith_detail::require_same_layout(destination, active_cells, "pops::lincomb_active", false);
  if (active_cells.ncomp() != 1)
    throw std::invalid_argument("pops::lincomb_active mask must have one component");
  for (std::size_t local = 0; local < destination.local_size(); ++local)
    for (int component = 0; component < destination.ncomp(); ++component)
      for_each_cell(
          destination.box(local),
          mf_arith_detail::ActiveLincombKernel<Dim>{
              destination.fab(local).view(), left.fab(local).view(), right.fab(local).view(),
              active_cells.fab(local).view(), left_factor, right_factor, component});
}

template <int Dim, class MemorySpace>
Real reduce_sum_local(const MultiFab<Dim, MemorySpace>& field, int component = 0) {
  mf_arith_detail::require_component(field, component, "pops::reduce_sum_local");
  Real result = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local)
    result += for_each_cell_reduce_sum(
        field.box(local), mf_arith_detail::SumKernel<Dim>{field.fab(local).view(), component});
  return result;
}

template <int Dim, class MemorySpace>
Real reduce_abs_sum_local(const MultiFab<Dim, MemorySpace>& field, int component = 0) {
  mf_arith_detail::require_component(field, component, "pops::reduce_abs_sum_local");
  Real result = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local)
    result += for_each_cell_reduce_sum(
        field.box(local), mf_arith_detail::AbsSumKernel<Dim>{field.fab(local).view(), component});
  return result;
}

template <int Dim, class MemorySpace>
Real reduce_max_local(const MultiFab<Dim, MemorySpace>& field, int component = 0) {
  mf_arith_detail::require_component(field, component, "pops::reduce_max_local");
  Real result = -std::numeric_limits<Real>::infinity();
  for (std::size_t local = 0; local < field.local_size(); ++local)
    result = std::max(result, for_each_cell_reduce_max(field.box(local),
                                                       mf_arith_detail::MaxKernel<Dim>{
                                                           field.fab(local).view(), component}));
  return result;
}

template <int Dim, class MemorySpace>
Real reduce_min_local(const MultiFab<Dim, MemorySpace>& field, int component = 0) {
  mf_arith_detail::require_component(field, component, "pops::reduce_min_local");
  Real result = std::numeric_limits<Real>::infinity();
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const Real negated = for_each_cell_reduce_max(
        field.box(local), mf_arith_detail::NegatedKernel<Dim>{field.fab(local).view(), component});
    result = std::min(result, -negated);
  }
  return result;
}

template <int Dim, class MemorySpace>
Real norm_inf(const MultiFab<Dim, MemorySpace>& field, int component = 0) {
  mf_arith_detail::require_component(field, component, "pops::norm_inf");
  Real result = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local)
    result = std::max(result, for_each_cell_reduce_max(field.box(local),
                                                       mf_arith_detail::NormInfKernel<Dim>{
                                                           field.fab(local).view(), component}));
  return result;
}

template <int Dim, class MemorySpace>
Real dot_local(const MultiFab<Dim, MemorySpace>& left, const MultiFab<Dim, MemorySpace>& right,
               int component = 0) {
  mf_arith_detail::require_same_layout(left, right, "pops::dot_local");
  mf_arith_detail::require_component(left, component, "pops::dot_local");
  Real result = 0;
  for (std::size_t local = 0; local < left.local_size(); ++local)
    result += for_each_cell_reduce_sum(
        left.box(local), mf_arith_detail::DotKernel<Dim>{left.fab(local).view(),
                                                         right.fab(local).view(), component});
  return result;
}

template <int Dim, class MemorySpace>
Real dot_all_local(const MultiFab<Dim, MemorySpace>& left,
                   const MultiFab<Dim, MemorySpace>& right) {
  mf_arith_detail::require_same_layout(left, right, "pops::dot_all_local");
  Real result = 0;
  for (int component = 0; component < left.ncomp(); ++component)
    result += dot_local(left, right, component);
  return result;
}

template <int Dim, class MemorySpace>
Real difference_sum_sq_all_local(const MultiFab<Dim, MemorySpace>& current,
                                 const MultiFab<Dim, MemorySpace>& previous) {
  mf_arith_detail::require_same_layout(current, previous, "pops::difference_sum_sq_all_local");
  Real result = 0;
  for (std::size_t local = 0; local < current.local_size(); ++local)
    for (int component = 0; component < current.ncomp(); ++component)
      result += for_each_cell_reduce_sum(
          current.box(local),
          mf_arith_detail::DifferenceSqKernel<Dim>{current.fab(local).view(),
                                                   previous.fab(local).view(), component});
  return result;
}

template <int Dim, class MemorySpace>
Real reduce_sum(const MultiFab<Dim, MemorySpace>& field, int component = 0) {
  mf_arith_detail::require_collective_identity(field, "pops::reduce_sum");
  return static_cast<Real>(all_reduce_sum(reduce_sum_local(field, component)));
}

template <int Dim, class MemorySpace>
Real reduce_abs_sum(const MultiFab<Dim, MemorySpace>& field, int component = 0) {
  mf_arith_detail::require_collective_identity(field, "pops::reduce_abs_sum");
  return static_cast<Real>(all_reduce_sum(reduce_abs_sum_local(field, component)));
}

template <int Dim, class MemorySpace>
Real reduce_max(const MultiFab<Dim, MemorySpace>& field, int component = 0) {
  mf_arith_detail::require_collective_identity(field, "pops::reduce_max");
  return static_cast<Real>(all_reduce_max(reduce_max_local(field, component)));
}

template <int Dim, class MemorySpace>
Real reduce_min(const MultiFab<Dim, MemorySpace>& field, int component = 0) {
  mf_arith_detail::require_collective_identity(field, "pops::reduce_min");
  return static_cast<Real>(all_reduce_min(reduce_min_local(field, component)));
}

template <int Dim, class MemorySpace>
Real reduce_norm_inf(const MultiFab<Dim, MemorySpace>& field, int component = 0) {
  mf_arith_detail::require_collective_identity(field, "pops::reduce_norm_inf");
  return static_cast<Real>(all_reduce_max(norm_inf(field, component)));
}

template <int Dim, class MemorySpace>
Real dot(const MultiFab<Dim, MemorySpace>& left, const MultiFab<Dim, MemorySpace>& right,
         int component = 0) {
  mf_arith_detail::require_same_layout(left, right, "pops::dot");
  mf_arith_detail::require_collective_identity(left, "pops::dot");
  return static_cast<Real>(all_reduce_sum(dot_local(left, right, component)));
}

template <int Dim, class MemorySpace>
Real dot_all(const MultiFab<Dim, MemorySpace>& left, const MultiFab<Dim, MemorySpace>& right) {
  mf_arith_detail::require_same_layout(left, right, "pops::dot_all");
  mf_arith_detail::require_collective_identity(left, "pops::dot_all");
  return static_cast<Real>(all_reduce_sum(dot_all_local(left, right)));
}

template <int Dim, class MemorySpace>
Real difference_sum_sq_all(const MultiFab<Dim, MemorySpace>& current,
                           const MultiFab<Dim, MemorySpace>& previous) {
  mf_arith_detail::require_same_layout(current, previous, "pops::difference_sum_sq_all");
  mf_arith_detail::require_collective_identity(current, "pops::difference_sum_sq_all");
  return static_cast<Real>(all_reduce_sum(difference_sum_sq_all_local(current, previous)));
}

template <int Dim, class MemorySpace>
Real reduce_sum(const MultiFab<Dim, MemorySpace>& field, int component,
                const RelativeCellMeasure<Dim, MemorySpace>& measure) {
  mf_arith_detail::require_component(field, component, "pops::reduce_sum(measure)");
  mf_arith_detail::validate_measure(field, measure, "pops::reduce_sum(measure)");
  if (measure.active_cells == nullptr)
    return reduce_sum(field, component);
  mf_arith_detail::require_collective_identity(field, "pops::reduce_sum(measure)");
  Real local_result = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const FieldView<const Real, Dim> inverse =
        measure.inverse_volume_fraction == nullptr
            ? FieldView<const Real, Dim>{}
            : measure.inverse_volume_fraction->fab(local).view();
    local_result += for_each_cell_reduce_sum(
        field.box(local),
        mf_arith_detail::MeasuredValueKernel<Dim>{
            field.fab(local).view(), measure.active_cells->fab(local).view(), inverse, component,
            false, measure.inverse_volume_fraction != nullptr});
  }
  return static_cast<Real>(all_reduce_sum(local_result));
}

template <int Dim, class MemorySpace>
Real reduce_abs_sum(const MultiFab<Dim, MemorySpace>& field, int component,
                    const RelativeCellMeasure<Dim, MemorySpace>& measure) {
  mf_arith_detail::require_component(field, component, "pops::reduce_abs_sum(measure)");
  mf_arith_detail::validate_measure(field, measure, "pops::reduce_abs_sum(measure)");
  if (measure.active_cells == nullptr)
    return reduce_abs_sum(field, component);
  mf_arith_detail::require_collective_identity(field, "pops::reduce_abs_sum(measure)");
  Real local_result = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const FieldView<const Real, Dim> inverse =
        measure.inverse_volume_fraction == nullptr
            ? FieldView<const Real, Dim>{}
            : measure.inverse_volume_fraction->fab(local).view();
    local_result += for_each_cell_reduce_sum(
        field.box(local),
        mf_arith_detail::MeasuredValueKernel<Dim>{
            field.fab(local).view(), measure.active_cells->fab(local).view(), inverse, component,
            true, measure.inverse_volume_fraction != nullptr});
  }
  return static_cast<Real>(all_reduce_sum(local_result));
}

template <int Dim, class MemorySpace>
Real dot_local(const MultiFab<Dim, MemorySpace>& left, const MultiFab<Dim, MemorySpace>& right,
               int component, const RelativeCellMeasure<Dim, MemorySpace>& measure) {
  mf_arith_detail::require_same_layout(left, right, "pops::dot(measure)");
  mf_arith_detail::require_component(left, component, "pops::dot(measure)");
  mf_arith_detail::validate_measure(left, measure, "pops::dot(measure)");
  if (measure.active_cells == nullptr)
    return dot_local(left, right, component);
  Real local_result = 0;
  for (std::size_t local = 0; local < left.local_size(); ++local) {
    const FieldView<const Real, Dim> inverse =
        measure.inverse_volume_fraction == nullptr
            ? FieldView<const Real, Dim>{}
            : measure.inverse_volume_fraction->fab(local).view();
    local_result += for_each_cell_reduce_sum(
        left.box(local), mf_arith_detail::MeasuredDotKernel<Dim>{
                             left.fab(local).view(), right.fab(local).view(),
                             measure.active_cells->fab(local).view(), inverse, component,
                             measure.inverse_volume_fraction != nullptr});
  }
  return local_result;
}

template <int Dim, class MemorySpace>
Real dot(const MultiFab<Dim, MemorySpace>& left, const MultiFab<Dim, MemorySpace>& right,
         int component, const RelativeCellMeasure<Dim, MemorySpace>& measure) {
  mf_arith_detail::require_collective_identity(left, "pops::dot(measure)");
  return static_cast<Real>(all_reduce_sum(dot_local(left, right, component, measure)));
}

template <int Dim, class MemorySpace>
Real dot_all(const MultiFab<Dim, MemorySpace>& left, const MultiFab<Dim, MemorySpace>& right,
             const RelativeCellMeasure<Dim, MemorySpace>& measure) {
  mf_arith_detail::require_same_layout(left, right, "pops::dot_all(measure)");
  mf_arith_detail::validate_measure(left, measure, "pops::dot_all(measure)");
  if (measure.active_cells == nullptr)
    return dot_all(left, right);
  mf_arith_detail::require_collective_identity(left, "pops::dot_all(measure)");
  Real local_result = 0;
  for (std::size_t local = 0; local < left.local_size(); ++local) {
    const FieldView<const Real, Dim> inverse =
        measure.inverse_volume_fraction == nullptr
            ? FieldView<const Real, Dim>{}
            : measure.inverse_volume_fraction->fab(local).view();
    for (int component = 0; component < left.ncomp(); ++component)
      local_result += for_each_cell_reduce_sum(
          left.box(local), mf_arith_detail::MeasuredDotKernel<Dim>{
                               left.fab(local).view(), right.fab(local).view(),
                               measure.active_cells->fab(local).view(), inverse, component,
                               measure.inverse_volume_fraction != nullptr});
  }
  return static_cast<Real>(all_reduce_sum(local_result));
}

template <int Dim, class MemorySpace>
Real difference_sum_sq_all(const MultiFab<Dim, MemorySpace>& current,
                           const MultiFab<Dim, MemorySpace>& previous,
                           const RelativeCellMeasure<Dim, MemorySpace>& measure) {
  mf_arith_detail::require_same_layout(current, previous, "pops::difference_sum_sq_all(measure)");
  mf_arith_detail::validate_measure(current, measure, "pops::difference_sum_sq_all(measure)");
  if (measure.active_cells == nullptr)
    return difference_sum_sq_all(current, previous);
  mf_arith_detail::require_collective_identity(current, "pops::difference_sum_sq_all(measure)");
  Real local_result = 0;
  for (std::size_t local = 0; local < current.local_size(); ++local) {
    const FieldView<const Real, Dim> inverse =
        measure.inverse_volume_fraction == nullptr
            ? FieldView<const Real, Dim>{}
            : measure.inverse_volume_fraction->fab(local).view();
    for (int component = 0; component < current.ncomp(); ++component)
      local_result += for_each_cell_reduce_sum(
          current.box(local), mf_arith_detail::MeasuredDifferenceSqKernel<Dim>{
                                  current.fab(local).view(), previous.fab(local).view(),
                                  measure.active_cells->fab(local).view(), inverse, component,
                                  measure.inverse_volume_fraction != nullptr});
  }
  return static_cast<Real>(all_reduce_sum(local_result));
}

template <int Dim, class MemorySpace>
Real reduce_max(const MultiFab<Dim, MemorySpace>& field, int component,
                const RelativeCellMeasure<Dim, MemorySpace>& measure) {
  mf_arith_detail::require_component(field, component, "pops::reduce_max(measure)");
  mf_arith_detail::validate_measure(field, measure, "pops::reduce_max(measure)");
  if (measure.active_cells == nullptr)
    return reduce_max(field, component);
  mf_arith_detail::require_collective_identity(field, "pops::reduce_max(measure)");
  Real local_result = -std::numeric_limits<Real>::infinity();
  for (std::size_t local = 0; local < field.local_size(); ++local)
    local_result = std::max(
        local_result,
        for_each_cell_reduce_max(
            field.box(local), mf_arith_detail::ActiveMaxKernel<Dim>{
                                  field.fab(local).view(), measure.active_cells->fab(local).view(),
                                  component, false, false}));
  return static_cast<Real>(all_reduce_max(local_result));
}

template <int Dim, class MemorySpace>
Real reduce_min(const MultiFab<Dim, MemorySpace>& field, int component,
                const RelativeCellMeasure<Dim, MemorySpace>& measure) {
  mf_arith_detail::require_component(field, component, "pops::reduce_min(measure)");
  mf_arith_detail::validate_measure(field, measure, "pops::reduce_min(measure)");
  if (measure.active_cells == nullptr)
    return reduce_min(field, component);
  mf_arith_detail::require_collective_identity(field, "pops::reduce_min(measure)");
  Real local_result = std::numeric_limits<Real>::infinity();
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const Real negated = for_each_cell_reduce_max(
        field.box(local), mf_arith_detail::ActiveMaxKernel<Dim>{
                              field.fab(local).view(), measure.active_cells->fab(local).view(),
                              component, true, false});
    local_result = std::min(local_result, -negated);
  }
  return static_cast<Real>(all_reduce_min(local_result));
}

template <int Dim, class MemorySpace>
Real reduce_norm_inf(const MultiFab<Dim, MemorySpace>& field, int component,
                     const RelativeCellMeasure<Dim, MemorySpace>& measure) {
  mf_arith_detail::require_component(field, component, "pops::reduce_norm_inf(measure)");
  mf_arith_detail::validate_measure(field, measure, "pops::reduce_norm_inf(measure)");
  if (measure.active_cells == nullptr)
    return reduce_norm_inf(field, component);
  mf_arith_detail::require_collective_identity(field, "pops::reduce_norm_inf(measure)");
  Real local_result = 0;
  for (std::size_t local = 0; local < field.local_size(); ++local)
    local_result = std::max(
        local_result,
        for_each_cell_reduce_max(
            field.box(local), mf_arith_detail::ActiveMaxKernel<Dim>{
                                  field.fab(local).view(), measure.active_cells->fab(local).view(),
                                  component, false, true}));
  return static_cast<Real>(all_reduce_max(local_result));
}

}  // namespace pops
