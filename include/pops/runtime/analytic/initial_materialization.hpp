/// @file
/// @brief Compile-time-ranked native materialization of validated analytic programs.

#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/analytic/expression.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::analytic {

using AnalyticOpcodeRows = std::vector<std::vector<std::string>>;
using AnalyticLiteralRows = std::vector<std::vector<double>>;

/// Compile the flat binding representation through the same strict native validator used by C++.
inline std::vector<AnalyticProgram> compile_component_programs(
    const AnalyticOpcodeRows& opcodes, const AnalyticLiteralRows& literals) {
  if (opcodes.empty() || opcodes.size() != literals.size())
    throw std::invalid_argument(
        "analytic initial state requires one opcode/literal row per component");
  std::vector<AnalyticProgram> result;
  result.reserve(opcodes.size());
  for (std::size_t component = 0; component < opcodes.size(); ++component) {
    if (opcodes[component].empty() || opcodes[component].size() != literals[component].size())
      throw std::invalid_argument("analytic initial component has mismatched opcode/literal rows");
    std::vector<AnalyticToken> tokens;
    tokens.reserve(opcodes[component].size());
    for (std::size_t index = 0; index < opcodes[component].size(); ++index) {
      const AnalyticOp op = analytic_op_from_name(opcodes[component][index]);
      const double raw = literals[component][index];
      if (!std::isfinite(raw))
        throw std::invalid_argument("analytic initial token literal must be finite");
      tokens.push_back(AnalyticToken{op, static_cast<Real>(raw)});
    }
    AnalyticProgram program = compile_analytic_postfix(tokens);
    if (program.result_type() != AnalyticValueType::Scalar)
      throw std::invalid_argument("analytic initial component must produce one scalar value");
    result.push_back(std::move(program));
  }
  return result;
}

namespace detail {

POPS_HD inline Real gauss_node(int index) {
  switch (index) {
    case 0:
      return Real(-0.861136311594052575223946488892809505);
    case 1:
      return Real(-0.339981043584856264802665759103244687);
    case 2:
      return Real(0.339981043584856264802665759103244687);
    default:
      return Real(0.861136311594052575223946488892809505);
  }
}

POPS_HD inline Real gauss_weight(int index) {
  return index == 0 || index == 3 ? Real(0.347854845137453857373063949221999408)
                                  : Real(0.652145154862546142626936050778000593);
}

template <int Dim>
struct AnalyticCellAverage {
  AnalyticProgramView program;
  Geometry<Dim> geometry;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    constexpr int sample_count = 1 << (2 * Dim);  // 4^Dim tensor quadrature points.
    constexpr Real normalization = Real(1) / static_cast<Real>(1 << Dim);
    const RealVector<Dim> center = geometry.cell_center(index);
    Real integral = Real(0);
    for (int sample = 0; sample < sample_count; ++sample) {
      int encoded = sample;
      Real weight = Real(1);
      RealVector<Dim> point = center;
      for (int axis = 0; axis < Dim; ++axis) {
        const int quadrature_index = encoded & 3;
        encoded >>= 2;
        weight *= gauss_weight(quadrature_index);
        point[axis] += Real(0.5) * geometry.spacing(axis) * gauss_node(quadrature_index);
      }
      integral += weight * program.eval(point);
    }
    return normalization * integral;
  }
};

template <int Dim>
struct AnalyticInitialKernel {
  FieldView<Real, Dim> values;
  int component;
  AnalyticCellAverage<Dim> average;

  POPS_HD void operator()(const Index<Dim>& index) const {
    values(index, component) = average(index);
  }
};

template <int Dim>
struct AnalyticInitialFiniteKernel {
  AnalyticCellAverage<Dim> average;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    return Kokkos::isfinite(average(index)) ? Real(0) : Real(1);
  }
};

template <int Dim, class MemorySpace>
struct AnalyticMappedInputField {
  const MultiFab<Dim, MemorySpace>* field = nullptr;
  int component = 0;
};

template <int Dim>
struct AnalyticMappedInputPatch {
  static_assert(std::is_trivially_copyable_v<FieldView<const Real, Dim>>,
                "mapped analytic input views must be device-copyable");

  FieldView<const Real, Dim> fields[kAnalyticMaxStack]{};
  int components[kAnalyticMaxStack]{};
  int count = 0;

  POPS_HD Real operator()(const Index<Dim>& index, int slot) const {
    return fields[slot](index, components[slot]);
  }
};

template <int Dim>
struct AnalyticMappedInitialKernel {
  AnalyticMappedInputPatch<Dim> inputs;
  FieldView<Real, Dim> values;
  int component;
  AnalyticProgramView program;
  Geometry<Dim> geometry;

  POPS_HD void operator()(const Index<Dim>& index) const {
    Real inputs[kAnalyticMaxStack];
    for (int slot = 0; slot < this->inputs.count; ++slot)
      inputs[slot] = this->inputs(index, slot);
    const AnalyticEvaluation result = program.eval_checked(
        index, geometry, inputs, static_cast<std::uint8_t>(this->inputs.count));
    values(index, component) = result.valid ? result.value : std::numeric_limits<Real>::quiet_NaN();
  }
};

template <int Dim>
struct AnalyticMappedInitialFiniteKernel {
  AnalyticMappedInputPatch<Dim> inputs;
  AnalyticProgramView program;
  Geometry<Dim> geometry;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    Real inputs[kAnalyticMaxStack];
    for (int slot = 0; slot < this->inputs.count; ++slot)
      inputs[slot] = this->inputs(index, slot);
    return program.eval_checked(index, geometry, inputs,
                                static_cast<std::uint8_t>(this->inputs.count))
                   .valid
               ? Real(0)
               : Real(1);
  }
};

template <int Dim>
struct GaussianCellAverage {
  Geometry<Dim> geometry;
  RealVector<Dim> center;
  Real background;
  Real amplitude;
  Real inverse_width;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real root = Kokkos::sqrt(inverse_width);
    const Real pi = Real(3.141592653589793238462643383279502884);
    Real average = Real(1);
    for (int axis = 0; axis < Dim; ++axis) {
      const Real spacing = geometry.spacing(axis);
      const Real lower = root * (geometry.face_coordinate(axis, index[axis]) - center[axis]);
      const Real upper = root * (geometry.face_coordinate(axis, index[axis] + 1) - center[axis]);
      const Real scale = Kokkos::sqrt(pi) / (Real(2) * root * spacing);
      average *= scale * (Kokkos::erf(upper) - Kokkos::erf(lower));
    }
    return background + amplitude * average;
  }
};

template <int Dim>
struct GaussianCellAverageKernel {
  FieldView<Real, Dim> values;
  GaussianCellAverage<Dim> average;

  POPS_HD void operator()(const Index<Dim>& index) const { values(index, 0) = average(index); }
};

template <int Dim>
struct GaussianCellAverageFiniteKernel {
  GaussianCellAverage<Dim> average;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    return Kokkos::isfinite(average(index)) ? Real(0) : Real(1);
  }
};

template <int Dim, class MemorySpace>
long invalid_materialization_target(const MultiFab<Dim, MemorySpace>& values,
                                    const Geometry<Dim>& geometry,
                                    const std::vector<AnalyticProgram>& programs) {
  bool invalid = programs.size() != static_cast<std::size_t>(values.ncomp());
  for (const AnalyticProgram& program : programs)
    invalid = invalid || program.required_dimension() > Dim;
  for (std::size_t local = 0; local < values.local_size(); ++local)
    invalid = invalid || !geometry.domain().contains(values.box(local));
  return all_reduce_sum(invalid ? 1L : 0L);
}

template <int Dim>
std::int64_t checked_layout_cell_count(const mesh::BoxArray<Dim>& layout) {
  std::int64_t total = 0;
  for (const Box<Dim>& box : layout.boxes()) {
    std::int64_t cells = 1;
    for (int axis = 0; axis < Dim; ++axis) {
      const std::int64_t length = box.length(axis);
      if (length <= 0 || cells > std::numeric_limits<std::int64_t>::max() / length)
        throw std::overflow_error("analytic materialization cell count exceeds int64_t");
      cells *= length;
    }
    if (total > std::numeric_limits<std::int64_t>::max() - cells)
      throw std::overflow_error("analytic materialization cell count exceeds int64_t");
    total += cells;
  }
  return total;
}

}  // namespace detail

/// Project each expression to cell averages with one Dim-generic tensor Gauss--Legendre algorithm.
template <int Dim, class MemorySpace>
std::int64_t materialize_cell_average(MultiFab<Dim, MemorySpace>& values,
                                      const Geometry<Dim>& geometry,
                                      const std::vector<AnalyticProgram>& programs) {
  if (detail::invalid_materialization_target(values, geometry, programs) != 0)
    throw std::invalid_argument("analytic initial materialization target/profile mismatch");
  long invalid_local = 0;
  for (std::size_t local = 0; local < values.local_size(); ++local)
    for (int component = 0; component < values.ncomp(); ++component)
      invalid_local += static_cast<long>(for_each_cell_reduce_sum(
          values.box(local),
          detail::AnalyticInitialFiniteKernel<Dim>{
              {programs[static_cast<std::size_t>(component)].view(), geometry}}));
  const long invalid = all_reduce_sum(invalid_local);
  if (invalid != 0)
    throw std::runtime_error("analytic initial expression produced non-finite cell values (count=" +
                             std::to_string(invalid) + ")");
  for (std::size_t local = 0; local < values.local_size(); ++local)
    for (int component = 0; component < values.ncomp(); ++component)
      for_each_cell(values.box(local),
                    detail::AnalyticInitialKernel<Dim>{
                        values.fab(local).view(),
                        component,
                        {programs[static_cast<std::size_t>(component)].view(), geometry}});
  device_fence();
  return detail::checked_layout_cell_count(values.layout()) * values.ncomp();
}

/// Evaluate mapped analytic state at cell centers using ranked native field views.
template <int Dim, class MemorySpace>
std::int64_t materialize_discrete_mapped_state(
    MultiFab<Dim, MemorySpace>& values, const Geometry<Dim>& geometry,
    const std::vector<AnalyticProgram>& programs,
    const std::vector<detail::AnalyticMappedInputField<Dim, MemorySpace>>& inputs) {
  bool target_invalid = detail::invalid_materialization_target(values, geometry, programs) != 0;
  target_invalid = target_invalid || inputs.size() > kAnalyticMaxStack;
  for (const auto& input : inputs)
    target_invalid =
        target_invalid || input.field == nullptr || input.component < 0 ||
        (input.field != nullptr && input.component >= input.field->ncomp()) ||
        (input.field != nullptr && input.field->layout() != values.layout()) ||
        (input.field != nullptr && input.field->distribution() != values.distribution()) ||
        (input.field != nullptr && input.field->local_rank() != values.local_rank()) ||
        (input.field != nullptr && input.field->local_size() != values.local_size());
  if (all_reduce_sum(target_invalid ? 1L : 0L) != 0)
    throw std::invalid_argument("analytic mapped initial state target/profile mismatch");
  long invalid_local = 0;
  for (std::size_t local = 0; local < values.local_size(); ++local) {
    detail::AnalyticMappedInputPatch<Dim> patch_inputs{};
    patch_inputs.count = static_cast<int>(inputs.size());
    for (std::size_t slot = 0; slot < inputs.size(); ++slot) {
      patch_inputs.fields[slot] = inputs[slot].field->fab(local).view();
      patch_inputs.components[slot] = inputs[slot].component;
    }
    for (int component = 0; component < values.ncomp(); ++component)
      invalid_local += static_cast<long>(for_each_cell_reduce_sum(
          values.box(local),
          detail::AnalyticMappedInitialFiniteKernel<Dim>{
              patch_inputs, programs[static_cast<std::size_t>(component)].view(), geometry}));
  }
  const long invalid = all_reduce_sum(invalid_local);
  if (invalid != 0)
    throw std::runtime_error(
        "analytic mapped initial expression produced non-finite cell values (count=" +
        std::to_string(invalid) + ")");
  for (std::size_t local = 0; local < values.local_size(); ++local) {
    detail::AnalyticMappedInputPatch<Dim> patch_inputs{};
    patch_inputs.count = static_cast<int>(inputs.size());
    for (std::size_t slot = 0; slot < inputs.size(); ++slot) {
      patch_inputs.fields[slot] = inputs[slot].field->fab(local).view();
      patch_inputs.components[slot] = inputs[slot].component;
    }
    const auto output = values.fab(local).view();
    for (int component = 0; component < values.ncomp(); ++component)
      for_each_cell(values.box(local),
                    detail::AnalyticMappedInitialKernel<Dim>{
                        patch_inputs, output, component,
                        programs[static_cast<std::size_t>(component)].view(), geometry});
  }
  device_fence();
  return detail::checked_layout_cell_count(values.layout()) * values.ncomp();
}

/// Materialize a separable Gaussian cell average in the selected compile-time rank.
template <int Dim, class MemorySpace>
std::int64_t materialize_gaussian_cell_average(MultiFab<Dim, MemorySpace>& values,
                                               const Geometry<Dim>& geometry,
                                               const RealVector<Dim>& center, Real background,
                                               Real amplitude, Real inverse_width) {
  bool invalid = values.ncomp() != 1 || !(inverse_width > Real(0)) ||
                 !std::isfinite(static_cast<double>(background)) ||
                 !std::isfinite(static_cast<double>(amplitude)) ||
                 !std::isfinite(static_cast<double>(inverse_width));
  for (int axis = 0; axis < Dim; ++axis)
    invalid = invalid || !std::isfinite(center[axis]);
  for (std::size_t local = 0; local < values.local_size(); ++local)
    invalid = invalid || !geometry.domain().contains(values.box(local));
  if (all_reduce_sum(invalid ? 1L : 0L) != 0)
    throw std::invalid_argument("analytic Gaussian initial profile is invalid");

  const detail::GaussianCellAverage<Dim> average{geometry, center, background, amplitude,
                                                 inverse_width};
  long invalid_local = 0;
  for (std::size_t local = 0; local < values.local_size(); ++local)
    invalid_local += static_cast<long>(for_each_cell_reduce_sum(
        values.box(local), detail::GaussianCellAverageFiniteKernel<Dim>{average}));
  const long non_finite = all_reduce_sum(invalid_local);
  if (non_finite != 0)
    throw std::runtime_error("analytic Gaussian profile produced non-finite cell averages (count=" +
                             std::to_string(non_finite) + ")");
  for (std::size_t local = 0; local < values.local_size(); ++local)
    for_each_cell(values.box(local),
                  detail::GaussianCellAverageKernel<Dim>{values.fab(local).view(), average});
  device_fence();
  return detail::checked_layout_cell_count(values.layout());
}

}  // namespace pops::analytic
