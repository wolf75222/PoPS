/// @file
/// @brief Native cell-centred materialization of validated analytic programs.

#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/runtime/analytic/expression.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
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
      throw std::invalid_argument(
          "analytic initial component has mismatched opcode/literal rows");
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
  return index == 0 || index == 3
             ? Real(0.347854845137453857373063949221999408)
             : Real(0.652145154862546142626936050778000593);
}

struct AnalyticCellAverage {
  AnalyticProgramView program;
  Real xlo, ylo, dx, dy;

  POPS_HD Real operator()(int i, int j) const {
    const Real x_center = xlo + (static_cast<Real>(i) + Real(0.5)) * dx;
    const Real y_center = ylo + (static_cast<Real>(j) + Real(0.5)) * dy;
    Real average = Real(0);
    for (int qy = 0; qy < 4; ++qy)
      for (int qx = 0; qx < 4; ++qx)
        average += gauss_weight(qx) * gauss_weight(qy) *
                   program.eval(x_center + Real(0.5) * dx * gauss_node(qx),
                                y_center + Real(0.5) * dy * gauss_node(qy));
    return Real(0.25) * average;
  }
};

struct AnalyticInitialKernel {
  Array4 values;
  int component;
  AnalyticCellAverage average;

  POPS_HD void operator()(int i, int j) const {
    values(i, j, component) = average(i, j);
  }
};

struct AnalyticInitialFiniteKernel {
  AnalyticCellAverage average;

  POPS_HD Real operator()(int i, int j) const {
    return Kokkos::isfinite(average(i, j)) ? Real(0) : Real(1);
  }
};

struct AnalyticInputBinding {
  int source = 0;  // 0 = current state, 1 = aux
  int component = 0;
};

struct AnalyticMappedInitialKernel {
  ConstArray4 state;
  ConstArray4 aux;
  Array4 values;
  int component;
  AnalyticProgramView program;
  const AnalyticInputBinding* bindings;
  int binding_count;
  Real xlo, ylo, dx, dy;

  POPS_HD void operator()(int i, int j) const {
    Real inputs[kAnalyticMaxStack];
    for (int slot = 0; slot < binding_count; ++slot) {
      const AnalyticInputBinding binding = bindings[slot];
      inputs[slot] = binding.source == 0 ? state(i, j, binding.component)
                                         : aux(i, j, binding.component);
    }
    const Real x_center = xlo + (static_cast<Real>(i) + Real(0.5)) * dx;
    const Real y_center = ylo + (static_cast<Real>(j) + Real(0.5)) * dy;
    const AnalyticEvaluation result =
        program.eval_checked(x_center, y_center, inputs, static_cast<std::uint8_t>(binding_count));
    values(i, j, component) =
        result.valid ? result.value : std::numeric_limits<Real>::quiet_NaN();
  }
};

struct AnalyticMappedInitialFiniteKernel {
  ConstArray4 state;
  ConstArray4 aux;
  AnalyticProgramView program;
  const AnalyticInputBinding* bindings;
  int binding_count;
  Real xlo, ylo, dx, dy;

  POPS_HD Real operator()(int i, int j) const {
    Real inputs[kAnalyticMaxStack];
    for (int slot = 0; slot < binding_count; ++slot) {
      const AnalyticInputBinding binding = bindings[slot];
      inputs[slot] = binding.source == 0 ? state(i, j, binding.component)
                                         : aux(i, j, binding.component);
    }
    const Real x_center = xlo + (static_cast<Real>(i) + Real(0.5)) * dx;
    const Real y_center = ylo + (static_cast<Real>(j) + Real(0.5)) * dy;
    const AnalyticEvaluation result =
        program.eval_checked(x_center, y_center, inputs, static_cast<std::uint8_t>(binding_count));
    return result.valid ? Real(0) : Real(1);
  }
};

struct GaussianCellAverage {
  Real xlo, ylo, dx, dy, center_x, center_y, background, amplitude, inverse_width;

  POPS_HD Real operator()(int i, int j) const {
    const Real root = Kokkos::sqrt(inverse_width);
    const Real ax = root * (xlo + static_cast<Real>(i) * dx - center_x);
    const Real bx = root * (xlo + static_cast<Real>(i + 1) * dx - center_x);
    const Real ay = root * (ylo + static_cast<Real>(j) * dy - center_y);
    const Real by = root * (ylo + static_cast<Real>(j + 1) * dy - center_y);
    const Real scale_x =
        Kokkos::sqrt(Real(3.141592653589793238462643383279502884)) /
        (Real(2) * root * dx);
    const Real scale_y =
        Kokkos::sqrt(Real(3.141592653589793238462643383279502884)) /
        (Real(2) * root * dy);
    return background + amplitude * scale_x * (Kokkos::erf(bx) - Kokkos::erf(ax)) * scale_y *
                            (Kokkos::erf(by) - Kokkos::erf(ay));
  }
};

struct GaussianCellAverageKernel {
  Array4 values;
  GaussianCellAverage average;

  POPS_HD void operator()(int i, int j) const { values(i, j, 0) = average(i, j); }
};

struct GaussianCellAverageFiniteKernel {
  GaussianCellAverage average;

  POPS_HD Real operator()(int i, int j) const {
    return Kokkos::isfinite(average(i, j)) ? Real(0) : Real(1);
  }
};

}  // namespace detail

/// Project each expression to a cell average with deterministic tensor Gauss--Legendre quadrature.
inline std::int64_t materialize_cell_average(
    MultiFab& values, Real xlo, Real ylo, Real dx, Real dy,
    const std::vector<AnalyticProgram>& programs) {
  const long invalid_target = all_reduce_sum(
      !(dx > Real(0)) || !(dy > Real(0)) ||
              !std::isfinite(static_cast<double>(xlo)) ||
              !std::isfinite(static_cast<double>(ylo)) ||
              programs.size() != static_cast<std::size_t>(values.ncomp())
          ? 1L
          : 0L);
  if (invalid_target != 0)
    throw std::invalid_argument("analytic initial materialization target/profile mismatch");
  long invalid_local = 0;
  for (int local = 0; local < values.local_size(); ++local) {
    const Box2D valid = values.box(local);
    for (int component = 0; component < values.ncomp(); ++component) {
      invalid_local += static_cast<long>(for_each_cell_reduce_sum(
          valid, detail::AnalyticInitialFiniteKernel{
                     {programs[static_cast<std::size_t>(component)].view(), xlo, ylo, dx, dy}}));
    }
  }
  const long invalid = all_reduce_sum(invalid_local);
  if (invalid != 0)
    throw std::runtime_error(
        "analytic initial expression produced non-finite cell values (count=" +
        std::to_string(static_cast<std::int64_t>(invalid)) + ")");
  for (int local = 0; local < values.local_size(); ++local) {
    const Box2D valid = values.box(local);
    for (int component = 0; component < values.ncomp(); ++component)
      for_each_cell(
          valid, detail::AnalyticInitialKernel{
                     values.fab(local).array(), component,
                     {programs[static_cast<std::size_t>(component)].view(), xlo, ylo, dx, dy}});
  }
  // AnalyticProgramView borrows device pointers from @p programs. The caller owns those programs as
  // setup-time locals, so projection kernels must finish before this function permits destruction.
  // This is a real asynchronous-device lifetime barrier, not a host fallback.
  device_fence();
  return values.box_array().num_cells() * values.ncomp();
}

inline std::int64_t materialize_discrete_mapped_state(
    MultiFab& values, const MultiFab& seed, const MultiFab& aux, Real xlo, Real ylo, Real dx,
    Real dy,
    const std::vector<AnalyticProgram>& programs,
    const std::vector<detail::AnalyticInputBinding>& bindings) {
  const long invalid_target = all_reduce_sum(
      !(dx > Real(0)) || !(dy > Real(0)) ||
              !std::isfinite(static_cast<double>(xlo)) ||
              !std::isfinite(static_cast<double>(ylo)) ||
              programs.size() != static_cast<std::size_t>(values.ncomp()) || bindings.empty() ||
              bindings.size() > kAnalyticMaxStack || values.local_size() != seed.local_size() ||
              values.local_size() != aux.local_size() || seed.ncomp() != values.ncomp()
          ? 1L
          : 0L);
  if (invalid_target != 0)
    throw std::invalid_argument("analytic mapped initial state target/profile mismatch");
  for (const auto& binding : bindings) {
    const long invalid_binding = all_reduce_sum(
        (binding.source != 0 && binding.source != 1) || binding.component < 0 ||
                (binding.source == 0 && binding.component >= seed.ncomp()) ||
                (binding.source == 1 && binding.component >= aux.ncomp())
            ? 1L
            : 0L);
    if (invalid_binding != 0)
      throw std::invalid_argument("analytic mapped initial state input binding is invalid");
  }
  using BindingStorage =
      std::vector<detail::AnalyticInputBinding, fab_allocator<detail::AnalyticInputBinding>>;
  BindingStorage device_bindings(bindings.begin(), bindings.end());
  long invalid_local = 0;
  for (int local = 0; local < values.local_size(); ++local) {
    const Box2D valid = values.box(local);
    const ConstArray4 state_view = seed.fab(local).const_array();
    const ConstArray4 aux_view = aux.fab(local).const_array();
    for (int component = 0; component < values.ncomp(); ++component) {
      invalid_local += static_cast<long>(for_each_cell_reduce_sum(
          valid, detail::AnalyticMappedInitialFiniteKernel{
                     state_view, aux_view,
                     programs[static_cast<std::size_t>(component)].view(),
                     device_bindings.data(), static_cast<int>(device_bindings.size()),
                     xlo, ylo, dx, dy}));
    }
  }
  const long invalid = all_reduce_sum(invalid_local);
  if (invalid != 0)
    throw std::runtime_error(
        "analytic mapped initial expression produced non-finite cell values (count=" +
        std::to_string(static_cast<std::int64_t>(invalid)) + ")");
  for (int local = 0; local < values.local_size(); ++local) {
    const Box2D valid = values.box(local);
    const ConstArray4 state_view = seed.fab(local).const_array();
    const ConstArray4 aux_view = aux.fab(local).const_array();
    Array4 output = values.fab(local).array();
    for (int component = 0; component < values.ncomp(); ++component)
      for_each_cell(valid, detail::AnalyticMappedInitialKernel{
                               state_view, aux_view, output, component,
                               programs[static_cast<std::size_t>(component)].view(),
                               device_bindings.data(), static_cast<int>(device_bindings.size()),
                               xlo, ylo, dx, dy});
  }
  device_fence();
  return values.box_array().num_cells() * values.ncomp();
}

inline std::int64_t materialize_gaussian_cell_average(
    MultiFab& values, Real xlo, Real ylo, Real dx, Real dy, Real center_x, Real center_y,
    Real background, Real amplitude, Real inverse_width) {
  const long invalid_arguments = all_reduce_sum(
      values.ncomp() != 1 || !(dx > Real(0)) || !(dy > Real(0)) ||
              !std::isfinite(static_cast<double>(xlo)) ||
              !std::isfinite(static_cast<double>(ylo)) ||
              !(inverse_width > Real(0)) || !std::isfinite(static_cast<double>(center_x)) ||
              !std::isfinite(static_cast<double>(center_y)) ||
              !std::isfinite(static_cast<double>(background)) ||
              !std::isfinite(static_cast<double>(amplitude)) ||
              !std::isfinite(static_cast<double>(inverse_width))
          ? 1L
          : 0L);
  if (invalid_arguments != 0)
    throw std::invalid_argument("analytic Gaussian initial profile is invalid");
  const detail::GaussianCellAverage average{
      xlo, ylo, dx, dy, center_x, center_y, background, amplitude, inverse_width};
  long invalid_local = 0;
  for (int local = 0; local < values.local_size(); ++local)
    invalid_local += static_cast<long>(for_each_cell_reduce_sum(
        values.box(local), detail::GaussianCellAverageFiniteKernel{average}));
  const long invalid = all_reduce_sum(invalid_local);
  if (invalid != 0)
    throw std::runtime_error(
        "analytic Gaussian profile produced non-finite cell averages (count=" +
        std::to_string(invalid) + ")");
  for (int local = 0; local < values.local_size(); ++local)
    for_each_cell(values.box(local),
                  detail::GaussianCellAverageKernel{values.fab(local).array(), average});
  // Give every one-shot initializer the same completion contract on asynchronous Kokkos devices.
  device_fence();
  return values.box_array().num_cells();
}

}  // namespace pops::analytic
