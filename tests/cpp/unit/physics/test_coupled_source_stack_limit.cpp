#include <pops/coupling/source/coupled_source_program.hpp>

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <string>

namespace {

using pops::CoupledFreqKernel;
using pops::CoupledSourceKernel;
using pops::CsOp;
using pops::CsProgram;
using pops::FieldView;
using pops::Index;
using pops::Real;
using pops::kCsMaxStack;

CsProgram multiply_registers() {
  CsProgram program{};
  program.len = 3;
  program.op[0] = static_cast<int>(CsOp::PushReg);
  program.arg[0] = 0;
  program.op[1] = static_cast<int>(CsOp::PushReg);
  program.arg[1] = 1;
  program.op[2] = static_cast<int>(CsOp::Mul);
  return program;
}

template <class Value, int Dim>
FieldView<Value, Dim> one_cell_view(Value* value, const Index<Dim>& index) {
  FieldView<Value, Dim> view{};
  view.data = value;
  view.origin = index;
  for (int axis = 0; axis < Dim; ++axis) {
    view.extents[axis] = 1;
    view.strides[axis] = 1;
  }
  view.ncomp = 1;
  view.component_stride = 1;
  return view;
}

template <int Dim>
void expect_exact_ranked_kernels() {
  Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis)
    index[axis] = 4 + axis;

  Real first = Real(3);
  Real second = Real(5);
  Real output = Real(7);

  CoupledSourceKernel<Dim> source{};
  source.in[0] = one_cell_view<const Real, Dim>(&first, index);
  source.in[1] = one_cell_view<const Real, Dim>(&second, index);
  source.n_in = 2;
  source.in_comp[0] = 0;
  source.in_comp[1] = 0;
  source.out[0] = one_cell_view<Real, Dim>(&output, index);
  source.out_comp[0] = 0;
  source.prog[0] = multiply_registers();
  source.n_terms = 1;
  source.dt = Real(0.25);
  source(index);
  EXPECT_DOUBLE_EQ(output, Real(10.75));

  CoupledFreqKernel<Dim> frequency{};
  frequency.in[0] = one_cell_view<const Real, Dim>(&first, index);
  frequency.in[1] = one_cell_view<const Real, Dim>(&second, index);
  frequency.n_in = 2;
  frequency.in_comp[0] = 0;
  frequency.in_comp[1] = 0;
  frequency.prog = multiply_registers();
  Real maximum = Real(4);
  frequency(index, maximum);
  EXPECT_DOUBLE_EQ(maximum, Real(15));
}

TEST(test_coupled_source_stack_limit, ExecutesOneTypedKernelInOneTwoAndThreeDimensions) {
  expect_exact_ranked_kernels<1>();
  expect_exact_ranked_kernels<2>();
  expect_exact_ranked_kernels<3>();
}

TEST(test_coupled_source_stack_limit, RejectsStackOverflowBeforeExecution) {
  CsProgram program{};
  program.len = kCsMaxStack + 1;
  for (int instruction = 0; instruction < program.len; ++instruction) {
    program.op[instruction] = static_cast<int>(CsOp::PushReg);
    program.arg[instruction] = 0;
  }

  try {
    pops::validate_cs_program_stack(program, "coupled source test");
    FAIL() << "stack overflow was accepted";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("stack overflow"), std::string::npos);
    EXPECT_NE(message.find(std::to_string(kCsMaxStack)), std::string::npos);
  }
}

}  // namespace
