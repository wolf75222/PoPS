// Compile with the installed Kokkos headers in both Real specializations.
#include <pops/runtime/program/generated_step_bound.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>

#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

int main() {
  using namespace pops::runtime::program;
  const double smallest = std::numeric_limits<double>::denorm_min();
  const double largest = std::numeric_limits<double>::max();
  assert(std::bit_cast<std::uint64_t>(decode_step_bound(encode_step_bound(smallest))) == 1);
  assert(decode_step_bound(encode_step_bound(largest)) == largest);
  assert(encode_step_bound(smallest) > encode_step_bound(largest));
  assert(encode_step_bound(largest) != 0);
  assert(decode_step_bound(0) == kUnboundedStep);
  for (double value :
       {0.0, -0.0, -1.0, kUnboundedStep, -kUnboundedStep, std::numeric_limits<double>::quiet_NaN()})
    assert(encode_step_bound(value) == 0);

  constexpr std::uint64_t index = (std::uint64_t{1} << 24) + 1;
  const StepBoundReduction first{largest, index + 1, 0};
  const StepBoundReduction second{smallest, index, kNoStepBoundCell};
  StepBoundReduction joined = first;
  joined.join(second);
  const auto [status, nonfinite, nonpositive, bound] =
      step_bound_words(StepBoundDispatch::kReady, joined);
  assert(status == 0);
  assert(kNoStepBoundCell - nonfinite == index);
  assert(kNoStepBoundCell - nonpositive == 0);
  assert(decode_step_bound(bound) == smallest);

  ProgramRuntimeState<2> program;
  program.advance_step_bound_identity_ = "retained-bound";
  program.advance_step_bound_ = [joined](StepBoundReduction* result) {
    *result = joined;
    return 0;
  };
  auto installed = program.capture_artifact_step_install();
  program.install_unverified_step([](double) {});
  assert(!program.advance_step_bound_ && program.advance_step_bound_identity_.empty());
  program.rollback_artifact_step_install(std::move(installed));
  StepBoundReduction restored;
  assert(program.advance_step_bound_identity_ == "retained-bound");
  assert(program.advance_step_bound_(&restored) == 0);
  assert(step_bound_words(StepBoundDispatch::kReady, restored) ==
         step_bound_words(StepBoundDispatch::kReady, joined));
}
