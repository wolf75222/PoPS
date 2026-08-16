#include <gtest/gtest.h>

#include <pops/coupling/source/coupled_source_program.hpp>
#include <pops/mesh/index/box.hpp>
#include <pops/mesh/layout/box_array.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/program/program_context.hpp>
#include <pops/runtime/program/wire_ids.hpp>

#include <Kokkos_Core.hpp>

#include <exception>
#include <string>
#include <typeinfo>
#include <vector>

namespace {

template <class ExpectedException, class F>
::testing::AssertionResult ThrowsWithMessage(F&& f, const std::vector<std::string>& needles) {
  try {
    f();
  } catch (const ExpectedException& error) {
    if (typeid(error) != typeid(ExpectedException))
      return ::testing::AssertionFailure()
             << "expected exact exception category " << typeid(ExpectedException).name()
             << ", received " << typeid(error).name();
    const std::string message = error.what();
    for (const std::string& needle : needles)
      if (message.find(needle) == std::string::npos)
        return ::testing::AssertionFailure() << "message missing '" << needle << "': " << message;
    return ::testing::AssertionSuccess();
  } catch (const std::exception& error) {
    return ::testing::AssertionFailure()
           << "expected exception category " << typeid(ExpectedException).name() << ", received "
           << typeid(error).name() << ": " << error.what();
  }
  return ::testing::AssertionFailure() << "expected an exception, none was thrown";
}

template <int Dim>
pops::mesh::RankSpace<Dim> one_rank_space() {
  pops::Extent<Dim> extent{};
  for (int axis = 0; axis < Dim; ++axis)
    extent[axis] = 1;
  return {pops::Index<Dim>{}, extent};
}

template <int Dim>
pops::Box<Dim> domain(int cells) {
  pops::Extent<Dim> extent{};
  for (int axis = 0; axis < Dim; ++axis)
    extent[axis] = cells;
  return pops::Box<Dim>::from_extents(extent);
}

}  // namespace

using namespace pops;
using namespace pops::mesh;

TEST(PublicValidationErrors, ProgramWireIdsNeverFallback) {
  using namespace pops::runtime::program;
  EXPECT_TRUE(ThrowsWithMessage<std::runtime_error>(
      [] { validate_linear_solve_method(99, "test"); }, {"test", "LinearSolveMethod", "99"}));
  EXPECT_TRUE(ThrowsWithMessage<std::runtime_error>(
      [] { validate_linear_solve_method(kLinearSolveReserved4, "test"); },
      {"test", "LinearSolveMethod", "4"}));
  EXPECT_TRUE(
      ThrowsWithMessage<std::runtime_error>([] { validate_prepared_field_slot("", "test"); },
                                            {"test", "prepared field-slot identity", "non-empty"}));
  EXPECT_NO_THROW(validate_linear_solve_method(kLinearSolveBicgstab, "test"));
  EXPECT_NO_THROW(validate_prepared_field_slot("pops.test.operator.extra-slot", "test"));
}

TEST(PublicValidationErrors, RankedFabRejectsInvalidMetadataAndForeignHostMirror) {
  const Box<2> valid = domain<2>(2);
  EXPECT_TRUE(
      ThrowsWithMessage<std::invalid_argument>([&] { Fab<2> bad(valid, /*ncomp=*/0, Extent<2>{}); },
                                               {"pops::Fab", "ncomp must be positive"}));
  EXPECT_TRUE(ThrowsWithMessage<std::invalid_argument>(
      [&] { Fab<2> bad(valid, /*ncomp=*/1, Extent<2>{1, -1}); },
      {"pops::Fab", "ghost extents must be non-negative"}));

  Fab<2, Kokkos::HostSpace> first(valid, 1, Extent<2>{});
  Fab<2, Kokkos::HostSpace> second(valid, 1, Extent<2>{});
  const auto foreign = first.create_host_mirror();
  EXPECT_TRUE(ThrowsWithMessage<std::invalid_argument>(
      [&] { second.copy_to_host(foreign); },
      {"pops::Fab host mirror", "does not match this Fab association"}));
}

TEST(PublicValidationErrors, RankedMultiFabRejectsInvalidMetadataAndIndices) {
  const Box<2> valid = domain<2>(2);
  const BoxArray<2> layout(std::vector<Box<2>>{valid});
  const auto ranks = one_rank_space<2>();
  const auto distribution = Distribution<2>::replicated(layout, ranks);
  EXPECT_TRUE(ThrowsWithMessage<std::invalid_argument>(
      [&] { MultiFab<2, Kokkos::HostSpace> bad(layout, distribution, Index<2>{}, 0, Extent<2>{}); },
      {"pops::MultiFab", "ncomp must be positive"}));
  EXPECT_TRUE(ThrowsWithMessage<std::invalid_argument>(
      [&] {
        MultiFab<2, Kokkos::HostSpace> bad(layout, distribution, Index<2>{}, 1, Extent<2>{1, -1});
      },
      {"pops::MultiFab", "ghost extents must be non-negative"}));
  EXPECT_TRUE(ThrowsWithMessage<std::invalid_argument>(
      [&] { (void)Distribution<2>::partitioned(layout, ranks, std::vector<Index<2>>{}); },
      {"Distribution", "owner count must equal its patch count"}));
  EXPECT_TRUE(ThrowsWithMessage<std::out_of_range>(
      [&] {
        (void)Distribution<2>::partitioned(layout, ranks, std::vector<Index<2>>{Index<2>{1, 0}});
      },
      {"Distribution owner", "outside the process space"}));

  MultiFab<2, Kokkos::HostSpace> fields(layout, distribution, Index<2>{}, 1, Extent<2>{});
  EXPECT_TRUE(ThrowsWithMessage<std::out_of_range>(
      [&] { (void)fields.fab(1); }, {"pops::MultiFab", "local patch index", "local storage"}));
}

TEST(PublicValidationErrors, RankedRefinementRejectsInvalidRatioAndComponentWidths) {
  const Box<2> coarse_domain = domain<2>(2);
  const Box<2> fine_domain = domain<2>(4);
  const BoxArray<2> coarse_layout(std::vector<Box<2>>{coarse_domain});
  const BoxArray<2> fine_layout(std::vector<Box<2>>{fine_domain});
  EXPECT_TRUE(ThrowsWithMessage<std::invalid_argument>(
      [&] { (void)coarsen(fine_layout, 0); }, {"pops::coarsen(layout)", "strictly positive"}));

  const auto ranks = one_rank_space<2>();
  const auto coarse_distribution = Distribution<2>::replicated(coarse_layout, ranks);
  const auto fine_distribution = Distribution<2>::replicated(fine_layout, ranks);
  MultiFab<2, Kokkos::HostSpace> coarse(coarse_layout, coarse_distribution, Index<2>{}, 1,
                                        Extent<2>{});
  MultiFab<2, Kokkos::HostSpace> fine(fine_layout, fine_distribution, Index<2>{}, 2, Extent<2>{});
  const CopyScheduleBudget budget{1, 1, 0, 0};
  EXPECT_TRUE(
      ThrowsWithMessage<std::invalid_argument>([&] { average_down(fine, coarse, 2, budget); },
                                               {"pops::average_down", "component counts differ"}));
  EXPECT_TRUE(
      ThrowsWithMessage<std::invalid_argument>([&] { interpolate(coarse, fine, 2, budget); },
                                               {"pops::interpolate", "component counts differ"}));
  EXPECT_TRUE(ThrowsWithMessage<std::invalid_argument>(
      [&] { parallel_copy(fine, coarse, budget); },
      {"pops::parallel_copy", "different component counts"}));
}

TEST(PublicValidationErrors, CsProgramStackValidationRejectsMalformedPrograms) {
  EXPECT_TRUE(ThrowsWithMessage<ValidationError>(
      [&] {
        CsProgram program;
        program.len = 1;
        program.op[0] = static_cast<int>(CsOp::Add);
        validate_cs_program_stack(program, "test CsProgram");
      },
      {"test CsProgram", "well-formed postfix stack program", "stack underflow"}));
  EXPECT_TRUE(ThrowsWithMessage<ValidationError>(
      [&] {
        CsProgram program;
        program.len = 2;
        program.op[0] = static_cast<int>(CsOp::PushReg);
        program.op[1] = static_cast<int>(CsOp::PushReg);
        validate_cs_program_stack(program, "test CsProgram");
      },
      {"test CsProgram", "exactly one result", "final stack_depth=2"}));
}
