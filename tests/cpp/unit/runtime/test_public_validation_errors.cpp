#include <gtest/gtest.h>

#include <pops/coupling/source/coupled_source_program.hpp>
#include <pops/mesh/layout/distribution.hpp>
#include <pops/mesh/layout/refinement.hpp>
#include <pops/mesh/storage/fab.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/runtime/program/wire_ids.hpp>

#include <cstdint>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace {

template <class F>
::testing::AssertionResult ThrowsWithMessage(F&& f, const std::vector<std::string>& needles) {
  try {
    f();
  } catch (const std::exception& error) {
    const std::string message = error.what();
    for (const std::string& needle : needles) {
      if (message.find(needle) == std::string::npos)
        return ::testing::AssertionFailure() << "message missing '" << needle << "': " << message;
    }
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << "expected an exception, none was thrown";
}

template <int Dim>
pops::Extent<Dim> filled_extent(std::int64_t value) {
  pops::Extent<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis)
    result[axis] = value;
  return result;
}

template <int Dim>
pops::Box<Dim> cube(int lower, int upper) {
  pops::Index<Dim> lo{};
  pops::Index<Dim> hi{};
  for (int axis = 0; axis < Dim; ++axis) {
    lo[axis] = lower;
    hi[axis] = upper;
  }
  return pops::Box<Dim>{lo, hi};
}

template <int Dim>
pops::mesh::RankSpace<Dim> one_rank_space() {
  return pops::mesh::RankSpace<Dim>{pops::Index<Dim>{}, filled_extent<Dim>(1)};
}

template <int Dim>
void prove_exact_rank_storage_and_transfer_validation() {
  const pops::Box<Dim> valid = cube<Dim>(0, 1);
  const pops::mesh::BoxArray<Dim> layout(std::vector<pops::Box<Dim>>{valid});
  const auto ranks = one_rank_space<Dim>();
  const auto distribution = pops::mesh::Distribution<Dim>::replicated(layout, ranks);

  EXPECT_TRUE(ThrowsWithMessage([&] { (void)pops::Fab<Dim>(valid, 0, pops::Extent<Dim>{}); },
                                {"pops::Fab", "ncomp", "positive"}));

  auto negative_ghosts = filled_extent<Dim>(0);
  negative_ghosts[Dim - 1] = -1;
  EXPECT_TRUE(ThrowsWithMessage([&] { (void)pops::Fab<Dim>(valid, 1, negative_ghosts); },
                                {"pops::Fab", "ghost extents", "non-negative"}));

  pops::Fab<Dim> first(valid, 1);
  pops::Fab<Dim> second(valid, 1);
  const auto first_mirror = first.create_host_mirror();
  EXPECT_TRUE(ThrowsWithMessage([&] { second.copy_to_host(first_mirror); },
                                {"pops::Fab", "host mirror", "association"}));

  EXPECT_TRUE(ThrowsWithMessage(
      [&] {
        (void)pops::MultiFab<Dim>(layout, distribution, pops::Index<Dim>{}, 0, pops::Extent<Dim>{});
      },
      {"pops::MultiFab", "ncomp", "positive"}));
  EXPECT_TRUE(ThrowsWithMessage(
      [&] {
        (void)pops::MultiFab<Dim>(layout, distribution, pops::Index<Dim>{}, 1, negative_ghosts);
      },
      {"pops::MultiFab", "ghost extents", "non-negative"}));

  const pops::mesh::BoxArray<Dim> other_layout(std::vector<pops::Box<Dim>>{cube<Dim>(0, 2)});
  const auto other_distribution = pops::mesh::Distribution<Dim>::replicated(other_layout, ranks);
  EXPECT_TRUE(ThrowsWithMessage(
      [&] {
        (void)pops::MultiFab<Dim>(layout, other_distribution, pops::Index<Dim>{}, 1,
                                  pops::Extent<Dim>{});
      },
      {"pops::MultiFab", "distribution layout", "exactly match"}));

  auto outside_owner = pops::Index<Dim>{};
  outside_owner[0] = 1;
  EXPECT_TRUE(ThrowsWithMessage(
      [&] {
        (void)pops::mesh::Distribution<Dim>::partitioned(
            layout, ranks, std::vector<pops::Index<Dim>>{outside_owner});
      },
      {"Distribution owner", "outside", "process space"}));

  pops::MultiFab<Dim> field(layout, distribution, pops::Index<Dim>{}, 1, pops::Extent<Dim>{});
  EXPECT_TRUE(ThrowsWithMessage([&] { (void)field.fab(1); },
                                {"pops::MultiFab", "local patch index", "outside"}));

  EXPECT_TRUE(ThrowsWithMessage([&] { (void)pops::coarsen(layout, 0); },
                                {"pops::coarsen(layout)", "strictly positive"}));

  const auto fine_layout = pops::refine(layout, 2);
  const auto fine_distribution = pops::mesh::Distribution<Dim>::replicated(fine_layout, ranks);
  pops::MultiFab<Dim> coarse(layout, distribution, pops::Index<Dim>{}, 1, pops::Extent<Dim>{});
  pops::MultiFab<Dim> fine(fine_layout, fine_distribution, pops::Index<Dim>{}, 2,
                           pops::Extent<Dim>{});
  const pops::CopyScheduleBudget one_job_budget{1, 1, 0, 0};
  EXPECT_TRUE(ThrowsWithMessage([&] { pops::average_down(fine, coarse, 2, one_job_budget); },
                                {"pops::average_down", "component counts differ"}));
  EXPECT_TRUE(ThrowsWithMessage([&] { pops::interpolate(coarse, fine, 2, one_job_budget); },
                                {"pops::interpolate", "component counts differ"}));

  pops::MultiFab<Dim> wider(layout, distribution, pops::Index<Dim>{}, 2, pops::Extent<Dim>{});
  EXPECT_TRUE(ThrowsWithMessage([&] { pops::parallel_copy(coarse, wider, one_job_budget); },
                                {"pops::parallel_copy", "different component counts"}));

  pops::MultiFab<Dim> larger(other_layout, other_distribution, pops::Index<Dim>{}, 1,
                             pops::Extent<Dim>{});
  EXPECT_TRUE(ThrowsWithMessage(
      [&] { (void)pops::prepare_copy_schedule(larger, coarse, one_job_budget); },
      {"pops::CopySchedule", "source and destination layouts", "same cells exactly"}));

  auto split_upper = pops::Index<Dim>{};
  for (int axis = 0; axis < Dim; ++axis)
    split_upper[axis] = 1;
  split_upper[0] = 3;
  auto split_grid = filled_extent<Dim>(2);
  const auto split_layout = pops::mesh::BoxArray<Dim>::from_domain(
      pops::Box<Dim>{pops::Index<Dim>{}, split_upper}, split_grid);
  auto rank_extent = filled_extent<Dim>(1);
  rank_extent[0] = 2;
  const pops::mesh::RankSpace<Dim> two_ranks{pops::Index<Dim>{}, rank_extent};
  auto rank_one = pops::Index<Dim>{};
  rank_one[0] = 1;
  const auto source_distribution = pops::mesh::Distribution<Dim>::partitioned(
      split_layout, two_ranks, {pops::Index<Dim>{}, rank_one});
  const auto destination_distribution = pops::mesh::Distribution<Dim>::partitioned(
      split_layout, two_ranks, {rank_one, pops::Index<Dim>{}});
  pops::MultiFab<Dim> remote_source(split_layout, source_distribution, pops::Index<Dim>{}, 1,
                                    pops::Extent<Dim>{});
  pops::MultiFab<Dim> remote_destination(split_layout, destination_distribution, pops::Index<Dim>{},
                                         1, pops::Extent<Dim>{});
  const auto remote_schedule = pops::prepare_copy_schedule(remote_destination, remote_source,
                                                           pops::CopyScheduleBudget{4, 2, 1, 1});
  ASSERT_TRUE(remote_schedule.has_remote_jobs());
  EXPECT_TRUE(ThrowsWithMessage(
      [&] { pops::parallel_copy(remote_destination, remote_source, remote_schedule); },
      {"pops::CopySchedule", "remote jobs", "prepared", "copy transport"}));
}

}  // namespace

using namespace pops;

TEST(PublicValidationErrors, ProgramWireIdsNeverFallback) {
  using namespace pops::runtime::program;
  EXPECT_TRUE(ThrowsWithMessage([] { validate_linear_solve_method(99, "test"); },
                                {"LinearSolveMethod", "99"}));
  EXPECT_TRUE(ThrowsWithMessage([] { validate_linear_solve_method(kLinearSolveReserved4, "test"); },
                                {"LinearSolveMethod", "4"}));
  EXPECT_TRUE(ThrowsWithMessage([] { validate_prepared_field_slot("", "test"); },
                                {"prepared field-slot identity", "non-empty"}));
  EXPECT_NO_THROW(validate_linear_solve_method(kLinearSolveBicgstab, "test"));
  EXPECT_NO_THROW(validate_prepared_field_slot("pops.test.operator.extra-slot", "test"));
}

TEST(PublicValidationErrors, ExactRankStorageAndTransfersRejectInvalidContractsIn1D2D3D) {
  prove_exact_rank_storage_and_transfer_validation<1>();
  prove_exact_rank_storage_and_transfer_validation<2>();
  prove_exact_rank_storage_and_transfer_validation<3>();
}

TEST(PublicValidationErrors, CsProgramStackValidationRejectsUnderflow) {
  EXPECT_TRUE(ThrowsWithMessage(
      [&] {
        CsProgram program;
        program.len = 1;
        program.op[0] = static_cast<int>(CsOp::Add);
        validate_cs_program_stack(program, "test CsProgram");
      },
      {"test CsProgram", "well-formed postfix stack program", "stack underflow"}));
}

TEST(PublicValidationErrors, CsProgramStackValidationRejectsUnusedResult) {
  EXPECT_TRUE(ThrowsWithMessage(
      [&] {
        CsProgram program;
        program.len = 2;
        program.op[0] = static_cast<int>(CsOp::PushReg);
        program.op[1] = static_cast<int>(CsOp::PushReg);
        validate_cs_program_stack(program, "test CsProgram");
      },
      {"test CsProgram", "exactly one result", "final stack_depth=2"}));
}
