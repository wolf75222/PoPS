#include <pops/runtime/program/field_read_transport.hpp>
#include <pops/runtime/program/field_read_transform.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <gtest/gtest.h>

namespace {
constexpr std::uint64_t capacity = 1ULL << 20;
using Field = pops::MultiFab<1>;
using Capture = pops::runtime::program::CapturedFieldReads<1>;
using Journal = pops::runtime::program::FieldReadCapture<1>;

Field source(const pops::ExecutionLane& lane, bool replicated) {
  const pops::mesh::BoxArray<1> boxes(std::vector<pops::Box<1>>{
      {pops::Index<1>{0}, pops::Index<1>{0}}, {pops::Index<1>{1}, pops::Index<1>{1}}});
  const pops::mesh::RankSpace<1> ranks(pops::Index<1>{}, pops::Extent<1>{lane.size()});
  const auto distribution =
      replicated ? pops::mesh::Distribution<1>::replicated(boxes, ranks)
                 : pops::mesh::Distribution<1>::partitioned(
                       boxes, ranks, {pops::Index<1>{0}, pops::Index<1>{1 % lane.size()}});
  return Field(boxes, distribution, ranks.coordinate(static_cast<std::size_t>(lane.rank())), 1,
               pops::Extent<1>{1});
}

struct WriteReads {
  pops::FieldView<pops::Real, 1> view;
  int mode, rank;
  POPS_HD void operator()(const pops::Index<1>& cell) const {
    if (mode == 1 && cell[0] == 0)
      return;
    Journal journal{view, cell, 2};
    journal.initialize();
    journal.activate();
    journal.record(0, pops::Real(cell[0]) + pops::Real(.25) + (mode == 2 ? rank : 0));
  }
};

struct EvaluateSquare {
  pops::FieldView<const pops::Real, 1> source;
  pops::FieldView<pops::Real, 1> output, inputs, coordinates;
  pops::Real factor;
  POPS_HD void operator()(const pops::Index<1>& cell) const {
    Journal free_reads{inputs, cell, 2};
    Journal coordinate_reads{coordinates, cell, 1};
    free_reads.initialize();
    coordinate_reads.initialize();
    free_reads.activate();
    coordinate_reads.activate();
    const auto q = source(cell, 0);
    coordinate_reads.record(0, q);
    free_reads.record(0, factor);
    output(cell, 0) = factor * q * q;
  }
};

auto evaluate_square(const Field& input, Field& output, pops::Real factor,
                     const pops::ExecutionLane& lane) {
  auto capture =
      pops::runtime::program::CapturedTransformReads<1>::prepare(input, 2, lane, capacity);
  for (std::size_t patch = 0; patch < input.local_size(); ++patch)
    pops::for_each_cell(input.box(patch),
                        EvaluateSquare{input.fab(patch).view(), output.fab(patch).view(),
                                       capture.inputs.journal.fab(patch).view(),
                                       capture.coordinates.journal.fab(patch).view(), factor});
  Kokkos::fence();
  return capture;
}

void write(Capture& capture, int mode, int rank) {
  for (std::size_t patch = 0; patch < capture.journal.local_size(); ++patch)
    pops::for_each_cell(capture.journal.box(patch),
                        WriteReads{capture.journal.fab(patch).view(), mode, rank});
  Kokkos::fence();
}
}  // namespace

TEST(FieldReadTransport, ActualKokkosReadsArePartitionInvariantIncludingEmptyRanks) {
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.field-read.partition");
  auto partitioned = source(lane, false);
  auto replicated = source(lane, true);
  auto left = Capture::prepare(partitioned, 2, lane, capacity);
  auto right = Capture::prepare(replicated, 2, lane, capacity);
  write(left, 0, lane.rank());
  write(right, 0, lane.rank());
  const std::array<std::string_view, 2> authorities{"parameter/a/float64", "unused-provider/v1"};
  const auto expected = left.close(partitioned, "equation/schema", "exact/logical-support",
                                   authorities, lane, capacity);
  EXPECT_EQ(right.close(replicated, "equation/schema", "exact/logical-support", authorities, lane,
                        capacity),
            expected);
  const std::array<std::string_view, 2> unused_changed{"parameter/a/float64", "unused-provider/v2"};
  EXPECT_EQ(left.close(partitioned, "equation/schema", "exact/logical-support", unused_changed,
                       lane, capacity),
            expected);
  const std::array<std::string_view, 2> used_changed{"parameter/b/float64", "unused-provider/v1"};
  EXPECT_NE(left.close(partitioned, "equation/schema", "exact/logical-support", used_changed, lane,
                       capacity),
            expected);
}

TEST(FieldReadTransport, MissingKernelRowsAndInsufficientWorkspaceRejectCollectively) {
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.field-read.reject");
  auto field = source(lane, false);
  auto capture = Capture::prepare(field, 2, lane, capacity);
  const std::array<std::string_view, 2> authorities{"parameter/a/float64", "unused"};
  EXPECT_ANY_THROW((void)capture.close(field, "schema", "support", authorities, lane, capacity));
  write(capture, 1, lane.rank());
  EXPECT_ANY_THROW((void)capture.close(field, "schema", "support", authorities, lane, capacity));
  write(capture, 0, lane.rank());
  EXPECT_ANY_THROW((void)capture.close(field, "schema", "support", authorities, lane,
                                       lane.rank() == 0 ? 1 : capacity));
  EXPECT_NO_THROW((void)capture.close(field, "schema", "support", authorities, lane, capacity));
  EXPECT_ANY_THROW((void)Capture::prepare(field, 2, lane, lane.rank() == 0 ? 1 : capacity));
}

TEST(FieldReadTransport, ReplicasMustAgreeExactlyBeforeCanonicalSelection) {
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.field-read.replicas");
  auto field = source(lane, true);
  auto capture = Capture::prepare(field, 2, lane, capacity);
  write(capture, 2, lane.rank());
  const std::array<std::string_view, 2> authorities{"parameter/a/float64", "unused"};
  if (lane.size() > 1) {
    EXPECT_ANY_THROW((void)capture.close(field, "schema", "support", authorities, lane, capacity));
  } else {
    EXPECT_NO_THROW((void)capture.close(field, "schema", "support", authorities, lane, capacity));
  }
}

TEST(FieldReadTransport, EqualLayoutDoesNotAuthenticateAReplacedSourceAllocation) {
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.field-read.incarnation");
  auto field = source(lane, false);
  auto lookalike = source(lane, false);
  auto capture = Capture::prepare(field, 2, lane, capacity);
  write(capture, 0, lane.rank());
  const std::array<std::string_view, 2> authorities{"parameter/a/float64", "unused"};
  EXPECT_ANY_THROW(
      (void)capture.close(lookalike, "schema", "support", authorities, lane, capacity));
  // Move a different live allocation into the same C++ object. Shape, owner and
  // values alone do not describe the input actually used by the kernel.
  field = std::move(lookalike);
  EXPECT_ANY_THROW((void)capture.close(field, "schema", "support", authorities, lane, capacity));
}

TEST(FieldReadTransport, HighestRankIncarnationAndNestedFabReplacementAreNotLayoutAliases) {
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.field-read.nested-aba");
  auto field = source(lane, false);
  auto capture = Capture::prepare(field, 2, lane, capacity);
  write(capture, 0, lane.rank());
  const std::array<std::string_view, 2> authorities{"parameter/a", "unused"};
  // At MPI3 this modifies only an empty rank. Pointer lists and exact layout
  // alone would both match; the retained Field incarnation must still reject.
  if (lane.rank() == lane.size() - 1)
    field = source(lane, false);
  EXPECT_ANY_THROW((void)capture.close(field, "schema", "support", authorities, lane, capacity));

  auto nested = source(lane, false);
  auto next = Capture::prepare(nested, 2, lane, capacity);
  write(next, 0, lane.rank());
  if (lane.rank() == 0) {
    const auto old_data = nested.fab(0).storage().data();
    auto replacement = source(lane, false);
    nested.fab(0) = replacement.fab(0);
    // The journal retains a const storage view, so the old allocation cannot
    // be reclaimed/reused under its address while this proof is outstanding.
    EXPECT_NE(nested.fab(0).storage().data(), old_data);
  }
  EXPECT_ANY_THROW((void)next.close(nested, "schema", "support", authorities, lane, capacity));
}

TEST(FieldReadTransport, TransformSeparatesActualCoordinatesFromOnlyUsedBindings) {
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.field-read.transform");
  auto field = source(lane, false);
  auto output = source(lane, false);
  field.set_val(2.);
  auto first = evaluate_square(field, output, 3., lane);
  const std::array<std::string_view, 2> bindings{"parameter/a", "unused"};
  const auto baseline =
      first.close(field, "a*q*q", "typed/q", "native/support", bindings, lane, capacity);
  auto changed_binding = evaluate_square(field, output, 4., lane);
  const auto rebound =
      changed_binding.close(field, "a*q*q", "typed/q", "native/support", bindings, lane, capacity);
  EXPECT_NE(baseline.inputs, rebound.inputs);
  EXPECT_EQ(baseline.coordinates, rebound.coordinates);
  const std::array<std::string_view, 2> unused_changed{"parameter/a", "different-unused"};
  const auto unaffected =
      first.close(field, "a*q*q", "typed/q", "native/support", unused_changed, lane, capacity);
  EXPECT_EQ(baseline.inputs, unaffected.inputs);
  EXPECT_EQ(baseline.coordinates, unaffected.coordinates);

  field.set_val(3.);
  auto changed_coordinate = evaluate_square(field, output, 3., lane);
  const auto moved = changed_coordinate.close(field, "a*q*q", "typed/q", "native/support", bindings,
                                              lane, capacity);
  EXPECT_EQ(baseline.inputs, moved.inputs);
  EXPECT_NE(baseline.coordinates, moved.coordinates);
  const auto foreign = changed_coordinate.close(field, "a*q*q", "typed/q", "foreign/support",
                                                bindings, lane, capacity);
  EXPECT_NE(moved.inputs, foreign.inputs);
  EXPECT_NE(moved.coordinates, foreign.coordinates);
}

TEST(FieldReadTransport, TransformJournalsAndClosuresShareAnExplicitWorkspace) {
  auto lane = pops::ExecutionLane::duplicate_world_collectively("test.field-read.transform-budget");
  auto field = source(lane, false);
  auto output = source(lane, false);
  field.set_val(2.);
  auto capture = evaluate_square(field, output, 3., lane);
  const std::array<std::string_view, 2> bindings{"parameter/a", "unused"};
  EXPECT_ANY_THROW((void)capture.close(field, "a*q*q", "typed/q", "support", bindings, lane,
                                       lane.rank() == 0 ? 1 : capacity));
  EXPECT_NO_THROW(
      (void)capture.close(field, "a*q*q", "typed/q", "support", bindings, lane, capacity));
  EXPECT_ANY_THROW((void)pops::runtime::program::CapturedTransformReads<1>::prepare(
      field, 2, lane, lane.rank() == 0 ? 1 : capacity));
}
