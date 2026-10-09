#include <pops/runtime/multiblock/interface_flux_scheduler.hpp>
#include <Kokkos_Core.hpp>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <stdexcept>
using namespace pops;
using namespace pops::runtime::multiblock;
namespace {
bool fail_next = false;
bool fail_with_error = false;
bool vote_error = false;
int allocations_in_gap = 0;
bool gap = false;
int votes = 0;
}  // namespace
void* operator new(std::size_t n) {
  if (fail_next) {
    fail_next = false;
    if (fail_with_error)
      throw std::runtime_error("injected preparation exception");
    throw std::bad_alloc();
  }
  if (gap)
    ++allocations_in_gap;
  if (void* p = std::malloc(n ? n : 1))
    return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t) noexcept {
  std::free(p);
}
extern "C" int MPI_Allreduce(const void* send, void* receive, int count, MPI_Datatype datatype,
                             MPI_Op op, MPI_Comm comm) {
  gap = false;
  ++votes;
  int rc = PMPI_Allreduce(send, receive, count, datatype, op, comm);
  if (votes == 1 && !vote_error)
    gap = true;
  return vote_error ? MPI_ERR_OTHER : rc;
}
template <int Dim>
void profile(const CommunicatorView& lane) {
  using S = InterfaceFluxScheduler<Dim>;
  Index<Dim> low{}, high{};
  Extent<Dim> ranks_extent{};
  RealVector<Dim> lower{}, upper{};
  for (int axis = 0; axis < Dim; ++axis) {
    low[axis] = -axis - 3;
    high[axis] = low[axis] + axis + 2;
    ranks_extent[axis] = 1;
    lower[axis] = axis - 2.;
    upper[axis] = lower[axis] + (axis + 3) * .75;
  }
  ranks_extent[0] = lane.size();
  Box<Dim> box(low, high);
  mesh::BoxArray<Dim> layout(std::vector<Box<Dim>>{box});
  mesh::RankSpace<Dim> ranks(Index<Dim>{}, ranks_extent);
  auto distribution =
      mesh::Distribution<Dim>::partitioned(layout, ranks, std::vector<Index<Dim>>{Index<Dim>{}});
  MultiFab<Dim> left(layout, distribution, ranks.coordinate(lane.rank()), 3, Extent<Dim>{});
  MultiFab<Dim> right(layout, distribution, ranks.coordinate(lane.rank()), 3, Extent<Dim>{});
  auto geometry = Geometry<Dim>::from_bounds(box, lower, upper);
  AxisAlignedInterface<Dim> route;
  route.identity = std::string("route\0full-key", 14);
  route.right_component_for_left = {2, 0, 1};
  route.left_block = 4;
  route.right_block = 9;
  route.level = 2;
  route.left_trace_projection_identity = std::string("left\0projection", 15);
  route.right_trace_projection_identity = "right/projection";
  const auto original = S::collective_plan_identity_(route, left, geometry, right, geometry, .75,
                                                     .75, 7, 3, "MPI_COMM_WORLD", lane.size());
  votes = 0;
  allocations_in_gap = 0;
  gap = false;
  auto identity = S::prepare_collective_plan_identity_(
      route, left, geometry, right, geometry, .75, .75, 7, 3, "MPI_COMM_WORLD", lane.size(), lane);
  assert(identity == original);
  ExactOrderedBytePair pair(std::string_view(route.identity), std::string_view(identity));
  assert(all_ranks_agree_exact_ordered_byte_pairs(std::span<const ExactOrderedBytePair>(&pair, 1),
                                                  lane));
  gap = false;
  assert(allocations_in_gap == 0);
  const int success_votes = votes;
  for (int kind = 0; kind < 2; ++kind) {
    for (int failing = 0; failing < lane.size(); ++failing) {
      votes = 0;
      gap = false;
      fail_next = lane.rank() == failing;
      fail_with_error = kind == 1;
      bool rejected = false, own_bad_alloc = false, own_marker = false;
      try {
        (void)S::prepare_collective_plan_identity_(route, left, geometry, right, geometry, .75, .75,
                                                   7, 3, "MPI_COMM_WORLD", lane.size(), lane);
      } catch (const std::bad_alloc&) {
        rejected = true;
        own_bad_alloc = true;
      } catch (const std::runtime_error& error) {
        rejected = true;
        own_marker = std::string_view(error.what()) == "injected preparation exception";
      }
      fail_next = false;
      fail_with_error = false;
      gap = false;
      assert(rejected && own_bad_alloc == (kind == 0 && lane.rank() == failing));
      assert(own_marker == (kind == 1 && lane.rank() == failing));
      assert(votes == (lane.size() > 1 ? 1 : 0));  // No byte-consensus collective reached.
      votes = 0;
      identity =
          S::prepare_collective_plan_identity_(route, left, geometry, right, geometry, .75, .75, 7,
                                               3, "MPI_COMM_WORLD", lane.size(), lane);
      assert(identity == original);
      gap = false;
    }
  }
  if (lane.size() > 1) {
    votes = 0;
    vote_error = true;
    bool rejected = false;
    try {
      (void)S::prepare_collective_plan_identity_(route, left, geometry, right, geometry, .75, .75,
                                                 7, 3, "MPI_COMM_WORLD", lane.size(), lane);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    vote_error = false;
    gap = false;
    assert(rejected && votes == 1);  // Injected MPI return-code error after real PMPI completion.
  }
  // The old non-voted serializer's rank-local allocation failure remains a separate control.
  votes = 0;
  fail_next = true;
  bool old_rejected = false;
  try {
    (void)S::collective_plan_identity_(route, left, geometry, right, geometry, .75, .75, 7, 3,
                                       "MPI_COMM_WORLD", lane.size());
  } catch (const std::bad_alloc&) {
    old_rejected = true;
  }
  fail_next = false;
  gap = false;
  assert(old_rejected && votes == 0);
  printf(
      "rank=%d ranks=%d Dim=%d exact_bytes=%zu success_votes=%d prep_failure_votes=%d retry=PASS "
      "old_unvoted=PASS gap_allocations=0\n",
      lane.rank(), lane.size(), Dim, original.size(), success_votes, lane.size() > 1 ? 1 : 0);
}
int main(int argc, char** argv) {
  comm_init();
  {
    Kokkos::ScopeGuard guard(argc, argv);
    auto lane = world_communicator_view();
    if (argc > 1 && std::string_view(argv[1]) == "--mpi2")
      assert(lane.size() == 2);
    profile<1>(lane);
    profile<2>(lane);
    profile<3>(lane);
  }
  comm_finalize();
}
