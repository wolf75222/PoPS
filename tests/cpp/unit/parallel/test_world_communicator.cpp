#include <gtest/gtest.h>

#include <pops/parallel/world_communicator.hpp>
#include <pops/runtime/output_piece_collective.hpp>

#include <atomic>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string rank_payload(int rank) {
  std::string result(static_cast<std::size_t>(rank + 1), static_cast<char>('A' + (rank % 26)));
  result.push_back('\0');
  result += std::to_string(rank);
  return result;
}

}  // namespace

TEST(WorldCommunicator, IsAnOpaqueProcessSingletonWithOwnedResources) {
  pops::WorldCommunicator& world = pops::WorldCommunicator::world();
  EXPECT_EQ(&world, &pops::WorldCommunicator::world());

#ifdef POPS_HAS_MPI
  EXPECT_TRUE(world.active());
  EXPECT_EQ(world.identity(), "MPI_COMM_WORLD");
  EXPECT_GE(world.rank(), 0);
  EXPECT_GT(world.size(), 0);
  EXPECT_TRUE(world.initialized_by_pops());
  EXPECT_TRUE(world.atexit_finalize_registered());
  EXPECT_GE(world.thread_level(), MPI_THREAD_MULTIPLE);
  const pops::NativeMpiDatatype& datatype = world.datatype_float64();
  EXPECT_EQ(datatype.identity(), "MPI_DOUBLE");
  EXPECT_TRUE(world.owns_float64_datatype(datatype));
  // MPI_Comm_c2f/MPI_Type_c2f may legally map predefined handles to zero.  Calling the projection
  // is itself the test: it must be native, active and non-throwing.
  EXPECT_NO_THROW((void)world.fortran_handle());
  EXPECT_NO_THROW((void)datatype.fortran_handle());
#else
  EXPECT_FALSE(world.active());
  EXPECT_EQ(world.identity(), "serial");
  EXPECT_EQ(world.rank(), 0);
  EXPECT_EQ(world.size(), 1);
  EXPECT_FALSE(world.initialized_by_pops());
  EXPECT_FALSE(world.atexit_finalize_registered());
  EXPECT_THROW((void)world.fortran_handle(), std::runtime_error);
  EXPECT_THROW((void)world.datatype_float64(), std::runtime_error);
#endif
}

TEST(WorldCommunicator, RequiresProcessGlobalThreadMultipleForNativeCallSites) {
  pops::WorldCommunicator& world = pops::WorldCommunicator::world();
#ifdef POPS_HAS_MPI
  ASSERT_GE(world.thread_level(), MPI_THREAD_MULTIPLE);
  const int expected_rank = world.rank();
  const int expected_size = world.size();
  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 4; ++worker) {
    workers.emplace_back([&] {
      for (int iteration = 0; iteration < 128; ++iteration) {
        if (!world.active() || world.rank() != expected_rank || world.size() != expected_size)
          failures.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (std::thread& worker : workers)
    worker.join();
  EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
#else
  EXPECT_EQ(world.thread_level(), 0);
#endif
}

TEST(WorldCommunicator, TransfersEmptyNullAndVariableSizedBytes) {
  pops::WorldCommunicator& world = pops::WorldCommunicator::world();
  const int rank = world.rank();
  const int size = world.size();

  const std::string root_payload("root\0bytes", 10);
  EXPECT_EQ(world.broadcast_bytes(rank == 0 ? root_payload : std::string("ignored"), 0),
            root_payload);
  EXPECT_TRUE(world.broadcast_bytes({}, 0).empty());

  const std::vector<std::string> gathered = world.allgather_bytes(rank_payload(rank));
  EXPECT_EQ(gathered.size(), static_cast<std::size_t>(size));
  if (gathered.size() == static_cast<std::size_t>(size)) {
    for (int source = 0; source < size; ++source)
      EXPECT_EQ(gathered[static_cast<std::size_t>(source)], rank_payload(source));
  }

  const std::optional<std::vector<std::string>> root_only =
      world.gather_bytes(rank_payload(rank), 0);
  if (rank == 0) {
    EXPECT_TRUE(root_only.has_value());
    if (root_only.has_value()) {
      EXPECT_EQ(root_only->size(), static_cast<std::size_t>(size));
      if (root_only->size() == static_cast<std::size_t>(size)) {
        for (int source = 0; source < size; ++source)
          EXPECT_EQ((*root_only)[static_cast<std::size_t>(source)], rank_payload(source));
      }
    }
  } else {
    EXPECT_FALSE(root_only.has_value());
  }
  EXPECT_NO_THROW(world.barrier());
  EXPECT_THROW((void)world.broadcast_bytes({}, size), std::out_of_range);
}

TEST(WorldCommunicator, GathersOutputPiecesOnlyOnRoot) {
  pops::WorldCommunicator& world = pops::WorldCommunicator::world();
#ifdef POPS_HAS_MPI
  const int rank = world.rank();
  const int size = world.size();
  std::vector<pops::OutputPiece> result = pops::output_pieces_to_root(
      world, pops::detail::output_collective_identity("test", "state", "tracer", 0), [rank] {
        pops::OutputPiece piece;
        piece.box = pops::PatchBox{0, rank, 0, rank, 0};
        piece.global_box_index = rank;
        piece.owner_rank = rank;
        piece.ncomp = 2;
        piece.values = {static_cast<double>(rank), static_cast<double>(rank) + 0.5};
        return std::vector<pops::OutputPiece>{std::move(piece)};
      });
  if (rank == 0) {
    EXPECT_EQ(result.size(), static_cast<std::size_t>(size));
    if (result.size() == static_cast<std::size_t>(size)) {
      for (int source = 0; source < size; ++source) {
        const pops::OutputPiece& piece = result[static_cast<std::size_t>(source)];
        EXPECT_EQ(piece.global_box_index, source);
        EXPECT_EQ(piece.owner_rank, source);
        EXPECT_EQ(piece.values, (std::vector<double>{static_cast<double>(source),
                                                     static_cast<double>(source) + 0.5}));
      }
    }
  } else {
    EXPECT_TRUE(result.empty());
  }
#else
  EXPECT_THROW((void)pops::output_pieces_to_root(
                   world, pops::detail::output_collective_identity("test", "state", "tracer", 0),
                   [] { return std::vector<pops::OutputPiece>{}; }),
               std::runtime_error);
#endif
}

TEST(WorldCommunicator, SelectsOneCanonicalReplicatedOutputContributor) {
  pops::WorldCommunicator& world = pops::WorldCommunicator::world();
#ifdef POPS_HAS_MPI
  const int rank = world.rank();
  std::vector<pops::OutputPiece> result = pops::output_pieces_to_root(
      world, pops::detail::output_collective_identity("test", "state", "replicated", 0), [rank] {
        pops::OutputPiece piece;
        piece.box = pops::PatchBox{0, 0, 0, 0, 0};
        piece.global_box_index = 0;
        piece.owner_rank = rank;
        piece.replicated = true;
        piece.ncomp = 1;
        piece.values = {42.0};
        return std::vector<pops::OutputPiece>{std::move(piece)};
      });
  if (rank == 0) {
    EXPECT_EQ(result.size(), 1U);
    if (result.size() == 1U) {
      EXPECT_EQ(result.front().owner_rank, 0);
      EXPECT_TRUE(result.front().replicated);
      EXPECT_EQ(result.front().values, (std::vector<double>{42.0}));
    }
  } else {
    EXPECT_TRUE(result.empty());
  }
#else
  EXPECT_THROW(
      (void)pops::output_pieces_to_root(
          world, pops::detail::output_collective_identity("test", "state", "replicated", 0),
          [] { return std::vector<pops::OutputPiece>{}; }),
      std::runtime_error);
#endif
}
