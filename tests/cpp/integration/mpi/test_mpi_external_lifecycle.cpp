#include <mpi.h>

#include <pops/parallel/comm.hpp>

#include <exception>
#include <iostream>
#include <string_view>

namespace {

void record(bool condition, std::string_view message, int rank, int& failures) {
  if (condition)
    return;
  ++failures;
  std::cerr << "rank " << rank << ": " << message << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  int provided = MPI_THREAD_SINGLE;
  const int init_code = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (init_code != MPI_SUCCESS) {
    std::cerr << "MPI_Init_thread(MPI_THREAD_MULTIPLE) failed with code " << init_code << '\n';
    return 1;
  }

  int rank = -1;
  int failures = 0;
  record(MPI_Comm_rank(MPI_COMM_WORLD, &rank) == MPI_SUCCESS,
         "MPI_Comm_rank failed after application-owned MPI_Init_thread", 0, failures);
  record(provided >= MPI_THREAD_MULTIPLE, "MPI implementation did not provide MPI_THREAD_MULTIPLE",
         rank, failures);
  record(!pops::mpi_initialized_by_pops(), "PoPS claimed ownership of application-initialized MPI",
         rank, failures);
  record(!pops::mpi_atexit_finalize_registered(),
         "PoPS registered an atexit finalizer for application-initialized MPI", rank, failures);

  try {
    pops::comm_init(&argc, &argv);
  } catch (const std::exception& error) {
    ++failures;
    std::cerr << "rank " << rank << ": pops::comm_init failed to attach: " << error.what() << '\n';
  }

  record(pops::comm_active(), "PoPS did not attach to active MPI_COMM_WORLD", rank, failures);
  record(!pops::mpi_initialized_by_pops(), "PoPS claimed MPI ownership while attaching", rank,
         failures);
  record(!pops::mpi_atexit_finalize_registered(),
         "PoPS registered an atexit finalizer while attaching", rank, failures);

  pops::comm_finalize();

  int initialized = 0;
  int finalized = 0;
  record(MPI_Initialized(&initialized) == MPI_SUCCESS && initialized != 0,
         "pops::comm_finalize deactivated application-owned MPI", rank, failures);
  record(MPI_Finalized(&finalized) == MPI_SUCCESS && finalized == 0,
         "pops::comm_finalize finalized application-owned MPI", rank, failures);
  record(pops::comm_active(), "MPI_COMM_WORLD is not usable after pops::comm_finalize", rank,
         failures);
  record(!pops::mpi_initialized_by_pops(), "PoPS ownership changed during pops::comm_finalize",
         rank, failures);
  record(!pops::mpi_atexit_finalize_registered(),
         "PoPS registered an atexit finalizer during pops::comm_finalize", rank, failures);

  record(MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS,
         "MPI_COMM_WORLD barrier failed after pops::comm_finalize", rank, failures);

  int global_failures = failures;
  if (MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD) !=
      MPI_SUCCESS) {
    ++global_failures;
    std::cerr << "rank " << rank << ": MPI_Allreduce failed before application finalization\n";
  }

  const int finalize_code = MPI_Finalize();
  if (finalize_code != MPI_SUCCESS) {
    std::cerr << "rank " << rank << ": application MPI_Finalize failed with code " << finalize_code
              << '\n';
    return 1;
  }

  finalized = 0;
  if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized == 0) {
    std::cerr << "rank " << rank << ": application MPI_Finalize did not finalize MPI\n";
    return 1;
  }
  return global_failures == 0 ? 0 : 1;
}
