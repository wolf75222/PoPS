#include <mpi.h>
#include <stddef.h>
#include <stdint.h>

static int active;
static int collectives;
static int traversals;

void pops_measure_selection_begin(void) {
  collectives = 0;
  traversals = 0;
  active = 1;
}

void pops_measure_selection_end(int* mpi, int* kokkos) {
  active = 0;
  *mpi = collectives;
  *kokkos = traversals;
}

#ifdef __APPLE__
static int measured_allreduce(const void* send, void* receive, int count, MPI_Datatype type,
                              MPI_Op op, MPI_Comm comm) {
#else
int MPI_Allreduce(const void* send, void* receive, int count, MPI_Datatype type, MPI_Op op,
                  MPI_Comm comm) {
#endif
  collectives += active;
  return PMPI_Allreduce(send, receive, count, type, op, comm);
}

#ifdef __APPLE__
__attribute__((used, section("__DATA,__interpose"))) static const struct {
  const void* replacement;
  const void* original;
} allreduce_interpose = {(const void*)measured_allreduce, (const void*)MPI_Allreduce};
#endif

// This tool and the PMPI interposer share their counters in the rank process.
void kokkosp_begin_parallel_reduce(const char* name, uint32_t device, uint64_t* event) {
  (void)name;
  (void)device;
  *event = 0;
  traversals += active;
}
