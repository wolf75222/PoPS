#include <pops/parallel/comm.hpp>

int main(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  const bool valid_world = pops::n_ranks() >= 1 && pops::my_rank() >= 0;
  pops::comm_finalize();
  return valid_world ? 0 : 1;
}
