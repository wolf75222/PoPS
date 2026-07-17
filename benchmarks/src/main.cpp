#include <pops_bench/benchmark_support.hpp>
#include <pops_bench/cases.hpp>

#include <Kokkos_Core.hpp>
#include <pops/parallel/comm.hpp>

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  long local_failure = 0;
  {
    try {
      const pops::bench::BenchmarkConfig config = pops::bench::parse_config(argc, argv);
      if (config.help) {
        if (pops::my_rank() == 0)
          pops::bench::print_help(std::cout);
      } else {
        pops::bench::JsonlWriter writer(config.output_path, pops::my_rank() == 0);
        const pops::bench::RuntimeMetadata metadata = pops::bench::collect_runtime_metadata();
        if (config.case_id == "all" || config.case_id == "arith_halo")
          pops::bench::run_arith_halo_case(config, metadata, writer);
        if (config.case_id == "all" || config.case_id == "tensor_krylov")
          pops::bench::run_tensor_krylov_case(config, metadata, writer);
      }
    } catch (const std::exception& error) {
      std::cerr << "pops_benchmark rank " << pops::my_rank() << ": " << error.what() << '\n';
      local_failure = 1;
    } catch (...) {
      std::cerr << "pops_benchmark rank " << pops::my_rank() << ": unknown exception\n";
      local_failure = 1;
    }
  }

  const long global_failure = pops::all_reduce_max(local_failure);
  pops::barrier();
  Kokkos::finalize();
  pops::comm_finalize();
  return global_failure == 0 ? 0 : 1;
}
