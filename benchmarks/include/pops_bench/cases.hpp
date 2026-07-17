#pragma once

#include <pops_bench/benchmark_support.hpp>

namespace pops::bench {

void run_arith_halo_case(const BenchmarkConfig& config, const RuntimeMetadata& metadata,
                         JsonlWriter& writer);
void run_tensor_krylov_case(const BenchmarkConfig& config, const RuntimeMetadata& metadata,
                            JsonlWriter& writer);

}  // namespace pops::bench
