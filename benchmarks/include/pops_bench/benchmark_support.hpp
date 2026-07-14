#pragma once

#include <pops/mesh/execution/for_each.hpp>
#include <pops/parallel/comm.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iosfwd>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::bench {

struct BenchmarkConfig {
  std::string case_id = "all";
  std::string output_path;
  int warmups = 2;
  int repetitions = 7;
  int arith_n = 1024;
  int arith_tile = 128;
  int arith_components = 4;
  int krylov_n = 128;
  int krylov_tile = 64;
  int krylov_max_iters = 300;
  double krylov_rel_tol = 1e-9;
  double krylov_abs_tol = 0.0;
  bool help = false;
};

BenchmarkConfig parse_config(int argc, char** argv);
void print_help(std::ostream& out);

struct RobustStats {
  std::size_t count = 0;
  double minimum = 0.0;
  double p10 = 0.0;
  double median = 0.0;
  double p90 = 0.0;
  double maximum = 0.0;
  double mad = 0.0;
  double trimmed_mean = 0.0;
};

RobustStats summarize(const std::vector<double>& samples);

struct PairedSamples {
  std::vector<double> a_seconds;
  std::vector<double> b_seconds;
  std::vector<double> a_over_b;
};

template <class Function>
double measure_max_rank_seconds(Function&& function) {
  // Drain setup work before aligning ranks. The timed interval contains only the benchmarked region;
  // the trailing fence makes asynchronous Kokkos work complete before the local clock is stopped.
  pops::device_fence();
  pops::barrier();
  const auto begin = std::chrono::steady_clock::now();
  std::invoke(std::forward<Function>(function));
  pops::device_fence();
  const auto end = std::chrono::steady_clock::now();
  pops::barrier();
  const double local = std::chrono::duration<double>(end - begin).count();
  return pops::all_reduce_max(local);
}

template <class Prepare, class Run, class Observe>
std::vector<double> run_repeated(int warmups, int repetitions, Prepare&& prepare, Run&& run,
                                 Observe&& observe) {
  for (int i = 0; i < warmups; ++i) {
    std::invoke(prepare);
    (void)measure_max_rank_seconds(run);
    std::invoke(observe, false);
  }
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repetitions));
  for (int i = 0; i < repetitions; ++i) {
    std::invoke(prepare);
    samples.push_back(measure_max_rank_seconds(run));
    std::invoke(observe, true);
  }
  return samples;
}

template <class PrepareA, class RunA, class ObserveA, class PrepareB, class RunB, class ObserveB>
PairedSamples run_paired_abba(int warmups, int repetitions, PrepareA&& prepare_a, RunA&& run_a,
                              ObserveA&& observe_a, PrepareB&& prepare_b, RunB&& run_b,
                              ObserveB&& observe_b) {
  auto time_a = [&](bool record, PairedSamples& samples) {
    std::invoke(prepare_a);
    const double elapsed = measure_max_rank_seconds(run_a);
    std::invoke(observe_a, record);
    if (record)
      samples.a_seconds.push_back(elapsed);
    return elapsed;
  };
  auto time_b = [&](bool record, PairedSamples& samples) {
    std::invoke(prepare_b);
    const double elapsed = measure_max_rank_seconds(run_b);
    std::invoke(observe_b, record);
    if (record)
      samples.b_seconds.push_back(elapsed);
    return elapsed;
  };

  PairedSamples samples;
  samples.a_seconds.reserve(static_cast<std::size_t>(2 * repetitions));
  samples.b_seconds.reserve(static_cast<std::size_t>(2 * repetitions));
  samples.a_over_b.reserve(static_cast<std::size_t>(repetitions));

  for (int i = 0; i < warmups; ++i) {
    (void)time_a(false, samples);
    (void)time_b(false, samples);
    (void)time_b(false, samples);
    (void)time_a(false, samples);
  }
  for (int i = 0; i < repetitions; ++i) {
    const double a1 = time_a(true, samples);
    const double b1 = time_b(true, samples);
    const double b2 = time_b(true, samples);
    const double a2 = time_a(true, samples);
    if (!(a1 > 0.0 && a2 > 0.0 && b1 > 0.0 && b2 > 0.0))
      throw std::runtime_error("ABBA produced a non-positive duration");
    // Geometric mean of the two position-balanced ratios: log-ratio averages A(first,last)
    // against B(second,third), cancelling first-order drift across an ABBA block.
    samples.a_over_b.push_back(
        std::exp(0.5 * (std::log(a1) + std::log(a2) - std::log(b1) - std::log(b2))));
  }
  return samples;
}

struct RuntimeMetadata {
  std::string timestamp_utc;
  std::string hostname;
  std::string git_sha;
  std::string compiler;
  std::string build_type;
  std::string execution_space;
  std::string slurm_job_id;
  int mpi_ranks = 1;
  int execution_concurrency = 1;
  int real_bytes = 0;
  bool source_dirty = false;
};

RuntimeMetadata collect_runtime_metadata();
std::string json_escape(std::string_view value);
std::string json_number(double value);
std::string json_number_array(const std::vector<double>& values);
std::string json_integer_array(const std::vector<int>& values);
std::string stats_json(const std::vector<double>& samples);
std::string metadata_json(const RuntimeMetadata& metadata);
std::string record_prefix(const RuntimeMetadata& metadata, std::string_view case_id,
                          std::string_view variant, std::string_view protocol,
                          std::string_view record_type = "measurement");

class JsonlWriter {
 public:
  JsonlWriter(std::string path, bool enabled);
  ~JsonlWriter();

  JsonlWriter(const JsonlWriter&) = delete;
  JsonlWriter& operator=(const JsonlWriter&) = delete;

  void write(const std::string& record);

 private:
  bool enabled_ = false;
  std::string path_;
  class Impl;
  Impl* impl_ = nullptr;
};

}  // namespace pops::bench
