// ADC-757 out-of-CI heterogeneous numerics campaign.
//
// This executable refuses non-accelerator or single-rank runs.  It uses PoPS' prepared stream
// authority for every measured kernel, performs real rank-to-rank migration for the load-balance
// scenario, and reports one baseline/candidate measurement.  The SLURM driver invokes it in ABBA
// order; assemble.py authenticates the ordering and builds the closure report.

#include <pops/parallel/comm.hpp>
#include <pops/runtime/accelerator/prepared_stream_executor.hpp>

#include <Kokkos_Core.hpp>

#ifndef POPS_HAS_MPI
#error "The ADC-757 heterogeneous campaign requires a real MPI build"
#endif
#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef POPS_ADC757_REVISION
#define POPS_ADC757_REVISION "unknown"
#endif
#ifndef POPS_ADC757_BUILD_ID
#define POPS_ADC757_BUILD_ID "unknown"
#endif
#ifndef POPS_ADC757_WHEEL_SHA256
#define POPS_ADC757_WHEEL_SHA256 ""
#endif
#ifndef POPS_ADC757_MODULE_ABI_SHA256
#define POPS_ADC757_MODULE_ABI_SHA256 ""
#endif

namespace {

using Executor = pops::runtime::accelerator::PreparedAcceleratorStreamExecutor<double>;
using Clock = std::chrono::steady_clock;

constexpr std::string_view kMeasurementSchema = "pops.adc757.heterogeneous-numerics.measurement.v1";
constexpr int kLocalSubsteps = 8;

enum class Scenario { PreparedLocalTime, CostAwareLoadBalance };
enum class Route { Baseline, Candidate };

struct Config {
  Scenario scenario = Scenario::PreparedLocalTime;
  Route route = Route::Baseline;
  std::int64_t extent = 32768;
  int inner_iterations = 96;
  int migration_values_per_task = 4096;
  std::string runtime_evidence_sha256;
};

struct Metrics {
  double time_to_solution_seconds = 0.0;
  double throughput_cell_updates_per_second = 0.0;
  double memory_traffic_bytes = 0.0;
  double kernel_launches = 0.0;
  double task_count = 0.0;
  double communication_bytes = 0.0;
  double communication_seconds = 0.0;
  double fallback_count = 0.0;
  double useful_work_cell_updates = 0.0;
  double imbalance_ratio = 1.0;
  double migration_bytes = 0.0;
  double migration_seconds = 0.0;
};

struct Correctness {
  bool passed = false;
  double mass_error = 0.0;
  double restart_max_error = 0.0;
  double rollback_max_error = 0.0;
  double ledger_balance_error = 0.0;
};

struct TimedResult {
  double seconds = 0.0;
  double communication_seconds = 0.0;
};

struct Task {
  int id = 0;
  int weight = 0;
  int baseline_owner = 0;
  int candidate_owner = 0;
};

void mpi_check(int code, const char* operation) {
  if (code == MPI_SUCCESS)
    return;
  char message[MPI_MAX_ERROR_STRING] = {};
  int length = 0;
  MPI_Error_string(code, message, &length);
  throw std::runtime_error(std::string(operation) + " failed: " +
                           std::string(message, static_cast<std::size_t>(std::max(length, 0))));
}

int parse_positive_int(const char* text, const char* option) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0 || value > 100'000'000)
    throw std::invalid_argument(std::string(option) + " requires a positive bounded integer");
  return static_cast<int>(value);
}

std::string parse_sha256(const char* text, const char* option) {
  const std::string value(text);
  if (value.size() != 64 || std::any_of(value.begin(), value.end(), [](char character) {
        return !((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f'));
      }))
    throw std::invalid_argument(std::string(option) + " requires one lowercase sha256 digest");
  return value;
}

Config parse_config(int argc, char** argv) {
  Config config;
  bool have_scenario = false;
  bool have_route = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    const auto value = [&](const char* prefix) -> const char* {
      const std::string key(prefix);
      return argument.rfind(key, 0) == 0 ? argument.c_str() + key.size() : nullptr;
    };
    if (const char* raw = value("--scenario=")) {
      have_scenario = true;
      if (std::string_view(raw) == "prepared_local_time")
        config.scenario = Scenario::PreparedLocalTime;
      else if (std::string_view(raw) == "cost_aware_load_balance")
        config.scenario = Scenario::CostAwareLoadBalance;
      else
        throw std::invalid_argument("unknown ADC-757 scenario: " + std::string(raw));
    } else if (const char* raw = value("--route=")) {
      have_route = true;
      if (std::string_view(raw) == "baseline")
        config.route = Route::Baseline;
      else if (std::string_view(raw) == "candidate")
        config.route = Route::Candidate;
      else
        throw std::invalid_argument("unknown ADC-757 route: " + std::string(raw));
    } else if (const char* raw = value("--extent=")) {
      config.extent = parse_positive_int(raw, "--extent");
    } else if (const char* raw = value("--inner-iterations=")) {
      config.inner_iterations = parse_positive_int(raw, "--inner-iterations");
    } else if (const char* raw = value("--migration-values-per-task=")) {
      config.migration_values_per_task = parse_positive_int(raw, "--migration-values-per-task");
    } else if (const char* raw = value("--runtime-evidence-sha256=")) {
      config.runtime_evidence_sha256 = parse_sha256(raw, "--runtime-evidence-sha256");
    } else {
      throw std::invalid_argument("unknown ADC-757 campaign option: " + argument);
    }
  }
  if (!have_scenario || !have_route || config.runtime_evidence_sha256.empty())
    throw std::invalid_argument("--scenario, --route and --runtime-evidence-sha256 are required");
  if (config.extent < 4096)
    throw std::invalid_argument("--extent must be at least 4096 cells");
  if (config.inner_iterations > 1'000'000)
    throw std::invalid_argument("--inner-iterations must not exceed 1000000");
  if (config.migration_values_per_task > 1'000'000)
    throw std::invalid_argument("--migration-values-per-task must not exceed 1000000");
  return config;
}

const char* scenario_name(Scenario scenario) {
  return scenario == Scenario::PreparedLocalTime ? "prepared_local_time"
                                                 : "cost_aware_load_balance";
}

const char* route_name(Route route) {
  return route == Route::Baseline ? "baseline" : "candidate";
}

struct UpdateKernel {
  double* values = nullptr;
  double increment = 0.0;
  int work = 0;

  KOKKOS_INLINE_FUNCTION void operator()(std::int64_t index) const {
    double burn = 1.0 + static_cast<double>(index % 97) * 1.0e-4;
    for (int iteration = 0; iteration < work; ++iteration)
      burn = burn * 1.00000011920928955078125 + 1.7e-7;
    values[index] += increment + burn * 1.0e-30;
  }
};

void reset_workspaces(Executor& executor, double value = 1.0) {
  for (std::size_t lane = 0; lane < executor.size(); ++lane)
    Kokkos::deep_copy(executor.instance(lane), executor.workspace(lane), value);
  executor.fence_all();
}

void launch_update(Executor& executor, std::size_t lane, std::int64_t extent, int work,
                   double increment, const char* label) {
  executor.launch_for(lane, label, extent,
                      UpdateKernel{executor.workspace_data(lane), increment, work});
}

void run_local_time_route(Executor& executor, const Config& config, Route route) {
  reset_workspaces(executor);
  if (route == Route::Baseline) {
    for (int substep = 0; substep < kLocalSubsteps; ++substep) {
      launch_update(executor, 0, config.extent, config.inner_iterations, 1.0 / kLocalSubsteps,
                    "pops_adc757_global_fast");
      executor.fence(0);
      launch_update(executor, 1, config.extent, config.inner_iterations, 1.0 / kLocalSubsteps,
                    "pops_adc757_global_slow");
      executor.fence(1);
    }
    return;
  }
  for (int substep = 0; substep < kLocalSubsteps; ++substep)
    launch_update(executor, 0, config.extent, config.inner_iterations, 1.0 / kLocalSubsteps,
                  "pops_adc757_local_fast");
  launch_update(executor, 1, config.extent, config.inner_iterations, 1.0, "pops_adc757_local_slow");
  executor.fence_all();
}

template <class View>
double maximum_error(const View& lhs, const View& rhs) {
  if (lhs.extent(0) != rhs.extent(0))
    throw std::logic_error("ADC-757 parity views have different extents");
  double error = 0.0;
  for (std::size_t index = 0; index < lhs.extent(0); ++index)
    error = std::max(error, std::fabs(lhs(index) - rhs(index)));
  return error;
}

template <class View>
double maximum_error_from_value(const View& values, double expected) {
  double error = 0.0;
  for (std::size_t index = 0; index < values.extent(0); ++index)
    error = std::max(error, std::fabs(values(index) - expected));
  return error;
}

template <class View>
double host_sum(const View& values) {
  double sum = 0.0;
  for (std::size_t index = 0; index < values.extent(0); ++index)
    sum += values(index);
  return sum;
}

Correctness validate_local_time(Executor& executor, const Config& config) {
  run_local_time_route(executor, config, Route::Baseline);
  const auto baseline_fast =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, executor.workspace(0));
  const auto baseline_slow =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, executor.workspace(1));

  run_local_time_route(executor, config, Route::Candidate);
  const auto candidate_fast =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, executor.workspace(0));
  const auto candidate_slow =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, executor.workspace(1));

  const double parity_error = std::max(maximum_error(baseline_fast, candidate_fast),
                                       maximum_error(baseline_slow, candidate_slow));
  const double mass_error = std::max(maximum_error_from_value(candidate_fast, 2.0),
                                     maximum_error_from_value(candidate_slow, 2.0));
  const double ledger_error = std::fabs((host_sum(baseline_fast) + host_sum(baseline_slow)) -
                                        (host_sum(candidate_fast) + host_sum(candidate_slow))) /
                              static_cast<double>(2 * config.extent);

  reset_workspaces(executor);
  for (int substep = 0; substep < kLocalSubsteps / 2; ++substep)
    launch_update(executor, 0, config.extent, config.inner_iterations, 1.0 / kLocalSubsteps,
                  "pops_adc757_restart_first_half");
  launch_update(executor, 1, config.extent, config.inner_iterations, 1.0,
                "pops_adc757_restart_slow");
  executor.fence_all();
  const auto accepted_fast =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, executor.workspace(0));
  const auto accepted_slow =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, executor.workspace(1));

  for (int substep = kLocalSubsteps / 2; substep < kLocalSubsteps; ++substep)
    launch_update(executor, 0, config.extent, config.inner_iterations, 1.0 / kLocalSubsteps,
                  "pops_adc757_restart_second_half");
  executor.fence_all();
  const auto restarted_fast =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, executor.workspace(0));
  const auto restarted_slow =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, executor.workspace(1));
  const double restart_error = std::max(maximum_error(candidate_fast, restarted_fast),
                                        maximum_error(candidate_slow, restarted_slow));

  launch_update(executor, 0, config.extent, config.inner_iterations, 17.0,
                "pops_adc757_rejected_attempt");
  launch_update(executor, 1, config.extent, config.inner_iterations, -11.0,
                "pops_adc757_rejected_attempt_slow");
  executor.fence_all();
  Kokkos::deep_copy(executor.instance(0), executor.workspace(0), accepted_fast);
  Kokkos::deep_copy(executor.instance(1), executor.workspace(1), accepted_slow);
  executor.fence_all();
  const auto rolled_back_fast =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, executor.workspace(0));
  const auto rolled_back_slow =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, executor.workspace(1));
  const double rollback_error = std::max(maximum_error(accepted_fast, rolled_back_fast),
                                         maximum_error(accepted_slow, rolled_back_slow));

  const double global_parity = pops::all_reduce_max(parity_error);
  Correctness result;
  result.mass_error = pops::all_reduce_max(mass_error);
  result.restart_max_error = pops::all_reduce_max(restart_error);
  result.rollback_max_error = pops::all_reduce_max(rollback_error);
  result.ledger_balance_error = pops::all_reduce_max(ledger_error);
  result.passed = global_parity <= 1.0e-11 && result.mass_error <= 1.0e-11 &&
                  result.restart_max_error <= 1.0e-11 && result.rollback_max_error <= 1.0e-11 &&
                  result.ledger_balance_error <= 1.0e-11;
  return result;
}

std::vector<Task> make_tasks(int ranks) {
  constexpr int tasks_per_rank = 8;
  const int task_count = tasks_per_rank * ranks;
  std::vector<Task> tasks(static_cast<std::size_t>(task_count));
  for (int task = 0; task < task_count; ++task) {
    const int baseline_owner = task % ranks;
    const int weight = baseline_owner == 0 ? 16 + (task % 3) : 1 + (task % 3);
    tasks[static_cast<std::size_t>(task)] = {task, weight, baseline_owner, -1};
  }

  std::vector<int> order(static_cast<std::size_t>(task_count));
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    return tasks[static_cast<std::size_t>(lhs)].weight >
           tasks[static_cast<std::size_t>(rhs)].weight;
  });
  std::vector<long> loads(static_cast<std::size_t>(ranks), 0);
  for (const int task_index : order) {
    const auto least = std::min_element(loads.begin(), loads.end());
    const int owner = static_cast<int>(std::distance(loads.begin(), least));
    tasks[static_cast<std::size_t>(task_index)].candidate_owner = owner;
    loads[static_cast<std::size_t>(owner)] += tasks[static_cast<std::size_t>(task_index)].weight;
  }
  return tasks;
}

std::vector<long> owner_loads(const std::vector<Task>& tasks, int ranks, Route route) {
  std::vector<long> loads(static_cast<std::size_t>(ranks), 0);
  for (const Task& task : tasks) {
    const int owner = route == Route::Baseline ? task.baseline_owner : task.candidate_owner;
    loads[static_cast<std::size_t>(owner)] += task.weight;
  }
  return loads;
}

double imbalance_ratio(const std::vector<long>& loads) {
  const double total = static_cast<double>(std::accumulate(loads.begin(), loads.end(), 0L));
  const double average = total / static_cast<double>(loads.size());
  return static_cast<double>(*std::max_element(loads.begin(), loads.end())) / average;
}

class MigrationPlan {
 public:
  MigrationPlan(const std::vector<Task>& tasks, int values_per_task)
      : send_counts_(static_cast<std::size_t>(pops::n_ranks()), 0),
        receive_counts_(static_cast<std::size_t>(pops::n_ranks()), 0),
        send_displacements_(static_cast<std::size_t>(pops::n_ranks()), 0),
        receive_displacements_(static_cast<std::size_t>(pops::n_ranks()), 0) {
    const std::size_t bytes_per_task = static_cast<std::size_t>(values_per_task) * sizeof(double);
    if (bytes_per_task > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("ADC-757 migration task payload exceeds MPI int count");
    for (const Task& task : tasks)
      if (task.baseline_owner == pops::my_rank() && task.candidate_owner != task.baseline_owner) {
        int& count = send_counts_[static_cast<std::size_t>(task.candidate_owner)];
        if (count > std::numeric_limits<int>::max() - static_cast<int>(bytes_per_task))
          throw std::overflow_error("ADC-757 migration send count overflows MPI int");
        count += static_cast<int>(bytes_per_task);
      }
    mpi_check(MPI_Alltoall(send_counts_.data(), 1, MPI_INT, receive_counts_.data(), 1, MPI_INT,
                           MPI_COMM_WORLD),
              "MPI_Alltoall(ADC-757 migration counts)");
    for (int rank = 1; rank < pops::n_ranks(); ++rank) {
      send_displacements_[static_cast<std::size_t>(rank)] =
          send_displacements_[static_cast<std::size_t>(rank - 1)] +
          send_counts_[static_cast<std::size_t>(rank - 1)];
      receive_displacements_[static_cast<std::size_t>(rank)] =
          receive_displacements_[static_cast<std::size_t>(rank - 1)] +
          receive_counts_[static_cast<std::size_t>(rank - 1)];
    }
    const int send_bytes = send_displacements_.back() + send_counts_.back();
    const int receive_bytes = receive_displacements_.back() + receive_counts_.back();
    send_.resize(static_cast<std::size_t>(send_bytes));
    receive_.resize(static_cast<std::size_t>(receive_bytes));

    std::vector<int> cursors = send_displacements_;
    for (const Task& task : tasks)
      if (task.baseline_owner == pops::my_rank() && task.candidate_owner != task.baseline_owner) {
        const int destination = task.candidate_owner;
        int& cursor = cursors[static_cast<std::size_t>(destination)];
        const unsigned char value = static_cast<unsigned char>((task.id % 251) + 1);
        std::fill_n(send_.data() + cursor, bytes_per_task, value);
        cursor += static_cast<int>(bytes_per_task);
      }

    const long local_bytes = static_cast<long>(send_.size());
    global_bytes_ = pops::all_reduce_sum(local_bytes);
    if (global_bytes_ <= 0)
      throw std::runtime_error("ADC-757 cost-aware plan did not migrate any task data");
  }

  double migrate() {
    const auto begin = Clock::now();
    mpi_check(MPI_Alltoallv(send_.data(), send_counts_.data(), send_displacements_.data(), MPI_BYTE,
                            receive_.data(), receive_counts_.data(), receive_displacements_.data(),
                            MPI_BYTE, MPI_COMM_WORLD),
              "MPI_Alltoallv(ADC-757 task migration)");
    const auto end = Clock::now();
    return std::chrono::duration<double>(end - begin).count();
  }

  [[nodiscard]] long global_bytes() const noexcept { return global_bytes_; }

  [[nodiscard]] double checksum_error() const {
    const double sent = std::accumulate(send_.begin(), send_.end(), 0.0);
    const double received = std::accumulate(receive_.begin(), receive_.end(), 0.0);
    return std::fabs(pops::all_reduce_sum(sent) - pops::all_reduce_sum(received));
  }

  [[nodiscard]] std::pair<double, double> restart_and_rollback_errors() {
    const std::vector<unsigned char> accepted = receive_;
    std::vector<unsigned char> checkpoint;
    checkpoint.reserve(sizeof(std::uint64_t) + accepted.size());
    const std::uint64_t extent = static_cast<std::uint64_t>(accepted.size());
    const auto* extent_bytes = reinterpret_cast<const unsigned char*>(&extent);
    checkpoint.insert(checkpoint.end(), extent_bytes, extent_bytes + sizeof(extent));
    checkpoint.insert(checkpoint.end(), accepted.begin(), accepted.end());

    std::fill(receive_.begin(), receive_.end(), 0xff);
    std::uint64_t restored_extent = 0;
    std::memcpy(&restored_extent, checkpoint.data(), sizeof(restored_extent));
    if (restored_extent != accepted.size())
      return {1.0, 1.0};
    std::copy(checkpoint.begin() + static_cast<std::ptrdiff_t>(sizeof(restored_extent)),
              checkpoint.end(), receive_.begin());
    const double restart_error = receive_ == accepted ? 0.0 : 1.0;

    for (unsigned char& value : receive_)
      value ^= 0x5a;
    receive_ = accepted;
    const double rollback_error = receive_ == accepted ? 0.0 : 1.0;
    return {pops::all_reduce_max(restart_error), pops::all_reduce_max(rollback_error)};
  }

 private:
  std::vector<int> send_counts_;
  std::vector<int> receive_counts_;
  std::vector<int> send_displacements_;
  std::vector<int> receive_displacements_;
  std::vector<unsigned char> send_;
  std::vector<unsigned char> receive_;
  long global_bytes_ = 0;
};

std::array<long, 2> split_candidate_load(const std::vector<Task>& tasks, int rank) {
  std::array<long, 2> lanes{0, 0};
  std::vector<int> local_weights;
  for (const Task& task : tasks)
    if (task.candidate_owner == rank)
      local_weights.push_back(task.weight);
  std::sort(local_weights.begin(), local_weights.end(), std::greater<>());
  for (const int weight : local_weights) {
    const std::size_t lane = lanes[0] <= lanes[1] ? 0u : 1u;
    lanes[lane] += weight;
  }
  return lanes;
}

int checked_kernel_work(long weight, int inner_iterations) {
  if (weight < 0 || weight > static_cast<long>(std::numeric_limits<int>::max() / inner_iterations))
    throw std::overflow_error("ADC-757 kernel work exceeds the prepared integer range");
  return static_cast<int>(weight * inner_iterations);
}

void run_load_balance_route(Executor& executor, const Config& config, Route route,
                            const std::vector<Task>& tasks, MigrationPlan& migration,
                            double& local_communication_seconds) {
  reset_workspaces(executor);
  if (route == Route::Baseline) {
    const auto loads = owner_loads(tasks, pops::n_ranks(), Route::Baseline);
    const long work = loads[static_cast<std::size_t>(pops::my_rank())];
    launch_update(executor, 0, config.extent, checked_kernel_work(work, config.inner_iterations),
                  1.0, "pops_adc757_round_robin_load");
    executor.fence(0);
    local_communication_seconds = 0.0;
    return;
  }

  local_communication_seconds = migration.migrate();
  const std::array<long, 2> lane_loads = split_candidate_load(tasks, pops::my_rank());
  for (std::size_t lane = 0; lane < lane_loads.size(); ++lane)
    if (lane_loads[lane] != 0)
      launch_update(executor, lane, config.extent,
                    checked_kernel_work(lane_loads[lane], config.inner_iterations), 1.0,
                    "pops_adc757_cost_aware_load");
  executor.fence_all();
}

Correctness validate_load_balance(MigrationPlan& migration, const std::vector<Task>& tasks) {
  migration.migrate();
  const double checksum_error = migration.checksum_error();
  const auto [restart_error, rollback_error] = migration.restart_and_rollback_errors();
  const long baseline_weight = std::accumulate(
      tasks.begin(), tasks.end(), 0L, [](long sum, const Task& task) { return sum + task.weight; });
  const auto candidate_loads = owner_loads(tasks, pops::n_ranks(), Route::Candidate);
  const long candidate_weight = std::accumulate(candidate_loads.begin(), candidate_loads.end(), 0L);
  Correctness result;
  result.mass_error = checksum_error;
  result.restart_max_error = restart_error;
  result.rollback_max_error = rollback_error;
  result.ledger_balance_error = std::fabs(static_cast<double>(baseline_weight - candidate_weight));
  result.passed = result.mass_error <= 1.0e-11 && result.restart_max_error <= 1.0e-11 &&
                  result.rollback_max_error <= 1.0e-11 && result.ledger_balance_error <= 1.0e-11;
  return result;
}

template <class Function>
TimedResult measure(Function&& function, Executor& executor) {
  executor.fence_all();
  pops::barrier();
  double local_communication_seconds = 0.0;
  const auto begin = Clock::now();
  function(local_communication_seconds);
  executor.fence_all();
  const auto end = Clock::now();
  pops::barrier();
  return {pops::all_reduce_max(std::chrono::duration<double>(end - begin).count()),
          pops::all_reduce_max(local_communication_seconds)};
}

double median(std::vector<double> values) {
  if (values.empty())
    throw std::logic_error("ADC-757 median requires samples");
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ? 0.5 * (values[middle - 1] + values[middle]) : values[middle];
}

bool observe_stream_overlap(Executor& executor, const Config& config) {
  const std::int64_t extent = std::min<std::int64_t>(config.extent, 4096);
  const int work = static_cast<int>(
      std::min<std::int64_t>(static_cast<std::int64_t>(config.inner_iterations) * 64, 100'000));
  auto run_sequential = [&](double&) {
    reset_workspaces(executor);
    launch_update(executor, 0, extent, work, 0.0, "pops_adc757_overlap_a0");
    executor.fence(0);
    launch_update(executor, 1, extent, work, 0.0, "pops_adc757_overlap_a1");
    executor.fence(1);
  };
  auto run_concurrent = [&](double&) {
    reset_workspaces(executor);
    launch_update(executor, 0, extent, work, 0.0, "pops_adc757_overlap_b0");
    launch_update(executor, 1, extent, work, 0.0, "pops_adc757_overlap_b1");
    executor.fence_all();
  };
  for (int warmup = 0; warmup < 2; ++warmup) {
    double ignored_communication_seconds = 0.0;
    run_sequential(ignored_communication_seconds);
    run_concurrent(ignored_communication_seconds);
  }
  std::vector<double> ratios;
  ratios.reserve(5);
  for (int block = 0; block < 5; ++block) {
    const double a1 = measure(run_sequential, executor).seconds;
    const double b1 = measure(run_concurrent, executor).seconds;
    const double b2 = measure(run_concurrent, executor).seconds;
    const double a2 = measure(run_sequential, executor).seconds;
    ratios.push_back(std::sqrt((b1 * b2) / (a1 * a2)));
  }
  return pops::all_reduce_max(median(std::move(ratios))) < 0.95;
}

std::vector<std::string> gather_device_uuids() {
  constexpr std::size_t capacity = 128;
  std::string local_uuid;
#if defined(KOKKOS_ENABLE_CUDA)
  int device = -1;
  cudaUUID_t uuid{};
  const cudaError_t device_status = cudaGetDevice(&device);
  const cudaError_t uuid_status =
      device_status == cudaSuccess ? cudaDeviceGetUuid(&uuid, device) : device_status;
  if (device_status == cudaSuccess && uuid_status == cudaSuccess) {
    std::ostringstream encoded;
    encoded << "GPU-" << std::hex << std::setfill('0');
    for (const char byte : uuid.bytes)
      encoded << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(byte));
    local_uuid = encoded.str();
  }
#else
  // CUDA supplies a stable physical UUID directly.  Other accelerator runtimes may inject an
  // equivalent rank-local identifier until their Kokkos device API standardizes one.
  const char* environment = std::getenv("POPS_ADC757_DEVICE_UUID");
  if (environment != nullptr)
    local_uuid = environment;
#endif
  const bool invalid = local_uuid.empty() || local_uuid.size() >= capacity;
  if (pops::all_reduce_max(static_cast<long>(invalid ? 1 : 0)) != 0)
    throw std::runtime_error("every rank requires one bounded physical accelerator UUID");
  std::array<char, capacity> local{};
  std::memcpy(local.data(), local_uuid.data(), local_uuid.size());
  std::vector<char> gathered(capacity * static_cast<std::size_t>(pops::n_ranks()));
  mpi_check(MPI_Allgather(local.data(), static_cast<int>(capacity), MPI_CHAR, gathered.data(),
                          static_cast<int>(capacity), MPI_CHAR, MPI_COMM_WORLD),
            "MPI_Allgather(ADC-757 device UUIDs)");
  std::vector<std::string> result;
  result.reserve(static_cast<std::size_t>(pops::n_ranks()));
  for (int rank = 0; rank < pops::n_ranks(); ++rank)
    result.emplace_back(gathered.data() + static_cast<std::size_t>(rank) * capacity);
  if (std::set<std::string>(result.begin(), result.end()).size() != result.size())
    throw std::runtime_error("ADC-757 requires one distinct accelerator UUID per MPI rank");
  return result;
}

std::string json_escape(std::string_view text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (const char character : text) {
    if (character == '"' || character == '\\')
      escaped.push_back('\\');
    escaped.push_back(character);
  }
  return escaped;
}

void write_metrics(std::ostream& output, const Metrics& metrics) {
  output << "{\"time_to_solution_seconds\":" << metrics.time_to_solution_seconds
         << ",\"throughput_cell_updates_per_second\":" << metrics.throughput_cell_updates_per_second
         << ",\"memory_traffic_bytes\":" << metrics.memory_traffic_bytes
         << ",\"kernel_launches\":" << metrics.kernel_launches
         << ",\"task_count\":" << metrics.task_count
         << ",\"communication_bytes\":" << metrics.communication_bytes
         << ",\"communication_seconds\":" << metrics.communication_seconds
         << ",\"fallback_count\":" << metrics.fallback_count
         << ",\"useful_work_cell_updates\":" << metrics.useful_work_cell_updates
         << ",\"imbalance_ratio\":" << metrics.imbalance_ratio
         << ",\"migration_bytes\":" << metrics.migration_bytes
         << ",\"migration_seconds\":" << metrics.migration_seconds << '}';
}

void write_correctness(std::ostream& output, const Correctness& correctness) {
  output << "{\"passed\":" << (correctness.passed ? "true" : "false")
         << ",\"mass_error\":" << correctness.mass_error
         << ",\"restart_max_error\":" << correctness.restart_max_error
         << ",\"rollback_max_error\":" << correctness.rollback_max_error
         << ",\"ledger_balance_error\":" << correctness.ledger_balance_error << '}';
}

int run(const Config& config) {
  static_cast<void>(parse_sha256(POPS_ADC757_WHEEL_SHA256, "installed wheel identity"));
  static_cast<void>(parse_sha256(POPS_ADC757_MODULE_ABI_SHA256, "installed module ABI identity"));
  if (pops::n_ranks() < 2)
    throw std::runtime_error("ADC-757 heterogeneous evidence requires at least two MPI ranks");
  if (!Executor::backend_can_partition_authentic_streams())
    throw std::runtime_error(std::string("ADC-757 refuses non-accelerator Kokkos backend ") +
                             Kokkos::DefaultExecutionSpace::name());

  std::unique_ptr<Executor> prepared_executor;
  std::string local_preparation_error;
  try {
    prepared_executor =
        std::make_unique<Executor>(Executor::prepare(2, static_cast<std::size_t>(config.extent)));
  } catch (const std::exception& error) {
    local_preparation_error = error.what();
  }
  if (pops::all_reduce_max(static_cast<long>(local_preparation_error.empty() ? 0 : 1)) != 0)
    throw std::runtime_error(
        "accelerator stream preparation failed on at least one MPI rank" +
        (local_preparation_error.empty() ? std::string{} : ": " + local_preparation_error));
  Executor& executor = *prepared_executor;
  const std::vector<std::string> device_uuids = gather_device_uuids();
  const bool overlap_observed = observe_stream_overlap(executor, config);

  std::vector<Task> tasks = make_tasks(pops::n_ranks());
  MigrationPlan migration(tasks, config.migration_values_per_task);
  const Correctness correctness = config.scenario == Scenario::PreparedLocalTime
                                      ? validate_local_time(executor, config)
                                      : validate_load_balance(migration, tasks);

  auto selected_route = [&](double& local_communication_seconds) {
    if (config.scenario == Scenario::PreparedLocalTime) {
      run_local_time_route(executor, config, config.route);
      local_communication_seconds = 0.0;
    } else {
      run_load_balance_route(executor, config, config.route, tasks, migration,
                             local_communication_seconds);
    }
  };
  for (int warmup = 0; warmup < 2; ++warmup) {
    double communication = 0.0;
    selected_route(communication);
    executor.fence_all();
    pops::barrier();
  }
  const TimedResult timing = measure(selected_route, executor);

  Metrics metrics;
  metrics.time_to_solution_seconds = timing.seconds;
  metrics.communication_seconds = timing.communication_seconds;
  if (config.scenario == Scenario::PreparedLocalTime) {
    const double updates_per_rank =
        static_cast<double>(config.extent) *
        (config.route == Route::Baseline ? 2.0 * kLocalSubsteps : kLocalSubsteps + 1.0);
    metrics.useful_work_cell_updates = updates_per_rank * pops::n_ranks();
    metrics.kernel_launches =
        (config.route == Route::Baseline ? 2.0 * kLocalSubsteps : kLocalSubsteps + 1.0) *
        pops::n_ranks();
    metrics.task_count = metrics.kernel_launches;
  } else {
    const long total_weight =
        std::accumulate(tasks.begin(), tasks.end(), 0L,
                        [](long sum, const Task& task) { return sum + task.weight; });
    metrics.useful_work_cell_updates = static_cast<double>(config.extent) * total_weight;
    metrics.task_count = static_cast<double>(tasks.size());
    metrics.kernel_launches = (config.route == Route::Baseline ? 1.0 : 2.0) * pops::n_ranks();
    metrics.imbalance_ratio = imbalance_ratio(owner_loads(tasks, pops::n_ranks(), config.route));
    if (config.route == Route::Candidate) {
      metrics.communication_bytes = static_cast<double>(migration.global_bytes());
      metrics.migration_bytes = static_cast<double>(migration.global_bytes());
      metrics.migration_seconds = timing.communication_seconds;
    }
  }
  metrics.memory_traffic_bytes = 2.0 * sizeof(double) * metrics.useful_work_cell_updates;
  metrics.throughput_cell_updates_per_second =
      metrics.useful_work_cell_updates / metrics.time_to_solution_seconds;

  const bool local_pass = correctness.passed && overlap_observed &&
                          executor.evidence().independent_streams &&
                          executor.evidence().disjoint_workspaces && timing.seconds > 0.0;
  const bool passed = pops::all_reduce_min(static_cast<long>(local_pass ? 1 : 0)) == 1;
  if (pops::my_rank() == 0) {
    std::ostringstream output;
    output << std::setprecision(17);
    output << "{\"schema\":\"" << kMeasurementSchema << "\",\"status\":\""
           << (passed ? "passed" : "failed") << "\",\"revision\":\""
           << json_escape(POPS_ADC757_REVISION) << "\",\"build_identity\":\""
           << json_escape(std::string(POPS_ADC757_BUILD_ID) + "-" +
                          Kokkos::DefaultExecutionSpace::name())
           << "\",\"installed_wheel_sha256\":\"" << POPS_ADC757_WHEEL_SHA256
           << "\",\"module_abi_sha256\":\"" << POPS_ADC757_MODULE_ABI_SHA256
           << "\",\"runtime_evidence_sha256\":\"" << config.runtime_evidence_sha256
           << "\",\"execution_space\":\"" << Kokkos::DefaultExecutionSpace::name()
           << "\",\"mpi_ranks\":" << pops::n_ranks() << ",\"scenario\":\""
           << scenario_name(config.scenario) << "\",\"route\":\"" << route_name(config.route)
           << "\",\"device_assignments\":[";
    for (int rank = 0; rank < pops::n_ranks(); ++rank) {
      if (rank != 0)
        output << ',';
      output << "{\"rank\":" << rank << ",\"uuid\":\""
             << json_escape(device_uuids[static_cast<std::size_t>(rank)]) << "\"}";
    }
    output << "],\"streams\":{\"identities\":[";
    for (std::size_t lane = 0; lane < executor.size(); ++lane) {
      if (lane != 0)
        output << ',';
      output << "\"" << json_escape(executor.stream_identity(lane)) << "\"";
    }
    output << "],\"correctness_parity\":" << (correctness.passed ? "true" : "false")
           << ",\"overlap_observed\":" << (overlap_observed ? "true" : "false")
           << ",\"workspace_disjoint\":"
           << (executor.evidence().disjoint_workspaces ? "true" : "false") << "},\"metrics\":";
    write_metrics(output, metrics);
    output << ",\"correctness\":";
    write_correctness(output, correctness);
    output << '}';
    std::cout << output.str() << '\n';
  }
  return passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  pops::comm_init(&argc, &argv);
  Kokkos::initialize(argc, argv);
  int failed = 0;
  try {
    failed = run(parse_config(argc, argv));
  } catch (const std::exception& error) {
    if (pops::my_rank() == 0)
      std::fprintf(stderr, "ADC-757 heterogeneous campaign failed: %s\n", error.what());
    failed = 1;
  }
  const long collective_failure = pops::all_reduce_max(static_cast<long>(failed));
  pops::barrier();
  Kokkos::finalize();
  pops::comm_finalize();
  return collective_failure == 0 ? 0 : 1;
}
