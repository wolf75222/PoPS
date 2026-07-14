#include <pops_bench/benchmark_support.hpp>

#include <Kokkos_Core.hpp>
#include <pops/core/foundation/types.hpp>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

#ifndef POPS_BENCH_GIT_SHA
#define POPS_BENCH_GIT_SHA "unknown"
#endif
#ifndef POPS_BENCH_GIT_DIRTY
#define POPS_BENCH_GIT_DIRTY 0
#endif
#ifndef POPS_BENCH_COMPILER
#define POPS_BENCH_COMPILER "unknown"
#endif
#ifndef POPS_BENCH_BUILD_TYPE
#define POPS_BENCH_BUILD_TYPE "unknown"
#endif

namespace pops::bench {
namespace {

std::string require_value(int& index, int argc, char** argv, std::string_view option,
                          std::string_view inline_value) {
  if (!inline_value.empty())
    return std::string(inline_value);
  if (index + 1 >= argc)
    throw std::invalid_argument(std::string(option) + " requires a value");
  return argv[++index];
}

int parse_int(std::string_view text, std::string_view option) {
  int result = 0;
  const char* begin = text.data();
  const char* end = text.data() + text.size();
  const auto [ptr, error] = std::from_chars(begin, end, result);
  if (error != std::errc{} || ptr != end)
    throw std::invalid_argument(std::string(option) + " requires an integer, got " +
                                std::string(text));
  return result;
}

double parse_double(std::string_view text, std::string_view option) {
  std::string owned(text);
  char* end = nullptr;
  errno = 0;
  const double result = std::strtod(owned.c_str(), &end);
  if (errno != 0 || end != owned.c_str() + owned.size() || !std::isfinite(result))
    throw std::invalid_argument(std::string(option) + " requires a finite real, got " + owned);
  return result;
}

double quantile(const std::vector<double>& sorted, double q) {
  if (sorted.empty())
    throw std::invalid_argument("cannot compute a quantile of an empty sample");
  const double position = q * static_cast<double>(sorted.size() - 1);
  const std::size_t lower = static_cast<std::size_t>(std::floor(position));
  const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
  const double weight = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

std::string environment_or(std::string_view name, std::string_view fallback) {
  const std::string key(name);
  if (const char* value = std::getenv(key.c_str()); value != nullptr && value[0] != '\0')
    return value;
  return std::string(fallback);
}

std::string utc_now() {
  const std::time_t raw = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm utc{};
#if defined(_WIN32)
  gmtime_s(&utc, &raw);
#else
  gmtime_r(&raw, &utc);
#endif
  std::ostringstream out;
  out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}

std::string json_bool(bool value) { return value ? "true" : "false"; }

}  // namespace

BenchmarkConfig parse_config(int argc, char** argv) {
  BenchmarkConfig config;
  for (int i = 1; i < argc; ++i) {
    const std::string_view raw(argv[i]);
    if (raw == "--help" || raw == "-h") {
      config.help = true;
      continue;
    }
    if (!raw.starts_with("--"))
      throw std::invalid_argument("unexpected positional argument: " + std::string(raw));
    const std::size_t equal = raw.find('=');
    const std::string_view option = raw.substr(0, equal);
    const std::string_view inline_value =
        equal == std::string_view::npos ? std::string_view{} : raw.substr(equal + 1);
    const std::string value = require_value(i, argc, argv, option, inline_value);

    if (option == "--case")
      config.case_id = value;
    else if (option == "--output")
      config.output_path = value;
    else if (option == "--warmups")
      config.warmups = parse_int(value, option);
    else if (option == "--repetitions")
      config.repetitions = parse_int(value, option);
    else if (option == "--arith-n")
      config.arith_n = parse_int(value, option);
    else if (option == "--arith-tile")
      config.arith_tile = parse_int(value, option);
    else if (option == "--arith-components")
      config.arith_components = parse_int(value, option);
    else if (option == "--krylov-n")
      config.krylov_n = parse_int(value, option);
    else if (option == "--krylov-tile")
      config.krylov_tile = parse_int(value, option);
    else if (option == "--krylov-max-iters")
      config.krylov_max_iters = parse_int(value, option);
    else if (option == "--krylov-rel-tol")
      config.krylov_rel_tol = parse_double(value, option);
    else if (option == "--krylov-abs-tol")
      config.krylov_abs_tol = parse_double(value, option);
    else
      throw std::invalid_argument("unknown option: " + std::string(option));
  }

  if (config.case_id != "all" && config.case_id != "arith_halo" &&
      config.case_id != "tensor_krylov")
    throw std::invalid_argument("--case must be all, arith_halo, or tensor_krylov");
  if (config.warmups < 0)
    throw std::invalid_argument("--warmups must be nonnegative");
  if (config.repetitions < 3)
    throw std::invalid_argument("--repetitions must be at least 3 for robust statistics");
  if (config.arith_n < 1 || config.arith_tile < 1 || config.arith_components < 1)
    throw std::invalid_argument("arith dimensions, tile, and component count must be positive");
  if (config.krylov_n < 4 || config.krylov_tile < 1 || config.krylov_max_iters < 1)
    throw std::invalid_argument("Krylov n>=4, tile>=1, and max-iters>=1 are required");
  if (!(config.krylov_rel_tol > 0.0) || !std::isfinite(config.krylov_rel_tol))
    throw std::invalid_argument("--krylov-rel-tol must be positive and finite");
  if (config.krylov_abs_tol < 0.0 || !std::isfinite(config.krylov_abs_tol))
    throw std::invalid_argument("--krylov-abs-tol must be nonnegative and finite");
  return config;
}

void print_help(std::ostream& out) {
  out << "PoPS benchmark harness\n"
      << "  --case all|arith_halo|tensor_krylov\n"
      << "  --warmups N                 discarded warmups (ABBA blocks for arith_halo)\n"
      << "  --repetitions N              samples (ABBA blocks for arith_halo), N >= 3\n"
      << "  --arith-n N                  square arithmetic/halo domain\n"
      << "  --arith-tile N               maximum box edge for arithmetic/halo\n"
      << "  --arith-components N         MultiFab component count\n"
      << "  --krylov-n N                 square manufactured Krylov problem\n"
      << "  --krylov-tile N              maximum box edge for Krylov\n"
      << "  --krylov-max-iters N         BiCGStab iteration cap\n"
      << "  --krylov-rel-tol X           relative residual tolerance\n"
      << "  --krylov-abs-tol X           absolute residual tolerance\n"
      << "  --output PATH                write rank-0 JSONL (stdout always receives it)\n";
}

RobustStats summarize(const std::vector<double>& samples) {
  if (samples.empty())
    throw std::invalid_argument("cannot summarize an empty sample");
  std::vector<double> sorted = samples;
  for (const double value : sorted)
    if (!std::isfinite(value))
      throw std::invalid_argument("cannot summarize non-finite samples");
  std::sort(sorted.begin(), sorted.end());
  RobustStats stats;
  stats.count = sorted.size();
  stats.minimum = sorted.front();
  stats.p10 = quantile(sorted, 0.10);
  stats.median = quantile(sorted, 0.50);
  stats.p90 = quantile(sorted, 0.90);
  stats.maximum = sorted.back();

  std::vector<double> deviations;
  deviations.reserve(sorted.size());
  for (const double value : sorted)
    deviations.push_back(std::fabs(value - stats.median));
  std::sort(deviations.begin(), deviations.end());
  stats.mad = quantile(deviations, 0.50);

  // Keep the statistic genuinely trimmed at the default seven repetitions while retaining enough
  // observations for a useful central estimate.
  const std::size_t trim = sorted.size() >= 5 ? std::max<std::size_t>(1, sorted.size() / 10) : 0;
  const auto first = sorted.begin() + static_cast<std::ptrdiff_t>(trim);
  const auto last = sorted.end() - static_cast<std::ptrdiff_t>(trim);
  stats.trimmed_mean = std::accumulate(first, last, 0.0) /
                       static_cast<double>(std::distance(first, last));
  return stats;
}

RuntimeMetadata collect_runtime_metadata() {
  RuntimeMetadata metadata;
  metadata.timestamp_utc = utc_now();
  metadata.hostname = environment_or("HOSTNAME", "unknown");
  metadata.git_sha = POPS_BENCH_GIT_SHA;
  metadata.compiler = POPS_BENCH_COMPILER;
  metadata.build_type = POPS_BENCH_BUILD_TYPE;
  metadata.execution_space = Kokkos::DefaultExecutionSpace::name();
  metadata.slurm_job_id = environment_or("SLURM_JOB_ID", "");
  metadata.mpi_ranks = pops::n_ranks();
  metadata.execution_concurrency = Kokkos::DefaultExecutionSpace().concurrency();
  metadata.real_bytes = static_cast<int>(sizeof(pops::Real));
  metadata.source_dirty = POPS_BENCH_GIT_DIRTY != 0;
  return metadata;
}

std::string json_escape(std::string_view value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char ch : value) {
    switch (ch) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20)
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch) << std::dec << std::setfill(' ');
        else
          out << static_cast<char>(ch);
    }
  }
  out << '"';
  return out.str();
}

std::string json_number(double value) {
  if (!std::isfinite(value))
    return "null";
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

std::string json_number_array(const std::vector<double>& values) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      out << ',';
    out << json_number(values[i]);
  }
  out << ']';
  return out.str();
}

std::string json_integer_array(const std::vector<int>& values) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0)
      out << ',';
    out << values[i];
  }
  out << ']';
  return out.str();
}

std::string stats_json(const std::vector<double>& samples) {
  const RobustStats stats = summarize(samples);
  std::ostringstream out;
  out << "{\"count\":" << stats.count << ",\"min\":" << json_number(stats.minimum)
      << ",\"p10\":" << json_number(stats.p10)
      << ",\"median\":" << json_number(stats.median)
      << ",\"p90\":" << json_number(stats.p90)
      << ",\"max\":" << json_number(stats.maximum)
      << ",\"mad\":" << json_number(stats.mad)
      << ",\"trimmed_mean\":" << json_number(stats.trimmed_mean)
      << ",\"samples\":" << json_number_array(samples) << '}';
  return out.str();
}

std::string metadata_json(const RuntimeMetadata& metadata) {
  std::ostringstream out;
  out << "{\"git_sha\":" << json_escape(metadata.git_sha)
      << ",\"source_dirty\":" << json_bool(metadata.source_dirty)
      << ",\"compiler\":" << json_escape(metadata.compiler)
      << ",\"build_type\":" << json_escape(metadata.build_type)
      << ",\"execution_space\":" << json_escape(metadata.execution_space)
      << ",\"execution_concurrency\":" << metadata.execution_concurrency
      << ",\"mpi_ranks\":" << metadata.mpi_ranks << ",\"real_bytes\":" << metadata.real_bytes
      << ",\"hostname\":" << json_escape(metadata.hostname)
      << ",\"slurm_job_id\":" << json_escape(metadata.slurm_job_id) << '}';
  return out.str();
}

std::string record_prefix(const RuntimeMetadata& metadata, std::string_view case_id,
                          std::string_view variant, std::string_view protocol,
                          std::string_view record_type) {
  std::ostringstream out;
  out << "{\"schema\":\"pops.benchmark.v1\",\"timestamp_utc\":"
      << json_escape(metadata.timestamp_utc) << ",\"record_type\":" << json_escape(record_type)
      << ",\"case\":" << json_escape(case_id) << ",\"variant\":" << json_escape(variant)
      << ",\"protocol\":" << json_escape(protocol)
      << ",\"metadata\":" << metadata_json(metadata);
  return out.str();
}

class JsonlWriter::Impl {
 public:
  std::ofstream file;
};

JsonlWriter::JsonlWriter(std::string path, bool enabled)
    : enabled_(enabled), path_(std::move(path)), impl_(enabled ? new Impl() : nullptr) {
  long local_failure = 0;
  if (enabled_ && !path_.empty()) {
    impl_->file.open(path_, std::ios::out | std::ios::trunc);
    if (!impl_->file)
      local_failure = 1;
  }
  // Construction happens collectively. Propagate a rank-0 filesystem error before another rank
  // can enter a benchmark collective while rank 0 unwinds.
  if (pops::all_reduce_max(local_failure) != 0) {
    delete impl_;
    impl_ = nullptr;
    throw std::runtime_error("cannot open JSONL output: " + path_);
  }
}

JsonlWriter::~JsonlWriter() { delete impl_; }

void JsonlWriter::write(const std::string& record) {
  long local_failure = 0;
  if (enabled_) {
    std::cout << record << '\n';
    std::cout.flush();
    if (!std::cout)
      local_failure = 1;
    if (impl_->file.is_open()) {
      impl_->file << record << '\n';
      impl_->file.flush();
      if (!impl_->file)
        local_failure = 1;
    }
  }
  // Every runner calls write on every rank. Make output failure a collective failure so no rank
  // proceeds to a later timed collective alone.
  if (pops::all_reduce_max(local_failure) != 0)
    throw std::runtime_error("failed while writing benchmark JSONL output");
}

}  // namespace pops::bench
