// ADC-700 pre-cutover native oracle.
//
// The candidate is intentionally not another C++ implementation.  It is authored by
// benchmarks/adc700/program_cutover.py and reaches the runtime through the public Python
// validate -> resolve -> compile(MODULE) -> bind -> AmrSystem.install_program path.  This source
// remains a small native oracle compiled only from the pinned baseline revision, so the ABBA
// comparison has one stable numerical reference and cannot accidentally compare two in-process
// hand-written candidate implementations.

#include <pops/parallel/comm.hpp>
#include <pops/parallel/world_communicator.hpp>
#include <pops/runtime/amr_system.hpp>
#include <pops/runtime/builders/compiled/amr_dsl_block.hpp>
#include <pops/physics/bricks/source.hpp>
#include <pops/physics/composition/composite.hpp>
#include <pops/physics/fluids/euler.hpp>

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef POPS_ADC700_REVISION
#define POPS_ADC700_REVISION "unknown"
#endif
#ifndef POPS_ADC700_ROUTE_NAME
#define POPS_ADC700_ROUTE_NAME "pre_cutover_native"
#endif

#if !POPS_ADC700_PRE_CUTOVER
#error "The ADC-700 C++ source is the pinned native oracle; use the Python driver for the candidate."
#endif
#ifndef POPS_ADC700_TOOLCHAIN_ATTESTED
#error "The ADC-700 baseline must pass the CMake toolchain-contract preflight before compiling."
#endif
#ifndef POPS_ADC700_TOOLCHAIN_RECEIPT_SHA256
#error "The ADC-700 baseline must bind the CMake-verified toolchain receipt digest."
#endif
#ifndef POPS_ADC700_TOOLCHAIN_REVISION
#error "The ADC-700 baseline must bind the CMake-verified candidate revision."
#endif

namespace {

using namespace pops;

struct ZeroElliptic {
  template <class State>
  POPS_HD Real rhs(const State&) const {
    return Real(0);
  }
};

using GasModel = CompositeModel<Euler, NoSource, ZeroElliptic>;

struct Config {
  int n = 128;
  int warmups = 4;
  int measured_steps = 40;
  double dt = 5.0e-4;
};

bool patch_less(const PatchBox& lhs, const PatchBox& rhs) {
  if (lhs.level != rhs.level)
    return lhs.level < rhs.level;
  if (lhs.ilo != rhs.ilo)
    return lhs.ilo < rhs.ilo;
  if (lhs.jlo != rhs.jlo)
    return lhs.jlo < rhs.jlo;
  if (lhs.ihi != rhs.ihi)
    return lhs.ihi < rhs.ihi;
  return lhs.jhi < rhs.jhi;
}

std::string patch_signature(const PatchBox& patch) {
  return std::to_string(patch.level) + ':' + std::to_string(patch.ilo) + ',' +
         std::to_string(patch.jlo) + ',' + std::to_string(patch.ihi) + ',' +
         std::to_string(patch.jhi);
}

std::string assigned_gpu_uuid() {
  const char* value = std::getenv("POPS_ADC700_GPU_UUID");
  if (value == nullptr || *value == '\0')
    throw std::runtime_error("POPS_ADC700_GPU_UUID is missing for this measurement run");
  const std::string uuid(value);
  if (uuid.find_first_of("\"\\\n\r\t") != std::string::npos)
    throw std::runtime_error("POPS_ADC700_GPU_UUID contains JSON-unsafe bytes");
  return uuid;
}

std::string required_json_env(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0')
    throw std::runtime_error(std::string(name) + " is missing from the toolchain receipt environment");
  const std::string result(value);
  if (result.find_first_of("\"\\\n\r\t") != std::string::npos)
    throw std::runtime_error(std::string(name) + " contains JSON-unsafe bytes");
  return result;
}

std::string toolchain_receipt_json() {
  const std::string path = required_json_env("POPS_ADC700_TOOLCHAIN_RECEIPT");
  if (path.front() != '/')
    throw std::runtime_error("POPS_ADC700_TOOLCHAIN_RECEIPT must be absolute");
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot open the authenticated ADC-700 toolchain receipt");
  std::string json((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  while (!json.empty() && (json.back() == '\n' || json.back() == '\r' || json.back() == ' ' ||
                           json.back() == '\t'))
    json.pop_back();
  if (json.size() < 2 || json.front() != '{' || json.back() != '}')
    throw std::runtime_error("ADC-700 toolchain receipt is not one JSON object");
  for (const unsigned char value : json)
    if (value < 0x20)
      throw std::runtime_error("ADC-700 toolchain receipt contains raw control bytes");
  return json;
}

std::string gpu_assignments_json(const std::string& local_uuid) {
  auto& world = WorldCommunicator::world();
  world.require_active_mpi_world();
  if (world.identity() != "MPI_COMM_WORLD" || world.size() != 4)
    throw std::runtime_error("ADC-700 requires four ranks on MPI_COMM_WORLD");
  const auto uuids = world.allgather_bytes(local_uuid);
  if (uuids.size() != 4)
    throw std::runtime_error("GPU identity allgather did not return four ranks");
  std::string json;
  for (std::size_t rank = 0; rank < uuids.size(); ++rank) {
    if (uuids[rank].empty() || uuids[rank].find_first_of("\"\\\n\r\t") != std::string::npos)
      throw std::runtime_error("GPU identity allgather returned an invalid UUID");
    if (rank > 0)
      json += ',';
    json += "{\"rank\":" + std::to_string(rank) + ",\"uuid\":\"" + uuids[rank] + "\"}";
  }
  for (std::size_t left = 0; left < uuids.size(); ++left)
    for (std::size_t right = left + 1; right < uuids.size(); ++right)
      if (uuids[left] == uuids[right])
        throw std::runtime_error("GPU identity allgather did not return four distinct UUIDs");
  return json;
}

int parse_positive_int(const char* text, const char* option) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0 || value > 1'000'000)
    throw std::invalid_argument(std::string(option) + " requires a positive bounded integer");
  return static_cast<int>(value);
}

double parse_positive_double(const char* text, const char* option) {
  char* end = nullptr;
  const double value = std::strtod(text, &end);
  if (end == text || *end != '\0' || !(value > 0.0) || !std::isfinite(value))
    throw std::invalid_argument(std::string(option) + " requires a positive finite value");
  return value;
}

Config parse_config(int argc, char** argv) {
  Config config;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    const auto value = [&](const char* prefix) -> const char* {
      const std::string key(prefix);
      return argument.rfind(key, 0) == 0 ? argument.c_str() + key.size() : nullptr;
    };
    if (const char* raw = value("--n="))
      config.n = parse_positive_int(raw, "--n");
    else if (const char* raw = value("--warmups="))
      config.warmups = parse_positive_int(raw, "--warmups");
    else if (const char* raw = value("--steps="))
      config.measured_steps = parse_positive_int(raw, "--steps");
    else if (const char* raw = value("--dt="))
      config.dt = parse_positive_double(raw, "--dt");
    else
      throw std::invalid_argument("unknown ADC-700 oracle option: " + argument);
  }
  if (config.n < 16 || config.n % 4 != 0)
    throw std::invalid_argument("--n must be >= 16 and divisible by four");
  return config;
}

std::vector<double> initial_state(int n) {
  const std::size_t cells = static_cast<std::size_t>(n) * n;
  std::vector<double> state(4 * cells, 0.0);
  constexpr double gamma = 1.4;
  constexpr double pi = 3.141592653589793238462643383279502884;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const std::size_t cell = static_cast<std::size_t>(j) * n + i;
      const double x = (static_cast<double>(i) + 0.5) / static_cast<double>(n);
      const double y = (static_cast<double>(j) + 0.5) / static_cast<double>(n);
      const double dx = x - 0.37;
      const double dy = y - 0.41;
      const double density = 1.0 + 0.35 * std::exp(-(dx * dx + dy * dy) / 0.008);
      const double pressure = 2.0 + 0.1 * std::cos(2.0 * pi * x) * std::cos(2.0 * pi * y);
      const double velocity_x = 0.15;
      const double velocity_y = -0.07;
      state[cell] = density;
      state[cells + cell] = density * velocity_x;
      state[2 * cells + cell] = density * velocity_y;
      state[3 * cells + cell] = pressure / (gamma - 1.0) +
                                0.5 * density * (velocity_x * velocity_x + velocity_y * velocity_y);
    }
  return state;
}

int run(const Config& config) {
  AmrSystemConfig native_config;
  native_config.n = config.n;
  native_config.L = 1.0;
  native_config.periodicity = {true, true};
  native_config.regrid_every = 0;
  native_config.distribute_coarse = true;
  native_config.coarse_max_grid = config.n / 2;
  AmrSystem system(native_config);
  system.set_temporal_relations({2}, {1}, {"integral_only"});
  add_compiled_model(system, "gas", GasModel{Euler{Real(1.4)}, NoSource{}, ZeroElliptic{}},
                     "minmod", "rusanov", "conservative", "euler", 1.4);
  system.set_conservative_state("gas", initial_state(config.n));

  const std::string gpu_uuid = assigned_gpu_uuid();
  const std::string gpu_assignments = gpu_assignments_json(gpu_uuid);
  const std::string toolchain_receipt_path = required_json_env("POPS_ADC700_TOOLCHAIN_RECEIPT");
  const std::string toolchain_receipt_sha256 = required_json_env("POPS_ADC700_TOOLCHAIN_RECEIPT_SHA256");
  const std::string toolchain_receipt_revision = required_json_env("POPS_ADC700_TOOLCHAIN_RECEIPT_REVISION");
  if (toolchain_receipt_sha256.size() != 64 || toolchain_receipt_revision.size() != 40 ||
      toolchain_receipt_sha256 != POPS_ADC700_TOOLCHAIN_RECEIPT_SHA256 ||
      toolchain_receipt_revision != POPS_ADC700_TOOLCHAIN_REVISION)
    throw std::runtime_error(
        "ADC-700 toolchain receipt metadata does not match the CMake preflight binding");
  const std::string toolchain = toolchain_receipt_json();

  const double initial_mass = system.mass();
  const int levels = system.n_levels();
  const int patches = system.n_patches();
  const int coarse_local_boxes = system.coarse_local_boxes();
  const int coarse_total_boxes = system.coarse_total_boxes();
  if (coarse_local_boxes <= 0 || coarse_total_boxes <= 0 || coarse_local_boxes > coarse_total_boxes)
    throw std::runtime_error("ADC-700 native oracle returned an invalid coarse box distribution");
  auto patch_boxes = system.patch_boxes();
  std::sort(patch_boxes.begin(), patch_boxes.end(), patch_less);
  std::string topology_boxes;
  for (const PatchBox& patch : patch_boxes) {
    if (!topology_boxes.empty())
      topology_boxes += ';';
    topology_boxes += patch_signature(patch);
  }

  for (int step = 0; step < config.warmups; ++step)
    system.step(config.dt);

  Kokkos::fence();
  barrier();
  const auto begin = std::chrono::steady_clock::now();
  for (int step = 0; step < config.measured_steps; ++step)
    system.step(config.dt);
  Kokkos::fence();
  const auto end = std::chrono::steady_clock::now();
  barrier();

  const double local_seconds = std::chrono::duration<double>(end - begin).count();
  const double seconds = all_reduce_max(local_seconds);
  const double final_mass = system.mass();
  const std::vector<double> state = system.block_level_state_global("gas", 0);

  double checksum = 0.0;
  double checksum_square = 0.0;
  double maximum = 0.0;
  bool finite = std::isfinite(final_mass);
  for (const double value : state) {
    finite = finite && std::isfinite(value);
    checksum += value;
    checksum_square += value * value;
    maximum = std::max(maximum, std::fabs(value));
  }
  const double mass_error = std::fabs(final_mass - initial_mass);
  const double mass_tolerance = 1.0e-9 * std::max(1.0, std::fabs(initial_mass));
  const bool passed =
      levels >= 1 && finite && maximum > 0.0 && mass_error <= mass_tolerance && seconds > 0.0;

  if (my_rank() == 0) {
    std::printf(
        "{\"schema\":\"pops.adc700.program_cutover.measurement.v1\","
        "\"route\":\"%s\",\"revision\":\"%s\",\"execution_space\":\"%s\","
        "\"mpi_ranks\":%d,\"mpi_communicator\":\"MPI_COMM_WORLD\","
        "\"toolchain_build_attested\":true,"
        "\"execution_concurrency\":%d,\"real_bytes\":%zu,"
        "\"parameters\":{\"n\":%d,\"warmups\":%d,\"measured_steps\":%d,\"dt\":%.17g},"
        "\"topology\":{\"levels\":%d,\"patches\":%d,\"boxes\":\"%s\","
        "\"distribute_coarse\":true,\"coarse_max_grid\":%d,"
        "\"coarse_local_boxes\":%d,\"coarse_total_boxes\":%d},"
        "\"toolchain_receipt\":{\"path\":\"%s\",\"sha256\":\"%s\",\"revision\":\"%s\"},"
        "\"toolchain\":%s,"
        "\"gpu\":{\"rank\":%d,\"uuid\":\"%s\"},\"gpu_uuid\":\"%s\","
        "\"gpu_assignments\":[%s],"
        "\"timing\":{\"seconds\":%.17g,\"per_step_seconds\":%.17g,"
        "\"rank_aggregation\":\"max\",\"device_fence\":\"before_and_after\","
        "\"mpi_barrier\":\"before_and_after\"},"
        "\"signature\":{\"mass\":%.17g,\"initial_mass\":%.17g,\"mass_error\":%.17g,"
        "\"checksum\":%.17g,\"checksum_square\":%.17g,\"maximum\":%.17g},"
        "\"validation\":{\"passed\":%s,\"mass_tolerance\":%.17g}}\n",
        POPS_ADC700_ROUTE_NAME, POPS_ADC700_REVISION, Kokkos::DefaultExecutionSpace::name(),
        n_ranks(), Kokkos::DefaultExecutionSpace().concurrency(), sizeof(Real), config.n,
        config.warmups, config.measured_steps, config.dt, levels, patches, topology_boxes.c_str(),
        config.n / 2, coarse_local_boxes, coarse_total_boxes, toolchain_receipt_path.c_str(),
        toolchain_receipt_sha256.c_str(), toolchain_receipt_revision.c_str(), toolchain.c_str(),
        my_rank(), gpu_uuid.c_str(), gpu_uuid.c_str(), gpu_assignments.c_str(),
        seconds, seconds / static_cast<double>(config.measured_steps), final_mass, initial_mass,
        mass_error, checksum, checksum_square, maximum, passed ? "true" : "false", mass_tolerance);
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
      std::fprintf(stderr, "ADC-700 native oracle failed: %s\n", error.what());
    failed = 1;
  }
  const long collective_failure = pops::all_reduce_max(static_cast<long>(failed));
  pops::barrier();
  Kokkos::finalize();
  pops::comm_finalize();
  return collective_failure == 0 ? 0 : 1;
}
