// A Kokkos Tools library for the residency tests of DM8: it reports, for each step, the deep
// copies that Kokkos makes between the start of the step and its commit, by kind of data.
//
// The runtime opens the region "pops::step" around System::step and AmrSystem::step, and the
// region "pops::program_region" around a Program region of a step that the caller opens and
// commits. Finite arrays label their columns "pops_finite_array:" (target memory space) and
// their host staging "pops_finite_host:"; O(1) control views are labelled
// "pops_finite_control:". A deep copy that reads or writes a view with one of the first two
// labels moves array data.
//
// Mesh arrays (G13 lot 2) label their storage "pops_mesh_array:". A mesh copy reaches the host
// when it goes through a host mirror: on a device build the mirror carries the suffix "_mirror",
// on a host build the mirror aliases the storage (same source and destination pointer); both
// count in "mesh_host". Other mesh copies stay in the target memory space ("mesh_device"). Halo
// transports that stage through pinned host memory label those buffers "pops_host_staging:":
// their copies are the reported host phase of PF2 ("host_staging"). Host buffers that also carry
// host work label themselves "pops_host_phase:": their copies are no admissible phase of PF2 and
// count apart ("host_phase"). Allocations labelled
// "pops_unified:" live in unified memory (SharedSpace): each window reports how many are alive
// when it opens and how many it makes, the arrays whose page migrations DM8 requires to stay at
// zero on a GPU target. When KOKKOS_TOOLS_LIBS names this library and
// POPS_KOKKOS_RESIDENCY_OUTPUT names a file, one JSON line is written per completed outermost
// window. The library needs no Kokkos header: it implements the C interface of Kokkos Tools.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace {

struct SpaceHandle {
  char name[64];
};

struct Tally {
  std::uint64_t count = 0;
  std::uint64_t bytes = 0;
};

struct Window {
  std::string region;
  Tally array_data, control, other, mesh_host, mesh_device, host_staging, host_phase;
  Tally unified_live, unified_allocations;
  std::set<std::string> array_labels;
  // Allocations of finite array columns made in the window, by memory space name.
  std::map<std::string, std::uint64_t> array_allocations;
};

std::mutex guard;
std::FILE* output = nullptr;
std::vector<std::string> regions;
int open_windows = 0;
std::uint64_t completed = 0;
Window current;
// Live unified-memory allocations, by pointer: their bytes.
std::map<const void*, std::uint64_t> unified;

bool starts_with(const char* text, const char* prefix) {
  return text && std::strncmp(text, prefix, std::strlen(prefix)) == 0;
}

bool array_label(const char* label) {
  return starts_with(label, "pops_finite_array:") || starts_with(label, "pops_finite_host:");
}

bool mesh_label(const char* label) {
  return starts_with(label, "pops_mesh_array:");
}

bool ends_with(const char* text, const char* suffix) {
  if (!text)
    return false;
  const std::size_t length = std::strlen(text), tail = std::strlen(suffix);
  return length >= tail && std::strcmp(text + length - tail, suffix) == 0;
}

bool window_region(const std::string& name) {
  return name == "pops::step" || name == "pops::program_region";
}

std::string quoted(const std::string& text) {
  std::string out = "\"";
  for (const char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      char escaped[8];
      std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(c));
      out += escaped;
    } else
      out += c;
  }
  return out + "\"";
}

std::string tally(const Tally& value) {
  return "{\"count\": " + std::to_string(value.count) +
         ", \"bytes\": " + std::to_string(value.bytes) + "}";
}

void write_window() {
  if (!output)
    return;
  std::string line =
      "{\"window\": " + std::to_string(completed) + ", \"region\": " + quoted(current.region) +
      ", \"array_data\": " + tally(current.array_data) +
      ", \"control\": " + tally(current.control) + ", \"other\": " + tally(current.other) +
      ", \"mesh_host\": " + tally(current.mesh_host) +
      ", \"mesh_device\": " + tally(current.mesh_device) +
      ", \"host_staging\": " + tally(current.host_staging) +
      ", \"host_phase\": " + tally(current.host_phase) +
      ", \"unified_live\": " + tally(current.unified_live) +
      ", \"unified_allocations\": " + tally(current.unified_allocations) + ", \"array_labels\": [";
  bool first = true;
  for (const auto& label : current.array_labels) {
    line += (first ? "" : ", ") + quoted(label);
    first = false;
  }
  line += "], \"array_allocations\": {";
  first = true;
  for (const auto& [space, count] : current.array_allocations) {
    line += (first ? "" : ", ") + quoted(space) + ": " + std::to_string(count);
    first = false;
  }
  line += "}}\n";
  std::fputs(line.c_str(), output);
  std::fflush(output);
}

}  // namespace

extern "C" void kokkosp_init_library(int, std::uint64_t, std::uint32_t, void*) {
  const std::lock_guard<std::mutex> lock(guard);
  const char* path = std::getenv("POPS_KOKKOS_RESIDENCY_OUTPUT");
  if (path && *path)
    output = std::fopen(path, "w");
}

extern "C" void kokkosp_finalize_library() {
  const std::lock_guard<std::mutex> lock(guard);
  if (output)
    std::fclose(output);
  output = nullptr;
}

extern "C" void kokkosp_push_profile_region(const char* name) {
  const std::lock_guard<std::mutex> lock(guard);
  regions.emplace_back(name ? name : "");
  if (!window_region(regions.back()))
    return;
  if (open_windows++ == 0) {
    current = Window{};
    current.region = regions.back();
    current.unified_live.count = unified.size();
    for (const auto& [pointer, bytes] : unified)
      current.unified_live.bytes += bytes;
  }
}

extern "C" void kokkosp_pop_profile_region() {
  const std::lock_guard<std::mutex> lock(guard);
  if (regions.empty())
    return;
  const bool closes = window_region(regions.back());
  regions.pop_back();
  if (closes && --open_windows == 0) {
    write_window();
    ++completed;
  }
}

extern "C" void kokkosp_begin_deep_copy(SpaceHandle, const char* destination,
                                        const void* destination_pointer, SpaceHandle,
                                        const char* source, const void* source_pointer,
                                        std::uint64_t bytes) {
  const std::lock_guard<std::mutex> lock(guard);
  if (open_windows == 0)
    return;
  Tally* target = &current.other;
  if (mesh_label(destination) || mesh_label(source)) {
    const bool host = ends_with(destination, "_mirror") || ends_with(source, "_mirror") ||
                      (destination_pointer != nullptr && destination_pointer == source_pointer);
    target = host ? &current.mesh_host : &current.mesh_device;
  } else if (starts_with(destination, "pops_host_phase:") ||
             starts_with(source, "pops_host_phase:"))
    target = &current.host_phase;
  else if (starts_with(destination, "pops_host_staging:") ||
           starts_with(source, "pops_host_staging:"))
    target = &current.host_staging;
  else if (array_label(destination) || array_label(source)) {
    target = &current.array_data;
    current.array_labels.insert(array_label(source) ? source : destination);
  } else if (starts_with(destination, "pops_finite_control:") ||
             starts_with(source, "pops_finite_control:"))
    target = &current.control;
  ++target->count;
  target->bytes += bytes;
}

extern "C" void kokkosp_allocate_data(SpaceHandle space, const char* label, const void* pointer,
                                      std::uint64_t bytes) {
  const std::lock_guard<std::mutex> lock(guard);
  if (starts_with(label, "pops_unified:")) {
    unified[pointer] = bytes;
    if (open_windows != 0) {
      ++current.unified_allocations.count;
      current.unified_allocations.bytes += bytes;
    }
  }
  if (open_windows == 0 || !starts_with(label, "pops_finite_array:"))
    return;
  ++current.array_allocations[std::string(space.name, strnlen(space.name, sizeof(space.name)))];
}

extern "C" void kokkosp_deallocate_data(SpaceHandle, const char* label, const void* pointer,
                                        std::uint64_t) {
  const std::lock_guard<std::mutex> lock(guard);
  if (starts_with(label, "pops_unified:"))
    unified.erase(pointer);
}
