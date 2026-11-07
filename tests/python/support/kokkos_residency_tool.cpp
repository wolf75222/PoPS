// A Kokkos Tools library for the residency tests of DM8: it reports, for each step, the deep
// copies that Kokkos makes between the start of the step and its commit, by kind of data.
//
// The runtime opens the region "pops::step" around System::step and AmrSystem::step, and the
// region "pops::program_region" around a Program region of a step that the caller opens and
// commits. Finite arrays label their columns "pops_finite_array:" (target memory space) and
// their host staging "pops_finite_host:"; O(1) control views are labelled
// "pops_finite_control:". A deep copy that reads or writes a view with one of the first two
// labels moves array data. When KOKKOS_TOOLS_LIBS names this library and
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
  Tally array_data, control, other;
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

bool starts_with(const char* text, const char* prefix) {
  return text && std::strncmp(text, prefix, std::strlen(prefix)) == 0;
}

bool array_label(const char* label) {
  return starts_with(label, "pops_finite_array:") || starts_with(label, "pops_finite_host:");
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
  std::string line = "{\"window\": " + std::to_string(completed) +
                     ", \"region\": " + quoted(current.region) +
                     ", \"array_data\": " + tally(current.array_data) +
                     ", \"control\": " + tally(current.control) +
                     ", \"other\": " + tally(current.other) + ", \"array_labels\": [";
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

extern "C" void kokkosp_begin_deep_copy(SpaceHandle, const char* destination, const void*,
                                        SpaceHandle, const char* source, const void*,
                                        std::uint64_t bytes) {
  const std::lock_guard<std::mutex> lock(guard);
  if (open_windows == 0)
    return;
  Tally* target = &current.other;
  if (array_label(destination) || array_label(source)) {
    target = &current.array_data;
    current.array_labels.insert(array_label(source) ? source : destination);
  } else if (starts_with(destination, "pops_finite_control:") ||
             starts_with(source, "pops_finite_control:"))
    target = &current.control;
  ++target->count;
  target->bytes += bytes;
}

extern "C" void kokkosp_allocate_data(SpaceHandle space, const char* label, const void*,
                                      std::uint64_t) {
  const std::lock_guard<std::mutex> lock(guard);
  if (open_windows == 0 || !starts_with(label, "pops_finite_array:"))
    return;
  ++current.array_allocations[std::string(space.name, strnlen(space.name, sizeof(space.name)))];
}
