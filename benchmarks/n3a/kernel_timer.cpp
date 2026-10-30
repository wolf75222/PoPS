// Minimal Kokkos Tools kernel timer for the N3a measurements (host backends only: a host
// kernel ends when its callback returns, so no fence is needed).
// Kernels are keyed by their label and by the call stack of the launch, so that kernels sharing
// the label "pops_for_each_cell" are told apart. Load with KOKKOS_TOOLS_LIBS=<this .so>; call
// n3a_timer_reset() / n3a_timer_dump(path) through ctypes.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Stat {
  double seconds = 0;
  std::uint64_t calls = 0;
  std::string kind;
  std::string label;
};
std::mutex lock;
std::map<std::string, Stat> stats;
std::unordered_map<std::string, std::string> symbolized;
struct Open {
  std::string key;
  std::string kind;
  std::string label;
  Clock::time_point start;
};
std::vector<Open> open;

std::string demangle(const char* mangled) {
  int status = 0;
  char* text = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
  std::string out = status == 0 && text ? text : mangled;
  std::free(text);
  return out;
}

std::string stack_key(std::string& chain) {
  void* frames[24];
  const int n = backtrace(frames, 24);
  std::string key;
  char buffer[32];
  for (int i = 3; i < n; ++i) {
    std::snprintf(buffer, sizeof buffer, "%p,", frames[i]);
    key += buffer;
  }
  auto found = symbolized.find(key);
  if (found != symbolized.end()) {
    chain = found->second;
    return key;
  }
  std::string text;
  for (int i = 3; i < n; ++i) {
    Dl_info info{};
    if (dladdr(frames[i], &info) && info.dli_sname)
      text += demangle(info.dli_sname);
    else
      text += "?";
    text += "\n";
  }
  symbolized.emplace(key, text);
  chain = text;
  return key;
}

void begin(const char* name, const char* kind, std::uint64_t* id) {
  std::lock_guard<std::mutex> guard(lock);
  std::string chain;
  const std::string key = std::string(kind) + "|" + name + "|" + stack_key(chain);
  *id = open.size();
  open.push_back({key, kind, std::string(name) + "\n" + chain, Clock::now()});
}
void end(std::uint64_t id) {
  const auto stop = Clock::now();
  std::lock_guard<std::mutex> guard(lock);
  if (id >= open.size())
    return;
  auto& item = open[id];
  auto& stat = stats[item.key];
  stat.kind = item.kind;
  stat.label = item.label;
  stat.seconds += std::chrono::duration<double>(stop - item.start).count();
  stat.calls += 1;
}
void escape(std::FILE* out, const std::string& text) {
  for (char c : text) {
    if (c == '"' || c == '\\')
      std::fputc('\\', out);
    if (c == '\n') {
      std::fputs("\\n", out);
      continue;
    }
    if (static_cast<unsigned char>(c) < 0x20)
      continue;
    std::fputc(c, out);
  }
}
}  // namespace

extern "C" {
void kokkosp_init_library(const int, const std::uint64_t, const std::uint32_t, void*) {}
void kokkosp_finalize_library() {}
void kokkosp_begin_parallel_for(const char* name, std::uint32_t, std::uint64_t* id) {
  begin(name, "for", id);
}
void kokkosp_end_parallel_for(std::uint64_t id) {
  end(id);
}
void kokkosp_begin_parallel_reduce(const char* name, std::uint32_t, std::uint64_t* id) {
  begin(name, "reduce", id);
}
void kokkosp_end_parallel_reduce(std::uint64_t id) {
  end(id);
}
void kokkosp_begin_parallel_scan(const char* name, std::uint32_t, std::uint64_t* id) {
  begin(name, "scan", id);
}
void kokkosp_end_parallel_scan(std::uint64_t id) {
  end(id);
}
void n3a_timer_reset() {
  std::lock_guard<std::mutex> guard(lock);
  stats.clear();
  open.clear();
}
int n3a_timer_dump(const char* path) {
  std::lock_guard<std::mutex> guard(lock);
  std::FILE* out = std::fopen(path, "w");
  if (!out)
    return 1;
  std::fprintf(out, "[\n");
  bool first = true;
  for (const auto& [key, stat] : stats) {
    std::fprintf(out, "%s {\"kind\": \"%s\", \"name\": \"", first ? "" : ",\n", stat.kind.c_str());
    escape(out, stat.label);
    std::fprintf(out, "\", \"seconds\": %.9e, \"calls\": %llu}", stat.seconds,
                 static_cast<unsigned long long>(stat.calls));
    first = false;
  }
  std::fprintf(out, "\n]\n");
  std::fclose(out);
  return 0;
}
}
