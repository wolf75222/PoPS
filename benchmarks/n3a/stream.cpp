// STREAM-style sustained bandwidth (copy, scale, add, triad) for the N3a roofline.
// Arrays of N doubles, first-touch by the same OpenMP threads; best of `reps` repetitions.
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
  const long n = argc > 1 ? std::atol(argv[1]) : 100'000'000L;
  const int reps = argc > 2 ? std::atoi(argv[2]) : 10;
  double* a = static_cast<double*>(std::aligned_alloc(64, n * sizeof(double)));
  double* b = static_cast<double*>(std::aligned_alloc(64, n * sizeof(double)));
  double* c = static_cast<double*>(std::aligned_alloc(64, n * sizeof(double)));
#pragma omp parallel for schedule(static)
  for (long i = 0; i < n; ++i) {
    a[i] = 1.0;
    b[i] = 2.0;
    c[i] = 0.0;
  }
  const double s = 3.0;
  double best[4] = {1e30, 1e30, 1e30, 1e30};
  for (int r = 0; r < reps; ++r) {
    auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i)
      c[i] = a[i];
    auto t1 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i)
      b[i] = s * c[i];
    auto t2 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i)
      c[i] = a[i] + b[i];
    auto t3 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i)
      a[i] = b[i] + s * c[i];
    auto t4 = std::chrono::steady_clock::now();
    const double d[4] = {std::chrono::duration<double>(t1 - t0).count(),
                         std::chrono::duration<double>(t2 - t1).count(),
                         std::chrono::duration<double>(t3 - t2).count(),
                         std::chrono::duration<double>(t4 - t3).count()};
    for (int k = 0; k < 4; ++k)
      best[k] = std::min(best[k], d[k]);
  }
  const double bytes[4] = {16.0 * n, 16.0 * n, 24.0 * n, 24.0 * n};
  const char* names[4] = {"copy", "scale", "add", "triad"};
  std::printf("{\"threads\": %d, \"n\": %ld", omp_get_max_threads(), n);
  for (int k = 0; k < 4; ++k)
    std::printf(", \"%s_GBps\": %.3f", names[k], bytes[k] / best[k] / 1e9);
  std::printf(", \"check\": %.1f}\n", a[n / 2] + b[n / 3] + c[n / 5]);
  std::free(a);
  std::free(b);
  std::free(c);
  return 0;
}
