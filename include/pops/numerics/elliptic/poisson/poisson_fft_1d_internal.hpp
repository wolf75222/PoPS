#pragma once

/// @file
/// @brief Internal host-side 1D FFT support for elliptic solvers.
///
/// This header is SDK support, not a standalone elliptic API.  It preserves the historical
/// forward/inverse convention: forward uses exp(-i*2*pi*k*j/n), inverse uses exp(+i*2*pi*k*j/n)
/// and applies the 1/n normalization.  Non-power-of-two lengths use the explicitly diagnosed
/// O(n^2) direct DFT path because the radix-2 butterfly requires an exact power-of-two extent.

#include <pops/core/foundation/types.hpp>
#include <pops/diagnostics/fallback_diagnostics.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <span>
#include <utility>
#include <vector>

namespace pops::elliptic::poisson::internal {

using host_fft_complex = std::complex<Real>;

inline bool host_fft_is_power_of_two(int extent) noexcept {
  return extent > 0 && (extent & (extent - 1)) == 0;
}

inline void host_direct_dft(std::span<host_fft_complex> values, bool inverse) {
  const int extent = static_cast<int>(values.size());
  std::vector<host_fft_complex> output(static_cast<std::size_t>(extent));
  const double sign = inverse ? 1.0 : -1.0;
  for (int frequency = 0; frequency < extent; ++frequency) {
    host_fft_complex accumulator(0.0, 0.0);
    for (int sample = 0; sample < extent; ++sample) {
      const double angle =
          sign * 2.0 * std::numbers::pi * (static_cast<double>(frequency) * sample / extent);
      accumulator += values[static_cast<std::size_t>(sample)] *
                     host_fft_complex(std::cos(angle), std::sin(angle));
    }
    output[static_cast<std::size_t>(frequency)] =
        inverse ? accumulator / static_cast<double>(extent) : accumulator;
  }
  for (int index = 0; index < extent; ++index)
    values[static_cast<std::size_t>(index)] = output[static_cast<std::size_t>(index)];
}

inline void host_fft1d(std::span<host_fft_complex> values, bool inverse) {
  const int extent = static_cast<int>(values.size());
  if (!host_fft_is_power_of_two(extent)) {
    record_fallback(FallbackCounter::kFftDirectDft);
    host_direct_dft(values, inverse);
    return;
  }
  for (int index = 1, reversed = 0; index < extent; ++index) {
    int bit = extent >> 1;
    for (; reversed & bit; bit >>= 1)
      reversed ^= bit;
    reversed ^= bit;
    if (index < reversed)
      std::swap(values[static_cast<std::size_t>(index)],
                values[static_cast<std::size_t>(reversed)]);
  }
  for (int length = 2; length <= extent; length <<= 1) {
    const double angle = 2.0 * std::numbers::pi / length * (inverse ? 1.0 : -1.0);
    const host_fft_complex root(std::cos(angle), std::sin(angle));
    for (int offset = 0; offset < extent; offset += length) {
      host_fft_complex weight(1.0, 0.0);
      for (int index = 0; index < length / 2; ++index) {
        const host_fft_complex even = values[static_cast<std::size_t>(offset + index)];
        const host_fft_complex odd =
            values[static_cast<std::size_t>(offset + index + length / 2)] * weight;
        values[static_cast<std::size_t>(offset + index)] = even + odd;
        values[static_cast<std::size_t>(offset + index + length / 2)] = even - odd;
        weight *= root;
      }
    }
  }
  if (inverse)
    for (auto& value : values)
      value /= extent;
}

inline void host_fft1d(std::vector<host_fft_complex>& values, bool inverse) {
  host_fft1d(std::span<host_fft_complex>(values), inverse);
}

}  // namespace pops::elliptic::poisson::internal
