#pragma once

// Kernel launch entry points have public, named enclosing functions so NVCC can
// form extended-lambda identities. Views/scalars are explicit snapshots of the
// private FFT plan; orchestration, allocation, fencing and MPI stay in that plan.
#include <pops/core/foundation/types.hpp>
#include <Kokkos_Array.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>
#include <array>
#include <cstddef>
#include <numbers>

namespace pops::detail {

template <int Dim, class MemorySpace>
struct PoissonFFTDeviceKernels {
  using int_array = std::array<int, Dim>;
  using real_array = std::array<double, Dim>;
  using complex_type = Kokkos::complex<double>;
  using device_view = Kokkos::View<complex_type*, MemorySpace>;

  static POPS_HD int reverse_bits_(int value, int extent) {
    int reversed = 0;
    for (int remaining = extent; remaining > 1; remaining >>= 1) {
      reversed = (reversed << 1) | (value & 1);
      value >>= 1;
    }
    return reversed;
  }

  static void local_dft(const device_view& input, const device_view& output, std::size_t stride,
                        int extent, bool inverse, double sign, std::size_t lines) {
    Kokkos::parallel_for(
        "pops_poisson_fft_local_dft", Kokkos::RangePolicy<>(0, lines),
        KOKKOS_LAMBDA(std::size_t line) {
          const std::size_t block = line / stride;
          const std::size_t offset = line % stride;
          const std::size_t base = block * stride * static_cast<std::size_t>(extent) + offset;
          for (int frequency = 0; frequency < extent; ++frequency) {
            complex_type sum(0.0, 0.0);
            for (int point = 0; point < extent; ++point) {
              const double angle = sign * 2.0 * std::numbers::pi * frequency * point / extent;
              sum += input[base + stride * static_cast<std::size_t>(point)] *
                     complex_type(Kokkos::cos(angle), Kokkos::sin(angle));
            }
            output[base + stride * static_cast<std::size_t>(frequency)] =
                inverse ? sum / static_cast<double>(extent) : sum;
          }
        });
  }

  static void bit_reverse(const device_view& bit_reversed_input,
                          const device_view& bit_reversed_output, std::size_t stride, int extent,
                          std::size_t local_count) {
    Kokkos::parallel_for(
        "pops_poisson_fft_bit_reverse", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(std::size_t ordinal) {
          const int coordinate = static_cast<int>((ordinal / stride) % extent);
          int reversed = 0;
          for (int value = coordinate, bit = extent >> 1; bit != 0; bit >>= 1) {
            reversed = (reversed << 1) | (value & 1);
            value >>= 1;
          }
          const std::ptrdiff_t delta = static_cast<std::ptrdiff_t>(reversed - coordinate) *
                                       static_cast<std::ptrdiff_t>(stride);
          bit_reversed_output[ordinal] = bit_reversed_input[ordinal + delta];
        });
  }

  static void local_radix_stage(const device_view& input, const device_view& output,
                                std::size_t stride, int extent, int length, int half, double sign,
                                std::size_t local_count) {
    Kokkos::parallel_for(
        "pops_poisson_fft_radix2_stage", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(std::size_t ordinal) {
          const int coordinate = static_cast<int>((ordinal / stride) % extent);
          const int group_begin = coordinate - coordinate % length;
          const int position = coordinate - group_begin;
          const int butterfly = position % half;
          const std::size_t top =
              ordinal + static_cast<std::ptrdiff_t>(group_begin + butterfly - coordinate) *
                            static_cast<std::ptrdiff_t>(stride);
          const std::size_t bottom = top + static_cast<std::size_t>(half) * stride;
          const double angle = sign * 2.0 * std::numbers::pi * butterfly / length;
          const complex_type even = input[top];
          const complex_type odd =
              input[bottom] * complex_type(Kokkos::cos(angle), Kokkos::sin(angle));
          output[ordinal] = position < half ? even + odd : even - odd;
        });
  }

  static void local_radix_normalize(const device_view& normalized, int extent,
                                    std::size_t local_count) {
    Kokkos::parallel_for(
        "pops_poisson_fft_radix2_normalize", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(std::size_t ordinal) { normalized[ordinal] /= static_cast<double>(extent); });
  }

  static void peer_dft(const device_view& source, const device_view& send, std::size_t transverse,
                       int local_last, int global_last, int rank, int destination, double sign,
                       std::size_t local_count) {
    Kokkos::parallel_for(
        "pops_poisson_fft_peer_dft", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(std::size_t ordinal) {
          const std::size_t transverse_index = ordinal % transverse;
          const int output_local = static_cast<int>(ordinal / transverse);
          const int output_global = destination * local_last + output_local;
          complex_type sum(0.0, 0.0);
          for (int input_local = 0; input_local < local_last; ++input_local) {
            const int input_global = rank * local_last + input_local;
            const double angle =
                sign * 2.0 * std::numbers::pi * output_global * input_global / global_last;
            sum += source[transverse_index + transverse * static_cast<std::size_t>(input_local)] *
                   complex_type(Kokkos::cos(angle), Kokkos::sin(angle));
          }
          send[ordinal] = sum;
        });
  }

  static void peer_accumulate(const device_view& result, const device_view& receive,
                              std::size_t local_count) {
    Kokkos::parallel_for(
        "pops_poisson_fft_peer_accumulate", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(std::size_t ordinal) { result[ordinal] += receive[ordinal]; });
  }

  static void inverse_normalize(const device_view& normalized, int global_last,
                                std::size_t local_count) {
    Kokkos::parallel_for(
        "pops_poisson_fft_inverse_normalize", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(std::size_t ordinal) {
          normalized[ordinal] /= static_cast<double>(global_last);
        });
  }

  static void last_inverse_normalize(const device_view& normalized, int global_last,
                                     std::size_t local_count) {
    Kokkos::parallel_for(
        "pops_poisson_fft_last_inverse_normalize", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(std::size_t ordinal) {
          normalized[ordinal] /= static_cast<double>(global_last);
        });
  }

  static void last_local_radix(const device_view& input, const device_view& output,
                               std::size_t transverse, int length, int half, double sign,
                               bool inverse, std::size_t local_count) {
    Kokkos::parallel_for(
        "pops_poisson_fft_last_local_radix", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(std::size_t ordinal) {
          const int coordinate = static_cast<int>(ordinal / transverse);
          const int group_begin = coordinate - coordinate % length;
          const int position = coordinate - group_begin;
          const int butterfly = position % half;
          const std::size_t top =
              static_cast<std::size_t>(group_begin + butterfly) * transverse + ordinal % transverse;
          const std::size_t bottom = top + static_cast<std::size_t>(half) * transverse;
          const double angle = sign * 2.0 * std::numbers::pi * butterfly / length;
          const complex_type even = input[top];
          const complex_type odd =
              input[bottom] * complex_type(Kokkos::cos(angle), Kokkos::sin(angle));
          if (inverse)
            output[ordinal] = position < half ? even + odd : even - odd;
          else
            output[ordinal] = position < half
                                  ? input[top] + input[bottom]
                                  : (input[top] - input[bottom]) *
                                        complex_type(Kokkos::cos(angle), Kokkos::sin(angle));
        });
  }

  static void last_distributed_radix(const device_view& local, const device_view& remote,
                                     std::size_t transverse, int local_last, int rank,
                                     int rank_offset, int half, int length, double sign,
                                     bool inverse, std::size_t local_count) {
    Kokkos::parallel_for(
        "pops_poisson_fft_last_distributed_radix", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(std::size_t ordinal) {
          const int coordinate = static_cast<int>(ordinal / transverse);
          const int global = rank * local_last + coordinate;
          const int within_half = global % half;
          const bool lower = (rank & rank_offset) == 0;
          const double angle = sign * 2.0 * std::numbers::pi * within_half / length;
          if (inverse) {
            const complex_type lower_value = lower ? local[ordinal] : remote[ordinal];
            const complex_type upper_value = lower ? remote[ordinal] : local[ordinal];
            const complex_type weighted_upper =
                upper_value * complex_type(Kokkos::cos(angle), Kokkos::sin(angle));
            local[ordinal] = lower ? lower_value + weighted_upper : lower_value - weighted_upper;
          } else {
            const complex_type lower_value = lower ? local[ordinal] : remote[ordinal];
            const complex_type upper_value = lower ? remote[ordinal] : local[ordinal];
            local[ordinal] = lower ? lower_value + upper_value
                                   : (lower_value - upper_value) *
                                         complex_type(Kokkos::cos(angle), Kokkos::sin(angle));
          }
        });
  }

  static void inverse_symbol(const device_view& values, const int_array& cells,
                             const real_array& spacing, std::size_t transverse, int local_last,
                             int rank, bool last_axis_bit_reversed, std::size_t local_count) {
    // std::array accessors are host-only in the supported NVCC toolchain. Keep
    // the host plan interface, and capture an owning device-callable snapshot.
    Kokkos::Array<int, Dim> device_cells{};
    Kokkos::Array<double, Dim> device_spacing{};
    for (int axis = 0; axis < Dim; ++axis) {
      device_cells[axis] = cells[axis];
      device_spacing[axis] = spacing[axis];
    }
    Kokkos::parallel_for(
        "pops_poisson_fft_symbol", Kokkos::RangePolicy<>(0, local_count),
        KOKKOS_LAMBDA(std::size_t ordinal) {
          std::size_t cursor = ordinal % transverse;
          double lambda = 0.0;
          for (int axis = 0; axis < Dim - 1; ++axis) {
            const int frequency = static_cast<int>(cursor % device_cells[axis]);
            cursor /= static_cast<std::size_t>(device_cells[axis]);
            lambda +=
                (2.0 * Kokkos::cos(2.0 * std::numbers::pi * frequency / device_cells[axis]) - 2.0) /
                (device_spacing[axis] * device_spacing[axis]);
          }
          const int stored_frequency = rank * local_last + static_cast<int>(ordinal / transverse);
          const int frequency = last_axis_bit_reversed
                                    ? reverse_bits_(stored_frequency, device_cells[Dim - 1])
                                    : stored_frequency;
          lambda +=
              (2.0 * Kokkos::cos(2.0 * std::numbers::pi * frequency / device_cells[Dim - 1]) - 2.0) /
              (device_spacing[Dim - 1] * device_spacing[Dim - 1]);
          values[ordinal] =
              Kokkos::abs(lambda) < 1e-14 ? complex_type(0.0, 0.0) : values[ordinal] / lambda;
        });
  }
};

}  // namespace pops::detail
