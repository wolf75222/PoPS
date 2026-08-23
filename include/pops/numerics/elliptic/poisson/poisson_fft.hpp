#pragma once

/// @file
/// @brief Device-resident Cartesian periodic FFT Poisson transform.
///
/// A single `PoissonFFT<Dim>` plan owns the same execution trace for Dim=1,2,3:
/// batched device transforms over every locally complete axis, a distributed transform over the
/// final slab axis, a discrete Cartesian symbol that is the sum of one cosine eigenvalue per axis,
/// then the reverse trace. Dim=3 is that same product over X, Y and Z; it is not a 2-D FFT with a
/// replicated Z loop. When FFTW3 was discovered in CONDA_PREFIX (or POPS_FFTW_ROOT) at configure,
/// locally complete axes use that production radix backend for every positive extent. Otherwise the
/// in-tree radix-2 stages remain, and other extents use the explicitly diagnosed direct-DFT path.
/// The distributed stage is slab-on-the-final-axis, not a closed pencil-decomposed production 3-D
/// MPI FFT. FFT+AMR stays refused.
/// No rank materializes another rank's slab and MPI sees only pinned staging buffers.

#include <pops/diagnostics/fallback_diagnostics.hpp>
#include <pops/core/foundation/types.hpp>
#include <pops/core/foundation/kokkos_env.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <Kokkos_Complex.hpp>
#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#ifdef POPS_HAS_FFTW
#include <fftw3.h>
#endif

#ifdef POPS_HAS_MPI
#include <mpi.h>
#endif

namespace pops {

inline bool is_pow2(int n) {
  return n > 0 && (n & (n - 1)) == 0;
}
inline void reset_poisson_fft_direct_dft_fallback_count() {
  fallback_counter(FallbackCounter::kFftDirectDft).store(0, std::memory_order_relaxed);
}
inline std::size_t poisson_fft_direct_dft_fallback_count() {
  return fallback_count(FallbackCounter::kFftDirectDft);
}
inline void record_poisson_fft_direct_dft_fallback() {
  record_fallback(FallbackCounter::kFftDirectDft);
}

inline constexpr bool poisson_fft_fftw_configured() noexcept {
#ifdef POPS_HAS_FFTW
  return true;
#else
  return false;
#endif
}

inline const char* poisson_fft_fftw_absence_reason() noexcept {
#ifdef POPS_HAS_FFTW
  return "";
#else
  return "FFTW3 was not found in CONDA_PREFIX or POPS_FFTW_ROOT at configure; "
         "PoissonFFT uses the in-tree radix-2 / diagnosed direct-DFT backend";
#endif
}

/// Bounded diagnostics used by MPI tests to prove that a rank-local allocation or device-stage
/// failure is published before the next collective. The default context is inert and is part of
/// the exact plan contract, so production ranks cannot accidentally select different diagnostics.
enum class PoissonFFTDiagnosticStage : std::uint8_t {
  none,
  workspace_allocation,
  peer_dft_launch,
  peer_accumulation,
  distributed_radix,
};

struct PoissonFFTDiagnosticContext {
  PoissonFFTDiagnosticStage stage = PoissonFFTDiagnosticStage::none;
  int fail_rank = -1;
};

template <int Dim, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class PoissonFFT {
 public:
  static_assert(Dim >= 1 && Dim <= 3);
  static constexpr int dimension = Dim;
  using int_array = std::array<int, Dim>;
  using real_array = std::array<double, Dim>;
  using complex_type = Kokkos::complex<double>;
  using device_view = Kokkos::View<complex_type*, MemorySpace>;
  using pinned_view = Kokkos::View<complex_type*, Kokkos::SharedHostPinnedSpace>;

  PoissonFFT(int_array cells, real_array lengths, const ExecutionLane& lane,
             std::string_view lane_identity, PoissonFFTDiagnosticContext diagnostic = {})
      : cells_(cells),
        lengths_(lengths),
        lane_(&lane),
        lane_borrow_(lane.borrow_immutably()),
        diagnostic_(diagnostic) {
    ranks_ = lane.size();
    rank_ = lane.rank();
    std::exception_ptr preparation_error;
    std::string request_contract;
    try {
      detail::ensure_kokkos_initialized();
      if (lane_identity.empty())
        throw std::invalid_argument("PoissonFFT requires a stable execution-lane identity");
      if (ranks_ < 1 || rank_ < 0 || rank_ >= ranks_)
        throw std::invalid_argument("PoissonFFT requires a valid execution-lane rank space");
      if ((diagnostic_.stage == PoissonFFTDiagnosticStage::none && diagnostic_.fail_rank != -1) ||
          (diagnostic_.stage != PoissonFFTDiagnosticStage::none &&
           (diagnostic_.fail_rank < 0 || diagnostic_.fail_rank >= ranks_)))
        throw std::invalid_argument("PoissonFFT diagnostic context is not rank-canonical");
      transverse_ = 1;
      for (int axis = 0; axis < Dim; ++axis) {
        if (cells_[axis] <= 0 || !std::isfinite(lengths_[axis]) || lengths_[axis] <= 0.0)
          throw std::invalid_argument("PoissonFFT requires positive Cartesian extents and lengths");
        spacing_[axis] = lengths_[axis] / static_cast<double>(cells_[axis]);
        if (axis < Dim - 1)
          transverse_ = checked_product_(transverse_, static_cast<std::size_t>(cells_[axis]));
      }
      if (cells_[Dim - 1] % ranks_ != 0)
        throw std::invalid_argument(
            "PoissonFFT communicator size must divide the final Cartesian axis "
            "(unique slabs, no replicated Z)");
      local_last_ = cells_[Dim - 1] / ranks_;
      local_count_ = checked_product_(transverse_, static_cast<std::size_t>(local_last_));
      if (local_count_ >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) / sizeof(complex_type))
        throw std::length_error("PoissonFFT local transfer exceeds MPI byte count range");

      ExactContractBuilder contract;
      contract.text("pops.poisson-fft.exact-request@3")
          .scalar(static_cast<std::int32_t>(Dim))
          .scalar(static_cast<std::int32_t>(ranks_))
          .sequence(cells_)
          .sequence(lengths_)
          .text(lane.identity())
          .text(lane_identity)
          .scalar(static_cast<std::uint8_t>(diagnostic_.stage))
          .scalar(static_cast<std::int32_t>(diagnostic_.fail_rank));
      request_contract = std::move(contract).release();
    } catch (...) {
      preparation_error = std::current_exception();
    }
    const long preparation_failures = all_reduce_sum(preparation_error ? 1L : 0L, lane);
    if (preparation_failures != 0) {
      if (ranks_ == 1 && preparation_error)
        std::rethrow_exception(preparation_error);
      throw std::runtime_error("PoissonFFT request preparation failed collectively on " +
                               std::to_string(preparation_failures) + " rank(s)");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"pops.poisson-fft.exact-request@3", request_contract}}, lane))
      throw std::invalid_argument("PoissonFFT request differs across communicator ranks");

    std::exception_ptr allocation_error;
    try {
      if (diagnostic_.stage == PoissonFFTDiagnosticStage::workspace_allocation &&
          diagnostic_.fail_rank == rank_)
        throw std::bad_alloc();
      values_ = device_view("poisson_fft_values", local_count_);
      scratch_ = device_view("poisson_fft_scratch", local_count_);
      peer_send_ = device_view("poisson_fft_peer_send", local_count_);
      peer_receive_ = device_view("poisson_fft_peer_receive", local_count_);
      host_send_ = pinned_view("poisson_fft_host_send", local_count_);
      host_receive_ = pinned_view("poisson_fft_host_receive", local_count_);
    } catch (...) {
      allocation_error = std::current_exception();
    }
    if (all_reduce_max(allocation_error ? 1L : 0L, lane) != 0) {
      if (ranks_ == 1 && allocation_error)
        std::rethrow_exception(allocation_error);
      throw std::runtime_error("PoissonFFT workspace allocation failed collectively");
    }
  }

  PoissonFFT(const PoissonFFT&) = delete;
  PoissonFFT& operator=(const PoissonFFT&) = delete;
  PoissonFFT(PoissonFFT&&) noexcept = default;
  PoissonFFT& operator=(PoissonFFT&&) noexcept = default;

  [[nodiscard]] const int_array& cells() const noexcept { return cells_; }
  [[nodiscard]] int local_last_extent() const noexcept { return local_last_; }
  [[nodiscard]] int local_last_begin() const noexcept { return rank_ * local_last_; }
  [[nodiscard]] std::size_t local_cell_count() const noexcept { return local_count_; }
  [[nodiscard]] const ExecutionLane& lane() const noexcept { return *lane_; }
  [[nodiscard]] bool uses_direct_dft_fallback() const noexcept {
    if (poisson_fft_fftw_configured()) {
      if (ranks_ == 1)
        return false;
      return !is_pow2(cells_[Dim - 1]);
    }
    for (const int extent : cells_)
      if (!is_pow2(extent))
        return true;
    return false;
  }
  [[nodiscard]] bool uses_fftw_backend() const noexcept {
    if (!poisson_fft_fftw_configured())
      return false;
    return ranks_ == 1 || Dim > 1;
  }

  /// Solve lap_h phi = rhs.  Both views are local final-axis slabs and remain device-resident.
  void solve(const device_view& rhs, const device_view& phi) {
    // A rank-local view mismatch must never let a strict subset of ranks enter the exchange.
    // Turn it into one lane-collective failure before the first MPI operation on `lane_`.
    const long view_mismatch =
        rhs.extent(0) != local_count_ || phi.extent(0) != local_count_ ? 1L : 0L;
    if (all_reduce_max(view_mismatch, *lane_) != 0)
      throw std::invalid_argument("PoissonFFT device views do not match the local slab");
    execute_local_stage_collectively_(
        [&] {
          Kokkos::deep_copy(values_, rhs);
          for (int axis = 0; axis < Dim - 1; ++axis)
            local_dft_axis_(axis, false);
        },
        "forward local-axis transform");
    distributed_last_dft_(false);
    execute_local_stage_collectively_([&] { apply_discrete_inverse_symbol_(); },
                                      "discrete inverse symbol");
    distributed_last_dft_(true);
    execute_local_stage_collectively_(
        [&] {
          for (int axis = Dim - 2; axis >= 0; --axis)
            local_dft_axis_(axis, true);
          Kokkos::deep_copy(phi, values_);
        },
        "inverse local-axis transform");
  }

 private:
  static std::size_t checked_product_(std::size_t left, std::size_t right) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
      throw std::length_error("PoissonFFT allocation overflow");
    return left * right;
  }

  template <class Operation>
  void execute_local_stage_collectively_(
      Operation&& operation, std::string_view label,
      PoissonFFTDiagnosticStage diagnostic_stage = PoissonFFTDiagnosticStage::none) {
    std::exception_ptr local_error;
    try {
      if (diagnostic_.stage == diagnostic_stage && diagnostic_.fail_rank == rank_)
        throw std::runtime_error("injected rank-local PoissonFFT device-stage failure");
      std::forward<Operation>(operation)();
      Kokkos::fence();
    } catch (...) {
      local_error = std::current_exception();
    }
    if (all_reduce_max(local_error ? 1L : 0L, lane_->communicator()) != 0) {
      if (ranks_ == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("PoissonFFT " + std::string(label) + " failed collectively");
    }
  }

  bool try_local_fftw_axis_(int axis, bool inverse) {
#ifdef POPS_HAS_FFTW
    const int extent = cells_[axis];
    if (extent <= 1)
      return extent == 1;
    Kokkos::fence();
    auto host = Kokkos::create_mirror_view(values_);
    Kokkos::deep_copy(host, values_);
    int shape[3]{};
    for (int lower = 0; lower < Dim - 1; ++lower)
      shape[lower] = cells_[lower];
    shape[Dim - 1] = local_last_;
    fftw_iodim dims{};
    fftw_iodim howmany[3]{};
    int howmany_rank = 0;
    int stride = 1;
    for (int current = 0; current < Dim; ++current) {
      if (current == axis) {
        dims.n = shape[current];
        dims.is = dims.os = stride;
      } else {
        howmany[howmany_rank].n = shape[current];
        howmany[howmany_rank].is = howmany[howmany_rank].os = stride;
        ++howmany_rank;
      }
      stride *= shape[current];
    }
    fftw_complex* data = reinterpret_cast<fftw_complex*>(host.data());
    const int sign = inverse ? FFTW_BACKWARD : FFTW_FORWARD;
    fftw_plan plan = fftw_plan_guru_dft(1, &dims, howmany_rank, howmany_rank == 0 ? nullptr : howmany,
                                        data, data, sign, FFTW_ESTIMATE);
    if (plan == nullptr)
      return false;
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    if (inverse) {
      const double scale = 1.0 / static_cast<double>(extent);
      for (std::size_t ordinal = 0; ordinal < local_count_; ++ordinal)
        host(ordinal) *= scale;
    }
    Kokkos::deep_copy(values_, host);
    return true;
#else
    (void)axis;
    (void)inverse;
    return false;
#endif
  }

 public:
  // NVCC requires the lexical parent of a KOKKOS_LAMBDA (__host__ __device__) to be public.
  // Keep every launch helper below public so the CUDA frontend can instantiate the same device
  // kernels as the Serial and OpenMP backends.  These underscore-suffixed implementation helpers
  // are not part of the supported solver interface; only their access changes here.  The plan
  // state remains private below.
  void local_dft_axis_(int axis, bool inverse) {
    if (try_local_fftw_axis_(axis, inverse))
      return;
    if (is_pow2(cells_[axis])) {
      local_radix2_axis_(axis, inverse);
      return;
    }
    record_poisson_fft_direct_dft_fallback();
    std::size_t stride = 1;
    for (int lower = 0; lower < axis; ++lower)
      stride *= static_cast<std::size_t>(cells_[lower]);
    const int extent = cells_[axis];
    const std::size_t lines = local_count_ / static_cast<std::size_t>(extent);
    const auto input = values_;
    const auto output = scratch_;
    const double sign = inverse ? 1.0 : -1.0;
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
    Kokkos::deep_copy(values_, scratch_);
  }

  void local_radix2_axis_(int axis, bool inverse) {
    std::size_t stride = 1;
    for (int lower = 0; lower < axis; ++lower)
      stride *= static_cast<std::size_t>(cells_[lower]);
    const int extent = cells_[axis];
    if (extent == 1)
      return;
    const auto bit_reversed_input = values_;
    const auto bit_reversed_output = scratch_;
    Kokkos::parallel_for(
        "pops_poisson_fft_bit_reverse", Kokkos::RangePolicy<>(0, local_count_),
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
    std::swap(values_, scratch_);
    for (int length = 2;; length <<= 1) {
      const auto input = values_;
      const auto output = scratch_;
      const int half = length / 2;
      const double sign = inverse ? 1.0 : -1.0;
      Kokkos::parallel_for(
          "pops_poisson_fft_radix2_stage", Kokkos::RangePolicy<>(0, local_count_),
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
      std::swap(values_, scratch_);
      if (length == extent)
        break;
    }
    if (inverse) {
      const auto normalized = values_;
      Kokkos::parallel_for(
          "pops_poisson_fft_radix2_normalize", Kokkos::RangePolicy<>(0, local_count_),
          KOKKOS_LAMBDA(std::size_t ordinal) {
            normalized[ordinal] /= static_cast<double>(extent);
          });
    }
  }

  void distributed_last_dft_(bool inverse) {
    if (ranks_ == 1) {
      local_dft_axis_(Dim - 1, inverse);
      return;
    }
    if (is_pow2(cells_[Dim - 1])) {
      distributed_last_radix2_(inverse);
      return;
    }
    record_poisson_fft_direct_dft_fallback();
    execute_local_stage_collectively_(
        [&] {
          Kokkos::deep_copy(scratch_, values_);
          Kokkos::deep_copy(values_, complex_type(0.0, 0.0));
        },
        "direct-DFT initialization");
    const auto source = scratch_;
    const auto send = peer_send_;
    const auto receive = peer_receive_;
    const auto result = values_;
    const std::size_t transverse = transverse_;
    const int local_last = local_last_;
    const int global_last = cells_[Dim - 1];
    const int rank = rank_;
    const double sign = inverse ? 1.0 : -1.0;
    for (int phase = 0; phase < ranks_; ++phase) {
      // Cyclic all-to-all schedule: every phase has one distinct destination and
      // one distinct source, so no rank waits for a peer that is staging itself.
      const int destination = (rank_ + phase) % ranks_;
      const int source_rank = (rank_ - phase + ranks_) % ranks_;
      execute_local_stage_collectively_(
          [&] {
            Kokkos::parallel_for(
                "pops_poisson_fft_peer_dft", Kokkos::RangePolicy<>(0, local_count_),
                KOKKOS_LAMBDA(std::size_t ordinal) {
                  const std::size_t transverse_index = ordinal % transverse;
                  const int output_local = static_cast<int>(ordinal / transverse);
                  const int output_global = destination * local_last + output_local;
                  complex_type sum(0.0, 0.0);
                  for (int input_local = 0; input_local < local_last; ++input_local) {
                    const int input_global = rank * local_last + input_local;
                    const double angle =
                        sign * 2.0 * std::numbers::pi * output_global * input_global / global_last;
                    sum += source[transverse_index +
                                  transverse * static_cast<std::size_t>(input_local)] *
                           complex_type(Kokkos::cos(angle), Kokkos::sin(angle));
                  }
                  send[ordinal] = sum;
                });
          },
          "peer-DFT launch", PoissonFFTDiagnosticStage::peer_dft_launch);
      exchange_peer_(destination, source_rank);
      execute_local_stage_collectively_(
          [&] {
            Kokkos::parallel_for(
                "pops_poisson_fft_peer_accumulate", Kokkos::RangePolicy<>(0, local_count_),
                KOKKOS_LAMBDA(std::size_t ordinal) { result[ordinal] += receive[ordinal]; });
          },
          "peer accumulation", PoissonFFTDiagnosticStage::peer_accumulation);
    }
    execute_local_stage_collectively_(
        [&] {
          if (inverse) {
            const auto normalized = values_;
            Kokkos::parallel_for(
                "pops_poisson_fft_inverse_normalize", Kokkos::RangePolicy<>(0, local_count_),
                KOKKOS_LAMBDA(std::size_t ordinal) {
                  normalized[ordinal] /= static_cast<double>(global_last);
                });
          }
        },
        "direct-DFT completion");
  }

  static POPS_HD int reverse_bits_(int value, int extent) {
    int reversed = 0;
    for (int remaining = extent; remaining > 1; remaining >>= 1) {
      reversed = (reversed << 1) | (value & 1);
      value >>= 1;
    }
    return reversed;
  }

  void distributed_last_radix2_(bool inverse) {
    const int global_last = cells_[Dim - 1];
    const int local_last = local_last_;
    if (!inverse) {
      // DIF: natural samples become bit-reversed Fourier frequencies.
      for (int length = global_last; length > local_last; length >>= 1)
        distributed_radix_stage_(length, false);
      execute_local_stage_collectively_(
          [&] {
            for (int length = local_last; length >= 2; length >>= 1)
              local_last_radix_stage_(length, false);
          },
          "forward local radix tail");
      return;
    }
    // DIT: bit-reversed Fourier frequencies return to natural samples.
    execute_local_stage_collectively_(
        [&] {
          if (local_last > 1)
            for (int length = 2;; length <<= 1) {
              local_last_radix_stage_(length, true);
              if (length == local_last)
                break;
            }
        },
        "inverse local radix prefix");
    for (int length = local_last; length < global_last;) {
      length <<= 1;
      distributed_radix_stage_(length, true);
    }
    execute_local_stage_collectively_(
        [&] {
          const auto normalized = values_;
          Kokkos::parallel_for(
              "pops_poisson_fft_last_inverse_normalize", Kokkos::RangePolicy<>(0, local_count_),
              KOKKOS_LAMBDA(std::size_t ordinal) {
                normalized[ordinal] /= static_cast<double>(global_last);
              });
        },
        "inverse distributed radix normalization");
  }

  void local_last_radix_stage_(int length, bool inverse) {
    const auto input = values_;
    const auto output = scratch_;
    const std::size_t transverse = transverse_;
    const int half = length / 2;
    const double sign = inverse ? 1.0 : -1.0;
    Kokkos::parallel_for(
        "pops_poisson_fft_last_local_radix", Kokkos::RangePolicy<>(0, local_count_),
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
    std::swap(values_, scratch_);
  }

  void distributed_radix_stage_(int length, bool inverse) {
    const int half = length / 2;
    const int rank_offset = half / local_last_;
    const int partner = rank_ ^ rank_offset;
    exchange_partner_(partner);
    const auto local = values_;
    const auto remote = peer_receive_;
    const std::size_t transverse = transverse_;
    const int local_last = local_last_;
    const int rank = rank_;
    const double sign = inverse ? 1.0 : -1.0;
    execute_local_stage_collectively_(
        [&] {
          Kokkos::parallel_for(
              "pops_poisson_fft_last_distributed_radix", Kokkos::RangePolicy<>(0, local_count_),
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
                  local[ordinal] =
                      lower ? lower_value + weighted_upper : lower_value - weighted_upper;
                } else {
                  const complex_type lower_value = lower ? local[ordinal] : remote[ordinal];
                  const complex_type upper_value = lower ? remote[ordinal] : local[ordinal];
                  local[ordinal] = lower ? lower_value + upper_value
                                         : (lower_value - upper_value) *
                                               complex_type(Kokkos::cos(angle), Kokkos::sin(angle));
                }
              });
        },
        "distributed radix stage", PoissonFFTDiagnosticStage::distributed_radix);
  }

  void exchange_peer_(int destination, int source_rank) {
    execute_local_stage_collectively_([&] { Kokkos::deep_copy(host_send_, peer_send_); },
                                      "peer-send staging");
    if (destination == rank_ && source_rank == rank_) {
      execute_local_stage_collectively_([&] { Kokkos::deep_copy(host_receive_, host_send_); },
                                        "local peer transfer");
    } else {
#ifdef POPS_HAS_MPI
      const std::size_t bytes = local_count_ * sizeof(complex_type);
      const int exchange_code = MPI_Sendrecv(
          host_send_.data(), static_cast<int>(bytes), MPI_BYTE, destination,
          ExecutionLane::translation_message_tag, host_receive_.data(), static_cast<int>(bytes),
          MPI_BYTE, source_rank, ExecutionLane::translation_message_tag, lane_->native_handle(),
          MPI_STATUS_IGNORE);
      if (all_reduce_max(exchange_code == MPI_SUCCESS ? 0L : 1L, lane_->communicator()) != 0)
        throw std::runtime_error("MPI_Sendrecv(PoissonFFT peer DFT) failed collectively");
#else
      throw std::logic_error("PoissonFFT peer exchange requires MPI support");
#endif
    }
    execute_local_stage_collectively_([&] { Kokkos::deep_copy(peer_receive_, host_receive_); },
                                      "peer-receive publication");
  }

  void exchange_partner_(int partner) {
    execute_local_stage_collectively_([&] { Kokkos::deep_copy(host_send_, values_); },
                                      "radix-send staging");
    if (partner == rank_) {
      execute_local_stage_collectively_([&] { Kokkos::deep_copy(host_receive_, host_send_); },
                                        "local radix transfer");
    } else {
#ifdef POPS_HAS_MPI
      const std::size_t bytes = local_count_ * sizeof(complex_type);
      const int exchange_code = MPI_Sendrecv(
          host_send_.data(), static_cast<int>(bytes), MPI_BYTE, partner,
          ExecutionLane::translation_message_tag, host_receive_.data(), static_cast<int>(bytes),
          MPI_BYTE, partner, ExecutionLane::translation_message_tag, lane_->native_handle(),
          MPI_STATUS_IGNORE);
      if (all_reduce_max(exchange_code == MPI_SUCCESS ? 0L : 1L, lane_->communicator()) != 0)
        throw std::runtime_error("MPI_Sendrecv(PoissonFFT radix stage) failed collectively");
#else
      throw std::logic_error("PoissonFFT radix exchange requires MPI support");
#endif
    }
    execute_local_stage_collectively_([&] { Kokkos::deep_copy(peer_receive_, host_receive_); },
                                      "radix-receive publication");
  }

  void apply_discrete_inverse_symbol_() {
    const auto values = values_;
    const auto cells = cells_;
    const auto spacing = spacing_;
    const std::size_t transverse = transverse_;
    const int local_last = local_last_;
    const int rank = rank_;
    const bool last_axis_bit_reversed = ranks_ > 1 && is_pow2(cells_[Dim - 1]);
    Kokkos::parallel_for(
        "pops_poisson_fft_symbol", Kokkos::RangePolicy<>(0, local_count_),
        KOKKOS_LAMBDA(std::size_t ordinal) {
          std::size_t cursor = ordinal % transverse;
          double lambda = 0.0;
          for (int axis = 0; axis < Dim - 1; ++axis) {
            const int frequency = static_cast<int>(cursor % cells[axis]);
            cursor /= static_cast<std::size_t>(cells[axis]);
            lambda += (2.0 * Kokkos::cos(2.0 * std::numbers::pi * frequency / cells[axis]) - 2.0) /
                      (spacing[axis] * spacing[axis]);
          }
          const int stored_frequency = rank * local_last + static_cast<int>(ordinal / transverse);
          const int frequency =
              last_axis_bit_reversed ? reverse_bits_(stored_frequency, cells[Dim - 1]) : stored_frequency;
          lambda += (2.0 * Kokkos::cos(2.0 * std::numbers::pi * frequency / cells[Dim - 1]) - 2.0) /
                    (spacing[Dim - 1] * spacing[Dim - 1]);
          values[ordinal] =
              Kokkos::abs(lambda) < 1e-14 ? complex_type(0.0, 0.0) : values[ordinal] / lambda;
        });
  }

 private:
  int_array cells_{};
  real_array lengths_{};
  real_array spacing_{};
  int ranks_ = 1;
  int rank_ = 0;
  int local_last_ = 0;
  std::size_t transverse_ = 0;
  std::size_t local_count_ = 0;
  const ExecutionLane* lane_ = nullptr;
  ExecutionLane::ImmutableBorrow lane_borrow_;
  device_view values_{};
  device_view scratch_{};
  device_view peer_send_{};
  device_view peer_receive_{};
  pinned_view host_send_{};
  pinned_view host_receive_{};
  PoissonFFTDiagnosticContext diagnostic_{};
};

}  // namespace pops
