/// @file
/// @brief Fixed-topology 1D space-time geometry, not an ALE execution provider.
#pragma once

#include <pops/core/foundation/types.hpp>

#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace pops {

/// Positive-x oriented swept face measures over ONE time interval. In 1D these
/// are face displacements. They are integrated measures, not velocities/rates.
/// Coordinate endpoints and supplied sweeps are independently checked so stale
/// duration-dependent fluxes cannot silently determine the new cell measure.
class SweptInterval {
 public:
  static SweptInterval prepare(Real left_old, Real right_old, Real left_new, Real right_new,
                               Real left_swept, Real right_swept, Real tolerance) {
    for (Real value : {left_old, right_old, left_new, right_new,
                       left_swept, right_swept, tolerance})
      if (!std::isfinite(value))
        throw std::invalid_argument("swept interval requires finite coordinates and measures");
    if (tolerance < Real(0) || !(right_old > left_old) || !(right_new > left_new))
      throw std::invalid_argument("swept interval requires positive endpoint cell measures");
    const Real old_volume = right_old - left_old;
    const Real new_volume = right_new - left_new;
    if (!std::isfinite(old_volume) || !std::isfinite(new_volume))
      throw std::invalid_argument("swept interval cell measure overflow");
    if (std::abs((left_new - left_old) - left_swept) > tolerance ||
        std::abs((right_new - right_old) - right_swept) > tolerance ||
        std::abs((new_volume - old_volume) - (right_swept - left_swept)) > tolerance)
      throw std::invalid_argument("swept interval endpoint displacement/GCL mismatch");
    return SweptInterval(old_volume, new_volume, left_swept, right_swept);
  }

  POPS_HD Real old_measure() const noexcept { return old_measure_; }
  POPS_HD Real new_measure() const noexcept { return new_measure_; }
  POPS_HD Real left_swept() const noexcept { return left_swept_; }
  POPS_HD Real right_swept() const noexcept { return right_swept_; }
  POPS_HD Real gcl_residual() const noexcept {
    return (new_measure_ - old_measure_) - (right_swept_ - left_swept_);
  }

  /// Reynolds balance Q+ = Qn - (physical_right - physical_left)
  ///                      + (U_right*swept_right - U_left*swept_left) + source.
  /// Every flux and source argument is already integrated over this interval.
  /// The reconstruction/physical flux/source projection belongs to its caller.
  POPS_HD Real updated_amount(Real old_amount, Real density_left, Real density_right,
                               Real physical_left, Real physical_right,
                               Real source_amount) const noexcept {
    return integrated_amount_update(old_amount, density_left, density_right,
                                    physical_left, physical_right, source_amount,
                                    left_swept_, right_swept_);
  }

  /// Device algebra used after the owning provider validated endpoint geometry.
  POPS_HD static Real integrated_amount_update(
      Real old_amount, Real density_left, Real density_right, Real physical_left,
      Real physical_right, Real source_amount, Real left_swept, Real right_swept) noexcept {
    return old_amount - (physical_right - physical_left) +
           density_right * right_swept - density_left * left_swept + source_amount;
  }

 private:
  POPS_HD SweptInterval(Real old_measure, Real new_measure, Real left_swept, Real right_swept)
      : old_measure_(old_measure), new_measure_(new_measure), left_swept_(left_swept),
        right_swept_(right_swept) {}
  Real old_measure_, new_measure_, left_swept_, right_swept_;
};

static_assert(std::is_trivially_copyable_v<SweptInterval>);
}  // namespace pops
