/// @file
/// @brief Transactional exact-clock linear interpolation between two qualified AMR states.

#pragma once

#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/numerics/time/amr/levels/amr_clock.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace pops::amr::transfer {

/// Host-side identity carried by every retained state and by the requested candidate image.
struct QualifiedTemporalState {
  std::string state_identity;
  std::string spatial_contract;
  std::uint64_t topology_generation = 0;
  std::uint64_t materialization_generation = 0;
  ClockStamp clock{};

  friend bool operator==(const QualifiedTemporalState&, const QualifiedTemporalState&) = default;
};

struct TemporalComponentRange {
  int older_begin = 0;
  int newer_begin = 0;
  int destination_begin = 0;
  int count = 1;

  constexpr bool operator==(const TemporalComponentRange&) const = default;
};

template <int Dim>
class LinearTemporalInterpolationProvider;

template <int Dim>
class PreparedLinearTemporalInterpolation {
 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedLinearTemporalInterpolation supports dimensions 1, 2, and 3");

  POPS_HD void operator()(const Index<Dim>& index) const {
    for (int component = 0; component < components_.count; ++component) {
      const Real older = older_(index, components_.older_begin + component);
      const Real newer = newer_(index, components_.newer_begin + component);
      destination_(index, components_.destination_begin + component) =
          older + alpha_ * (newer - older);
    }
  }

  POPS_HD const Box<Dim>& destination_region() const { return destination_region_; }
  POPS_HD Real alpha() const { return alpha_; }
  POPS_HD const RefinementRatio<Dim>& refinement_ratio() const { return ratio_; }
  POPS_HD TemporalComponentRange components() const { return components_; }

 private:
  template <int>
  friend class LinearTemporalInterpolationProvider;

  POPS_HD PreparedLinearTemporalInterpolation(FieldView<const Real, Dim> older,
                                              FieldView<const Real, Dim> newer,
                                              FieldView<Real, Dim> destination,
                                              Box<Dim> destination_region,
                                              RefinementRatio<Dim> ratio,
                                              TemporalComponentRange components, Real alpha)
      : older_(older),
        newer_(newer),
        destination_(destination),
        destination_region_(destination_region),
        ratio_(ratio),
        components_(components),
        alpha_(alpha) {}

  FieldView<const Real, Dim> older_{};
  FieldView<const Real, Dim> newer_{};
  FieldView<Real, Dim> destination_{};
  Box<Dim> destination_region_{};
  RefinementRatio<Dim> ratio_{};
  TemporalComponentRange components_{};
  Real alpha_ = Real(0);
};

template <int Dim>
class LinearTemporalInterpolationProvider {
 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "LinearTemporalInterpolationProvider supports dimensions 1, 2, and 3");

  static constexpr TransferCapabilities capabilities() {
    return {2, 0, true, true, SlopeLimiter::None};
  }

  PreparedLinearTemporalInterpolation<Dim> prepare(
      FieldView<const Real, Dim> older, FieldView<const Real, Dim> newer,
      FieldView<Real, Dim> destination, const Box<Dim>& destination_region,
      RefinementRatio<Dim> transition_ratio, const QualifiedTemporalState& older_state,
      const QualifiedTemporalState& newer_state, const QualifiedTemporalState& target_state,
      TemporalComponentRange components = {}) const {
    validate_qualifications_(older_state, newer_state, target_state);
    if (!transition_ratio.refines_any_axis())
      throw std::invalid_argument(
          "prepared temporal interpolation requires a non-identity AMR transition ratio");
    const auto older_view = detail::validate_view(older);
    const auto newer_view = detail::validate_view(newer);
    const auto destination_view = detail::validate_view(destination);
    if (older_view.box != newer_view.box || destination_region.empty() ||
        !older_view.box.contains(destination_region) ||
        !destination_view.box.contains(destination_region))
      throw std::invalid_argument(
          "prepared temporal interpolation requires one common non-empty spatial image");
    if (components.older_begin < 0 || components.newer_begin < 0 ||
        components.destination_begin < 0 || components.count < 1 ||
        components.older_begin > older.ncomp - components.count ||
        components.newer_begin > newer.ncomp - components.count ||
        components.destination_begin > destination.ncomp - components.count)
      throw std::invalid_argument(
          "prepared temporal interpolation component interval is outside its states");
    if ((older_view.begin < destination_view.end && destination_view.begin < older_view.end) ||
        (newer_view.begin < destination_view.end && destination_view.begin < newer_view.end) ||
        (older_view.begin < newer_view.end && newer_view.begin < older_view.end))
      throw std::invalid_argument(
          "prepared temporal interpolation requires two immutable sources and candidate storage");

    const Rational exact_alpha =
        ClockWindow{older_state.clock, newer_state.clock}.alpha(target_state.clock);
    return PreparedLinearTemporalInterpolation<Dim>(older, newer, destination, destination_region,
                                                    transition_ratio, components,
                                                    static_cast<Real>(exact_alpha.value()));
  }

 private:
  static void validate_qualifications_(const QualifiedTemporalState& older,
                                       const QualifiedTemporalState& newer,
                                       const QualifiedTemporalState& target) {
    if (older.state_identity.empty() || older.spatial_contract.empty() ||
        older.state_identity != newer.state_identity ||
        older.state_identity != target.state_identity ||
        older.spatial_contract != newer.spatial_contract ||
        older.spatial_contract != target.spatial_contract ||
        older.topology_generation != newer.topology_generation ||
        older.topology_generation != target.topology_generation ||
        older.materialization_generation != newer.materialization_generation ||
        older.materialization_generation != target.materialization_generation)
      throw std::invalid_argument(
          "linear temporal interpolation states do not share one exact spatial authority");

    const double older_time = older.clock.physical_time;
    const double newer_time = newer.clock.physical_time;
    const double target_time = target.clock.physical_time;
    if (!std::isfinite(older_time) || !std::isfinite(newer_time) || !std::isfinite(target_time) ||
        !(older_time < newer_time))
      throw std::invalid_argument(
          "linear temporal interpolation requires finite increasing physical timestamps");
    const Rational exact_alpha = ClockWindow{older.clock, newer.clock}.alpha(target.clock);
    const double expected = older_time + exact_alpha.value() * (newer_time - older_time);
    double scale = 1.0 + std::abs(older_time) + std::abs(newer_time) + std::abs(target_time);
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() * scale;
    if (std::abs(expected - target_time) > tolerance)
      throw std::invalid_argument(
          "linear temporal interpolation physical time disagrees with its exact clock phase");
  }
};

}  // namespace pops::amr::transfer
