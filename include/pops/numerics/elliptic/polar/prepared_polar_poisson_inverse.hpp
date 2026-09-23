#pragma once

/// @file
/// @brief Fixed, bounded FV disk-Poisson inverse for an explicitly selected preconditioner.

#include <pops/core/foundation/allocator.hpp>
#include <pops/mesh/boundary/physical_bc.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/numerics/elliptic/linear/prepared_affine_problem.hpp>
#include <pops/numerics/elliptic/nd/cartesian_tensor_operator.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::elliptic::polar {

/// Invert P0=-div(diag(r,1/r) grad) on the complete cell-centered disk root grid.
/// This is an approximate inverse for a different full tensor operator, never a replacement
/// for that operator or its residual/stopping checks. The angular symbol is the FV sine symbol.
/// The fixed real DFT costs O(Nr*Ntheta^2); every rank retains a bounded complete work image.
///
/// The prototype, boundary and execution lane must outlive this session. Construction does
/// not allocate or communicate; prepare() collectively authenticates/allocates/factors once.
/// Sessions own separate mutable buffers. apply() never allocates, rebuilds factors, changes
/// the input, uses a warm start, or borrows a process-global communicator. Replicated inputs
/// require exactly equal finite numerical values (signed zero is numerically equivalent).
template <int Dim>
class PreparedPolarPoissonInverse {
 public:
  using field_type = MultiFab<Dim>;
  using options_type = nd::CartesianTensorStencilOptions;

  static constexpr std::size_t maximum_radial_cells = 4096;
  static constexpr std::size_t maximum_angular_cells = 1024;
  static constexpr std::size_t maximum_cells = std::size_t{1} << 18;
  static constexpr std::size_t maximum_transform_products = std::size_t{1} << 26;
  static constexpr std::size_t maximum_storage_bytes = std::size_t{64} << 20;
  static constexpr std::size_t maximum_patches = 4096;

  PreparedPolarPoissonInverse(const field_type& prototype, const Geometry<Dim>& geometry,
                             const PreparedPhysicalBoundary<Dim>& boundary,
                             options_type options, const ExecutionLane& lane) noexcept
      : prototype_(&prototype), geometry_(geometry), boundary_(&boundary), options_(options),
        lane_(&lane) {}

  PreparedPolarPoissonInverse(const PreparedPolarPoissonInverse&) = delete;
  PreparedPolarPoissonInverse& operator=(const PreparedPolarPoissonInverse&) = delete;
  PreparedPolarPoissonInverse(PreparedPolarPoissonInverse&&) noexcept = default;
  PreparedPolarPoissonInverse& operator=(PreparedPolarPoissonInverse&&) noexcept = default;

  void prepare() {
    const long has_state = state_ ? 1L : 0L;
    const long minimum_state = all_reduce_min(has_state, *lane_);
    const long maximum_state = all_reduce_max(has_state, *lane_);
    if (minimum_state != maximum_state)
      throw std::invalid_argument("polar inverse preparation state differs between ranks");
    if (state_) {
      const long changed = matches_(*prototype_) ? 0L : 1L;
      if (all_reduce_max(changed, *lane_) != 0)
        throw std::invalid_argument("polar inverse prototype changed after preparation");
      return;
    }

    Dimensions dimensions{};
    std::string contract;
    long invalid = 0;
    try {
      dimensions = validate_();
      ExactContractBuilder builder;
      builder.text("pops.prepared-fv-polar-poisson-inverse@1")
          .text("real-half-DFT;FV-sine-symbol;radial-Thomas;exact-numeric-replicas")
          .text(lane_->identity()).scalar(Dim).scalar(prototype_->ncomp())
          .scalar(options_.zero_flux_faces).scalar(options_.dirichlet_faces)
          .scalar(options_.arithmetic_diagonal).scalar(prototype_->distribution().mode())
          .scalar(maximum_radial_cells).scalar(maximum_angular_cells).scalar(maximum_cells)
          .scalar(maximum_transform_products).scalar(maximum_storage_bytes)
          .scalar(maximum_patches).scalar(dimensions.storage_bytes);
      for (int axis = 0; axis < Dim; ++axis) {
        builder.scalar(geometry_.domain().lo[axis]).scalar(geometry_.domain().hi[axis])
            .scalar(geometry_.lower()[axis]).scalar(geometry_.upper()[axis])
            .scalar(geometry_.spacing(axis)).scalar(prototype_->ghosts()[axis])
            .scalar(prototype_->rank_space().origin()[axis])
            .scalar(prototype_->rank_space().extent()[axis])
            .scalar(boundary_->schedule().ghosts()[axis]);
        for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
          const Face<Dim> face{axis, side};
          const auto& law = boundary_->conditions().at(face);
          builder.scalar(boundary_->conditions().topology().is_periodic(face))
              .scalar(law.kind).scalar(law.value).scalar(law.alpha).scalar(law.beta);
        }
      }
      builder.scalar(prototype_->layout().size());
      for (const auto& box : prototype_->layout().boxes())
        for (int axis = 0; axis < Dim; ++axis)
          builder.scalar(box.lo[axis]).scalar(box.hi[axis]);
      for (const auto& owner : prototype_->distribution().owners())
        for (int axis = 0; axis < Dim; ++axis)
          builder.scalar(owner[axis]);
      contract = std::move(builder).release();
    } catch (...) {
      invalid = 1;
    }
    if (all_reduce_max(invalid, *lane_) != 0)
      throw std::invalid_argument("polar inverse geometry, boundary, ownership or size is unsupported");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"pops.prepared-fv-polar-poisson-inverse@1", contract}}, *lane_))
      throw std::invalid_argument("polar inverse prepared inputs differ between ranks");

    std::unique_ptr<State> candidate;
    long allocation_failed = 0;
    try {
      candidate = std::make_unique<State>(*prototype_, dimensions);
    } catch (...) {
      allocation_failed = 1;
    }
    if (all_reduce_max(allocation_failed, *lane_) != 0)
      throw std::runtime_error("polar inverse workspace allocation failed collectively");
    long factor_failed = 0;
    try {
      factor_(*candidate);
    } catch (...) {
      factor_failed = 1;
    }
    if (all_reduce_max(factor_failed, *lane_) != 0)
      throw std::invalid_argument("polar inverse requires finite positive FV Thomas pivots");
    state_ = std::move(candidate);
  }

  [[nodiscard]] PreparedApplyResult apply(field_type& out, const field_type& in) noexcept {
    try {
      const long malformed = !state_ || !matches_(in) || !matches_(out) ? 1L : 0L;
      if (all_reduce_max(malformed, *lane_) != 0)
        return failure_(out, "polar-inverse:unprepared-or-layout");
      auto& state = *state_;
      const auto values = state.values;
      const auto domain = geometry_.domain();
      const std::size_t nr = state.shape.nr;
      const std::size_t nt = state.shape.nt;
      const std::size_t nm = state.shape.nm;

      long pack_failed = 0;
      try {
        Kokkos::deep_copy(values, Real(0));
        for (std::size_t local = 0; local < in.local_size(); ++local) {
          const auto source = in.fab(local).view();
          const auto box = in.box(local);
          const std::size_t width = static_cast<std::size_t>(box.length(0));
          Kokkos::parallel_for(state.pack_label, Range(0, static_cast<std::size_t>(box.numPts())),
              KOKKOS_LAMBDA(const std::size_t ordinal) {
            Index<Dim> index = box.lo;
            index[0] += static_cast<int>(ordinal % width);
            index[1] += static_cast<int>(ordinal / width);
            const std::size_t i = static_cast<std::size_t>(
                static_cast<std::int64_t>(index[0]) - domain.lo[0]);
            const std::size_t j = static_cast<std::size_t>(
                static_cast<std::int64_t>(index[1]) - domain.lo[1]);
            values(j * nr + i) = source(index, 0);
          });
        }
        Kokkos::fence();
        Kokkos::deep_copy(state.host_values, values);
        for (std::size_t index = 0; index < state.shape.cells; ++index)
          if (!std::isfinite(state.host_values(index)))
            pack_failed = 1;
      } catch (...) {
        pack_failed = 1;
      }
      // All local packing/validation errors agree before the ownership data collective.
      if (all_reduce_max(pack_failed, *lane_) != 0)
        return failure_(out, "polar-inverse:nonfinite-or-pack");

      long reduction_failed = 0;
      try {
        if (state.distribution.replicated()) {
          for (std::size_t index = 0; index < state.shape.cells; ++index) {
            state.consensus(index) = state.host_values(index);
            state.consensus(state.shape.cells + index) = -state.host_values(index);
          }
          all_reduce_max_inplace(state.consensus.data(), 2 * state.shape.cells, *lane_);
          for (std::size_t index = 0; index < state.shape.cells; ++index) {
            const Real maximum = state.consensus(index);
            const Real minimum = -state.consensus(state.shape.cells + index);
            if (!std::isfinite(maximum) || !std::isfinite(minimum) || maximum != minimum)
              reduction_failed = 1;
            state.host_values(index) = maximum;
          }
        } else {
          // Exact full tiling and rank-space checks proved one contribution per cell.
          all_reduce_sum_inplace(state.host_values.data(), state.shape.cells, *lane_);
        }
      } catch (...) {
        reduction_failed = 1;
      }
      if (all_reduce_max(reduction_failed, *lane_) != 0)
        return failure_(out, "polar-inverse:replica-or-reduction");

      long solve_failed = 0;
      try {
        Kokkos::deep_copy(values, state.host_values);
        const auto coefficients = state.coefficients;
        const auto real = state.real;
        const auto imaginary = state.imaginary;
        const std::size_t sine_offset = state.shape.table;
        const std::size_t pivot_offset = 2 * state.shape.table;
        const std::size_t upper_offset = pivot_offset + state.shape.modal;
        const std::size_t lower_offset = upper_offset + state.shape.modal;
        Kokkos::parallel_for(state.forward_label, Range(0, state.shape.modal),
            KOKKOS_LAMBDA(const std::size_t ordinal) {
              const std::size_t mode = ordinal / nr;
              const std::size_t i = ordinal % nr;
              Real cosine_sum = 0;
              Real sine_sum = 0;
              for (std::size_t j = 0; j < nt; ++j) {
                const Real value = values(j * nr + i);
                cosine_sum += value * coefficients(mode * nt + j);
                sine_sum += value * coefficients(sine_offset + mode * nt + j);
              }
              real(ordinal) = cosine_sum;
              imaginary(ordinal) = sine_sum;
            });
        Kokkos::parallel_for(state.radial_label, Range(0, nm),
            KOKKOS_LAMBDA(const std::size_t mode) {
              const std::size_t base = mode * nr;
              Real previous_real = 0;
              Real previous_imaginary = 0;
              for (std::size_t i = 0; i < nr; ++i) {
                const std::size_t ordinal = base + i;
                const Real lower = coefficients(lower_offset + i);
                const Real inverse_pivot = coefficients(pivot_offset + ordinal);
                previous_real = (real(ordinal) - lower * previous_real) * inverse_pivot;
                previous_imaginary =
                    (imaginary(ordinal) - lower * previous_imaginary) * inverse_pivot;
                real(ordinal) = previous_real;
                imaginary(ordinal) = previous_imaginary;
              }
              for (std::size_t i = nr - 1; i > 0; --i) {
                const std::size_t ordinal = base + i - 1;
                const Real upper = coefficients(upper_offset + ordinal);
                real(ordinal) -= upper * real(ordinal + 1);
                imaginary(ordinal) -= upper * imaginary(ordinal + 1);
              }
            });
        Kokkos::parallel_for(state.backward_label, Range(0, state.shape.cells),
            KOKKOS_LAMBDA(const std::size_t ordinal) {
              const std::size_t j = ordinal / nr;
              const std::size_t i = ordinal % nr;
              Real sum = real(i);
              for (std::size_t mode = 1; mode < nm; ++mode) {
                const Real weight = 2 * mode == nt ? Real(1) : Real(2);
                sum += weight * (real(mode * nr + i) * coefficients(mode * nt + j) +
                    imaginary(mode * nr + i) * coefficients(sine_offset + mode * nt + j));
              }
              values(ordinal) = sum / static_cast<Real>(nt);
            });
        Kokkos::fence();
        Kokkos::deep_copy(state.host_values, values);
        for (std::size_t index = 0; index < state.shape.cells; ++index)
          if (!std::isfinite(state.host_values(index)))
            solve_failed = 1;
      } catch (...) {
        solve_failed = 1;
      }
      if (all_reduce_max(solve_failed, *lane_) != 0)
        return failure_(out, "polar-inverse:nonfinite-or-transform");

      long scatter_failed = 0;
      try {
        for (std::size_t local = 0; local < out.local_size(); ++local) {
          const auto target = out.fab(local).view();
          const auto box = out.box(local);
          const std::size_t width = static_cast<std::size_t>(box.length(0));
          Kokkos::parallel_for(state.scatter_label, Range(0, static_cast<std::size_t>(box.numPts())),
              KOKKOS_LAMBDA(const std::size_t ordinal) {
            Index<Dim> index = box.lo;
            index[0] += static_cast<int>(ordinal % width);
            index[1] += static_cast<int>(ordinal / width);
            const std::size_t i = static_cast<std::size_t>(
                static_cast<std::int64_t>(index[0]) - domain.lo[0]);
            const std::size_t j = static_cast<std::size_t>(
                static_cast<std::int64_t>(index[1]) - domain.lo[1]);
            target(index, 0) = values(j * nr + i);
          });
        }
        Kokkos::fence();
      } catch (...) {
        scatter_failed = 1;
      }
      if (all_reduce_max(scatter_failed, *lane_) != 0)
        return failure_(out, "polar-inverse:scatter");
      return PreparedApplyResult::success();
    } catch (...) {
      return failure_(out, "polar-inverse:collective");
    }
  }

  /// Seven persistent Kokkos allocations on every rank, including empty owners. Metadata is
  /// allocated only in prepare(); direct storage calls also enter PoPS allocation-event counters.
  [[nodiscard]] std::size_t allocation_count() const noexcept { return state_ ? 7 : 0; }
  [[nodiscard]] std::size_t storage_bytes() const noexcept {
    return state_ ? state_->shape.storage_bytes : 0;
  }

 private:
  using device_view = Kokkos::View<Real*>;
  using pinned_view = Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace>;
  using Range = Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<std::size_t>>;

  struct Dimensions {
    std::size_t nr = 0, nt = 0, nm = 0, cells = 0, modal = 0, table = 0;
    std::size_t coefficient_count = 0, storage_bytes = 0;
  };

  struct State {
    typename field_type::layout_type layout;
    typename field_type::distribution_type distribution;
    Index<Dim> local_rank;
    std::vector<std::size_t> local_indices;
    Dimensions shape;
    device_view values, coefficients, real, imaginary;
    pinned_view host_values, consensus, host_coefficients;
    // Keep kernel-name storage out of repeated Kokkos call argument conversions.
    std::string pack_label = "polar_inverse_pack";
    std::string forward_label = "polar_inverse_forward";
    std::string radial_label = "polar_inverse_radial";
    std::string backward_label = "polar_inverse_backward";
    std::string scatter_label = "polar_inverse_scatter";

    State(const field_type& prototype, Dimensions dimensions)
        : layout(prototype.layout()), distribution(prototype.distribution()),
          local_rank(prototype.local_rank()), local_indices(prototype.local_global_indices()),
          shape(dimensions) {
      values = device_view("polar_inverse_values", shape.cells);
      ::pops::detail::record_fab_allocation(shape.cells * sizeof(Real));
      coefficients = device_view("polar_inverse_coefficients", shape.coefficient_count);
      ::pops::detail::record_fab_allocation(shape.coefficient_count * sizeof(Real));
      real = device_view("polar_inverse_real", shape.modal);
      ::pops::detail::record_fab_allocation(shape.modal * sizeof(Real));
      imaginary = device_view("polar_inverse_imaginary", shape.modal);
      ::pops::detail::record_fab_allocation(shape.modal * sizeof(Real));
      host_values = pinned_view("polar_inverse_host_values", shape.cells);
      ::pops::detail::record_communication_allocation(shape.cells * sizeof(Real));
      consensus = pinned_view("polar_inverse_consensus", 2 * shape.cells);
      ::pops::detail::record_communication_allocation(2 * shape.cells * sizeof(Real));
      host_coefficients = pinned_view("polar_inverse_host_coefficients", shape.coefficient_count);
      ::pops::detail::record_communication_allocation(shape.coefficient_count * sizeof(Real));
    }
  };

  static std::size_t product_(std::size_t left, std::size_t right) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
      throw std::length_error("polar inverse size product exceeds size_t");
    return left * right;
  }
  static std::size_t sum_(std::size_t left, std::size_t right) {
    if (left > std::numeric_limits<std::size_t>::max() - right)
      throw std::length_error("polar inverse size sum exceeds size_t");
    return left + right;
  }

  Dimensions validate_() const {
    if constexpr (Dim != 2) {
      throw std::invalid_argument("polar inverse supports only two dimensions");
    } else {
      Dimensions result;
      const auto& domain = geometry_.domain();
      const auto& conditions = boundary_->conditions();
      if (domain.length(0) < 2 || domain.length(1) < 2 || geometry_.lower()[0] != Real(0) ||
          geometry_.upper()[1] - geometry_.lower()[1] != Real(2) * std::numbers::pi_v<Real> ||
          options_.zero_flux_faces != 1u || options_.dirichlet_faces != 2u ||
          !options_.arithmetic_diagonal || prototype_->ncomp() != 1 ||
          boundary_->schedule().domain() != domain)
        throw std::invalid_argument("polar inverse requires the arithmetic-face full disk stencil");
      for (int axis = 0; axis < Dim; ++axis) {
        const Real inverse_spacing = Real(1) / geometry_.spacing(axis);
        if (!std::isfinite(inverse_spacing) || !(inverse_spacing > 0) ||
            conditions.spacing()[axis] != geometry_.spacing(axis))
          throw std::invalid_argument("polar inverse spacing is incoherent");
        for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
          const Face<Dim> face{axis, side};
          const auto& law = conditions.at(face);
          if (conditions.topology().is_periodic(face) != (axis == 1) ||
              !std::isfinite(law.value) || !std::isfinite(law.alpha) || !std::isfinite(law.beta))
            throw std::invalid_argument("polar inverse topology is unsupported");
        }
      }
      const auto& pole = conditions.at(Face<Dim>{0, BoundarySide::lower});
      const auto& wall = conditions.at(Face<Dim>{0, BoundarySide::upper});
      if (pole.kind != PhysicalBoundaryKind::neumann || pole.value != 0 ||
          wall.kind != PhysicalBoundaryKind::dirichlet || wall.value != 0)
        throw std::invalid_argument("polar inverse requires zero pole flux and a grounded wall");
      result.nr = static_cast<std::size_t>(domain.length(0));
      result.nt = static_cast<std::size_t>(domain.length(1));
      if (result.nr > maximum_radial_cells || result.nt > maximum_angular_cells)
        throw std::length_error("polar inverse transform extent exceeds its fixed bound");
      result.nm = result.nt / 2 + 1;
      result.cells = product_(result.nr, result.nt);
      result.modal = product_(result.nr, result.nm);
      result.table = product_(result.nt, result.nm);
      if (result.cells > maximum_cells ||
          product_(result.nr, result.table) > maximum_transform_products)
        throw std::length_error("polar inverse direct-transform work exceeds its fixed bound");
      result.coefficient_count = sum_(sum_(product_(2, result.table), product_(2, result.modal)),
                                      result.nr);
      const std::size_t real_values = sum_(sum_(product_(4, result.cells),
                                               product_(2, result.coefficient_count)),
                                         product_(2, result.modal));
      const std::size_t metadata = sum_(sum_(sizeof(State), 128),
          product_(prototype_->layout().size(),
              2 * sizeof(Box<Dim>) + sizeof(Index<Dim>) + sizeof(std::size_t)));
      result.storage_bytes = sum_(product_(real_values, sizeof(Real)), metadata);
      if (result.storage_bytes > maximum_storage_bytes ||
          result.cells > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 2)
        throw std::length_error("polar inverse persistent storage exceeds its fixed bound");
      if (prototype_->rank_space().size() != static_cast<std::size_t>(lane_->size()) ||
          prototype_->rank_space().linear_rank(prototype_->local_rank()) !=
              static_cast<std::size_t>(lane_->rank()) ||
          !prototype_->distribution().matches_layout(prototype_->layout()))
        throw std::invalid_argument("polar inverse field rank space differs from its execution lane");
      if (!prototype_->layout().tiles_exactly(domain,
              {maximum_patches, maximum_patches * (maximum_patches - 1) / 2}))
        throw std::invalid_argument("polar inverse requires a disjoint complete coarse tiling");
      return result;
    }
  }

  void factor_(State& state) const {
    const std::size_t nr = state.shape.nr, nt = state.shape.nt, nm = state.shape.nm;
    const std::size_t sine_offset = state.shape.table;
    const std::size_t pivot_offset = 2 * state.shape.table;
    const std::size_t upper_offset = pivot_offset + state.shape.modal;
    const std::size_t lower_offset = upper_offset + state.shape.modal;
    auto host = state.host_coefficients;
    const Real radial_inverse_spacing = Real(1) / geometry_.spacing(0);
    const Real angular_inverse_spacing = Real(1) / geometry_.spacing(1);
    const Real pi = std::numbers::pi_v<Real>;
    for (std::size_t mode = 0; mode < nm; ++mode) {
      for (std::size_t j = 0; j < nt; ++j) {
        const Real angle = Real(2) * pi * static_cast<Real>(mode) * static_cast<Real>(j) /
                           static_cast<Real>(nt);
        // These two columns have no sine partner; keep their exact normalization explicit.
        host(mode * nt + j) = mode == 0 ? Real(1) :
            (2 * mode == nt ? (j % 2 == 0 ? Real(1) : Real(-1)) : std::cos(angle));
        host(sine_offset + mode * nt + j) =
            mode == 0 || 2 * mode == nt ? Real(0) : std::sin(angle);
      }
      const Real sine = std::sin(pi * static_cast<Real>(mode) / static_cast<Real>(nt));
      const Real lambda = ((Real(4) * sine * sine) * angular_inverse_spacing) *
                          angular_inverse_spacing;
      Real previous_upper = 0;
      for (std::size_t i = 0; i < nr; ++i) {
        const int index = static_cast<int>(static_cast<std::int64_t>(geometry_.domain().lo[0]) + i);
        const Real radius = geometry_.cell_coordinate(0, index);
        // Match native sampled metric arithmetic, including one-sided physical-face
        // extrapolation. In particular, do not replace these with ideal (i+1)*h faces.
        const Real lower_face = i == 0 ? Real(0) :
            Real(0.5) * (geometry_.cell_coordinate(0, index - 1) + radius);
        const Real upper_face = i + 1 == nr ?
            Real(1.5) * radius - Real(0.5) * geometry_.cell_coordinate(0, index - 1) :
            Real(0.5) * (radius + geometry_.cell_coordinate(0, index + 1));
        const Real lower = (-lower_face * radial_inverse_spacing) * radial_inverse_spacing;
        const Real upper = i + 1 == nr ? Real(0) :
            (-upper_face * radial_inverse_spacing) * radial_inverse_spacing;
        const Real diagonal = ((lower_face + (i + 1 == nr ? Real(2) : Real(1)) * upper_face) *
            radial_inverse_spacing) * radial_inverse_spacing + lambda * (Real(1) / radius);
        const Real pivot = diagonal - lower * previous_upper;
        const Real inverse_pivot = Real(1) / pivot;
        if (!std::isfinite(radius) || !(radius > 0) || !std::isfinite(lower) ||
            !std::isfinite(upper) || !std::isfinite(pivot) || !(pivot > 0) ||
            !std::isfinite(inverse_pivot) || !(inverse_pivot > 0))
          throw std::invalid_argument("polar inverse FV factor is nonfinite or nonpositive");
        previous_upper = upper * inverse_pivot;
        if (!std::isfinite(previous_upper))
          throw std::invalid_argument("polar inverse normalized upper factor is nonfinite");
        host(pivot_offset + mode * nr + i) = inverse_pivot;
        host(upper_offset + mode * nr + i) = previous_upper;
        host(lower_offset + i) = lower;
      }
    }
    Kokkos::deep_copy(state.coefficients, host);
    Kokkos::fence();
  }

  bool matches_(const field_type& field) const noexcept {
    return state_ && field.ncomp() == 1 && field.layout() == state_->layout &&
           field.distribution() == state_->distribution && field.local_rank() == state_->local_rank &&
           field.local_global_indices() == state_->local_indices &&
           field.local_size() == state_->local_indices.size();
  }

  static PreparedApplyResult failure_(field_type& out, std::string_view phase) noexcept {
    try {
      out.set_val(std::numeric_limits<Real>::quiet_NaN());
      Kokkos::fence();
    } catch (...) {
    }
    auto result = PreparedApplyResult::unknown_failure();
    const std::size_t count = phase.size() < result.kPhaseCapacity ? phase.size() : result.kPhaseCapacity;
    std::memcpy(result.phase_bytes.data(), phase.data(), count);
    result.phase_size = static_cast<std::uint8_t>(count);
    result.phase_truncated = count != phase.size();
    return result;
  }

  const field_type* prototype_;
  Geometry<Dim> geometry_;
  const PreparedPhysicalBoundary<Dim>* boundary_;
  options_type options_;
  const ExecutionLane* lane_;
  std::unique_ptr<State> state_;
};

}  // namespace pops::elliptic::polar
