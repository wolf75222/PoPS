#pragma once

/// @file
/// @brief MultiFab adapter that hosts PoissonFFT on one uniform periodic level.
///
/// This is the only honest FFT-on-AMR seam: the coarsest uniform periodic box (or a replicated
/// copy of that box rewritten onto canonical last-axis slabs) may invert the constant-k Poisson
/// operator with PoissonFFTSolver. A global FFT over a sparse AMR hierarchy, covered cells, or
/// any non-uniform coarse layout stays refused. Variable-k, embedded-boundary, and Helmholtz
/// (reaction > 0) coarse levels keep Jacobi / GeometricMG.

#include <pops/mesh/execution/for_each.hpp>
#include <pops/numerics/elliptic/poisson/poisson_fft_solver.hpp>
#include <pops/numerics/elliptic/poisson/poisson_operator.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops::elliptic {

enum class PoissonFftBottomKind : unsigned char {
  none,
  native_solver,
  replicated_slab_rewrite,
};

struct PoissonFftBottomClassification {
  PoissonFftBottomKind kind = PoissonFftBottomKind::none;

  constexpr bool eligible() const noexcept { return kind != PoissonFftBottomKind::none; }
};

namespace fft_multifab_detail {

template <int Dim>
Extent<Dim> unit_ghosts() {
  Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    ghosts[axis] = 1;
  return ghosts;
}

template <int Dim>
EllipticBuildRequest<Dim> with_fft_ghosts(EllipticBuildRequest<Dim> request) {
  request.rhs_ghosts = {};
  request.phi_ghosts = unit_ghosts<Dim>();
  return request;
}

template <int Dim>
bool single_full_domain_box(const EllipticBuildRequest<Dim>& request) {
  return request.boxes.size() == 1 && request.boxes[0] == request.geometry.domain();
}

template <int Dim>
bool last_axis_divides_ranks(const Geometry<Dim>& geometry, int ranks) {
  return ranks >= 1 && geometry.domain().length(Dim - 1) % ranks == 0;
}

template <int Dim>
Index<Dim> index_from_ordinal(const Box<Dim>& box, std::size_t ordinal) {
  Index<Dim> result{};
  for (int axis = 0; axis < Dim; ++axis) {
    const auto length = static_cast<std::size_t>(box.length(axis));
    result[axis] = box.lo[axis] + static_cast<int>(ordinal % length);
    ordinal /= length;
  }
  return result;
}

template <int Dim>
std::size_t storage_ordinal(const Box<Dim>& storage, const Index<Dim>& index) {
  std::size_t ordinal = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    ordinal += static_cast<std::size_t>(index[axis] - storage.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(storage.length(axis));
  }
  return ordinal;
}

template <int Dim>
void copy_overlapping_valid(const MultiFab<Dim>& source, MultiFab<Dim>& destination) {
  if (source.ncomp() != 1 || destination.ncomp() != 1)
    throw std::invalid_argument("Poisson FFT MultiFab overlap copy requires scalar fields");
  for (std::size_t dst = 0; dst < destination.local_size(); ++dst) {
    const Box<Dim> dst_box = destination.box(dst);
    const auto out = destination.fab(dst).view();
    for (std::size_t src = 0; src < source.local_size(); ++src) {
      const Box<Dim> overlap = dst_box.intersect(source.box(src));
      if (overlap.empty())
        continue;
      const auto in = source.fab(src).view();
      for_each_cell(overlap, [=] POPS_HD(const Index<Dim>& cell) { out(cell, 0) = in(cell, 0); });
    }
  }
  ::pops::device_fence();
}

template <int Dim>
void allreduce_sum_valid(MultiFab<Dim>& field, const ExecutionLane& lane) {
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    auto& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& storage = fab.grown_box();
    const std::size_t count = static_cast<std::size_t>(valid.numPts());
    std::vector<double> values(count, 0.0);
    for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
      const Index<Dim> cell = index_from_ordinal(valid, ordinal);
      values[ordinal] = static_cast<double>(host(storage_ordinal(storage, cell)));
    }
    all_reduce_sum_inplace(values.data(), values.size(), lane);
    for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
      const Index<Dim> cell = index_from_ordinal(valid, ordinal);
      host(storage_ordinal(storage, cell)) = static_cast<Real>(values[ordinal]);
    }
    fab.copy_from_host(host);
  }
}

template <int Dim>
EllipticBuildRequest<Dim> unique_last_axis_slab_request(const EllipticBuildRequest<Dim>& source,
                                                        const ExecutionLane& lane) {
  const int ranks = lane.size();
  const Box<Dim>& domain = source.geometry.domain();
  if (!last_axis_divides_ranks(source.geometry, ranks))
    throw std::invalid_argument(
        "Poisson FFT slab rewrite requires the communicator size to divide the last axis");
  std::vector<Box<Dim>> slabs;
  slabs.reserve(static_cast<std::size_t>(ranks));
  const int local_last = domain.length(Dim - 1) / ranks;
  for (int rank = 0; rank < ranks; ++rank) {
    Box<Dim> slab = domain;
    slab.lo[Dim - 1] = domain.lo[Dim - 1] + rank * local_last;
    slab.hi[Dim - 1] = slab.lo[Dim - 1] + local_last - 1;
    slabs.push_back(slab);
  }
  mesh::BoxArray<Dim> layout(std::move(slabs));
  Extent<Dim> rank_extent{};
  for (int axis = 0; axis < Dim; ++axis)
    rank_extent[axis] = 1;
  rank_extent[Dim - 1] = ranks;
  const mesh::RankSpace<Dim> rank_space{Index<Dim>{}, rank_extent};
  std::vector<Index<Dim>> owners;
  owners.reserve(static_cast<std::size_t>(ranks));
  for (int rank = 0; rank < ranks; ++rank)
    owners.push_back(rank_space.coordinate(static_cast<std::size_t>(rank)));
  const mesh::Distribution<Dim> distribution =
      mesh::Distribution<Dim>::partitioned(layout, rank_space, std::move(owners));
  const std::size_t pairs =
      layout.size() < 2 ? 0 : layout.size() * (layout.size() - 1) / 2;
  EllipticBuildRequest<Dim> request = with_fft_ghosts(source);
  request.boxes = std::move(layout);
  request.distribution = distribution;
  request.local_rank = rank_space.coordinate(static_cast<std::size_t>(lane.rank()));
  request.layout_budget = {request.boxes.size(), pairs};
  return request;
}

}  // namespace fft_multifab_detail

/// Copies a MultiFab residual onto the device-resident PoissonFFT brick and writes the correction
/// back. Eligibility is the PoissonFFTSolver layout contract; ghosts stay with the caller so the
/// adapter never opens a second HaloExchange on a lane that already hosts FAC/MG.
template <int Dim>
class PoissonFftMultiFabAdapter {
 public:
  static_assert(Dim >= 1 && Dim <= 3, "PoissonFftMultiFabAdapter supports dimensions 1, 2, and 3");

  using field_type = MultiFab<Dim>;
  using request_type = EllipticBuildRequest<Dim>;
  using complex_type = typename PoissonFFT<Dim>::complex_type;
  using device_view = typename PoissonFFT<Dim>::device_view;

  static PoissonFftBottomClassification classify(const request_type& request,
                                                 const ExecutionLane& lane, Real reaction,
                                                 bool has_coefficient,
                                                 bool has_embedded_boundary) noexcept {
    PoissonFftBottomClassification result;
    try {
      if (has_coefficient || has_embedded_boundary || reaction != Real(0))
        return result;
      const request_type normalized = fft_multifab_detail::with_fft_ghosts(request);
      fft_solver_detail::validate_periodic_boundary(normalized.geometry, normalized.boundary);
      if (PoissonFFTSolver<Dim>::supports(normalized, lane)) {
        result.kind = PoissonFftBottomKind::native_solver;
        return result;
      }
      if (normalized.distribution.replicated() &&
          fft_multifab_detail::single_full_domain_box(normalized) && lane.size() > 1 &&
          fft_multifab_detail::last_axis_divides_ranks(normalized.geometry, lane.size()) &&
          PoissonFFTSolver<Dim>::supports(
              fft_multifab_detail::unique_last_axis_slab_request(normalized, lane), lane)) {
        result.kind = PoissonFftBottomKind::replicated_slab_rewrite;
        return result;
      }
    } catch (...) {
      return {};
    }
    return result;
  }

  static std::unique_ptr<PoissonFftMultiFabAdapter<Dim>> try_make(const request_type& request,
                                                                  const ExecutionLane& lane,
                                                                  Real reaction,
                                                                  bool has_coefficient,
                                                                  bool has_embedded_boundary) {
    const auto classification =
        classify(request, lane, reaction, has_coefficient, has_embedded_boundary);
    const long eligible = classification.eligible() ? 1 : 0;
    if (all_reduce_min(eligible, lane) != all_reduce_max(eligible, lane))
      throw std::invalid_argument(
          "Poisson FFT MultiFab adapter eligibility differs across communicator ranks");
    if (eligible == 0)
      return nullptr;
    return std::make_unique<PoissonFftMultiFabAdapter<Dim>>(request, lane, reaction,
                                                            has_coefficient, has_embedded_boundary);
  }

  PoissonFftMultiFabAdapter(const request_type& request, const ExecutionLane& lane, Real reaction,
                            bool has_coefficient, bool has_embedded_boundary)
      : lane_(&lane),
        geometry_(request.geometry),
        classification_(classify(request, lane, reaction, has_coefficient, has_embedded_boundary)) {
    if (!classification_.eligible())
      throw std::invalid_argument(
          "Poisson FFT MultiFab adapter requires a uniform periodic constant-k coarse level");
    fft_.emplace(fft_solver_detail::fft_cells(geometry_), fft_solver_detail::fft_lengths(geometry_),
                 lane, "pops.poisson-fft/multifab-bottom");
    local_slab_ = geometry_.domain();
    local_slab_.lo[Dim - 1] = geometry_.domain().lo[Dim - 1] + fft_->local_last_begin();
    local_slab_.hi[Dim - 1] = local_slab_.lo[Dim - 1] + fft_->local_last_extent() - 1;
    fft_rhs_ = device_view("pops.poisson-fft.multifab-rhs", fft_->local_cell_count());
    fft_phi_ = device_view("pops.poisson-fft.multifab-phi", fft_->local_cell_count());
    cell_count_ = static_cast<Real>(geometry_.domain().numPts());
  }

  PoissonFftBottomKind kind() const noexcept { return classification_.kind; }

  SolveReport apply(const field_type& rhs, field_type& correction) {
    if (!fft_)
      throw std::logic_error("Poisson FFT MultiFab adapter has no prepared plan");
    SolveReport report;
    const Real reference = static_cast<Real>(
        all_reduce_max(static_cast<double>(::pops::norm_inf(rhs)), *lane_));
    report.reference_residual_norm = reference;
    const Real mean = rhs_mean_(rhs);
    const Real envelope = fft_solver_detail::kDirectResidualSafetyFactor *
                          std::numeric_limits<Real>::epsilon() *
                          std::sqrt(std::max(Real(1), cell_count_)) * std::max(Real(1), reference);
    if (std::abs(static_cast<double>(mean)) > static_cast<double>(envelope)) {
      report.mark_failed(SolveStatus::kIncompatibleRhs, SolveAction::kFailRun,
                         "poisson_fft_multifab_incompatible_rhs");
      return report;
    }

    Kokkos::deep_copy(fft_rhs_, complex_type(0.0, 0.0));
    pack_slab_(rhs, fft_rhs_, Real(-1));
    fft_->solve(fft_rhs_, fft_phi_);
    if (classification_.kind == PoissonFftBottomKind::replicated_slab_rewrite)
      correction.set_val(Real(0));
    unpack_slab_(fft_phi_, correction);
    if (classification_.kind == PoissonFftBottomKind::replicated_slab_rewrite)
      fft_multifab_detail::allreduce_sum_valid(correction, *lane_);
    subtract_mean_(correction);
    report.evaluations = 1;
    report.mark_solved("poisson_fft_multifab_discrete_direct");
    return report;
  }

 private:
  Real rhs_mean_(const field_type& field) const {
    Real local = ::pops::reduce_sum_local(field, 0);
    if (field.distribution().replicated())
      local /= static_cast<Real>(std::max(1, lane_->size()));
    return static_cast<Real>(all_reduce_sum(static_cast<double>(local), *lane_)) / cell_count_;
  }

  void pack_slab_(const field_type& source, const device_view& destination, Real sign) const {
    const Box<Dim> slab = local_slab_;
    const auto dest = destination;
    for (std::size_t local = 0; local < source.local_size(); ++local) {
      const Box<Dim> overlap = source.box(local).intersect(slab);
      if (overlap.empty())
        continue;
      const auto in = source.fab(local).view();
      for_each_cell(overlap, [=] POPS_HD(const Index<Dim>& cell) {
        dest[fft_solver_detail::local_slab_ordinal(slab, cell)] =
            complex_type(static_cast<double>(sign * in(cell, 0)), 0.0);
      });
    }
    ::pops::device_fence();
  }

  void unpack_slab_(const device_view& source, field_type& destination) const {
    const Box<Dim> slab = local_slab_;
    const auto src = source;
    for (std::size_t local = 0; local < destination.local_size(); ++local) {
      const Box<Dim> overlap = destination.box(local).intersect(slab);
      if (overlap.empty())
        continue;
      const auto out = destination.fab(local).view();
      for_each_cell(overlap, [=] POPS_HD(const Index<Dim>& cell) {
        out(cell, 0) = static_cast<Real>(src[fft_solver_detail::local_slab_ordinal(slab, cell)].real());
      });
    }
    ::pops::device_fence();
  }

  void subtract_mean_(field_type& field) const {
    const Real mean = rhs_mean_(field);
    ::pops::elliptic::mg::add_scalar_valid(field, -mean);
  }

  const ExecutionLane* lane_ = nullptr;
  Geometry<Dim> geometry_;
  PoissonFftBottomClassification classification_{};
  std::optional<PoissonFFT<Dim>> fft_{};
  device_view fft_rhs_{};
  device_view fft_phi_{};
  Box<Dim> local_slab_{};
  Real cell_count_ = Real(1);
};

}  // namespace pops::elliptic
