#pragma once
#include <pops/numerics/elliptic/interface/amr_field_newton_krylov.hpp>
#include <pops/numerics/spatial/nd/face_field.hpp>
#include <functional>

namespace pops::runtime::program {
/// Full temporal F(q)=Q(q)-U_n-tau R(Q(q)) on one immutable active hierarchy.
/// Newton/GMRES is an algorithm here: no physical field slot or gauge is introduced.
template <int Dim>
class PreparedAmrSpatialResidual final {
 public:
  using field_type = MultiFab<Dim>;
  using hierarchy_type = std::vector<field_type>;
  using faces_type = std::vector<nd::FaceField<Dim>>;
  using evaluation_type = std::function<const field_type*(field_type&, const field_type&,
                                                          field_type&, const field_type*)>;
  PreparedAmrSpatialResidual(std::span<const field_type* const> layouts,
                             std::span<const field_type* const> masks,
                             std::span<const Real> measures, FieldNewtonOptions options,
                             Real difference_step, std::span<const field_type* const> coverage = {})
      : masks_(masks.begin(), masks.end()),
        measures_(measures.begin(), measures.end()),
        difference_step_(difference_step) {
    if (!std::isfinite(difference_step_) || !(difference_step_ > Real(0)))
      throw std::invalid_argument("composite spatial residual requires a finite positive FD step");
    if (layouts.empty() || measures_.size() != layouts.size() || masks_.size() != layouts.size() ||
        (!coverage.empty() && coverage.size() != layouts.size()))
      throw std::invalid_argument("composite spatial residual has an incomplete mask hierarchy");
    const int components = layouts.front() ? layouts.front()->ncomp() : 0;
    for (const auto* layout : layouts)
      if (!layout || components <= 0 || layout->ncomp() != components)
        throw std::invalid_argument("composite spatial levels require one exact vector width");
    for (const auto* layout : layouts)
      for (auto* tower : {&candidate_, &previous_, &perturbed_, &plus_, &minus_})
        tower->emplace_back(layout->layout(), layout->distribution(), layout->local_rank(),
                            components, layout->ghosts());
    owned_masks_.reserve(layouts.size());
    for (std::size_t level = 0; level < layouts.size(); ++level) {
      const auto& prototype = *layouts[level];
      owned_masks_.emplace_back(prototype.layout(), prototype.distribution(),
                                prototype.local_rank(), 1, Extent<Dim>{});
      if (masks_[level])
        lincomb(owned_masks_.back(), Real(1), *masks_[level], Real(0), *masks_[level]);
      else
        owned_masks_.back().set_val(Real(1));
      if (!coverage.empty()) {
        const auto* active = coverage[level];
        if (!active || active->ncomp() != 1 || active->layout() != prototype.layout() ||
            active->distribution() != prototype.distribution() ||
            active->local_rank() != prototype.local_rank())
          throw std::invalid_argument("composite spatial coverage differs from its level carrier");
        // Preserve EB inactivity and remove covered parent DOFs once per prepared topology.
        for (std::size_t local = 0; local < prototype.local_size(); ++local) {
          const auto destination = owned_masks_.back().fab(local).view();
          const auto covered = active->fab(local).view();
          for_each_cell(prototype.box(local), [=] POPS_HD(const Index<Dim>& cell) {
            if (!(covered(cell, 0) >= Real(0.5)))
              destination(cell, 0) = Real(0);
          });
        }
      }
    }
    for (std::size_t level = 0; level < layouts.size(); ++level)
      masks_[level] = &owned_masks_[level];
    newton_ =
        std::make_unique<AmrFieldNewtonKrylovWorkspace<Dim>>(layouts, masks_, measures_, options);
    for (auto& candidate : candidate_)
      destinations_.push_back(&candidate);
    mappings.resize(layouts.size());
    conserved.resize(layouts.size());
    evaluations.resize(layouts.size());
    accepted.resize(layouts.size());
    faces.resize(layouts.size(), nullptr);
  }
  void stage(std::size_t level, const field_type& previous, const field_type* seed) {
    authenticate_(previous, level);
    if (seed)
      authenticate_(*seed, level);
    lincomb(previous_[level], Real(1), previous, Real(0), previous);
    if (seed) {
      lincomb(candidate_[level], Real(1), *seed, Real(0), *seed);
    } else
      candidate_[level].set_val(Real(0));
  }
  template <class Evaluate>
  SolveReport solve(Evaluate&& evaluate, const ExecutionLane& lane) {
    residual_evaluations_ = derivative_evaluations_ = 0;
    auto full = [&](const hierarchy_type& q, hierarchy_type& result, int evaluation) {
      increment_(residual_evaluations_);
      evaluate(q, previous_, result, evaluation);
    };
    auto defect = [&](const hierarchy_type& q, hierarchy_type& result, int evaluation) {
      full(q, result, evaluation);
      for (auto& level : result)
        scale(level, Real(-1));
    };
    auto jvp = [&](const hierarchy_type& q, const hierarchy_type& direction, hierarchy_type& result,
                   int evaluation) {
      increment_(derivative_evaluations_);
      const Real norm_q = norm_(q, lane), norm_v = norm_(direction, lane);
      if (norm_v == Real(0)) {
        for (auto& level : result)
          level.set_val(Real(0));
        return;
      }
      const Real step = difference_step_ * std::max(Real(1), norm_q) / norm_v;
      if (!std::isfinite(step) || !(step > Real(0))) {
        for (auto& level : result)
          level.set_val(std::numeric_limits<Real>::quiet_NaN());
        return;
      }
      for (std::size_t level = 0; level < q.size(); ++level)
        lincomb(perturbed_[level], Real(1), q[level], step, direction[level]);
      full(perturbed_, plus_, evaluation);
      for (std::size_t level = 0; level < q.size(); ++level)
        lincomb(perturbed_[level], Real(1), q[level], -step, direction[level]);
      full(perturbed_, minus_, evaluation);
      for (std::size_t level = 0; level < q.size(); ++level)
        lincomb(result[level], Real(0.5) / step, plus_[level], -Real(0.5) / step, minus_[level]);
    };
    auto no_gauge = [](hierarchy_type&) {};
    auto report = newton_->solve(destinations_, defect, jvp, no_gauge, lane);
    report.evaluations = residual_evaluations_;
    return report;
  }
  const field_type& candidate(std::size_t level) const { return candidate_.at(level); }
  field_type& previous(std::size_t level) { return previous_.at(level); }
  std::size_t levels() const noexcept { return candidate_.size(); }
  int residual_evaluations() const noexcept { return residual_evaluations_; }
  int derivative_evaluations() const noexcept { return derivative_evaluations_; }
  std::vector<std::function<field_type*(const field_type&)>> mappings;
  std::vector<field_type*> conserved;
  std::vector<evaluation_type> evaluations;
  std::vector<std::function<void()>> accepted;
  std::vector<const faces_type*> faces;
  Real residual_face_weight = Real(0);
  struct Interface {
    std::size_t parent;
    int axis;
    Index<Dim> coarse_face, coarse_cell;
    std::vector<Index<Dim>> fine_faces;
    Real coarse_area, fine_area, inverse_volume, divergence_sign;
    std::size_t coarse_sample = 0;
    std::vector<std::size_t> fine_samples;
  };
  std::vector<Interface> interfaces;
  struct FaceSample {
    std::size_t level, local_patch;
    int axis;
    Index<Dim> face;
    bool owned;
  };
  struct FaceGatherEntry {
    FieldView<const Real, Dim> density;
    Index<Dim> face;
    bool owned;
  };
  std::vector<FaceSample> face_samples;
  Kokkos::View<FaceGatherEntry*> face_gather;
  Kokkos::View<FaceGatherEntry*, Kokkos::HostSpace> face_gather_host;
  Kokkos::View<Real*> face_values;
  Kokkos::View<Real*, Kokkos::HostSpace> face_values_host;

 private:
  static void increment_(int& count) {
    if (count == std::numeric_limits<int>::max())
      throw std::length_error("composite spatial evaluation counter overflow");
    ++count;
  }
  void authenticate_(const field_type& field, std::size_t level) const {
    const auto& expected = candidate_.at(level);
    if (field.ncomp() != expected.ncomp() || field.layout() != expected.layout() ||
        field.distribution() != expected.distribution() ||
        field.local_rank() != expected.local_rank())
      throw std::invalid_argument("composite spatial input differs from its exact-ranked layout");
  }
  Real norm_(const hierarchy_type& values, const ExecutionLane& lane) const {
    Real local_sum = Real(0);
    for (std::size_t level = 0; level < values.size(); ++level) {
      if (values[level].distribution().replicated() &&
          values[level].local_rank() != values[level].rank_space().coordinate(0))
        continue;
      for (int component = 0; component < values[level].ncomp(); ++component)
        local_sum += measures_[level] *
                     dot_active_local(values[level], values[level], component, masks_[level]);
    }
    return std::sqrt(all_reduce_sum(local_sum, lane));
  }
  std::unique_ptr<AmrFieldNewtonKrylovWorkspace<Dim>> newton_;
  hierarchy_type owned_masks_;
  std::vector<const field_type*> masks_;
  std::vector<Real> measures_;
  hierarchy_type candidate_, previous_, perturbed_, plus_, minus_;
  std::vector<field_type*> destinations_;
  Real difference_step_;
  int residual_evaluations_ = 0, derivative_evaluations_ = 0;
};
}  // namespace pops::runtime::program
