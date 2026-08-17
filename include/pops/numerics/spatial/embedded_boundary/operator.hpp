/// @file
/// @brief Exact-ranked embedded-boundary transport over the canonical ranked operator.

#pragma once

#include <pops/mesh/geometry/prepared_metric_provider.hpp>
#include <pops/numerics/spatial/operators/masked_operator.hpp>
#include <pops/parallel/execution_lane.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pops::nd {

/// Proof that partitioned state ghosts were filled through the Cartesian FV halo schedule.
/// MultiFab assemble without this token refuses remote owners instead of using stale ghosts.
struct PreparedEbPartitionHalo {
  const ExecutionLane* lane = nullptr;
};

namespace embedded_operator_detail {

template <int Dim, class MemorySpace>
void require_prepared_eb_ghost_contract(const MultiFab<Dim, MemorySpace>& state,
                                        const MultiFab<Dim, MemorySpace>& active_cells,
                                        const MultiFab<Dim, MemorySpace>& inverse_volume_fraction) {
  for (std::size_t local = 0; local < state.local_size(); ++local) {
    const auto& state_ghosts = state.fab(local).ghosts();
    const auto& mask_ghosts = active_cells.fab(local).ghosts();
    const auto& inverse_ghosts = inverse_volume_fraction.fab(local).ghosts();
    for (int axis = 0; axis < Dim; ++axis) {
      if (state_ghosts[axis] < 1)
        throw std::invalid_argument(
            "prepared EB residual requires state ghosts on every axis");
      if (mask_ghosts[axis] < 1)
        throw std::invalid_argument(
            "prepared EB residual requires active_mask ghosts on every axis");
      if (inverse_ghosts[axis] != 0)
        throw std::invalid_argument(
            "prepared EB inverse_volume_fraction is valid-cell-only and must not advertise ghosts");
    }
  }
}

template <int Dim, class MemorySpace>
bool prepared_eb_has_remote_box_owner(const MultiFab<Dim, MemorySpace>& field) {
  if (field.distribution().replicated() || field.layout().empty())
    return false;
  const auto& local = field.local_rank();
  for (std::size_t box = 0; box < field.layout().size(); ++box) {
    if (!(field.distribution().owner(box) == local))
      return true;
  }
  return false;
}

template <int Dim, class MemorySpace>
void require_prepared_eb_assemble_authority(const MultiFab<Dim, MemorySpace>& state,
                                            const MultiFab<Dim, MemorySpace>& active_cells,
                                            const MultiFab<Dim, MemorySpace>& inverse_volume,
                                            bool partition_halo_authorized) {
  require_prepared_eb_ghost_contract(state, active_cells, inverse_volume);
  if (!partition_halo_authorized && prepared_eb_has_remote_box_owner(state))
    throw std::invalid_argument(
        "prepared EB MultiFab assemble refuses partitioned ghosts without halo authority");
}

template <int Dim, class MemorySpace>
void require_prepared_eb_aperture_contract(const MultiFab<Dim, MemorySpace>& state,
                                           const MultiFab<Dim, MemorySpace>& face_aperture_lower,
                                           const MultiFab<Dim, MemorySpace>& face_aperture_upper) {
  masked_operator_detail::require_same_multifab_layout(
      state, face_aperture_lower, "prepared EB state and face-aperture-lower layouts differ");
  masked_operator_detail::require_same_multifab_layout(
      state, face_aperture_upper, "prepared EB state and face-aperture-upper layouts differ");
  if (face_aperture_lower.ncomp() != Dim || face_aperture_upper.ncomp() != Dim)
    throw std::invalid_argument("prepared EB face apertures must have one component per axis");
  for (std::size_t local = 0; local < state.local_size(); ++local) {
    const auto& lower_ghosts = face_aperture_lower.fab(local).ghosts();
    const auto& upper_ghosts = face_aperture_upper.fab(local).ghosts();
    for (int axis = 0; axis < Dim; ++axis) {
      if (lower_ghosts[axis] != 0 || upper_ghosts[axis] != 0)
        throw std::invalid_argument(
            "prepared EB face apertures are valid-cell-only and must not advertise ghosts");
    }
  }
}

}  // namespace embedded_operator_detail


template <int Dim>
struct EmbeddedBoundaryCapabilities {
  static_assert(Dim >= 1 && Dim <= 3);
  static constexpr int dimension = Dim;
  bool centre_sampled_activity = true;
  bool binary_face_aperture = false;
  bool prepared_inverse_volume = true;
};

/// Metric decoration for one prepared cut-cell patch.
///
/// Cell measure uses the immutable inverse-volume fraction from `cut_cell_fractions_from_samples`.
/// Oriented face area is the Cartesian area scaled by the independent `CutCellFractions` face
/// aperture (continuous in [0, 1]).  A null aperture view is treated as aperture 1, which is a
/// valid continuous value, not a binary mask.  Face closure between active/inactive neighbours
/// still belongs to `PreparedMaskedCartesianOperator`.  Dim=3 volume is the ranked
/// cube-triangulation of that same sampled level set.  Polar/Disc stay planar.
template <int Dim, class BaseMetric>
  requires PreparedMetricProvider<Dim, BaseMetric>
class PreparedEmbeddedBoundaryMetric {
 public:
  static constexpr int dimension = Dim;
  static constexpr int logical_dimension = Dim;
  static constexpr int embedding_dimension = BaseMetric::embedding_dimension;
  using PhysicalPoint = typename BaseMetric::PhysicalPoint;
  using Identity = PreparedMetricIdentity<Dim, embedding_dimension>;

  static constexpr PreparedMetricCapabilities capabilities() { return BaseMetric::capabilities(); }

  POPS_HD Identity identity() const { return base_.identity(); }
  POPS_HD ReferenceCell<Dim> reference_cell(const Index<Dim>& index) const {
    return base_.reference_cell(index);
  }
  POPS_HD PhysicalPoint cell_center(const Index<Dim>& index) const {
    return base_.cell_center(index);
  }
  template <int Axis, MetricFaceSide Side>
  POPS_HD PhysicalPoint face_center(const Index<Dim>& index) const {
    return base_.template face_center<Axis, Side>(index);
  }
  POPS_HD CoordinateJacobian<Dim, embedding_dimension> jacobian(const Index<Dim>& index) const {
    return base_.jacobian(index);
  }
  template <int Axis, MetricFaceSide Side>
  POPS_HD Real face_aperture(const Index<Dim>& index) const {
    const FieldView<const Real, Dim>& field =
        Side == MetricFaceSide::Lower ? face_aperture_lower_ : face_aperture_upper_;
    if (field.data == nullptr)
      return Real(1);
    return field(index, Axis);
  }
  template <int Axis, MetricFaceSide Side>
  POPS_HD PhysicalPoint oriented_face_area_vector(const Index<Dim>& index) const {
    auto area = base_.template oriented_face_area_vector<Axis, Side>(index);
    const Real aperture = face_aperture<Axis, Side>(index);
    for (int physical = 0; physical < embedding_dimension; ++physical)
      area[physical] *= aperture;
    return area;
  }
  POPS_HD InverseMapResult<Dim> inverse_map(const PhysicalPoint& physical) const {
    return base_.inverse_map(physical);
  }

  POPS_HD Real cell_measure(const Index<Dim>& index) const {
    const Real inverse = inverse_volume_fraction_(index);
    if (!(inverse > Real(0)) || !Kokkos::isfinite(inverse))
      return std::numeric_limits<Real>::quiet_NaN();
    return base_.cell_measure(index) / inverse;
  }

  POPS_HD const BaseMetric& base_metric() const { return base_; }
  POPS_HD FieldView<const Real, Dim> inverse_volume_fraction() const {
    return inverse_volume_fraction_;
  }
  POPS_HD FieldView<const Real, Dim> face_aperture_lower() const { return face_aperture_lower_; }
  POPS_HD FieldView<const Real, Dim> face_aperture_upper() const { return face_aperture_upper_; }

  template <class MemorySpace>
  static PreparedEmbeddedBoundaryMetric prepare(
      BaseMetric base, const Fab<Dim, MemorySpace>& inverse_volume_fraction, const Box<Dim>& cells) {
    if (inverse_volume_fraction.ncomp() != 1 || !(inverse_volume_fraction.box() == cells))
      throw std::invalid_argument(
          "prepared embedded-boundary inverse-volume field does not match the patch");
    return PreparedEmbeddedBoundaryMetric(std::move(base), inverse_volume_fraction.view(), {}, {});
  }

  template <class MemorySpace>
  static PreparedEmbeddedBoundaryMetric prepare(
      BaseMetric base, const Fab<Dim, MemorySpace>& inverse_volume_fraction,
      const Fab<Dim, MemorySpace>& face_aperture_lower,
      const Fab<Dim, MemorySpace>& face_aperture_upper, const Box<Dim>& cells) {
    if (inverse_volume_fraction.ncomp() != 1 || !(inverse_volume_fraction.box() == cells))
      throw std::invalid_argument(
          "prepared embedded-boundary inverse-volume field does not match the patch");
    if (face_aperture_lower.ncomp() != Dim || face_aperture_upper.ncomp() != Dim ||
        !(face_aperture_lower.box() == cells) || !(face_aperture_upper.box() == cells))
      throw std::invalid_argument(
          "prepared embedded-boundary face apertures do not match the patch");
    return PreparedEmbeddedBoundaryMetric(std::move(base), inverse_volume_fraction.view(),
                                          face_aperture_lower.view(), face_aperture_upper.view());
  }

 private:
  POPS_HD PreparedEmbeddedBoundaryMetric(BaseMetric base,
                                         FieldView<const Real, Dim> inverse_volume_fraction,
                                         FieldView<const Real, Dim> face_aperture_lower,
                                         FieldView<const Real, Dim> face_aperture_upper)
      : base_(std::move(base)),
        inverse_volume_fraction_(inverse_volume_fraction),
        face_aperture_lower_(face_aperture_lower),
        face_aperture_upper_(face_aperture_upper) {}

  BaseMetric base_;
  FieldView<const Real, Dim> inverse_volume_fraction_{};
  FieldView<const Real, Dim> face_aperture_lower_{};
  FieldView<const Real, Dim> face_aperture_upper_{};
};

/// Prepared centre-sampled EB transport capability.
///
/// This type composes a prepared metric decoration with the exact-ranked masked operator.
template <int Dim, class Model, class BaseMetric, class Reconstruction = NoSlope,
          class NumericalFlux = RusanovFlux,
          ReconstructionVariables Variables = ReconstructionVariables::Conservative>
  requires(ConservationLaw<Dim, Model> && PreparedMetricProvider<Dim, BaseMetric> &&
           ReconstructionPolicy<Reconstruction>)
class PreparedEmbeddedBoundaryOperator {
 public:
  static constexpr int dimension = Dim;
  static constexpr EmbeddedBoundaryCapabilities<Dim> capabilities() { return {}; }

  PreparedEmbeddedBoundaryOperator(Model model, BaseMetric metric,
                                   Reconstruction reconstruction = {},
                                   NumericalFlux numerical_flux = {},
                                   Real positivity_floor = Real(0))
      : model_(std::move(model)),
        metric_(std::move(metric)),
        reconstruction_(std::move(reconstruction)),
        numerical_flux_(std::move(numerical_flux)),
        positivity_floor_(positivity_floor) {}

  template <class MemorySpace>
  void assemble_residual(const Fab<Dim, MemorySpace>& state, const Fab<Dim, MemorySpace>& active_cells,
                         const Fab<Dim, MemorySpace>& inverse_volume_fraction,
                         Fab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const
    requires(flux_provider_count<Model> == 0)
  {
    const auto cut_metric = PreparedEmbeddedBoundaryMetric<Dim, BaseMetric>::prepare(
        metric_, inverse_volume_fraction, state.box());
    PreparedMaskedCartesianOperator<Dim, Model, decltype(cut_metric), Reconstruction, NumericalFlux,
                                    Variables>
        masked(model_, cut_metric, reconstruction_, numerical_flux_, positivity_floor_);
    masked.assemble_residual(state, active_cells, residual, omission);
  }

  template <class MemorySpace>
  void assemble_residual(const Fab<Dim, MemorySpace>& state,
                         const Fab<Dim, MemorySpace>& providers,
                         const Fab<Dim, MemorySpace>& active_cells,
                         const Fab<Dim, MemorySpace>& inverse_volume_fraction,
                         Fab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const {
    const auto cut_metric = PreparedEmbeddedBoundaryMetric<Dim, BaseMetric>::prepare(
        metric_, inverse_volume_fraction, state.box());
    PreparedMaskedCartesianOperator<Dim, Model, decltype(cut_metric), Reconstruction, NumericalFlux,
                                    Variables>
        masked(model_, cut_metric, reconstruction_, numerical_flux_, positivity_floor_);
    masked.assemble_residual(state, providers, active_cells, residual, omission);
  }

  template <class MemorySpace, int Count>
  void assemble_residual(const Fab<Dim, MemorySpace>& state,
                         const ProviderStorageView<Dim, Count>& providers,
                         const Fab<Dim, MemorySpace>& active_cells,
                         const Fab<Dim, MemorySpace>& inverse_volume_fraction,
                         Fab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const
    requires(Count == flux_provider_count<Model>)
  {
    const auto cut_metric = PreparedEmbeddedBoundaryMetric<Dim, BaseMetric>::prepare(
        metric_, inverse_volume_fraction, state.box());
    PreparedMaskedCartesianOperator<Dim, Model, decltype(cut_metric), Reconstruction, NumericalFlux,
                                    Variables>
        masked(model_, cut_metric, reconstruction_, numerical_flux_, positivity_floor_);
    masked.assemble_residual(state, providers, active_cells, residual, omission);
  }

  template <class MemorySpace>
  void assemble_residual(const Fab<Dim, MemorySpace>& state, const Fab<Dim, MemorySpace>& active_cells,
                         const Fab<Dim, MemorySpace>& inverse_volume_fraction,
                         const Fab<Dim, MemorySpace>& face_aperture_lower,
                         const Fab<Dim, MemorySpace>& face_aperture_upper,
                         Fab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const
    requires(flux_provider_count<Model> == 0)
  {
    const auto cut_metric = PreparedEmbeddedBoundaryMetric<Dim, BaseMetric>::prepare(
        metric_, inverse_volume_fraction, face_aperture_lower, face_aperture_upper, state.box());
    PreparedMaskedCartesianOperator<Dim, Model, decltype(cut_metric), Reconstruction, NumericalFlux,
                                    Variables>
        masked(model_, cut_metric, reconstruction_, numerical_flux_, positivity_floor_);
    masked.assemble_residual(state, active_cells, residual, omission);
  }

  template <class MemorySpace>
  void assemble_residual(const Fab<Dim, MemorySpace>& state,
                         const Fab<Dim, MemorySpace>& providers,
                         const Fab<Dim, MemorySpace>& active_cells,
                         const Fab<Dim, MemorySpace>& inverse_volume_fraction,
                         const Fab<Dim, MemorySpace>& face_aperture_lower,
                         const Fab<Dim, MemorySpace>& face_aperture_upper,
                         Fab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const {
    const auto cut_metric = PreparedEmbeddedBoundaryMetric<Dim, BaseMetric>::prepare(
        metric_, inverse_volume_fraction, face_aperture_lower, face_aperture_upper, state.box());
    PreparedMaskedCartesianOperator<Dim, Model, decltype(cut_metric), Reconstruction, NumericalFlux,
                                    Variables>
        masked(model_, cut_metric, reconstruction_, numerical_flux_, positivity_floor_);
    masked.assemble_residual(state, providers, active_cells, residual, omission);
  }

  template <class MemorySpace, int Count>
  void assemble_residual(const Fab<Dim, MemorySpace>& state,
                         const ProviderStorageView<Dim, Count>& providers,
                         const Fab<Dim, MemorySpace>& active_cells,
                         const Fab<Dim, MemorySpace>& inverse_volume_fraction,
                         const Fab<Dim, MemorySpace>& face_aperture_lower,
                         const Fab<Dim, MemorySpace>& face_aperture_upper,
                         Fab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const
    requires(Count == flux_provider_count<Model>)
  {
    const auto cut_metric = PreparedEmbeddedBoundaryMetric<Dim, BaseMetric>::prepare(
        metric_, inverse_volume_fraction, face_aperture_lower, face_aperture_upper, state.box());
    PreparedMaskedCartesianOperator<Dim, Model, decltype(cut_metric), Reconstruction, NumericalFlux,
                                    Variables>
        masked(model_, cut_metric, reconstruction_, numerical_flux_, positivity_floor_);
    masked.assemble_residual(state, providers, active_cells, residual, omission);
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         const MultiFab<Dim, MemorySpace>& inverse_volume_fraction,
                         MultiFab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const
    requires(flux_provider_count<Model> == 0)
  {
    assemble_residual(state, active_cells, inverse_volume_fraction, residual, omission, false);
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         const MultiFab<Dim, MemorySpace>& inverse_volume_fraction,
                         MultiFab<Dim, MemorySpace>& residual, BoundaryFaceOmission<Dim> omission,
                         PreparedEbPartitionHalo) const
    requires(flux_provider_count<Model> == 0)
  {
    assemble_residual(state, active_cells, inverse_volume_fraction, residual, omission, true);
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& providers,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         const MultiFab<Dim, MemorySpace>& inverse_volume_fraction,
                         MultiFab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const {
    assemble_residual(state, providers, active_cells, inverse_volume_fraction, residual, omission,
                      false);
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& providers,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         const MultiFab<Dim, MemorySpace>& inverse_volume_fraction,
                         MultiFab<Dim, MemorySpace>& residual, BoundaryFaceOmission<Dim> omission,
                         PreparedEbPartitionHalo) const {
    assemble_residual(state, providers, active_cells, inverse_volume_fraction, residual, omission,
                      true);
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         const MultiFab<Dim, MemorySpace>& inverse_volume_fraction,
                         const MultiFab<Dim, MemorySpace>& face_aperture_lower,
                         const MultiFab<Dim, MemorySpace>& face_aperture_upper,
                         MultiFab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const
    requires(flux_provider_count<Model> == 0)
  {
    assemble_residual(state, active_cells, inverse_volume_fraction, face_aperture_lower,
                      face_aperture_upper, residual, omission, false);
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         const MultiFab<Dim, MemorySpace>& inverse_volume_fraction,
                         const MultiFab<Dim, MemorySpace>& face_aperture_lower,
                         const MultiFab<Dim, MemorySpace>& face_aperture_upper,
                         MultiFab<Dim, MemorySpace>& residual, BoundaryFaceOmission<Dim> omission,
                         PreparedEbPartitionHalo) const
    requires(flux_provider_count<Model> == 0)
  {
    assemble_residual(state, active_cells, inverse_volume_fraction, face_aperture_lower,
                      face_aperture_upper, residual, omission, true);
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& providers,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         const MultiFab<Dim, MemorySpace>& inverse_volume_fraction,
                         const MultiFab<Dim, MemorySpace>& face_aperture_lower,
                         const MultiFab<Dim, MemorySpace>& face_aperture_upper,
                         MultiFab<Dim, MemorySpace>& residual,
                         BoundaryFaceOmission<Dim> omission = {}) const {
    assemble_residual(state, providers, active_cells, inverse_volume_fraction, face_aperture_lower,
                      face_aperture_upper, residual, omission, false);
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& providers,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         const MultiFab<Dim, MemorySpace>& inverse_volume_fraction,
                         const MultiFab<Dim, MemorySpace>& face_aperture_lower,
                         const MultiFab<Dim, MemorySpace>& face_aperture_upper,
                         MultiFab<Dim, MemorySpace>& residual, BoundaryFaceOmission<Dim> omission,
                         PreparedEbPartitionHalo) const {
    assemble_residual(state, providers, active_cells, inverse_volume_fraction, face_aperture_lower,
                      face_aperture_upper, residual, omission, true);
  }

 private:
  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         const MultiFab<Dim, MemorySpace>& inverse_volume_fraction,
                         MultiFab<Dim, MemorySpace>& residual, BoundaryFaceOmission<Dim> omission,
                         bool partition_halo_authorized) const
    requires(flux_provider_count<Model> == 0)
  {
    embedded_operator_detail::require_prepared_eb_assemble_authority(
        state, active_cells, inverse_volume_fraction, partition_halo_authorized);
    masked_operator_detail::require_same_multifab_layout(
        state, active_cells, "prepared EB state and active-mask layouts differ");
    masked_operator_detail::require_same_multifab_layout(
        state, inverse_volume_fraction, "prepared EB state and inverse-volume layouts differ");
    masked_operator_detail::require_same_multifab_layout(
        state, residual, "prepared EB state and residual layouts differ");
    if (active_cells.ncomp() != 1 || inverse_volume_fraction.ncomp() != 1 ||
        state.ncomp() != Model::n_vars || residual.ncomp() != Model::n_vars ||
        state.shares_storage_with(residual))
      throw std::invalid_argument("prepared EB MultiFab components differ or alias");

    MultiFab<Dim, MemorySpace> candidate(residual.layout(), residual.distribution(),
                                         residual.local_rank(), Model::n_vars, residual.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      assemble_residual(state.fab(local), active_cells.fab(local),
                        inverse_volume_fraction.fab(local), candidate.fab(local), omission);
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local),
                    cartesian_operator_detail::CopyCellField<Dim>{
                        static_cast<const Fab<Dim, MemorySpace>&>(candidate.fab(local)).view(),
                        residual.fab(local).view(), Model::n_vars});
    device_fence();
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& providers,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         const MultiFab<Dim, MemorySpace>& inverse_volume_fraction,
                         MultiFab<Dim, MemorySpace>& residual, BoundaryFaceOmission<Dim> omission,
                         bool partition_halo_authorized) const {
    embedded_operator_detail::require_prepared_eb_assemble_authority(
        state, active_cells, inverse_volume_fraction, partition_halo_authorized);
    masked_operator_detail::require_same_multifab_layout(
        state, providers, "prepared EB state and provider layouts differ");
    masked_operator_detail::require_same_multifab_layout(
        state, active_cells, "prepared EB state and active-mask layouts differ");
    masked_operator_detail::require_same_multifab_layout(
        state, inverse_volume_fraction, "prepared EB state and inverse-volume layouts differ");
    masked_operator_detail::require_same_multifab_layout(
        state, residual, "prepared EB state and residual layouts differ");
    if (providers.ncomp() < flux_provider_count<Model> || active_cells.ncomp() != 1 ||
        inverse_volume_fraction.ncomp() != 1 || state.ncomp() != Model::n_vars ||
        residual.ncomp() != Model::n_vars || state.shares_storage_with(residual))
      throw std::invalid_argument("prepared EB MultiFab components differ or alias");

    MultiFab<Dim, MemorySpace> candidate(residual.layout(), residual.distribution(),
                                         residual.local_rank(), Model::n_vars, residual.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      assemble_residual(state.fab(local), providers.fab(local), active_cells.fab(local),
                        inverse_volume_fraction.fab(local), candidate.fab(local), omission);
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local),
                    cartesian_operator_detail::CopyCellField<Dim>{
                        static_cast<const Fab<Dim, MemorySpace>&>(candidate.fab(local)).view(),
                        residual.fab(local).view(), Model::n_vars});
    device_fence();
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         const MultiFab<Dim, MemorySpace>& inverse_volume_fraction,
                         const MultiFab<Dim, MemorySpace>& face_aperture_lower,
                         const MultiFab<Dim, MemorySpace>& face_aperture_upper,
                         MultiFab<Dim, MemorySpace>& residual, BoundaryFaceOmission<Dim> omission,
                         bool partition_halo_authorized) const
    requires(flux_provider_count<Model> == 0)
  {
    embedded_operator_detail::require_prepared_eb_assemble_authority(
        state, active_cells, inverse_volume_fraction, partition_halo_authorized);
    embedded_operator_detail::require_prepared_eb_aperture_contract(state, face_aperture_lower,
                                                                    face_aperture_upper);
    masked_operator_detail::require_same_multifab_layout(
        state, active_cells, "prepared EB state and active-mask layouts differ");
    masked_operator_detail::require_same_multifab_layout(
        state, inverse_volume_fraction, "prepared EB state and inverse-volume layouts differ");
    masked_operator_detail::require_same_multifab_layout(
        state, residual, "prepared EB state and residual layouts differ");
    if (active_cells.ncomp() != 1 || inverse_volume_fraction.ncomp() != 1 ||
        state.ncomp() != Model::n_vars || residual.ncomp() != Model::n_vars ||
        state.shares_storage_with(residual))
      throw std::invalid_argument("prepared EB MultiFab components differ or alias");

    MultiFab<Dim, MemorySpace> candidate(residual.layout(), residual.distribution(),
                                         residual.local_rank(), Model::n_vars, residual.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      assemble_residual(state.fab(local), active_cells.fab(local),
                        inverse_volume_fraction.fab(local), face_aperture_lower.fab(local),
                        face_aperture_upper.fab(local), candidate.fab(local), omission);
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local),
                    cartesian_operator_detail::CopyCellField<Dim>{
                        static_cast<const Fab<Dim, MemorySpace>&>(candidate.fab(local)).view(),
                        residual.fab(local).view(), Model::n_vars});
    device_fence();
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<Dim, MemorySpace>& state,
                         const MultiFab<Dim, MemorySpace>& providers,
                         const MultiFab<Dim, MemorySpace>& active_cells,
                         const MultiFab<Dim, MemorySpace>& inverse_volume_fraction,
                         const MultiFab<Dim, MemorySpace>& face_aperture_lower,
                         const MultiFab<Dim, MemorySpace>& face_aperture_upper,
                         MultiFab<Dim, MemorySpace>& residual, BoundaryFaceOmission<Dim> omission,
                         bool partition_halo_authorized) const {
    embedded_operator_detail::require_prepared_eb_assemble_authority(
        state, active_cells, inverse_volume_fraction, partition_halo_authorized);
    embedded_operator_detail::require_prepared_eb_aperture_contract(state, face_aperture_lower,
                                                                    face_aperture_upper);
    masked_operator_detail::require_same_multifab_layout(
        state, providers, "prepared EB state and provider layouts differ");
    masked_operator_detail::require_same_multifab_layout(
        state, active_cells, "prepared EB state and active-mask layouts differ");
    masked_operator_detail::require_same_multifab_layout(
        state, inverse_volume_fraction, "prepared EB state and inverse-volume layouts differ");
    masked_operator_detail::require_same_multifab_layout(
        state, residual, "prepared EB state and residual layouts differ");
    if (providers.ncomp() < flux_provider_count<Model> || active_cells.ncomp() != 1 ||
        inverse_volume_fraction.ncomp() != 1 || state.ncomp() != Model::n_vars ||
        residual.ncomp() != Model::n_vars || state.shares_storage_with(residual))
      throw std::invalid_argument("prepared EB MultiFab components differ or alias");

    MultiFab<Dim, MemorySpace> candidate(residual.layout(), residual.distribution(),
                                         residual.local_rank(), Model::n_vars, residual.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      assemble_residual(state.fab(local), providers.fab(local), active_cells.fab(local),
                        inverse_volume_fraction.fab(local), face_aperture_lower.fab(local),
                        face_aperture_upper.fab(local), candidate.fab(local), omission);
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local),
                    cartesian_operator_detail::CopyCellField<Dim>{
                        static_cast<const Fab<Dim, MemorySpace>&>(candidate.fab(local)).view(),
                        residual.fab(local).view(), Model::n_vars});
    device_fence();
  }

  Model model_;
  BaseMetric metric_;
  Reconstruction reconstruction_;
  NumericalFlux numerical_flux_;
  Real positivity_floor_ = Real(0);
};

template <class Model, class BaseMetric, class Reconstruction = NoSlope,
          class NumericalFlux = RusanovFlux,
          ReconstructionVariables Variables = ReconstructionVariables::Conservative>
auto prepare_embedded_boundary_operator(Model model, BaseMetric metric,
                                        Reconstruction reconstruction = {},
                                        NumericalFlux numerical_flux = {},
                                        Real positivity_floor = Real(0)) {
  return PreparedEmbeddedBoundaryOperator<Model::dimension, Model, BaseMetric, Reconstruction,
                                          NumericalFlux, Variables>(
      std::move(model), std::move(metric), std::move(reconstruction), std::move(numerical_flux),
      positivity_floor);
}

}  // namespace pops::nd
