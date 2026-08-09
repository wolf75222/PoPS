/// @file
/// @brief Explicit 2D embedded-boundary capability over the canonical ranked operator.

#pragma once

#include <pops/mesh/geometry/prepared_metric_provider.hpp>
#include <pops/numerics/spatial/operators/masked_operator.hpp>

#include <Kokkos_MathematicalFunctions.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pops::nd {

struct EmbeddedBoundaryCapabilities2D {
  static constexpr int dimension = 2;
  bool centre_sampled_activity = true;
  bool binary_face_aperture = true;
  bool prepared_inverse_volume = true;
};

/// Metric decoration for one prepared cut-cell patch.
///
/// Face closure belongs to `PreparedMaskedCartesianOperator`; this provider changes only the cell
/// measure by the immutable inverse-volume fraction.  It therefore cannot introduce another face
/// index convention or raw-storage contract.
template <class BaseMetric>
  requires PreparedMetricProvider<2, BaseMetric>
class PreparedEmbeddedBoundaryMetric2D {
 public:
  static constexpr int logical_dimension = 2;
  static constexpr int embedding_dimension = BaseMetric::embedding_dimension;
  using PhysicalPoint = typename BaseMetric::PhysicalPoint;
  using Identity = PreparedMetricIdentity<2, embedding_dimension>;

  static constexpr PreparedMetricCapabilities capabilities() { return BaseMetric::capabilities(); }

  POPS_HD Identity identity() const { return base_.identity(); }
  POPS_HD ReferenceCell<2> reference_cell(const Index<2>& index) const {
    return base_.reference_cell(index);
  }
  POPS_HD PhysicalPoint cell_center(const Index<2>& index) const {
    return base_.cell_center(index);
  }
  template <int Axis, MetricFaceSide Side>
  POPS_HD PhysicalPoint face_center(const Index<2>& index) const {
    return base_.template face_center<Axis, Side>(index);
  }
  POPS_HD CoordinateJacobian<2, embedding_dimension> jacobian(const Index<2>& index) const {
    return base_.jacobian(index);
  }
  template <int Axis, MetricFaceSide Side>
  POPS_HD PhysicalPoint oriented_face_area_vector(const Index<2>& index) const {
    return base_.template oriented_face_area_vector<Axis, Side>(index);
  }
  POPS_HD InverseMapResult<2> inverse_map(const PhysicalPoint& physical) const {
    return base_.inverse_map(physical);
  }

  POPS_HD Real cell_measure(const Index<2>& index) const {
    const Real inverse = inverse_volume_fraction_(index);
    if (!(inverse > Real(0)) || !Kokkos::isfinite(inverse))
      return std::numeric_limits<Real>::quiet_NaN();
    return base_.cell_measure(index) / inverse;
  }

  POPS_HD const BaseMetric& base_metric() const { return base_; }
  POPS_HD FieldView<const Real, 2> inverse_volume_fraction() const {
    return inverse_volume_fraction_;
  }

  template <class MemorySpace>
  static PreparedEmbeddedBoundaryMetric2D prepare(
      BaseMetric base, const Fab<2, MemorySpace>& inverse_volume_fraction, const Box<2>& cells) {
    if (inverse_volume_fraction.ncomp() != 1 || !(inverse_volume_fraction.box() == cells))
      throw std::invalid_argument(
          "prepared embedded-boundary inverse-volume field does not match the patch");
    return PreparedEmbeddedBoundaryMetric2D(std::move(base), inverse_volume_fraction.view());
  }

 private:
  POPS_HD PreparedEmbeddedBoundaryMetric2D(BaseMetric base,
                                           FieldView<const Real, 2> inverse_volume_fraction)
      : base_(std::move(base)), inverse_volume_fraction_(inverse_volume_fraction) {}

  BaseMetric base_;
  FieldView<const Real, 2> inverse_volume_fraction_{};
};

/// Prepared centre-sampled EB transport capability.
///
/// This type is intentionally fixed to rank two.  It composes a prepared metric decoration with
/// the generic masked operator; it is not a partial `Dim` implementation and never enters the
/// Cartesian core as an authority.
template <class Model, class BaseMetric, class Reconstruction = NoSlope,
          class NumericalFlux = RusanovFlux,
          ReconstructionVariables Variables = ReconstructionVariables::Conservative>
  requires(ConservationLaw<2, Model> && PreparedMetricProvider<2, BaseMetric> &&
           ReconstructionPolicy<Reconstruction>)
class PreparedEmbeddedBoundaryOperator2D {
 public:
  static constexpr int dimension = 2;
  static constexpr EmbeddedBoundaryCapabilities2D capabilities() { return {}; }

  PreparedEmbeddedBoundaryOperator2D(Model model, BaseMetric metric,
                                     Reconstruction reconstruction = {},
                                     NumericalFlux numerical_flux = {},
                                     Real positivity_floor = Real(0))
      : model_(std::move(model)),
        metric_(std::move(metric)),
        reconstruction_(std::move(reconstruction)),
        numerical_flux_(std::move(numerical_flux)),
        positivity_floor_(positivity_floor) {}

  template <class MemorySpace>
  void assemble_residual(const Fab<2, MemorySpace>& state, const Fab<2, MemorySpace>& active_cells,
                         const Fab<2, MemorySpace>& inverse_volume_fraction,
                         Fab<2, MemorySpace>& residual,
                         BoundaryFaceOmission<2> omission = {}) const
    requires(flux_provider_count<Model> == 0)
  {
    const auto cut_metric = PreparedEmbeddedBoundaryMetric2D<BaseMetric>::prepare(
        metric_, inverse_volume_fraction, state.box());
    PreparedMaskedCartesianOperator<2, Model, decltype(cut_metric), Reconstruction, NumericalFlux,
                                    Variables>
        masked(model_, cut_metric, reconstruction_, numerical_flux_, positivity_floor_);
    masked.assemble_residual(state, active_cells, residual, omission);
  }

  template <class MemorySpace>
  void assemble_residual(const Fab<2, MemorySpace>& state,
                         const Fab<2, MemorySpace>& providers,
                         const Fab<2, MemorySpace>& active_cells,
                         const Fab<2, MemorySpace>& inverse_volume_fraction,
                         Fab<2, MemorySpace>& residual,
                         BoundaryFaceOmission<2> omission = {}) const {
    const auto cut_metric = PreparedEmbeddedBoundaryMetric2D<BaseMetric>::prepare(
        metric_, inverse_volume_fraction, state.box());
    PreparedMaskedCartesianOperator<2, Model, decltype(cut_metric), Reconstruction, NumericalFlux,
                                    Variables>
        masked(model_, cut_metric, reconstruction_, numerical_flux_, positivity_floor_);
    masked.assemble_residual(state, providers, active_cells, residual, omission);
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<2, MemorySpace>& state,
                         const MultiFab<2, MemorySpace>& active_cells,
                         const MultiFab<2, MemorySpace>& inverse_volume_fraction,
                         MultiFab<2, MemorySpace>& residual,
                         BoundaryFaceOmission<2> omission = {}) const
    requires(flux_provider_count<Model> == 0)
  {
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

    MultiFab<2, MemorySpace> candidate(residual.layout(), residual.distribution(),
                                       residual.local_rank(), Model::n_vars, residual.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      assemble_residual(state.fab(local), active_cells.fab(local),
                        inverse_volume_fraction.fab(local), candidate.fab(local), omission);
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local),
                    cartesian_operator_detail::CopyCellField<2>{
                        static_cast<const Fab<2, MemorySpace>&>(candidate.fab(local)).view(),
                        residual.fab(local).view(), Model::n_vars});
    device_fence();
  }

  template <class MemorySpace>
  void assemble_residual(const MultiFab<2, MemorySpace>& state,
                         const MultiFab<2, MemorySpace>& providers,
                         const MultiFab<2, MemorySpace>& active_cells,
                         const MultiFab<2, MemorySpace>& inverse_volume_fraction,
                         MultiFab<2, MemorySpace>& residual,
                         BoundaryFaceOmission<2> omission = {}) const {
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

    MultiFab<2, MemorySpace> candidate(residual.layout(), residual.distribution(),
                                       residual.local_rank(), Model::n_vars, residual.ghosts());
    for (std::size_t local = 0; local < state.local_size(); ++local)
      assemble_residual(state.fab(local), providers.fab(local), active_cells.fab(local),
                        inverse_volume_fraction.fab(local), candidate.fab(local), omission);
    for (std::size_t local = 0; local < state.local_size(); ++local)
      for_each_cell(state.box(local),
                    cartesian_operator_detail::CopyCellField<2>{
                        static_cast<const Fab<2, MemorySpace>&>(candidate.fab(local)).view(),
                        residual.fab(local).view(), Model::n_vars});
    device_fence();
  }

 private:
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
  return PreparedEmbeddedBoundaryOperator2D<Model, BaseMetric, Reconstruction, NumericalFlux,
                                            Variables>(
      std::move(model), std::move(metric), std::move(reconstruction), std::move(numerical_flux),
      positivity_floor);
}

}  // namespace pops::nd
