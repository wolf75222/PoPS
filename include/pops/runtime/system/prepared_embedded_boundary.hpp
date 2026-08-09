/// @file
/// @brief Immutable exact-ranked embedded-boundary geometry prepared outside numerical hot paths.

#pragma once

#include <pops/mesh/geometry/geometry.hpp>
#include <pops/mesh/storage/multifab.hpp>
#include <pops/mesh/topology/boundary_topology.hpp>
#include <pops/parallel/execution_lane.hpp>
#include <pops/runtime/analytic/level_set.hpp>
#include <pops/runtime/numerical_defaults.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops::runtime::system {

/// Routing policy selected by a prepared embedded-boundary geometry.
enum class PreparedEmbeddedBoundaryMode : unsigned char {
  inactive = 0,
  staircase = 1,
  cut_cell = 2,
};

[[nodiscard]] PreparedEmbeddedBoundaryMode parse_prepared_embedded_boundary_mode(
    std::string_view mode);
[[nodiscard]] std::string_view prepared_embedded_boundary_mode_name(
    PreparedEmbeddedBoundaryMode mode) noexcept;

/// Deep-owning immutable geometry package for one compile-time spatial rank.
///
/// The analytic program remains owned so `level_set()` always returns a valid device view. The
/// four exact-ranked fields share one layout/distribution identity. `phi` and `active_mask` retain
/// one ghost in every axis; `volume_fraction` and `inverse_volume_fraction` contain valid cells
/// only. Inactive cells store zero in both metric fields. Active cells store raw kappa and
/// `1/max(kappa,kappa_min)` respectively.
template <int Dim>
class PreparedEmbeddedBoundaryGeometry final {
 public:
  static_assert(Dim >= 1 && Dim <= 3,
                "PreparedEmbeddedBoundaryGeometry supports dimensions 1, 2, and 3");

  static constexpr int dimension = Dim;
  using field_type = MultiFab<Dim>;

  PreparedEmbeddedBoundaryGeometry(const PreparedEmbeddedBoundaryGeometry&) = delete;
  PreparedEmbeddedBoundaryGeometry& operator=(const PreparedEmbeddedBoundaryGeometry&) = delete;
  PreparedEmbeddedBoundaryGeometry(PreparedEmbeddedBoundaryGeometry&&) = delete;
  PreparedEmbeddedBoundaryGeometry& operator=(PreparedEmbeddedBoundaryGeometry&&) = delete;

  [[nodiscard]] analytic::AnalyticLevelSet<Dim> level_set() const {
    return analytic::make_analytic_level_set<Dim>(program_);
  }
  [[nodiscard]] const analytic::AnalyticProgram& program() const noexcept { return program_; }
  [[nodiscard]] const Geometry<Dim>& geometry() const noexcept { return geometry_; }
  [[nodiscard]] const BoundaryTopology<Dim>& topology() const noexcept { return topology_; }
  [[nodiscard]] PreparedEmbeddedBoundaryMode mode() const noexcept { return mode_; }
  [[nodiscard]] const EbThresholds& thresholds() const noexcept { return thresholds_; }
  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
  [[nodiscard]] const std::string& semantic_digest() const noexcept { return semantic_digest_; }
  [[nodiscard]] const std::string& digest() const noexcept { return digest_; }

  [[nodiscard]] const field_type& phi() const noexcept { return phi_; }
  [[nodiscard]] const field_type& active_mask() const noexcept { return active_mask_; }
  [[nodiscard]] const field_type& volume_fraction() const noexcept { return volume_fraction_; }
  [[nodiscard]] const field_type& inverse_volume_fraction() const noexcept {
    return inverse_volume_fraction_;
  }

 private:
  PreparedEmbeddedBoundaryGeometry(analytic::AnalyticProgram program, Geometry<Dim> geometry,
                                   BoundaryTopology<Dim> topology,
                                   PreparedEmbeddedBoundaryMode mode, EbThresholds thresholds,
                                   std::uint64_t generation, std::string semantic_digest,
                                   std::string digest, field_type phi, field_type active_mask,
                                   field_type volume_fraction, field_type inverse_volume_fraction)
      : program_(std::move(program)),
        geometry_(geometry),
        topology_(topology),
        mode_(mode),
        thresholds_(thresholds),
        generation_(generation),
        semantic_digest_(std::move(semantic_digest)),
        digest_(std::move(digest)),
        phi_(std::move(phi)),
        active_mask_(std::move(active_mask)),
        volume_fraction_(std::move(volume_fraction)),
        inverse_volume_fraction_(std::move(inverse_volume_fraction)) {}

  analytic::AnalyticProgram program_;
  Geometry<Dim> geometry_;
  BoundaryTopology<Dim> topology_;
  PreparedEmbeddedBoundaryMode mode_ = PreparedEmbeddedBoundaryMode::inactive;
  EbThresholds thresholds_{};
  std::uint64_t generation_ = 0;
  std::string semantic_digest_;
  std::string digest_;
  field_type phi_;
  field_type active_mask_;
  field_type volume_fraction_;
  field_type inverse_volume_fraction_;

  template <int Rank>
  friend std::shared_ptr<const PreparedEmbeddedBoundaryGeometry<Rank>>
  prepare_embedded_boundary_geometry_collectively(const std::vector<std::string>&,
                                                  const std::vector<double>&, const Geometry<Rank>&,
                                                  const BoundaryTopology<Rank>&,
                                                  const MultiFab<Rank>&,
                                                  PreparedEmbeddedBoundaryMode, const EbThresholds&,
                                                  std::uint64_t, const ExecutionLane&);
};

/// Prepare one exact-ranked EB owner collectively and publish nothing on failure.
///
/// `prototype` supplies the immutable global layout, distribution, and local process coordinate.
/// The expression is sampled over every locally allocated valid and ghost cell. Internal and
/// periodic ghosts with an authoritative same-level source are then overwritten by the exact halo
/// plan. Sparse in-domain ghosts retain the analytic value until a parent transfer is prepared;
/// physical ghosts retain the topology-qualified analytic value.
template <int Dim>
[[nodiscard]] std::shared_ptr<const PreparedEmbeddedBoundaryGeometry<Dim>>
prepare_embedded_boundary_geometry_collectively(
    const std::vector<std::string>& opcodes, const std::vector<double>& literals,
    const Geometry<Dim>& geometry, const BoundaryTopology<Dim>& topology,
    const MultiFab<Dim>& prototype, PreparedEmbeddedBoundaryMode mode,
    const EbThresholds& thresholds, std::uint64_t generation, const ExecutionLane& lane);

/// Strong replacement seam: destination changes only after complete collective preparation.
template <int Dim>
void replace_prepared_embedded_boundary_geometry_collectively(
    std::shared_ptr<const PreparedEmbeddedBoundaryGeometry<Dim>>& destination,
    const std::vector<std::string>& opcodes, const std::vector<double>& literals,
    const Geometry<Dim>& geometry, const BoundaryTopology<Dim>& topology,
    const MultiFab<Dim>& prototype, PreparedEmbeddedBoundaryMode mode,
    const EbThresholds& thresholds, std::uint64_t generation, const ExecutionLane& lane);

}  // namespace pops::runtime::system
