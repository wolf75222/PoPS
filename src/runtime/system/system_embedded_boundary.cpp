/// @file
/// @brief Uniform System authoring facade for prepared embedded-boundary geometry.

#include <pops/runtime/system/prepared_embedded_boundary.hpp>

#include "system_impl.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pops {
namespace {

template <int Dim>
struct CutCellSystemCapability {
  static void require(runtime::system::PreparedEmbeddedBoundaryMode mode) {
    if (mode == runtime::system::PreparedEmbeddedBoundaryMode::cut_cell)
      throw std::invalid_argument(
          "cut-cell transport has no exact native provider for this spatial rank");
  }
};

template <>
struct CutCellSystemCapability<2> {
  static void require(runtime::system::PreparedEmbeddedBoundaryMode) noexcept {}
};

template <int Dim>
struct DiscLevelSetCapability {
  static std::pair<std::vector<std::string>, std::vector<double>> make(double, double, double) {
    throw std::invalid_argument("Disc is an exact rank-two authoring capability");
  }
};

template <>
struct DiscLevelSetCapability<2> {
  static std::pair<std::vector<std::string>, std::vector<double>> make(double cx, double cy,
                                                                       double radius) {
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(radius) || !(radius > 0.0))
      throw std::invalid_argument("disc center and positive radius must be finite");
    return {{"x", "constant", "sub", "y", "constant", "sub", "hypot", "constant", "sub"},
            {0.0, cx, 0.0, 0.0, cy, 0.0, 0.0, radius, 0.0}};
  }
};

EbThresholds resolved_eb_thresholds(double kappa_min, double face_open_eps, double cut_theta_min) {
  EbThresholds result;
  if (kappa_min > 0.0)
    result.kappa_min = static_cast<Real>(kappa_min);
  if (face_open_eps > 0.0)
    result.face_open_eps = static_cast<Real>(face_open_eps);
  if (cut_theta_min > 0.0)
    result.cut_theta_min = static_cast<Real>(cut_theta_min);
  return result;
}

std::uint64_t next_embedded_boundary_generation(std::uint64_t current) {
  if (current == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("System embedded-boundary generation overflow");
  return current + 1;
}

}  // namespace

template <int Dim>
void System<Dim>::set_analytic_level_set(const std::vector<std::string>& opcodes,
                                         const std::vector<double>& literals,
                                         const std::string& mode, double kappa_min,
                                         double face_open_eps, double cut_theta_min) {
  require_assembling(p_->lifecycle_, "set_analytic_level_set");
  const auto prepared_mode = runtime::system::parse_prepared_embedded_boundary_mode(mode);
  CutCellSystemCapability<Dim>::require(prepared_mode);
  const EbThresholds thresholds = resolved_eb_thresholds(kappa_min, face_open_eps, cut_theta_min);
  const std::uint64_t generation =
      next_embedded_boundary_generation(p_->embedded_boundary_generation_);

  std::vector<std::string> staged_opcodes(opcodes);
  std::vector<double> staged_literals(literals);
  if (!p_->embedded_boundary_lane_)
    p_->embedded_boundary_lane_.emplace(
        ExecutionLane::duplicate_world_collectively("pops.system.embedded-boundary"));
  const BoundaryTopology<Dim> topology = BoundaryTopology<Dim>::axis_periodic(p_->periodicity);
  auto prepared = runtime::system::prepare_embedded_boundary_geometry_collectively(
      staged_opcodes, staged_literals, p_->geom, topology, p_->aux, prepared_mode, thresholds,
      generation, *p_->embedded_boundary_lane_);

  p_->embedded_boundary_ = std::move(prepared);
  p_->embedded_boundary_opcodes_ = std::move(staged_opcodes);
  p_->embedded_boundary_literals_ = std::move(staged_literals);
  p_->embedded_boundary_thresholds_ = thresholds;
  p_->embedded_boundary_generation_ = generation;
}

template <int Dim>
void System<Dim>::set_disc_domain(double cx, double cy, double radius, const std::string& mode,
                                  double kappa_min, double face_open_eps, double cut_theta_min) {
  auto [opcodes, literals] = DiscLevelSetCapability<Dim>::make(cx, cy, radius);
  set_analytic_level_set(opcodes, literals, mode, kappa_min, face_open_eps, cut_theta_min);
}

template <int Dim>
void System<Dim>::set_geometry_mode(const std::string& mode) {
  require_assembling(p_->lifecycle_, "set_geometry_mode");
  const auto prepared_mode = runtime::system::parse_prepared_embedded_boundary_mode(mode);
  CutCellSystemCapability<Dim>::require(prepared_mode);
  if (!p_->embedded_boundary_) {
    if (prepared_mode == runtime::system::PreparedEmbeddedBoundaryMode::inactive)
      return;
    throw std::runtime_error("System geometry mode requires a prepared analytic level set");
  }
  const std::uint64_t generation =
      next_embedded_boundary_generation(p_->embedded_boundary_generation_);
  const BoundaryTopology<Dim> topology = BoundaryTopology<Dim>::axis_periodic(p_->periodicity);
  auto prepared = runtime::system::prepare_embedded_boundary_geometry_collectively(
      p_->embedded_boundary_opcodes_, p_->embedded_boundary_literals_, p_->geom, topology, p_->aux,
      prepared_mode, p_->embedded_boundary_thresholds_, generation, *p_->embedded_boundary_lane_);
  p_->embedded_boundary_ = std::move(prepared);
  p_->embedded_boundary_generation_ = generation;
}

template <int Dim>
std::vector<double> System<Dim>::disc_mask() const {
  if (p_->embedded_boundary_)
    return p_->blocks_.copy_comp0(p_->embedded_boundary_->active_mask());
  MultiFab<Dim> active(p_->ba, p_->dm, p_->local_rank, 1, Extent<Dim>{});
  active.set_val(Real(1));
  return p_->blocks_.copy_comp0(active);
}

template void System<kNativeDimension>::set_analytic_level_set(const std::vector<std::string>&,
                                                               const std::vector<double>&,
                                                               const std::string&, double, double,
                                                               double);
template void System<kNativeDimension>::set_disc_domain(double, double, double, const std::string&,
                                                        double, double, double);
template void System<kNativeDimension>::set_geometry_mode(const std::string&);
template std::vector<double> System<kNativeDimension>::disc_mask() const;

}  // namespace pops
