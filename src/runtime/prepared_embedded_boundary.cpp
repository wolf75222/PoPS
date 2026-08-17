/// @file
/// @brief Collective materialization of immutable exact-ranked embedded-boundary geometry.

#include <pops/runtime/system/prepared_embedded_boundary.hpp>

#include <pops/numerics/spatial/embedded_boundary/cut_geometry.hpp>

#include <pops/core/identity/sha256.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/runtime/analytic/collective_preflight.hpp>

#include <Kokkos_Core.hpp>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pops::runtime::system {
namespace {

template <class Function>
[[nodiscard]] auto collective_stage(std::string_view operation, Function&& function,
                                    const CommunicatorView& communicator)
    -> std::invoke_result_t<Function> {
  using Result = std::invoke_result_t<Function>;
  static_assert(!std::is_void_v<Result>);

  std::optional<Result> staged;
  std::exception_ptr local_failure;
  try {
    staged.emplace(std::invoke(std::forward<Function>(function)));
  } catch (...) {
    local_failure = std::current_exception();
  }
  const long failures = all_reduce_sum(local_failure ? 1L : 0L, communicator);
  if (failures != 0) {
    if (communicator.size() == 1 && local_failure)
      std::rethrow_exception(local_failure);
    throw std::runtime_error(std::string(operation) + " failed collectively on " +
                             std::to_string(failures) + " rank(s)");
  }
  return std::move(*staged);
}

inline std::size_t checked_product(std::size_t left, std::size_t right,
                                   std::string_view operation) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    throw std::length_error(std::string(operation));
  return left * right;
}

template <int Dim>
Extent<Dim> unit_ghosts() {
  Extent<Dim> ghosts{};
  for (int axis = 0; axis < Dim; ++axis)
    ghosts[axis] = 1;
  return ghosts;
}

template <int Dim>
Extent<Dim> zero_ghosts() {
  return Extent<Dim>{};
}

template <int Dim>
HaloScheduleBudget exact_halo_budget(const MultiFab<Dim>& field, const Geometry<Dim>& geometry,
                                     const BoundaryTopology<Dim>& topology) {
  const std::size_t boxes = field.layout().size();
  const std::size_t pairs = checked_product(boxes, boxes, "EB halo patch-pair overflow");
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    std::size_t count = 1;
    const Face<Dim> lower{axis, BoundarySide::lower};
    if (topology.is_periodic(lower) && field.ghosts()[axis] > 0) {
      const std::int64_t length = geometry.domain().length(axis);
      if (length <= 0)
        throw std::invalid_argument("EB halo has an empty periodic axis");
      const std::int64_t wraps = 1 + (field.ghosts()[axis] - 1) / length;
      count = 1 + checked_product(2, static_cast<std::size_t>(wraps),
                                  "EB halo periodic-image overflow");
    }
    images = checked_product(images, count, "EB halo periodic-image product overflow");
  }
  const std::size_t work = checked_product(pairs, images, "EB halo work overflow");
  const std::size_t jobs =
      checked_product(work, static_cast<std::size_t>(2 * Dim), "EB halo job overflow");
  const std::int64_t signed_cells = geometry.domain().numPts();
  if (signed_cells <= 0)
    throw std::invalid_argument("EB halo requires a non-empty geometry domain");
  const std::size_t elements =
      checked_product(jobs, static_cast<std::size_t>(signed_cells), "EB halo element overflow");
  return {{boxes, pairs},
          work,
          jobs,
          images,
          checked_product(boxes, std::size_t{2}, "EB halo peer overflow"),
          elements,
          elements,
          elements};
}

inline void append_signed(std::string& payload, std::int64_t value) {
  analytic::detail::append_analytic_u64(payload, std::bit_cast<std::uint64_t>(value));
}

inline void append_real(std::string& payload, Real value) {
  static_assert(sizeof(Real) == sizeof(std::uint64_t));
  static_assert(std::numeric_limits<Real>::is_iec559);
  analytic::detail::append_analytic_u64(payload, std::bit_cast<std::uint64_t>(value));
}

template <int Dim>
std::string canonical_semantic_request(const std::vector<std::string>& opcodes,
                                       const std::vector<double>& literals,
                                       const Geometry<Dim>& geometry,
                                       const BoundaryTopology<Dim>& topology,
                                       PreparedEmbeddedBoundaryMode mode,
                                       const EbThresholds& thresholds) {
  const analytic::AnalyticOpcodeRows opcode_rows{opcodes};
  const analytic::AnalyticLiteralRows literal_rows{literals};
  std::string payload = analytic::detail::canonical_analytic_request(
      "prepare_embedded_boundary_geometry", std::span<const analytic::AnalyticTextMetadata>{},
      std::span<const analytic::AnalyticRealMetadata>{}, opcode_rows, literal_rows);
  analytic::detail::append_analytic_bytes(payload, "pops.prepared-eb-semantic.v1");
  analytic::detail::append_analytic_u64(payload, static_cast<std::uint64_t>(Dim));
  analytic::detail::append_analytic_u64(payload, static_cast<std::uint64_t>(mode));
  append_real(payload, thresholds.kappa_min);
  append_real(payload, thresholds.face_open_eps);
  append_real(payload, thresholds.cut_theta_min);

  for (int axis = 0; axis < Dim; ++axis) {
    append_signed(payload, geometry.domain().lo[axis]);
    append_signed(payload, geometry.domain().hi[axis]);
    append_real(payload, geometry.lower()[axis]);
    append_real(payload, geometry.upper()[axis]);
  }
  for (const auto& face : topology.faces()) {
    analytic::detail::append_analytic_u64(payload, static_cast<std::uint64_t>(face.kind));
    append_signed(payload, face.partner.ordinal());
  }
  return payload;
}

template <int Dim>
std::string canonical_request(std::string_view semantic, const MultiFab<Dim>& prototype,
                              std::uint64_t generation, std::string_view lane_identity) {
  std::string payload;
  analytic::detail::append_analytic_bytes(payload, "pops.prepared-eb-materialization.v1");
  analytic::detail::append_analytic_bytes(payload, semantic);
  analytic::detail::append_analytic_u64(payload, generation);
  analytic::detail::append_analytic_bytes(payload, lane_identity);

  analytic::detail::append_analytic_size(payload, prototype.layout().size());
  for (const Box<Dim>& box : prototype.layout().boxes())
    for (int axis = 0; axis < Dim; ++axis) {
      append_signed(payload, box.lo[axis]);
      append_signed(payload, box.hi[axis]);
    }

  const auto& distribution = prototype.distribution();
  analytic::detail::append_analytic_u64(payload, static_cast<std::uint64_t>(distribution.mode()));
  for (int axis = 0; axis < Dim; ++axis) {
    append_signed(payload, distribution.rank_space().origin()[axis]);
    append_signed(payload, distribution.rank_space().extent()[axis]);
  }
  analytic::detail::append_analytic_size(payload, distribution.owners().size());
  for (const Index<Dim>& owner : distribution.owners())
    for (int axis = 0; axis < Dim; ++axis)
      append_signed(payload, owner[axis]);
  return payload;
}

std::string digest_request(std::string_view canonical) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(canonical.size());
  for (const char value : canonical)
    bytes.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(value)));
  return "pops.prepared-eb-geometry.v1:sha256:" + identity::sha256_hex(bytes);
}

std::string semantic_digest_request(std::string_view canonical) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(canonical.size());
  for (const char value : canonical)
    bytes.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(value)));
  return "pops.prepared-eb-semantic.v1:sha256:" + identity::sha256_hex(bytes);
}

bool finite(Real value) noexcept {
  return std::isfinite(static_cast<double>(value));
}

void validate_thresholds(const EbThresholds& thresholds) {
  if (!finite(thresholds.kappa_min) || !(thresholds.kappa_min > Real(0)) ||
      thresholds.kappa_min > Real(1))
    throw std::invalid_argument("prepared EB kappa_min must be finite and in (0,1]");
  if (!finite(thresholds.face_open_eps) || thresholds.face_open_eps < Real(0) ||
      thresholds.face_open_eps > Real(1))
    throw std::invalid_argument("prepared EB face_open_eps must be finite and in [0,1]");
  if (!finite(thresholds.cut_theta_min) || !(thresholds.cut_theta_min > Real(0)) ||
      thresholds.cut_theta_min > Real(1))
    throw std::invalid_argument("prepared EB cut_theta_min must be finite and in (0,1]");
}

template <int Dim>
void validate_request(const Geometry<Dim>& geometry, const BoundaryTopology<Dim>& topology,
                      const MultiFab<Dim>& prototype, PreparedEmbeddedBoundaryMode mode,
                      const EbThresholds& thresholds, std::uint64_t generation,
                      const ExecutionLane& lane) {
  if (mode != PreparedEmbeddedBoundaryMode::inactive &&
      mode != PreparedEmbeddedBoundaryMode::staircase &&
      mode != PreparedEmbeddedBoundaryMode::cut_cell)
    throw std::invalid_argument("prepared EB mode is invalid");
  validate_thresholds(thresholds);
  if (generation == 0)
    throw std::invalid_argument("prepared EB generation must be non-zero");
  if (geometry.domain().empty() || prototype.layout().empty())
    throw std::invalid_argument("prepared EB requires a non-empty geometry and patch layout");
  if (prototype.distribution().rank_space().size() != static_cast<std::size_t>(lane.size()))
    throw std::invalid_argument("prepared EB rank space does not match the execution lane");
  const std::size_t expected_rank =
      prototype.distribution().rank_space().linear_rank(prototype.local_rank());
  if (expected_rank != static_cast<std::size_t>(lane.rank()))
    throw std::invalid_argument("prepared EB local rank does not match the execution lane");
  for (int axis = 0; axis < Dim; ++axis) {
    const Face<Dim> lower{axis, BoundarySide::lower};
    const Face<Dim> upper{axis, BoundarySide::upper};
    if (topology.is_periodic(lower) != topology.is_periodic(upper))
      throw std::invalid_argument("prepared EB topology has an incomplete periodic axis");
  }
}

template <int Dim>
struct SampleAnalyticKernel {
  analytic::AnalyticLevelSet<Dim> level_set;
  Geometry<Dim> geometry;
  FieldView<Real, Dim> phi;

  POPS_HD void operator()(const Index<Dim>& index) const {
    phi(index) = level_set(geometry.cell_center(index));
  }
};

POPS_HD inline int wrapped_index(int value, int lower, int upper) {
  const int extent = upper - lower + 1;
  int offset = (value - lower) % extent;
  if (offset < 0)
    offset += extent;
  return lower + offset;
}

template <int Dim>
struct SamplePhysicalGhostKernel {
  analytic::AnalyticLevelSet<Dim> level_set;
  Geometry<Dim> geometry;
  BoundaryTopology<Dim> topology;
  FieldView<Real, Dim> phi;

  POPS_HD void operator()(const Index<Dim>& index) const {
    Index<Dim> sampled = index;
    bool crosses_physical_face = false;
    for (int axis = 0; axis < Dim; ++axis) {
      if (index[axis] >= geometry.domain().lo[axis] && index[axis] <= geometry.domain().hi[axis])
        continue;
      const Face<Dim> lower{axis, BoundarySide::lower};
      if (topology.is_periodic(lower)) {
        sampled[axis] =
            wrapped_index(index[axis], geometry.domain().lo[axis], geometry.domain().hi[axis]);
      } else {
        crosses_physical_face = true;
      }
    }
    if (crosses_physical_face)
      phi(index) = level_set(geometry.cell_center(sampled));
  }
};

template <int Dim>
struct NonFiniteIndicator {
  FieldView<const Real, Dim> field;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    return Kokkos::isfinite(field(index)) ? Real(0) : Real(1);
  }
};

template <int Dim>
struct MaterializeMaskKernel {
  FieldView<const Real, Dim> phi;
  FieldView<Real, Dim> mask;

  POPS_HD void operator()(const Index<Dim>& index) const {
    mask(index) = phi(index) < Real(0) ? Real(1) : Real(0);
  }
};

template <int Dim>
struct MaskPhiMismatchKernel {
  FieldView<const Real, Dim> phi;
  FieldView<const Real, Dim> mask;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real expected = phi(index) < Real(0) ? Real(1) : Real(0);
    return expected == mask(index) ? Real(0) : Real(1);
  }
};

template <int Dim>
struct MaterializeMetricKernel {
  FieldView<const Real, Dim> phi;
  FieldView<Real, Dim> kappa;
  FieldView<Real, Dim> inverse_volume;
  FieldView<Real, Dim> face_lower;
  FieldView<Real, Dim> face_upper;
  Real kappa_min = Real(0);
  Real cut_theta_min = Real(0);

  POPS_HD void operator()(const Index<Dim>& index) const {
    const Real center = phi(index);
    if (!(center < Real(0))) {
      kappa(index) = Real(0);
      inverse_volume(index) = Real(0);
      for (int axis = 0; axis < Dim; ++axis) {
        face_lower(index, axis) = Real(0);
        face_upper(index, axis) = Real(0);
      }
      return;
    }
    RealVector<Dim> lower_samples{};
    RealVector<Dim> upper_samples{};
    for (int axis = 0; axis < Dim; ++axis) {
      Index<Dim> lower = index;
      Index<Dim> upper = index;
      --lower[axis];
      ++upper[axis];
      lower_samples[axis] = phi(lower);
      upper_samples[axis] = phi(upper);
    }
    const auto fractions = nd::cut_cell_fractions_from_samples<Dim>(
        center, lower_samples, upper_samples, cut_theta_min);
    kappa(index) = fractions.volume_fraction;
    const Real effective =
        fractions.volume_fraction > kappa_min ? fractions.volume_fraction : kappa_min;
    inverse_volume(index) = Real(1) / effective;
    for (int axis = 0; axis < Dim; ++axis) {
      face_lower(index, axis) = fractions.face_lower[axis];
      face_upper(index, axis) = fractions.face_upper[axis];
    }
  }
};

template <int Dim>
struct ActiveIndicator {
  FieldView<const Real, Dim> mask;

  POPS_HD Real operator()(const Index<Dim>& index) const { return mask(index); }
};

template <int Dim>
struct MetricInvalidIndicator {
  FieldView<const Real, Dim> mask;
  FieldView<const Real, Dim> kappa;
  FieldView<const Real, Dim> inverse_volume;
  FieldView<const Real, Dim> face_lower;
  FieldView<const Real, Dim> face_upper;

  POPS_HD Real operator()(const Index<Dim>& index) const {
    const Real active = mask(index);
    const Real volume = kappa(index);
    const Real inverse = inverse_volume(index);
    const bool valid_active = active == Real(0) || active == Real(1);
    const bool valid_metric = active == Real(0)
                                  ? volume == Real(0) && inverse == Real(0)
                                  : volume > Real(0) && volume <= Real(1) && inverse >= Real(1) &&
                                        Kokkos::isfinite(volume) && Kokkos::isfinite(inverse);
    bool valid_apertures = true;
    for (int axis = 0; axis < Dim; ++axis) {
      const Real lower = face_lower(index, axis);
      const Real upper = face_upper(index, axis);
      if (active == Real(0))
        valid_apertures = valid_apertures && lower == Real(0) && upper == Real(0);
      else
        valid_apertures = valid_apertures && lower >= Real(0) && lower <= Real(1) &&
                          upper >= Real(0) && upper <= Real(1) && Kokkos::isfinite(lower) &&
                          Kokkos::isfinite(upper);
    }
    return valid_active && valid_metric && valid_apertures ? Real(0) : Real(1);
  }
};

template <int Dim>
struct StagedFields {
  MultiFab<Dim> phi;
  MultiFab<Dim> mask;
  MultiFab<Dim> kappa;
  MultiFab<Dim> inverse;
  MultiFab<Dim> face_lower;
  MultiFab<Dim> face_upper;
};

template <int Dim>
StagedFields<Dim> allocate_fields(const MultiFab<Dim>& prototype) {
  return {MultiFab<Dim>(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
                        unit_ghosts<Dim>()),
          MultiFab<Dim>(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
                        unit_ghosts<Dim>()),
          MultiFab<Dim>(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
                        zero_ghosts<Dim>()),
          MultiFab<Dim>(prototype.layout(), prototype.distribution(), prototype.local_rank(), 1,
                        zero_ghosts<Dim>()),
          MultiFab<Dim>(prototype.layout(), prototype.distribution(), prototype.local_rank(), Dim,
                        zero_ghosts<Dim>()),
          MultiFab<Dim>(prototype.layout(), prototype.distribution(), prototype.local_rank(), Dim,
                        zero_ghosts<Dim>())};
}

}  // namespace

PreparedEmbeddedBoundaryMode parse_prepared_embedded_boundary_mode(std::string_view mode) {
  if (mode == "none")
    return PreparedEmbeddedBoundaryMode::inactive;
  if (mode == "staircase")
    return PreparedEmbeddedBoundaryMode::staircase;
  if (mode == "cutcell")
    return PreparedEmbeddedBoundaryMode::cut_cell;
  throw std::invalid_argument("unknown prepared EB mode '" + std::string(mode) +
                              "' (expected none|staircase|cutcell)");
}

std::string_view prepared_embedded_boundary_mode_name(PreparedEmbeddedBoundaryMode mode) noexcept {
  switch (mode) {
    case PreparedEmbeddedBoundaryMode::inactive:
      return "none";
    case PreparedEmbeddedBoundaryMode::staircase:
      return "staircase";
    case PreparedEmbeddedBoundaryMode::cut_cell:
      return "cutcell";
  }
  return "invalid";
}

template <int Dim>
std::shared_ptr<const PreparedEmbeddedBoundaryGeometry<Dim>>
prepare_embedded_boundary_geometry_collectively(
    const std::vector<std::string>& opcodes, const std::vector<double>& literals,
    const Geometry<Dim>& geometry, const BoundaryTopology<Dim>& topology,
    const MultiFab<Dim>& prototype, PreparedEmbeddedBoundaryMode mode,
    const EbThresholds& thresholds, std::uint64_t generation, const ExecutionLane& lane) {
  const CommunicatorView communicator = lane.communicator();
  std::string semantic_canonical;
  std::string canonical;
  analytic::AnalyticProgram program = analytic::collectively_prepare_exact_analytic_request(
      "prepare_embedded_boundary_geometry",
      [&]() {
        validate_request(geometry, topology, prototype, mode, thresholds, generation, lane);
        semantic_canonical =
            canonical_semantic_request(opcodes, literals, geometry, topology, mode, thresholds);
        canonical = canonical_request(semantic_canonical, prototype, generation, lane.identity());
        std::vector<analytic::AnalyticProgram> programs =
            analytic::compile_component_programs({opcodes}, {literals});
        if (programs.size() != 1)
          throw std::logic_error("prepared EB compilation did not produce one scalar program");
        (void)analytic::make_analytic_level_set<Dim>(programs.front());
        return std::move(programs.front());
      },
      [&]() { return canonical; }, communicator);
  const std::string digest = collective_stage(
      "prepared EB digest", [&]() { return digest_request(canonical); }, communicator);
  const std::string semantic_digest = collective_stage(
      "prepared EB semantic digest", [&]() { return semantic_digest_request(semantic_canonical); },
      communicator);

  StagedFields<Dim> fields = collective_stage(
      "prepared EB field allocation", [&]() { return allocate_fields(prototype); }, communicator);
  const analytic::AnalyticLevelSet<Dim> level_set = analytic::make_analytic_level_set<Dim>(program);
  // Sample the complete owned allocation first.  Same-level and periodic halo jobs overwrite the
  // cells for which another patch is authoritative.  Any sparse in-domain ghost with no same-level
  // neighbour retains the exact analytic geometry, which is required while preparing a fine AMR
  // patch before a parent transfer exists.
  for (std::size_t local = 0; local < fields.phi.local_size(); ++local)
    for_each_cell(fields.phi.fab(local).grown_box(),
                  SampleAnalyticKernel<Dim>{level_set, geometry, fields.phi.fab(local).view()});

  HaloSchedule<Dim> schedule = collective_stage(
      "prepared EB halo schedule",
      [&]() {
        const HaloScheduleBudget budget = exact_halo_budget(fields.phi, geometry, topology);
        const HaloLayoutCoverage coverage =
            fields.phi.layout().tiles_exactly(geometry.domain(), budget.layout)
                ? HaloLayoutCoverage::full_domain
                : HaloLayoutCoverage::sparse_level;
        return prepare_halo_schedule(fields.phi, geometry.domain(), topology, coverage, budget);
      },
      communicator);
  const bool distributed = all_reduce_max(schedule.has_remote_jobs() ? 1L : 0L, communicator) != 0;
  if (distributed) {
    fill_boundary(fields.phi, schedule, lane,
                  HaloExchangeContext{generation, generation, ExecutionLane::halo_message_tag});
  } else {
    fill_boundary(fields.phi, schedule);
  }

  for (std::size_t local = 0; local < fields.phi.local_size(); ++local)
    for_each_cell(fields.phi.fab(local).grown_box(),
                  SamplePhysicalGhostKernel<Dim>{level_set, geometry, topology,
                                                 fields.phi.fab(local).view()});

  Real local_non_finite = Real(0);
  for (std::size_t local = 0; local < fields.phi.local_size(); ++local)
    local_non_finite =
        Kokkos::fmax(local_non_finite,
                     for_each_cell_reduce_max(
                         fields.phi.fab(local).grown_box(),
                         NonFiniteIndicator<Dim>{
                             static_cast<const MultiFab<Dim>&>(fields.phi).fab(local).view()}));
  if (all_reduce_max(static_cast<double>(local_non_finite), communicator) != 0.0)
    throw std::domain_error(
        "prepared EB expression produced a non-finite value on the distributed mesh or ghost "
        "layer");

  Real local_active = Real(0);
  Real local_invalid_metric = Real(0);
  for (std::size_t local = 0; local < fields.phi.local_size(); ++local) {
    const auto phi = static_cast<const MultiFab<Dim>&>(fields.phi).fab(local).view();
    for_each_cell(fields.mask.fab(local).grown_box(),
                  MaterializeMaskKernel<Dim>{phi, fields.mask.fab(local).view()});
    for_each_cell(fields.kappa.box(local),
                  MaterializeMetricKernel<Dim>{phi, fields.kappa.fab(local).view(),
                                               fields.inverse.fab(local).view(),
                                               fields.face_lower.fab(local).view(),
                                               fields.face_upper.fab(local).view(),
                                               thresholds.kappa_min, thresholds.cut_theta_min});
    local_active += for_each_cell_reduce_sum(
        fields.mask.box(local),
        ActiveIndicator<Dim>{static_cast<const MultiFab<Dim>&>(fields.mask).fab(local).view()});
    local_invalid_metric =
        Kokkos::fmax(local_invalid_metric,
                     for_each_cell_reduce_max(
                         fields.kappa.box(local),
                         MetricInvalidIndicator<Dim>{
                             static_cast<const MultiFab<Dim>&>(fields.mask).fab(local).view(),
                             static_cast<const MultiFab<Dim>&>(fields.kappa).fab(local).view(),
                             static_cast<const MultiFab<Dim>&>(fields.inverse).fab(local).view(),
                             static_cast<const MultiFab<Dim>&>(fields.face_lower).fab(local).view(),
                             static_cast<const MultiFab<Dim>&>(fields.face_upper)
                                 .fab(local)
                                 .view()}));
  }
  const double global_active = all_reduce_sum(static_cast<double>(local_active), communicator);
  const double global_invalid_metric =
      all_reduce_max(static_cast<double>(local_invalid_metric), communicator);
  if (global_invalid_metric != 0.0)
    throw std::domain_error("prepared EB metric materialization produced an invalid value");
  if (mode != PreparedEmbeddedBoundaryMode::inactive && !(global_active > 0.0))
    throw std::domain_error("active prepared EB geometry contains no cells");

  return collective_stage(
      "prepared EB immutable publication",
      [&]() -> std::shared_ptr<const PreparedEmbeddedBoundaryGeometry<Dim>> {
        return std::shared_ptr<const PreparedEmbeddedBoundaryGeometry<Dim>>(
            new PreparedEmbeddedBoundaryGeometry<Dim>(
                std::move(program), geometry, topology, mode, thresholds, generation,
                semantic_digest, digest, std::move(fields.phi), std::move(fields.mask),
                std::move(fields.kappa), std::move(fields.inverse), std::move(fields.face_lower),
                std::move(fields.face_upper)));
      },
      communicator);
}

template <int Dim>
void replace_prepared_embedded_boundary_geometry_collectively(
    std::shared_ptr<const PreparedEmbeddedBoundaryGeometry<Dim>>& destination,
    const std::vector<std::string>& opcodes, const std::vector<double>& literals,
    const Geometry<Dim>& geometry, const BoundaryTopology<Dim>& topology,
    const MultiFab<Dim>& prototype, PreparedEmbeddedBoundaryMode mode,
    const EbThresholds& thresholds, std::uint64_t generation, const ExecutionLane& lane) {
  auto staged = prepare_embedded_boundary_geometry_collectively(
      opcodes, literals, geometry, topology, prototype, mode, thresholds, generation, lane);
  destination = std::move(staged);
}

template <int Dim>
void fill_prepared_eb_transport_state_ghosts(MultiFab<Dim>& state,
                                             const PreparedEmbeddedBoundaryGeometry<Dim>& embedded,
                                             const ExecutionLane& lane) {
  if (!(state.layout() == embedded.phi().layout()) ||
      !(state.distribution() == embedded.phi().distribution()) ||
      !(state.local_rank() == embedded.phi().local_rank()))
    throw std::invalid_argument("prepared EB transport state layout does not match geometry");
  for (int axis = 0; axis < Dim; ++axis) {
    if (state.ghosts()[axis] < 1)
      throw std::invalid_argument("prepared EB transport state requires one ghost on every axis");
  }

  const HaloScheduleBudget budget =
      exact_halo_budget(state, embedded.geometry(), embedded.topology());
  const HaloLayoutCoverage coverage =
      state.layout().tiles_exactly(embedded.geometry().domain(), budget.layout)
          ? HaloLayoutCoverage::full_domain
          : HaloLayoutCoverage::sparse_level;
  const HaloSchedule<Dim> schedule = prepare_halo_schedule(
      state, embedded.geometry().domain(), embedded.topology(), coverage, budget);
  if (schedule.has_remote_jobs()) {
    fill_boundary(state, schedule, lane,
                  HaloExchangeContext{embedded.generation(), embedded.generation(),
                                      ExecutionLane::halo_message_tag});
  } else {
    fill_boundary(state, schedule);
  }
}

template <int Dim>
void require_prepared_eb_active_mask_matches_phi(
    const PreparedEmbeddedBoundaryGeometry<Dim>& embedded, const ExecutionLane& lane) {
  Real local_mismatch = Real(0);
  for (std::size_t local = 0; local < embedded.phi().local_size(); ++local) {
    local_mismatch = Kokkos::fmax(
        local_mismatch,
        for_each_cell_reduce_max(embedded.phi().fab(local).grown_box(),
                                 MaskPhiMismatchKernel<Dim>{embedded.phi().fab(local).view(),
                                                            embedded.active_mask().fab(local).view()}));
  }
  if (all_reduce_max(static_cast<double>(local_mismatch), lane.communicator()) != 0.0)
    throw std::invalid_argument(
        "prepared EB active_mask ghosts do not match phi after the Cartesian halo");
}

template std::shared_ptr<const PreparedEmbeddedBoundaryGeometry<1>>
prepare_embedded_boundary_geometry_collectively(const std::vector<std::string>&,
                                                const std::vector<double>&, const Geometry<1>&,
                                                const BoundaryTopology<1>&, const MultiFab<1>&,
                                                PreparedEmbeddedBoundaryMode, const EbThresholds&,
                                                std::uint64_t, const ExecutionLane&);
template std::shared_ptr<const PreparedEmbeddedBoundaryGeometry<2>>
prepare_embedded_boundary_geometry_collectively(const std::vector<std::string>&,
                                                const std::vector<double>&, const Geometry<2>&,
                                                const BoundaryTopology<2>&, const MultiFab<2>&,
                                                PreparedEmbeddedBoundaryMode, const EbThresholds&,
                                                std::uint64_t, const ExecutionLane&);
template std::shared_ptr<const PreparedEmbeddedBoundaryGeometry<3>>
prepare_embedded_boundary_geometry_collectively(const std::vector<std::string>&,
                                                const std::vector<double>&, const Geometry<3>&,
                                                const BoundaryTopology<3>&, const MultiFab<3>&,
                                                PreparedEmbeddedBoundaryMode, const EbThresholds&,
                                                std::uint64_t, const ExecutionLane&);

template void replace_prepared_embedded_boundary_geometry_collectively(
    std::shared_ptr<const PreparedEmbeddedBoundaryGeometry<1>>&, const std::vector<std::string>&,
    const std::vector<double>&, const Geometry<1>&, const BoundaryTopology<1>&, const MultiFab<1>&,
    PreparedEmbeddedBoundaryMode, const EbThresholds&, std::uint64_t, const ExecutionLane&);
template void replace_prepared_embedded_boundary_geometry_collectively(
    std::shared_ptr<const PreparedEmbeddedBoundaryGeometry<2>>&, const std::vector<std::string>&,
    const std::vector<double>&, const Geometry<2>&, const BoundaryTopology<2>&, const MultiFab<2>&,
    PreparedEmbeddedBoundaryMode, const EbThresholds&, std::uint64_t, const ExecutionLane&);
template void replace_prepared_embedded_boundary_geometry_collectively(
    std::shared_ptr<const PreparedEmbeddedBoundaryGeometry<3>>&, const std::vector<std::string>&,
    const std::vector<double>&, const Geometry<3>&, const BoundaryTopology<3>&, const MultiFab<3>&,
    PreparedEmbeddedBoundaryMode, const EbThresholds&, std::uint64_t, const ExecutionLane&);

template void fill_prepared_eb_transport_state_ghosts(MultiFab<1>&,
                                                      const PreparedEmbeddedBoundaryGeometry<1>&,
                                                      const ExecutionLane&);
template void fill_prepared_eb_transport_state_ghosts(MultiFab<2>&,
                                                      const PreparedEmbeddedBoundaryGeometry<2>&,
                                                      const ExecutionLane&);
template void fill_prepared_eb_transport_state_ghosts(MultiFab<3>&,
                                                      const PreparedEmbeddedBoundaryGeometry<3>&,
                                                      const ExecutionLane&);

template void require_prepared_eb_active_mask_matches_phi(
    const PreparedEmbeddedBoundaryGeometry<1>&, const ExecutionLane&);
template void require_prepared_eb_active_mask_matches_phi(
    const PreparedEmbeddedBoundaryGeometry<2>&, const ExecutionLane&);
template void require_prepared_eb_active_mask_matches_phi(
    const PreparedEmbeddedBoundaryGeometry<3>&, const ExecutionLane&);

}  // namespace pops::runtime::system
