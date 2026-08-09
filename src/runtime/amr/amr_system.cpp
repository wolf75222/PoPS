/// @file
/// @brief Exact compile-time-ranked AMR facade over runtime::amr::AmrRuntime<Dim>.

#include <pops/runtime/amr_system.hpp>

#include <pops/amr/hierarchy/amr_hierarchy.hpp>
#include <pops/amr/tagging/berger_rigoutsos.hpp>
#include <pops/amr/transfer/transfer_provider.hpp>
#include <pops/core/foundation/native_dimension.hpp>
#include <pops/core/identity/sha256.hpp>
#include <pops/core/state/aux_names.hpp>
#include <pops/mesh/boundary/fill_boundary.hpp>
#include <pops/mesh/execution/for_each.hpp>
#include <pops/mesh/storage/mf_arith.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_builtins.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_prepare.hpp>
#include <pops/numerics/elliptic/interface/field_boundary_kernel.hpp>
#include <pops/numerics/elliptic/interface/field_nonlinear.hpp>
#include <pops/numerics/elliptic/linear/solve_outcome.hpp>
#include <pops/numerics/elliptic/interface/field_nullspace_provider.hpp>
#include <pops/parallel/comm.hpp>
#include <pops/parallel/solve_report_consensus.hpp>
#include <pops/runtime/amr/amr_runtime.hpp>
#include <pops/runtime/amr/amr_tensor_elliptic.hpp>
#include <pops/runtime/amr/composite_reduction.hpp>
#include <pops/runtime/amr/persistent_tagging_state.hpp>
#include <pops/runtime/analytic/initial_materialization.hpp>
#include <pops/runtime/builders/compiled/generated_amr_system_block.hpp>
#include <pops/runtime/named_field_output.hpp>
#include <pops/runtime/named_field_publication.hpp>
#include <pops/runtime/output_piece_collective.hpp>
#include <pops/runtime/program/amr_program_checkpoint.hpp>
#include <pops/runtime/program/program_runtime_state.hpp>
#include <pops/runtime/program/profiler.hpp>
#include <pops/runtime/system/system_boundary_registry.hpp>
#include <pops/runtime/system/system_lifecycle.hpp>
#include <pops/runtime/system/prepared_field_solver_component.hpp>
#include <pops/runtime/system/prepared_embedded_boundary.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pops {
namespace {

inline void require_amr_assembling(const runtime::system::SystemLifecycle& lifecycle,
                                   const char* operation) {
  if (lifecycle.frozen())
    throw std::runtime_error(std::string("AmrSystem::") + operation +
                             ": composition is frozen after bind");
}

template <int Dim>
void validate_amr_config(const AmrSystemConfig<Dim>& config) {
  config.validate_spatial_domain();
  if (config.coordinate_system != runtime_config_detail::cartesian_coordinate_system<Dim>())
    throw std::invalid_argument(
        "AmrSystem Cartesian core requires a dimension-qualified Cartesian provider");
  if (config.level_count < 1)
    throw std::invalid_argument("AmrSystem level_count must be positive");
  if (config.regrid_every < 0)
    throw std::invalid_argument("AmrSystem regrid_every must be non-negative");

  const std::size_t transitions = static_cast<std::size_t>(config.level_count - 1);
  if (config.transition_ratios.size() != transitions ||
      config.transition_buffers.size() != transitions ||
      config.transition_lookaheads.size() != transitions)
    throw std::invalid_argument(
        "AmrSystem transition tables must contain exactly level_count - 1 ranked rows");
  for (std::size_t transition = 0; transition < transitions; ++transition)
    for (int axis = 0; axis < Dim; ++axis) {
      if (config.transition_ratios[transition][axis] < 2 ||
          config.transition_ratios[transition][axis] > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "AmrSystem transition refinement ratios must be at least two on every axis");
      if (config.transition_buffers[transition][axis] < 0 ||
          config.transition_lookaheads[transition][axis] < 0)
        throw std::invalid_argument(
            "AmrSystem transition buffers and lookaheads must be non-negative");
      const auto reach = static_cast<std::uint64_t>(config.transition_buffers[transition][axis]) +
                         static_cast<std::uint64_t>(config.transition_lookaheads[transition][axis]);
      if (reach > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
          reach > (static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) - 1u) / 2u)
        throw std::overflow_error(
            "AmrSystem transition buffer plus lookahead exceeds the ranked neighborhood budget");
    }
  for (int axis = 0; axis < Dim; ++axis)
    if (config.coarse_max_grid[axis] < 0)
      throw std::invalid_argument("AmrSystem coarse_max_grid must be non-negative");
}

template <int Dim>
mesh::RankSpace<Dim> process_rank_space(const ExecutionLane& lane) {
  Extent<Dim> shape = runtime_config_detail::filled_extent<Dim>(1);
  shape[0] = lane.size();
  return mesh::RankSpace<Dim>(Index<Dim>{}, shape);
}

template <int Dim>
struct AmrCutCellCapability {
  static void validate_provider(const PreparedAmrSystemBlock<Dim>& block) {
    if (!block.cut_cell_provider_identity.empty())
      throw std::invalid_argument(
          "prepared AMR block advertises cut-cell transport outside its exact provider rank");
  }

  static void require(runtime::system::PreparedEmbeddedBoundaryMode mode,
                      const PreparedAmrSystemBlock<Dim>&) {
    if (mode == runtime::system::PreparedEmbeddedBoundaryMode::cut_cell)
      throw std::invalid_argument(
          "AMR cut-cell transport has no exact provider for this spatial rank");
  }
};

template <>
struct AmrCutCellCapability<2> {
  static void validate_provider(const PreparedAmrSystemBlock<2>& block) {
    if (block.cut_cell_provider_identity.empty())
      throw std::invalid_argument(
          "prepared rank-two AMR block requires its exact cut-cell provider identity");
  }

  static void require(runtime::system::PreparedEmbeddedBoundaryMode mode,
                      const PreparedAmrSystemBlock<2>& block) {
    if (mode == runtime::system::PreparedEmbeddedBoundaryMode::cut_cell &&
        block.cut_cell_provider_identity.empty())
      throw std::invalid_argument(
          "AMR cut-cell transport requires an authenticated rank-two provider");
  }
};

template <int Dim>
struct AmrDiscLevelSetCapability {
  static std::pair<std::vector<std::string>, std::vector<double>> make(double, double, double) {
    throw std::invalid_argument("Disc is an exact rank-two AMR authoring capability");
  }
};

template <>
struct AmrDiscLevelSetCapability<2> {
  static std::pair<std::vector<std::string>, std::vector<double>> make(double cx, double cy,
                                                                       double radius) {
    if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(radius) || !(radius > 0.0))
      throw std::invalid_argument("AMR disc center and positive radius must be finite");
    return {{"x", "constant", "sub", "y", "constant", "sub", "hypot", "constant", "sub"},
            {0.0, cx, 0.0, 0.0, cy, 0.0, 0.0, radius, 0.0}};
  }
};

EbThresholds resolved_amr_eb_thresholds(double kappa_min, double face_open_eps,
                                        double cut_theta_min) {
  if (!std::isfinite(kappa_min) || kappa_min < 0.0 || !std::isfinite(face_open_eps) ||
      face_open_eps < 0.0 || !std::isfinite(cut_theta_min) || cut_theta_min < 0.0)
    throw std::invalid_argument(
        "AMR embedded-boundary threshold overrides must be finite and non-negative");
  EbThresholds result;
  if (kappa_min > 0.0)
    result.kappa_min = static_cast<Real>(kappa_min);
  if (face_open_eps > 0.0)
    result.face_open_eps = static_cast<Real>(face_open_eps);
  if (cut_theta_min > 0.0)
    result.cut_theta_min = static_cast<Real>(cut_theta_min);
  return result;
}

std::uint64_t next_amr_eb_generation(std::uint64_t current) {
  if (current == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("AmrSystem embedded-boundary generation overflow");
  return current + 1;
}

std::string prefixed_sha256(std::string_view prefix, std::string_view contract) {
  std::vector<std::uint8_t> bytes;
  bytes.reserve(contract.size());
  for (const char value : contract)
    bytes.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(value)));
  return std::string(prefix) + identity::sha256_hex(bytes);
}

std::vector<int> resolved_composite_levels(std::size_t count, const std::vector<int>& requested) {
  std::vector<int> levels = requested;
  if (levels.empty()) {
    levels.reserve(count);
    for (std::size_t level = 0; level < count; ++level)
      levels.push_back(static_cast<int>(level));
  }
  int previous = -1;
  for (const int level : levels) {
    if (level < 0 || static_cast<std::size_t>(level) >= count || level <= previous)
      throw std::invalid_argument(
          "AMR composite levels must be strictly increasing live hierarchy indices");
    previous = level;
  }
  return levels;
}

runtime::amr::CompositeReductionKind composite_reduction_kind(std::string_view kind) {
  if (kind == "sum")
    return runtime::amr::CompositeReductionKind::Sum;
  if (kind == "abs_sum")
    return runtime::amr::CompositeReductionKind::AbsoluteSum;
  if (kind == "sum_sq")
    return runtime::amr::CompositeReductionKind::SumSquares;
  if (kind == "min")
    return runtime::amr::CompositeReductionKind::Minimum;
  if (kind == "max")
    return runtime::amr::CompositeReductionKind::Maximum;
  if (kind == "abs_max")
    return runtime::amr::CompositeReductionKind::AbsoluteMaximum;
  throw std::invalid_argument("AMR composite reduction kind is unknown: " + std::string(kind));
}

struct PreparedAmrEbAuthoring {
  std::string configuration_contract;
  std::string semantic_digest;
};

template <int Dim>
PreparedAmrEbAuthoring prepare_amr_eb_authoring(const AmrSystemConfig<Dim>& config,
                                                const PreparedAmrSystemBlock<Dim>& block,
                                                const std::vector<std::string>& opcodes,
                                                const std::vector<double>& literals,
                                                runtime::system::PreparedEmbeddedBoundaryMode mode,
                                                const EbThresholds& thresholds,
                                                std::uint64_t generation) {
  PreparedAmrEbAuthoring result;
  AmrCutCellCapability<Dim>::require(mode, block);
  if (mode != runtime::system::PreparedEmbeddedBoundaryMode::inactive &&
      std::any_of(config.periodicity.begin(), config.periodicity.end(),
                  [](bool periodic) { return !periodic; }))
    throw std::invalid_argument(
        "active AMR embedded transport has no EB-qualified physical-boundary provider");
  if (!std::isfinite(thresholds.kappa_min) || !(thresholds.kappa_min > Real(0)) ||
      thresholds.kappa_min > Real(1) || !std::isfinite(thresholds.face_open_eps) ||
      thresholds.face_open_eps < Real(0) || thresholds.face_open_eps > Real(1) ||
      !std::isfinite(thresholds.cut_theta_min) || !(thresholds.cut_theta_min > Real(0)) ||
      thresholds.cut_theta_min > Real(1))
    throw std::invalid_argument("AMR embedded-boundary thresholds are outside their unit ranges");
  std::vector<analytic::AnalyticProgram> programs =
      analytic::compile_component_programs({opcodes}, {literals});
  if (programs.size() != 1)
    throw std::logic_error("AMR embedded-boundary authoring did not compile one scalar program");
  (void)analytic::make_analytic_level_set<Dim>(programs.front());

  ExactContractBuilder semantic_builder;
  semantic_builder.text("pops.amr.embedded-boundary-semantic")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .text(runtime::system::prepared_embedded_boundary_mode_name(mode))
      .scalar(thresholds.kappa_min)
      .scalar(thresholds.face_open_eps)
      .scalar(thresholds.cut_theta_min)
      .text(block.staircase_provider_identity)
      .text(block.cut_cell_provider_identity)
      .scalar(static_cast<std::uint64_t>(opcodes.size()));
  for (const std::string& opcode : opcodes)
    semantic_builder.text(opcode);
  semantic_builder.scalar(static_cast<std::uint64_t>(literals.size()));
  for (const double literal : literals)
    semantic_builder.scalar(literal);
  for (int axis = 0; axis < Dim; ++axis)
    semantic_builder.scalar(config.index_domain().lo[axis])
        .scalar(config.index_domain().hi[axis])
        .scalar(config.lower[axis])
        .scalar(config.upper[axis])
        .scalar(config.periodicity[axis]);
  std::string semantic = std::move(semantic_builder).release();
  result.semantic_digest = prefixed_sha256("pops.prepared-eb-semantic.v1:sha256:", semantic);
  ExactContractBuilder configuration;
  configuration.text("pops.amr.embedded-boundary-configuration")
      .scalar(std::uint32_t{1})
      .bytes(semantic)
      .scalar(generation);
  result.configuration_contract = std::move(configuration).release();
  return result;
}

template <int Dim>
std::int64_t checked_layout_cells(const mesh::BoxArray<Dim>& layout) {
  std::int64_t total = 0;
  for (const Box<Dim>& box : layout.boxes()) {
    const std::int64_t cells = box.numPts();
    if (cells < 1 || cells > std::numeric_limits<std::int64_t>::max() - total)
      throw std::overflow_error("AmrSystem layout cell budget exceeds int64_t");
    total += cells;
  }
  return total;
}

inline std::size_t checked_pair_count(std::size_t count) {
  if (count > 1 && count - 1 > std::numeric_limits<std::size_t>::max() / count)
    throw std::length_error("AmrSystem layout pair budget exceeds size_t");
  return count < 2 ? 0 : count * (count - 1) / 2;
}

inline std::size_t checked_square_count(std::size_t count) {
  if (count != 0 && count > std::numeric_limits<std::size_t>::max() / count)
    throw std::length_error("AmrSystem hierarchy pair budget exceeds size_t");
  return count * count;
}

inline std::size_t checked_size_sum(std::size_t left, std::size_t right, const char* operation) {
  if (right > std::numeric_limits<std::size_t>::max() - left)
    throw std::length_error(operation);
  return left + right;
}

inline std::size_t checked_size_product(std::size_t left, std::size_t right,
                                        const char* operation) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right)
    throw std::length_error(operation);
  return left * right;
}

template <int Dim>
std::size_t checked_cells(const Box<Dim>& box) {
  const std::int64_t cells = box.numPts();
  if (cells < 0 || static_cast<std::uint64_t>(cells) >
                       static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::overflow_error("AmrSystem ranked field exceeds size_t");
  return static_cast<std::size_t>(cells);
}

template <int Dim>
Index<Dim> unflatten(const Box<Dim>& box, std::size_t linear);

template <int Dim>
std::size_t offset(const Index<Dim>& index, const Box<Dim>& box);

template <int Dim>
amr::RefinementRatio<Dim> refinement_ratio(const Extent<Dim>& authored) {
  std::array<int, Dim> values{};
  for (int axis = 0; axis < Dim; ++axis) {
    if (authored[axis] < 1 || authored[axis] > std::numeric_limits<int>::max())
      throw std::invalid_argument("AMR transition ratio exceeds the exact ranked index range");
    values[static_cast<std::size_t>(axis)] = static_cast<int>(authored[axis]);
  }
  return amr::RefinementRatio<Dim>(values);
}

template <int Dim>
amr::tagging::TagMaskBudget exact_tag_mask_budget(const amr::hierarchy::LevelLayout<Dim>& layout,
                                                  const Index<Dim>& rank) {
  const std::vector<std::size_t> local = layout.distribution().local_box_indices(rank);
  std::size_t maximum_patch_cells = 0;
  std::size_t owned_cells = 0;
  for (const std::size_t global : local) {
    const std::size_t cells = checked_cells(layout.patches()[global]);
    maximum_patch_cells = std::max(maximum_patch_cells, cells);
    owned_cells = checked_size_sum(owned_cells, cells, "AMR tag-mask owned cells exceed size_t");
  }
  std::size_t identity_bytes = checked_size_product(layout.patches().size(), sizeof(Box<Dim>),
                                                    "AMR tag-mask layout identity exceeds size_t");
  identity_bytes = checked_size_sum(
      identity_bytes,
      checked_size_product(layout.distribution().owners().size(), sizeof(Index<Dim>),
                           "AMR tag-mask ownership identity exceeds size_t"),
      "AMR tag-mask identity exceeds size_t");
  identity_bytes = checked_size_sum(
      identity_bytes,
      checked_size_product(local.size(), sizeof(amr::tagging::PatchTagIdentity<Dim>),
                           "AMR tag-mask patch identity exceeds size_t"),
      "AMR tag-mask identity exceeds size_t");
  identity_bytes =
      checked_size_sum(identity_bytes, owned_cells, "AMR tag-mask identity exceeds size_t");
  return {layout.patches().size(), local.size(), maximum_patch_cells, owned_cells, owned_cells,
          identity_bytes};
}

template <int Dim>
runtime::amr::PreparedTaggingExecutionBudget exact_tagging_budget(
    const amr::hierarchy::LevelLayout<Dim>& layout, const Index<Dim>& rank) {
  const auto mask = exact_tag_mask_budget(layout, rank);
  const std::size_t consensus =
      layout.distribution().replicated()
          ? checked_size_product(mask.owned_cells, 2, "AMR replicated tag consensus exceeds size_t")
          : 0;
  return {mask, mask.owned_cells, consensus};
}

template <int Dim>
std::vector<char> gather_tag_mask(const amr::tagging::TagMask<Dim>& mask, const Box<Dim>& domain,
                                  const CommunicatorView& communicator) {
  std::vector<char> global(checked_cells(domain), char{0});
  const bool contributes =
      mask.level_identity().distribution_mode != mesh::DistributionMode::replicated ||
      communicator.rank() == 0;
  if (contributes)
    for (const auto& patch : mask.patches())
      for (std::size_t ordinal = 0; ordinal < patch.tags.size(); ++ordinal)
        if (patch.tags[ordinal] != 0)
          global[offset(unflatten(patch.box, ordinal), domain)] = char{1};
  all_reduce_max_inplace(global.data(), global.size(), communicator);
  return global;
}

template <int Dim>
struct SparseFieldImage {
  Box<Dim> domain{};
  int components = 0;
  std::vector<double> values{};
  std::vector<char> populated{};
};

template <int Dim>
SparseFieldImage<Dim> gather_sparse_field(const MultiFab<Dim>& field, const Box<Dim>& domain,
                                          const CommunicatorView& communicator) {
  const std::size_t cells = checked_cells(domain);
  SparseFieldImage<Dim> result{domain, field.ncomp(), {}, std::vector<char>(cells, char{0})};
  result.values.assign(checked_size_product(static_cast<std::size_t>(field.ncomp()), cells,
                                            "AMR transfer image exceeds size_t"),
                       0.0);
  const bool contributes = !field.distribution().replicated() || communicator.rank() == 0;
  if (contributes)
    for (std::size_t local = 0; local < field.local_size(); ++local) {
      const Fab<Dim>& fab = field.fab(local);
      auto host = fab.create_host_mirror();
      fab.copy_to_host(host);
      const Box<Dim>& valid = fab.box();
      const Box<Dim>& grown = fab.grown_box();
      const std::size_t component_stride = checked_cells(grown);
      for (std::size_t ordinal = 0; ordinal < checked_cells(valid); ++ordinal) {
        const Index<Dim> index = unflatten(valid, ordinal);
        const std::size_t global = offset(index, domain);
        result.populated[global] = char{1};
        for (int component = 0; component < field.ncomp(); ++component)
          result.values[static_cast<std::size_t>(component) * cells + global] = static_cast<double>(
              host(static_cast<std::size_t>(component) * component_stride + offset(index, grown)));
      }
    }
  all_reduce_sum_inplace(result.values.data(), result.values.size(), communicator);
  all_reduce_max_inplace(result.populated.data(), result.populated.size(), communicator);
  return result;
}

template <int Dim>
Fab<Dim> gather_transfer_source(const MultiFab<Dim>& field,
                                const amr::hierarchy::LevelLayout<Dim>& layout,
                                const CommunicatorView& communicator, int source_radius) {
  if (source_radius < 0 || source_radius > 1)
    throw std::invalid_argument("AMR prepared transfer requested an unsupported source radius");
  Extent<Dim> required_ghosts{};
  for (int axis = 0; axis < Dim; ++axis) {
    required_ghosts[axis] = source_radius;
    if (field.ghosts()[axis] < required_ghosts[axis])
      throw std::invalid_argument("AMR prepared transfer lacks its exact parent ghost stencil");
  }
  Fab<Dim> dense(layout.domain(), field.ncomp(), required_ghosts);
  const Box<Dim>& dense_box = dense.grown_box();
  const std::size_t dense_cells = checked_cells(dense_box);
  std::vector<double> values(
      checked_size_product(static_cast<std::size_t>(field.ncomp()), dense_cells,
                           "AMR linear transfer source exceeds size_t"),
      0.0);
  std::vector<char> populated(dense_cells, char{0});

  const bool replicated = field.distribution().replicated();
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    const std::size_t global = field.global_index(local);
    if (replicated && communicator.rank() != 0)
      continue;
    const Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t component_stride = checked_cells(grown);
    for (std::size_t ordinal = 0; ordinal < dense_cells; ++ordinal) {
      const Index<Dim> index = unflatten(dense_box, ordinal);
      std::size_t selected = layout.patches().size();
      for (std::size_t patch = 0; patch < layout.patches().size(); ++patch)
        if (layout.patches()[patch].contains(index)) {
          selected = patch;
          break;
        }
      if (selected == layout.patches().size())
        for (std::size_t patch = 0; patch < layout.patches().size(); ++patch) {
          Box<Dim> patch_grown = layout.patches()[patch];
          for (int axis = 0; axis < Dim; ++axis)
            patch_grown = patch_grown.grow(axis, static_cast<int>(field.ghosts()[axis]));
          if (patch_grown.contains(index)) {
            selected = patch;
            break;
          }
        }
      if (selected != global || !grown.contains(index))
        continue;
      populated[ordinal] = char{1};
      for (int component = 0; component < field.ncomp(); ++component)
        values[static_cast<std::size_t>(component) * dense_cells + ordinal] = static_cast<double>(
            host(static_cast<std::size_t>(component) * component_stride + offset(index, grown)));
    }
  }
  all_reduce_sum_inplace(values.data(), values.size(), communicator);
  all_reduce_max_inplace(populated.data(), populated.size(), communicator);
  if (std::any_of(populated.begin(), populated.end(), [](char value) { return value == 0; }))
    throw std::runtime_error(
        "AMR prepared transfer source ghosts were not materialized collectively");

  auto dense_host = dense.create_host_mirror();
  dense.copy_to_host(dense_host);
  for (std::size_t index = 0; index < values.size(); ++index)
    dense_host(index) = static_cast<Real>(values[index]);
  dense.copy_from_host(dense_host);
  return dense;
}

template <int Dim>
MultiFab<Dim> transfer_regridded_state(const MultiFab<Dim>& parent,
                                       const amr::hierarchy::LevelLayout<Dim>& parent_layout,
                                       const amr::hierarchy::LevelLayout<Dim>& child_layout,
                                       const std::optional<SparseFieldImage<Dim>>& previous_child,
                                       const CommunicatorView& communicator,
                                       amr::transfer::TransferKind transfer_kind) {
  MultiFab<Dim> child(child_layout.patches(), child_layout.distribution(), parent.local_rank(),
                      parent.ncomp(), parent.ghosts());
  child.set_val(Real(0));
  const int source_radius = transfer_kind == amr::transfer::TransferKind::ConstantInjection ? 0 : 1;
  Fab<Dim> dense_parent =
      gather_transfer_source(parent, parent_layout, communicator, source_radius);
  const amr::RefinementRatio<Dim>& ratio = child_layout.ratio_from_parent();
  amr::transfer::IndexMapping<Dim> mapping;
  mapping.coarse_origin = parent_layout.domain().lo;
  mapping.fine_origin = child_layout.domain().lo;
  const amr::transfer::ComponentRange components{0, 0, parent.ncomp()};
  const amr::transfer::TransferProvider<Dim, amr::transfer::Centering::Cell> provider(
      transfer_kind);
  for (std::size_t local = 0; local < child.local_size(); ++local) {
    Fab<Dim>& fab = child.fab(local);
    const auto prepared = provider.prepare(std::as_const(dense_parent).view(), fab.view(),
                                           fab.box(), ratio, mapping, components);
    for_each_cell(fab.box(), prepared);
  }
  device_fence();

  const std::size_t old_cells =
      previous_child ? checked_cells(previous_child->domain) : std::size_t{0};
  if (!previous_child)
    return child;
  for (std::size_t local = 0; local < child.local_size(); ++local) {
    Fab<Dim>& fab = child.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t component_stride = checked_cells(grown);
    for (std::size_t ordinal = 0; ordinal < checked_cells(valid); ++ordinal) {
      const Index<Dim> fine = unflatten(valid, ordinal);
      const std::size_t fine_global = offset(fine, child_layout.domain());
      const bool reuse = previous_child->domain == child_layout.domain() &&
                         previous_child->populated[fine_global] != 0;
      if (!reuse)
        continue;
      for (int component = 0; component < parent.ncomp(); ++component) {
        const double value =
            previous_child->values[static_cast<std::size_t>(component) * old_cells + fine_global];
        host(static_cast<std::size_t>(component) * component_stride + offset(fine, grown)) =
            static_cast<Real>(value);
      }
    }
    fab.copy_from_host(host);
  }
  return child;
}

inline std::string direct_amr_state_identity(std::string_view block) {
  return "pops://runtime/amr-direct-state/" + std::to_string(block.size()) + ":" +
         std::string(block);
}

template <int Dim>
bool layout_contains(const amr::hierarchy::LevelLayout<Dim>& layout, const Index<Dim>& index) {
  for (const Box<Dim>& patch : layout.patches().boxes())
    if (patch.contains(index))
      return true;
  return false;
}

template <int Dim>
std::vector<char> child_coverage_on_parent(
    const amr::hierarchy::LevelLayout<Dim>& parent,
    const std::optional<amr::hierarchy::LevelLayout<Dim>>& child) {
  std::vector<char> covered(checked_cells(parent.domain()), char{0});
  if (!child)
    return covered;
  for (const Box<Dim>& fine : child->patches().boxes()) {
    const Box<Dim> footprint = amr::hierarchy::coarsen_box(fine, child->ratio_from_parent());
    for (std::size_t ordinal = 0; ordinal < checked_cells(footprint); ++ordinal)
      covered[offset(unflatten(footprint, ordinal), parent.domain())] = char{1};
  }
  return covered;
}

template <int Dim>
amr::tagging::ClusterOptions<Dim> exact_cluster_options(
    const AmrSystemConfig<Dim>& config, const amr::hierarchy::LevelLayout<Dim>& parent) {
  const std::size_t cells = checked_cells(parent.domain());
  const std::size_t patches = parent.patches().size();
  const std::size_t ranks = parent.distribution().rank_space().size();
  const std::size_t nodes = checked_size_sum(
      checked_size_product(std::max<std::size_t>(cells, 1), static_cast<std::size_t>(2 * Dim),
                           "AMR clustering recursion budget exceeds size_t"),
      1, "AMR clustering recursion budget exceeds size_t");
  const std::size_t visits = checked_size_product(
      std::max<std::size_t>(cells, 1), nodes, "AMR clustering cell-visit budget exceeds size_t");
  if (visits > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::length_error("AMR clustering cell-visit budget exceeds exact signed counters");
  const auto add_identity = [](std::size_t total, std::size_t count, std::size_t width) {
    return checked_size_sum(
        total, checked_size_product(count, width, "AMR clustering identity budget exceeds size_t"),
        "AMR clustering identity budget exceeds size_t");
  };
  std::size_t identity_bytes =
      std::string_view{amr::tagging::BergerRigoutsosProvider<Dim>::kIdentity}.size();
  identity_bytes = add_identity(identity_bytes, patches, sizeof(Box<Dim>));
  identity_bytes =
      add_identity(identity_bytes, parent.distribution().owners().size(), sizeof(Index<Dim>));
  identity_bytes = add_identity(identity_bytes, cells, sizeof(Box<Dim>));
  identity_bytes = add_identity(identity_bytes, ranks, sizeof(amr::tagging::TagShardIdentity<Dim>));
  identity_bytes =
      add_identity(identity_bytes, patches, sizeof(amr::tagging::PatchTagIdentity<Dim>));
  identity_bytes = checked_size_sum(
      identity_bytes, static_cast<std::size_t>(checked_layout_cells(parent.patches())),
      "AMR clustering identity budget exceeds size_t");
  amr::tagging::ClusterOptions<Dim> options;
  options.min_efficiency =
      config.cluster_min_efficiency > 0.0 ? config.cluster_min_efficiency : 0.7;
  const int minimum = config.cluster_min_box_size > 0 ? config.cluster_min_box_size : 1;
  const int maximum = config.cluster_max_box_size > 0 ? config.cluster_max_box_size : 32;
  for (int axis = 0; axis < Dim; ++axis) {
    options.min_box_size[static_cast<std::size_t>(axis)] = minimum;
    options.max_box_size[static_cast<std::size_t>(axis)] = maximum;
  }
  options.budget = {ranks, nodes, visits, std::max<std::size_t>(cells, 1),
                    std::max<std::size_t>(identity_bytes, 1)};
  return options;
}

template <int Dim>
amr::regridding::RegridPreparationBudget exact_regrid_budget(
    const amr::hierarchy::LevelLayout<Dim>& parent, const amr::RefinementRatio<Dim>& ratio,
    const amr::tagging::ClusterResult<Dim>& clustered) {
  std::int64_t fine_cells = 0;
  for (const Box<Dim>& box : clustered.boxes.boxes()) {
    const std::int64_t cells = amr::hierarchy::refine_box(box, ratio).numPts();
    if (cells < 0 || cells > std::numeric_limits<std::int64_t>::max() - fine_cells)
      throw std::length_error("AMR regrid load-balance weight exceeds int64_t");
    fine_cells += cells;
  }
  const std::size_t boxes = clustered.boxes.size();
  const mesh::BoxArrayValidationBudget layout_budget{boxes, checked_pair_count(boxes)};
  return {
      layout_budget, layout_budget, {boxes, parent.distribution().rank_space().size(), fine_cells}};
}

template <int Dim>
std::size_t exact_hierarchy_pair_budget(const AmrSystemConfig<Dim>& config,
                                        std::size_t coarse_patches) {
  Box<Dim> parent_domain = config.index_domain();
  std::size_t parent_patch_bound = coarse_patches;
  std::size_t pairs = 0;
  for (std::size_t transition = 0; transition < static_cast<std::size_t>(config.level_count - 1);
       ++transition) {
    const std::size_t child_patch_bound = checked_cells(parent_domain);
    pairs = checked_size_sum(
        pairs,
        checked_size_product(parent_patch_bound, child_patch_bound,
                             "AMR hierarchy parent/child pair budget exceeds size_t"),
        "AMR hierarchy parent/child pair budget exceeds size_t");
    parent_patch_bound = child_patch_bound;
    parent_domain = amr::hierarchy::refine_box(
        parent_domain, refinement_ratio(config.transition_ratios[transition]));
  }
  return pairs;
}

template <int Dim>
Index<Dim> unflatten(const Box<Dim>& box, std::size_t linear) {
  Index<Dim> index{};
  for (int axis = 0; axis < Dim; ++axis) {
    const std::size_t extent = static_cast<std::size_t>(box.length(axis));
    index[axis] = box.lo[axis] + static_cast<int>(linear % extent);
    linear /= extent;
  }
  return index;
}

template <int Dim>
std::size_t offset(const Index<Dim>& index, const Box<Dim>& box) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    result += static_cast<std::size_t>(index[axis] - box.lo[axis]) * stride;
    stride *= static_cast<std::size_t>(box.length(axis));
  }
  return result;
}

template <int Dim>
std::vector<double> gather_field(const MultiFab<Dim>& field, const Box<Dim>& domain,
                                 int components) {
  if (components < 1 || components > field.ncomp())
    throw std::invalid_argument("AmrSystem gather component count is invalid");
  const std::size_t domain_cells = checked_cells(domain);
  if (static_cast<std::size_t>(components) > std::numeric_limits<std::size_t>::max() / domain_cells)
    throw std::overflow_error("AmrSystem gather buffer exceeds size_t");
  std::vector<double> result(static_cast<std::size_t>(components) * domain_cells, 0.0);
  const bool contributes = !field.distribution().replicated() || my_rank() == 0;
  for (std::size_t local = 0; contributes && local < field.local_size(); ++local) {
    const Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t local_cells = checked_cells(valid);
    const std::size_t component_stride = checked_cells(grown);
    for (int component = 0; component < components; ++component)
      for (std::size_t linear = 0; linear < local_cells; ++linear) {
        const Index<Dim> index = unflatten(valid, linear);
        result[static_cast<std::size_t>(component) * domain_cells + offset(index, domain)] =
            static_cast<double>(host(static_cast<std::size_t>(component) * component_stride +
                                     offset(index, grown)));
      }
  }
  all_reduce_sum_inplace(result.data(), result.size());
  return result;
}

template <int Dim>
void write_field(MultiFab<Dim>& field, const Box<Dim>& domain, const std::vector<double>& values,
                 int components) {
  const std::size_t domain_cells = checked_cells(domain);
  if (components < 1 || components > field.ncomp() ||
      values.size() != static_cast<std::size_t>(components) * domain_cells)
    throw std::invalid_argument("AmrSystem field input differs from its exact ranked shape");
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t local_cells = checked_cells(valid);
    const std::size_t component_stride = checked_cells(grown);
    for (int component = 0; component < components; ++component)
      for (std::size_t linear = 0; linear < local_cells; ++linear) {
        const Index<Dim> index = unflatten(valid, linear);
        host(static_cast<std::size_t>(component) * component_stride + offset(index, grown)) =
            static_cast<Real>(
                values[static_cast<std::size_t>(component) * domain_cells + offset(index, domain)]);
      }
    fab.copy_from_host(host);
  }
}

template <int Dim>
void write_component(MultiFab<Dim>& field, const Box<Dim>& domain,
                     const std::vector<double>& values, int component) {
  const std::size_t domain_cells = checked_cells(domain);
  if (component < 0 || component >= field.ncomp() || values.size() != domain_cells)
    throw std::invalid_argument("AmrSystem component input differs from its exact ranked shape");
  for (std::size_t local = 0; local < field.local_size(); ++local) {
    Fab<Dim>& fab = field.fab(local);
    auto host = fab.create_host_mirror();
    fab.copy_to_host(host);
    const Box<Dim>& valid = fab.box();
    const Box<Dim>& grown = fab.grown_box();
    const std::size_t local_cells = checked_cells(valid);
    const std::size_t component_stride = checked_cells(grown);
    for (std::size_t linear = 0; linear < local_cells; ++linear) {
      const Index<Dim> index = unflatten(valid, linear);
      host(static_cast<std::size_t>(component) * component_stride + offset(index, grown)) =
          static_cast<Real>(values[offset(index, domain)]);
    }
    fab.copy_from_host(host);
  }
}

template <int Dim>
double cell_measure(const AmrSystemConfig<Dim>& config, const Box<Dim>& domain) {
  double measure = 1.0;
  for (int axis = 0; axis < Dim; ++axis)
    measure *= static_cast<double>(config.upper[axis] - config.lower[axis]) /
               static_cast<double>(domain.length(axis));
  return measure;
}

template <int Dim>
std::size_t periodic_image_bound(const Box<Dim>& domain, const Extent<Dim>& ghosts,
                                 const BoundaryTopology<Dim>& topology) {
  std::size_t images = 1;
  for (int axis = 0; axis < Dim; ++axis) {
    std::size_t axis_images = 1;
    if (topology.is_periodic(Face<Dim>{axis, BoundarySide::lower}) && ghosts[axis] > 0) {
      const std::int64_t length = domain.length(axis);
      if (length <= 0)
        throw std::invalid_argument("AMR halo budget requires a non-empty periodic domain");
      const std::int64_t wraps = 1 + (ghosts[axis] - 1) / length;
      if (wraps > static_cast<std::int64_t>((std::numeric_limits<std::size_t>::max() - 1) / 2))
        throw std::length_error("AMR halo periodic-image budget exceeds size_t");
      axis_images = 1 + 2 * static_cast<std::size_t>(wraps);
    }
    images =
        checked_size_product(images, axis_images, "AMR halo periodic-image product exceeds size_t");
  }
  return images;
}

template <int Dim>
std::size_t grown_layout_elements(const mesh::BoxArray<Dim>& layout, const Extent<Dim>& ghosts,
                                  int ncomp) {
  if (ncomp < 1)
    throw std::invalid_argument("AMR halo element budget requires positive components");
  std::size_t cells = 0;
  for (const Box<Dim>& valid : layout.boxes()) {
    Box<Dim> grown = valid;
    for (int axis = 0; axis < Dim; ++axis) {
      if (ghosts[axis] < 0 || ghosts[axis] > std::numeric_limits<int>::max())
        throw std::invalid_argument("AMR halo ghost budget exceeds native coordinates");
      grown = grown.grow(axis, static_cast<int>(ghosts[axis]));
    }
    cells =
        checked_size_sum(cells, checked_cells(grown), "AMR halo grown-cell budget exceeds size_t");
  }
  return checked_size_product(cells, static_cast<std::size_t>(ncomp),
                              "AMR halo element budget exceeds size_t");
}

template <int Dim>
HaloScheduleBudget exact_halo_budget(const MultiFab<Dim>& field, const Box<Dim>& domain,
                                     const BoundaryTopology<Dim>& topology) {
  const std::size_t patches = field.layout().size();
  const std::size_t images = periodic_image_bound(domain, field.ghosts(), topology);
  const std::size_t patch_pairs =
      checked_size_product(patches, patches, "AMR halo patch-pair budget exceeds size_t");
  const std::size_t work =
      checked_size_product(patch_pairs, images, "AMR halo box-image budget exceeds size_t");
  const std::size_t jobs = checked_size_product(work, static_cast<std::size_t>(2 * Dim),
                                                "AMR halo job budget exceeds size_t");
  const std::size_t elements = checked_size_product(
      checked_size_product(grown_layout_elements(field.layout(), field.ghosts(), field.ncomp()),
                           std::max<std::size_t>(patches, 1),
                           "AMR halo patch-element budget exceeds size_t"),
      images, "AMR halo image-element budget exceeds size_t");
  return HaloScheduleBudget{
      mesh::BoxArrayValidationBudget{patches, checked_pair_count(patches)},
      work,
      jobs,
      images,
      field.rank_space().size(),
      elements,
      elements,
      elements,
  };
}

template <int Dim>
runtime::amr::AmrGhostFillBudget exact_amr_ghost_budget(const MultiFab<Dim>& coarse,
                                                        const MultiFab<Dim>& fine,
                                                        const Box<Dim>& coarse_domain,
                                                        const Box<Dim>& fine_domain,
                                                        const BoundaryTopology<Dim>& topology) {
  const std::size_t coarse_patches = coarse.layout().size();
  const std::size_t fine_patches = fine.layout().size();
  const std::size_t ranks = fine.rank_space().size();
  const std::size_t images = periodic_image_bound(fine_domain, fine.ghosts(), topology);
  const std::size_t regions =
      checked_size_product(checked_size_product(fine_patches, static_cast<std::size_t>(2 * Dim),
                                                "AMR coarse/fine region budget exceeds size_t"),
                           images, "AMR coarse/fine periodic-region budget exceeds size_t");
  const std::size_t cross_pairs = checked_size_product(
      coarse_patches, fine_patches, "AMR coarse/fine patch-pair budget exceeds size_t");
  const std::size_t pair_budget =
      std::max({cross_pairs, checked_pair_count(coarse_patches), checked_pair_count(fine_patches)});
  const std::size_t jobs = checked_size_product(
      checked_size_product(std::max<std::size_t>(regions, fine_patches),
                           std::max<std::size_t>(coarse_patches, 1),
                           "AMR coarse/fine job budget exceeds size_t"),
      std::max<std::size_t>(ranks, 1), "AMR coarse/fine rank-job budget exceeds size_t");
  const std::size_t elements = checked_size_product(
      checked_size_product(
          checked_size_product(checked_cells(coarse_domain), static_cast<std::size_t>(fine.ncomp()),
                               "AMR coarse/fine component budget exceeds size_t"),
          std::max<std::size_t>(fine_patches, 1),
          "AMR coarse/fine patch-element budget exceeds size_t"),
      checked_size_product(std::max<std::size_t>(images, 1), std::max<std::size_t>(ranks, 1),
                           "AMR coarse/fine image-rank budget exceeds size_t"),
      "AMR coarse/fine element budget exceeds size_t");

  runtime::amr::AmrGhostFillBudget budget;
  budget.coarse_fine = runtime::amr::CoarseFineGhostScheduleBudget{
      fine_patches, regions, pair_budget, jobs, ranks, elements, elements, elements};
  budget.same_level = exact_halo_budget(fine, fine_domain, topology);
  return budget;
}

template <int Dim>
std::vector<const Real*> field_storage_identity(const MultiFab<Dim>& field) {
  std::vector<const Real*> identity;
  identity.reserve(field.local_size());
  for (std::size_t local = 0; local < field.local_size(); ++local)
    identity.push_back(field.fab(local).view().data);
  return identity;
}

template <int Dim>
bool field_storage_matches(const MultiFab<Dim>& field,
                           const std::vector<const Real*>& identity) noexcept {
  try {
    if (field.local_size() != identity.size())
      return false;
    for (std::size_t local = 0; local < field.local_size(); ++local)
      if (field.fab(local).view().data != identity[local])
        return false;
    return true;
  } catch (...) {
    return false;
  }
}

template <int Dim>
bool same_field_shape(const MultiFab<Dim>& left, const MultiFab<Dim>& right) noexcept {
  return left.layout() == right.layout() && left.distribution() == right.distribution() &&
         left.local_rank() == right.local_rank() && left.local_size() == right.local_size() &&
         left.ncomp() == right.ncomp();
}

template <int Dim>
bool same_field_contract(const MultiFab<Dim>& left, const MultiFab<Dim>& right) noexcept {
  return same_field_shape(left, right) && left.ghosts() == right.ghosts();
}

template <int Dim>
void copy_full_field_in_place(const MultiFab<Dim>& source, MultiFab<Dim>& destination) {
  if (!same_field_contract(source, destination))
    throw std::invalid_argument("AMR full-field copy requires one exact ranked field contract");

  // Complete validation precedes the first device write so a malformed candidate cannot partially
  // mutate storage pinned by a prepared halo provider.
  for (std::size_t local = 0; local < source.local_size(); ++local) {
    const Fab<Dim>& source_fab = source.fab(local);
    const Fab<Dim>& destination_fab = destination.fab(local);
    if (source.global_index(local) != destination.global_index(local) ||
        source_fab.box() != destination_fab.box() ||
        source_fab.grown_box() != destination_fab.grown_box() ||
        source_fab.size() != destination_fab.size())
      throw std::invalid_argument(
          "AMR full-field copy encountered different exact local patch storage");
  }
  for (std::size_t local = 0; local < source.local_size(); ++local)
    Kokkos::deep_copy(destination.fab(local).storage(), source.fab(local).storage());
  Kokkos::fence();
}

template <int Dim>
void restore_exact_field_collectively(std::optional<MultiFab<Dim>>& backup, MultiFab<Dim>& live,
                                      const CommunicatorView& communicator) {
  if (!backup)
    return;
  std::exception_ptr restore_error;
  long restore_failure = 0;
  try {
    copy_full_field_in_place(*backup, live);
  } catch (...) {
    restore_failure = 1;
    restore_error = std::current_exception();
  }
  if (all_reduce_max(restore_failure, communicator) != 0) {
    if (restore_error)
      std::rethrow_exception(restore_error);
    throw std::runtime_error("prepared AMR live-state restoration failed collectively");
  }
}

template <int Dim>
std::optional<MultiFab<Dim>> stage_exact_field_collectively(const MultiFab<Dim>& candidate,
                                                            MultiFab<Dim>& live,
                                                            const CommunicatorView& communicator) {
  const long staged = &candidate == &live ? 0L : 1L;
  if (all_reduce_min(staged, communicator) != all_reduce_max(staged, communicator))
    throw std::invalid_argument("prepared AMR candidate/live selection differs between MPI ranks");
  if (staged == 0)
    return std::nullopt;

  std::optional<MultiFab<Dim>> backup;
  std::exception_ptr preflight_error;
  long preflight_failure = 0;
  try {
    if (!same_field_contract(candidate, live))
      throw std::invalid_argument(
          "prepared AMR candidate differs from its exact live level contract");
    backup.emplace(live);
  } catch (...) {
    preflight_failure = 1;
    preflight_error = std::current_exception();
  }
  if (all_reduce_max(preflight_failure, communicator) != 0) {
    if (preflight_error)
      std::rethrow_exception(preflight_error);
    throw std::runtime_error("prepared AMR candidate staging preflight failed collectively");
  }

  std::exception_ptr staging_error;
  long staging_failure = 0;
  try {
    copy_full_field_in_place(candidate, live);
  } catch (...) {
    staging_failure = 1;
    staging_error = std::current_exception();
  }
  if (all_reduce_max(staging_failure, communicator) != 0) {
    restore_exact_field_collectively(backup, live, communicator);
    if (staging_error)
      std::rethrow_exception(staging_error);
    throw std::runtime_error("prepared AMR candidate staging failed collectively");
  }
  return backup;
}

template <int Dim>
void copy_valid_field(const MultiFab<Dim>& source, MultiFab<Dim>& destination) {
  if (!same_field_shape(source, destination))
    throw std::invalid_argument("AMR auxiliary copy requires one exact ranked field shape");
  for (std::size_t local = 0; local < source.local_size(); ++local) {
    const Fab<Dim>& source_fab = source.fab(local);
    Fab<Dim>& destination_fab = destination.fab(local);
    auto source_host = source_fab.create_host_mirror();
    auto destination_host = destination_fab.create_host_mirror();
    source_fab.copy_to_host(source_host);
    destination_fab.copy_to_host(destination_host);
    const Box<Dim>& valid = source_fab.box();
    const Box<Dim>& source_storage = source_fab.grown_box();
    const Box<Dim>& destination_storage = destination_fab.grown_box();
    const std::size_t source_stride = checked_cells(source_storage);
    const std::size_t destination_stride = checked_cells(destination_storage);
    const std::size_t cells = checked_cells(valid);
    for (int component = 0; component < source.ncomp(); ++component)
      for (std::size_t linear = 0; linear < cells; ++linear) {
        const Index<Dim> index = unflatten(valid, linear);
        destination_host(static_cast<std::size_t>(component) * destination_stride +
                         offset(index, destination_storage)) =
            source_host(static_cast<std::size_t>(component) * source_stride +
                        offset(index, source_storage));
      }
    destination_fab.copy_from_host(destination_host);
  }
}

template <int Dim>
std::string exact_root_ghost_contract(const HaloSchedule<Dim>& schedule,
                                      std::string_view field_identity,
                                      std::uint64_t topology_generation,
                                      std::uint64_t materialization_generation,
                                      std::string_view lane_identity) {
  ExactContractBuilder contract;
  contract.text("pops.generated-amr-root-ghost-fill")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .text(field_identity)
      .text(lane_identity)
      .scalar(topology_generation)
      .scalar(materialization_generation)
      .scalar(schedule.coverage())
      .scalar(std::int32_t{schedule.ncomp()});
  for (int axis = 0; axis < Dim; ++axis)
    contract.scalar(std::int64_t{schedule.domain().lo[axis]})
        .scalar(std::int64_t{schedule.domain().hi[axis]})
        .scalar(std::int64_t{schedule.ghosts()[axis]})
        .scalar(schedule.topology().kind(Face<Dim>{axis, BoundarySide::lower}))
        .scalar(schedule.topology().kind(Face<Dim>{axis, BoundarySide::upper}));
  contract.sequence(schedule.layout().boxes(), [](ExactContractBuilder& item, const Box<Dim>& box) {
    for (int axis = 0; axis < Dim; ++axis)
      item.scalar(std::int64_t{box.lo[axis]}).scalar(std::int64_t{box.hi[axis]});
  });
  contract.sequence(schedule.distribution().owners(),
                    [](ExactContractBuilder& item, const Index<Dim>& owner) {
                      for (int axis = 0; axis < Dim; ++axis)
                        item.scalar(std::int64_t{owner[axis]});
                    });
  contract.sequence(schedule.canonical_jobs(), [](ExactContractBuilder& item, const auto& job) {
    item.scalar(static_cast<std::uint64_t>(job.source_box))
        .scalar(static_cast<std::uint64_t>(job.destination_box))
        .scalar(static_cast<std::uint64_t>(job.elements));
    for (int axis = 0; axis < Dim; ++axis)
      item.scalar(std::int64_t{job.destination_region.lo[axis]})
          .scalar(std::int64_t{job.destination_region.hi[axis]})
          .scalar(std::int64_t{job.source_from_destination[axis]});
  });
  return std::move(contract).release();
}

template <int Dim>
struct GeneratedRootGhostSource {
  using field_type = MultiFab<Dim>;

  struct State {
    std::optional<HaloSchedule<Dim>> schedule;
    const ExecutionLane* lane = nullptr;
    std::optional<ExecutionLane::ImmutableBorrow> lane_borrow;
    std::optional<HaloExchange<Dim>> exchange;
    std::vector<const Real*> storage;
    std::string contract;

    void execute(field_type& field, const runtime::multiblock::BoundaryEvaluationPoint& point) {
      const long binding_invalid = !schedule || lane == nullptr ? 1L : 0L;
      if (all_reduce_max(binding_invalid) != 0)
        throw std::logic_error("prepared AMR root ghost provider lost its immutable binding");
      const long invalid = point.level != 0 || !field_storage_matches(field, storage) ? 1L : 0L;
      if (all_reduce_max(invalid, lane->communicator()) != 0)
        throw std::invalid_argument(
            "prepared AMR root ghost provider received stale storage or a non-root level");
      if (exchange)
        fill_boundary(field, *exchange, *lane);
      else
        fill_boundary(field, *schedule);
    }
  };

  std::shared_ptr<State> state;

  [[nodiscard]] static PreparedProviderIdentity provider_identity() noexcept {
    return {"pops.generated.amr.root-ghost-fill", 1};
  }

  void serialize_exact_parameters(ExactContractBuilder& contract) const {
    if (!state)
      throw std::logic_error("generated AMR root ghost source is empty");
    contract.bytes(state->contract);
  }

  void operator()(field_type& field,
                  const runtime::multiblock::BoundaryEvaluationPoint& point) const {
    if (!state)
      throw std::logic_error("generated AMR root ghost source is empty");
    state->execute(field, point);
  }
};

inline std::uint64_t exchange_generation(std::uint64_t generation, const char* label) {
  if (generation == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error(std::string(label) + " cannot form a halo exchange generation");
  return generation + 1;
}

template <int Dim>
PreparedRootAmrGhostFill<Dim> prepare_root_ghost_fill(MultiFab<Dim>& field, const Box<Dim>& domain,
                                                      const BoundaryTopology<Dim>& topology,
                                                      std::string field_identity,
                                                      std::uint64_t topology_generation,
                                                      std::uint64_t materialization_generation,
                                                      const ExecutionLane& lane) {
  using source_type = GeneratedRootGhostSource<Dim>;
  std::shared_ptr<typename source_type::State> state;
  std::exception_ptr metadata_error;
  long metadata_failure = 0;
  try {
    state = std::make_shared<typename source_type::State>();
    state->schedule.emplace(prepare_halo_schedule(field, domain, topology,
                                                  HaloLayoutCoverage::full_domain,
                                                  exact_halo_budget(field, domain, topology)));
    state->lane = &lane;
    state->storage = field_storage_identity(field);
    state->contract =
        exact_root_ghost_contract(*state->schedule, field_identity, topology_generation,
                                  materialization_generation, lane.identity());
  } catch (...) {
    metadata_failure = 1;
    metadata_error = std::current_exception();
  }
  if (all_reduce_max(metadata_failure, lane.communicator()) != 0) {
    if (metadata_error)
      std::rethrow_exception(metadata_error);
    throw std::runtime_error("generated AMR root ghost metadata failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("generated-amr-root-ghost"), std::string_view(state->contract)}},
          lane.communicator()))
    throw std::invalid_argument("generated AMR root ghost contracts differ between ranks");

  const long remote_any =
      all_reduce_max(state->schedule->has_remote_jobs() ? 1L : 0L, lane.communicator());
  std::exception_ptr transport_error;
  long transport_failure = 0;
  try {
    if (remote_any != 0) {
      state->exchange.emplace(
          *state->schedule, lane,
          HaloExchangeContext{
              .context_generation = exchange_generation(topology_generation, "topology"),
              .schedule_generation =
                  exchange_generation(materialization_generation, "materialization"),
          });
    } else {
      state->lane_borrow.emplace(lane.borrow_immutably());
    }
  } catch (...) {
    transport_failure = 1;
    transport_error = std::current_exception();
  }
  if (all_reduce_max(transport_failure, lane.communicator()) != 0) {
    if (transport_error)
      std::rethrow_exception(transport_error);
    throw std::runtime_error("generated AMR root ghost transport failed collectively");
  }
  std::optional<PreparedRootAmrGhostFill<Dim>> prepared;
  std::exception_ptr provider_error;
  long provider_failure = 0;
  try {
    prepared.emplace(source_type{std::move(state)});
  } catch (...) {
    provider_failure = 1;
    provider_error = std::current_exception();
  }
  if (all_reduce_max(provider_failure, lane.communicator()) != 0) {
    if (provider_error)
      std::rethrow_exception(provider_error);
    throw std::runtime_error("generated AMR root ghost provider failed collectively");
  }
  return std::move(*prepared);
}

void validate_variable_set(const VariableSet& variables, VariableKind expected, int ncomp,
                           const char* label) {
  if (variables.kind != expected || variables.size != ncomp ||
      variables.names.size() != static_cast<std::size_t>(ncomp) ||
      variables.roles.size() != static_cast<std::size_t>(ncomp) ||
      (!variables.user_roles.empty() &&
       variables.user_roles.size() != static_cast<std::size_t>(ncomp)))
    throw std::invalid_argument(std::string("prepared AMR block ") + label +
                                " variable metadata differs from its component count");
  std::set<std::string> names;
  for (const std::string& name : variables.names)
    if (name.empty() || !names.insert(name).second)
      throw std::invalid_argument(std::string("prepared AMR block ") + label +
                                  " variable names must be unique and non-empty");
}

template <int Dim>
void validate_prepared_amr_block(const PreparedAmrSystemBlock<Dim>& block) {
  if (block.name.empty() || block.provider_identity.empty() ||
      block.staircase_provider_identity.empty() || block.collective_contract.empty())
    throw std::invalid_argument(
        "prepared AMR block requires non-empty block, Cartesian, staircase, and collective "
        "identities");
  AmrCutCellCapability<Dim>::validate_provider(block);
  if (block.ncomp < 1 || block.aux_components < 1)
    throw std::invalid_argument(
        "prepared AMR block requires positive state and auxiliary component counts");
  if (!std::isfinite(block.gamma) || !(block.gamma > 0.0) || block.substeps < 1 ||
      block.stride < 1 || block.time_route.empty())
    throw std::invalid_argument("prepared AMR block gamma, cadence, or time route is invalid");
  for (int axis = 0; axis < Dim; ++axis)
    if (block.ghosts[axis] < 1)
      throw std::invalid_argument(
          "prepared AMR block must declare a positive ghost extent on every native axis");
  validate_variable_set(block.conservative_variables, VariableKind::Conservative, block.ncomp,
                        "conservative");
  validate_variable_set(block.primitive_variables, VariableKind::Primitive, block.ncomp,
                        "primitive");
  if (!block.materialize_level || !block.primitive_to_conservative ||
      !block.conservative_to_primitive || !block.batch_conservative_to_primitive)
    throw std::invalid_argument(
        "prepared AMR block does not implement its complete exact-ranked execution contract");
}

template <int Dim>
std::string exact_hyperbolic_boundary_contract(const PreparedHyperbolicBoundary<Dim>& boundary) {
  ExactContractBuilder contract;
  contract.text("pops.prepared-hyperbolic-boundary")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .scalar(std::int32_t{boundary.ncomp()})
      .scalar(boundary.corner_policy());
  for (int component = 0; component < boundary.ncomp(); ++component) {
    const auto& transform = boundary.component_transform(component);
    contract.scalar(transform.parity).scalar(std::int32_t{transform.axis});
  }
  for (int axis = 0; axis < Dim; ++axis) {
    for (const int side : {-1, 1}) {
      const PreparedHyperbolicFace& face = boundary.face(axis, side);
      contract.scalar(std::int32_t{axis})
          .scalar(std::int32_t{side})
          .scalar(face.law)
          .text(face.identity)
          .scalar(face.identity_token)
          .scalar(face.authored_representation)
          .text(face.converter_identity)
          .scalar(face.fixed_state_converted)
          .text(face.analytic_clock)
          .scalar(static_cast<std::uint64_t>(face.fixed_state.size()));
      for (const Real value : face.fixed_state)
        contract.scalar(value);
      contract.scalar(static_cast<std::uint64_t>(face.analytic_state.size()));
      for (const analytic::AnalyticProgram& program : face.analytic_state) {
        const analytic::AnalyticProgramView view = program.view();
        contract.scalar(static_cast<std::uint64_t>(program.instruction_count()))
            .scalar(static_cast<std::uint64_t>(program.literal_count()))
            .scalar(static_cast<std::uint64_t>(program.required_stack()))
            .scalar(std::uint8_t{static_cast<std::uint8_t>(program.required_dimension())})
            .scalar(program.result_type());
        for (std::size_t index = 0; index < program.instruction_count(); ++index)
          contract.scalar(view.instructions[index].op).scalar(view.instructions[index].operand);
        for (std::size_t index = 0; index < program.literal_count(); ++index)
          contract.scalar(view.literals[index]);
      }
    }
  }
  return std::move(contract).release();
}

}  // namespace

template <int Dim>
struct AmrSystem<Dim>::Impl {
  using engine_type = runtime::amr::AmrRuntime<Dim>;
  using field_type = MultiFab<Dim>;
  using boundary_registry_type = runtime::system::SystemBoundaryRegistry<Dim>;
  using prepared_block_type = PreparedAmrSystemBlock<Dim>;
  using level_block_type = PreparedGeneratedAmrLevelBlock<Dim>;
  using evaluation_type = PreparedAmrLevelEvaluation<Dim>;
  using exact_field_solver_type = runtime::amr::ExactAmrFieldSolver<Dim>;
  using exact_field_provider_type = runtime::amr::ExactAmrFieldSolverProvider<Dim>;
  using exact_field_registry_type = runtime::amr::ExactAmrFieldSolverRegistry<Dim>;
  using hierarchy_tensor_provider_type = runtime::program::HierarchyTensorSolverProvider<Dim>;
  using hierarchy_tensor_registry_type =
      runtime::program::HierarchyTensorSolverProviderRegistry<Dim>;

  struct BlockSpec {
    std::string name;
    int ncomp = 0;
    double gamma = static_cast<double>(kPhysicalDefaultGamma);
    int substeps = 1;
    int stride = 1;
    int required_ghost_depth = 1;
    Extent<Dim> ghosts{};
    std::string time = "euler";
    bool has_density = false;
    std::vector<double> density;
    bool has_state = false;
    std::vector<double> state;
  };

  struct GlobalDtBound {
    std::string label;
    std::function<double()> evaluate;
  };

  struct BootstrapTransferRoute {
    std::string identity;
    std::vector<std::string> subjects;
    std::string provider_identity;
    std::string space;
    std::string centering;
    std::string representation;
    std::string storage;
    std::string operation;
    std::string kernel;
    int order = 0;
    Extent<Dim> ghost_depth{};
    Extent<Dim> refinement_ratio{};
  };

  struct FieldProviderBinding {
    std::string identity;
    std::string block;
    std::string key;
    double coefficient = 0.0;
  };

  struct PreparedFieldRhs {
    std::function<void(const field_type&, field_type&)> evaluate;
    Real coefficient = Real(1);
  };

  struct TaggingSpec {
    std::vector<std::string> leaf_subject_kinds;
    std::vector<std::string> leaf_subject_identities;
    std::vector<std::string> leaf_blocks;
    std::vector<std::string> leaf_variables;
    std::vector<int> leaf_field_component_indices;
    std::vector<int> leaf_ops;
    std::vector<double> leaf_thresholds;
    std::vector<int> leaf_stencil_indices;
    std::vector<typename runtime::amr::PreparedTaggingProgram<Dim>::Stencil> stencils;
    std::vector<std::int32_t> refine_ops;
    std::vector<std::int32_t> refine_args;
    std::vector<std::int32_t> coarsen_ops;
    std::vector<std::int32_t> coarsen_args;
    int min_cycles = 0;
    int equality_policy = 0;
    int conflict_policy = 0;
    std::string clock_identity;
    std::string provider_identity;
  };

  enum class TaggingFieldKind : std::uint8_t { state, auxiliary };

  struct ResolvedTaggingField {
    TaggingFieldKind kind = TaggingFieldKind::state;
    std::string qualified_identity;
  };

  struct ResolvedTaggingProgram {
    runtime::amr::PreparedTaggingProgram<Dim> program;
    std::vector<ResolvedTaggingField> fields;
  };

  struct PreparedBoundaryContext {
    std::vector<const field_type*> states;
    std::vector<FieldDistribution> state_distributions;
    std::vector<std::string> state_identities;
    std::vector<const field_type*> fields;
    std::vector<FieldDistribution> field_distributions;
    std::vector<std::string> field_identities;
    const std::vector<Real>* parameters = nullptr;
    FieldLogicalTimePoint point{};
    FieldBoundaryFailure<Dim> failure{};

    FieldBoundaryExecutionContext<Dim> view() {
      if (states.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          fields.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
          (parameters != nullptr &&
           parameters->size() > static_cast<std::size_t>(std::numeric_limits<int>::max())))
        throw std::overflow_error("AMR field boundary dependency pack exceeds native int");
      return {point,
              states.empty() ? nullptr : states.data(),
              state_distributions.empty() ? nullptr : state_distributions.data(),
              state_identities.empty() ? nullptr : state_identities.data(),
              static_cast<int>(states.size()),
              fields.empty() ? nullptr : fields.data(),
              field_distributions.empty() ? nullptr : field_distributions.data(),
              field_identities.empty() ? nullptr : field_identities.data(),
              static_cast<int>(fields.size()),
              parameters,
              parameters == nullptr ? 0 : static_cast<int>(parameters->size()),
              &failure};
    }
  };

  struct FieldPlan {
    std::string plan_identity;
    std::string provider_identity;
    std::string output_owner_identity;
    std::string output_block;
    std::string output_key;
    std::vector<FieldProviderBinding> providers;
    std::string solver_route;
    AmrFieldHierarchyPolicyAuthority hierarchy_policy;
    AmrFieldSolverOptions solver_options;
    std::optional<runtime::field::NamedFieldOutput<Dim>> output;
    std::vector<std::vector<PreparedFieldRhs>> rhs_by_block;
    bool use_prepared_level_rhs = false;
    double reaction = 0.0;
    bool has_reaction = false;
    std::string nullspace_provider_identity;
    PreparedProviderOptions nullspace_options;
    std::string topology_provider_kind;
    std::string topology_provenance;
    std::string topology_digest;
    std::vector<std::string> boundary_kind;
    std::vector<double> boundary_alpha;
    std::vector<double> boundary_beta;
    std::vector<double> boundary_value;
    std::vector<std::string> boundary_state_blocks;
    std::vector<int> boundary_state_components;
    std::vector<std::string> boundary_field_blocks;
    std::vector<std::string> boundary_field_keys;
    std::vector<int> boundary_field_components;
    std::vector<Real> boundary_parameters;
    std::optional<CompiledFieldBoundaryKernel<Dim>> boundary_kernel;
    std::optional<FieldLogicalTimePoint> boundary_point;
    std::optional<FieldNewtonOptions> newton;

    std::unique_ptr<exact_field_solver_type> prepared_solver;
    std::vector<std::unique_ptr<field_type>> accepted_potential;
    std::vector<std::unique_ptr<field_type>> candidate_auxiliary;
    std::vector<std::unique_ptr<field_type>> contribution_scratch;
    std::vector<std::shared_ptr<const field_type>> active_coverage;
    std::vector<PreparedBoundaryContext> boundary_context_storage;
    std::string prepared_contract;
    std::string prepared_nullspace_contract;
    std::uint64_t topology_epoch = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t materialization_generation = std::numeric_limits<std::uint64_t>::max();
    bool candidate_ready = false;

    bool materialized_for(const engine_type& engine) const noexcept {
      return prepared_solver && topology_epoch == engine.topology_epoch() &&
             materialization_generation == engine.materialization_generation();
    }

    void discard_materialization() noexcept {
      prepared_solver.reset();
      accepted_potential.clear();
      candidate_auxiliary.clear();
      contribution_scratch.clear();
      active_coverage.clear();
      boundary_context_storage.clear();
      prepared_contract.clear();
      prepared_nullspace_contract.clear();
      topology_epoch = std::numeric_limits<std::uint64_t>::max();
      materialization_generation = std::numeric_limits<std::uint64_t>::max();
      candidate_ready = false;
    }
  };

  struct PreparedHierarchy {
    // The lane is declared first so every provider that pins it is destroyed before MPI_Comm_free.
    std::optional<ExecutionLane> lane;
    std::vector<std::unique_ptr<field_type>> auxiliary;
    std::vector<std::shared_ptr<const runtime::system::PreparedEmbeddedBoundaryGeometry<Dim>>>
        embedded_boundary;
    std::vector<std::shared_ptr<const field_type>> active_coverage;
    std::vector<level_block_type> levels;
    std::vector<std::optional<evaluation_type>> evaluations;
    std::vector<std::vector<const Real*>> state_storage;
    std::vector<std::vector<const Real*>> auxiliary_storage;
    std::vector<std::string> state_field_identities;
    std::vector<std::string> auxiliary_field_identities;
    std::string spatial_contract;
    std::string package_contract;
    std::string collective_contract;
    std::string embedded_boundary_configuration_contract;
    std::string embedded_boundary_materialization_digest;
    std::uint64_t topology_epoch = 0;
    std::uint64_t materialization_generation = 0;

    bool matches(const engine_type& live, std::string_view expected_package,
                 std::string_view expected_embedded) const noexcept {
      try {
        if (!lane || spatial_contract != live.spatial_contract() ||
            package_contract != expected_package ||
            embedded_boundary_configuration_contract != expected_embedded ||
            topology_epoch != live.topology_epoch() ||
            materialization_generation != live.materialization_generation() ||
            levels.size() != live.hierarchy().num_levels() ||
            embedded_boundary.size() != live.hierarchy().num_levels() ||
            active_coverage.size() != live.hierarchy().num_levels() ||
            state_storage.size() != live.hierarchy().num_levels() ||
            auxiliary_storage.size() != auxiliary.size())
          return false;
        for (std::size_t level = 0; level < state_storage.size(); ++level)
          if (!field_storage_matches(live.hierarchy().state(level), state_storage[level]) ||
              !auxiliary[level] ||
              !field_storage_matches(*auxiliary[level], auxiliary_storage[level]))
            return false;
        return true;
      } catch (...) {
        return false;
      }
    }
  };

  using auxiliary_snapshot_type = std::vector<field_type>;
  struct AcceptedSnapshot;

  AmrSystemConfig<Dim> cfg;
  std::shared_ptr<const PreparedLoadBalanceAuthority<Dim>> load_balance;
  std::vector<BlockSpec> blocks;
  boundary_registry_type boundary_registry;
  std::shared_ptr<exact_field_registry_type> field_solver_providers;
  std::shared_ptr<FieldNullspaceProviderRegistry<Dim>> field_nullspace_providers;
  std::shared_ptr<hierarchy_tensor_registry_type> hierarchy_tensor_solver_providers;
  mutable std::map<std::string, FieldPlan> field_plans;
  std::string default_field_slot;
  std::string default_nullspace_provider_identity;
  PreparedProviderOptions default_nullspace_options;
  mutable std::string active_field_slot;
  runtime::program::ProgramRuntimeState<Dim> program;
  runtime::system::SystemLifecycle lifecycle;
  mutable std::unique_ptr<engine_type> engine;
  std::map<std::string, BootstrapTransferRoute> bootstrap_transfer_routes;
  std::map<std::pair<std::string, std::string>, std::string> bootstrap_subject_routes;
  std::optional<prepared_block_type> prepared_block;
  std::vector<std::string> embedded_boundary_opcodes;
  std::vector<double> embedded_boundary_literals;
  runtime::system::PreparedEmbeddedBoundaryMode embedded_boundary_mode =
      runtime::system::PreparedEmbeddedBoundaryMode::inactive;
  EbThresholds embedded_boundary_thresholds{};
  std::uint64_t embedded_boundary_generation = 0;
  std::string embedded_boundary_configuration_contract;
  std::string embedded_boundary_semantic_digest;
  mutable std::unique_ptr<PreparedHierarchy> prepared_hierarchy;
  mutable std::shared_ptr<const auxiliary_snapshot_type> pending_auxiliary_restore;
  std::vector<GlobalDtBound> dt_bounds;
  std::vector<CouplingOperatorView> coupling_views;
  double accepted_time = 0.0;
  int macro_step = 0;
  std::string last_dt_reason;
  mutable std::vector<std::uint8_t> program_accepted_bytes;
  mutable std::uint64_t program_accepted_revision = 0;
  mutable bool program_accepted_bytes_runtime_owned = false;
  std::optional<TaggingSpec> tagging_spec;
  mutable std::optional<ResolvedTaggingProgram> resolved_tagging;
  mutable std::unique_ptr<runtime::amr::PreparedTaggingExecutionPlan<Dim>> tagging_plan;
  mutable runtime::amr::PersistentTaggingState<Dim> tagging_state;
  mutable std::unique_ptr<AcceptedSnapshot> bootstrap_transaction;
  mutable bool automatic_bootstrap_complete = false;

  struct AcceptedSnapshot {
    std::optional<typename engine_type::Snapshot> engine;
    std::shared_ptr<const auxiliary_snapshot_type> auxiliary;
    runtime::program::ProgramRuntimeState<Dim> program;
    double accepted_time = 0.0;
    int macro_step = 0;
    std::vector<std::uint8_t> program_accepted_bytes;
    std::uint64_t program_accepted_revision = 0;
    bool program_accepted_bytes_runtime_owned = false;
    std::map<std::string, std::vector<field_type>> field_potentials;
    runtime::amr::PersistentTaggingState<Dim> tagging_state;
    bool automatic_bootstrap_complete = false;

    explicit AcceptedSnapshot(const Impl& owner)
        : engine(owner.engine
                     ? std::optional<typename engine_type::Snapshot>(owner.engine->snapshot())
                     : std::nullopt),
          auxiliary(owner.snapshot_auxiliary()),
          program(owner.program),
          accepted_time(owner.accepted_time),
          macro_step(owner.macro_step),
          program_accepted_bytes(owner.program_accepted_bytes),
          program_accepted_revision(owner.program_accepted_revision),
          program_accepted_bytes_runtime_owned(owner.program_accepted_bytes_runtime_owned),
          tagging_state(owner.tagging_state),
          automatic_bootstrap_complete(owner.automatic_bootstrap_complete) {
      if (!owner.active_field_slot.empty())
        throw std::logic_error(
            "AmrSystem cannot snapshot an unconsumed exact field solve candidate");
      for (const auto& [slot, plan] : owner.field_plans) {
        if (plan.accepted_potential.empty())
          continue;
        auto& levels = field_potentials[slot];
        levels.reserve(plan.accepted_potential.size());
        for (const auto& level : plan.accepted_potential) {
          if (!level)
            throw std::logic_error(
                "AmrSystem materialized field plan contains an empty accepted level");
          levels.push_back(*level);
        }
      }
    }

    void restore(Impl& owner) {
      if (engine.has_value() != static_cast<bool>(owner.engine))
        throw std::logic_error("AmrSystem transaction changed engine materialization");
      owner.prepared_hierarchy.reset();
      if (engine) {
        owner.engine->restore(*engine);
        owner.pending_auxiliary_restore = auxiliary;
      }
      owner.program = program;
      owner.accepted_time = accepted_time;
      owner.macro_step = macro_step;
      owner.program_accepted_bytes = program_accepted_bytes;
      owner.program_accepted_revision = program_accepted_revision;
      owner.program_accepted_bytes_runtime_owned = program_accepted_bytes_runtime_owned;
      owner.tagging_state = tagging_state;
      owner.automatic_bootstrap_complete = automatic_bootstrap_complete;
      owner.tagging_plan.reset();
      for (const auto& [slot, levels] : field_potentials) {
        auto found = owner.field_plans.find(slot);
        if (found == owner.field_plans.end())
          throw std::logic_error("AmrSystem transaction changed a materialized field plan");
        if (owner.engine && !found->second.materialized_for(*owner.engine)) {
          found->second.discard_materialization();
          continue;
        }
        if (found->second.accepted_potential.size() != levels.size())
          throw std::logic_error("AmrSystem transaction changed a materialized field plan");
        for (std::size_t level = 0; level < levels.size(); ++level)
          copy_full_field_in_place(levels[level], *found->second.accepted_potential[level]);
        found->second.candidate_ready = false;
      }
      owner.active_field_slot.clear();
    }
  };

  std::unique_ptr<AcceptedSnapshot> external_step_transaction;
  bool external_step_committed = false;

  explicit Impl(const AmrSystemConfig<Dim>& config)
      : cfg(config),
        load_balance(std::make_shared<const PreparedLoadBalanceAuthority<Dim>>(
            prepare_load_balance_authority<Dim>(cfg.load_balance_route, cfg.load_balance_identity,
                                                cfg.load_balance_options))),
        field_solver_providers(std::make_shared<exact_field_registry_type>()),
        field_nullspace_providers(make_default_field_nullspace_provider_registry<Dim>()),
        hierarchy_tensor_solver_providers(
            runtime::program::make_default_hierarchy_tensor_solver_provider_registry<Dim>()) {
    field_solver_providers->add(runtime::amr::make_builtin_exact_amr_field_solver_provider<Dim>());
    const FieldNullspaceProviderSelection selection = operator_topology_zero_mean_nullspace();
    default_nullspace_provider_identity = selection.provider_identity;
    default_nullspace_options = selection.options;
  }

  BlockSpec& block(const std::string& name) {
    for (BlockSpec& candidate : blocks)
      if (candidate.name == name || (blocks.size() == 1 && name.empty()))
        return candidate;
    throw std::runtime_error("AmrSystem has no block named '" + name + "'");
  }

  const BlockSpec& block(const std::string& name) const {
    return const_cast<Impl*>(this)->block(name);
  }

  std::shared_ptr<const auxiliary_snapshot_type> snapshot_auxiliary() const {
    if (!prepared_hierarchy)
      return {};
    auto snapshot = std::make_shared<auxiliary_snapshot_type>();
    snapshot->reserve(prepared_hierarchy->auxiliary.size());
    for (const std::unique_ptr<field_type>& level : prepared_hierarchy->auxiliary) {
      if (!level)
        throw std::logic_error("prepared AMR hierarchy contains an empty auxiliary owner");
      snapshot->push_back(*level);
    }
    return snapshot;
  }

  BoundaryTopology<Dim> topology() const {
    return BoundaryTopology<Dim>::axis_periodic(cfg.periodicity);
  }

  static std::string exact_field_plan_contract(std::string_view slot, const FieldPlan& plan) {
    ExactContractBuilder contract;
    contract.text("pops.amr.exact-ranked-field-plan")
        .scalar(std::uint32_t{2})
        .scalar(std::int32_t{Dim})
        .text(slot)
        .text(plan.plan_identity)
        .text(plan.provider_identity)
        .text(plan.output_owner_identity)
        .text(plan.output_block)
        .text(plan.output_key)
        .sequence(plan.providers,
                  [](ExactContractBuilder& item, const FieldProviderBinding& binding) {
                    item.text(binding.identity)
                        .text(binding.block)
                        .text(binding.key)
                        .scalar(binding.coefficient);
                  })
        .text(plan.solver_route)
        .bytes(plan.hierarchy_policy.exact_contract())
        .bytes(plan.solver_options.exact_contract())
        .presence(plan.has_reaction);
    if (plan.has_reaction)
      contract.scalar(plan.reaction);
    contract.text(plan.nullspace_provider_identity)
        .presence(!plan.nullspace_provider_identity.empty());
    if (!plan.nullspace_provider_identity.empty())
      contract.bytes(plan.nullspace_options.exact_contract());
    contract.text(plan.topology_provider_kind)
        .text(plan.topology_provenance)
        .text(plan.topology_digest)
        .sequence(plan.boundary_kind,
                  [](ExactContractBuilder& item, const std::string& kind) { item.text(kind); })
        .sequence(plan.boundary_alpha)
        .sequence(plan.boundary_beta)
        .sequence(plan.boundary_value)
        .sequence(plan.boundary_state_blocks,
                  [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
        .sequence(plan.boundary_state_components)
        .sequence(plan.boundary_field_blocks,
                  [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
        .sequence(plan.boundary_field_keys,
                  [](ExactContractBuilder& item, const std::string& value) { item.text(value); })
        .sequence(plan.boundary_field_components)
        .sequence(plan.boundary_parameters)
        .presence(plan.output.has_value());
    if (plan.output) {
      contract.scalar(plan.output->gradient_sign())
          .scalar(static_cast<std::uint64_t>(plan.output->component_count()));
      for (std::size_t component = 0; component < plan.output->component_count(); ++component)
        contract.scalar(plan.output->components()[component]);
    }
    contract.presence(plan.use_prepared_level_rhs).presence(plan.boundary_kernel.has_value());
    if (plan.boundary_kernel)
      contract.text(plan.boundary_kernel->identity)
          .text(plan.boundary_kernel->residual_identity)
          .text(plan.boundary_kernel->jvp_identity)
          .presence(plan.boundary_kernel->observes_iteration);
    contract.presence(plan.newton.has_value());
    if (plan.newton)
      contract.scalar(plan.newton->tolerance)
          .scalar(plan.newton->max_iterations)
          .scalar(plan.newton->linear_tolerance)
          .scalar(plan.newton->linear_max_iterations)
          .scalar(plan.newton->restart)
          .scalar(plan.newton->armijo)
          .scalar(plan.newton->minimum_step);
    return std::move(contract).release();
  }

  FieldPlan& resolve_field_plan(std::string_view field) {
    auto found = field_plans.find(std::string(field));
    if (found != field_plans.end())
      return found->second;
    for (auto& [slot, plan] : field_plans) {
      (void)slot;
      if (plan.output_key == field)
        return plan;
    }
    throw std::out_of_range("AmrSystem has no exact field route '" + std::string(field) + "'");
  }

  const FieldPlan& resolve_field_plan(std::string_view field) const {
    return const_cast<Impl*>(this)->resolve_field_plan(field);
  }

  std::string resolve_field_slot(std::string_view field) const {
    const auto exact = field_plans.find(std::string(field));
    if (exact != field_plans.end())
      return exact->first;
    for (const auto& [slot, plan] : field_plans)
      if (plan.output_key == field)
        return slot;
    throw std::out_of_range("AmrSystem has no exact field route '" + std::string(field) + "'");
  }

  static Extent<Dim> unit_ghosts() {
    Extent<Dim> result{};
    for (int axis = 0; axis < Dim; ++axis)
      result[axis] = 1;
    return result;
  }

  PhysicalBoundaryConditions<Dim> field_boundary(const FieldPlan& plan,
                                                 const Geometry<Dim>& geometry) const {
    const BoundaryTopology<Dim> exact_topology = topology();
    std::array<PhysicalBoundaryFace, static_cast<std::size_t>(2 * Dim)> faces{};
    const std::size_t face_count = static_cast<std::size_t>(2 * Dim);
    if (!plan.boundary_kind.empty() &&
        (plan.boundary_kind.size() != face_count || plan.boundary_alpha.size() != face_count ||
         plan.boundary_beta.size() != face_count || plan.boundary_value.size() != face_count))
      throw std::invalid_argument(
          "AMR field boundary plan must cover both faces of every exact axis");

    for (int axis = 0; axis < Dim; ++axis)
      for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
        const Face<Dim> face{axis, side};
        const std::size_t ordinal = static_cast<std::size_t>(face.ordinal());
        const bool periodic = exact_topology.is_periodic(face);
        const std::string kind = plan.boundary_kind.empty() ? (periodic ? "periodic" : "dirichlet")
                                                            : plan.boundary_kind[ordinal];
        if ((kind == "periodic") != periodic)
          throw std::invalid_argument(
              "AMR field boundary periodicity differs from the hierarchy topology");
        if (periodic) {
          faces[ordinal] = {};
          continue;
        }
        const Real value =
            plan.boundary_value.empty() ? Real(0) : static_cast<Real>(plan.boundary_value[ordinal]);
        if (kind == "dirichlet")
          faces[ordinal] = {PhysicalBoundaryKind::dirichlet, value, Real(1), Real(0)};
        else if (kind == "neumann")
          faces[ordinal] = {PhysicalBoundaryKind::neumann, value, Real(0), Real(1)};
        else if (kind == "mixed")
          faces[ordinal] = {PhysicalBoundaryKind::robin, value,
                            static_cast<Real>(plan.boundary_alpha[ordinal]),
                            static_cast<Real>(plan.boundary_beta[ordinal])};
        else
          throw std::invalid_argument("AMR field boundary kind is unknown");
      }

    RealVector<Dim> spacing{};
    for (int axis = 0; axis < Dim; ++axis)
      spacing[axis] = geometry.spacing(axis);
    return PhysicalBoundaryConditions<Dim>(exact_topology, faces, spacing);
  }

  static std::vector<std::shared_ptr<const field_type>> prepare_active_coverage(
      const engine_type& source, std::span<const int> selected_levels) {
    const auto& hierarchy = source.hierarchy();
    if (selected_levels.empty())
      throw std::invalid_argument("AMR composite coverage requires at least one selected level");
    int previous = -1;
    for (const int level : selected_levels) {
      if (level < 0 || static_cast<std::size_t>(level) >= hierarchy.num_levels() ||
          level <= previous)
        throw std::invalid_argument(
            "AMR composite coverage requires strictly increasing live hierarchy levels");
      previous = level;
    }

    std::vector<std::shared_ptr<field_type>> writable;
    writable.reserve(selected_levels.size());
    for (const int level : selected_levels) {
      const field_type& state = hierarchy.state(static_cast<std::size_t>(level));
      auto mask = std::make_shared<field_type>(state.layout(), state.distribution(),
                                               state.local_rank(), 1, Extent<Dim>{});
      mask->set_val(Real(1));
      writable.push_back(std::move(mask));
    }
    for (std::size_t position = 1; position < selected_levels.size(); ++position) {
      const int coarse_level = selected_levels[position - 1];
      const int fine_level = selected_levels[position];
      field_type& parent = *writable[position - 1];
      const auto& fine_layout = hierarchy.layout(static_cast<std::size_t>(fine_level));
      for (const Box<Dim>& fine_patch : fine_layout.patches().boxes()) {
        Box<Dim> footprint = fine_patch;
        for (int current = fine_level; current > coarse_level; --current)
          footprint = amr::hierarchy::coarsen_box(
              footprint, hierarchy.layout(static_cast<std::size_t>(current)).ratio_from_parent());
        for (std::size_t local = 0; local < parent.local_size(); ++local) {
          const Box<Dim> overlap = parent.box(local).intersect(footprint);
          if (overlap.empty())
            continue;
          const auto values = parent.fab(local).view();
          for_each_cell(overlap,
                        [=] POPS_HD(const Index<Dim>& cell) { values(cell, 0) = Real(0); });
        }
      }
    }
    Kokkos::fence();
    std::vector<std::shared_ptr<const field_type>> result;
    result.reserve(writable.size());
    for (auto& mask : writable)
      result.push_back(std::move(mask));
    return result;
  }

  static std::vector<std::shared_ptr<const field_type>> prepare_active_coverage(
      const engine_type& source) {
    std::vector<int> levels;
    levels.reserve(source.hierarchy().num_levels());
    for (std::size_t level = 0; level < source.hierarchy().num_levels(); ++level)
      levels.push_back(static_cast<int>(level));
    return prepare_active_coverage(source, levels);
  }

  const std::vector<std::shared_ptr<const field_type>>& active_coverage() const {
    if (!prepared_hierarchy ||
        prepared_hierarchy->active_coverage.size() != engine->hierarchy().num_levels())
      throw std::logic_error("AMR composite coverage is not prepared for the live hierarchy");
    return prepared_hierarchy->active_coverage;
  }

  static void copy_scalar_component(const field_type& source, int source_component,
                                    field_type& destination, int destination_component) {
    if (source.layout() != destination.layout() ||
        source.distribution() != destination.distribution() ||
        source.local_rank() != destination.local_rank() || source_component < 0 ||
        source_component >= source.ncomp() || destination_component < 0 ||
        destination_component >= destination.ncomp())
      throw std::invalid_argument(
          "AMR field component copy requires one exact layout and valid components");
    for (std::size_t local = 0; local < source.local_size(); ++local) {
      const auto input = source.fab(local).view();
      const auto output = destination.fab(local).view();
      for_each_cell(source.box(local), [=] POPS_HD(const Index<Dim>& cell) {
        output(cell, destination_component) = input(cell, source_component);
      });
    }
    Kokkos::fence();
  }

  FieldNullspaceOperatorFacts nullspace_operator_facts(const FieldPlan& plan) const {
    std::vector<FieldBoundaryNullspaceFact> facts;
    facts.reserve(static_cast<std::size_t>(2 * Dim));
    ExactContractBuilder identity;
    identity.text("pops.amr.field-boundary-set").scalar(std::uint32_t{1}).scalar(std::int32_t{Dim});
    const BoundaryTopology<Dim> exact_topology = topology();
    for (int axis = 0; axis < Dim; ++axis)
      for (const BoundarySide side : {BoundarySide::lower, BoundarySide::upper}) {
        const Face<Dim> face{axis, side};
        const std::size_t ordinal = static_cast<std::size_t>(face.ordinal());
        const std::string face_identity =
            "axis:" + std::to_string(axis) + (side == BoundarySide::lower ? ":lower" : ":upper");
        const std::string kind = plan.boundary_kind.empty()
                                     ? (exact_topology.is_periodic(face) ? "periodic" : "dirichlet")
                                     : plan.boundary_kind[ordinal];
        FieldBoundaryNullspaceBehavior behavior =
            FieldBoundaryNullspaceBehavior::ConstrainsConstantMode;
        if (kind == "periodic" || kind == "neumann" ||
            (kind == "mixed" && plan.boundary_alpha[ordinal] == 0.0))
          behavior = FieldBoundaryNullspaceBehavior::PreservesConstantMode;
        facts.push_back({face_identity, behavior});
        identity.text(face_identity).text(kind);
      }
    return make_field_nullspace_operator_facts(std::move(identity).release(), std::move(facts),
                                               plan.has_reaction);
  }

  FieldNullspaceProviderRequest<Dim> make_field_nullspace_request(
      const std::string& slot, const FieldPlan& plan, exact_field_solver_type& solver,
      const std::vector<std::shared_ptr<const field_type>>& coverage,
      std::vector<PreparedVectorDistribution<Dim>>& distributions) const {
    FieldNullspaceProviderRequest<Dim> request;
    request.plan_identity = plan.plan_identity;
    request.operator_facts = nullspace_operator_facts(plan);
    request.topology.identity =
        plan.topology_digest.empty() ? plan.plan_identity + ":amr-topology" : plan.topology_digest;
    request.topology.field_component = 0;
    request.topology.first_level = 0;
    request.topology.coverage = coverage;
    request.topology.layouts.reserve(static_cast<std::size_t>(solver.level_count()));
    request.topology.cell_measure.reserve(static_cast<std::size_t>(solver.level_count()));
    request.topology.coverage_contracts.reserve(static_cast<std::size_t>(solver.level_count()));
    distributions.clear();
    distributions.reserve(static_cast<std::size_t>(solver.level_count()));
    ExactContractBuilder layout_contract;
    layout_contract.text("pops.amr.field-nullspace-layout")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(slot)
        .bytes(engine->spatial_contract())
        .scalar(engine->topology_epoch())
        .scalar(engine->materialization_generation());
    ExactContractBuilder connected;
    connected.text("pops.amr.connected-cartesian-hierarchy")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(static_cast<std::uint64_t>(solver.level_count()));
    for (int level = 0; level < solver.level_count(); ++level) {
      const field_type& layout = solver.rhs_level(level);
      request.topology.layouts.push_back(&layout);
      PreparedVectorDistribution<Dim> distribution =
          layout.distribution().replicated() ? PreparedVectorDistribution<Dim>::replicated()
                                             : PreparedVectorDistribution<Dim>::distributed();
      layout_contract.bytes(distribution.layout_contract(layout));
      distributions.push_back(distribution);
      Geometry<Dim> geometry = Geometry<Dim>::from_bounds(
          engine->hierarchy().layout(static_cast<std::size_t>(level)).domain(), cfg.lower,
          cfg.upper);
      Real measure = Real(1);
      for (int axis = 0; axis < Dim; ++axis)
        measure *= geometry.spacing(axis);
      request.topology.cell_measure.push_back(measure);
      ExactContractBuilder coverage_contract;
      coverage_contract.text("pops.amr.active-coverage")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .scalar(level)
          .scalar(engine->topology_epoch())
          .scalar(engine->materialization_generation())
          .bytes(distribution.layout_contract(*coverage[static_cast<std::size_t>(level)]));
      request.topology.coverage_contracts.push_back(std::move(coverage_contract).release());
      const Box<Dim>& domain = engine->hierarchy().layout(static_cast<std::size_t>(level)).domain();
      for (int axis = 0; axis < Dim; ++axis)
        connected.scalar(domain.lo[axis]).scalar(domain.hi[axis]);
    }
    request.topology.exact_layout_contract = std::move(layout_contract).release();
    request.topology.connected_component_contract = std::move(connected).release();
    request.topology.level_distributions = distributions;
    return request;
  }

  void materialize_field(const std::string& slot) {
    if (!embedded_boundary_configuration_contract.empty() &&
        embedded_boundary_mode != runtime::system::PreparedEmbeddedBoundaryMode::inactive)
      throw std::runtime_error(
          "AMR Cartesian field solvers have no authenticated embedded-boundary operator");
    ensure_engine();
    auto found = field_plans.find(slot);
    if (found == field_plans.end())
      throw std::out_of_range("AmrSystem has no exact field provider slot '" + slot + "'");
    FieldPlan& plan = found->second;
    if (plan.materialized_for(*engine))
      return;

    std::unique_ptr<exact_field_solver_type> prepared_solver;
    std::vector<std::unique_ptr<field_type>> accepted_potential;
    std::vector<std::unique_ptr<field_type>> candidate_auxiliary;
    std::vector<std::unique_ptr<field_type>> contribution_scratch;
    std::vector<std::shared_ptr<const field_type>> coverage;
    runtime::amr::ExactAmrFieldSolverBuildRequest<Dim> request;
    std::shared_ptr<const exact_field_provider_type> provider;
    std::optional<FieldNullspaceProviderRequest<Dim>> nullspace_request;
    std::vector<PreparedVectorDistribution<Dim>> distributions;
    std::string expected_contract;
    std::string nullspace_contract;
    std::string provider_contract;
    std::string support_contract;

    const auto finish_local_phase = [](std::exception_ptr local_error, std::string_view phase) {
      if (all_reduce_max(local_error ? 1L : 0L) == 0)
        return;
      if (n_ranks() == 1 && local_error)
        std::rethrow_exception(local_error);
      throw std::runtime_error("AMR exact field " + std::string(phase) + " failed collectively");
    };

    std::exception_ptr local_error;
    try {
      if (!plan.output)
        throw std::logic_error("AMR exact field plan has no registered output carrier");
      if ((!plan.boundary_state_blocks.empty() || !plan.boundary_field_blocks.empty()) &&
          !plan.boundary_kernel)
        throw std::logic_error(
            "AMR field boundary dependencies require a compiled dynamic boundary kernel");
      if (!field_solver_providers || !field_nullspace_providers)
        throw std::logic_error("AMR exact field provider registries are absent");
      if (prepared_hierarchy->levels.size() != engine->hierarchy().num_levels() ||
          prepared_hierarchy->auxiliary.size() != engine->hierarchy().num_levels())
        throw std::logic_error("AMR field materialization sees an incomplete prepared hierarchy");

      request.mode = plan.hierarchy_policy.policy_id == "pops.field-hierarchy.level-local"
                         ? runtime::amr::ExactFieldHierarchyMode::level_local
                         : runtime::amr::ExactFieldHierarchyMode::composite;
      if (plan.hierarchy_policy.policy_id != "pops.field-hierarchy.level-local" &&
          plan.hierarchy_policy.policy_id != "pops.field-hierarchy.composite")
        throw std::invalid_argument("AMR exact field hierarchy policy is unsupported: " +
                                    plan.hierarchy_policy.policy_id);
      plan.hierarchy_policy.validate();
      request.provider_options = plan.solver_options;
      request.reaction = static_cast<Real>(plan.has_reaction ? plan.reaction : 0.0);
      request.use_contract = exact_field_plan_contract(slot, plan);
      request.spatial_contract.assign(engine->spatial_contract());
      request.hierarchy.levels.reserve(engine->hierarchy().num_levels());
      request.hierarchy.ratios.reserve(engine->hierarchy().num_levels() - 1);
      for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level) {
        const auto& layout = engine->hierarchy().layout(level);
        const field_type& state = engine->hierarchy().state(level);
        const Geometry<Dim> geometry =
            Geometry<Dim>::from_bounds(layout.domain(), cfg.lower, cfg.upper);
        request.hierarchy.levels.push_back(EllipticBuildRequest<Dim>{
            geometry,
            layout.patches(),
            layout.distribution(),
            state.local_rank(),
            field_boundary(plan, geometry),
            Extent<Dim>{},
            unit_ghosts(),
            {layout.patches().size(), checked_pair_count(layout.patches().size())}});
        if (level != 0)
          request.hierarchy.ratios.push_back(layout.ratio_from_parent());
      }

      provider = field_solver_providers->find(plan.solver_route);
      if (!provider)
        throw std::invalid_argument("unknown exact AMR field solver provider '" +
                                    plan.solver_route + "'");
      const PreparedProviderSupport support = provider->supports(request);
      support_contract = exact_prepared_provider_support(support);
      if (!support.well_formed())
        throw std::runtime_error("exact AMR field solver returned malformed support metadata");
      if (!support.accepted())
        throw std::runtime_error("exact AMR field solver rejected the hierarchy: " +
                                 std::string(support.reason));
      ExactContractBuilder provider_declaration;
      provider_declaration.text("pops.amr.exact-field-provider-declaration")
          .scalar(std::uint32_t{1})
          .text(provider->identity())
          .scalar(provider->interface_version())
          .text(provider->collective_contract());
      provider_contract = std::move(provider_declaration).release();
      expected_contract = provider->expected_prepared_contract(request);
      if (expected_contract.empty())
        throw std::runtime_error(
            "exact AMR field solver accepted the request without a prepared contract");
    } catch (...) {
      local_error = std::current_exception();
    }
    finish_local_phase(local_error, "provider declaration");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"amr-exact-field-use", request.use_contract},
             {"amr-exact-field-spatial", request.spatial_contract},
             {"amr-exact-field-provider", provider_contract},
             {"amr-exact-field-support", support_contract},
             {"amr-exact-field-expected", expected_contract}}))
      throw std::runtime_error("AMR exact field provider declaration differs between MPI ranks");

    local_error = {};
    try {
      prepared_solver = provider->build(request);
      if (!prepared_solver || prepared_solver->provider_identity() != provider->identity() ||
          prepared_solver->exact_prepared_contract() != expected_contract ||
          prepared_solver->level_count() != static_cast<int>(engine->hierarchy().num_levels()))
        throw std::runtime_error("exact AMR field solver published an invalid prepared contract");

      coverage = active_coverage();
      nullspace_request.emplace(
          make_field_nullspace_request(slot, plan, *prepared_solver, coverage, distributions));
    } catch (...) {
      local_error = std::current_exception();
    }
    finish_local_phase(local_error, "solver preparation");

    const FieldNullspaceProviderSelection selection{
        plan.nullspace_provider_identity.empty() ? default_nullspace_provider_identity
                                                 : plan.nullspace_provider_identity,
        plan.nullspace_provider_identity.empty() ? default_nullspace_options
                                                 : plan.nullspace_options};
    PreparedFieldNullspace<Dim> prepared_nullspace = prepare_field_nullspace_collectively<Dim>(
        *field_nullspace_providers, selection, std::move(*nullspace_request));
    nullspace_request.reset();
    nullspace_contract = prepared_nullspace.exact_prepared_contract;

    local_error = {};
    try {
      prepared_solver->install_nullspace(std::move(prepared_nullspace), std::move(distributions));
      if (plan.newton)
        prepared_solver->install_newton(*plan.newton);
      if (plan.boundary_kernel)
        prepared_solver->install_boundary_kernel(*plan.boundary_kernel);

      accepted_potential.reserve(engine->hierarchy().num_levels());
      candidate_auxiliary.reserve(engine->hierarchy().num_levels());
      contribution_scratch.reserve(engine->hierarchy().num_levels());
      for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level) {
        field_type& candidate = prepared_solver->candidate_level(static_cast<int>(level));
        auto accepted = std::make_unique<field_type>(candidate.layout(), candidate.distribution(),
                                                     candidate.local_rank(), candidate.ncomp(),
                                                     candidate.ghosts());
        accepted->set_val(Real(0));
        const field_type& live_auxiliary = *prepared_hierarchy->auxiliary[level];
        plan.output->validate_width(live_auxiliary.ncomp(), "AMR exact field output");
        copy_scalar_component(live_auxiliary, plan.output->potential_component(), *accepted, 0);
        copy_full_field_in_place(*accepted, candidate);
        auto auxiliary = std::make_unique<field_type>(
            live_auxiliary.layout(), live_auxiliary.distribution(), live_auxiliary.local_rank(),
            live_auxiliary.ncomp(), live_auxiliary.ghosts());
        copy_full_field_in_place(live_auxiliary, *auxiliary);
        field_type& rhs = prepared_solver->rhs_level(static_cast<int>(level));
        auto scratch = std::make_unique<field_type>(rhs.layout(), rhs.distribution(),
                                                    rhs.local_rank(), 1, rhs.ghosts());
        scratch->set_val(Real(0));
        accepted_potential.push_back(std::move(accepted));
        candidate_auxiliary.push_back(std::move(auxiliary));
        contribution_scratch.push_back(std::move(scratch));
      }
    } catch (...) {
      local_error = std::current_exception();
    }
    finish_local_phase(local_error, "workspace installation");
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{"amr-exact-field-prepared", expected_contract},
             {"amr-exact-field-nullspace", nullspace_contract}}))
      throw std::runtime_error("AMR exact field materialization differs between MPI ranks");

    plan.prepared_solver = std::move(prepared_solver);
    plan.accepted_potential = std::move(accepted_potential);
    plan.candidate_auxiliary = std::move(candidate_auxiliary);
    plan.contribution_scratch = std::move(contribution_scratch);
    plan.active_coverage = std::move(coverage);
    plan.prepared_contract = std::move(expected_contract);
    plan.prepared_nullspace_contract = std::move(nullspace_contract);
    plan.topology_epoch = engine->topology_epoch();
    plan.materialization_generation = engine->materialization_generation();
    plan.candidate_ready = false;
  }

  static FieldDistribution field_distribution(const field_type& field) noexcept {
    return field.distribution().replicated() ? FieldDistribution::Replicated
                                             : FieldDistribution::Distributed;
  }

  void prepare_field_boundary_contexts(
      const std::string& slot, FieldPlan& plan, int active_level,
      const std::vector<const field_type*>& stage_overrides,
      const runtime::multiblock::BoundaryEvaluationPoint* evaluation_point) {
    plan.boundary_context_storage.clear();
    if (!plan.boundary_kernel)
      return;
    if (evaluation_point == nullptr && !plan.boundary_point)
      throw std::logic_error("AMR dynamic field boundary has no exact logical evaluation point");
    if (evaluation_point != nullptr &&
        (evaluation_point->tick < 0 || evaluation_point->tick > std::numeric_limits<int>::max()))
      throw std::overflow_error("AMR dynamic field boundary tick exceeds native int");

    const std::size_t level_count = engine->hierarchy().num_levels();
    plan.boundary_context_storage.reserve(level_count);
    for (std::size_t level = 0; level < level_count; ++level) {
      PreparedBoundaryContext context;
      context.parameters = &plan.boundary_parameters;
      if (plan.boundary_point)
        context.point = *plan.boundary_point;
      if (evaluation_point != nullptr) {
        context.point.time = static_cast<Real>(evaluation_point->physical_time);
        context.point.dt = static_cast<Real>(evaluation_point->dt);
        context.point.stage_slot = evaluation_point->stage;
        context.point.step = static_cast<int>(evaluation_point->tick);
        context.point.substep = evaluation_point->substep;
      }
      context.point.level = static_cast<int>(level);
      context.point.iteration = 0;

      context.states.reserve(plan.boundary_state_blocks.size());
      context.state_distributions.reserve(plan.boundary_state_blocks.size());
      context.state_identities.reserve(plan.boundary_state_blocks.size());
      for (std::size_t dependency = 0; dependency < plan.boundary_state_blocks.size();
           ++dependency) {
        const auto block =
            std::find_if(blocks.begin(), blocks.end(), [&](const BlockSpec& candidate) {
              return candidate.name == plan.boundary_state_blocks[dependency];
            });
        if (block == blocks.end())
          throw std::runtime_error("AMR dynamic boundary names an unknown state dependency block");
        const std::size_t block_index =
            static_cast<std::size_t>(std::distance(blocks.begin(), block));
        const field_type* state = &engine->hierarchy().state(level);
        if (static_cast<int>(level) == active_level && !stage_overrides.empty() &&
            stage_overrides[block_index] != nullptr)
          state = stage_overrides[block_index];
        const int component = plan.boundary_state_components[dependency];
        if (!same_field_contract(*state, engine->hierarchy().state(level)) || component < 0 ||
            component >= state->ncomp())
          throw std::invalid_argument(
              "AMR dynamic boundary state dependency is not materialized exactly");
        context.states.push_back(state);
        context.state_distributions.push_back(field_distribution(*state));
        context.state_identities.push_back(block->name + "/" +
                                           prepared_hierarchy->state_field_identities[level]);
      }

      context.fields.reserve(plan.boundary_field_blocks.size());
      context.field_distributions.reserve(plan.boundary_field_blocks.size());
      context.field_identities.reserve(plan.boundary_field_blocks.size());
      for (std::size_t dependency = 0; dependency < plan.boundary_field_blocks.size();
           ++dependency) {
        if (plan.boundary_field_components[dependency] != 0)
          throw std::invalid_argument(
              "AMR scalar field boundary dependency must select component zero");
        const auto dependency_plan =
            std::find_if(field_plans.begin(), field_plans.end(), [&](const auto& candidate) {
              return candidate.second.output_block == plan.boundary_field_blocks[dependency] &&
                     candidate.second.output_key == plan.boundary_field_keys[dependency];
            });
        if (dependency_plan == field_plans.end())
          throw std::runtime_error(
              "AMR dynamic boundary names an unknown owner-qualified field dependency");
        if (dependency_plan->first == slot)
          throw std::logic_error(
              "AMR dynamic boundary cannot name its iterate as an external field dependency");
        const FieldPlan& dependency_field = dependency_plan->second;
        if (!dependency_field.materialized_for(*engine) ||
            dependency_field.accepted_potential.size() != level_count)
          throw std::logic_error(
              "AMR dynamic boundary field dependency must be solved and materialized first");
        const field_type& values = *dependency_field.accepted_potential[level];
        context.fields.push_back(&values);
        context.field_distributions.push_back(field_distribution(values));
        context.field_identities.push_back(dependency_plan->first + "/" +
                                           dependency_field.provider_identity);
      }
      plan.boundary_context_storage.push_back(std::move(context));
    }

    std::vector<FieldBoundaryExecutionContext<Dim>> views;
    views.reserve(plan.boundary_context_storage.size());
    for (PreparedBoundaryContext& context : plan.boundary_context_storage)
      views.push_back(context.view());
    plan.prepared_solver->set_boundary_contexts(std::move(views));
  }

  SolveReport solve_field_candidate(
      const std::string& slot, int active_level,
      const std::vector<const field_type*>& stage_overrides,
      const runtime::multiblock::BoundaryEvaluationPoint* evaluation_point = nullptr) {
    materialize_field(slot);
    FieldPlan& plan = field_plans.at(slot);
    if (!active_field_slot.empty())
      throw std::logic_error(
          "AMR exact field solves are sequential until the prior SolveOutcome is consumed");
    if (active_level < 0 ||
        static_cast<std::size_t>(active_level) >= engine->hierarchy().num_levels())
      throw std::out_of_range("AMR exact field solve active level is outside the hierarchy");
    if (!stage_overrides.empty() && stage_overrides.size() != blocks.size())
      throw std::invalid_argument(
          "AMR exact field stage vector must cover the runtime block registry");

    bool has_rhs = false;
    prepare_field_boundary_contexts(slot, plan, active_level, stage_overrides, evaluation_point);
    active_field_slot = slot;
    plan.candidate_ready = false;
    try {
      for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level) {
        field_type& rhs = plan.prepared_solver->rhs_level(static_cast<int>(level));
        field_type& candidate = plan.prepared_solver->candidate_level(static_cast<int>(level));
        rhs.set_val(Real(0));
        copy_full_field_in_place(*plan.accepted_potential[level], candidate);
        const field_type* state = &engine->hierarchy().state(level);
        if (static_cast<int>(level) == active_level && !stage_overrides.empty() &&
            stage_overrides.front() != nullptr)
          state = stage_overrides.front();
        if (!same_field_contract(*state, engine->hierarchy().state(level)))
          throw std::invalid_argument(
              "AMR exact field stage override differs from its live level contract");

        if (plan.use_prepared_level_rhs) {
          prepared_hierarchy->levels[level].add_poisson_rhs(*state, rhs);
          has_rhs = true;
        }
        if (!plan.rhs_by_block.empty()) {
          if (plan.rhs_by_block.size() != blocks.size())
            throw std::logic_error("AMR exact field RHS registry changed after preparation");
          for (const PreparedFieldRhs& provider : plan.rhs_by_block.front()) {
            field_type& scratch = *plan.contribution_scratch[level];
            scratch.set_val(Real(0));
            provider.evaluate(*state, scratch);
            saxpy(rhs, provider.coefficient, scratch);
            has_rhs = true;
          }
        }
      }
      if (!has_rhs)
        throw std::runtime_error("AMR exact field has no prepared RHS provider");
      Kokkos::fence();
      SolveReport report = plan.prepared_solver->solve();
      if (!report.solved_value_available()) {
        active_field_slot.clear();
        return report;
      }
      for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level) {
        field_type& candidate_auxiliary = *plan.candidate_auxiliary[level];
        copy_full_field_in_place(*prepared_hierarchy->auxiliary[level], candidate_auxiliary);
        const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(
            engine->hierarchy().layout(level).domain(), cfg.lower, cfg.upper);
        runtime::field::publish_named_field(
            plan.prepared_solver->candidate_level(static_cast<int>(level)), candidate_auxiliary,
            geometry, *plan.output);
      }
      plan.candidate_ready = true;
      return report;
    } catch (...) {
      plan.candidate_ready = false;
      active_field_slot.clear();
      throw;
    }
  }

  void validate_field_candidate() const {
    if (active_field_slot.empty())
      throw std::logic_error("AMR exact field publication candidate is absent");
    const FieldPlan& plan = field_plans.at(active_field_slot);
    if (!engine || !prepared_hierarchy || !plan.materialized_for(*engine) ||
        !plan.candidate_ready ||
        plan.candidate_auxiliary.size() != prepared_hierarchy->auxiliary.size())
      throw std::logic_error("AMR exact field publication candidate is stale");
    for (std::size_t level = 0; level < plan.candidate_auxiliary.size(); ++level)
      if (!same_field_contract(*plan.candidate_auxiliary[level],
                               *prepared_hierarchy->auxiliary[level]))
        throw std::runtime_error("AMR exact field publication layout changed after solve");
  }

  void accept_field_candidate() noexcept {
    try {
      validate_field_candidate();
      FieldPlan& plan = field_plans.at(active_field_slot);
      for (std::size_t level = 0; level < plan.accepted_potential.size(); ++level) {
        copy_full_field_in_place(plan.prepared_solver->candidate_level(static_cast<int>(level)),
                                 *plan.accepted_potential[level]);
        copy_full_field_in_place(*plan.candidate_auxiliary[level],
                                 *prepared_hierarchy->auxiliary[level]);
      }
      plan.candidate_ready = false;
      active_field_slot.clear();
    } catch (...) {
      std::terminate();
    }
  }

  void reject_field_candidate() noexcept {
    try {
      if (!active_field_slot.empty())
        field_plans.at(active_field_slot).candidate_ready = false;
      active_field_slot.clear();
    } catch (...) {
      std::terminate();
    }
  }

  SolveOutcome make_field_outcome(SolveReport report) {
    FieldPlan* plan = active_field_slot.empty() ? nullptr : &field_plans.at(active_field_slot);
    if (!solve_report_is_publishable(report, plan ? plan->prepared_solver->maximum_iterations()
                                                  : std::numeric_limits<int>::max())) {
      reject_field_candidate();
      throw std::runtime_error("AMR exact field solver published a malformed SolveReport");
    }
    ExactSolveReportConsensusScratch consensus;
    if (!consensus.agrees(report)) {
      reject_field_candidate();
      throw std::runtime_error("AMR exact field SolveReport differs between MPI ranks");
    }
    if (!report.solved_value_available())
      return SolveOutcome::collective_lane(std::move(report), *prepared_hierarchy->lane);
    validate_field_candidate();
    return SolveOutcome::collective_lane(
        std::move(report), *prepared_hierarchy->lane,
        SolveOutcome::PublicationHooks{
            this,
            [](void* context) noexcept { static_cast<Impl*>(context)->accept_field_candidate(); },
            [](void* context) { static_cast<Impl*>(context)->reject_field_candidate(); },
            nullptr,
            {},
            [](void* context) { static_cast<Impl*>(context)->validate_field_candidate(); }});
  }

  std::unique_ptr<PreparedHierarchy> prepare_hierarchy_graph(
      engine_type& candidate_engine, const PreparedHierarchy* previous) const {
    if (!prepared_block)
      throw std::logic_error("AmrSystem has no retained generated package");

    const std::size_t level_count = candidate_engine.hierarchy().num_levels();
    std::unique_ptr<PreparedHierarchy> candidate;
    std::string lane_identity;
    std::shared_ptr<const PreparedHyperbolicBoundary<Dim>> boundary;
    std::string boundary_identity;
    std::exception_ptr allocation_error;
    long allocation_failure = 0;
    try {
      candidate = std::make_unique<PreparedHierarchy>();
      candidate->spatial_contract.assign(candidate_engine.spatial_contract());
      candidate->package_contract = prepared_block->collective_contract;
      candidate->embedded_boundary_configuration_contract =
          embedded_boundary_configuration_contract;
      candidate->topology_epoch = candidate_engine.topology_epoch();
      candidate->materialization_generation = candidate_engine.materialization_generation();
      candidate->auxiliary.reserve(level_count);
      candidate->embedded_boundary.resize(level_count);
      candidate->levels.reserve(level_count);
      candidate->evaluations.resize(level_count);
      candidate->state_storage.reserve(level_count);
      candidate->auxiliary_storage.reserve(level_count);
      candidate->state_field_identities.reserve(level_count);
      candidate->auxiliary_field_identities.reserve(level_count);

      const std::string& state_route = boundary_registry.state_route(prepared_block->name);
      const auto* installed_boundary = boundary_registry.find_boundary(prepared_block->name);
      if (installed_boundary != nullptr) {
        boundary = installed_boundary->authority;
        boundary_identity = installed_boundary->identity;
      }
      lane_identity = "pops.generated-amr-levels/" + std::to_string(candidate->topology_epoch) +
                      "/" + std::to_string(candidate->materialization_generation);
      if (pending_auxiliary_restore && pending_auxiliary_restore->size() != level_count)
        throw std::invalid_argument(
            "AMR rollback auxiliary image differs from the restored hierarchy depth");
      for (std::size_t level = 0; level < level_count; ++level) {
        field_type& state = candidate_engine.hierarchy().state(level);
        auto auxiliary =
            std::make_unique<field_type>(state.layout(), state.distribution(), state.local_rank(),
                                         prepared_block->aux_components, prepared_block->ghosts);
        auxiliary->set_val(Real(0));
        if (pending_auxiliary_restore) {
          const field_type& restored = (*pending_auxiliary_restore)[level];
          if (!same_field_shape(restored, *auxiliary))
            throw std::invalid_argument(
                "AMR rollback auxiliary image differs from the restored level layout");
          copy_valid_field(restored, *auxiliary);
        } else if (previous != nullptr && level < previous->auxiliary.size() &&
                   previous->auxiliary[level] &&
                   same_field_shape(*previous->auxiliary[level], *auxiliary)) {
          copy_valid_field(*previous->auxiliary[level], *auxiliary);
        }
        candidate->state_storage.push_back(field_storage_identity(state));
        candidate->auxiliary_storage.push_back(field_storage_identity(*auxiliary));
        candidate->state_field_identities.push_back(state_route + "/level/" +
                                                    std::to_string(level));
        candidate->auxiliary_field_identities.push_back(
            prepared_block->provider_identity + "/auxiliary/level/" + std::to_string(level));
        candidate->auxiliary.push_back(std::move(auxiliary));
      }
      candidate->active_coverage = prepare_active_coverage(candidate_engine);
    } catch (...) {
      allocation_failure = 1;
      allocation_error = std::current_exception();
    }
    if (all_reduce_max(allocation_failure) != 0) {
      if (allocation_error)
        std::rethrow_exception(allocation_error);
      throw std::runtime_error(
          "generated AMR hierarchy allocation failed collectively before lane publication");
    }

    candidate->lane.emplace(ExecutionLane::duplicate_world_collectively(lane_identity));
    const BoundaryTopology<Dim> exact_topology = topology();

    if (!embedded_boundary_configuration_contract.empty()) {
      ExactContractBuilder materializations;
      materializations.text("pops.amr.embedded-boundary-levels")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(embedded_boundary_configuration_contract)
          .scalar(static_cast<std::uint64_t>(level_count));
      for (std::size_t level = 0; level < level_count; ++level) {
        const Box<Dim>& level_domain = candidate_engine.hierarchy().layout(level).domain();
        const Geometry<Dim> geometry =
            Geometry<Dim>::from_bounds(level_domain, cfg.lower, cfg.upper);
        candidate->embedded_boundary[level] =
            runtime::system::prepare_embedded_boundary_geometry_collectively(
                embedded_boundary_opcodes, embedded_boundary_literals, geometry, exact_topology,
                candidate_engine.hierarchy().state(level), embedded_boundary_mode,
                embedded_boundary_thresholds, embedded_boundary_generation, *candidate->lane);
        materializations.text(candidate->embedded_boundary[level]->digest());
      }
      candidate->embedded_boundary_materialization_digest = prefixed_sha256(
          "pops.prepared-eb-geometry.v1:sha256:", std::move(materializations).release());
    }

    for (std::size_t level = 0; level < level_count; ++level) {
      std::optional<level_block_type> prepared_level;
      std::exception_ptr level_error;
      long level_failure = 0;
      try {
        field_type& state = candidate_engine.hierarchy().state(level);
        field_type& auxiliary = *candidate->auxiliary[level];
        const Box<Dim>& level_domain = candidate_engine.hierarchy().layout(level).domain();
        runtime::amr::PreparedAmrGhostFill<Dim> state_ghost_fill;
        runtime::amr::PreparedAmrGhostFill<Dim> auxiliary_ghost_fill;
        PreparedRootAmrGhostFill<Dim> root_state_ghost_fill;
        PreparedRootAmrGhostFill<Dim> root_auxiliary_ghost_fill;
        if (level == 0) {
          root_state_ghost_fill = prepare_root_ghost_fill(
              state, level_domain, exact_topology, candidate->state_field_identities[level],
              candidate->topology_epoch, candidate->materialization_generation, *candidate->lane);
          root_auxiliary_ghost_fill = prepare_root_ghost_fill(
              auxiliary, level_domain, exact_topology, candidate->auxiliary_field_identities[level],
              candidate->topology_epoch, candidate->materialization_generation, *candidate->lane);
        } else {
          const Box<Dim>& coarse_domain = candidate_engine.hierarchy().layout(level - 1).domain();
          const auto& ratio = candidate_engine.hierarchy().layout(level).ratio_from_parent();
          state_ghost_fill = runtime::amr::prepare_amr_ghost_fill(
              candidate_engine.hierarchy().state(level - 1), state,
              runtime::amr::AmrGhostFillPreparation<Dim>{
                  .fine_level = static_cast<int>(level),
                  .coarse_domain = coarse_domain,
                  .fine_domain = level_domain,
                  .ratio = ratio,
                  .topology = exact_topology,
                  .topology_generation = candidate->topology_epoch,
                  .materialization_generation = candidate->materialization_generation,
                  .field_identity = candidate->state_field_identities[level],
                  .budget =
                      exact_amr_ghost_budget(candidate_engine.hierarchy().state(level - 1), state,
                                             coarse_domain, level_domain, exact_topology),
              },
              *candidate->lane);
          auxiliary_ghost_fill = runtime::amr::prepare_amr_ghost_fill(
              *candidate->auxiliary[level - 1], auxiliary,
              runtime::amr::AmrGhostFillPreparation<Dim>{
                  .fine_level = static_cast<int>(level),
                  .coarse_domain = coarse_domain,
                  .fine_domain = level_domain,
                  .ratio = ratio,
                  .topology = exact_topology,
                  .topology_generation = candidate->topology_epoch,
                  .materialization_generation = candidate->materialization_generation,
                  .field_identity = candidate->auxiliary_field_identities[level],
                  .budget = exact_amr_ghost_budget(*candidate->auxiliary[level - 1], auxiliary,
                                                   coarse_domain, level_domain, exact_topology),
              },
              *candidate->lane);
        }
        GeneratedAmrLevelContext<Dim> context{
            .level = level,
            .geometry = Geometry<Dim>::from_bounds(level_domain, cfg.lower, cfg.upper),
            .topology = exact_topology,
            .auxiliary = &auxiliary,
            .state_ghost_fill = std::move(state_ghost_fill),
            .auxiliary_ghost_fill = std::move(auxiliary_ghost_fill),
            .root_state_ghost_fill = std::move(root_state_ghost_fill),
            .root_auxiliary_ghost_fill = std::move(root_auxiliary_ghost_fill),
            .physical_boundary = boundary,
            .embedded_boundary = candidate->embedded_boundary[level],
            .state_identity = candidate->state_field_identities[level],
            .auxiliary_identity = candidate->auxiliary_field_identities[level],
            .boundary_identity = boundary_identity,
            .embedded_boundary_provider_identity =
                !candidate->embedded_boundary[level] ||
                        embedded_boundary_mode ==
                            runtime::system::PreparedEmbeddedBoundaryMode::inactive
                    ? std::string{}
                : embedded_boundary_mode == runtime::system::PreparedEmbeddedBoundaryMode::staircase
                    ? prepared_block->staircase_provider_identity
                    : prepared_block->cut_cell_provider_identity,
        };
        prepared_level.emplace(prepared_block->prepare_level(candidate_engine, std::move(context)));
      } catch (...) {
        level_failure = 1;
        level_error = std::current_exception();
      }
      if (all_reduce_max(level_failure, candidate->lane->communicator()) != 0) {
        if (level_error)
          std::rethrow_exception(level_error);
        throw std::runtime_error("generated AMR level preparation failed collectively");
      }
      candidate->levels.push_back(std::move(*prepared_level));
    }

    std::exception_ptr contract_error;
    long contract_failure = 0;
    try {
      ExactContractBuilder contract;
      contract.text("pops.generated-amr-hierarchy-graph")
          .scalar(std::uint32_t{1})
          .scalar(std::int32_t{Dim})
          .bytes(candidate->spatial_contract)
          .bytes(candidate->package_contract)
          .bytes(candidate->embedded_boundary_configuration_contract)
          .text(candidate->embedded_boundary_materialization_digest)
          .text(candidate->lane->identity())
          .scalar(candidate->topology_epoch)
          .scalar(candidate->materialization_generation)
          .scalar(static_cast<std::uint64_t>(candidate->levels.size()));
      for (const level_block_type& level : candidate->levels)
        contract.bytes(level.collective_contract());
      candidate->collective_contract = std::move(contract).release();
    } catch (...) {
      contract_failure = 1;
      contract_error = std::current_exception();
    }
    if (all_reduce_max(contract_failure, candidate->lane->communicator()) != 0) {
      if (contract_error)
        std::rethrow_exception(contract_error);
      throw std::runtime_error("generated AMR hierarchy contract failed collectively");
    }
    if (!all_ranks_agree_exact_ordered_byte_pairs(
            {{std::string_view("generated-amr-hierarchy"),
              std::string_view(candidate->collective_contract)}},
            candidate->lane->communicator()))
      throw std::invalid_argument("generated AMR hierarchy contracts differ between MPI ranks");
    return candidate;
  }

  void refresh_prepared_hierarchy() const {
    if (!engine || !prepared_block)
      throw std::logic_error("AmrSystem cannot prepare levels before package materialization");
    const long has_graph = prepared_hierarchy ? 1L : 0L;
    if (all_reduce_min(has_graph) != all_reduce_max(has_graph))
      throw std::runtime_error("prepared AMR graph publication differs between MPI ranks");
    const long stale = prepared_hierarchy && prepared_hierarchy->matches(
                                                 *engine, prepared_block->collective_contract,
                                                 embedded_boundary_configuration_contract)
                           ? 0L
                           : 1L;
    if (all_reduce_max(stale) == 0)
      return;
    std::unique_ptr<PreparedHierarchy> candidate =
        prepare_hierarchy_graph(*engine, prepared_hierarchy.get());
    prepared_hierarchy.swap(candidate);
    pending_auxiliary_restore.reset();
    for (auto& [slot, plan] : field_plans) {
      (void)slot;
      plan.discard_materialization();
    }
    active_field_slot.clear();
  }

  void discard_level_evaluations() const noexcept {
    if (!prepared_hierarchy)
      return;
    for (std::optional<evaluation_type>& evaluation : prepared_hierarchy->evaluations)
      evaluation.reset();
  }

  const ResolvedTaggingProgram& resolve_tagging_program() const {
    if (resolved_tagging)
      return *resolved_tagging;
    if (!tagging_spec || !prepared_block || blocks.size() != 1)
      throw std::logic_error(
          "AMR prepared tagging requires one installed exact-ranked block and graph");

    ResolvedTaggingProgram candidate;
    candidate.program.stencils = tagging_spec->stencils;
    candidate.program.refine_ops = tagging_spec->refine_ops;
    candidate.program.refine_args = tagging_spec->refine_args;
    candidate.program.coarsen_ops = tagging_spec->coarsen_ops;
    candidate.program.coarsen_args = tagging_spec->coarsen_args;
    candidate.program.minimum_cycles = tagging_spec->min_cycles;
    candidate.program.equality_policy = tagging_spec->equality_policy;
    candidate.program.conflict_policy = tagging_spec->conflict_policy;
    candidate.program.non_finite_policy = POPS_TAGGING_NON_FINITE_REJECT_V1;
    candidate.program.clock_identity = tagging_spec->clock_identity;
    candidate.program.provider_identity = tagging_spec->provider_identity;
    candidate.program.prepared = true;

    const auto bind_field = [&](TaggingFieldKind kind, const std::string& identity) -> std::size_t {
      for (std::size_t index = 0; index < candidate.fields.size(); ++index)
        if (candidate.fields[index].qualified_identity == identity) {
          if (candidate.fields[index].kind != kind)
            throw std::invalid_argument(
                "AMR tagging qualified identity resolves to more than one storage authority");
          return index;
        }
      candidate.fields.push_back({kind, identity});
      return candidate.fields.size() - 1;
    };

    for (std::size_t leaf_index = 0; leaf_index < tagging_spec->leaf_subject_kinds.size();
         ++leaf_index) {
      const std::string& kind = tagging_spec->leaf_subject_kinds[leaf_index];
      const std::string& identity = tagging_spec->leaf_subject_identities[leaf_index];
      const std::string& block_name = tagging_spec->leaf_blocks[leaf_index];
      const std::string& variable = tagging_spec->leaf_variables[leaf_index];
      std::size_t field_index = 0;
      int component = -1;
      if (kind == "state") {
        if (block_name != blocks.front().name)
          throw std::invalid_argument("AMR tagging state leaf names another native block");
        const std::string& installed = boundary_registry.state_route(block_name);
        if (identity != installed && identity != direct_amr_state_identity(block_name))
          throw std::invalid_argument(
              "AMR tagging state leaf differs from its exact qualified storage route");
        const auto found = std::find(prepared_block->conservative_variables.names.begin(),
                                     prepared_block->conservative_variables.names.end(), variable);
        if (found == prepared_block->conservative_variables.names.end())
          throw std::invalid_argument("AMR tagging names an unknown conservative variable");
        component = static_cast<int>(
            std::distance(prepared_block->conservative_variables.names.begin(), found));
        field_index = bind_field(TaggingFieldKind::state, identity);
      } else if (kind == "aux") {
        if (identity != "pops://runtime/amr/shared-aux" || !block_name.empty())
          throw std::invalid_argument("AMR tagging auxiliary leaf lacks the shared exact route");
        component = aux_canonical_index<Dim>(variable);
        if (component < 0 || component >= prepared_block->aux_components)
          throw std::invalid_argument("AMR tagging names an unavailable exact auxiliary component");
        field_index = bind_field(TaggingFieldKind::auxiliary, identity);
      } else if (kind == "field") {
        if (!block_name.empty())
          throw std::invalid_argument("AMR tagging field leaf cannot carry a state block route");
        const std::string& slot = boundary_registry.field_storage_route(identity);
        const auto plan = field_plans.find(slot);
        if (plan == field_plans.end() || !plan->second.output)
          throw std::invalid_argument("AMR tagging field leaf has no exact published output");
        const int output_index = tagging_spec->leaf_field_component_indices[leaf_index];
        if (output_index < 0 ||
            static_cast<std::size_t>(output_index) >= plan->second.output->component_count())
          throw std::out_of_range("AMR tagging field output slot is outside its exact carrier");
        component = plan->second.output->components()[static_cast<std::size_t>(output_index)];
        field_index = bind_field(TaggingFieldKind::auxiliary, identity);
      } else {
        throw std::invalid_argument("AMR tagging leaf has an unknown subject kind");
      }
      const int stencil = tagging_spec->leaf_stencil_indices[leaf_index];
      candidate.program.leaves.push_back(
          {field_index, static_cast<std::size_t>(component),
           static_cast<std::int32_t>(tagging_spec->leaf_ops[leaf_index]),
           tagging_spec->leaf_thresholds[leaf_index],
           stencil < 0 ? POPS_TAGGING_NO_STENCIL_V1 : static_cast<std::size_t>(stencil)});
    }
    resolved_tagging.emplace(std::move(candidate));
    return *resolved_tagging;
  }

  std::vector<Box<Dim>> live_tagging_parent_domains() const {
    if (!engine)
      throw std::logic_error("AMR tagging checkpoint requires a materialized hierarchy");
    const std::size_t configured_parents =
        static_cast<std::size_t>(std::max(cfg.level_count - 1, 0));
    const std::size_t parent_count = std::min(engine->hierarchy().num_levels(), configured_parents);
    std::vector<Box<Dim>> domains;
    domains.reserve(parent_count);
    for (std::size_t level = 0; level < parent_count; ++level)
      domains.push_back(engine->hierarchy().layout(level).domain());
    return domains;
  }

  runtime::program::AmrProgramAcceptedState<Dim> minimal_program_accepted_state() const {
    if (!engine)
      throw std::logic_error("AMR Program checkpoint requires a materialized hierarchy");
    runtime::program::AmrProgramAcceptedState<Dim> state;
    state.spatial_contract = engine->spatial_contract();
    state.topology_epoch = engine->topology_epoch();
    state.materialization_generation = engine->materialization_generation();
    state.level_clocks.reserve(engine->hierarchy().num_levels());
    for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level)
      state.level_clocks.push_back(
          {static_cast<int>(level), macro_step, ::pops::amr::Rational(0, 1), accepted_time});
    return state;
  }

  void requalify_runtime_owned_program_state(
      runtime::program::AmrProgramAcceptedState<Dim>& state) const {
    if (!engine)
      throw std::logic_error("AMR Program checkpoint requires a materialized hierarchy");
    state.spatial_contract = engine->spatial_contract();
    state.topology_epoch = engine->topology_epoch();
    state.materialization_generation = engine->materialization_generation();
    state.level_clocks.clear();
    state.level_clocks.reserve(engine->hierarchy().num_levels());
    for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level)
      state.level_clocks.push_back(
          {static_cast<int>(level), macro_step, ::pops::amr::Rational(0, 1), accepted_time});
  }

  void publish_tagging_checkpoint() const {
    if (!tagging_spec || (tagging_spec->min_cycles == 0 && program_accepted_bytes.empty()))
      return;
    runtime::program::AmrProgramAcceptedState<Dim> state;
    const bool synthesize = program_accepted_bytes.empty();
    if (synthesize) {
      state = minimal_program_accepted_state();
    } else {
      state = runtime::program::deserialize_amr_program_accepted_state<Dim>(program_accepted_bytes);
      if (program_accepted_bytes_runtime_owned)
        requalify_runtime_owned_program_state(state);
      else
        runtime::program::require_live_amr_program_checkpoint(state, *engine);
    }
    state.tagging_hysteresis_state =
        tagging_state.encode(tagging_spec->min_cycles, tagging_spec->provider_identity);
    runtime::program::require_collective_amr_program_checkpoint_consensus(
        state, *prepared_hierarchy->lane);
    std::vector<std::uint8_t> candidate =
        runtime::program::serialize_amr_program_accepted_state(state);
    if (candidate == program_accepted_bytes)
      return;
    if (program_accepted_revision == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AmrSystem Program accepted-state revision overflow");
    program_accepted_bytes = std::move(candidate);
    ++program_accepted_revision;
    if (synthesize)
      program_accepted_bytes_runtime_owned = true;
  }

  amr::transfer::TransferKind regrid_transfer_kind(int parent_level) const {
    if (blocks.size() != 1)
      throw std::logic_error("AMR regrid transfer requires one exact-ranked block package");
    const std::string& subject = boundary_registry.state_route(blocks.front().name);
    const auto subject_route =
        bootstrap_subject_routes.find(std::make_pair(subject, std::string("prolongation")));
    if (subject_route == bootstrap_subject_routes.end())
      return amr::transfer::TransferKind::LinearProlongation;
    const auto route = bootstrap_transfer_routes.find(subject_route->second);
    if (route == bootstrap_transfer_routes.end())
      throw std::logic_error("AMR regrid transfer route lost its exact provider authority");
    const std::size_t transition = static_cast<std::size_t>(parent_level);
    if (transition >= cfg.transition_ratios.size() ||
        route->second.refinement_ratio != cfg.transition_ratios[transition])
      throw std::invalid_argument(
          "AMR regrid transfer route does not authenticate this ranked refinement ratio");
    if (route->second.kernel == "conservative_linear")
      return amr::transfer::TransferKind::LinearProlongation;
    if (route->second.kernel == "conservative_injection")
      return amr::transfer::TransferKind::ConstantInjection;
    throw std::invalid_argument("AMR regrid selected an unsupported prolongation kernel");
  }

  std::uint64_t tagging_generation() const {
    if (!engine)
      throw std::logic_error("AMR tagging generation requires a materialized hierarchy");
    if (engine->materialization_generation() == std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("AMR tagging generation exceeds uint64_t");
    return engine->materialization_generation() + 1;
  }

  void prepare_tagging_execution() const {
    if (tagging_plan && tagging_plan->topology_generation() == tagging_generation())
      return;
    const ResolvedTaggingProgram& resolved = resolve_tagging_program();
    if (!prepared_hierarchy || !prepared_hierarchy->lane ||
        prepared_hierarchy->auxiliary.size() != engine->hierarchy().num_levels())
      throw std::logic_error("AMR tagging requires the exact prepared hierarchy graph");
    using TaggingField =
        runtime::amr::PreparedTaggingField<Dim,
                                           typename Kokkos::DefaultExecutionSpace::memory_space>;
    std::vector<std::vector<TaggingField>> fields_by_level;
    std::vector<amr::hierarchy::LevelLayout<Dim>> layouts;
    std::vector<runtime::amr::PreparedTaggingExecutionBudget> budgets;
    fields_by_level.reserve(engine->hierarchy().num_levels());
    layouts.reserve(engine->hierarchy().num_levels());
    budgets.reserve(engine->hierarchy().num_levels());
    for (std::size_t level = 0; level < engine->hierarchy().num_levels(); ++level) {
      std::vector<TaggingField> fields;
      fields.reserve(resolved.fields.size());
      for (const ResolvedTaggingField& field : resolved.fields)
        fields.push_back(
            {field.qualified_identity, field.kind == TaggingFieldKind::state
                                           ? &engine->hierarchy().state(level)
                                           : prepared_hierarchy->auxiliary[level].get()});
      fields_by_level.push_back(std::move(fields));
      layouts.push_back(engine->hierarchy().layout(level));
      budgets.push_back(exact_tagging_budget(engine->hierarchy().layout(level),
                                             engine->hierarchy().state(level).local_rank()));
    }
    auto candidate = runtime::amr::PreparedTaggingExecutionPlan<Dim>::prepare(
        resolved.program, fields_by_level, layouts, budgets, tagging_generation(),
        prepared_hierarchy->lane->communicator());
    tagging_plan =
        std::make_unique<runtime::amr::PreparedTaggingExecutionPlan<Dim>>(std::move(candidate));
  }

  runtime::amr::PreparedTaggerCandidates<Dim> execute_tagging(int parent_level) const {
    if (!tagging_spec)
      throw std::logic_error("AMR hierarchy has no prepared tagging authority");
    if (!prepared_hierarchy || !prepared_hierarchy->lane)
      throw std::logic_error("AMR tagging lacks its prepared collective execution lane");
    const CommunicatorView communicator = prepared_hierarchy->lane->communicator();
    const long invalid_parent =
        parent_level < 0 ||
                static_cast<std::size_t>(parent_level) >= engine->hierarchy().num_levels() ||
                parent_level >= cfg.level_count - 1
            ? 1L
            : 0L;
    const long minimum_parent = all_reduce_min(static_cast<long>(parent_level), communicator);
    const long maximum_parent = all_reduce_max(static_cast<long>(parent_level), communicator);
    if (all_reduce_max(invalid_parent, communicator) != 0)
      throw std::out_of_range("AMR tagging parent lies outside the live configured hierarchy");
    if (minimum_parent != maximum_parent)
      throw std::invalid_argument("AMR tagging parent level differs between MPI ranks");
    prepare_tagging_execution();
    runtime::multiblock::BoundaryEvaluationPoint point;
    point.clock = tagging_spec->clock_identity;
    point.tick = macro_step;
    point.level = parent_level;
    point.stage_fraction = {0, 1};
    point.dt = 0.0;
    point.physical_time = accepted_time;
    field_type& state = engine->hierarchy().state(static_cast<std::size_t>(parent_level));
    prepared_hierarchy->levels[static_cast<std::size_t>(parent_level)].prepare(point, state);
    std::array<Real, Dim> spacing{};
    const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(
        engine->hierarchy().layout(static_cast<std::size_t>(parent_level)).domain(), cfg.lower,
        cfg.upper);
    for (int axis = 0; axis < Dim; ++axis)
      spacing[static_cast<std::size_t>(axis)] = geometry.spacing(axis);
    return tagging_plan->execute(static_cast<std::size_t>(parent_level),
                                 engine->hierarchy().layout(static_cast<std::size_t>(parent_level)),
                                 spacing, tagging_generation());
  }

  std::vector<amr::tagging::TagMask<Dim>> prepare_cluster_shards(
      int parent_level, const runtime::amr::PreparedTaggerCandidates<Dim>& candidates,
      runtime::amr::PersistentTaggingState<Dim>& staged_state) const {
    const auto& layout = engine->hierarchy().layout(static_cast<std::size_t>(parent_level));
    const CommunicatorView communicator = prepared_hierarchy->lane->communicator();
    std::vector<char> refine = gather_tag_mask(candidates.refine, layout.domain(), communicator);
    std::vector<char> coarsen = gather_tag_mask(candidates.coarsen, layout.domain(), communicator);
    const std::vector<char> refine_equalities =
        gather_tag_mask(candidates.refine_equalities, layout.domain(), communicator);
    const std::vector<char> coarsen_equalities =
        gather_tag_mask(candidates.coarsen_equalities, layout.domain(), communicator);
    if (tagging_spec->equality_policy == 1)
      for (std::size_t index = 0; index < refine.size(); ++index)
        refine[index] = static_cast<char>(refine[index] != 0 || refine_equalities[index] != 0 ||
                                          coarsen_equalities[index] != 0);
    else if (tagging_spec->equality_policy == 2)
      for (std::size_t index = 0; index < coarsen.size(); ++index)
        coarsen[index] = static_cast<char>(coarsen[index] != 0 || refine_equalities[index] != 0 ||
                                           coarsen_equalities[index] != 0);

    std::optional<amr::hierarchy::LevelLayout<Dim>> child;
    if (static_cast<std::size_t>(parent_level + 1) < engine->hierarchy().num_levels())
      child = engine->hierarchy().layout(static_cast<std::size_t>(parent_level + 1));
    const std::vector<char> current = child_coverage_on_parent(layout, child);
    std::vector<char> target = current;
    for (std::size_t ordinal = 0; ordinal < target.size(); ++ordinal) {
      bool requests_refine = refine[ordinal] != 0;
      bool requests_coarsen = coarsen[ordinal] != 0;
      if (requests_refine && requests_coarsen) {
        if (tagging_spec->conflict_policy == 0)
          throw std::runtime_error("AMR tagging refine/coarsen predicates conflict");
        if (tagging_spec->conflict_policy == 1)
          requests_refine = requests_coarsen = false;
        else if (tagging_spec->conflict_policy == 2)
          requests_coarsen = false;
        else
          requests_refine = false;
      }
      const bool desired = requests_refine    ? true
                           : requests_coarsen ? false
                                              : current[ordinal] != 0;
      if (desired == (current[ordinal] != 0))
        continue;
      const Index<Dim> cell = unflatten(layout.domain(), ordinal);
      typename runtime::amr::PersistentTaggingState<Dim>::CellKey key{parent_level, cell};
      if (!staged_state.transition_allowed(key, tagging_spec->min_cycles))
        continue;
      target[ordinal] = desired ? char{1} : char{0};
      staged_state.record(key,
                          desired ? runtime::amr::PersistentTaggingState<Dim>::Decision::Refine
                                  : runtime::amr::PersistentTaggingState<Dim>::Decision::Coarsen,
                          tagging_spec->min_cycles);
    }

    std::array<std::size_t, Dim> reach{};
    std::array<std::size_t, Dim> width{};
    std::size_t neighborhood = 1;
    const std::size_t transition = static_cast<std::size_t>(parent_level);
    for (int axis = 0; axis < Dim; ++axis) {
      reach[axis] =
          checked_size_sum(static_cast<std::size_t>(cfg.transition_buffers[transition][axis]),
                           static_cast<std::size_t>(cfg.transition_lookaheads[transition][axis]),
                           "AMR tagging transition reach exceeds size_t");
      width[axis] =
          checked_size_sum(checked_size_product(std::size_t{2}, reach[axis],
                                                "AMR tagging transition width exceeds size_t"),
                           std::size_t{1}, "AMR tagging transition width exceeds size_t");
      neighborhood = checked_size_product(neighborhood, width[axis],
                                          "AMR tagging transition neighborhood exceeds size_t");
    }
    std::vector<char> buffered = target;
    for (std::size_t ordinal = 0; ordinal < target.size(); ++ordinal) {
      if (target[ordinal] == 0)
        continue;
      const Index<Dim> center = unflatten(layout.domain(), ordinal);
      for (std::size_t neighbor = 0; neighbor < neighborhood; ++neighbor) {
        std::size_t quotient = neighbor;
        std::optional<Index<Dim>> canonical = center;
        for (int axis = 0; axis < Dim; ++axis) {
          const std::size_t digit = quotient % width[axis];
          quotient /= width[axis];
          const std::int64_t delta =
              static_cast<std::int64_t>(digit) - static_cast<std::int64_t>(reach[axis]);
          const std::int64_t proposed = static_cast<std::int64_t>(center[axis]) + delta;
          if (proposed >= layout.domain().lo[axis] && proposed <= layout.domain().hi[axis]) {
            (*canonical)[axis] = static_cast<int>(proposed);
            continue;
          }
          if (!topology().is_periodic(Face<Dim>{axis, BoundarySide::lower})) {
            canonical.reset();
            break;
          }
          const std::int64_t length = layout.domain().length(axis);
          const std::int64_t shifted = proposed - layout.domain().lo[axis];
          const std::int64_t wrapped = ((shifted % length) + length) % length;
          (*canonical)[axis] =
              static_cast<int>(static_cast<std::int64_t>(layout.domain().lo[axis]) + wrapped);
        }
        if (canonical && layout_contains(layout, *canonical))
          buffered[offset(*canonical, layout.domain())] = char{1};
      }
    }

    std::vector<amr::tagging::TagMask<Dim>> shards;
    const mesh::RankSpace<Dim>& ranks = layout.distribution().rank_space();
    shards.reserve(ranks.size());
    for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
      const Index<Dim> coordinate = ranks.coordinate(rank);
      shards.emplace_back(layout, coordinate, exact_tag_mask_budget(layout, coordinate));
      for (const auto& patch : shards.back().patches())
        for (std::size_t ordinal = 0; ordinal < patch.tags.size(); ++ordinal) {
          const Index<Dim> cell = unflatten(patch.box, ordinal);
          if (buffered[offset(cell, layout.domain())] != 0)
            shards.back().set(patch.global_patch, cell);
        }
    }
    return shards;
  }

  bool regrid_parent(
      int parent_level, const std::optional<SparseFieldImage<Dim>>& previous_child = std::nullopt,
      runtime::amr::PersistentTaggingState<Dim>* hierarchy_cycle_state = nullptr) const {
    runtime::amr::PreparedTaggerCandidates<Dim> candidates = execute_tagging(parent_level);
    std::optional<SparseFieldImage<Dim>> retained_child = previous_child;
    const std::size_t live_child = static_cast<std::size_t>(parent_level + 1);
    if (!retained_child && live_child < engine->hierarchy().num_levels())
      retained_child = gather_sparse_field(engine->hierarchy().state(live_child),
                                           engine->hierarchy().layout(live_child).domain(),
                                           prepared_hierarchy->lane->communicator());
    runtime::amr::PersistentTaggingState<Dim> staged_state =
        hierarchy_cycle_state ? *hierarchy_cycle_state : tagging_state;
    if (hierarchy_cycle_state == nullptr)
      staged_state.begin_cycle(tagging_spec->min_cycles);
    std::vector<amr::tagging::TagMask<Dim>> shards =
        prepare_cluster_shards(parent_level, candidates, staged_state);
    const auto& parent_layout = engine->hierarchy().layout(static_cast<std::size_t>(parent_level));
    const amr::tagging::ClusterOptions<Dim> cluster_options =
        exact_cluster_options(cfg, parent_layout);
    const amr::tagging::BergerRigoutsosProvider<Dim> clustering;
    amr::tagging::ClusterResult<Dim> clustered = clustering.cluster(shards, cluster_options);
    const amr::RefinementRatio<Dim> ratio =
        refinement_ratio(cfg.transition_ratios[static_cast<std::size_t>(parent_level)]);
    const amr::regridding::RegridPreparationBudget budget =
        exact_regrid_budget(parent_layout, ratio, clustered);
    auto prepared = engine->prepare_regrid(static_cast<std::size_t>(parent_level), ratio,
                                           std::move(clustered), budget, *prepared_hierarchy->lane);
    std::optional<field_type> child_state;
    if (!prepared.removes_fine_level())
      child_state.emplace(transfer_regridded_state(
          engine->hierarchy().state(static_cast<std::size_t>(parent_level)), parent_layout,
          *prepared.fine_layout(), retained_child, prepared_hierarchy->lane->communicator(),
          regrid_transfer_kind(parent_level)));
    engine->publish_regrid(static_cast<std::size_t>(parent_level), std::move(prepared),
                           std::move(child_state));
    if (hierarchy_cycle_state != nullptr)
      *hierarchy_cycle_state = std::move(staged_state);
    else
      tagging_state = std::move(staged_state);
    tagging_plan.reset();
    refresh_prepared_hierarchy();
    program.refresh_hierarchy_state("AmrSystem::regrid_from_prepared_tagging");
    if (hierarchy_cycle_state == nullptr)
      publish_tagging_checkpoint();
    return static_cast<std::size_t>(parent_level + 1) < engine->hierarchy().num_levels();
  }

  void automatic_bootstrap() const {
    if (automatic_bootstrap_complete || cfg.explicit_bootstrap || !tagging_spec)
      return;
    if (std::any_of(tagging_spec->leaf_subject_kinds.begin(),
                    tagging_spec->leaf_subject_kinds.end(),
                    [](const std::string& kind) { return kind == "field"; }))
      throw std::logic_error(
          "automatic AMR bootstrap cannot evaluate a field tagging leaf before its exact field "
          "plan is materialized; enable explicit bootstrap and publish the field first");
    AcceptedSnapshot snapshot(*this);
    try {
      runtime::amr::PersistentTaggingState<Dim> staged_state = tagging_state;
      staged_state.begin_cycle(tagging_spec->min_cycles);
      for (int parent_level = 0; parent_level < cfg.level_count - 1; ++parent_level)
        if (!regrid_parent(parent_level, std::nullopt, &staged_state))
          break;
      tagging_state = std::move(staged_state);
      publish_tagging_checkpoint();
      automatic_bootstrap_complete = true;
    } catch (...) {
      snapshot.restore(*const_cast<Impl*>(this));
      throw;
    }
  }

  void ensure_engine() const {
    const ExecutionLane world_lane = ExecutionLane::world();
    const long materialized = engine ? 1L : 0L;
    if (all_reduce_min(materialized) != all_reduce_max(materialized))
      throw std::runtime_error("AmrSystem hierarchy materialization differs between MPI ranks");
    if (materialized != 0) {
      refresh_prepared_hierarchy();
      automatic_bootstrap();
      return;
    }
    if (blocks.empty())
      throw std::logic_error(
          "AmrSystem requires a dimension-qualified block before materialization");
    if (blocks.size() != 1)
      throw std::logic_error(
          "AmrSystem exact-ranked core requires a prepared multi-block hierarchy provider");

    std::unique_ptr<engine_type> engine_candidate;
    std::exception_ptr engine_error;
    long engine_failure = 0;
    try {
      const BlockSpec& block = blocks.front();
      const Box<Dim> domain = cfg.index_domain();
      const mesh::BoxArray<Dim> patches(cfg.materialized_boxes());
      const mesh::RankSpace<Dim> ranks = process_rank_space<Dim>(world_lane);
      const Index<Dim> local_rank = ranks.coordinate(static_cast<std::size_t>(world_lane.rank()));
      mesh::Distribution<Dim> distribution;
      if (cfg.distribute_coarse) {
        const parallel::LoadBalancePreparationBudget budget{patches.size(), ranks.size(),
                                                            checked_layout_cells(patches)};
        distribution =
            load_balance->prepare(patches, ranks, budget, {}, world_lane).plan().distribution();
      } else {
        distribution = mesh::Distribution<Dim>::replicated(patches, ranks);
      }

      const mesh::BoxArrayValidationBudget layout_budget{patches.size(),
                                                         checked_pair_count(patches.size())};
      const amr::hierarchy::LevelLayout<Dim> coarse(0, domain, patches, distribution,
                                                    amr::RefinementRatio<Dim>{}, layout_budget);
      field_type state(patches, distribution, local_rank, block.ncomp, block.ghosts);
      if (block.has_state)
        write_field(state, domain, block.state, block.ncomp);
      else if (block.has_density)
        write_component(state, domain, block.density, 0);

      const std::size_t hierarchy_pairs = exact_hierarchy_pair_budget(cfg, patches.size());
      auto hierarchy = amr::hierarchy::AmrHierarchy<Dim>::from_coarse(
          coarse, std::move(state),
          amr::hierarchy::HierarchyValidationBudget{static_cast<std::size_t>(cfg.level_count),
                                                    hierarchy_pairs});
      engine_candidate = std::make_unique<engine_type>(std::move(hierarchy), load_balance,
                                                       "pops.amr-system.exact-ranked");
    } catch (...) {
      engine_failure = 1;
      engine_error = std::current_exception();
    }
    if (all_reduce_max(engine_failure) != 0) {
      if (engine_error)
        std::rethrow_exception(engine_error);
      throw std::runtime_error("AmrSystem hierarchy preparation failed collectively");
    }
    std::unique_ptr<PreparedHierarchy> hierarchy_candidate =
        prepare_hierarchy_graph(*engine_candidate, nullptr);
    engine = std::move(engine_candidate);
    prepared_hierarchy = std::move(hierarchy_candidate);
    pending_auxiliary_restore.reset();
    automatic_bootstrap();
  }

  template <class Function>
  decltype(auto) execute_transaction(Function&& function) {
    ensure_engine();
    AcceptedSnapshot snapshot(*this);
    try {
      return std::forward<Function>(function)();
    } catch (...) {
      snapshot.restore(*this);
      throw;
    }
  }
};

template <int Dim>
AmrSystem<Dim>::AmrSystem(const AmrSystemConfig<Dim>& config) {
  validate_amr_config(config);
  p_ = std::make_unique<Impl>(config);
}

template <int Dim>
AmrSystem<Dim>::~AmrSystem() = default;

template <int Dim>
AmrSystem<Dim>::AmrSystem(AmrSystem&&) noexcept = default;

template <int Dim>
AmrSystem<Dim>& AmrSystem<Dim>::operator=(AmrSystem&&) noexcept = default;

template <int Dim>
void AmrSystem<Dim>::add_block(const std::string&, const ModelSpec&, const std::string&,
                               const std::string&, const std::string&, const std::string&, int, int,
                               const std::vector<std::string>&, const std::vector<std::string>&,
                               const NewtonOptions&, bool, double, double, bool) {
  throw std::runtime_error(
      "AmrSystem ModelSpec installation requires a dimension-qualified compiled block provider");
}

template <int Dim>
void AmrSystem<Dim>::set_compiled_block(int ncomp, double gamma, int substeps,
                                        AmrCompiledBlockBuilder<Dim> runtime_builder,
                                        const std::string& name, bool, const std::string& time,
                                        int stride, const std::vector<std::string>& implicit_vars,
                                        const std::vector<std::string>& implicit_roles, double,
                                        double, bool) {
  (void)ncomp;
  (void)gamma;
  (void)substeps;
  (void)runtime_builder;
  (void)name;
  (void)time;
  (void)stride;
  (void)implicit_vars;
  (void)implicit_roles;
  throw std::logic_error(
      "AmrSystem::set_compiled_block is retired: install one complete "
      "PreparedAmrSystemBlock<Dim> through the exact generated-package seam");
}

template <int Dim>
void AmrSystem<Dim>::install_prepared_amr_block(PreparedBlock prepared) {
  std::exception_ptr preparation_error;
  long preparation_failure = 0;
  std::vector<typename Impl::BlockSpec> block_candidate;
  std::optional<PreparedBlock> prepared_candidate;
  std::shared_ptr<const HyperbolicBoundary> converted_boundary;
  std::string install_contract;
  bool has_boundary = false;
  try {
    require_amr_assembling(p_->lifecycle, "install_prepared_amr_block");
    validate_prepared_amr_block(prepared);
    if (p_->engine || p_->prepared_hierarchy || p_->prepared_block || !p_->blocks.empty())
      throw std::logic_error("AmrSystem accepts exactly one prepared generated block");

    const auto route = p_->boundary_registry.state_routes().find(prepared.name);
    if (route == p_->boundary_registry.state_routes().end())
      throw std::runtime_error("prepared AMR block lacks its exact pre-installed state identity");
    const auto* installed_boundary = p_->boundary_registry.find_boundary(prepared.name);
    const BoundaryTopology<Dim> exact_topology =
        BoundaryTopology<Dim>::axis_periodic(p_->cfg.periodicity);
    if (generated_amr_detail::has_physical_faces(exact_topology) && installed_boundary == nullptr)
      throw std::runtime_error(
          "prepared AMR block with physical faces requires a model-qualified boundary");
    if (installed_boundary != nullptr) {
      if (installed_boundary->state_identity != route->second ||
          installed_boundary->authority->ncomp() != prepared.ncomp ||
          installed_boundary->authority->periodic_axes() != p_->cfg.periodicity)
        throw std::invalid_argument(
            "prepared AMR block boundary differs from its exact state/domain contract");
      for (int axis = 0; axis < Dim; ++axis)
        if (prepared.ghosts[axis] < installed_boundary->required_depth)
          throw std::invalid_argument(
              "prepared AMR block ghosts are narrower than its boundary requirement");
      converted_boundary = std::make_shared<HyperbolicBoundary>(
          installed_boundary->authority->with_converted_fixed_states(
              prepared.primitive_to_conservative));
      has_boundary = true;
    }

    typename Impl::BlockSpec block;
    block.name = prepared.name;
    block.ncomp = prepared.ncomp;
    block.gamma = prepared.gamma;
    block.substeps = prepared.substeps;
    block.stride = prepared.stride;
    block.ghosts = prepared.ghosts;
    block.time = prepared.time_route;
    block.required_ghost_depth = 0;
    for (int axis = 0; axis < Dim; ++axis)
      block.required_ghost_depth =
          std::max(block.required_ghost_depth, static_cast<int>(prepared.ghosts[axis]));
    block_candidate.reserve(1);
    block_candidate.push_back(std::move(block));

    ExactContractBuilder contract;
    contract.text("pops.amr-system.prepared-install")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .bytes(prepared.collective_contract)
        .text(route->second)
        .scalar(has_boundary);
    if (installed_boundary != nullptr)
      contract.text(installed_boundary->identity)
          .scalar(std::int32_t{installed_boundary->required_depth})
          .scalar(std::int32_t{installed_boundary->authority->ncomp()})
          .bytes(exact_hyperbolic_boundary_contract(*converted_boundary));
    install_contract = std::move(contract).release();
    prepared_candidate.emplace(std::move(prepared));
  } catch (...) {
    preparation_failure = 1;
    preparation_error = std::current_exception();
  }
  if (all_reduce_max(preparation_failure) != 0) {
    if (preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("prepared AMR block validation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("prepared-amr-install"), std::string_view(install_contract)}}))
    throw std::invalid_argument("prepared AMR block install contracts differ between MPI ranks");

  p_->blocks.swap(block_candidate);
  p_->prepared_block.swap(prepared_candidate);
  if (has_boundary)
    p_->boundary_registry.boundary(p_->blocks.front().name).authority =
        std::move(converted_boundary);
}

template <int Dim>
void AmrSystem<Dim>::set_bootstrap_tagging(
    const std::vector<std::string>& leaf_subject_kinds,
    const std::vector<std::string>& leaf_subject_identities,
    const std::vector<std::string>& leaf_blocks, const std::vector<std::string>& leaf_variables,
    const std::vector<int>& leaf_field_component_indices, const std::vector<int>& leaf_ops,
    const std::vector<double>& leaf_thresholds, const std::vector<int>& leaf_stencil_indices,
    const std::vector<typename runtime::amr::PreparedTaggingProgram<Dim>::Stencil>& stencils,
    const std::vector<std::int32_t>& refine_ops, const std::vector<std::int32_t>& refine_args,
    const std::vector<std::int32_t>& coarsen_ops, const std::vector<std::int32_t>& coarsen_args,
    int min_cycles, const std::string& equality_policy, const std::string& conflict_policy,
    const std::string& clock_identity, const std::string& provider_identity) {
  typename Impl::TaggingSpec candidate;
  std::string contract;
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    require_amr_assembling(p_->lifecycle, "set_bootstrap_tagging");
    const std::size_t leaves = leaf_subject_kinds.size();
    if (p_->engine || p_->tagging_spec || leaves == 0 || leaf_subject_identities.size() != leaves ||
        leaf_blocks.size() != leaves || leaf_variables.size() != leaves ||
        leaf_field_component_indices.size() != leaves || leaf_ops.size() != leaves ||
        leaf_thresholds.size() != leaves || leaf_stencil_indices.size() != leaves ||
        refine_ops.empty() || refine_ops.size() != refine_args.size() ||
        coarsen_ops.size() != coarsen_args.size() || min_cycles < 0 || clock_identity.empty() ||
        provider_identity.empty())
      throw std::invalid_argument(
          "AMR prepared tagging requires one complete unique pre-materialization graph");
    const int equality = equality_policy == "hold"      ? 0
                         : equality_policy == "refine"  ? 1
                         : equality_policy == "coarsen" ? 2
                                                        : -1;
    const int conflict = conflict_policy == "error"          ? 0
                         : conflict_policy == "hold"         ? 1
                         : conflict_policy == "refine_wins"  ? 2
                         : conflict_policy == "coarsen_wins" ? 3
                                                             : -1;
    if (equality < 0 || conflict < 0)
      throw std::invalid_argument("AMR prepared tagging has an unknown decision policy");
    for (std::size_t index = 0; index < leaves; ++index) {
      const std::string& kind = leaf_subject_kinds[index];
      if ((kind != "state" && kind != "aux" && kind != "field") ||
          leaf_subject_identities[index].empty() || leaf_variables[index].empty() ||
          !std::isfinite(leaf_thresholds[index]) ||
          !pops_tagging_opcode_is_leaf_v1(leaf_ops[index]) ||
          (kind == "state" && leaf_blocks[index].empty()) ||
          (kind != "state" && !leaf_blocks[index].empty()) ||
          (kind == "field" && leaf_field_component_indices[index] < 0) ||
          (kind != "field" && leaf_field_component_indices[index] != -1))
        throw std::invalid_argument("AMR prepared tagging has an invalid qualified leaf");
      if (kind == "field" && !p_->cfg.explicit_bootstrap)
        throw std::invalid_argument(
            "AMR field tagging requires explicit bootstrap so its exact field plan can be "
            "materialized before the first predicate evaluation");
      const bool gradient = leaf_ops[index] == POPS_TAGGING_GRADIENT_ABOVE_V1 ||
                            leaf_ops[index] == POPS_TAGGING_GRADIENT_BELOW_V1;
      const int stencil = leaf_stencil_indices[index];
      if (gradient != (stencil >= 0) ||
          (stencil >= 0 && static_cast<std::size_t>(stencil) >= stencils.size()))
        throw std::invalid_argument("AMR prepared tagging leaf/stencil relation is invalid");
    }
    const auto validate_bytecode = [&](const std::vector<std::int32_t>& ops,
                                       const std::vector<std::int32_t>& args, bool required) {
      if (ops.empty()) {
        if (required)
          throw std::invalid_argument("AMR prepared tagging has no refine predicate");
        return;
      }
      int depth = 0;
      for (std::size_t instruction = 0; instruction < ops.size(); ++instruction) {
        const int opcode = ops[instruction];
        const int argument = args[instruction];
        if (pops_tagging_opcode_is_leaf_v1(opcode)) {
          if (argument < 0 || static_cast<std::size_t>(argument) >= leaves ||
              leaf_ops[static_cast<std::size_t>(argument)] != opcode)
            throw std::invalid_argument("AMR prepared tagging bytecode names an invalid leaf");
          ++depth;
        } else if (opcode == POPS_TAGGING_NOT_V1) {
          if (argument != 1 || depth < 1)
            throw std::invalid_argument("AMR prepared tagging bytecode has an invalid NOT");
        } else if (opcode == POPS_TAGGING_ANY_OF_V1 || opcode == POPS_TAGGING_ALL_OF_V1) {
          if (argument < 2 || depth < argument)
            throw std::invalid_argument("AMR prepared tagging bytecode has an invalid arity");
          depth -= argument - 1;
        } else {
          throw std::invalid_argument("AMR prepared tagging bytecode has an unknown opcode");
        }
      }
      if (depth != 1)
        throw std::invalid_argument("AMR prepared tagging bytecode has an invalid final depth");
    };
    validate_bytecode(refine_ops, refine_args, true);
    validate_bytecode(coarsen_ops, coarsen_args, false);
    candidate = {leaf_subject_kinds,
                 leaf_subject_identities,
                 leaf_blocks,
                 leaf_variables,
                 leaf_field_component_indices,
                 leaf_ops,
                 leaf_thresholds,
                 leaf_stencil_indices,
                 stencils,
                 refine_ops,
                 refine_args,
                 coarsen_ops,
                 coarsen_args,
                 min_cycles,
                 equality,
                 conflict,
                 clock_identity,
                 provider_identity};
    ExactContractBuilder exact;
    exact.text("pops.amr-system.prepared-tagging-authoring")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .sequence(leaf_subject_kinds,
                  [](ExactContractBuilder& row, const std::string& value) { row.text(value); })
        .sequence(leaf_subject_identities,
                  [](ExactContractBuilder& row, const std::string& value) { row.text(value); })
        .sequence(leaf_blocks,
                  [](ExactContractBuilder& row, const std::string& value) { row.text(value); })
        .sequence(leaf_variables,
                  [](ExactContractBuilder& row, const std::string& value) { row.text(value); })
        .sequence(leaf_field_component_indices)
        .sequence(leaf_ops)
        .sequence(leaf_thresholds)
        .sequence(leaf_stencil_indices)
        .scalar(static_cast<std::uint64_t>(stencils.size()));
    for (const auto& stencil : stencils) {
      exact.text(stencil.identity)
          .text(stencil.route)
          .text(stencil.norm)
          .text(stencil.scale)
          .text(stencil.boundary_mode);
      for (int axis = 0; axis < Dim; ++axis) {
        const auto& row = stencil.axes[static_cast<std::size_t>(axis)];
        exact.scalar(row.axis)
            .scalar(row.derivative_order)
            .scalar(row.formal_order)
            .scalar(static_cast<std::uint64_t>(row.ghost_lower))
            .scalar(static_cast<std::uint64_t>(row.ghost_upper))
            .sequence(row.offsets)
            .sequence(row.coefficients);
      }
    }
    exact.sequence(refine_ops)
        .sequence(refine_args)
        .sequence(coarsen_ops)
        .sequence(coarsen_args)
        .scalar(min_cycles)
        .scalar(equality)
        .scalar(conflict)
        .text(clock_identity)
        .text(provider_identity);
    contract = std::move(exact).release();
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_failure) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR prepared tagging authoring failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("prepared-amr-tagging"), std::string_view(contract)}}))
    throw std::invalid_argument("AMR prepared tagging contracts differ between MPI ranks");
  p_->tagging_spec.emplace(std::move(candidate));
  p_->resolved_tagging.reset();
  p_->tagging_plan.reset();
  p_->tagging_state.clear();
  p_->automatic_bootstrap_complete = false;
}

template <int Dim>
void AmrSystem<Dim>::set_analytic_level_set(const std::vector<std::string>& opcodes,
                                            const std::vector<double>& literals,
                                            const std::string& mode, double kappa_min,
                                            double face_open_eps, double cut_theta_min) {
  runtime::system::PreparedEmbeddedBoundaryMode prepared_mode =
      runtime::system::PreparedEmbeddedBoundaryMode::inactive;
  EbThresholds thresholds{};
  std::uint64_t generation = 0;
  std::vector<std::string> staged_opcodes;
  std::vector<double> staged_literals;
  PreparedAmrEbAuthoring authored;
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    require_amr_assembling(p_->lifecycle, "set_analytic_level_set");
    if (!p_->prepared_block)
      throw std::logic_error(
          "AmrSystem embedded geometry requires an installed exact generated block provider");
    if (p_->engine || p_->prepared_hierarchy)
      throw std::logic_error(
          "AmrSystem embedded geometry must be authored before hierarchy materialization");
    prepared_mode = runtime::system::parse_prepared_embedded_boundary_mode(mode);
    thresholds = resolved_amr_eb_thresholds(kappa_min, face_open_eps, cut_theta_min);
    generation = next_amr_eb_generation(p_->embedded_boundary_generation);
    staged_opcodes = opcodes;
    staged_literals = literals;
    authored = prepare_amr_eb_authoring(p_->cfg, *p_->prepared_block, staged_opcodes,
                                        staged_literals, prepared_mode, thresholds, generation);
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_failure) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR embedded-boundary authoring failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"amr-eb-configuration", authored.configuration_contract},
           {"amr-eb-semantic", authored.semantic_digest}}))
    throw std::invalid_argument(
        "AMR embedded-boundary authoring contracts differ between MPI ranks");

  p_->embedded_boundary_opcodes = std::move(staged_opcodes);
  p_->embedded_boundary_literals = std::move(staged_literals);
  p_->embedded_boundary_mode = prepared_mode;
  p_->embedded_boundary_thresholds = thresholds;
  p_->embedded_boundary_generation = generation;
  p_->embedded_boundary_configuration_contract = std::move(authored.configuration_contract);
  p_->embedded_boundary_semantic_digest = std::move(authored.semantic_digest);
}

template <int Dim>
void AmrSystem<Dim>::set_disc_domain(double cx, double cy, double radius, const std::string& mode,
                                     double kappa_min, double face_open_eps, double cut_theta_min) {
  std::pair<std::vector<std::string>, std::vector<double>> staged;
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    staged = AmrDiscLevelSetCapability<Dim>::make(cx, cy, radius);
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_failure) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR disc authoring failed collectively");
  }
  set_analytic_level_set(staged.first, staged.second, mode, kappa_min, face_open_eps,
                         cut_theta_min);
}

template <int Dim>
void AmrSystem<Dim>::set_geometry_mode(const std::string& mode) {
  runtime::system::PreparedEmbeddedBoundaryMode prepared_mode =
      runtime::system::PreparedEmbeddedBoundaryMode::inactive;
  std::uint64_t generation = 0;
  PreparedAmrEbAuthoring authored;
  bool unchanged_inactive = false;
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    require_amr_assembling(p_->lifecycle, "set_geometry_mode");
    prepared_mode = runtime::system::parse_prepared_embedded_boundary_mode(mode);
    if (p_->embedded_boundary_configuration_contract.empty()) {
      if (prepared_mode == runtime::system::PreparedEmbeddedBoundaryMode::inactive) {
        unchanged_inactive = true;
      } else {
        throw std::logic_error("AmrSystem geometry mode requires an analytic level set");
      }
    } else {
      if (!p_->prepared_block || p_->engine || p_->prepared_hierarchy)
        throw std::logic_error(
            "AmrSystem geometry mode must be selected on an assembled exact block before build");
      generation = next_amr_eb_generation(p_->embedded_boundary_generation);
      authored =
          prepare_amr_eb_authoring(p_->cfg, *p_->prepared_block, p_->embedded_boundary_opcodes,
                                   p_->embedded_boundary_literals, prepared_mode,
                                   p_->embedded_boundary_thresholds, generation);
    }
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_failure) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR embedded-boundary mode selection failed collectively");
  }
  if (all_reduce_min(unchanged_inactive ? 1L : 0L) != all_reduce_max(unchanged_inactive ? 1L : 0L))
    throw std::invalid_argument("AMR embedded-boundary mode state differs between MPI ranks");
  if (unchanged_inactive)
    return;
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"amr-eb-configuration", authored.configuration_contract},
           {"amr-eb-semantic", authored.semantic_digest}}))
    throw std::invalid_argument("AMR embedded-boundary mode contracts differ between MPI ranks");
  p_->embedded_boundary_mode = prepared_mode;
  p_->embedded_boundary_generation = generation;
  p_->embedded_boundary_configuration_contract = std::move(authored.configuration_contract);
  p_->embedded_boundary_semantic_digest = std::move(authored.semantic_digest);
}

template <int Dim>
void AmrSystem<Dim>::refresh_prepared_amr_levels() {
  p_->ensure_engine();
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedLevelEvaluation& AmrSystem<Dim>::evaluate_prepared_amr_level(
    const runtime::multiblock::BoundaryEvaluationPoint& point) {
  p_->ensure_engine();
  if (point.level < 0 ||
      static_cast<std::size_t>(point.level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("prepared AMR evaluation level lies outside the live hierarchy");
  return evaluate_prepared_amr_level_at(
      point, p_->engine->hierarchy().state(static_cast<std::size_t>(point.level)));
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedLevelEvaluation&
AmrSystem<Dim>::evaluate_prepared_amr_level_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab<Dim>& state) {
  p_->ensure_engine();
  std::string point_contract;
  std::exception_ptr point_error;
  long point_failure = 0;
  try {
    ExactContractBuilder contract;
    contract.text("pops.generated-amr-evaluation-point")
        .scalar(std::uint32_t{1})
        .text(point.clock)
        .scalar(point.tick)
        .scalar(std::int32_t{point.level})
        .scalar(std::int32_t{point.substep})
        .scalar(std::int32_t{point.stage})
        .scalar(point.stage_fraction.numerator)
        .scalar(point.stage_fraction.denominator)
        .scalar(point.dt)
        .scalar(point.physical_time);
    point_contract = std::move(contract).release();
  } catch (...) {
    point_failure = 1;
    point_error = std::current_exception();
  }
  if (all_reduce_max(point_failure, p_->prepared_hierarchy->lane->communicator()) != 0) {
    if (point_error)
      std::rethrow_exception(point_error);
    throw std::runtime_error("prepared AMR evaluation point failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("generated-amr-evaluation-point"), std::string_view(point_contract)}},
          p_->prepared_hierarchy->lane->communicator()))
    throw std::invalid_argument("prepared AMR evaluation points differ between MPI ranks");
  if (point.level < 0 ||
      static_cast<std::size_t>(point.level) >= p_->prepared_hierarchy->levels.size())
    throw std::out_of_range("prepared AMR evaluation level lies outside the live hierarchy");
  const std::size_t level_index = static_cast<std::size_t>(point.level);
  MultiFab<Dim>& live = p_->engine->hierarchy().state(level_index);
  std::optional<MultiFab<Dim>> live_backup =
      stage_exact_field_collectively(state, live, p_->prepared_hierarchy->lane->communicator());
  std::optional<PreparedLevelEvaluation> candidate;
  std::exception_ptr evaluation_error;
  long evaluation_failure = 0;
  try {
    candidate.emplace(p_->prepared_hierarchy->levels[level_index].evaluate(point));
  } catch (...) {
    evaluation_failure = 1;
    evaluation_error = std::current_exception();
  }
  restore_exact_field_collectively(live_backup, live, p_->prepared_hierarchy->lane->communicator());
  if (all_reduce_max(evaluation_failure, p_->prepared_hierarchy->lane->communicator()) != 0) {
    if (evaluation_error)
      std::rethrow_exception(evaluation_error);
    throw std::runtime_error("prepared AMR level evaluation failed collectively");
  }
  std::optional<PreparedLevelEvaluation>& published =
      p_->prepared_hierarchy->evaluations[level_index];
  published.swap(candidate);
  return *published;
}

template <int Dim>
void AmrSystem<Dim>::prepare_generated_amr_level_state(
    const runtime::multiblock::BoundaryEvaluationPoint& point, MultiFab<Dim>& state) {
  p_->ensure_engine();
  if (point.level < 0 ||
      static_cast<std::size_t>(point.level) >= p_->prepared_hierarchy->levels.size())
    throw std::out_of_range("prepared AMR state-preparation level lies outside the live hierarchy");

  std::string point_contract;
  std::exception_ptr point_error;
  long point_failure = 0;
  try {
    ExactContractBuilder contract;
    contract.text("pops.generated-amr-state-preparation-point")
        .scalar(std::uint32_t{1})
        .text(point.clock)
        .scalar(point.tick)
        .scalar(std::int32_t{point.level})
        .scalar(std::int32_t{point.substep})
        .scalar(std::int32_t{point.stage})
        .scalar(point.stage_fraction.numerator)
        .scalar(point.stage_fraction.denominator)
        .scalar(point.dt)
        .scalar(point.physical_time);
    point_contract = std::move(contract).release();
  } catch (...) {
    point_failure = 1;
    point_error = std::current_exception();
  }
  if (all_reduce_max(point_failure, p_->prepared_hierarchy->lane->communicator()) != 0) {
    if (point_error)
      std::rethrow_exception(point_error);
    throw std::runtime_error("prepared AMR state-preparation point failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("generated-amr-state-preparation-point"),
            std::string_view(point_contract)}},
          p_->prepared_hierarchy->lane->communicator()))
    throw std::invalid_argument("prepared AMR state-preparation points differ between MPI ranks");

  const std::size_t level_index = static_cast<std::size_t>(point.level);
  MultiFab<Dim>& live = p_->engine->hierarchy().state(level_index);
  const long staged = &state == &live ? 0L : 1L;
  if (all_reduce_min(staged, p_->prepared_hierarchy->lane->communicator()) !=
      all_reduce_max(staged, p_->prepared_hierarchy->lane->communicator()))
    throw std::invalid_argument("prepared AMR state-preparation target differs between MPI ranks");

  std::optional<MultiFab<Dim>> prepared_candidate;
  std::exception_ptr candidate_error;
  long candidate_failure = 0;
  try {
    if (staged != 0)
      prepared_candidate.emplace(state);
  } catch (...) {
    candidate_failure = 1;
    candidate_error = std::current_exception();
  }
  if (all_reduce_max(candidate_failure, p_->prepared_hierarchy->lane->communicator()) != 0) {
    if (candidate_error)
      std::rethrow_exception(candidate_error);
    throw std::runtime_error("prepared AMR state candidate allocation failed collectively");
  }

  MultiFab<Dim>& prepared_state = prepared_candidate ? *prepared_candidate : state;
  std::optional<MultiFab<Dim>> live_backup = stage_exact_field_collectively(
      prepared_state, live, p_->prepared_hierarchy->lane->communicator());
  std::exception_ptr preparation_error;
  long preparation_failure = 0;
  try {
    p_->prepared_hierarchy->levels[level_index].prepare(point, live);
    if (prepared_candidate)
      copy_full_field_in_place(live, *prepared_candidate);
  } catch (...) {
    preparation_failure = 1;
    preparation_error = std::current_exception();
  }
  restore_exact_field_collectively(live_backup, live, p_->prepared_hierarchy->lane->communicator());
  if (all_reduce_max(preparation_failure, p_->prepared_hierarchy->lane->communicator()) != 0) {
    if (preparation_error)
      std::rethrow_exception(preparation_error);
    throw std::runtime_error("prepared AMR state preparation failed collectively");
  }
  if (prepared_candidate)
    state = std::move(*prepared_candidate);
}

template <int Dim>
const typename AmrSystem<Dim>::PreparedLevelEvaluation&
AmrSystem<Dim>::prepared_amr_level_evaluation(int level) const {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->prepared_hierarchy->evaluations.size())
    throw std::out_of_range("prepared AMR ledger level lies outside the live hierarchy");
  const std::optional<PreparedLevelEvaluation>& evaluation =
      p_->prepared_hierarchy->evaluations[static_cast<std::size_t>(level)];
  if (!evaluation)
    throw std::logic_error("prepared AMR level has no published residual/flux evaluation");
  return *evaluation;
}

template <int Dim>
Geometry<Dim> AmrSystem<Dim>::prepared_amr_level_geometry(int level) const {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("prepared AMR geometry level lies outside the live hierarchy");
  return Geometry<Dim>::from_bounds(
      p_->engine->hierarchy().layout(static_cast<std::size_t>(level)).domain(), p_->cfg.lower,
      p_->cfg.upper);
}

template <int Dim>
BoundaryTopology<Dim> AmrSystem<Dim>::prepared_amr_boundary_topology() const {
  return p_->topology();
}

template <int Dim>
Real AmrSystem<Dim>::prepared_amr_level_maximum_speed(int level, const MultiFab<Dim>& state) const {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->prepared_hierarchy->levels.size())
    throw std::out_of_range("prepared AMR speed level lies outside the live hierarchy");
  return p_->prepared_hierarchy->levels[static_cast<std::size_t>(level)].maximum_speed(state);
}

template <int Dim>
MultiFab<Dim>& AmrSystem<Dim>::prepared_amr_level_auxiliary(int level) {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->prepared_hierarchy->auxiliary.size())
    throw std::out_of_range("prepared AMR auxiliary level lies outside the live hierarchy");
  p_->discard_level_evaluations();
  return *p_->prepared_hierarchy->auxiliary[static_cast<std::size_t>(level)];
}

template <int Dim>
const MultiFab<Dim>& AmrSystem<Dim>::prepared_amr_level_auxiliary(int level) const {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->prepared_hierarchy->auxiliary.size())
    throw std::out_of_range("prepared AMR auxiliary level lies outside the live hierarchy");
  return *p_->prepared_hierarchy->auxiliary[static_cast<std::size_t>(level)];
}

template <int Dim>
void AmrSystem<Dim>::add_prepared_amr_poisson_rhs(int level, MultiFab<Dim>& rhs) {
  p_->ensure_engine();
  if (all_reduce_min(static_cast<long>(level)) != all_reduce_max(static_cast<long>(level)))
    throw std::invalid_argument("prepared AMR Poisson RHS levels differ between MPI ranks");
  if (level < 0 || static_cast<std::size_t>(level) >= p_->prepared_hierarchy->levels.size())
    throw std::out_of_range("prepared AMR Poisson RHS level lies outside the live hierarchy");
  p_->prepared_hierarchy->levels[static_cast<std::size_t>(level)].add_poisson_rhs(rhs);
}

template <int Dim>
void AmrSystem<Dim>::install_block_state_route(const std::string& name,
                                               const std::string& state_identity) {
  require_amr_assembling(p_->lifecycle, "install_block_state_route");
  if (!p_->blocks.empty())
    throw std::logic_error("AmrSystem state routes must be installed before their block");
  p_->boundary_registry.install_state_route(name, state_identity);
}

template <int Dim>
void AmrSystem<Dim>::install_field_storage_route(const std::string& field_identity,
                                                 const std::string& provider_slot) {
  require_amr_assembling(p_->lifecycle, "install_field_storage_route");
  p_->boundary_registry.install_field_storage_route(field_identity, provider_slot);
}

template <int Dim>
void AmrSystem<Dim>::register_field_nullspace_provider(
    std::shared_ptr<const FieldNullspaceProvider<Dim>> provider) {
  require_amr_assembling(p_->lifecycle, "register_field_nullspace_provider");
  if (!p_->field_nullspace_providers)
    throw std::logic_error("AmrSystem field-nullspace registry is absent");
  p_->field_nullspace_providers->add(std::move(provider));
}

template <int Dim>
void AmrSystem<Dim>::register_field_solver_provider(
    std::shared_ptr<const runtime::amr::ExactAmrFieldSolverProvider<Dim>> provider) {
  require_amr_assembling(p_->lifecycle, "register_field_solver_provider");
  if (!p_->field_solver_providers)
    throw std::logic_error("AmrSystem exact field-solver registry is absent");
  p_->field_solver_providers->add(std::move(provider));
}

template <int Dim>
std::string AmrSystem<Dim>::register_field_solver_provider(
    const std::string& provider_slot, runtime::field::PreparedFieldSolverSpec,
    std::shared_ptr<component::LoadedComponent>, std::shared_ptr<component::LoadedComponent>) {
  require_amr_assembling(p_->lifecycle, "register_field_solver_provider");
  if (provider_slot.empty())
    throw std::invalid_argument("AMR component field provider slot must be non-empty");
  throw std::logic_error(
      "AMR component FieldTopology/FieldSolver pairs require an exact-ranked hierarchy provider");
}

template <int Dim>
void AmrSystem<Dim>::set_default_field_nullspace(const std::string& nullspace_provider_identity,
                                                 const PreparedProviderOptions& options) {
  require_amr_assembling(p_->lifecycle, "set_default_field_nullspace");
  if (nullspace_provider_identity.empty())
    throw std::invalid_argument("AMR default field nullspace provider identity must be non-empty");
  (void)options.exact_contract();
  p_->default_nullspace_provider_identity = nullspace_provider_identity;
  p_->default_nullspace_options = options;
}

template <int Dim>
void AmrSystem<Dim>::register_hierarchy_tensor_solver_provider(
    std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider<Dim>> provider) {
  require_amr_assembling(p_->lifecycle, "register_hierarchy_tensor_solver_provider");
  if (!p_->hierarchy_tensor_solver_providers)
    throw std::logic_error("AmrSystem hierarchy tensor-solver registry is absent");
  p_->hierarchy_tensor_solver_providers->add(std::move(provider));
}

template <int Dim>
void AmrSystem<Dim>::register_program_hierarchy_tensor_solver_provider(
    std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider<Dim>> provider) {
  if (!p_->hierarchy_tensor_solver_providers)
    throw std::logic_error("AmrSystem hierarchy tensor-solver registry is absent");
  p_->hierarchy_tensor_solver_providers->add_collectively(std::move(provider));
}

template <int Dim>
std::shared_ptr<const runtime::program::HierarchyTensorSolverProviderRegistry<Dim>>
AmrSystem<Dim>::hierarchy_tensor_solver_provider_registry() const {
  if (!p_->hierarchy_tensor_solver_providers)
    throw std::logic_error("AmrSystem hierarchy tensor-solver registry is absent");
  return p_->hierarchy_tensor_solver_providers;
}

template <int Dim>
void AmrSystem<Dim>::set_poisson(const std::string& rhs, const std::string& solver,
                                 const std::string& bc,
                                 const AmrFieldSolverOptions& solver_options) {
  require_amr_assembling(p_->lifecycle, "set_poisson");
  if (rhs != "charge_density" && rhs != "composite")
    throw std::invalid_argument(
        "AMR exact Poisson supports charge_density or composite RHS routes");
  if (solver != "geometric_mg")
    throw std::invalid_argument("AMR exact Poisson requires the geometric_mg provider");
  if (bc != "auto" && bc != "periodic" && bc != "dirichlet" && bc != "neumann")
    throw std::invalid_argument("AMR exact Poisson boundary mode is unknown");
  if (!p_->default_field_slot.empty())
    throw std::logic_error("AMR default field is already configured");
  if (bc == "periodic" && std::any_of(p_->cfg.periodicity.begin(), p_->cfg.periodicity.end(),
                                      [](bool periodic) { return !periodic; }))
    throw std::invalid_argument(
        "AMR periodic Poisson requires every exact topology axis to be periodic");

  typename Impl::FieldPlan plan;
  plan.plan_identity = "pops.amr.default-field-plan";
  plan.provider_identity = "pops.amr.default-field";
  plan.output_owner_identity = "pops.amr.shared-auxiliary";
  plan.output_block = "amr";
  plan.output_key = "fields_from_state";
  plan.solver_route = solver;
  plan.hierarchy_policy = {
      "pops.field-hierarchy.composite", 1, {"pops.field-hierarchy.options.empty@1", {}}};
  plan.solver_options =
      solver_options.schema_identity.empty()
          ? geometric_mg_amr_field_solver_options(GeometricMgOptions{}, CompositeFacOptions{})
          : solver_options;
  (void)plan.solver_options.exact_contract();
  std::vector<int> outputs;
  outputs.reserve(static_cast<std::size_t>(Dim + 1));
  for (int component = 0; component <= Dim; ++component)
    outputs.push_back(component);
  plan.output.emplace(outputs, 1);
  plan.use_prepared_level_rhs = true;
  if (bc != "auto") {
    const std::size_t faces = static_cast<std::size_t>(2 * Dim);
    plan.boundary_kind.assign(faces, bc);
    plan.boundary_alpha.assign(faces, bc == "dirichlet" ? 1.0 : 0.0);
    plan.boundary_beta.assign(faces, bc == "neumann" ? 1.0 : 0.0);
    plan.boundary_value.assign(faces, 0.0);
  }
  const std::string slot = "pops.amr.default-field";
  p_->field_plans.emplace(slot, std::move(plan));
  p_->default_field_slot = slot;
}

template <int Dim>
void AmrSystem<Dim>::set_field_solver_plan(
    const std::string& provider_slot, const std::string& plan_identity,
    const std::string& provider_identity, const std::string& output_owner_identity,
    const std::string& output_block, const std::string& output_key,
    const std::vector<std::string>& provider_identities,
    const std::vector<std::string>& provider_blocks, const std::vector<std::string>& provider_keys,
    const std::vector<double>& provider_coefficients, const std::string& solver,
    const AmrFieldHierarchyPolicyAuthority& hierarchy_policy,
    const AmrFieldSolverOptions& solver_options) {
  require_amr_assembling(p_->lifecycle, "set_field_solver_plan");
  const std::size_t count = provider_identities.size();
  if (provider_slot.empty() || plan_identity.empty() || provider_identity.empty() ||
      output_owner_identity.empty() || output_block.empty() || output_key.empty() ||
      solver.empty() || count == 0 || provider_blocks.size() != count ||
      provider_keys.size() != count || provider_coefficients.size() != count)
    throw std::invalid_argument("AMR exact field solver plan is incomplete");
  if (p_->field_plans.contains(provider_slot))
    throw std::runtime_error("AMR exact field provider slot is already installed: " +
                             provider_slot);
  for (const auto& [slot, existing] : p_->field_plans) {
    (void)slot;
    if (existing.output_block == output_block && existing.output_key == output_key)
      throw std::runtime_error("AMR exact field output block/key is already owned by another slot");
  }
  hierarchy_policy.validate();
  (void)solver_options.exact_contract();
  typename Impl::FieldPlan plan;
  plan.plan_identity = plan_identity;
  plan.provider_identity = provider_identity;
  plan.output_owner_identity = output_owner_identity;
  plan.output_block = output_block;
  plan.output_key = output_key;
  plan.solver_route = solver;
  plan.hierarchy_policy = hierarchy_policy;
  plan.solver_options = solver_options;
  plan.providers.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    if (provider_identities[index].empty() || provider_blocks[index].empty() ||
        provider_keys[index].empty() || !std::isfinite(provider_coefficients[index]))
      throw std::invalid_argument("AMR exact field provider binding is incomplete");
    plan.providers.push_back({provider_identities[index], provider_blocks[index],
                              provider_keys[index], provider_coefficients[index]});
  }
  plan.rhs_by_block.resize(p_->blocks.size());
  p_->field_plans.emplace(provider_slot, std::move(plan));
}

template <int Dim>
AmrFieldSolverConfiguration AmrSystem<Dim>::field_solver_configuration(
    const std::string& provider_slot) const {
  const auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("unknown exact AMR field provider slot");
  const auto& plan = found->second;
  return {plan.plan_identity, plan.provider_identity, plan.solver_route, plan.hierarchy_policy,
          plan.solver_options};
}

template <int Dim>
void AmrSystem<Dim>::set_field_reaction(const std::string& provider_slot, double reaction) {
  require_amr_assembling(p_->lifecycle, "set_field_reaction");
  if (!std::isfinite(reaction) || reaction <= 0.0)
    throw std::invalid_argument("AMR screened field reaction must be finite and strictly positive");
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field reaction names an unknown provider slot");
  if (found->second.prepared_solver)
    throw std::logic_error("AMR field reaction cannot change after solver materialization");
  found->second.reaction = reaction;
  found->second.has_reaction = true;
}

template <int Dim>
void AmrSystem<Dim>::set_field_topology_authority(const std::string& provider_slot,
                                                  const std::string& provider_kind,
                                                  const std::string& provenance,
                                                  const std::string& topology_digest) {
  require_amr_assembling(p_->lifecycle, "set_field_topology_authority");
  if (provider_kind.empty() || provenance.empty() || topology_digest.empty())
    throw std::invalid_argument("AMR field topology authority is incomplete");
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field topology authority names an unknown provider slot");
  if (found->second.prepared_solver)
    throw std::logic_error("AMR field topology cannot change after solver materialization");
  found->second.topology_provider_kind = provider_kind;
  found->second.topology_provenance = provenance;
  found->second.topology_digest = topology_digest;
}

template <int Dim>
std::vector<runtime::field::FieldTopologyReportRow> AmrSystem<Dim>::field_topology_report(
    const std::string& provider_slot) const {
  if (!p_->field_plans.contains(provider_slot))
    throw std::out_of_range("unknown exact AMR field provider slot");
  return {};
}

template <int Dim>
void AmrSystem<Dim>::set_field_boundary_plan(const std::string& provider_slot,
                                             const std::vector<std::string>& kind,
                                             const std::vector<double>& alpha,
                                             const std::vector<double>& beta,
                                             const std::vector<double>& value) {
  require_amr_assembling(p_->lifecycle, "set_field_boundary_plan");
  const std::size_t faces = static_cast<std::size_t>(2 * Dim);
  if (kind.size() != faces || alpha.size() != faces || beta.size() != faces ||
      value.size() != faces)
    throw std::invalid_argument(
        "AMR field boundary plan must cover both faces of every exact axis");
  for (std::size_t face = 0; face < faces; ++face) {
    if (kind[face] != "periodic" && kind[face] != "dirichlet" && kind[face] != "neumann" &&
        kind[face] != "mixed")
      throw std::invalid_argument("AMR field boundary kind is unknown");
    if (!std::isfinite(alpha[face]) || !std::isfinite(beta[face]) || !std::isfinite(value[face]))
      throw std::invalid_argument("AMR field boundary coefficients must be finite");
  }
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field boundary names an unknown provider slot");
  if (found->second.prepared_solver)
    throw std::logic_error("AMR field boundary cannot change after solver materialization");
  found->second.boundary_kind = kind;
  found->second.boundary_alpha = alpha;
  found->second.boundary_beta = beta;
  found->second.boundary_value = value;
}

template <int Dim>
void AmrSystem<Dim>::set_field_boundary_dependencies(const std::string& provider_slot,
                                                     const std::vector<std::string>& state_blocks,
                                                     const std::vector<int>& state_components,
                                                     const std::vector<std::string>& field_blocks,
                                                     const std::vector<std::string>& field_keys,
                                                     const std::vector<int>& field_components) {
  require_amr_assembling(p_->lifecycle, "set_field_boundary_dependencies");
  if (state_blocks.size() != state_components.size() || field_blocks.size() != field_keys.size() ||
      field_blocks.size() != field_components.size())
    throw std::invalid_argument("AMR field boundary dependency vectors differ in length");
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field boundary dependencies name an unknown provider slot");
  if (found->second.prepared_solver)
    throw std::logic_error(
        "AMR field boundary dependencies cannot change after solver materialization");
  found->second.boundary_state_blocks = state_blocks;
  found->second.boundary_state_components = state_components;
  found->second.boundary_field_blocks = field_blocks;
  found->second.boundary_field_keys = field_keys;
  found->second.boundary_field_components = field_components;
}

template <int Dim>
void AmrSystem<Dim>::set_field_boundary_kernel(const std::string& provider_slot,
                                               const CompiledFieldBoundaryKernel<Dim>& kernel) {
  require_amr_assembling(p_->lifecycle, "set_field_boundary_kernel");
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field boundary kernel names an unknown provider slot");
  if (found->second.prepared_solver || found->second.boundary_kernel)
    throw std::logic_error("AMR field boundary kernel is already fixed");
  kernel.validate();
  found->second.boundary_kernel = kernel;
}

template <int Dim>
void AmrSystem<Dim>::set_field_logical_timepoint(const std::string& provider_slot,
                                                 const FieldLogicalTimePoint& point) {
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field logical timepoint names an unknown provider slot");
  const bool invalid = !std::isfinite(static_cast<double>(point.time)) ||
                       !std::isfinite(static_cast<double>(point.dt)) || point.dt <= Real(0) ||
                       point.clock_slot < 0 || point.partition_slot < 0 || point.stage_slot < 0 ||
                       point.level < 0 || point.step < 0 || point.substep < 0 ||
                       point.iteration < 0;
  if (all_reduce_max(invalid ? 1L : 0L) != 0)
    throw std::invalid_argument("AMR field logical timepoint is incomplete");
  found->second.boundary_point = point;
}

template <int Dim>
void AmrSystem<Dim>::set_field_boundary_parameters(const std::string& provider_slot,
                                                   const std::vector<double>& parameters) {
  require_amr_assembling(p_->lifecycle, "set_field_boundary_parameters");
  if (!std::all_of(parameters.begin(), parameters.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::invalid_argument("AMR field boundary parameters must be finite");
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field boundary parameters name an unknown provider slot");
  if (found->second.prepared_solver)
    throw std::logic_error(
        "AMR field boundary parameters cannot change after solver materialization");
  found->second.boundary_parameters.assign(parameters.begin(), parameters.end());
}

template <int Dim>
void AmrSystem<Dim>::set_field_newton_plan(const std::string& provider_slot, double tolerance,
                                           int max_iterations, double linear_tolerance,
                                           int linear_max_iterations, int restart, double armijo,
                                           double minimum_step) {
  require_amr_assembling(p_->lifecycle, "set_field_newton_plan");
  FieldNewtonOptions options{
      static_cast<Real>(tolerance),   max_iterations, static_cast<Real>(linear_tolerance),
      linear_max_iterations,          restart,        static_cast<Real>(armijo),
      static_cast<Real>(minimum_step)};
  validate_field_newton_options(options);
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field Newton plan names an unknown provider slot");
  if (found->second.prepared_solver)
    throw std::logic_error("AMR field Newton plan cannot change after solver materialization");
  found->second.newton = options;
}

template <int Dim>
void AmrSystem<Dim>::set_field_nullspace(const std::string& provider_slot,
                                         const std::string& nullspace_provider_identity,
                                         const PreparedProviderOptions& options) {
  require_amr_assembling(p_->lifecycle, "set_field_nullspace");
  if (nullspace_provider_identity.empty())
    throw std::invalid_argument("AMR field nullspace provider identity must be non-empty");
  (void)options.exact_contract();
  auto found = p_->field_plans.find(provider_slot);
  if (found == p_->field_plans.end())
    throw std::out_of_range("AMR field nullspace names an unknown provider slot");
  if (found->second.prepared_solver)
    throw std::logic_error("AMR field nullspace cannot change after solver materialization");
  found->second.nullspace_provider_identity = nullspace_provider_identity;
  found->second.nullspace_options = options;
}

template <int Dim>
void AmrSystem<Dim>::register_elliptic_field(const std::string& block_name,
                                             const std::string& provider_key,
                                             const std::vector<int>& output_components,
                                             int gradient_sign) {
  require_amr_assembling(p_->lifecycle, "register_elliptic_field");
  if (p_->engine)
    throw std::runtime_error(
        "AmrSystem cannot register a named elliptic field after hierarchy materialization");
  if (provider_key.empty())
    throw std::invalid_argument("AmrSystem named elliptic field identity must be non-empty");
  (void)p_->block(block_name);
  const runtime::field::NamedFieldOutput<Dim> output(output_components, gradient_sign);
  const std::string slot = p_->resolve_field_slot(provider_key);
  typename Impl::FieldPlan& plan = p_->field_plans.at(slot);
  if (plan.output)
    throw std::logic_error("AMR exact field output is already registered");
  if (plan.output_block != block_name || plan.output_key != provider_key)
    throw std::invalid_argument(
        "AMR exact field output registration differs from its resolved plan");
  plan.output = output;
  plan.rhs_by_block.resize(p_->blocks.size());
}

template <int Dim>
void AmrSystem<Dim>::set_block_elliptic_field(
    const std::string& block_name, const std::string& field,
    std::function<void(const MultiFab<Dim>&, MultiFab<Dim>&)> rhs) {
  require_amr_assembling(p_->lifecycle, "set_block_elliptic_field");
  if (p_->engine)
    throw std::runtime_error(
        "AmrSystem cannot install a named elliptic RHS after hierarchy materialization");
  if (field.empty() || !rhs)
    throw std::invalid_argument(
        "AmrSystem named elliptic RHS requires a field identity and prepared closure");
  (void)p_->block(block_name);
  const std::string slot = field == "fields_from_state" && !p_->default_field_slot.empty()
                               ? p_->default_field_slot
                               : p_->resolve_field_slot(field);
  typename Impl::FieldPlan& plan = p_->field_plans.at(slot);
  if (plan.prepared_solver)
    throw std::logic_error("AMR exact field RHS cannot change after solver materialization");
  Real coefficient = Real(0);
  bool contributes = false;
  if (slot == p_->default_field_slot) {
    coefficient = Real(1);
    contributes = true;
    plan.use_prepared_level_rhs = false;
  } else {
    for (const typename Impl::FieldProviderBinding& binding : plan.providers)
      if (binding.block == block_name) {
        coefficient += static_cast<Real>(binding.coefficient);
        contributes = true;
      }
  }
  if (!contributes)
    throw std::invalid_argument("AMR exact field RHS has no resolved provider binding");
  plan.rhs_by_block.resize(p_->blocks.size());
  const std::size_t block_index = static_cast<std::size_t>(std::distance(
      p_->blocks.begin(), std::find_if(p_->blocks.begin(), p_->blocks.end(),
                                       [&](const typename Impl::BlockSpec& candidate) {
                                         return candidate.name == block_name;
                                       })));
  if (block_index >= plan.rhs_by_block.size())
    throw std::logic_error("AMR exact field RHS lost its runtime block identity");
  plan.rhs_by_block[block_index].push_back({std::move(rhs), coefficient});
}

template <int Dim>
SolveOutcome AmrSystem<Dim>::solve_program_field_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
    int active_level, const MultiFab<Dim>* stage_override) {
  if (point.clock.empty() || point.level != active_level || point.stage < 0 ||
      point.stage_fraction.denominator <= 0 || !std::isfinite(point.dt) || point.dt <= 0.0 ||
      !std::isfinite(point.physical_time))
    throw std::invalid_argument("AMR exact field solve has an invalid evaluation point");
  if (provider_slot.empty())
    throw std::invalid_argument("AMR exact field solve requires a provider slot");
  ExactContractBuilder request;
  request.text("pops.amr.program-field-solve")
      .scalar(std::uint32_t{1})
      .scalar(std::int32_t{Dim})
      .text(provider_slot)
      .text(point.clock)
      .scalar(point.tick)
      .scalar(point.level)
      .scalar(point.substep)
      .scalar(point.stage)
      .scalar(point.stage_fraction.numerator)
      .scalar(point.stage_fraction.denominator)
      .scalar(point.dt)
      .scalar(point.physical_time)
      .presence(stage_override != nullptr);
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{"amr-program-field-solve", std::move(request).release()}}))
    throw std::invalid_argument("AMR exact field solve request differs between MPI ranks");

  const std::string slot = p_->resolve_field_slot(provider_slot);
  std::vector<const MultiFab<Dim>*> stages(p_->blocks.size(), nullptr);
  if (stage_override != nullptr) {
    if (stages.size() != 1)
      throw std::logic_error(
          "AMR exact field stage override requires the prepared multi-block hierarchy provider");
    stages.front() = stage_override;
  }
  SolveReport report;
  std::exception_ptr local_error;
  try {
    report = p_->solve_field_candidate(slot, active_level, stages, &point);
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L) != 0) {
    p_->reject_field_candidate();
    if (n_ranks() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR exact field solve failed on at least one MPI rank");
  }
  return p_->make_field_outcome(std::move(report));
}

template <int Dim>
SolveOutcome AmrSystem<Dim>::solve_program_field_from_blocks_at(
    const runtime::multiblock::BoundaryEvaluationPoint& point, const std::string& provider_slot,
    int active_level, const std::vector<const MultiFab<Dim>*>& stage_overrides) {
  if (stage_overrides.size() != p_->blocks.size())
    throw std::invalid_argument(
        "AMR exact simultaneous field solve must cover the runtime block registry");
  if (stage_overrides.size() != 1)
    throw std::logic_error(
        "AMR exact simultaneous field solve requires the prepared multi-block hierarchy provider");
  return solve_program_field_at(point, provider_slot, active_level, stage_overrides.front());
}

template <int Dim>
SolveOutcome AmrSystem<Dim>::solve_program_default_field(int active_level) {
  if (p_->default_field_slot.empty())
    throw std::logic_error("AmrSystem has no configured default exact field");
  p_->ensure_engine();
  if (active_level < 0 ||
      static_cast<std::size_t>(active_level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AMR default field active level lies outside the hierarchy");
  SolveReport report;
  std::exception_ptr local_error;
  try {
    report =
        p_->solve_field_candidate(p_->default_field_slot, active_level,
                                  std::vector<const MultiFab<Dim>*>(p_->blocks.size(), nullptr));
  } catch (...) {
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_error ? 1L : 0L) != 0) {
    p_->reject_field_candidate();
    if (n_ranks() == 1 && local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR default exact field solve failed on at least one MPI rank");
  }
  return p_->make_field_outcome(std::move(report));
}

template <int Dim>
std::vector<std::string> AmrSystem<Dim>::field_provider_slots() const {
  std::vector<std::string> result;
  result.reserve(p_->field_plans.size());
  for (const auto& [slot, plan] : p_->field_plans) {
    (void)plan;
    result.push_back(slot);
  }
  return result;
}

template <int Dim>
int AmrSystem<Dim>::field_provider_levels(const std::string& provider_slot) {
  const std::string slot = p_->resolve_field_slot(provider_slot);
  p_->materialize_field(slot);
  return p_->field_plans.at(slot).prepared_solver->level_count();
}

template <int Dim>
void AmrSystem<Dim>::set_field_potential_level(const std::string& provider_slot, int level,
                                               const std::vector<double>& phi) {
  const std::string slot = p_->resolve_field_slot(provider_slot);
  p_->materialize_field(slot);
  typename Impl::FieldPlan& plan = p_->field_plans.at(slot);
  if (!p_->active_field_slot.empty())
    throw std::logic_error("AMR field potential cannot be restored during an active solve");
  if (level < 0 || level >= plan.prepared_solver->level_count())
    throw std::out_of_range("AMR field potential restore level lies outside the hierarchy");
  const Box<Dim>& domain = p_->engine->hierarchy().layout(static_cast<std::size_t>(level)).domain();
  write_field(*plan.accepted_potential[static_cast<std::size_t>(level)], domain, phi, 1);
  copy_full_field_in_place(*plan.accepted_potential[static_cast<std::size_t>(level)],
                           plan.prepared_solver->candidate_level(level));
}

template <int Dim>
void AmrSystem<Dim>::set_field_potential(const std::string& provider_slot,
                                         const std::vector<double>& phi) {
  set_field_potential_level(provider_slot, 0, phi);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::field_potential_level_global(const std::string& provider_slot,
                                                                 int level) {
  const std::string slot = p_->resolve_field_slot(provider_slot);
  p_->materialize_field(slot);
  const typename Impl::FieldPlan& plan = p_->field_plans.at(slot);
  if (level < 0 || level >= plan.prepared_solver->level_count())
    throw std::out_of_range("AMR field potential level lies outside the hierarchy");
  return gather_field(*plan.accepted_potential[static_cast<std::size_t>(level)],
                      p_->engine->hierarchy().layout(static_cast<std::size_t>(level)).domain(), 1);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::field_potential_global(const std::string& provider_slot) {
  return field_potential_level_global(provider_slot, 0);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::named_field_values(const std::string& field) {
  const std::string slot = p_->resolve_field_slot(field);
  runtime::multiblock::BoundaryEvaluationPoint point;
  point.clock = "pops.amr.direct-field-read";
  point.level = 0;
  point.stage_fraction = {0, 1};
  point.dt = 1.0;
  point.physical_time = p_->accepted_time;
  (void)consume_solve_outcome(solve_program_field_at(point, slot, 0, nullptr));
  return field_potential_level_global(slot, 0);
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_field_local_pieces(
    const std::string& provider_slot, int level) {
  const std::string slot = p_->resolve_field_slot(provider_slot);
  p_->materialize_field(slot);
  const typename Impl::FieldPlan& plan = p_->field_plans.at(slot);
  if (level < 0 || level >= plan.prepared_solver->level_count())
    throw std::out_of_range("AMR field output level lies outside the hierarchy");
  const MultiFab<Dim>& field = *plan.accepted_potential[static_cast<std::size_t>(level)];
  return output_local_pieces(field, level, field.distribution().replicated());
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_field_root_pieces(
    const ObserverMpiLane& lane, const std::string& provider_slot, int level) {
  return output_pieces_to_root(
      lane, detail::output_collective_identity("amr_system", "field", provider_slot, level),
      [&] { return output_field_local_pieces(provider_slot, level); });
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::level_potential(int level) {
  return level_potential_global(level);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::level_potential_global(int level) {
  if (p_->default_field_slot.empty())
    throw std::logic_error("AmrSystem has no configured default exact field");
  return field_potential_level_global(p_->default_field_slot, level);
}

template <int Dim>
void AmrSystem<Dim>::set_level_potential(int level, const std::vector<double>& phi) {
  if (p_->default_field_slot.empty())
    throw std::logic_error("AmrSystem has no configured default exact field");
  set_field_potential_level(p_->default_field_slot, level, phi);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::potential() {
  if (p_->default_field_slot.empty())
    throw std::logic_error("AmrSystem has no configured default exact field");
  (void)consume_solve_outcome(solve_program_default_field(0));
  return field_potential_level_global(p_->default_field_slot, 0);
}

template <int Dim>
void AmrSystem<Dim>::install_hyperbolic_boundary(
    const std::string& name, const std::string& identity, int required_depth,
    const std::vector<std::string>& face_types, const std::vector<double>& face_values,
    const std::vector<std::string>& face_identities,
    const std::vector<std::string>& component_roles, const std::string& state_identity,
    const std::vector<std::string>& face_representations,
    const std::vector<std::string>& face_converter_identities,
    const std::vector<std::vector<std::string>>& face_analytic_opcodes,
    const std::vector<std::vector<double>>& face_analytic_literals,
    const std::vector<std::string>& face_analytic_clocks) {
  require_amr_assembling(p_->lifecycle, "install_hyperbolic_boundary");
  if (!p_->blocks.empty() || p_->engine)
    throw std::logic_error("AmrSystem boundaries must be prepared before their block");
  p_->boundary_registry.install_boundary(
      name, identity, required_depth, face_types, face_values, face_identities, component_roles,
      state_identity, face_representations, face_converter_identities, face_analytic_opcodes,
      face_analytic_literals, face_analytic_clocks);
}

template <int Dim>
void AmrSystem<Dim>::install_prepared_hyperbolic_boundary(
    const std::string& name, const std::string& identity, int required_depth,
    const std::string& state_identity, std::shared_ptr<const HyperbolicBoundary> boundary) {
  require_amr_assembling(p_->lifecycle, "install_prepared_hyperbolic_boundary");
  if (!p_->blocks.empty() || p_->engine)
    throw std::logic_error("AmrSystem boundaries must be prepared before their block");
  p_->boundary_registry.install_boundary(name, identity, required_depth, state_identity,
                                         std::move(boundary));
}

template <int Dim>
void AmrSystem<Dim>::discard_hyperbolic_boundaries() {
  require_amr_assembling(p_->lifecycle, "discard_hyperbolic_boundaries");
  if (!p_->blocks.empty())
    throw std::logic_error("AmrSystem cannot discard state routes after block publication");
  p_->boundary_registry.discard_transaction();
}

template <int Dim>
void AmrSystem<Dim>::set_density(const std::string& name, const std::vector<double>& density) {
  typename Impl::BlockSpec& block = p_->block(name);
  if (density.size() != checked_cells(p_->cfg.index_domain()))
    throw std::invalid_argument("AmrSystem density differs from the exact coarse shape");
  if (p_->engine)
    write_component(p_->engine->hierarchy().state(0), p_->cfg.index_domain(), density, 0);
  p_->discard_level_evaluations();
  block.density = density;
  block.has_density = true;
}

template <int Dim>
void AmrSystem<Dim>::set_conservative_state(const std::string& name,
                                            const std::vector<double>& state) {
  typename Impl::BlockSpec& block = p_->block(name);
  const std::size_t cells = checked_cells(p_->cfg.index_domain());
  if (state.size() != static_cast<std::size_t>(block.ncomp) * cells)
    throw std::invalid_argument("AmrSystem state differs from the exact coarse shape");
  if (p_->engine)
    write_field(p_->engine->hierarchy().state(0), p_->cfg.index_domain(), state, block.ncomp);
  p_->discard_level_evaluations();
  block.state = state;
  block.has_state = true;
}

template <int Dim>
runtime::amr::PreparedTaggerCandidates<Dim> AmrSystem<Dim>::execute_prepared_tagging(
    int parent_level) {
  p_->ensure_engine();
  return p_->execute_tagging(parent_level);
}

template <int Dim>
bool AmrSystem<Dim>::regrid_from_prepared_tagging(int parent_level) {
  return p_->execute_transaction([&] { return p_->regrid_parent(parent_level); });
}

template <int Dim>
void AmrSystem<Dim>::begin_bootstrap_plan() {
  if (!p_->cfg.explicit_bootstrap)
    throw std::logic_error("AmrSystem explicit bootstrap is disabled by the resolved layout");
  if (p_->accepted_time != 0.0 || p_->macro_step != 0)
    throw std::logic_error("AmrSystem bootstrap requires the accepted t=0/step=0 state");
  if (!p_->tagging_spec)
    throw std::logic_error("AmrSystem bootstrap requires a prepared tagging authority");
  if (p_->bootstrap_transaction)
    throw std::logic_error("AmrSystem bootstrap transaction is already active");
  p_->ensure_engine();
  auto transaction = std::make_unique<typename Impl::AcceptedSnapshot>(*p_);
  runtime::amr::PersistentTaggingState<Dim> staged_state = p_->tagging_state;
  staged_state.begin_cycle(p_->tagging_spec->min_cycles);
  p_->bootstrap_transaction = std::move(transaction);
  p_->tagging_state = std::move(staged_state);
}

template <int Dim>
bool AmrSystem<Dim>::bootstrap_next_level() {
  if (!p_->bootstrap_transaction)
    throw std::logic_error("AmrSystem bootstrap level requires an active transaction");
  if (p_->accepted_time != 0.0 || p_->macro_step != 0)
    throw std::logic_error("AmrSystem bootstrap cannot advance the authoritative clock");
  const int parent_level = static_cast<int>(p_->engine->hierarchy().num_levels()) - 1;
  if (parent_level >= p_->cfg.level_count - 1)
    throw std::out_of_range("AmrSystem bootstrap would exceed the resolved hierarchy depth");
  return p_->regrid_parent(parent_level, std::nullopt, &p_->tagging_state);
}

template <int Dim>
void AmrSystem<Dim>::commit_bootstrap_level() {
  if (!p_->bootstrap_transaction)
    throw std::logic_error("AmrSystem bootstrap commit has no active transaction");
  if (p_->accepted_time != 0.0 || p_->macro_step != 0)
    throw std::logic_error("AmrSystem bootstrap commit requires the accepted t=0/step=0 state");
  p_->program.refresh_hierarchy_state("AmrSystem::commit_bootstrap_level");
  p_->publish_tagging_checkpoint();
  p_->bootstrap_transaction.reset();
  p_->automatic_bootstrap_complete = true;
}

template <int Dim>
void AmrSystem<Dim>::rollback_bootstrap_level() {
  if (!p_->bootstrap_transaction)
    throw std::logic_error("AmrSystem bootstrap rollback has no active transaction");
  p_->bootstrap_transaction->restore(*p_);
  p_->bootstrap_transaction.reset();
}

template <int Dim>
void AmrSystem<Dim>::register_bootstrap_transfer_route(
    const std::string& identity, const std::vector<std::string>& subjects,
    const std::string& provider_identity, const std::string& space, const std::string& centering,
    const std::string& representation, const std::string& storage, const std::string& operation,
    const std::string& kernel, int order, const Extent<Dim>& ghost_depth,
    const Extent<Dim>& refinement_ratio) {
  std::map<std::string, typename Impl::BootstrapTransferRoute> staged_routes;
  std::map<std::pair<std::string, std::string>, std::string> staged_subjects;
  std::string exact;
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    require_amr_assembling(p_->lifecycle, "register_bootstrap_transfer_route");
    if (p_->engine || identity.empty() || subjects.empty() || provider_identity.empty() ||
        space.empty() || centering.empty() || representation.empty() || storage.empty() ||
        operation.empty() || kernel.empty() || order < 1)
      throw std::invalid_argument(
          "AMR bootstrap transfer requires one complete pre-materialization provider route");
    std::set<std::string> unique_subjects;
    for (const std::string& subject : subjects)
      if (subject.empty() || !unique_subjects.insert(subject).second)
        throw std::invalid_argument("AMR bootstrap transfer subjects must be non-empty and unique");
    bool ratio_matches_transition = p_->cfg.transition_ratios.empty();
    for (int axis = 0; axis < Dim; ++axis)
      if (ghost_depth[axis] < 0 || refinement_ratio[axis] < 1)
        throw std::invalid_argument(
            "AMR bootstrap transfer ghosts and ranked refinement ratio are invalid");
    for (const Extent<Dim>& configured : p_->cfg.transition_ratios)
      ratio_matches_transition = ratio_matches_transition || configured == refinement_ratio;
    if (!ratio_matches_transition)
      throw std::invalid_argument(
          "AMR bootstrap transfer route names no configured ranked hierarchy transition");
    if (operation == "prolongation") {
      if (space != "cell" || centering != "cell" || representation != "conservative" ||
          storage != "dense")
        throw std::invalid_argument(
            "AMR state prolongation requires the exact cell-centered conservative dense route");
      if (kernel == "conservative_linear") {
        bool lacks_stencil = false;
        for (int axis = 0; axis < Dim; ++axis)
          lacks_stencil = lacks_stencil || ghost_depth[axis] < 1;
        if (order != 2 || lacks_stencil)
          throw std::invalid_argument(
              "AMR conservative-linear prolongation requires order two and one ranked ghost");
      } else if (kernel == "conservative_injection") {
        bool has_stencil = false;
        for (int axis = 0; axis < Dim; ++axis)
          has_stencil = has_stencil || ghost_depth[axis] != 0;
        if (order != 1 || has_stencil)
          throw std::invalid_argument(
              "AMR conservative injection requires explicit order one and zero ghosts");
      } else {
        throw std::invalid_argument("AMR prolongation kernel has no exact native provider");
      }
    }

    typename Impl::BootstrapTransferRoute candidate{identity, subjects,    provider_identity,
                                                    space,    centering,   representation,
                                                    storage,  operation,   kernel,
                                                    order,    ghost_depth, refinement_ratio};
    ExactContractBuilder contract;
    contract.text("pops.amr-system.bootstrap-transfer-route")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .text(identity)
        .text(provider_identity)
        .text(space)
        .text(centering)
        .text(representation)
        .text(storage)
        .text(operation)
        .text(kernel)
        .scalar(order)
        .sequence(subjects, [](ExactContractBuilder& item, const std::string& subject) {
          item.text(subject);
        });
    for (int axis = 0; axis < Dim; ++axis)
      contract.scalar(ghost_depth[axis]).scalar(refinement_ratio[axis]);
    exact = std::move(contract).release();

    staged_routes = p_->bootstrap_transfer_routes;
    staged_subjects = p_->bootstrap_subject_routes;
    if (!staged_routes.emplace(identity, std::move(candidate)).second)
      throw std::invalid_argument("AMR bootstrap transfer route identity is not unique");
    for (const std::string& subject : subjects)
      if (!staged_subjects.emplace(std::make_pair(subject, operation), identity).second)
        throw std::invalid_argument(
            "AMR bootstrap transfer subject already owns this operation route");
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  if (all_reduce_max(local_failure) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR bootstrap transfer route failed validation on another MPI rank");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("bootstrap-transfer-route"), std::string_view(exact)}}))
    throw std::invalid_argument("AMR bootstrap transfer route differs between MPI ranks");
  p_->bootstrap_transfer_routes = std::move(staged_routes);
  p_->bootstrap_subject_routes = std::move(staged_subjects);
}

template <int Dim>
void AmrSystem<Dim>::add_dt_bound(const std::string& label, std::function<double()> evaluate) {
  require_amr_assembling(p_->lifecycle, "add_dt_bound");
  if (label.empty() || !evaluate)
    throw std::invalid_argument("AmrSystem dt bound requires a label and provider");
  p_->dt_bounds.push_back(typename Impl::GlobalDtBound{label, std::move(evaluate)});
}

template <int Dim>
std::string AmrSystem<Dim>::last_dt_bound() const {
  return p_->last_dt_reason;
}

template <int Dim>
void AmrSystem<Dim>::step(double dt) {
  p_->program.require_step_installed("AmrSystem::step");
  runtime::program::ProfileScope scope(p_->program.profiler_, "step");
  p_->program.profiler_.count("steps");
  if (p_->bootstrap_transaction)
    throw std::logic_error("AmrSystem cannot step during an active bootstrap transaction");
  p_->execute_transaction([&] {
    p_->program.dispatch_cadence_step(p_->accepted_time, p_->macro_step, dt, "AmrSystem");
    if (!p_->tagging_spec || p_->cfg.regrid_every == 0 ||
        p_->macro_step % p_->cfg.regrid_every != 0)
      return;
    std::vector<std::optional<SparseFieldImage<Dim>>> previous(
        p_->engine->hierarchy().num_levels());
    for (std::size_t level = 1; level < p_->engine->hierarchy().num_levels(); ++level)
      previous[level] = gather_sparse_field(p_->engine->hierarchy().state(level),
                                            p_->engine->hierarchy().layout(level).domain(),
                                            p_->prepared_hierarchy->lane->communicator());
    runtime::amr::PersistentTaggingState<Dim> staged_state = p_->tagging_state;
    staged_state.begin_cycle(p_->tagging_spec->min_cycles);
    for (int parent_level = 0; parent_level < p_->cfg.level_count - 1; ++parent_level) {
      const std::size_t child = static_cast<std::size_t>(parent_level + 1);
      const std::optional<SparseFieldImage<Dim>> old_child =
          child < previous.size() ? previous[child] : std::nullopt;
      if (!p_->regrid_parent(parent_level, old_child, &staged_state))
        break;
    }
    p_->tagging_state = std::move(staged_state);
    p_->publish_tagging_checkpoint();
  });
  p_->discard_level_evaluations();
}

template <int Dim>
void AmrSystem<Dim>::advance(double dt, int nsteps) {
  p_->program.require_step_installed("AmrSystem::advance");
  if (nsteps < 0)
    throw std::invalid_argument("AmrSystem::advance requires a non-negative step count");
  for (int step_index = 0; step_index < nsteps; ++step_index)
    step(dt);
}

template <int Dim>
double AmrSystem<Dim>::step_cfl(double cfl, double speed_floor, double max_dt, double min_dt) {
  std::string request_contract;
  std::exception_ptr request_error;
  long request_failure = 0;
  try {
    p_->program.require_step_installed("AmrSystem::step_cfl");
    if (!std::isfinite(cfl) || cfl <= 0.0 || !std::isfinite(speed_floor) || speed_floor <= 0.0)
      throw std::invalid_argument("AmrSystem::step_cfl requires positive finite CFL inputs");
    if (std::isnan(max_dt) || max_dt <= 0.0 || !std::isfinite(min_dt) || min_dt < 0.0)
      throw std::invalid_argument("AmrSystem::step_cfl received invalid strategy bounds");
    if (p_->blocks.size() != 1 || !p_->prepared_block)
      throw std::logic_error("AmrSystem::step_cfl requires one retained generated block");

    ExactContractBuilder contract;
    contract.text("pops.amr-system.step-cfl-request")
        .scalar(std::uint32_t{1})
        .scalar(std::int32_t{Dim})
        .scalar(cfl)
        .scalar(speed_floor)
        .scalar(max_dt)
        .scalar(min_dt)
        .bytes(p_->prepared_block->collective_contract)
        .scalar(static_cast<std::uint64_t>(p_->dt_bounds.size()));
    for (const typename Impl::GlobalDtBound& bound : p_->dt_bounds)
      contract.text(bound.label);
    contract.scalar(static_cast<bool>(p_->program.dt_bound_));
    request_contract = std::move(contract).release();
  } catch (...) {
    request_failure = 1;
    request_error = std::current_exception();
  }
  if (all_reduce_max(request_failure) != 0) {
    if (request_error)
      std::rethrow_exception(request_error);
    throw std::runtime_error("AmrSystem::step_cfl request validation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-step-cfl-request"), std::string_view(request_contract)}}))
    throw std::invalid_argument(
        "AmrSystem::step_cfl inputs or prepared bound authorities differ between MPI ranks");

  p_->ensure_engine();
  const long invalid_hierarchy =
      !p_->prepared_hierarchy || p_->prepared_hierarchy->levels.empty() ? 1L : 0L;
  if (all_reduce_max(invalid_hierarchy) != 0)
    throw std::logic_error("AmrSystem::step_cfl requires one live prepared hierarchy graph");

  enum class BoundKind : std::int32_t {
    degenerate,
    transport,
    source_frequency,
    stability_dt,
    global,
    program,
    maximum_dt,
  };
  const typename Impl::BlockSpec& block = p_->blocks.front();
  double selected = std::numeric_limits<double>::infinity();
  BoundKind reason_kind = BoundKind::degenerate;
  std::size_t global_reason_index = std::numeric_limits<std::size_t>::max();
  for (std::size_t level = 0; level < p_->prepared_hierarchy->levels.size(); ++level) {
    const Box<Dim>& domain = p_->engine->hierarchy().layout(level).domain();
    const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, p_->cfg.lower, p_->cfg.upper);
    Real spacing = geometry.spacing(0);
    for (int axis = 1; axis < Dim; ++axis)
      spacing = std::min(spacing, geometry.spacing(axis));

    const typename Impl::level_block_type& prepared_level = p_->prepared_hierarchy->levels[level];
    const Real speed = std::max(prepared_level.maximum_speed(), static_cast<Real>(speed_floor));
    double level_dt = cfl * static_cast<double>(spacing) * block.substeps /
                      (static_cast<double>(block.stride) * static_cast<double>(speed));
    const char* level_reason = "transport";
    if (const std::optional<Real> frequency = prepared_level.source_frequency();
        frequency && *frequency > Real(0)) {
      const double source_dt =
          cfl * block.substeps /
          (static_cast<double>(block.stride) * static_cast<double>(*frequency));
      if (source_dt < level_dt) {
        level_dt = source_dt;
        level_reason = "source_frequency";
      }
    }
    if (const std::optional<Real> admissible = prepared_level.stability_dt();
        admissible && *admissible > Real(0)) {
      const double stability_dt =
          static_cast<double>(*admissible) * block.substeps / static_cast<double>(block.stride);
      if (stability_dt < level_dt) {
        level_dt = stability_dt;
        level_reason = "stability_dt";
      }
    }
    if (level_dt < selected) {
      selected = level_dt;
      if (std::string_view(level_reason) == "source_frequency")
        reason_kind = BoundKind::source_frequency;
      else if (std::string_view(level_reason) == "stability_dt")
        reason_kind = BoundKind::stability_dt;
      else
        reason_kind = BoundKind::transport;
    }
  }
  if (!std::isfinite(selected))
    throw std::runtime_error("AmrSystem::step_cfl found no finite generated stability bound");

  for (std::size_t bound_index = 0; bound_index < p_->dt_bounds.size(); ++bound_index) {
    const typename Impl::GlobalDtBound& bound = p_->dt_bounds[bound_index];
    double candidate = std::numeric_limits<double>::infinity();
    std::exception_ptr bound_error;
    long bound_failure = 0;
    try {
      candidate = bound.evaluate();
    } catch (...) {
      bound_failure = 1;
      bound_error = std::current_exception();
    }
    if (all_reduce_max(bound_failure) != 0) {
      if (bound_error)
        std::rethrow_exception(bound_error);
      throw std::runtime_error("AmrSystem global dt-bound evaluation failed collectively");
    }
    const long active = std::isfinite(candidate) && candidate > 0.0 ? 1L : 0L;
    if (all_reduce_min(active) != all_reduce_max(active))
      throw std::invalid_argument("AmrSystem global dt-bound activity differs between MPI ranks");
    if (active == 0)
      candidate = std::numeric_limits<double>::infinity();
    candidate = all_reduce_min(candidate);
    if (candidate < selected) {
      selected = candidate;
      reason_kind = BoundKind::global;
      global_reason_index = bound_index;
    }
  }
  const bool has_program_bound = static_cast<bool>(p_->program.dt_bound_);
  if (has_program_bound) {
    double candidate = std::numeric_limits<double>::infinity();
    std::exception_ptr bound_error;
    long bound_failure = 0;
    try {
      candidate = static_cast<double>(p_->program.dt_bound_(static_cast<Real>(cfl)));
    } catch (...) {
      bound_failure = 1;
      bound_error = std::current_exception();
    }
    if (all_reduce_max(bound_failure) != 0) {
      if (bound_error)
        std::rethrow_exception(bound_error);
      throw std::runtime_error("AmrSystem Program dt-bound evaluation failed collectively");
    }
    const long active = std::isfinite(candidate) && candidate > 0.0 ? 1L : 0L;
    if (all_reduce_min(active) != all_reduce_max(active))
      throw std::invalid_argument("AmrSystem Program dt-bound activity differs between MPI ranks");
    if (active == 0)
      candidate = std::numeric_limits<double>::infinity();
    candidate = all_reduce_min(candidate);
    if (std::isfinite(candidate) && candidate > 0.0 && candidate < selected) {
      selected = candidate;
      reason_kind = BoundKind::program;
    }
  }
  if (max_dt < selected) {
    selected = max_dt;
    reason_kind = BoundKind::maximum_dt;
  }
  if (selected < min_dt)
    throw std::runtime_error("AmrSystem::step_cfl stability bound is below declared min_dt");

  std::string reason;
  std::string decision_contract;
  std::exception_ptr decision_error;
  long decision_failure = 0;
  try {
    switch (reason_kind) {
      case BoundKind::transport:
        reason = "transport:" + block.name;
        break;
      case BoundKind::source_frequency:
        reason = "source_frequency:" + block.name;
        break;
      case BoundKind::stability_dt:
        reason = "stability_dt:" + block.name;
        break;
      case BoundKind::global:
        if (global_reason_index >= p_->dt_bounds.size())
          throw std::logic_error("AmrSystem::step_cfl lost its selected global bound identity");
        reason = "global:" + p_->dt_bounds[global_reason_index].label;
        break;
      case BoundKind::program:
        reason = "program:dt_bound";
        break;
      case BoundKind::maximum_dt:
        reason = "strategy:max_dt";
        break;
      case BoundKind::degenerate:
        reason = "degenerate";
        break;
    }
    ExactContractBuilder contract;
    contract.text("pops.amr-system.step-cfl-decision")
        .scalar(std::uint32_t{1})
        .scalar(selected)
        .scalar(reason_kind)
        .scalar(static_cast<std::uint64_t>(global_reason_index))
        .text(reason);
    decision_contract = std::move(contract).release();
  } catch (...) {
    decision_failure = 1;
    decision_error = std::current_exception();
  }
  if (all_reduce_max(decision_failure) != 0) {
    if (decision_error)
      std::rethrow_exception(decision_error);
    throw std::runtime_error("AmrSystem::step_cfl decision preparation failed collectively");
  }
  if (!all_ranks_agree_exact_ordered_byte_pairs(
          {{std::string_view("amr-step-cfl-decision"), std::string_view(decision_contract)}}))
    throw std::runtime_error("AmrSystem::step_cfl selected different bounds across MPI ranks");
  p_->last_dt_reason = std::move(reason);
  step(selected);
  return selected;
}

template <int Dim>
void AmrSystem<Dim>::begin_step_transaction() {
  p_->ensure_engine();
  if (p_->external_step_transaction)
    throw std::runtime_error("AmrSystem step transaction is already active");
  p_->external_step_transaction = std::make_unique<typename Impl::AcceptedSnapshot>(*p_);
  p_->external_step_committed = false;
}

template <int Dim>
void AmrSystem<Dim>::commit_step_transaction() {
  if (!p_->external_step_transaction || p_->external_step_committed)
    throw std::runtime_error("AmrSystem has no active uncommitted step transaction");
  p_->external_step_committed = true;
}

template <int Dim>
void AmrSystem<Dim>::finalize_step_transaction() {
  if (!p_->external_step_transaction || !p_->external_step_committed)
    throw std::runtime_error("AmrSystem has no committed step transaction");
  p_->external_step_transaction.reset();
  p_->external_step_committed = false;
}

template <int Dim>
void AmrSystem<Dim>::rollback_step_transaction() {
  if (!p_->external_step_transaction)
    throw std::runtime_error("AmrSystem has no active step transaction");
  p_->external_step_transaction->restore(*p_);
  p_->external_step_transaction.reset();
  p_->external_step_committed = false;
}

template <int Dim>
bool AmrSystem<Dim>::has_active_step_transaction() const noexcept {
  return static_cast<bool>(p_->external_step_transaction);
}

template <int Dim>
void AmrSystem<Dim>::restore_active_step_transaction_for_program() {
  if (!p_->external_step_transaction)
    throw std::runtime_error("AmrSystem has no outer Program rollback image");
  p_->external_step_transaction->restore(*p_);
}

template <int Dim>
std::map<std::string, double> AmrSystem<Dim>::step_change_l2() const {
  if (!p_->external_step_transaction || !p_->engine || !p_->external_step_transaction->engine)
    throw std::runtime_error("AmrSystem::step_change_l2 requires an active transaction");
  if (p_->engine->hierarchy().num_levels() != 1)
    throw std::runtime_error(
        "AmrSystem::step_change_l2 requires a prepared composite coverage provider");
  const MultiFab<Dim>& current = p_->engine->hierarchy().state(0);
  const MultiFab<Dim>& previous = p_->external_step_transaction->engine->hierarchy.state(0);
  const double value = std::sqrt(cell_measure(p_->cfg, p_->cfg.index_domain()) *
                                 static_cast<double>(difference_sum_sq_all(current, previous)));
  return {{p_->blocks.front().name, value}};
}

template <int Dim>
void AmrSystem<Dim>::install_program_step(std::function<void(double)> step) {
  p_->program.install_unverified_step(std::move(step));
}

template <int Dim>
void AmrSystem<Dim>::install_program_hierarchy_refresh(std::function<void()> refresh) {
  p_->program.install_hierarchy_refresh(std::move(refresh), "AmrSystem");
}

template <int Dim>
void AmrSystem<Dim>::install_program_restart_hooks(std::function<void()> preflight,
                                                   std::function<void()> regrid,
                                                   std::function<void()> resync) {
  p_->program.install_restart_hooks(std::move(preflight), std::move(regrid), std::move(resync),
                                    "AmrSystem");
}

template <int Dim>
void AmrSystem<Dim>::set_program_cadence(int substeps, int stride) {
  require_amr_assembling(p_->lifecycle, "set_program_cadence");
  p_->program.set_cadence(substeps, stride, "AmrSystem");
}

template <int Dim>
int AmrSystem<Dim>::program_substeps() const {
  return p_->program.substeps_;
}

template <int Dim>
int AmrSystem<Dim>::program_stride() const {
  return p_->program.stride_;
}

template <int Dim>
double AmrSystem<Dim>::program_cadence_window_dt() const {
  return p_->program.cadence_window_dt_;
}

template <int Dim>
int AmrSystem<Dim>::program_cadence_window_steps() const {
  return p_->program.cadence_window_steps_;
}

template <int Dim>
double AmrSystem<Dim>::program_cadence_window_start_time() const {
  return p_->program.cadence_window_start_time_;
}

template <int Dim>
double AmrSystem<Dim>::program_last_dt() const {
  return static_cast<double>(p_->program.last_dt_);
}

template <int Dim>
void AmrSystem<Dim>::restore_program_cadence_window(double accumulated_dt, int held_steps,
                                                    double window_start_time,
                                                    double accepted_last_dt, double accepted_time,
                                                    int macro_step) {
  p_->program.restore_cadence_window(accumulated_dt, held_steps, window_start_time,
                                     accepted_last_dt, accepted_time, macro_step, "AmrSystem");
}

template <int Dim>
void AmrSystem<Dim>::set_program_block_map(const std::vector<int>& program_to_runtime) {
  for (std::size_t program = 0; program < program_to_runtime.size(); ++program) {
    const int block = program_to_runtime[program];
    if (block < 0 || block >= static_cast<int>(p_->blocks.size()))
      throw std::out_of_range("AmrSystem Program block map is out of range");
    for (std::size_t previous = 0; previous < program; ++previous)
      if (program_to_runtime[previous] == block)
        throw std::invalid_argument("AmrSystem Program block map contains duplicate routes");
  }
  p_->program.block_map_ = program_to_runtime;
}

template <int Dim>
const std::vector<int>& AmrSystem<Dim>::program_block_map() const {
  return p_->program.block_map_;
}

template <int Dim>
std::string AmrSystem<Dim>::installed_program_hash() const {
  return p_->program.installed_hash_;
}

template <int Dim>
void AmrSystem<Dim>::seed_program_params(int block, const std::vector<double>& defaults) {
  p_->program.seed_params(block, defaults);
}

template <int Dim>
void AmrSystem<Dim>::set_program_params(int block, const std::vector<double>& values) {
  p_->program.set_params(block, values, "AmrSystem");
}

template <int Dim>
RuntimeParams AmrSystem<Dim>::program_params(int block) const {
  return p_->program.params(block);
}

template <int Dim>
runtime::amr::AmrRuntime<Dim>* AmrSystem<Dim>::engine() const {
  p_->ensure_engine();
  return p_->engine.get();
}

template <int Dim>
bool AmrSystem<Dim>::uses_runtime_engine() const {
  return static_cast<bool>(p_->engine);
}

template <int Dim>
runtime::program::Profiler& AmrSystem<Dim>::profiler_handle() {
  return p_->program.profiler_;
}

template <int Dim>
runtime::program::ProgramRuntimeState<Dim>& AmrSystem<Dim>::program_runtime_state_() {
  return p_->program;
}

template <int Dim>
void AmrSystem<Dim>::record_program_diagnostic(const std::string& name, double value) {
  p_->program.record_diagnostic(name, static_cast<Real>(value));
}

template <int Dim>
void AmrSystem<Dim>::record_program_balance_term(const std::string& route, const std::string& term,
                                                 double value) {
  p_->program.record_balance_term(route, term, static_cast<Real>(value), "AmrSystem");
}

template <int Dim>
bool AmrSystem<Dim>::program_balance_consumer_is_due(const std::string& contract,
                                                     const std::string& route, int every_n) const {
  return p_->program.balance_consumer_is_due(contract, route, every_n, "AmrSystem");
}

template <int Dim>
double AmrSystem<Dim>::program_diagnostic(const std::string& name) const {
  return static_cast<double>(p_->program.diagnostic(name, "AmrSystem"));
}

template <int Dim>
std::map<std::string, double> AmrSystem<Dim>::program_diagnostics() const {
  std::map<std::string, double> result;
  for (const auto& [name, value] : p_->program.diagnostics())
    result.emplace(name, static_cast<double>(value));
  return result;
}

template <int Dim>
void AmrSystem<Dim>::begin_step_projection_report() {
  p_->program.begin_step_projection_report();
}

template <int Dim>
void AmrSystem<Dim>::note_step_projection(const std::string& name) {
  p_->program.note_step_projection(name);
}

template <int Dim>
std::vector<std::string> AmrSystem<Dim>::consume_step_projections() {
  return p_->program.consume_step_projections();
}

template <int Dim>
void AmrSystem<Dim>::mark_bound() {
  if (p_->lifecycle.frozen())
    p_->lifecycle.to_bound();
  p_->ensure_engine();
  const auto& routes = p_->boundary_registry.state_routes();
  if (!routes.empty() && routes.size() != p_->blocks.size())
    throw std::runtime_error("AmrSystem state routes do not exactly cover its blocks");
  for (const typename Impl::BlockSpec& block : p_->blocks) {
    if (!routes.empty() && !routes.contains(block.name))
      throw std::runtime_error("AmrSystem block lacks its exact state route");
    const auto* boundary = p_->boundary_registry.find_boundary(block.name);
    if (boundary == nullptr)
      continue;
    if (boundary->authority->ncomp() != block.ncomp ||
        boundary->authority->periodic_axes() != p_->cfg.periodicity)
      throw std::runtime_error("AmrSystem boundary differs from its block/domain contract");
    for (std::size_t level = 0; level < p_->engine->hierarchy().num_levels(); ++level)
      for (int axis = 0; axis < Dim; ++axis)
        if (p_->engine->hierarchy().state(level).ghosts()[axis] < boundary->required_depth)
          throw std::runtime_error("AmrSystem boundary depth exceeds level storage");
  }
  p_->lifecycle.to_bound();
}

template <int Dim>
std::string AmrSystem<Dim>::lifecycle_state() const {
  return p_->lifecycle.state(p_->macro_step);
}

template <int Dim>
Extent<Dim> AmrSystem<Dim>::spatial_shape() const {
  return p_->cfg.shape;
}

template <int Dim>
double AmrSystem<Dim>::time() const {
  return p_->accepted_time;
}

template <int Dim>
int AmrSystem<Dim>::macro_step() const {
  return p_->macro_step;
}

template <int Dim>
void AmrSystem<Dim>::set_clock(double accepted_time, int macro_step) {
  if (!std::isfinite(accepted_time) || macro_step < 0)
    throw std::invalid_argument("AmrSystem clock requires finite time and non-negative step");
  p_->program.consume_cadence_clock_restore(accepted_time, macro_step, "AmrSystem");
  p_->accepted_time = accepted_time;
  p_->macro_step = macro_step;
}

template <int Dim>
void AmrSystem<Dim>::enable_profiling() {
  p_->program.profiler_.enable();
}

template <int Dim>
void AmrSystem<Dim>::disable_profiling() {
  p_->program.profiler_.disable();
}

template <int Dim>
bool AmrSystem<Dim>::is_profiling() const {
  return p_->program.profiler_.enabled();
}

template <int Dim>
void AmrSystem<Dim>::reset_profiling() {
  p_->program.profiler_.reset();
}

template <int Dim>
std::string AmrSystem<Dim>::profile_report() const {
  return p_->program.profiler_.report();
}

template <int Dim>
int AmrSystem<Dim>::n_blocks() const {
  return static_cast<int>(p_->blocks.size());
}

template <int Dim>
std::vector<std::string> AmrSystem<Dim>::block_names() const {
  std::vector<std::string> names;
  names.reserve(p_->blocks.size());
  for (const typename Impl::BlockSpec& block : p_->blocks)
    names.push_back(block.name);
  return names;
}

template <int Dim>
EffectiveOptionsReport AmrSystem<Dim>::effective_options_report() const {
  if (!p_->embedded_boundary_configuration_contract.empty())
    p_->ensure_engine();
  EffectiveOptionsReport report;
  report.runtime = "amr_system";
  report.has_amr = true;
  report.topology.dimension = Dim;
  report.topology.periodicity.reserve(Dim);
  for (int axis = 0; axis < Dim; ++axis)
    report.topology.periodicity.push_back(p_->cfg.periodicity[axis]);
  report.poisson.solver = "geometric_mg";
  report.poisson.solver_option_schema = "pops.amr.field-solver-options.geometric-mg@1";
  if (!p_->embedded_boundary_configuration_contract.empty()) {
    report.eb.enabled = true;
    report.eb.geometry_mode = std::string(
        runtime::system::prepared_embedded_boundary_mode_name(p_->embedded_boundary_mode));
    report.eb.kappa_min = static_cast<double>(p_->embedded_boundary_thresholds.kappa_min);
    report.eb.face_open_eps = static_cast<double>(p_->embedded_boundary_thresholds.face_open_eps);
    report.eb.cut_theta_min = static_cast<double>(p_->embedded_boundary_thresholds.cut_theta_min);
    report.eb.semantic_digest = p_->embedded_boundary_semantic_digest;
    report.eb.materialization_digest =
        p_->prepared_hierarchy->embedded_boundary_materialization_digest;
    report.eb.generation = p_->embedded_boundary_generation;
  }
  report.blocks.reserve(p_->blocks.size());
  for (const typename Impl::BlockSpec& block : p_->blocks) {
    EffectiveBlockOptions row;
    row.name = block.name;
    row.ncomp = block.ncomp;
    row.n_ghost = block.required_ghost_depth;
    row.gamma = block.gamma;
    row.substeps = block.substeps;
    row.stride = block.stride;
    row.time = block.time;
    report.blocks.push_back(std::move(row));
  }
  return report;
}

template <int Dim>
int AmrSystem<Dim>::n_levels() {
  p_->ensure_engine();
  return static_cast<int>(p_->engine->hierarchy().num_levels());
}

template <int Dim>
int AmrSystem<Dim>::max_levels() {
  return p_->cfg.level_count;
}

template <int Dim>
int AmrSystem<Dim>::configured_n_levels() {
  return p_->cfg.level_count;
}

template <int Dim>
int AmrSystem<Dim>::n_vars() {
  if (p_->blocks.size() != 1)
    throw std::runtime_error("AmrSystem::n_vars requires exactly one block");
  return p_->blocks.front().ncomp;
}

template <int Dim>
int AmrSystem<Dim>::block_n_vars(const std::string& name) {
  return p_->block(name).ncomp;
}

template <int Dim>
int AmrSystem<Dim>::n_patches() {
  p_->ensure_engine();
  const auto& hierarchy = p_->engine->hierarchy();
  return static_cast<int>(hierarchy.layout(hierarchy.num_levels() - 1).patches().size());
}

template <int Dim>
std::vector<AmrPatch<Dim>> AmrSystem<Dim>::patch_boxes() {
  p_->ensure_engine();
  std::vector<AmrPatch<Dim>> result;
  for (std::size_t level = 1; level < p_->engine->hierarchy().num_levels(); ++level)
    for (const Box<Dim>& box : p_->engine->hierarchy().layout(level).patches().boxes())
      result.push_back({static_cast<int>(level), box});
  return result;
}

template <int Dim>
std::vector<AmrPatch<Dim>> AmrSystem<Dim>::output_geometry_boxes() {
  p_->ensure_engine();
  std::vector<AmrPatch<Dim>> result;
  for (std::size_t level = 0; level < p_->engine->hierarchy().num_levels(); ++level)
    for (const Box<Dim>& box : p_->engine->hierarchy().layout(level).patches().boxes())
      result.push_back({static_cast<int>(level), box});
  return result;
}

template <int Dim>
int AmrSystem<Dim>::coarse_local_boxes() {
  p_->ensure_engine();
  return static_cast<int>(p_->engine->hierarchy().state(0).local_size());
}

template <int Dim>
int AmrSystem<Dim>::coarse_total_boxes() {
  p_->ensure_engine();
  return static_cast<int>(p_->engine->hierarchy().layout(0).patches().size());
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::level_state(int level) {
  return level_state_global(level);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::level_state_global(int level) {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem level is out of range");
  const auto& hierarchy = p_->engine->hierarchy();
  return gather_field(hierarchy.state(static_cast<std::size_t>(level)),
                      hierarchy.layout(static_cast<std::size_t>(level)).domain(),
                      hierarchy.state(static_cast<std::size_t>(level)).ncomp());
}

template <int Dim>
void AmrSystem<Dim>::set_level_state(int level, const std::vector<double>& state) {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem level is out of range");
  auto& hierarchy = p_->engine->hierarchy();
  write_field(hierarchy.state(static_cast<std::size_t>(level)),
              hierarchy.layout(static_cast<std::size_t>(level)).domain(), state,
              hierarchy.state(static_cast<std::size_t>(level)).ncomp());
  p_->discard_level_evaluations();
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::block_level_state(const std::string& name, int level) {
  (void)p_->block(name);
  return level_state(level);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::block_level_state_global(const std::string& name, int level) {
  (void)p_->block(name);
  return level_state_global(level);
}

template <int Dim>
void AmrSystem<Dim>::set_block_level_state(const std::string& name, int level,
                                           const std::vector<double>& state) {
  (void)p_->block(name);
  set_level_state(level, state);
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_state_local_pieces(const std::string& name,
                                                                        int level) {
  (void)p_->block(name);
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem output level is out of range");
  const MultiFab<Dim>& state = p_->engine->hierarchy().state(static_cast<std::size_t>(level));
  return output_local_pieces(state, level, state.distribution().replicated());
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_embedded_boundary_local_pieces(
    const std::string& name, int level) {
  p_->ensure_engine();
  if (level < 0 ||
      static_cast<std::size_t>(level) >= p_->prepared_hierarchy->embedded_boundary.size())
    throw std::out_of_range("AmrSystem embedded-boundary output level is out of range");
  const auto& embedded = p_->prepared_hierarchy->embedded_boundary[static_cast<std::size_t>(level)];
  if (!embedded)
    throw std::logic_error("AmrSystem has no prepared embedded-boundary geometry");
  const MultiFab<Dim>* field = nullptr;
  if (name == "pops_active")
    field = &embedded->active_mask();
  else if (name == "pops_phi")
    field = &embedded->phi();
  else if (name == "pops_kappa")
    field = &embedded->volume_fraction();
  else
    throw std::invalid_argument("unknown AMR embedded-boundary sidecar: " + name);
  return output_local_pieces(*field, level, field->distribution().replicated());
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_state_root_pieces(const ObserverMpiLane& lane,
                                                                       const std::string& name,
                                                                       int level) {
  return output_pieces_to_root(
      lane, detail::output_collective_identity("amr_system", "state", name, level),
      [&] { return output_state_local_pieces(name, level); });
}

template <int Dim>
std::vector<OutputPiece<Dim>> AmrSystem<Dim>::output_embedded_boundary_root_pieces(
    const ObserverMpiLane& lane, const std::string& name, int level) {
  if (lane.size() == 1)
    return output_embedded_boundary_local_pieces(name, level);
  return output_pieces_to_root(
      lane, detail::output_collective_identity("amr_system", "embedded-boundary", name, level),
      [&] { return output_embedded_boundary_local_pieces(name, level); });
}

template <int Dim>
double AmrSystem<Dim>::composite_reduce(const std::string& name, const std::string& kind,
                                        int component,
                                        const std::vector<int>& requested_levels) const {
  const typename Impl::BlockSpec& block = p_->block(name);
  p_->ensure_engine();
  if (kind == "sum_all" || kind == "abs_sum_all" || kind == "sum_sq_all" || kind == "abs_max_all") {
    const std::string base = kind == "sum_all"       ? "sum"
                             : kind == "abs_sum_all" ? "abs_sum"
                             : kind == "sum_sq_all"  ? "sum_sq"
                                                     : "abs_max";
    double result = 0.0;
    for (int current = 0; current < block.ncomp; ++current) {
      const double value = composite_reduce(name, base, current, requested_levels);
      result = kind == "abs_max_all" ? std::max(result, value) : result + value;
    }
    return result;
  }
  if (component < 0 || component >= block.ncomp)
    throw std::out_of_range("AMR composite reduction component is outside the block state");
  const auto levels =
      resolved_composite_levels(p_->engine->hierarchy().num_levels(), requested_levels);
  std::vector<std::shared_ptr<const MultiFab<Dim>>> selected_coverage;
  const std::vector<std::shared_ptr<const MultiFab<Dim>>>* coverage =
      &p_->prepared_hierarchy->active_coverage;
  if (!requested_levels.empty()) {
    selected_coverage = Impl::prepare_active_coverage(*p_->engine, levels);
    coverage = &selected_coverage;
  }
  std::vector<runtime::amr::CompositeLevelView<Dim, memory_space>> views;
  views.reserve(levels.size());
  for (std::size_t position = 0; position < levels.size(); ++position) {
    const int level = levels[position];
    const std::size_t index = static_cast<std::size_t>(level);
    const Box<Dim>& domain = p_->engine->hierarchy().layout(index).domain();
    const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, p_->cfg.lower, p_->cfg.upper);
    std::array<Real, Dim> extent{};
    for (int axis = 0; axis < Dim; ++axis)
      extent[static_cast<std::size_t>(axis)] = geometry.spacing(axis);
    const MultiFab<Dim>* relative = nullptr;
    const auto& embedded = p_->prepared_hierarchy->embedded_boundary[index];
    if (embedded && embedded->mode() == runtime::system::PreparedEmbeddedBoundaryMode::staircase)
      relative = &embedded->active_mask();
    else if (embedded &&
             embedded->mode() == runtime::system::PreparedEmbeddedBoundaryMode::cut_cell)
      relative = &embedded->volume_fraction();
    const std::size_t coverage_index = requested_levels.empty() ? index : position;
    views.push_back({&p_->engine->hierarchy().state(index), (*coverage)[coverage_index].get(),
                     extent, relative});
  }
  return static_cast<double>(
      runtime::amr::composite_reduce<Dim, memory_space>(
          views, component, composite_reduction_kind(kind), *p_->prepared_hierarchy->lane)
          .value);
}

template <int Dim>
double AmrSystem<Dim>::composite_reduce_field(const std::string& provider_slot,
                                              const std::string& kind, int component,
                                              const std::vector<int>& requested_levels) {
  const std::string slot = p_->resolve_field_slot(provider_slot);
  p_->materialize_field(slot);
  const typename Impl::FieldPlan& plan = p_->field_plans.at(slot);
  const auto levels = resolved_composite_levels(plan.accepted_potential.size(), requested_levels);
  std::vector<std::shared_ptr<const MultiFab<Dim>>> selected_coverage;
  const std::vector<std::shared_ptr<const MultiFab<Dim>>>* coverage = &plan.active_coverage;
  if (!requested_levels.empty()) {
    selected_coverage = Impl::prepare_active_coverage(*p_->engine, levels);
    coverage = &selected_coverage;
  }
  const int ncomp = plan.accepted_potential.empty() ? 0 : plan.accepted_potential.front()->ncomp();
  if (component < 0 || component >= ncomp)
    throw std::out_of_range("AMR composite field component is outside the prepared field");
  if (kind == "sum_all" || kind == "abs_sum_all" || kind == "sum_sq_all" || kind == "abs_max_all") {
    const std::string base = kind == "sum_all"       ? "sum"
                             : kind == "abs_sum_all" ? "abs_sum"
                             : kind == "sum_sq_all"  ? "sum_sq"
                                                     : "abs_max";
    double result = 0.0;
    for (int current = 0; current < ncomp; ++current) {
      const double value = composite_reduce_field(slot, base, current, requested_levels);
      result = kind == "abs_max_all" ? std::max(result, value) : result + value;
    }
    return result;
  }
  std::vector<runtime::amr::CompositeLevelView<Dim, memory_space>> views;
  views.reserve(levels.size());
  for (std::size_t position = 0; position < levels.size(); ++position) {
    const int level = levels[position];
    const std::size_t index = static_cast<std::size_t>(level);
    const Box<Dim>& domain = p_->engine->hierarchy().layout(index).domain();
    const Geometry<Dim> geometry = Geometry<Dim>::from_bounds(domain, p_->cfg.lower, p_->cfg.upper);
    std::array<Real, Dim> extent{};
    for (int axis = 0; axis < Dim; ++axis)
      extent[static_cast<std::size_t>(axis)] = geometry.spacing(axis);
    const std::size_t coverage_index = requested_levels.empty() ? index : position;
    views.push_back(
        {plan.accepted_potential[index].get(), (*coverage)[coverage_index].get(), extent, nullptr});
  }
  return static_cast<double>(
      runtime::amr::composite_reduce<Dim, memory_space>(
          views, component, composite_reduction_kind(kind), *p_->prepared_hierarchy->lane)
          .value);
}

template <int Dim>
std::vector<int> AmrSystem<Dim>::level_owner_ranks(int level) {
  p_->ensure_engine();
  if (level < 0 || static_cast<std::size_t>(level) >= p_->engine->hierarchy().num_levels())
    throw std::out_of_range("AmrSystem owner level is out of range");
  const auto& distribution =
      p_->engine->hierarchy().layout(static_cast<std::size_t>(level)).distribution();
  if (distribution.replicated())
    throw std::runtime_error("replicated AMR levels have no unique owner ranks");
  std::vector<int> result;
  result.reserve(distribution.owners().size());
  for (const Index<Dim>& owner : distribution.owners())
    result.push_back(static_cast<int>(distribution.rank_space().linear_rank(owner)));
  return result;
}

template <int Dim>
double AmrSystem<Dim>::mass() {
  if (p_->blocks.empty())
    throw std::logic_error("AmrSystem has no block");
  return mass(p_->blocks.front().name);
}

template <int Dim>
double AmrSystem<Dim>::mass(const std::string& name) {
  return composite_reduce(name, "sum", 0);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::density() {
  if (p_->blocks.empty())
    throw std::logic_error("AmrSystem has no block");
  return density(p_->blocks.front().name);
}

template <int Dim>
std::vector<double> AmrSystem<Dim>::density(const std::string& name) {
  (void)p_->block(name);
  p_->ensure_engine();
  return gather_field(p_->engine->hierarchy().state(0), p_->cfg.index_domain(), 1);
}

template <int Dim>
std::vector<std::uint8_t> AmrSystem<Dim>::program_accepted_state() const {
  return p_->program_accepted_bytes;
}

template <int Dim>
void AmrSystem<Dim>::copy_program_accepted_state_into(std::vector<std::uint8_t>& state) const {
  state = p_->program_accepted_bytes;
}

template <int Dim>
void AmrSystem<Dim>::restore_program_accepted_state(const std::vector<std::uint8_t>& state) {
  if (p_->program_accepted_revision == std::numeric_limits<std::uint64_t>::max())
    throw std::overflow_error("AmrSystem Program accepted-state revision overflow");
  p_->program_accepted_bytes = state;
  p_->program_accepted_bytes_runtime_owned = false;
  ++p_->program_accepted_revision;
}

template <int Dim>
void AmrSystem<Dim>::restore_checkpoint_accepted_state(const std::vector<std::uint8_t>& state) {
  p_->ensure_engine();
  std::optional<runtime::program::AmrProgramAcceptedState<Dim>> decoded;
  std::optional<runtime::amr::PersistentTaggingState<Dim>> decoded_tagging;
  std::exception_ptr local_error;
  long local_failure = 0;
  try {
    decoded.emplace(runtime::program::deserialize_amr_program_accepted_state<Dim>(state));
    runtime::program::require_live_amr_program_checkpoint(*decoded, *p_->engine);
    if (p_->tagging_spec) {
      decoded_tagging.emplace(runtime::amr::PersistentTaggingState<Dim>::decode(
          decoded->tagging_hysteresis_state, p_->tagging_spec->min_cycles,
          p_->tagging_spec->provider_identity, p_->live_tagging_parent_domains()));
    } else {
      if (!decoded->tagging_hysteresis_state.empty())
        throw std::invalid_argument(
            "AMR checkpoint carries tagging hysteresis without a bound tagging authority");
      decoded_tagging.emplace();
    }
  } catch (...) {
    local_failure = 1;
    local_error = std::current_exception();
  }
  const ExecutionLane& lane = *p_->prepared_hierarchy->lane;
  if (all_reduce_max(local_failure, lane.communicator()) != 0) {
    if (local_error)
      std::rethrow_exception(local_error);
    throw std::runtime_error("AMR checkpoint tagging state failed validation on another MPI rank");
  }
  runtime::program::require_collective_amr_program_checkpoint_consensus(*decoded, lane);
  const long exhausted =
      p_->program_accepted_revision == std::numeric_limits<std::uint64_t>::max() ? 1L : 0L;
  if (all_reduce_max(exhausted, lane.communicator()) != 0)
    throw std::overflow_error("AmrSystem Program accepted-state revision overflow");
  p_->program_accepted_bytes = state;
  p_->program_accepted_bytes_runtime_owned = false;
  p_->tagging_state = std::move(*decoded_tagging);
  ++p_->program_accepted_revision;
}

template <int Dim>
std::uint64_t AmrSystem<Dim>::program_accepted_state_revision() const {
  return p_->program_accepted_revision;
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_accepted_state_manifest() const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_clock_manifest() const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_temporal_partition_manifest() const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_flux_ledger_manifest() const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_interface_flux_ledger_manifest()
    const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::program_sync_manifest() const {
  return {};
}

template <int Dim>
std::vector<std::vector<std::string>> AmrSystem<Dim>::checkpoint_transfer_routes() const {
  std::vector<std::vector<std::string>> rows;
  rows.reserve(p_->bootstrap_subject_routes.size());
  for (const auto& [key, route_identity] : p_->bootstrap_subject_routes) {
    const auto found = p_->bootstrap_transfer_routes.find(route_identity);
    if (found == p_->bootstrap_transfer_routes.end())
      throw std::logic_error("AMR bootstrap transfer manifest lost a registered provider");
    const typename Impl::BootstrapTransferRoute& route = found->second;
    std::string ghosts;
    std::string ratio;
    for (int axis = 0; axis < Dim; ++axis) {
      if (axis != 0) {
        ghosts.push_back(',');
        ratio.push_back(',');
      }
      ghosts += std::to_string(route.ghost_depth[axis]);
      ratio += std::to_string(route.refinement_ratio[axis]);
    }
    rows.push_back({key.first, key.second, route.identity, route.provider_identity, route.kernel,
                    route.space, route.centering, route.representation, route.storage,
                    route.operation, std::to_string(route.order), std::move(ghosts),
                    std::to_string(Dim), std::move(ratio)});
  }
  return rows;
}

template <int Dim>
const std::vector<CouplingOperatorView>& AmrSystem<Dim>::coupled_operators() const {
  return p_->coupling_views;
}

template AmrSystem<kNativeDimension>::AmrSystem(const AmrSystemConfig<kNativeDimension>&);
template AmrSystem<kNativeDimension>::~AmrSystem();
template AmrSystem<kNativeDimension>::AmrSystem(AmrSystem&&) noexcept;
template AmrSystem<kNativeDimension>& AmrSystem<kNativeDimension>::operator=(AmrSystem&&) noexcept;
template void AmrSystem<kNativeDimension>::add_block(
    const std::string&, const ModelSpec&, const std::string&, const std::string&,
    const std::string&, const std::string&, int, int, const std::vector<std::string>&,
    const std::vector<std::string>&, const NewtonOptions&, bool, double, double, bool);
template void AmrSystem<kNativeDimension>::set_compiled_block(
    int, double, int, AmrCompiledBlockBuilder<kNativeDimension>, const std::string&, bool,
    const std::string&, int, const std::vector<std::string>&, const std::vector<std::string>&,
    double, double, bool);
template void AmrSystem<kNativeDimension>::install_prepared_amr_block(
    PreparedAmrSystemBlock<kNativeDimension>);
template void AmrSystem<kNativeDimension>::set_bootstrap_tagging(
    const std::vector<std::string>&, const std::vector<std::string>&,
    const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<int>&,
    const std::vector<int>&, const std::vector<double>&, const std::vector<int>&,
    const std::vector<runtime::amr::PreparedTaggingProgram<kNativeDimension>::Stencil>&,
    const std::vector<std::int32_t>&, const std::vector<std::int32_t>&,
    const std::vector<std::int32_t>&, const std::vector<std::int32_t>&, int, const std::string&,
    const std::string&, const std::string&, const std::string&);
template runtime::amr::PreparedTaggerCandidates<kNativeDimension>
AmrSystem<kNativeDimension>::execute_prepared_tagging(int);
template bool AmrSystem<kNativeDimension>::regrid_from_prepared_tagging(int);
template void AmrSystem<kNativeDimension>::begin_bootstrap_plan();
template bool AmrSystem<kNativeDimension>::bootstrap_next_level();
template void AmrSystem<kNativeDimension>::commit_bootstrap_level();
template void AmrSystem<kNativeDimension>::rollback_bootstrap_level();
template void AmrSystem<kNativeDimension>::register_bootstrap_transfer_route(
    const std::string&, const std::vector<std::string>&, const std::string&, const std::string&,
    const std::string&, const std::string&, const std::string&, const std::string&,
    const std::string&, int, const Extent<kNativeDimension>&, const Extent<kNativeDimension>&);
template void AmrSystem<kNativeDimension>::set_analytic_level_set(const std::vector<std::string>&,
                                                                  const std::vector<double>&,
                                                                  const std::string&, double,
                                                                  double, double);
template void AmrSystem<kNativeDimension>::set_disc_domain(double, double, double,
                                                           const std::string&, double, double,
                                                           double);
template void AmrSystem<kNativeDimension>::set_geometry_mode(const std::string&);
template void AmrSystem<kNativeDimension>::refresh_prepared_amr_levels();
template const PreparedAmrLevelEvaluation<kNativeDimension>&
AmrSystem<kNativeDimension>::evaluate_prepared_amr_level(
    const runtime::multiblock::BoundaryEvaluationPoint&);
template void AmrSystem<kNativeDimension>::prepare_generated_amr_level_state(
    const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&);
template const PreparedAmrLevelEvaluation<kNativeDimension>&
AmrSystem<kNativeDimension>::evaluate_prepared_amr_level_at(
    const runtime::multiblock::BoundaryEvaluationPoint&, MultiFab<kNativeDimension>&);
template const PreparedAmrLevelEvaluation<kNativeDimension>&
AmrSystem<kNativeDimension>::prepared_amr_level_evaluation(int) const;
template Geometry<kNativeDimension> AmrSystem<kNativeDimension>::prepared_amr_level_geometry(
    int) const;
template BoundaryTopology<kNativeDimension>
AmrSystem<kNativeDimension>::prepared_amr_boundary_topology() const;
template Real AmrSystem<kNativeDimension>::prepared_amr_level_maximum_speed(
    int, const MultiFab<kNativeDimension>&) const;
template MultiFab<kNativeDimension>& AmrSystem<kNativeDimension>::prepared_amr_level_auxiliary(int);
template const MultiFab<kNativeDimension>&
AmrSystem<kNativeDimension>::prepared_amr_level_auxiliary(int) const;
template void AmrSystem<kNativeDimension>::add_prepared_amr_poisson_rhs(
    int, MultiFab<kNativeDimension>&);
template void AmrSystem<kNativeDimension>::install_block_state_route(const std::string&,
                                                                     const std::string&);
template void AmrSystem<kNativeDimension>::install_field_storage_route(const std::string&,
                                                                       const std::string&);
template void AmrSystem<kNativeDimension>::register_field_nullspace_provider(
    std::shared_ptr<const FieldNullspaceProvider<kNativeDimension>>);
template void AmrSystem<kNativeDimension>::register_hierarchy_tensor_solver_provider(
    std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider<kNativeDimension>>);
template void AmrSystem<kNativeDimension>::register_program_hierarchy_tensor_solver_provider(
    std::shared_ptr<const runtime::program::HierarchyTensorSolverProvider<kNativeDimension>>);
template std::shared_ptr<
    const runtime::program::HierarchyTensorSolverProviderRegistry<kNativeDimension>>
AmrSystem<kNativeDimension>::hierarchy_tensor_solver_provider_registry() const;
template void AmrSystem<kNativeDimension>::register_field_solver_provider(
    std::shared_ptr<const runtime::amr::ExactAmrFieldSolverProvider<kNativeDimension>>);
template std::string AmrSystem<kNativeDimension>::register_field_solver_provider(
    const std::string&, runtime::field::PreparedFieldSolverSpec,
    std::shared_ptr<component::LoadedComponent>, std::shared_ptr<component::LoadedComponent>);
template void AmrSystem<kNativeDimension>::set_default_field_nullspace(
    const std::string&, const PreparedProviderOptions&);
template void AmrSystem<kNativeDimension>::set_poisson(const std::string&, const std::string&,
                                                       const std::string&,
                                                       const AmrFieldSolverOptions&);
template void AmrSystem<kNativeDimension>::set_field_solver_plan(
    const std::string&, const std::string&, const std::string&, const std::string&,
    const std::string&, const std::string&, const std::vector<std::string>&,
    const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<double>&,
    const std::string&, const AmrFieldHierarchyPolicyAuthority&, const AmrFieldSolverOptions&);
template AmrFieldSolverConfiguration AmrSystem<kNativeDimension>::field_solver_configuration(
    const std::string&) const;
template void AmrSystem<kNativeDimension>::set_field_reaction(const std::string&, double);
template void AmrSystem<kNativeDimension>::set_field_topology_authority(const std::string&,
                                                                        const std::string&,
                                                                        const std::string&,
                                                                        const std::string&);
template std::vector<runtime::field::FieldTopologyReportRow>
AmrSystem<kNativeDimension>::field_topology_report(const std::string&) const;
template void AmrSystem<kNativeDimension>::set_field_boundary_plan(const std::string&,
                                                                   const std::vector<std::string>&,
                                                                   const std::vector<double>&,
                                                                   const std::vector<double>&,
                                                                   const std::vector<double>&);
template void AmrSystem<kNativeDimension>::set_field_boundary_dependencies(
    const std::string&, const std::vector<std::string>&, const std::vector<int>&,
    const std::vector<std::string>&, const std::vector<std::string>&, const std::vector<int>&);
template void AmrSystem<kNativeDimension>::set_field_boundary_kernel(
    const std::string&, const CompiledFieldBoundaryKernel<kNativeDimension>&);
template void AmrSystem<kNativeDimension>::set_field_logical_timepoint(
    const std::string&, const FieldLogicalTimePoint&);
template void AmrSystem<kNativeDimension>::set_field_boundary_parameters(
    const std::string&, const std::vector<double>&);
template void AmrSystem<kNativeDimension>::set_field_newton_plan(const std::string&, double, int,
                                                                 double, int, int, double, double);
template void AmrSystem<kNativeDimension>::set_field_nullspace(const std::string&,
                                                               const std::string&,
                                                               const PreparedProviderOptions&);
template void AmrSystem<kNativeDimension>::register_elliptic_field(const std::string&,
                                                                   const std::string&,
                                                                   const std::vector<int>&, int);
template void AmrSystem<kNativeDimension>::set_block_elliptic_field(
    const std::string&, const std::string&,
    std::function<void(const MultiFab<kNativeDimension>&, MultiFab<kNativeDimension>&)>);
template SolveOutcome AmrSystem<kNativeDimension>::solve_program_field_at(
    const runtime::multiblock::BoundaryEvaluationPoint&, const std::string&, int,
    const MultiFab<kNativeDimension>*);
template SolveOutcome AmrSystem<kNativeDimension>::solve_program_field_from_blocks_at(
    const runtime::multiblock::BoundaryEvaluationPoint&, const std::string&, int,
    const std::vector<const MultiFab<kNativeDimension>*>&);
template SolveOutcome AmrSystem<kNativeDimension>::solve_program_default_field(int);
template std::vector<double> AmrSystem<kNativeDimension>::named_field_values(const std::string&);
template std::vector<std::string> AmrSystem<kNativeDimension>::field_provider_slots() const;
template int AmrSystem<kNativeDimension>::field_provider_levels(const std::string&);
template void AmrSystem<kNativeDimension>::set_field_potential(const std::string&,
                                                               const std::vector<double>&);
template void AmrSystem<kNativeDimension>::set_field_potential_level(const std::string&, int,
                                                                     const std::vector<double>&);
template std::vector<double> AmrSystem<kNativeDimension>::field_potential_global(
    const std::string&);
template std::vector<double> AmrSystem<kNativeDimension>::field_potential_level_global(
    const std::string&, int);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_field_local_pieces(const std::string&, int);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_field_root_pieces(const ObserverMpiLane&, const std::string&,
                                                      int);
template void AmrSystem<kNativeDimension>::install_hyperbolic_boundary(
    const std::string&, const std::string&, int, const std::vector<std::string>&,
    const std::vector<double>&, const std::vector<std::string>&, const std::vector<std::string>&,
    const std::string&, const std::vector<std::string>&, const std::vector<std::string>&,
    const std::vector<std::vector<std::string>>&, const std::vector<std::vector<double>>&,
    const std::vector<std::string>&);
template void AmrSystem<kNativeDimension>::install_prepared_hyperbolic_boundary(
    const std::string&, const std::string&, int, const std::string&,
    std::shared_ptr<const HyperbolicBoundary>);
template void AmrSystem<kNativeDimension>::discard_hyperbolic_boundaries();
template void AmrSystem<kNativeDimension>::set_density(const std::string&,
                                                       const std::vector<double>&);
template void AmrSystem<kNativeDimension>::set_conservative_state(const std::string&,
                                                                  const std::vector<double>&);
template void AmrSystem<kNativeDimension>::add_dt_bound(const std::string&,
                                                        std::function<double()>);
template std::string AmrSystem<kNativeDimension>::last_dt_bound() const;
template void AmrSystem<kNativeDimension>::step(double);
template void AmrSystem<kNativeDimension>::advance(double, int);
template double AmrSystem<kNativeDimension>::step_cfl(double, double, double, double);
template void AmrSystem<kNativeDimension>::begin_step_transaction();
template void AmrSystem<kNativeDimension>::commit_step_transaction();
template void AmrSystem<kNativeDimension>::finalize_step_transaction();
template void AmrSystem<kNativeDimension>::rollback_step_transaction();
template bool AmrSystem<kNativeDimension>::has_active_step_transaction() const noexcept;
template void AmrSystem<kNativeDimension>::restore_active_step_transaction_for_program();
template std::map<std::string, double> AmrSystem<kNativeDimension>::step_change_l2() const;
template void AmrSystem<kNativeDimension>::install_program_step(std::function<void(double)>);
template void AmrSystem<kNativeDimension>::install_program_hierarchy_refresh(std::function<void()>);
template void AmrSystem<kNativeDimension>::install_program_restart_hooks(std::function<void()>,
                                                                         std::function<void()>,
                                                                         std::function<void()>);
template void AmrSystem<kNativeDimension>::set_program_cadence(int, int);
template int AmrSystem<kNativeDimension>::program_substeps() const;
template int AmrSystem<kNativeDimension>::program_stride() const;
template double AmrSystem<kNativeDimension>::program_cadence_window_dt() const;
template int AmrSystem<kNativeDimension>::program_cadence_window_steps() const;
template double AmrSystem<kNativeDimension>::program_cadence_window_start_time() const;
template double AmrSystem<kNativeDimension>::program_last_dt() const;
template void AmrSystem<kNativeDimension>::restore_program_cadence_window(double, int, double,
                                                                          double, double, int);
template void AmrSystem<kNativeDimension>::set_program_block_map(const std::vector<int>&);
template const std::vector<int>& AmrSystem<kNativeDimension>::program_block_map() const;
template std::string AmrSystem<kNativeDimension>::installed_program_hash() const;
template void AmrSystem<kNativeDimension>::seed_program_params(int, const std::vector<double>&);
template void AmrSystem<kNativeDimension>::set_program_params(int, const std::vector<double>&);
template RuntimeParams AmrSystem<kNativeDimension>::program_params(int) const;
template runtime::amr::AmrRuntime<kNativeDimension>* AmrSystem<kNativeDimension>::engine() const;
template bool AmrSystem<kNativeDimension>::uses_runtime_engine() const;
template runtime::program::Profiler& AmrSystem<kNativeDimension>::profiler_handle();
template runtime::program::ProgramRuntimeState<kNativeDimension>&
AmrSystem<kNativeDimension>::program_runtime_state_();
template void AmrSystem<kNativeDimension>::record_program_diagnostic(const std::string&, double);
template void AmrSystem<kNativeDimension>::record_program_balance_term(const std::string&,
                                                                       const std::string&, double);
template bool AmrSystem<kNativeDimension>::program_balance_consumer_is_due(const std::string&,
                                                                           const std::string&,
                                                                           int) const;
template double AmrSystem<kNativeDimension>::program_diagnostic(const std::string&) const;
template std::map<std::string, double> AmrSystem<kNativeDimension>::program_diagnostics() const;
template void AmrSystem<kNativeDimension>::begin_step_projection_report();
template void AmrSystem<kNativeDimension>::note_step_projection(const std::string&);
template std::vector<std::string> AmrSystem<kNativeDimension>::consume_step_projections();
template void AmrSystem<kNativeDimension>::mark_bound();
template std::string AmrSystem<kNativeDimension>::lifecycle_state() const;
template Extent<kNativeDimension> AmrSystem<kNativeDimension>::spatial_shape() const;
template double AmrSystem<kNativeDimension>::time() const;
template int AmrSystem<kNativeDimension>::macro_step() const;
template void AmrSystem<kNativeDimension>::set_clock(double, int);
template void AmrSystem<kNativeDimension>::enable_profiling();
template void AmrSystem<kNativeDimension>::disable_profiling();
template bool AmrSystem<kNativeDimension>::is_profiling() const;
template void AmrSystem<kNativeDimension>::reset_profiling();
template std::string AmrSystem<kNativeDimension>::profile_report() const;
template int AmrSystem<kNativeDimension>::n_blocks() const;
template std::vector<std::string> AmrSystem<kNativeDimension>::block_names() const;
template EffectiveOptionsReport AmrSystem<kNativeDimension>::effective_options_report() const;
template int AmrSystem<kNativeDimension>::n_levels();
template int AmrSystem<kNativeDimension>::max_levels();
template int AmrSystem<kNativeDimension>::configured_n_levels();
template int AmrSystem<kNativeDimension>::n_vars();
template int AmrSystem<kNativeDimension>::block_n_vars(const std::string&);
template int AmrSystem<kNativeDimension>::n_patches();
template std::vector<AmrPatch<kNativeDimension>> AmrSystem<kNativeDimension>::patch_boxes();
template std::vector<AmrPatch<kNativeDimension>>
AmrSystem<kNativeDimension>::output_geometry_boxes();
template int AmrSystem<kNativeDimension>::coarse_local_boxes();
template int AmrSystem<kNativeDimension>::coarse_total_boxes();
template std::vector<double> AmrSystem<kNativeDimension>::level_state(int);
template std::vector<double> AmrSystem<kNativeDimension>::level_state_global(int);
template void AmrSystem<kNativeDimension>::set_level_state(int, const std::vector<double>&);
template std::vector<double> AmrSystem<kNativeDimension>::level_potential(int);
template std::vector<double> AmrSystem<kNativeDimension>::level_potential_global(int);
template void AmrSystem<kNativeDimension>::set_level_potential(int, const std::vector<double>&);
template std::vector<double> AmrSystem<kNativeDimension>::block_level_state(const std::string&,
                                                                            int);
template std::vector<double> AmrSystem<kNativeDimension>::block_level_state_global(
    const std::string&, int);
template void AmrSystem<kNativeDimension>::set_block_level_state(const std::string&, int,
                                                                 const std::vector<double>&);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_state_local_pieces(const std::string&, int);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_state_root_pieces(const ObserverMpiLane&, const std::string&,
                                                      int);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_embedded_boundary_local_pieces(const std::string&, int);
template std::vector<OutputPiece<kNativeDimension>>
AmrSystem<kNativeDimension>::output_embedded_boundary_root_pieces(const ObserverMpiLane&,
                                                                  const std::string&, int);
template double AmrSystem<kNativeDimension>::composite_reduce(const std::string&,
                                                              const std::string&, int,
                                                              const std::vector<int>&) const;
template double AmrSystem<kNativeDimension>::composite_reduce_field(const std::string&,
                                                                    const std::string&, int,
                                                                    const std::vector<int>&);
template std::vector<int> AmrSystem<kNativeDimension>::level_owner_ranks(int);
template double AmrSystem<kNativeDimension>::mass();
template double AmrSystem<kNativeDimension>::mass(const std::string&);
template std::vector<double> AmrSystem<kNativeDimension>::density();
template std::vector<double> AmrSystem<kNativeDimension>::density(const std::string&);
template std::vector<double> AmrSystem<kNativeDimension>::potential();
template std::vector<std::uint8_t> AmrSystem<kNativeDimension>::program_accepted_state() const;
template void AmrSystem<kNativeDimension>::copy_program_accepted_state_into(
    std::vector<std::uint8_t>&) const;
template void AmrSystem<kNativeDimension>::restore_program_accepted_state(
    const std::vector<std::uint8_t>&);
template void AmrSystem<kNativeDimension>::restore_checkpoint_accepted_state(
    const std::vector<std::uint8_t>&);
template std::uint64_t AmrSystem<kNativeDimension>::program_accepted_state_revision() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_accepted_state_manifest() const;
template std::vector<std::vector<std::string>> AmrSystem<kNativeDimension>::program_clock_manifest()
    const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_temporal_partition_manifest() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_flux_ledger_manifest() const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::program_interface_flux_ledger_manifest() const;
template std::vector<std::vector<std::string>> AmrSystem<kNativeDimension>::program_sync_manifest()
    const;
template std::vector<std::vector<std::string>>
AmrSystem<kNativeDimension>::checkpoint_transfer_routes() const;
template const std::vector<CouplingOperatorView>& AmrSystem<kNativeDimension>::coupled_operators()
    const;

}  // namespace pops
